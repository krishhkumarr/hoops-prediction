
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
from flask_cors import CORS

CSV_PATH = "NCAA.csv"
raw_df = pd.read_csv(CSV_PATH, encoding='latin1')

# Drop post-tournament columns to mirror app.py/prediction_dashboard.py
DROP_COLS = [
    'round', 'champion_share', 'make_tournament',
    'ap_weighted_score', 'ap_sos', 'under_12_ap_rec', 'sum_under_12_ap_games'
]
DROP_COLS = [c for c in DROP_COLS if c in raw_df.columns]
df = raw_df.drop(columns=DROP_COLS)
# De-duplicate columns if any
df = df.loc[:, ~df.columns.duplicated()]

# Prepare features and target exactly like app.py
TARGET_COL = 'champion' if 'champion' in df.columns else None
feature_exclude = {'season', 'Team', 'conference'} | ({TARGET_COL} if TARGET_COL else set())
features_df = df.drop(columns=[c for c in feature_exclude if c in df.columns])
X = features_df.select_dtypes(include='number')
y = df[TARGET_COL] if TARGET_COL and TARGET_COL in df.columns else pd.Series(np.zeros(len(df)))

# Fit models globally (not per-season), on scaled features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# Predictions like app.py
df['rf_pred'] = rf.predict(X_scaled)
df['ridge_pred'] = ridge.predict(X_scaled)

# Baseline 1/seed_tournament (or nearest seed-like column)
seed_col = 'seed_tournament' if 'seed_tournament' in df.columns else None
if seed_col is None:
    for c in df.columns:
        if 'seed' in c.lower():
            seed_col = c
            break

def compute_baseline(row):
    if seed_col and not pd.isna(row.get(seed_col)):
        try:
            s = float(row.get(seed_col))
            return 0.0 if s <= 0 else 1.0 / s
        except Exception:
            return 0.0
    return 0.0

df['baseline_pred'] = df.apply(compute_baseline, axis=1).astype(float)

# Feature importances (RF)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})

api = Flask(__name__)
CORS(api, resources={r"/api/*": {"origins": "*"}})

@api.get("/api/seasons")
def seasons():
    ss = sorted([int(s) for s in df['season'].dropna().unique().tolist()])
    return jsonify({"seasons": ss})


def season_slice(season: int) -> pd.DataFrame:
    return df[df['season'] == int(season)].copy()


def as_percent(series: pd.Series) -> pd.Series:
    return (series.astype(float) * 100.0).round(2)


@api.get("/api/leaderboard")
def leaderboard():
    season = request.args.get("season", type=int, default=None)
    model = request.args.get("model", default="rf")  # rf | ridge | baseline
    if season is None:
        season = int(sorted(df['season'].dropna().unique())[-1])
    d = season_slice(season)
    pred_map = {"rf": "rf_pred", "ridge": "ridge_pred", "baseline": "baseline_pred"}
    pred_col = pred_map.get(model, "rf_pred")
    d['win_probability'] = as_percent(d[pred_col])
    if 'champion' not in d.columns:
        d['champion'] = 0
    out = d.sort_values('win_probability', ascending=False)[['Team','conference','win_probability','champion']].head(50)
    return jsonify({"season": season, "model": model, "leaderboard": out.to_dict(orient="records")})


@api.get("/api/model_comparison")
def model_comparison():
    season = request.args.get("season", type=int, default=None)
    if season is None:
        season = int(sorted(df['season'].dropna().unique())[-1])
    d = season_slice(season)
    # Top 10 by rf_pred just like the dashboard
    top = d.nlargest(10, 'rf_pred')[['Team','rf_pred','ridge_pred','baseline_pred']].copy()
    top.rename(columns={'rf_pred':'pred_rf','ridge_pred':'pred_ridge','baseline_pred':'pred_baseline'}, inplace=True)
    for c in ['pred_rf','pred_ridge','pred_baseline']:
        top[c] = as_percent(top[c])
    return jsonify({"season": season, "rows": top.to_dict(orient="records")})


@api.get("/api/conferences")
def conferences():
    season = request.args.get("season", type=int, default=None)
    if season is None:
        season = int(sorted(df['season'].dropna().unique())[-1])
    d = season_slice(season)
    g = d.groupby('conference')[['rf_pred','ridge_pred','baseline_pred']].mean().reset_index()
    g.rename(columns={'rf_pred':'pred_rf','ridge_pred':'pred_ridge','baseline_pred':'pred_baseline'}, inplace=True)
    for c in ['pred_rf','pred_ridge','pred_baseline']:
        g[c] = as_percent(g[c])
    g = g.sort_values('pred_rf', ascending=False)
    return jsonify({"season": season, "conferences": g.to_dict(orient='records')})


@api.get("/api/feature_importance")
def feature_importance():
    # Return top 25 features by importance
    top = importance_df.sort_values('importance', ascending=False).head(25)
    return jsonify({"features": top.to_dict(orient='records')})


@api.get("/")
def root():
    return "ok", 200


@api.get("/api/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    api.run(host="0.0.0.0", port=8000, debug=True)
