# -*- coding: utf-8 -*-
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load data
raw_df = pd.read_csv("NCAA.csv", encoding='latin1')

# Preprocess: drop post-tournament info
drop_cols = [
    'round', 'champion_share', 'make_tournament',
    'ap_weighted_score', 'ap_sos', 'under_12_ap_rec', 'sum_under_12_ap_games'
]
drop_cols = [col for col in drop_cols if col in raw_df.columns]
df = raw_df.drop(columns=drop_cols)

# Drop duplicate season column if it exists
df = df.loc[:, ~df.columns.duplicated()]

# Prepare features
features = df.drop(columns=['season', 'Team', 'conference', 'champion'])
X = features.select_dtypes(include='number')
y = df['champion']

# Fit Random Forest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
df['rf_pred'] = rf.predict(X_scaled)

# Fit Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
df['ridge_pred'] = ridge.predict(X_scaled)

# Seed-based baseline
df['baseline_pred'] = 1 / df['seed_tournament']

# Feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})

# Initialize Dash app
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("🏀 NCAA Champion Predictions Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Season:"),
        dcc.Dropdown(id='season-dropdown', options=[
            {'label': str(s), 'value': s} for s in sorted(df['season'].unique())
        ], value=sorted(df['season'].unique())[-1]),

        html.Label("Model:"),
        dcc.RadioItems(id='model-choice', options=[
            {'label': 'Random Forest', 'value': 'rf'},
            {'label': 'Ridge Regression', 'value': 'ridge'},
            {'label': 'Seed-Based Baseline', 'value': 'baseline'}
        ], value='rf', inline=True),

        html.Label("Top N Teams to Display:"),
        dcc.Slider(id='top-n-slider', min=1, max=10, step=1, value=5,
                   marks={i: str(i) for i in range(1, 11)}),

        html.Label("Filter by Conference (optional):"),
        dcc.Dropdown(id='conf-filter', options=[
            {'label': c, 'value': c} for c in sorted(df['conference'].dropna().unique())
        ], multi=True)
    ], style={'padding': 20}),

    html.H3("Top Predicted Teams"),
    dash_table.DataTable(id='prediction-table',
                         style_table={'overflowX': 'auto'},
                         style_data_conditional=[{
                             'if': {'filter_query': '{champion} = 1'},
                             'backgroundColor': '#FFD700',
                             'fontWeight': 'bold'
                         }]),

    html.H3("Top Feature Importances (Random Forest)"),
    dcc.Slider(id='num-features', min=5, max=25, step=1, value=15,
               marks={i: str(i) for i in range(5, 26)}),
    dcc.Graph(id='importance-graph')
])

@app.callback(
    Output('prediction-table', 'data'),
    Output('prediction-table', 'columns'),
    Input('season-dropdown', 'value'),
    Input('model-choice', 'value'),
    Input('top-n-slider', 'value'),
    Input('conf-filter', 'value')
)
def update_table(season, model, top_n, confs):
    d = df[df['season'] == season]
    if confs:
        d = d[d['conference'].isin(confs)]
    pred_col = {'rf': 'rf_pred', 'ridge': 'ridge_pred', 'baseline': 'baseline_pred'}[model]
    d_sorted = d.sort_values(by=pred_col, ascending=False).head(top_n)
    cols = ["Team", pred_col, "champion"]
    return d_sorted[cols].to_dict('records'), [{"name": c, "id": c} for c in cols]

@app.callback(
    Output('importance-graph', 'figure'),
    Input('num-features', 'value')
)
def update_importance_chart(n):
    top_feats = importance_df.sort_values(by='importance', ascending=False).head(n)
    fig = px.bar(top_feats, x='importance', y='feature', orientation='h',
                 title='Feature Importances', labels={'importance': 'Importance'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

if __name__ == '__main__':
    app.run(debug=True)

