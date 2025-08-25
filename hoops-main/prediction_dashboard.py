# -*- coding: utf-8 -*-
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import traceback

try:
    # Load data
    print("Loading data...")
    raw_df = pd.read_csv("NCAA.csv", encoding='latin1')
    print("Data loaded successfully!")
    print(f"Columns in dataset: {raw_df.columns.tolist()}")

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
    print("Preparing features...")
    features = df.drop(columns=['season', 'Team', 'conference', 'champion'])
    X = features.select_dtypes(include='number')
    y = df['champion']

    # Fit Random Forest
    print("Training Random Forest model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    df['rf_pred'] = rf.predict(X_scaled)

    # Fit Ridge Regression
    print("Training Ridge Regression model...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    df['ridge_pred'] = ridge.predict(X_scaled)

    # Seed-based baseline
    print("Calculating seed-based baseline...")
    df['baseline_pred'] = 1 / df['seed_tournament']

    # Initialize Dash app
    print("Initializing Dash app...")
    app = Dash(__name__)
    server = app.server  # This is important for deployment

    app.layout = html.Div([
        html.H1("🏀 NCAA Championship Probability Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
        
        html.Div([
            html.Div([
                html.Label("Select Season:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='season-dropdown',
                    options=[{'label': str(s), 'value': s} for s in sorted(df['season'].unique())],
                    value=sorted(df['season'].unique())[-1],
                    style={'width': '100%'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
        ], style={'padding': '20px'}),

        html.Div([
            html.Div([
                html.H3("Random Forest Model Predictions", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dash_table.DataTable(
                    id='rf-table',
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': '#2c3e50',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{champion} = 1'},
                            'backgroundColor': '#FFD700',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ],
                    page_size=10
                ),
            ], style={'width': '33%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.H3("Ridge Regression Predictions", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dash_table.DataTable(
                    id='ridge-table',
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': '#2c3e50',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{champion} = 1'},
                            'backgroundColor': '#FFD700',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ],
                    page_size=10
                ),
            ], style={'width': '33%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.H3("Seed-Based Predictions", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dash_table.DataTable(
                    id='baseline-table',
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': '#2c3e50',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{champion} = 1'},
                            'backgroundColor': '#FFD700',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ],
                    page_size=10
                ),
            ], style={'width': '33%', 'display': 'inline-block', 'margin': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),

        html.Div([
            html.Div([
                html.H3("Model Comparison", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='model-comparison')
            ], style={'width': '100%', 'margin': '20px'}),
        ]),

        html.Div([
            html.Div([
                html.H3("Conference Analysis", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='conference-analysis')
            ], style={'width': '100%', 'margin': '20px'}),
        ])
    ])

    @app.callback(
        Output('rf-table', 'data'),
        Output('rf-table', 'columns'),
        Output('ridge-table', 'data'),
        Output('ridge-table', 'columns'),
        Output('baseline-table', 'data'),
        Output('baseline-table', 'columns'),
        Input('season-dropdown', 'value')
    )
    def update_tables(season):
        try:
            d = df[df['season'] == season].copy()
            
            # Function to prepare table data
            def prepare_table_data(pred_col):
                d_copy = d.copy()
                d_copy['win_probability'] = (d_copy[pred_col] * 100).round(2)
                d_sorted = d_copy.sort_values(by='win_probability', ascending=False)
                cols = ["Team", "conference", "win_probability", "champion"]
                d_sorted = d_sorted[cols]
                columns = [
                    {"name": "Team", "id": "Team"},
                    {"name": "Conference", "id": "conference"},
                    {"name": "Win Probability (%)", "id": "win_probability"},
                    {"name": "Champion", "id": "champion"}
                ]
                return d_sorted.to_dict('records'), columns
            
            # Get data for each model
            rf_data, rf_cols = prepare_table_data('rf_pred')
            ridge_data, ridge_cols = prepare_table_data('ridge_pred')
            baseline_data, baseline_cols = prepare_table_data('baseline_pred')
            
            return rf_data, rf_cols, ridge_data, ridge_cols, baseline_data, baseline_cols
            
        except Exception as e:
            print(f"Error in update_tables: {str(e)}")
            print(traceback.format_exc())
            return [], [], [], [], [], []

    @app.callback(
        Output('model-comparison', 'figure'),
        Input('season-dropdown', 'value')
    )
    def update_model_comparison(season):
        try:
            d = df[df['season'] == season].copy()
            
            # Get top 10 teams by Random Forest prediction
            top_teams = d.nlargest(10, 'rf_pred')['Team'].tolist()
            d_filtered = d[d['Team'].isin(top_teams)]
            
            # Create comparison plot
            fig = px.bar(
                d_filtered,
                x='Team',
                y=['rf_pred', 'ridge_pred', 'baseline_pred'],
                title=f'Top 10 Teams: Model Comparison ({season})',
                barmode='group',
                labels={
                    'value': 'Championship Probability',
                    'variable': 'Model',
                    'Team': 'Team'
                }
            )
            
            fig.update_layout(
                xaxis_title="Team",
                yaxis_title="Championship Probability",
                xaxis_tickangle=45,
                legend_title="Model"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error in update_model_comparison: {str(e)}")
            print(traceback.format_exc())
            return {}

    @app.callback(
        Output('conference-analysis', 'figure'),
        Input('season-dropdown', 'value')
    )
    def update_conference_analysis(season):
        try:
            d = df[df['season'] == season].copy()
            
            # Calculate average probability by conference for each model
            conf_analysis = d.groupby('conference')[['rf_pred', 'ridge_pred', 'baseline_pred']].mean().reset_index()
            conf_analysis = conf_analysis.sort_values(by='rf_pred', ascending=False)
            
            fig = px.bar(
                conf_analysis,
                x='conference',
                y=['rf_pred', 'ridge_pred', 'baseline_pred'],
                title=f'Conference Analysis ({season})',
                barmode='group',
                labels={
                    'value': 'Average Championship Probability',
                    'variable': 'Model',
                    'conference': 'Conference'
                }
            )
            
            fig.update_layout(
                xaxis_title="Conference",
                yaxis_title="Average Championship Probability",
                xaxis_tickangle=45,
                legend_title="Model"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error in update_conference_analysis: {str(e)}")
            print(traceback.format_exc())
            return {}

    if __name__ == '__main__':
        print("Starting the Dash server...")
        app.run(debug=True, host='0.0.0.0', port=8050)

except Exception as e:
    print(f"Error in main execution: {str(e)}")
    print(traceback.format_exc()) 