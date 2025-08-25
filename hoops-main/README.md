# NCAA Championship Probability Dashboard

This dashboard predicts the probability of teams winning the NCAA championship using machine learning models.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
python prediction_dashboard.py
```

3. Open http://localhost:8050 in your browser

## Deployment Instructions

### Deploying to Render.com

1. Create a new account on [Render.com](https://render.com)

2. Create a new Web Service:
   - Connect your GitHub repository
   - Select the repository containing this project
   - Choose "Python" as the runtime
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `gunicorn prediction_dashboard:server`
   - Choose the free tier

3. Click "Create Web Service"

The dashboard will be automatically deployed and you'll get a URL where it's accessible.

### Required Files for Deployment

- `prediction_dashboard.py`: The main application file
- `requirements.txt`: Python dependencies
- `Procfile`: Deployment configuration
- `NCAA.csv`: The dataset file

## Features

- Team rankings by championship probability
- Probability distribution visualization
- Conference analysis
- Multiple model comparison (Random Forest, Ridge Regression, Seed-Based Baseline) # college-hoops-precdictor
# college-hoops-precdictor
# college-hoops-precdictor
# college-hoops-precdictor
# vovovovo
# vovovovo
# college-hoops-precdictor
