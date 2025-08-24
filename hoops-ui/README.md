# Hoops UI (React + Recharts)

A modern, Robinhood-style frontend for your NCAA probability models.

## Run the backend API
```bash
python api_server.py  # serves on http://localhost:8000
```

## Run the frontend
```bash
cd hoops-ui
npm install
# set the API URL if your backend isn't on localhost:8000
# echo 'VITE_API_URL="http://localhost:8000"' > .env.local
npm run dev
```

Open the URL printed by Vite.
