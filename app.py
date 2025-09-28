# Import the FastAPI app from api/index.py
from api.index import app

# This is the main entrypoint that Vercel will use
# The app is already configured with CORS and all endpoints

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
