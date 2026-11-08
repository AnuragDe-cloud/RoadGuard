
from fastapi import FastAPI

app = FastAPI(
    title="RoadGuard API",
    description="AI-powered Road Safety and Incident Intelligence System",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to RoadGuard API!",
        "status": "Running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "RoadGuard"
    }
