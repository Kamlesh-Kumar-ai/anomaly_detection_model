from fastapi import FastAPI
from app.routes.proxy_routes import router as proxy_router

# Create FastAPI app
app = FastAPI(
    title="Proxy Detection API",
    description="ML-based Proxy Detection using LightGBM",
    version="1.0.0"
)

# Include routes
app.include_router(proxy_router)


# ========= ROOT ENDPOINT =========
@app.get("/")
def root():
    return {
        "message": "Proxy Detection API is running 🚀",
        "status": "success"
    }


# ========= HEALTH CHECK =========
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "proxy-detection",
    }