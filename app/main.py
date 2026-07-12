# pyrefly: ignore [missing-import]
import uvicorn
import sys
import os
# pyrefly: ignore [missing-import]
import uvicorn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    app = FastAPI(
        title="Network Intrusion Detection System",
        description="A production-grade AI REST API for predicting cyber attacks from network flow data.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Setup CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Allow all origins for development, restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router)

    return app

# The main application instance used by Uvicorn
app = create_app()

if __name__ == "__main__":
   
    # Start the server locally
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
