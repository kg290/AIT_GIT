"""
Medical AI Gateway 2.0 - Main Application Entry Point

A medical intelligence layer that turns messy documents into
structured, explainable, time-aware clinical insight.
"""
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.config import settings
from backend.database import init_db
from backend.api import documents, patients, query, analytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical AI Gateway 2.0",
    description="""
    A comprehensive medical document processing system that provides:
    
    * **Document Intake** - Multi-format support (PDF, images)
    * **OCR Processing** - Google Vision AI with confidence tracking
    * **Text Cleaning** - Medical abbreviation expansion, error correction
    * **Entity Extraction** - Medications, dosages, diagnoses, symptoms
    * **Drug Normalization** - Brand to generic mapping
    * **Prescription Structuring** - Free-text to structured format
    * **Drug Safety Analysis** - Interaction checking, allergy alerts
    * **Temporal Reasoning** - Timeline building, medication tracking
    * **Knowledge Graph** - Patient-medication-condition relationships
    * **Conversational Querying** - Natural language medical queries
    * **Audit & Compliance** - Complete audit trail
    * **Analytics Dashboard** - Visualization and insights
    """,
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(documents.router, prefix="/api")
app.include_router(patients.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.AUDIT_LOG_PATH.mkdir(parents=True, exist_ok=True)

# Mount static files
static_dir = Path(__file__).parent / "frontend" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    logger.info("Starting Medical AI Gateway 2.0...")
    init_db()
    logger.info("Database initialized")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"API documentation available at /api/docs")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend application"""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical AI Gateway 2.0</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2563eb; }
            .links { margin-top: 30px; }
            .links a { display: block; margin: 10px 0; color: #2563eb; }
        </style>
    </head>
    <body>
        <h1>🏥 Medical AI Gateway 2.0</h1>
        <p>A medical intelligence layer that turns messy documents into structured, explainable, time-aware clinical insight.</p>
        
        <div class="links">
            <h3>Quick Links:</h3>
            <a href="/api/docs">📚 API Documentation (Swagger)</a>
            <a href="/api/redoc">📖 API Documentation (ReDoc)</a>
            <a href="/dashboard">📊 Dashboard</a>
        </div>
        
        <div style="margin-top: 30px;">
            <h3>Features:</h3>
            <ul>
                <li>Multi-format document intake (PDF, images)</li>
                <li>Advanced OCR with Google Vision AI</li>
                <li>Medical entity extraction</li>
                <li>Drug interaction checking</li>
                <li>Patient timeline building</li>
                <li>Knowledge graph relationships</li>
                <li>Natural language querying</li>
                <li>Complete audit trail</li>
            </ul>
        </div>
    </body>
    </html>
    """)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "database": "ok",
            "ocr": "ok",
            "knowledge_graph": "ok"
        }
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard page"""
    dashboard_path = Path(__file__).parent / "frontend" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    
    # Return inline dashboard if file doesn't exist
    return FileResponse(Path(__file__).parent / "frontend" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
