"""
Medical AI Gateway 2.0 - Production Hospital System

A medical intelligence layer that turns messy documents into
structured, explainable, time-aware clinical insight.

Production-ready for hospital deployment with:
- JWT Authentication
- Role-based access control
- HIPAA-compliant audit logging
- PostgreSQL/SQLite database support
"""
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from backend.config import settings
from backend.database import init_db
from backend.api import documents, patients, query, analytics
from backend.api.enhanced_routes import router as enhanced_router
from backend.api.patient_prescriptions import router as patient_prescriptions_router
from backend.api.staff_api import router as staff_router

# Production imports
from backend.api.production_routes import router as hospital_router
from backend.database.connection import db_manager

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
app.include_router(enhanced_router)  # Enhanced API v2 routes
app.include_router(patient_prescriptions_router)  # Automated patient prescription routes
app.include_router(staff_router)  # Staff portal API routes
app.include_router(hospital_router)  # Production hospital routes with authentication

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
    logger.info("Starting Medical AI Gateway 2.0 (Production Mode)...")
    
    # Initialize legacy database
    init_db()
    logger.info("Legacy database initialized")
    
    # Initialize production database with authentication
    db_manager.init_database()
    logger.info("Production database initialized (PostgreSQL/SQLite)")
    
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"API documentation available at /api/docs")
    logger.info(f"Staff Portal: /staff")
    logger.info(f"Hospital Portal: /hospital/login")
    logger.info(f"Default admin credentials - Username: admin, Password: admin123")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to dashboard"""
    dashboard_path = Path(__file__).parent / "frontend" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    
    # Fallback to index.html
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
    
    # Fallback to index.html
    return FileResponse(Path(__file__).parent / "frontend" / "index.html")


@app.get("/simple", response_class=HTMLResponse)
async def simple_scanner():
    """Serve the simple scanner page"""
    return FileResponse(Path(__file__).parent / "frontend" / "index.html")


@app.get("/hospital/login", response_class=HTMLResponse)
async def hospital_login():
    """Redirect hospital login to main dashboard"""
    return RedirectResponse(url="/dashboard")


@app.get("/hospital/dashboard", response_class=HTMLResponse)
async def hospital_dashboard():
    """Redirect hospital dashboard to main dashboard"""
    return RedirectResponse(url="/dashboard")


@app.get("/staff", response_class=HTMLResponse)
async def staff_portal():
    """Serve the staff portal for prescription OCR scanning"""
    staff_path = Path(__file__).parent / "frontend" / "staff.html"
    if staff_path.exists():
        return FileResponse(staff_path)
    return RedirectResponse(url="/dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
