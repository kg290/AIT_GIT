"""
Document API Routes - Production-ready for hospital use
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import os
import uuid
import shutil
from datetime import datetime

from backend.config import settings
from backend.services.simple_processor import SimpleDocumentProcessor
from backend.services.prescription_extractor import PrescriptionExtractor
from backend.services.drug_interaction_service import DrugInteractionService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Query(None, description="Patient ID for context"),
    allergies: Optional[str] = Query(None, description="Comma-separated allergies")
):
    """
    Upload and process a prescription document
    
    Accepts: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP
    
    Returns:
        Extracted prescription data including:
        - Patient information (name, age, gender)
        - Prescription date
        - Doctor details
        - Medications with dosage, frequency, duration
        - Diagnosis and vitals
        - Drug interaction warnings
    """
    # Validate file type
    allowed_types = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = settings.UPLOAD_DIR / f"{file_id}{file_ext}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Parse allergies if provided
    patient_allergies = None
    if allergies:
        patient_allergies = [a.strip() for a in allergies.split(',')]
    
    # Process document
    try:
        processor = SimpleDocumentProcessor()
        result = processor.process(
            file_path=str(file_path),
            patient_allergies=patient_allergies
        )
        
        response_data = result.to_dict()
        response_data['filename'] = file.filename
        response_data['upload_time'] = datetime.utcnow().isoformat()
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/extract-text")
async def extract_from_text(
    text: str,
    allergies: Optional[str] = Query(None, description="Comma-separated allergies")
):
    """
    Extract prescription data from raw text (manual entry)
    """
    try:
        extractor = PrescriptionExtractor()
        extracted = extractor.extract(text)
        
        result = {
            'success': True,
            'patient_name': extracted.patient_name,
            'patient_age': extracted.patient_age,
            'patient_gender': extracted.patient_gender,
            'prescription_date': extracted.prescription_date,
            'doctor_name': extracted.doctor_name,
            'doctor_qualification': extracted.doctor_qualification,
            'clinic_name': extracted.clinic_name,
            'diagnosis': extracted.diagnosis,
            'vitals': extracted.vitals,
            'medications': [med.to_dict() for med in extracted.medications],
            'advice': extracted.advice,
            'follow_up': extracted.follow_up_date,
            'confidence': extracted.extraction_confidence
        }
        
        # Drug safety check
        if extracted.medications:
            drug_service = DrugInteractionService()
            patient_allergies = [a.strip() for a in allergies.split(',')] if allergies else []
            
            med_names = [m.name for m in extracted.medications if m.name]
            safety = drug_service.analyze_safety(med_names, patient_allergies)
            
            result['drug_interactions'] = [
                {
                    'drug1': i.drug1,
                    'drug2': i.drug2,
                    'severity': i.severity.value,
                    'description': i.description
                }
                for i in safety.interactions
            ]
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Medical AI Gateway"}
