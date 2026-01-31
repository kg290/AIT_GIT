"""
Document API Routes - Complete prescription processing endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import uuid
import shutil
from datetime import datetime

from backend.config import settings
from backend.services.complete_processor import CompleteDocumentProcessor, complete_processor
from backend.services.ai_extractor import AIExtractor
from backend.services.drug_database import find_all_interactions, find_allergy_alerts

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None, description="Patient ID for context"),
    allergies: Optional[str] = Form(None, description="Comma-separated allergies")
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
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Parse allergies if provided
    patient_allergies = None
    if allergies:
        patient_allergies = [a.strip() for a in allergies.split(',') if a.strip()]
    
    # Process document using complete processor
    try:
        result = complete_processor.process(
            file_path=str(file_path),
            patient_allergies=patient_allergies,
            patient_id=patient_id,
            save_to_db=False  # Disable DB for now until schema is ready
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
    text: str = Form(...),
    allergies: Optional[str] = Form(None, description="Comma-separated allergies")
):
    """
    Extract prescription data from raw text (manual entry)
    """
    try:
        # Parse allergies
        patient_allergies = None
        if allergies:
            patient_allergies = [a.strip() for a in allergies.split(',') if a.strip()]
        
        # Use AI extractor
        extractor = AIExtractor()
        extracted = extractor.extract(text)
        
        # Build medications list
        medications = [
            med.to_dict() if hasattr(med, 'to_dict') else med 
            for med in extracted.medications
        ]
        
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
            'medications': medications,
            'advice': extracted.advice,
            'follow_up': extracted.follow_up,
            'confidence': extracted.confidence,
            'extraction_method': extracted.extraction_method
        }
        
        # Drug safety check
        if medications:
            med_names = [m.get('name', '') for m in medications if m.get('name')]
            
            interactions = find_all_interactions(med_names)
            result['drug_interactions'] = [
                {
                    'drug1': i.drug1,
                    'drug2': i.drug2,
                    'severity': i.severity.value,
                    'description': i.description
                }
                for i in interactions
            ]
            
            if patient_allergies:
                allergy_alerts = find_allergy_alerts(med_names, patient_allergies)
                result['allergy_alerts'] = [
                    {
                        'drug': a.drug,
                        'allergen': a.allergen,
                        'cross_reactivity': a.cross_reactivity
                    }
                    for a in allergy_alerts
                ]
        
        return JSONResponse(content=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Medical AI Gateway"}
