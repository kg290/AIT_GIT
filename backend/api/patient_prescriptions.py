"""
Automated Patient Prescription API
Handles multi-prescription upload and automatic timeline building
Uses unified database service for all operations
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import uuid
import shutil
from datetime import datetime

from backend.config import settings
from backend.services.complete_processor import complete_processor
from backend.services.unified_patient_service import get_unified_patient_service

router = APIRouter(prefix="/api/v2/patient-prescriptions", tags=["Patient Prescriptions"])


@router.post("/scan")
async def scan_and_save_prescription(
    file: UploadFile = File(...),
    patient_id: str = Form(..., description="Patient ID (create new or use existing)"),
    patient_name: Optional[str] = Form(None, description="Patient name (for new patients)"),
    allergies: Optional[str] = Form(None, description="Comma-separated allergies"),
    conditions: Optional[str] = Form(None, description="Comma-separated chronic conditions")
):
    """
    Scan a prescription and automatically:
    1. Extract all data using OCR + AI
    2. Save/update patient record
    3. Add medications to patient's list
    4. Track changes from previous prescriptions
    5. Build timeline automatically
    6. Run safety analysis
    
    Returns complete analysis with changes detected.
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
    
    # Parse allergies and conditions
    patient_allergies = []
    if allergies:
        patient_allergies = [a.strip() for a in allergies.split(',') if a.strip()]
    
    patient_conditions = []
    if conditions:
        patient_conditions = [c.strip() for c in conditions.split(',') if c.strip()]
    
    try:
        # Process document using OCR + AI
        result = complete_processor.process(
            file_path=str(file_path),
            patient_allergies=patient_allergies,
            patient_id=patient_id,
            save_to_db=False
        )
        
        # Convert to dict
        prescription_data = result.to_dict()
        prescription_data['filename'] = file.filename
        prescription_data['allergies'] = patient_allergies
        
        # Use extracted UHID/patient_id if available, otherwise use the form input
        # This captures IDs like "UHID 23672" from the prescription
        extracted_patient_id = result.patient_id
        if extracted_patient_id and extracted_patient_id.strip():
            # Use extracted ID (UHID from prescription)
            final_patient_id = extracted_patient_id.strip()
            prescription_data['extracted_uhid'] = final_patient_id
        else:
            # Use provided patient_id from form
            final_patient_id = patient_id
        
        # Override patient name if provided
        if patient_name:
            prescription_data['patient_name'] = patient_name
        
        # Use UNIFIED service for database operations
        service = get_unified_patient_service()
        
        # Create/update patient in database
        patient = service.get_or_create_patient(
            patient_uid=final_patient_id,
            name=prescription_data.get('patient_name'),
            age=prescription_data.get('patient_age'),
            gender=prescription_data.get('patient_gender'),
            phone=prescription_data.get('patient_phone'),
            address=prescription_data.get('patient_address'),
            allergies=patient_allergies,
            conditions=patient_conditions
        )
        
        # Add prescription to database
        add_result = service.add_prescription(final_patient_id, prescription_data)
        
        # Get updated patient summary from database
        patient_summary = service.get_patient_summary(final_patient_id)
        
        # Get timeline from database
        timeline = service.get_patient_timeline(final_patient_id, limit=10)
        
        return JSONResponse(content={
            'success': True,
            'message': f"Prescription #{add_result.get('prescription_number', 1)} added successfully",
            'prescription_data': prescription_data,
            'patient_id': final_patient_id,
            'extracted_uhid': extracted_patient_id,
            'analysis': add_result,
            'patient_summary': patient_summary,
            'timeline': timeline
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/scan-multiple")
async def scan_multiple_prescriptions(
    files: List[UploadFile] = File(...),
    patient_id: str = Form(...),
    patient_name: Optional[str] = Form(None),
    allergies: Optional[str] = Form(None),
    conditions: Optional[str] = Form(None)
):
    """
    Upload multiple prescription images/PDFs at once.
    Each will be processed in order and added to the patient's timeline.
    
    Ideal for uploading a patient's complete prescription history.
    """
    allowed_types = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
    
    # Validate all files first
    for f in files:
        file_ext = os.path.splitext(f.filename)[1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File {f.filename} has unsupported type. Allowed: {', '.join(allowed_types)}"
            )
    
    # Parse allergies and conditions
    patient_allergies = []
    if allergies:
        patient_allergies = [a.strip() for a in allergies.split(',') if a.strip()]
    
    patient_conditions = []
    if conditions:
        patient_conditions = [c.strip() for c in conditions.split(',') if c.strip()]
    
    service = get_unified_patient_service()
    
    # Create/update patient first in database
    service.get_or_create_patient(
        patient_uid=patient_id,
        name=patient_name,
        allergies=patient_allergies,
        conditions=patient_conditions
    )
    
    results = []
    errors = []
    
    # Process each file
    for idx, file in enumerate(files):
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_id = str(uuid.uuid4())
        file_path = settings.UPLOAD_DIR / f"{file_id}{file_ext}"
        
        try:
            # Save file
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process with OCR
            result = complete_processor.process(
                file_path=str(file_path),
                patient_allergies=patient_allergies,
                patient_id=patient_id,
                save_to_db=False
            )
            
            prescription_data = result.to_dict()
            prescription_data['filename'] = file.filename
            prescription_data['allergies'] = patient_allergies
            
            # Use extracted UHID if available
            extracted_id = result.patient_id
            final_patient_id = extracted_id.strip() if extracted_id and extracted_id.strip() else patient_id
            
            if patient_name and not prescription_data.get('patient_name'):
                prescription_data['patient_name'] = patient_name
            
            # Add to database
            add_result = service.add_prescription(final_patient_id, prescription_data)
            
            results.append({
                'file': file.filename,
                'success': True,
                'prescription_uid': add_result.get('prescription_uid'),
                'medications_added': add_result.get('medications_added', [])
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append({
                'file': file.filename,
                'success': False,
                'error': str(e)
            })
    
    # Get final summary from database
    patient_summary = service.get_patient_summary(patient_id)
    timeline = service.get_patient_timeline(patient_id)
    
    return JSONResponse(content={
        'success': len(errors) == 0,
        'message': f"Processed {len(results)} of {len(files)} prescriptions",
        'results': results,
        'errors': errors,
        'patient_summary': patient_summary,
        'complete_timeline': timeline
    })


@router.get("/patients")
async def list_patients():
    """Get list of all patients with prescription counts"""
    service = get_unified_patient_service()
    return service.get_all_patients()


@router.get("/patients/{patient_id}")
async def get_patient_details(patient_id: str):
    """Get complete patient profile with all history"""
    service = get_unified_patient_service()
    patient = service.get_patient_by_uid(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient


@router.get("/patients/{patient_id}/summary")
async def get_patient_summary(patient_id: str):
    """Get comprehensive patient summary with statistics"""
    service = get_unified_patient_service()
    summary = service.get_patient_summary(patient_id)
    
    if 'error' in summary:
        raise HTTPException(status_code=404, detail=summary['error'])
    
    return summary


@router.get("/patients/{patient_id}/timeline")
async def get_patient_timeline(
    patient_id: str,
    limit: int = Query(100, description="Maximum events to return"),
    event_type: Optional[str] = Query(None, description="Filter by event type")
):
    """Get patient's complete medical timeline"""
    service = get_unified_patient_service()
    timeline = service.get_patient_timeline(patient_id, limit=limit)
    
    if event_type:
        timeline = [t for t in timeline if t['event_type'] == event_type]
    
    return {
        'patient_id': patient_id,
        'total_events': len(timeline),
        'timeline': timeline
    }


@router.get("/patients/{patient_id}/medications")
async def get_patient_medications(
    patient_id: str,
    include_historical: bool = Query(False, description="Include discontinued medications")
):
    """Get patient's current and optionally historical medications"""
    service = get_unified_patient_service()
    
    medications = service.get_patient_medications(patient_id, active_only=not include_historical)
    
    active_meds = [m for m in medications if m.get('is_active', True)]
    historical_meds = [m for m in medications if not m.get('is_active', True)]
    
    result = {
        'patient_id': patient_id,
        'current_medications': active_meds
    }
    
    if include_historical:
        result['historical_medications'] = historical_meds
    
    return result


@router.get("/patients/{patient_id}/prescriptions")
async def get_patient_prescriptions(patient_id: str):
    """Get all prescriptions for a patient"""
    service = get_unified_patient_service()
    prescriptions = service.get_patient_prescriptions(patient_id)
    
    return {
        'patient_id': patient_id,
        'total_prescriptions': len(prescriptions),
        'prescriptions': prescriptions
    }


@router.get("/patients/{patient_id}/safety")
async def get_patient_safety_analysis(patient_id: str):
    """Get current safety analysis for patient's medications"""
    service = get_unified_patient_service()
    patient = service.get_patient_by_uid(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get medications and allergies
    medications = service.get_patient_medications(patient_id)
    allergies = service.get_patient_allergies(patient_id)
    conditions = service.get_patient_conditions(patient_id)
    
    return {
        'patient_id': patient_id,
        'current_medications': [m['name'] for m in medications],
        'allergies': allergies,
        'conditions': conditions
    }


@router.post("/patients/{patient_id}/allergies")
async def add_patient_allergy(
    patient_id: str,
    allergy: str = Form(...)
):
    """Add an allergy to patient's record"""
    service = get_unified_patient_service()
    
    # Update patient with new allergy
    result = service.get_or_create_patient(
        patient_uid=patient_id,
        allergies=[allergy]
    )
    
    allergies = service.get_patient_allergies(patient_id)
    
    return {'success': True, 'allergies': allergies}


@router.post("/patients/{patient_id}/conditions")
async def add_patient_condition(
    patient_id: str,
    condition: str = Form(...)
):
    """Add a chronic condition to patient's record"""
    service = get_unified_patient_service()
    
    # Update patient with new condition
    result = service.get_or_create_patient(
        patient_uid=patient_id,
        conditions=[condition]
    )
    
    conditions = service.get_patient_conditions(patient_id)
    
    return {'success': True, 'conditions': conditions}
