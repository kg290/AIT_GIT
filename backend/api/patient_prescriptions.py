"""
Automated Patient Prescription API
Handles multi-prescription upload and automatic timeline building
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
from backend.services.patient_prescription_service import get_patient_prescription_service

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
        
        # Override patient name if provided
        if patient_name:
            prescription_data['patient_name'] = patient_name
        
        # Get service and add prescription
        service = get_patient_prescription_service()
        
        # Update patient with allergies and conditions
        patient = service.get_or_create_patient(
            patient_id=patient_id,
            name=prescription_data.get('patient_name'),
            allergies=patient_allergies,
            chronic_conditions=patient_conditions
        )
        
        # Add prescription and get analysis
        add_result = service.add_prescription(patient_id, prescription_data)
        
        # Get updated patient data
        patient_summary = service.get_patient_summary(patient_id)
        
        return JSONResponse(content={
            'success': True,
            'message': f"Prescription #{add_result['prescription_number']} added successfully",
            'prescription_data': prescription_data,
            'analysis': add_result,
            'patient_summary': patient_summary,
            'timeline': service.get_patient_timeline(patient_id)[:10]  # Last 10 events
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
    
    service = get_patient_prescription_service()
    
    # Create/update patient first
    service.get_or_create_patient(
        patient_id=patient_id,
        name=patient_name,
        allergies=patient_allergies,
        chronic_conditions=patient_conditions
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
            
            if patient_name and not prescription_data.get('patient_name'):
                prescription_data['patient_name'] = patient_name
            
            # Add to patient
            add_result = service.add_prescription(patient_id, prescription_data)
            
            results.append({
                'file': file.filename,
                'success': True,
                'prescription_number': add_result['prescription_number'],
                'changes': add_result['changes_detected'],
                'safety': add_result['safety_analysis']
            })
            
        except Exception as e:
            errors.append({
                'file': file.filename,
                'success': False,
                'error': str(e)
            })
    
    # Get final summary
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
    service = get_patient_prescription_service()
    return service.get_all_patients()


@router.get("/patients/{patient_id}")
async def get_patient_details(patient_id: str):
    """Get complete patient profile with all history"""
    service = get_patient_prescription_service()
    patient = service.get_patient(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient.to_dict()


@router.get("/patients/{patient_id}/summary")
async def get_patient_summary(patient_id: str):
    """Get comprehensive patient summary with statistics"""
    service = get_patient_prescription_service()
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
    service = get_patient_prescription_service()
    timeline = service.get_patient_timeline(patient_id)
    
    if event_type:
        timeline = [t for t in timeline if t['event_type'] == event_type]
    
    return {
        'patient_id': patient_id,
        'total_events': len(timeline),
        'timeline': timeline[:limit]
    }


@router.get("/patients/{patient_id}/medications")
async def get_patient_medications(
    patient_id: str,
    include_historical: bool = Query(False, description="Include discontinued medications")
):
    """Get patient's current and optionally historical medications"""
    service = get_patient_prescription_service()
    patient = service.get_patient(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    result = {
        'patient_id': patient_id,
        'current_medications': [m.to_dict() for m in patient.current_medications]
    }
    
    if include_historical:
        result['historical_medications'] = [m.to_dict() for m in patient.historical_medications]
    
    return result


@router.get("/patients/{patient_id}/prescriptions")
async def get_patient_prescriptions(patient_id: str):
    """Get all prescriptions for a patient"""
    service = get_patient_prescription_service()
    patient = service.get_patient(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return {
        'patient_id': patient_id,
        'total_prescriptions': len(patient.prescriptions),
        'prescriptions': [p.to_dict() for p in patient.prescriptions]
    }


@router.get("/patients/{patient_id}/safety")
async def get_patient_safety_analysis(patient_id: str):
    """Get current safety analysis for patient's medications"""
    service = get_patient_prescription_service()
    patient = service.get_patient(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Re-run safety analysis
    safety = service._run_safety_analysis(patient)
    
    return {
        'patient_id': patient_id,
        'current_medications': [m.name for m in patient.current_medications],
        'allergies': patient.allergies,
        'conditions': patient.chronic_conditions,
        'safety_analysis': safety,
        'active_alerts': patient.safety_alerts
    }


@router.post("/patients/{patient_id}/allergies")
async def add_patient_allergy(
    patient_id: str,
    allergy: str = Form(...)
):
    """Add an allergy to patient's record"""
    service = get_patient_prescription_service()
    patient = service.get_patient(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if allergy not in patient.allergies:
        patient.allergies.append(allergy)
        
        # Re-run safety analysis
        service._run_safety_analysis(patient)
    
    return {'success': True, 'allergies': patient.allergies}


@router.post("/patients/{patient_id}/conditions")
async def add_patient_condition(
    patient_id: str,
    condition: str = Form(...)
):
    """Add a chronic condition to patient's record"""
    service = get_patient_prescription_service()
    patient = service.get_patient(patient_id)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if condition not in patient.chronic_conditions:
        patient.chronic_conditions.append(condition)
        
        # Re-run safety analysis
        service._run_safety_analysis(patient)
    
    return {'success': True, 'conditions': patient.chronic_conditions}
