"""
Staff Portal API
Handles patient registration, QR code generation, and prescription management
"""
import os
import uuid
import shutil
import logging
import io
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image

from backend.config import settings
from backend.services.complete_processor import complete_processor
from backend.services.unified_patient_service import get_unified_patient_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/staff", tags=["Staff Portal"])


def decode_qr_from_image(image_bytes: bytes) -> Optional[str]:
    """Decode QR code from image bytes using pyzbar"""
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        import cv2
        import numpy as np
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            # Try with PIL
            pil_img = Image.open(io.BytesIO(image_bytes))
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Decode QR codes
        decoded_objects = pyzbar_decode(img)
        
        if decoded_objects:
            # Return the first QR code data
            return decoded_objects[0].data.decode('utf-8')
        
        # Try with grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar_decode(gray)
        
        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8')
        
        return None
        
    except ImportError as e:
        logger.error(f"QR decoding libraries not installed: {e}")
        raise HTTPException(
            status_code=500,
            detail="QR code decoding libraries not installed. Please install pyzbar and opencv-python."
        )
    except Exception as e:
        logger.error(f"Error decoding QR code: {e}")
        return None


def generate_patient_uid() -> str:
    """Generate a unique patient identifier"""
    # Format: PTYYYYMMDD-XXXX (e.g., PT20260130-A1B2)
    date_part = datetime.now().strftime("%Y%m%d")
    random_part = uuid.uuid4().hex[:4].upper()
    return f"PT{date_part}-{random_part}"


@router.post("/decode-qr")
async def decode_qr_code(
    file: UploadFile = File(..., description="QR code image file")
):
    """
    Decode a QR code from an uploaded image file.
    
    Useful for Windows users who want to upload a QR code image
    instead of using camera scanning.
    
    Returns the decoded patient UID and patient info if found.
    """
    # Validate file type
    allowed_types = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Decode QR code
        decoded_data = decode_qr_from_image(image_bytes)
        
        if not decoded_data:
            raise HTTPException(
                status_code=400,
                detail="No QR code found in the image. Please ensure the image contains a clear QR code."
            )
        
        # The QR code contains the patient UID
        patient_uid = decoded_data.strip()
        
        # Try to look up the patient
        service = get_unified_patient_service()
        patient = service.get_patient_by_uid(patient_uid)
        
        if patient:
            summary = service.get_patient_summary(patient_uid)
            active_meds = summary.get('active_medications', []) if summary else []
            
            return JSONResponse(content={
                'success': True,
                'decoded_uid': patient_uid,
                'patient_found': True,
                'patient': {
                    'uid': patient_uid,
                    'name': patient.get('name') or f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip(),
                    'age': patient.get('age'),
                    'gender': patient.get('gender'),
                    'phone': patient.get('phone'),
                    'allergies': patient.get('allergies', []),
                    'conditions': patient.get('conditions', []),
                    'prescriptions_count': summary.get('total_prescriptions', 0) if summary else 0,
                    'active_medications': active_meds
                }
            })
        else:
            return JSONResponse(content={
                'success': True,
                'decoded_uid': patient_uid,
                'patient_found': False,
                'message': f"QR code decoded successfully. UID: {patient_uid}. Patient not found in database."
            })
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing QR code: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing QR code: {str(e)}")


@router.post("/create-patient")
async def create_patient_with_prescription(
    first_name: str = Form(..., description="Patient first name"),
    last_name: str = Form(..., description="Patient last name"),
    phone: str = Form(..., description="Patient phone number"),
    age: Optional[str] = Form(None, description="Patient age"),
    gender: Optional[str] = Form(None, description="Patient gender"),
    blood_group: Optional[str] = Form(None, description="Blood group"),
    email: Optional[str] = Form(None, description="Email address"),
    address: Optional[str] = Form(None, description="Full address"),
    allergies: Optional[str] = Form(None, description="Comma-separated allergies"),
    conditions: Optional[str] = Form(None, description="Comma-separated chronic conditions"),
    emergency_contact_name: Optional[str] = Form(None, description="Emergency contact name"),
    emergency_contact_phone: Optional[str] = Form(None, description="Emergency contact phone"),
    file: UploadFile = File(..., description="Prescription image or PDF")
):
    """
    Create a new patient with their first prescription.
    
    This endpoint:
    1. Creates a new patient record with a unique UID
    2. Processes the prescription using OCR + AI
    3. Saves both patient and prescription to the database
    4. Returns the patient UID for QR code generation
    
    The frontend will generate the QR code client-side using the returned UID.
    """
    # Validate file type
    allowed_types = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Generate unique patient UID
    patient_uid = generate_patient_uid()
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = settings.UPLOAD_DIR / f"{file_id}{file_ext}"
    
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Parse allergies and conditions
    patient_allergies = []
    if allergies:
        patient_allergies = [a.strip() for a in allergies.split(',') if a.strip()]
    
    patient_conditions = []
    if conditions:
        patient_conditions = [c.strip() for c in conditions.split(',') if c.strip()]
    
    try:
        # Process prescription using OCR + AI
        result = complete_processor.process(
            file_path=str(file_path),
            patient_allergies=patient_allergies,
            patient_id=patient_uid,
            save_to_db=False
        )
        
        prescription_data = result.to_dict()
        prescription_data['filename'] = file.filename
        prescription_data['allergies'] = patient_allergies
        
        # Get unified patient service
        service = get_unified_patient_service()
        
        # Parse age
        patient_age = None
        if age:
            try:
                patient_age = int(age)
            except ValueError:
                pass
        
        # Full patient name
        full_name = f"{first_name} {last_name}"
        
        # Create patient in database
        patient = service.get_or_create_patient(
            patient_uid=patient_uid,
            name=full_name,
            age=patient_age,
            gender=gender,
            phone=phone,
            address=address,
            allergies=patient_allergies,
            conditions=patient_conditions
        )
        
        # Update patient with additional info (email, blood group, emergency contact)
        _update_patient_extra_info(
            patient_uid=patient_uid,
            email=email,
            blood_group=blood_group,
            emergency_contact_name=emergency_contact_name,
            emergency_contact_phone=emergency_contact_phone
        )
        
        # Add prescription to database
        add_result = service.add_prescription(patient_uid, prescription_data)
        
        # Get updated patient summary
        patient_summary = service.get_patient_summary(patient_uid)
        
        return JSONResponse(content={
            'success': True,
            'message': 'Patient created and prescription processed successfully',
            'patient': {
                'uid': patient_uid,
                'name': full_name,
                'age': patient_age,
                'gender': gender,
                'phone': phone,
                'allergies': patient_allergies,
                'conditions': patient_conditions,
                'prescriptions_count': 1
            },
            'prescription': {
                'prescription_id': add_result.get('prescription_id'),
                'medications': prescription_data.get('medications', []),
                'diagnosis': prescription_data.get('diagnosis', []),
                'doctor_name': prescription_data.get('doctor_name'),
                'confidence': prescription_data.get('confidence')
            },
            'qr_data': {
                'uid': patient_uid,
                'name': full_name
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to create patient: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create patient: {str(e)}")


@router.get("/patient/{patient_uid}")
async def get_patient_by_uid(patient_uid: str):
    """
    Look up a patient by their UID (from QR code or manual entry).
    
    Returns patient information including:
    - Basic demographics
    - Allergies and conditions
    - Prescription count
    - Active medications
    """
    try:
        service = get_unified_patient_service()
        
        # Get patient from database
        patient = service.get_patient_by_uid(patient_uid)
        
        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient with UID '{patient_uid}' not found"
            )
        
        # Get patient summary for additional details
        summary = service.get_patient_summary(patient_uid)
        
        # Get active medications
        active_meds = []
        if summary and summary.get('active_medications'):
            active_meds = summary.get('active_medications', [])
        
        return JSONResponse(content={
            'success': True,
            'patient': {
                'uid': patient_uid,
                'name': patient.get('name') or f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip(),
                'age': patient.get('age'),
                'gender': patient.get('gender'),
                'phone': patient.get('phone'),
                'email': patient.get('email'),
                'address': patient.get('address'),
                'blood_group': patient.get('blood_group'),
                'allergies': patient.get('allergies', []),
                'conditions': patient.get('conditions', []),
                'prescriptions_count': summary.get('total_prescriptions', 0) if summary else 0,
                'active_medications': active_meds
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to lookup patient: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to lookup patient: {str(e)}")


@router.post("/add-prescription")
async def add_prescription_to_patient(
    patient_uid: str = Form(..., description="Patient UID"),
    file: UploadFile = File(..., description="Prescription image or PDF")
):
    """
    Add a single prescription to an existing patient.
    For multiple prescriptions, use /add-prescriptions endpoint.
    """
    # Validate file type
    allowed_types = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        service = get_unified_patient_service()
        
        # Verify patient exists
        patient = service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient with UID '{patient_uid}' not found"
            )
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = settings.UPLOAD_DIR / f"{file_id}{file_ext}"
        
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get patient allergies for safety checking
        patient_allergies = patient.get('allergies', [])
        
        # Process prescription using OCR + AI
        result = complete_processor.process(
            file_path=str(file_path),
            patient_allergies=patient_allergies,
            patient_id=patient_uid,
            save_to_db=False
        )
        
        # Build comprehensive prescription data for AI context
        prescription_data = _build_comprehensive_prescription_data(
            result, file.filename, patient_allergies, patient
        )
        
        # Add prescription to database
        add_result = service.add_prescription(patient_uid, prescription_data)
        
        # Get updated patient summary
        summary = service.get_patient_summary(patient_uid)
        
        return JSONResponse(content={
            'success': True,
            'message': f"Prescription #{add_result.get('prescription_number', 1)} added successfully",
            'patient': {
                'uid': patient_uid,
                'name': patient.get('name') or f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip(),
                'age': patient.get('age'),
                'gender': patient.get('gender'),
                'phone': patient.get('phone'),
                'allergies': patient.get('allergies', []),
                'prescriptions_count': summary.get('total_prescriptions', 0) if summary else 1,
                'active_medications': summary.get('active_medications', []) if summary else []
            },
            'prescription': {
                'prescription_id': add_result.get('prescription_uid'),
                'prescription_number': add_result.get('prescription_number', 1),
                'medications': prescription_data.get('medications', []),
                'diagnosis': prescription_data.get('diagnosis', []),
                'doctor_name': prescription_data.get('doctor_name'),
                'confidence': prescription_data.get('confidence')
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to add prescription: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add prescription: {str(e)}")


@router.post("/add-prescriptions")
async def add_multiple_prescriptions(
    patient_uid: str = Form(..., description="Patient UID"),
    files: List[UploadFile] = File(..., description="Multiple prescription images or PDFs")
):
    """
    Add multiple prescriptions to an existing patient at once.
    
    This endpoint:
    1. Verifies the patient exists
    2. Processes each prescription using OCR + AI
    3. Saves all prescriptions with comprehensive data for AI context
    4. Returns summary of all processed prescriptions
    
    Each prescription is saved with full context including:
    - Patient demographics and medical history
    - Complete OCR text for reference
    - Extracted structured data
    - Drug interactions and safety alerts
    - Doctor and clinic information
    - Timestamps and confidence scores
    """
    # Validate all files first
    allowed_types = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
    
    for f in files:
        file_ext = os.path.splitext(f.filename)[1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File {f.filename} has unsupported type. Allowed: {', '.join(allowed_types)}"
            )
    
    try:
        service = get_unified_patient_service()
        
        # Verify patient exists
        patient = service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient with UID '{patient_uid}' not found"
            )
        
        # Get patient allergies for safety checking
        patient_allergies = patient.get('allergies', [])
        patient_conditions = patient.get('conditions', [])
        
        # Process each file
        results = []
        errors = []
        total_medications = 0
        
        for idx, file in enumerate(files):
            file_ext = os.path.splitext(file.filename)[1].lower()
            file_id = str(uuid.uuid4())
            file_path = settings.UPLOAD_DIR / f"{file_id}{file_ext}"
            
            try:
                # Save file
                os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process with OCR + AI
                result = complete_processor.process(
                    file_path=str(file_path),
                    patient_allergies=patient_allergies,
                    patient_id=patient_uid,
                    save_to_db=False
                )
                
                # Build comprehensive prescription data
                prescription_data = _build_comprehensive_prescription_data(
                    result, file.filename, patient_allergies, patient
                )
                
                # Add to database
                add_result = service.add_prescription(patient_uid, prescription_data)
                
                med_count = len(prescription_data.get('medications', []))
                total_medications += med_count
                
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'prescription_uid': add_result.get('prescription_uid'),
                    'prescription_number': add_result.get('prescription_number'),
                    'medications_count': med_count,
                    'medications': prescription_data.get('medications', []),
                    'diagnosis': prescription_data.get('diagnosis', []),
                    'doctor_name': prescription_data.get('doctor_name'),
                    'prescription_date': prescription_data.get('prescription_date'),
                    'confidence': prescription_data.get('confidence', 0),
                    'drug_interactions': prescription_data.get('drug_interactions', []),
                    'allergy_alerts': prescription_data.get('allergy_alerts', [])
                })
                
                logger.info(f"Processed prescription {idx + 1}/{len(files)}: {file.filename}")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                errors.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        # Get updated patient summary
        summary = service.get_patient_summary(patient_uid)
        
        return JSONResponse(content={
            'success': True,
            'message': f"Processed {len(results)} of {len(files)} prescriptions successfully",
            'total_processed': len(results),
            'total_failed': len(errors),
            'total_medications_added': total_medications,
            'patient': {
                'uid': patient_uid,
                'name': patient.get('name') or f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip(),
                'prescriptions_count': summary.get('total_prescriptions', 0) if summary else len(results),
                'active_medications': summary.get('active_medications', []) if summary else []
            },
            'prescriptions': results,
            'errors': errors
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to add prescriptions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add prescriptions: {str(e)}")


def _build_comprehensive_prescription_data(
    result, filename: str, patient_allergies: List[str], patient: Dict
) -> Dict[str, Any]:
    """
    Build comprehensive prescription data with full context for AI.
    This ensures all relevant information is saved for future AI queries.
    """
    result_dict = result.to_dict()
    
    return {
        # === File & Processing Info ===
        'filename': filename,
        'processed_at': datetime.utcnow().isoformat(),
        'document_id': result_dict.get('document_id'),
        
        # === Patient Context (for AI reference) ===
        'patient_context': {
            'patient_uid': patient.get('uid') or patient.get('patient_uid'),
            'name': patient.get('name'),
            'age': patient.get('age'),
            'gender': patient.get('gender'),
            'known_allergies': patient_allergies,
            'chronic_conditions': patient.get('conditions', []),
            'blood_group': patient.get('blood_group')
        },
        
        # === Extracted Patient Info from Prescription ===
        'patient_name': result_dict.get('patient_name'),
        'patient_age': result_dict.get('patient_age'),
        'patient_gender': result_dict.get('patient_gender'),
        'patient_address': result_dict.get('patient_address'),
        'patient_phone': result_dict.get('patient_phone'),
        
        # === Doctor & Clinic Info ===
        'doctor_name': result_dict.get('doctor_name'),
        'doctor_qualification': result_dict.get('doctor_qualification'),
        'doctor_reg_no': result_dict.get('doctor_reg_no'),
        'clinic_name': result_dict.get('clinic_name'),
        
        # === Prescription Details ===
        'prescription_date': result_dict.get('prescription_date'),
        
        # === Clinical Data ===
        'diagnosis': result_dict.get('diagnosis', []),
        'chief_complaints': result_dict.get('chief_complaints', []),
        'vitals': result_dict.get('vitals', {}),
        
        # === Medications with Full Details ===
        'medications': result_dict.get('medications', []),
        
        # === Additional Clinical Info ===
        'advice': result_dict.get('advice', []),
        'follow_up': result_dict.get('follow_up'),
        'investigations': result_dict.get('investigations', []),
        
        # === Safety Analysis ===
        'drug_interactions': result_dict.get('drug_interactions', []),
        'allergy_alerts': result_dict.get('allergy_alerts', []),
        'safety_alerts': result_dict.get('safety_alerts', []),
        
        # === Raw Data for AI Reference ===
        'raw_ocr_text': result_dict.get('raw_ocr_text', ''),
        
        # === Quality & Confidence ===
        'confidence': result_dict.get('confidence', 0),
        'ocr_confidence': result_dict.get('ocr_confidence', 0),
        'extraction_method': result_dict.get('extraction_method', 'unknown'),
        'processing_time_ms': result_dict.get('processing_time_ms', 0),
        
        # === Review Flags ===
        'needs_review': result_dict.get('needs_review', False),
        'review_reasons': result_dict.get('review_reasons', []),
        
        # === Errors & Warnings ===
        'errors': result_dict.get('errors', []),
        'warnings': result_dict.get('warnings', []),
        
        # === Known Allergies (passed in) ===
        'allergies': patient_allergies
    }


@router.get("/patients")
async def list_all_patients(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None
):
    """
    List all patients with basic info.
    Supports pagination and search by name or UID.
    """
    try:
        service = get_unified_patient_service()
        patients = service.get_all_patients()
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            patients = [
                p for p in patients
                if search_lower in (p.get('name', '') or '').lower() 
                or search_lower in (p.get('patient_id', '') or '').lower()
            ]
        
        # Apply pagination
        total = len(patients)
        patients = patients[offset:offset + limit]
        
        return JSONResponse(content={
            'success': True,
            'total': total,
            'limit': limit,
            'offset': offset,
            'patients': patients
        })
        
    except Exception as e:
        logger.error(f"Failed to list patients: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list patients: {str(e)}")


@router.get("/patient/{patient_uid}/prescriptions")
async def get_patient_prescriptions(patient_uid: str, limit: int = 20):
    """
    Get all prescriptions for a patient.
    """
    try:
        service = get_unified_patient_service()
        
        # Verify patient exists
        patient = service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient with UID '{patient_uid}' not found"
            )
        
        # Get prescriptions (no limit param in service, limit here)
        prescriptions = service.get_patient_prescriptions(patient_uid)
        prescriptions = prescriptions[:limit] if limit else prescriptions
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'total': len(prescriptions),
            'prescriptions': prescriptions
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prescriptions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prescriptions: {str(e)}")


@router.get("/patient/{patient_uid}/timeline")
async def get_patient_timeline(patient_uid: str, limit: int = 20):
    """
    Get patient's medical timeline (all events).
    """
    try:
        service = get_unified_patient_service()
        
        # Verify patient exists
        patient = service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient with UID '{patient_uid}' not found"
            )
        
        # Get timeline
        timeline = service.get_patient_timeline(patient_uid, limit=limit)
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'timeline': timeline
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get timeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")


@router.get("/patient/{patient_uid}/full-details")
async def get_patient_full_details(patient_uid: str):
    """
    Get complete patient details including all prescriptions with full data.
    Used by doctors to view complete patient medical history after QR scan.
    
    Returns:
    - Patient demographics
    - All prescriptions with medications, diagnosis, doctor info
    - Active medications
    - Allergies and conditions
    - Timeline events
    - Drug interactions and safety alerts
    """
    try:
        service = get_unified_patient_service()
        
        # Get patient
        patient = service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient with UID '{patient_uid}' not found"
            )
        
        # Get full summary
        summary = service.get_patient_summary(patient_uid)
        
        # Get all prescriptions
        prescriptions = service.get_patient_prescriptions(patient_uid)
        
        # Get timeline
        timeline = service.get_patient_timeline(patient_uid, limit=50)
        
        # Get active medications
        active_meds = summary.get('active_medications', []) if summary else []
        
        return JSONResponse(content={
            'success': True,
            'patient': {
                'uid': patient_uid,
                'name': patient.get('name') or f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip(),
                'age': patient.get('age'),
                'gender': patient.get('gender'),
                'phone': patient.get('phone'),
                'email': patient.get('email'),
                'address': patient.get('address'),
                'blood_group': patient.get('blood_group'),
                'allergies': patient.get('allergies', []),
                'conditions': patient.get('conditions', []),
                'emergency_contact_name': patient.get('emergency_contact_name'),
                'emergency_contact_phone': patient.get('emergency_contact_phone')
            },
            'summary': {
                'total_prescriptions': len(prescriptions),
                'active_medications_count': len(active_meds),
                'last_visit': prescriptions[0].get('prescription_date') if prescriptions else None
            },
            'active_medications': active_meds,
            'prescriptions': prescriptions,
            'timeline': timeline
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get patient full details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get patient details: {str(e)}")


@router.post("/doctor/scan-qr")
async def doctor_scan_qr(
    file: UploadFile = File(..., description="QR code image file")
):
    """
    Doctor scans patient QR code to view full prescription history.
    Combines QR decoding with full patient details retrieval.
    """
    # Validate file type
    allowed_types = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Decode QR code
        decoded_data = decode_qr_from_image(image_bytes)
        
        if not decoded_data:
            raise HTTPException(
                status_code=400,
                detail="No QR code found in the image. Please ensure the image contains a clear QR code."
            )
        
        patient_uid = decoded_data.strip()
        
        # Get full patient details
        service = get_unified_patient_service()
        patient = service.get_patient_by_uid(patient_uid)
        
        if not patient:
            return JSONResponse(content={
                'success': False,
                'decoded_uid': patient_uid,
                'patient_found': False,
                'message': f"QR decoded (UID: {patient_uid}) but patient not found in database"
            })
        
        # Get all patient data
        summary = service.get_patient_summary(patient_uid)
        prescriptions = service.get_patient_prescriptions(patient_uid)
        timeline = service.get_patient_timeline(patient_uid, limit=50)
        active_meds = summary.get('active_medications', []) if summary else []
        
        return JSONResponse(content={
            'success': True,
            'decoded_uid': patient_uid,
            'patient_found': True,
            'patient': {
                'uid': patient_uid,
                'name': patient.get('name') or f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip(),
                'age': patient.get('age'),
                'gender': patient.get('gender'),
                'phone': patient.get('phone'),
                'email': patient.get('email'),
                'address': patient.get('address'),
                'blood_group': patient.get('blood_group'),
                'allergies': patient.get('allergies', []),
                'conditions': patient.get('conditions', [])
            },
            'summary': {
                'total_prescriptions': len(prescriptions),
                'active_medications_count': len(active_meds),
                'last_visit': prescriptions[0].get('prescription_date') if prescriptions else None
            },
            'active_medications': active_meds,
            'prescriptions': prescriptions,
            'timeline': timeline
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in doctor QR scan: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing QR code: {str(e)}")


@router.get("/patient/{patient_uid}/ai-context")
async def get_patient_ai_context(patient_uid: str):
    """
    Get patient data formatted for AI assistant context.
    Returns all relevant information in a structured format optimized for AI queries.
    """
    try:
        service = get_unified_patient_service()
        
        patient = service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient with UID '{patient_uid}' not found"
            )
        
        summary = service.get_patient_summary(patient_uid)
        prescriptions = service.get_patient_prescriptions(patient_uid)
        active_meds = summary.get('active_medications', []) if summary else []
        
        # Build AI-friendly context
        ai_context = {
            'patient_info': {
                'uid': patient_uid,
                'name': patient.get('name'),
                'age': patient.get('age'),
                'gender': patient.get('gender'),
                'blood_group': patient.get('blood_group')
            },
            'medical_profile': {
                'allergies': patient.get('allergies', []),
                'chronic_conditions': patient.get('conditions', [])
            },
            'current_medications': [
                {
                    'name': med.get('name'),
                    'dosage': med.get('dosage'),
                    'frequency': med.get('frequency'),
                    'prescriber': med.get('prescriber')
                }
                for med in active_meds
            ],
            'prescription_history': [
                {
                    'date': presc.get('prescription_date'),
                    'doctor': presc.get('doctor_name'),
                    'diagnosis': presc.get('diagnosis'),
                    'medications': [
                        f"{med.get('name')} {med.get('dosage')} {med.get('frequency')}"
                        for med in presc.get('medications', [])
                    ]
                }
                for presc in prescriptions[:10]  # Last 10 prescriptions
            ],
            'summary_text': _build_patient_summary_text(patient, active_meds, prescriptions)
        }
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'ai_context': ai_context
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get AI context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI context: {str(e)}")


def _build_patient_summary_text(patient: Dict, active_meds: List, prescriptions: List) -> str:
    """Build a natural language summary of patient for AI context"""
    name = patient.get('name', 'Unknown')
    age = patient.get('age', 'Unknown')
    gender = patient.get('gender', 'Unknown')
    
    summary = f"Patient {name}, {age} years old, {gender}. "
    
    allergies = patient.get('allergies', [])
    if allergies:
        summary += f"Known allergies: {', '.join(allergies)}. "
    else:
        summary += "No known allergies. "
    
    conditions = patient.get('conditions', [])
    if conditions:
        summary += f"Chronic conditions: {', '.join(conditions)}. "
    
    if active_meds:
        med_list = [f"{m.get('name')} {m.get('dosage', '')}" for m in active_meds[:5]]
        summary += f"Currently taking: {', '.join(med_list)}. "
    
    summary += f"Total prescriptions on record: {len(prescriptions)}."
    
    return summary


def _update_patient_extra_info(
    patient_uid: str,
    email: Optional[str] = None,
    blood_group: Optional[str] = None,
    emergency_contact_name: Optional[str] = None,
    emergency_contact_phone: Optional[str] = None
):
    """Update patient with extra information not handled by unified service"""
    try:
        from backend.database.connection import db_manager
        from backend.database.models import Patient
        
        session = db_manager.get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if patient:
                if email:
                    patient.email = email
                if blood_group:
                    patient.blood_group = blood_group
                if emergency_contact_name:
                    patient.emergency_contact_name = emergency_contact_name
                if emergency_contact_phone:
                    patient.emergency_contact_phone = emergency_contact_phone
                
                session.commit()
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Failed to update extra patient info: {e}")
