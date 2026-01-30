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

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from backend.config import settings
from backend.services.complete_processor import complete_processor
from backend.services.unified_patient_service import get_unified_patient_service
from backend.services.clinical_decision_support_service import clinical_decision_support
from backend.services.treatment_outcome_service import treatment_outcome_service, OutcomeType, VitalType
from backend.services.neo4j_visualization_service import get_neo4j_visualization_service

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
    file: Optional[UploadFile] = File(None, description="Prescription image or PDF (optional)")
):
    """
    Create a new patient, optionally with their first prescription.
    
    This endpoint:
    1. Creates a new patient record with a unique UID
    2. Optionally processes the prescription using OCR + AI
    3. Saves both patient and prescription to the database
    4. Returns the patient UID for QR code generation
    
    The frontend will generate the QR code client-side using the returned UID.
    """
    # Generate unique patient UID
    patient_uid = generate_patient_uid()
    
    # Parse allergies and conditions
    patient_allergies = []
    if allergies:
        patient_allergies = [a.strip() for a in allergies.split(',') if a.strip()]
    
    patient_conditions = []
    if conditions:
        patient_conditions = [c.strip() for c in conditions.split(',') if c.strip()]
    
    # Parse age
    patient_age = None
    if age:
        try:
            patient_age = int(age)
        except ValueError:
            pass
    
    # Full patient name
    full_name = f"{first_name} {last_name}"
    
    prescription_data = None
    prescription_result = None
    
    # Process prescription if file is provided
    if file and file.filename:
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
            logger.error(f"Failed to save file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
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
        except Exception as e:
            logger.error(f"Failed to process prescription: {e}")
            # Continue without prescription data - patient will still be created
    
    try:
        # Get unified patient service
        service = get_unified_patient_service()
        
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
        
        prescriptions_count = 0
        prescription_response = None
        
        # Add prescription to database if we have prescription data
        if prescription_data:
            add_result = service.add_prescription(patient_uid, prescription_data)
            prescriptions_count = 1
            prescription_response = {
                'prescription_id': add_result.get('prescription_id'),
                'medications': prescription_data.get('medications', []),
                'diagnosis': prescription_data.get('diagnosis', []),
                'doctor_name': prescription_data.get('doctor_name'),
                'confidence': prescription_data.get('confidence')
            }
        
        # Build response message
        if prescription_data:
            message = 'Patient created and prescription processed successfully'
        else:
            message = 'Patient created successfully (no prescription uploaded)'
        
        return JSONResponse(content={
            'success': True,
            'message': message,
            'patient': {
                'uid': patient_uid,
                'name': full_name,
                'age': patient_age,
                'gender': gender,
                'phone': phone,
                'allergies': patient_allergies,
                'conditions': patient_conditions,
                'prescriptions_count': prescriptions_count
            },
            'prescription': prescription_response,
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


# ========================================
# CLINICAL DECISION SUPPORT ENDPOINTS
# ========================================

@router.post("/patient/{patient_uid}/clinical-decision-support")
async def get_clinical_decision_support(
    patient_uid: str
):
    """
    Get AI-powered clinical decision support for a patient.
    
    Provides:
    - Evidence-based treatment alternatives
    - Guideline compliance scoring
    - Pharmacogenomic alerts
    - Treatment optimization suggestions
    
    This goes beyond obvious insights to provide actionable clinical intelligence.
    """
    try:
        # Get unified patient service
        patient_service = get_unified_patient_service()
        
        # Get patient data
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Get current medications
        medications = []
        active_meds = patient_service.get_patient_medications(patient_uid, active_only=True)
        for med in active_meds:
            medications.append(med.get('name', '') or med.get('drug_name', ''))
        
        # Get conditions
        conditions = patient.get('conditions', [])
        if isinstance(conditions, str):
            conditions = [conditions]
        
        # Build patient profile
        patient_profile = {
            'age': patient.get('age') or 50,
            'gender': patient.get('gender') or 'unknown',
            'bmi': patient.get('bmi') or 25,
            'conditions': conditions,
            'allergies': patient.get('allergies') or []
        }
        
        # Generate clinical decision support report
        report = clinical_decision_support.generate_full_report(
            patient_id=patient_uid,
            medications=medications,
            conditions=conditions,
            patient_profile=patient_profile,
            genetic_data=patient.get('genetic_data', {})  # If available
        )
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'clinical_decision_support': report.to_dict(),
            'summary': _generate_cds_summary(report)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get clinical decision support: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_uid}/guideline-compliance")
async def get_guideline_compliance(
    patient_uid: str
):
    """
    Get guideline compliance assessment for a patient's current treatment.
    
    Scores prescription alignment with medical guidelines (AHA, ADA, ACC, etc.)
    and identifies gaps in care.
    """
    try:
        patient_service = get_unified_patient_service()
        
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Get medications
        medications = []
        active_meds = patient_service.get_patient_medications(patient_uid, active_only=True)
        for med in active_meds:
            medications.append(med.get('name', '') or med.get('drug_name', ''))
        
        conditions = patient.get('conditions', [])
        if isinstance(conditions, str):
            conditions = [conditions]
        
        patient_profile = {
            'age': patient.get('age', 50),
            'gender': patient.get('gender', 'unknown')
        }
        
        # Get compliance assessments
        assessments = clinical_decision_support.assess_guideline_compliance(
            medications=medications,
            conditions=conditions,
            patient_profile=patient_profile
        )
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'conditions_assessed': conditions,
            'medications_assessed': medications,
            'guideline_compliance': [a.to_dict() for a in assessments],
            'overall_score': sum(a.overall_score for a in assessments) / len(assessments) if assessments else 0
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get guideline compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_uid}/treatment-alternatives")
async def get_treatment_alternatives(
    patient_uid: str
):
    """
    Get evidence-based treatment alternatives for current medications.
    
    Provides alternatives based on:
    - Better outcomes for patient's specific profile
    - Newer guideline recommendations
    - Cost/benefit considerations
    """
    try:
        patient_service = get_unified_patient_service()
        
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Get medications
        medications = []
        active_meds = patient_service.get_patient_medications(patient_uid, active_only=True)
        for med in active_meds:
            medications.append(med.get('name', '') or med.get('drug_name', ''))
        
        conditions = patient.get('conditions', [])
        patient_profile = {
            'age': patient.get('age', 50),
            'conditions': conditions
        }
        
        # Get alternatives
        alternatives = clinical_decision_support.get_treatment_alternatives(
            medications=medications,
            conditions=conditions,
            patient_profile=patient_profile
        )
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'current_medications': medications,
            'alternatives': [a.to_dict() for a in alternatives],
            'total_alternatives_found': len(alternatives)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get treatment alternatives: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_uid}/pharmacogenomic-alerts")
async def get_pharmacogenomic_alerts(
    patient_uid: str
):
    """
    Get pharmacogenomic alerts for patient's medications.
    
    Identifies drugs that may require:
    - Genetic testing consideration
    - Dosage adjustments based on metabolism
    - Alternative drug selection
    """
    try:
        patient_service = get_unified_patient_service()
        
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Get medications
        medications = []
        active_meds = patient_service.get_patient_medications(patient_uid, active_only=True)
        for med in active_meds:
            medications.append(med.get('name', '') or med.get('drug_name', ''))
        
        # Get genetic data if available
        genetic_data = patient.get('genetic_data', {})
        
        # Get alerts
        alerts = clinical_decision_support.get_pharmacogenomic_alerts(
            medications=medications,
            genetic_data=genetic_data
        )
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'medications_assessed': medications,
            'genetic_data_available': bool(genetic_data),
            'pharmacogenomic_alerts': [a.to_dict() for a in alerts],
            'total_alerts': len(alerts)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pharmacogenomic alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# TREATMENT OUTCOME TRACKING ENDPOINTS
# ========================================

@router.post("/patient/{patient_uid}/record-outcome")
async def record_treatment_outcome(
    patient_uid: str,
    medication: str = Form(...),
    outcome_type: str = Form(...),
    description: str = Form(...),
    prescription_id: Optional[str] = Form(None),
    side_effects: Optional[str] = Form(None)  # Comma-separated
):
    """
    Record a treatment outcome for a patient.
    
    Links prescription to clinical outcome (improved, resolved, worsened, etc.)
    
    outcome_type: improved, stable, worsened, resolved, discontinued, adverse_event
    """
    try:
        # Validate outcome type
        try:
            outcome_enum = OutcomeType(outcome_type.lower())
        except ValueError:
            valid_types = [t.value for t in OutcomeType]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid outcome type. Valid: {valid_types}"
            )
        
        # Parse side effects
        side_effects_list = []
        if side_effects:
            side_effects_list = [s.strip() for s in side_effects.split(',') if s.strip()]
        
        # Record outcome
        outcome = treatment_outcome_service.record_outcome(
            patient_id=patient_uid,
            prescription_id=prescription_id or f"RX-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            medication=medication,
            outcome_type=outcome_enum,
            description=description,
            side_effects=side_effects_list
        )
        
        return JSONResponse(content={
            'success': True,
            'message': 'Treatment outcome recorded successfully',
            'outcome': outcome.to_dict()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patient/{patient_uid}/record-vital")
async def record_vital_reading(
    patient_uid: str,
    vital_type: str = Form(...),
    value: float = Form(...),
    unit: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """
    Record a vital sign reading for a patient.
    
    vital_type options: bp_systolic, bp_diastolic, heart_rate, blood_glucose, 
                        hba1c, ldl_cholesterol, hdl_cholesterol, weight, 
                        pain_score, egfr, creatinine, etc.
    """
    try:
        # Validate vital type
        try:
            vital_enum = VitalType(vital_type.lower())
        except ValueError:
            valid_types = [t.value for t in VitalType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vital type. Valid: {valid_types}"
            )
        
        # Default units
        default_units = {
            'bp_systolic': 'mmHg',
            'bp_diastolic': 'mmHg',
            'heart_rate': 'bpm',
            'blood_glucose': 'mg/dL',
            'hba1c': '%',
            'ldl_cholesterol': 'mg/dL',
            'hdl_cholesterol': 'mg/dL',
            'weight': 'kg',
            'pain_score': '0-10',
            'egfr': 'mL/min/1.73m²'
        }
        
        final_unit = unit or default_units.get(vital_type.lower(), 'units')
        
        # Record vital
        reading = treatment_outcome_service.record_vital_reading(
            patient_id=patient_uid,
            vital_type=vital_enum,
            value=value,
            unit=final_unit,
            notes=notes
        )
        
        return JSONResponse(content={
            'success': True,
            'message': 'Vital reading recorded successfully',
            'reading': reading.to_dict()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record vital: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_uid}/outcome-timeline")
async def get_outcome_timeline(
    patient_uid: str,
    months: int = 12
):
    """
    Get treatment outcome timeline for a patient.
    
    Shows:
    - All recorded treatment outcomes
    - Vital sign trends over time
    - Overall health trend analysis
    """
    try:
        timeline = treatment_outcome_service.get_patient_outcome_timeline(
            patient_id=patient_uid,
            months=months
        )
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'timeline': timeline.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Failed to get outcome timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_uid}/treatment-prediction")
async def get_treatment_prediction(
    patient_uid: str,
    medication: str,
    condition: Optional[str] = None
):
    """
    Get ML-based treatment success prediction.
    
    Predicts likelihood of treatment success based on:
    - Patient profile and history
    - Historical outcome data from similar patients
    - Known success/failure factors
    """
    try:
        patient_service = get_unified_patient_service()
        
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Build patient profile
        conditions = patient.get('conditions', [])
        patient_profile = {
            'age': patient.get('age', 50),
            'gender': patient.get('gender', 'unknown'),
            'bmi': patient.get('bmi', 25),
            'conditions': conditions,
            'smoker': patient.get('smoker', False),
            'adherence_rate': patient.get('adherence_rate', 0.7)
        }
        
        # Use provided condition or first from patient's conditions
        target_condition = condition or (conditions[0] if conditions else 'general')
        
        # Get prediction
        prediction = treatment_outcome_service.predict_treatment_success(
            medication=medication,
            condition=target_condition,
            patient_profile=patient_profile
        )
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'prediction': prediction.to_dict()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get treatment prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_uid}/comprehensive-outcome-report")
async def get_comprehensive_outcome_report(
    patient_uid: str
):
    """
    Get comprehensive outcome analysis and predictions for a patient.
    
    Combines:
    - Outcome timeline
    - Treatment predictions for all current medications
    - Vital sign trend analysis
    - Actionable insights
    """
    try:
        patient_service = get_unified_patient_service()
        
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Get medications
        medications = []
        active_meds = patient_service.get_patient_medications(patient_uid, active_only=True)
        for med in active_meds:
            medications.append(med.get('name', '') or med.get('drug_name', ''))
        
        conditions = patient.get('conditions', [])
        
        patient_profile = {
            'age': patient.get('age', 50),
            'gender': patient.get('gender', 'unknown'),
            'bmi': patient.get('bmi', 25),
            'conditions': conditions
        }
        
        # Generate comprehensive report
        report = treatment_outcome_service.generate_comprehensive_outcome_report(
            patient_id=patient_uid,
            medications=medications,
            conditions=conditions,
            patient_profile=patient_profile
        )
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'report': report
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patient/{patient_uid}/analyze-treatment-vitals")
async def analyze_treatment_vital_changes(
    patient_uid: str,
    medication: str = Form(...),
    start_date: str = Form(...),
    end_date: Optional[str] = Form(None)
):
    """
    Analyze vital sign changes during a treatment period.
    
    Generates analysis like:
    "Started Amlodipine 3 months ago → BP improved from 160/100 to 130/85"
    """
    try:
        analysis = treatment_outcome_service.analyze_vital_changes_for_treatment(
            patient_id=patient_uid,
            medication=medication,
            start_date=start_date,
            end_date=end_date
        )
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Failed to analyze vital changes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_cds_summary(report) -> Dict:
    """Generate a natural language summary of clinical decision support report"""
    summary = {
        'key_findings': [],
        'action_items': [],
        'guideline_score': None
    }
    
    # Add alternatives count
    if report.alternatives:
        summary['key_findings'].append(
            f"Found {len(report.alternatives)} evidence-based treatment alternatives to consider"
        )
        for alt in report.alternatives[:2]:
            summary['action_items'].append(
                f"Consider {alt.alternative_drug}: {alt.reason}"
            )
    
    # Add guideline compliance
    if report.guideline_compliance:
        score = report.guideline_compliance.overall_score
        summary['guideline_score'] = score
        
        if score >= 80:
            summary['key_findings'].append(
                f"Prescription {score:.0f}% aligned with {report.guideline_compliance.guideline_source}"
            )
        else:
            summary['key_findings'].append(
                f"Guideline alignment: {score:.0f}% - opportunities for optimization"
            )
            for gap in report.guideline_compliance.gaps[:2]:
                summary['action_items'].append(gap.get('item', ''))
    
    # Add pharmacogenomic alerts
    if report.pharmacogenomic_alerts:
        summary['key_findings'].append(
            f"{len(report.pharmacogenomic_alerts)} pharmacogenomic considerations identified"
        )
        for alert in report.pharmacogenomic_alerts[:2]:
            summary['action_items'].append(
                f"{alert.drug}: {alert.recommendation}"
            )
    
    # Add optimization suggestions
    if report.optimization_suggestions:
        high_priority = [s for s in report.optimization_suggestions if s.get('priority') == 'high']
        if high_priority:
            summary['key_findings'].append(
                f"{len(high_priority)} high-priority treatment optimizations recommended"
            )
            for opt in high_priority[:2]:
                summary['action_items'].append(opt.get('recommendation', opt.get('title', '')))
    
    return summary


# ==================== Knowledge Graph Endpoints ====================

@router.get("/patient/{patient_uid}/knowledge-graph")
async def get_patient_knowledge_graph(patient_uid: str):
    """
    Get interactive knowledge graph for a patient
    
    Returns vis.js compatible graph data with:
    - Patient node (center)
    - Medication nodes
    - Condition nodes
    - Allergy nodes
    - Drug interaction nodes
    - Relationships between all entities
    """
    try:
        patient_service = get_unified_patient_service()
        
        # Get patient data
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Get medications
        medications = patient_service.get_patient_medications(patient_uid, active_only=True)
        
        # Get conditions
        conditions = patient.get('conditions', [])
        if isinstance(conditions, str):
            conditions = [c.strip() for c in conditions.split(',') if c.strip()]
        
        # Get allergies
        allergies = patient.get('allergies', [])
        if isinstance(allergies, str):
            allergies = [a.strip() for a in allergies.split(',') if a.strip()]
        
        # Get prescriptions with dates
        prescriptions = patient_service.get_patient_prescriptions(patient_uid)
        
        # Get drug interactions if any
        interactions = []
        try:
            from backend.services.drug_interaction_service import get_drug_interaction_service
            interaction_service = get_drug_interaction_service()
            drug_names = [m.get('name') or m.get('drug_name', '') for m in medications if m.get('name') or m.get('drug_name')]
            if len(drug_names) >= 2:
                interaction_result = interaction_service.check_interactions(drug_names)
                interactions = interaction_result.get('interactions', [])
        except Exception as e:
            logger.warning(f"Could not check interactions: {e}")
        
        # Build knowledge graph
        viz_service = get_neo4j_visualization_service()
        graph_data = viz_service.build_patient_graph(
            patient_uid=patient_uid,
            patient_data={
                'name': patient.get('name', 'Unknown'),
                'age': patient.get('age') or 'N/A',
                'gender': patient.get('gender') or 'N/A',
                'blood_group': patient.get('blood_group') or 'N/A',
            },
            medications=medications,
            conditions=conditions,
            allergies=allergies,
            interactions=interactions,
            prescriptions=prescriptions,
            include_drug_relationships=True
        )
        
        # Also generate Cypher export for actual Neo4j import
        cypher_export = viz_service.export_cypher(patient_uid)
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'patient_name': patient.get('name', 'Unknown'),
            'graph': graph_data,
            'cypher_export': cypher_export,
            'neo4j_ready': True
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/demo")
async def get_demo_knowledge_graph():
    """
    Get a demo knowledge graph to show visualization capabilities
    """
    viz_service = get_neo4j_visualization_service()
    
    # Create demo data
    demo_medications = [
        {'name': 'Metformin', 'dosage': '500mg', 'frequency': 'twice daily'},
        {'name': 'Lisinopril', 'dosage': '10mg', 'frequency': 'once daily'},
        {'name': 'Atorvastatin', 'dosage': '20mg', 'frequency': 'at bedtime'},
        {'name': 'Aspirin', 'dosage': '81mg', 'frequency': 'once daily'},
    ]
    
    demo_conditions = ['Type 2 Diabetes', 'Hypertension', 'Hyperlipidemia']
    demo_allergies = ['Penicillin', 'Sulfa drugs']
    demo_interactions = [
        {
            'drug1': 'Aspirin',
            'drug2': 'Lisinopril',
            'severity': 'moderate',
            'description': 'May reduce antihypertensive effect'
        }
    ]
    
    graph_data = viz_service.build_patient_graph(
        patient_uid='DEMO-001',
        patient_data={
            'name': 'Demo Patient',
            'age': 65,
            'gender': 'Male',
            'blood_group': 'A+'
        },
        medications=demo_medications,
        conditions=demo_conditions,
        allergies=demo_allergies,
        interactions=demo_interactions
    )
    
    return JSONResponse(content={
        'success': True,
        'patient_uid': 'DEMO-001',
        'patient_name': 'Demo Patient',
        'graph': graph_data,
        'is_demo': True
    })


# ==================== Timeline Endpoints ====================

@router.get("/patient/{patient_uid}/timeline")
async def get_patient_timeline_by_uid(patient_uid: str):
    """
    Get patient medical timeline events by UID
    Returns prescriptions, medications, and events with dates
    """
    try:
        patient_service = get_unified_patient_service()
        
        # Get patient
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Get timeline events
        timeline = patient_service.get_patient_timeline(patient_uid)
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'patient_name': patient.get('name', 'Unknown'),
            'events': timeline
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_uid}/gantt")
async def get_patient_gantt_by_uid(patient_uid: str):
    """
    Get Gantt chart data for medication timeline visualization
    Returns medications with start/end dates for Gantt rendering
    """
    try:
        patient_service = get_unified_patient_service()
        
        # Get patient
        patient = patient_service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        # Get all medications (including inactive)
        medications = patient_service.get_patient_medications(patient_uid, active_only=False)
        
        # Format for Gantt chart
        gantt_data = []
        for med in medications:
            start_date = med.get('start_date', '')
            end_date = med.get('end_date', '')
            
            # Format dates
            start_display = 'Unknown'
            end_display = 'Ongoing'
            try:
                if start_date:
                    from dateutil import parser
                    if isinstance(start_date, str):
                        start_dt = parser.parse(start_date)
                    else:
                        start_dt = start_date
                    start_display = start_dt.strftime('%d %b %Y')
            except:
                pass
                
            try:
                if end_date:
                    from dateutil import parser
                    if isinstance(end_date, str):
                        end_dt = parser.parse(end_date)
                    else:
                        end_dt = end_date
                    end_display = end_dt.strftime('%d %b %Y')
            except:
                pass
            
            gantt_data.append({
                'name': med.get('name', 'Unknown'),
                'dosage': med.get('dosage', ''),
                'frequency': med.get('frequency', ''),
                'start': start_display,
                'end': end_display,
                'start_date': start_date,
                'end_date': end_date,
                'active': med.get('is_active', True),
                'prescriber': med.get('prescriber', '')
            })
        
        # Sort by start date (most recent first)
        gantt_data.sort(key=lambda x: x.get('start_date') or '', reverse=True)
        
        return JSONResponse(content={
            'success': True,
            'patient_uid': patient_uid,
            'patient_name': patient.get('name', 'Unknown'),
            'medications': gantt_data,
            'total_medications': len(gantt_data),
            'active_medications': sum(1 for m in gantt_data if m.get('active'))
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get gantt data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Patient Medical Data Management ====================

class MedicalDataRequest(BaseModel):
    """Request for adding medical data"""
    name: str
    category: Optional[str] = None
    severity: Optional[str] = None
    notes: Optional[str] = None


@router.post("/patient/{patient_uid}/allergy")
async def add_patient_allergy(patient_uid: str, data: MedicalDataRequest):
    """Add an allergy to a patient's record"""
    try:
        service = get_unified_patient_service()
        result = service.add_allergy(patient_uid, data.name)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Create timeline event
        _create_timeline_event(patient_uid, 'allergy_added', f"Allergy added: {data.name}")
        
        return JSONResponse(content={'success': True, 'message': result.get('message')})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add allergy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/patient/{patient_uid}/allergy/{allergy_name}")
async def remove_patient_allergy(patient_uid: str, allergy_name: str):
    """Remove an allergy from a patient's record"""
    try:
        service = get_unified_patient_service()
        result = service.remove_allergy(patient_uid, allergy_name)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content={'success': True, 'message': result.get('message')})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove allergy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patient/{patient_uid}/condition")
async def add_patient_condition(patient_uid: str, data: MedicalDataRequest):
    """Add a chronic condition to a patient's record"""
    try:
        service = get_unified_patient_service()
        result = service.add_condition(patient_uid, data.name)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Create timeline event
        _create_timeline_event(patient_uid, 'condition_added', f"Condition diagnosed: {data.name}")
        
        return JSONResponse(content={'success': True, 'message': result.get('message')})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add condition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/patient/{patient_uid}/condition/{condition_name}")
async def remove_patient_condition(patient_uid: str, condition_name: str):
    """Remove a condition from a patient's record"""
    try:
        service = get_unified_patient_service()
        result = service.remove_condition(patient_uid, condition_name)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content={'success': True, 'message': result.get('message')})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove condition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patient/{patient_uid}/symptom")
async def add_patient_symptom(patient_uid: str, data: MedicalDataRequest):
    """Add a symptom to a patient's record"""
    try:
        service = get_unified_patient_service()
        result = service.add_symptom(patient_uid, data.name, data.severity)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content={'success': True, 'message': result.get('message')})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add symptom: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patient/{patient_uid}/medication")
async def add_patient_medication_manual(patient_uid: str, data: Dict[str, Any] = Body(...)):
    """Manually add a medication to patient's record (not from prescription)"""
    try:
        service = get_unified_patient_service()
        result = service.add_medication_manual(
            patient_uid=patient_uid,
            medication_name=data.get('name'),
            dosage=data.get('dosage'),
            frequency=data.get('frequency'),
            prescriber=data.get('prescriber'),
            reason=data.get('reason'),
            start_date=data.get('start_date')
        )
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content={'success': True, 'message': result.get('message'), 'data': result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add medication: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/patient/{patient_uid}/medication/{medication_id}/stop")
async def stop_patient_medication(patient_uid: str, medication_id: int, data: Dict[str, Any] = Body(...)):
    """Stop/discontinue a medication"""
    try:
        service = get_unified_patient_service()
        result = service.stop_medication(
            patient_uid=patient_uid,
            medication_id=medication_id,
            reason=data.get('reason', 'Discontinued by physician')
        )
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content={'success': True, 'message': result.get('message')})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop medication: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_uid}/medical-summary")
async def get_patient_medical_summary(patient_uid: str):
    """Get complete medical summary for a patient including all relationships"""
    try:
        service = get_unified_patient_service()
        
        patient = service.get_patient_by_uid(patient_uid)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
        
        medications = service.get_patient_medications(patient_uid, active_only=False)
        allergies = service.get_patient_allergies(patient_uid)
        conditions = service.get_patient_conditions(patient_uid)
        prescriptions = service.get_patient_prescriptions(patient_uid)
        timeline = service.get_patient_timeline(patient_uid, limit=20)
        
        return JSONResponse(content={
            'success': True,
            'patient': patient,
            'medications': {
                'active': [m for m in medications if m.get('is_active')],
                'inactive': [m for m in medications if not m.get('is_active')],
                'total': len(medications)
            },
            'allergies': allergies,
            'conditions': conditions,
            'prescriptions': prescriptions[:10],
            'timeline': timeline,
            'stats': {
                'total_prescriptions': len(prescriptions),
                'active_medications': sum(1 for m in medications if m.get('is_active')),
                'allergies_count': len(allergies),
                'conditions_count': len(conditions)
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get medical summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/patient/{patient_uid}/update")
async def update_patient_info(patient_uid: str, data: Dict[str, Any] = Body(...)):
    """Update patient basic information"""
    try:
        service = get_unified_patient_service()
        result = service.update_patient(patient_uid, data)
        
        if result.get('error'):
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content={'success': True, 'message': 'Patient updated successfully', 'patient': result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _create_timeline_event(patient_uid: str, event_type: str, description: str):
    """Helper to create timeline events"""
    try:
        from backend.database import SessionLocal
        from backend.database.models import Patient, TimelineEvent, AlertSeverity
        from datetime import datetime
        
        session = SessionLocal()
        patient = session.query(Patient).filter(Patient.patient_uid == patient_uid).first()
        if patient:
            event = TimelineEvent(
                patient_id=patient.id,
                event_type=event_type,
                event_date=datetime.utcnow(),
                description=description,
                severity=AlertSeverity.INFO
            )
            session.add(event)
            session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"Failed to create timeline event: {e}")


# ==================== Real-Time Drug Safety Check ====================

class DrugCheckRequest(BaseModel):
    drug_name: str
    patient_allergies: List[str] = []
    patient_conditions: List[str] = []
    current_medications: List[str] = []


@router.post("/check-drug-safety")
async def check_drug_safety(request: DrugCheckRequest):
    """
    Real-time drug safety check - call when doctor adds a drug to prescription.
    Checks for:
    - Allergy conflicts (direct and cross-reactivity)
    - Drug-drug interactions with current medications
    - Drug-condition contraindications
    - Duplicate therapy
    
    Returns immediate alerts for the doctor.
    """
    try:
        from backend.services.drug_interaction_service import DrugInteractionService, InteractionSeverity
        
        service = DrugInteractionService()
        
        # Combine new drug with current medications for analysis
        all_meds = request.current_medications + [request.drug_name]
        
        # Analyze safety
        result = service.analyze_safety(
            medications=all_meds,
            patient_allergies=request.patient_allergies,
            patient_conditions=request.patient_conditions,
            suppress_low_value=False  # Show all alerts for real-time check
        )
        
        alerts = []
        
        # Check allergy alerts - these are CRITICAL
        for alert in result.allergy_alerts:
            if alert.drug.lower() == request.drug_name.lower() or request.drug_name.lower() in alert.drug.lower():
                alerts.append({
                    "type": "ALLERGY",
                    "severity": "CRITICAL",
                    "icon": "🚨",
                    "title": f"ALLERGY ALERT: {alert.allergen}",
                    "message": alert.description,
                    "risk_type": alert.risk_type,
                    "alternatives": alert.alternatives[:3] if alert.alternatives else [],
                    "action": "DO NOT PRESCRIBE without patient consent and monitoring"
                })
        
        # Check drug interactions - only for the new drug
        for interaction in result.interactions:
            if request.drug_name.lower() in interaction.drug1.lower() or request.drug_name.lower() in interaction.drug2.lower():
                severity_map = {
                    InteractionSeverity.CONTRAINDICATED: ("CRITICAL", "🚨"),
                    InteractionSeverity.MAJOR: ("HIGH", "⚠️"),
                    InteractionSeverity.MODERATE: ("MODERATE", "⚡"),
                    InteractionSeverity.MINOR: ("LOW", "ℹ️")
                }
                sev_label, icon = severity_map.get(interaction.severity, ("MODERATE", "⚡"))
                
                other_drug = interaction.drug2 if request.drug_name.lower() in interaction.drug1.lower() else interaction.drug1
                
                alerts.append({
                    "type": "INTERACTION",
                    "severity": sev_label,
                    "icon": icon,
                    "title": f"Interaction with {other_drug}",
                    "message": interaction.description,
                    "mechanism": interaction.mechanism,
                    "clinical_effects": interaction.clinical_effects,
                    "action": interaction.management
                })
        
        # Check contraindications
        for contra in result.contraindications:
            if request.drug_name.lower() in contra.get('drug', '').lower():
                alerts.append({
                    "type": "CONTRAINDICATION",
                    "severity": "HIGH" if contra.get('level') == 'contraindicated' else "MODERATE",
                    "icon": "🚫" if contra.get('level') == 'contraindicated' else "⚠️",
                    "title": f"Contraindicated: {contra.get('condition', 'Unknown condition')}",
                    "message": contra.get('description', ''),
                    "action": contra.get('recommendation', 'Use with caution or avoid')
                })
        
        # Check duplicate therapy
        for dup in result.duplicate_therapies:
            if request.drug_name.lower() in ' '.join(dup.drugs).lower():
                alerts.append({
                    "type": "DUPLICATE",
                    "severity": "MODERATE",
                    "icon": "🔄",
                    "title": f"Duplicate: {dup.drug_class}",
                    "message": dup.description,
                    "action": dup.recommendation
                })
        
        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
        alerts.sort(key=lambda x: severity_order.get(x["severity"], 99))
        
        return {
            "success": True,
            "drug": request.drug_name,
            "is_safe": len([a for a in alerts if a["severity"] in ["CRITICAL", "HIGH"]]) == 0,
            "alerts_count": len(alerts),
            "critical_count": len([a for a in alerts if a["severity"] == "CRITICAL"]),
            "high_count": len([a for a in alerts if a["severity"] == "HIGH"]),
            "alerts": alerts,
            "overall_risk": result.overall_risk_level
        }
        
    except Exception as e:
        logger.error(f"Drug safety check error: {e}")
        return {
            "success": False,
            "drug": request.drug_name,
            "is_safe": True,  # Default to safe if check fails
            "alerts_count": 0,
            "alerts": [],
            "error": str(e)
        }


@router.get("/patient/{patient_uid}/safety-summary")
async def get_patient_safety_summary(patient_uid: str):
    """
    Get comprehensive patient safety summary for doctor's quick view.
    Includes:
    - Critical allergies (with severity)
    - Active conditions affecting prescribing
    - Current medications with drug classes
    - Recent safety alerts
    """
    try:
        service = get_unified_patient_service()
        patient = await service.get_patient_by_uid(patient_uid)
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Categorize allergies by severity
        critical_allergies = []
        other_allergies = []
        
        allergy_severity_keywords = {
            'critical': ['penicillin', 'sulfa', 'latex', 'iodine', 'contrast', 'shellfish', 'peanut', 'bee', 'wasp'],
            'high': ['aspirin', 'nsaid', 'codeine', 'morphine', 'cephalosporin']
        }
        
        for allergy in (patient.get('allergies') or []):
            allergy_lower = allergy.lower()
            is_critical = any(kw in allergy_lower for kw in allergy_severity_keywords['critical'])
            is_high = any(kw in allergy_lower for kw in allergy_severity_keywords['high'])
            
            if is_critical:
                critical_allergies.append({
                    "name": allergy,
                    "severity": "CRITICAL",
                    "cross_reactivity": _get_cross_reactivity(allergy)
                })
            elif is_high:
                critical_allergies.append({
                    "name": allergy,
                    "severity": "HIGH",
                    "cross_reactivity": _get_cross_reactivity(allergy)
                })
            else:
                other_allergies.append({
                    "name": allergy,
                    "severity": "STANDARD",
                    "cross_reactivity": []
                })
        
        # Categorize conditions by prescribing impact
        high_impact_conditions = []
        other_conditions = []
        
        condition_impact_keywords = {
            'kidney': 'Affects dosing of many drugs',
            'renal': 'Affects dosing of many drugs',
            'liver': 'Affects drug metabolism',
            'hepatic': 'Affects drug metabolism',
            'heart': 'Avoid NSAIDs, some calcium channel blockers',
            'cardiac': 'Avoid NSAIDs, some calcium channel blockers',
            'pregnant': 'Many drugs contraindicated',
            'pregnancy': 'Many drugs contraindicated',
            'asthma': 'Avoid beta-blockers',
            'diabetes': 'Monitor with steroids, some antibiotics',
            'bleeding': 'Avoid anticoagulants, NSAIDs',
            'ulcer': 'Avoid NSAIDs, steroids'
        }
        
        for condition in (patient.get('chronic_conditions') or []):
            condition_lower = condition.lower()
            impact = None
            for kw, desc in condition_impact_keywords.items():
                if kw in condition_lower:
                    impact = desc
                    break
            
            if impact:
                high_impact_conditions.append({
                    "name": condition,
                    "impact": impact,
                    "level": "HIGH"
                })
            else:
                other_conditions.append({
                    "name": condition,
                    "impact": None,
                    "level": "STANDARD"
                })
        
        # Get current medications with drug classes
        from backend.services.drug_normalization_service import DrugNormalizationService
        normalizer = DrugNormalizationService()
        
        current_meds_detailed = []
        for med in (patient.get('current_medications') or []):
            med_name = med.get('name', med) if isinstance(med, dict) else med
            norm = normalizer.normalize(med_name)
            current_meds_detailed.append({
                "name": med_name,
                "generic": norm.generic_name if norm else med_name,
                "class": norm.drug_class if norm else 'Unknown',
                "dosage": med.get('dosage', '') if isinstance(med, dict) else '',
                "frequency": med.get('frequency', '') if isinstance(med, dict) else ''
            })
        
        return {
            "success": True,
            "patient": {
                "uid": patient_uid,
                "name": patient.get('name'),
                "age": patient.get('age'),
                "gender": patient.get('gender'),
                "blood_group": patient.get('blood_group'),
                "weight_kg": patient.get('weight_kg'),
                "height_cm": patient.get('height_cm')
            },
            "allergies": {
                "critical": critical_allergies,
                "other": other_allergies,
                "total": len(critical_allergies) + len(other_allergies)
            },
            "conditions": {
                "high_impact": high_impact_conditions,
                "other": other_conditions,
                "total": len(high_impact_conditions) + len(other_conditions)
            },
            "current_medications": current_meds_detailed,
            "warnings_count": len(critical_allergies) + len(high_impact_conditions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Safety summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_cross_reactivity(allergy: str) -> List[str]:
    """Get cross-reactive drugs for an allergy"""
    cross_map = {
        'penicillin': ['Amoxicillin', 'Ampicillin', 'Piperacillin', 'Cephalosporins (10% risk)'],
        'sulfa': ['Sulfamethoxazole', 'Sulfasalazine', 'Some diuretics'],
        'aspirin': ['Ibuprofen', 'Naproxen', 'Diclofenac', 'All NSAIDs'],
        'cephalosporin': ['Cefixime', 'Ceftriaxone', 'Penicillins (low risk)'],
        'codeine': ['Morphine', 'Tramadol', 'Hydrocodone'],
        'latex': ['Banana', 'Avocado', 'Kiwi (food cross-reactivity)']
    }
    
    for key, drugs in cross_map.items():
        if key in allergy.lower():
            return drugs
    return []


