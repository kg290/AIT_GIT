"""
Production API Routes
Hospital-ready endpoints with authentication and audit logging
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import os
import uuid
import shutil

from backend.database.connection import get_db, db_manager
from backend.database.models import (
    User, Patient, Prescription, PrescriptionStatus, 
    SafetyAlert, UserRole, AuditLog
)
from backend.services.auth_service import (
    auth_service, get_current_user, get_optional_user,
    require_admin, require_clinical_staff, audit_service
)
from backend.services.production_patient_service import get_production_patient_service
from backend.services.complete_processor import complete_processor
from backend.config import settings


router = APIRouter(prefix="/api/hospital", tags=["Hospital API"])


# ==================== Pydantic Models ====================

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: str = "receptionist"
    department: Optional[str] = None
    employee_id: Optional[str] = None
    phone: Optional[str] = None


class PatientCreate(BaseModel):
    patient_uid: str
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    allergies: Optional[List[str]] = []
    conditions: Optional[List[str]] = []


class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None


# ==================== Authentication ====================

@router.post("/auth/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Login with username and password.
    Returns JWT token for subsequent requests.
    """
    user = auth_service.authenticate_user(db, login_data.username, login_data.password)
    
    if not user:
        audit_service.log(
            db=db,
            action="login_failed",
            resource_type="auth",
            description=f"Failed login attempt for username: {login_data.username}",
            request=request,
            success=False
        )
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create token
    token = auth_service.create_access_token(
        data={"sub": user.username, "role": user.role.value}
    )
    
    # Audit log
    audit_service.log(
        db=db,
        action="login",
        resource_type="auth",
        description=f"User {user.username} logged in",
        user=user,
        request=request
    )
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role.value,
            "department": user.department,
            "email": user.email
        }
    }


@router.post("/auth/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout (invalidate token on client side)"""
    audit_service.log(
        db=db,
        action="logout",
        resource_type="auth",
        description=f"User {current_user.username} logged out",
        user=current_user,
        request=request
    )
    return {"message": "Logged out successfully"}


@router.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
        "department": current_user.department,
        "email": current_user.email,
        "employee_id": current_user.employee_id
    }


# ==================== User Management (Admin only) ====================

@router.post("/users")
async def create_user(
    request: Request,
    user_data: UserCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new user (Admin only)"""
    # Check if username exists
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check email
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    try:
        role = UserRole(user_data.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role. Valid roles: {[r.value for r in UserRole]}")
    
    user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=auth_service.hash_password(user_data.password),
        full_name=user_data.full_name,
        role=role,
        department=user_data.department,
        employee_id=user_data.employee_id,
        phone=user_data.phone
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    audit_service.log(
        db=db,
        action="create",
        resource_type="user",
        resource_id=user.id,
        description=f"Created user {user.username} with role {user.role.value}",
        user=current_user,
        request=request
    )
    
    return {"message": "User created successfully", "user_id": user.id}


@router.get("/users")
async def list_users(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """List all users (Admin only)"""
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "full_name": u.full_name,
            "role": u.role.value,
            "department": u.department,
            "is_active": u.is_active,
            "last_login": u.last_login.isoformat() if u.last_login else None
        }
        for u in users
    ]


# ==================== Patient Management ====================

@router.post("/patients")
async def create_patient(
    request: Request,
    patient_data: PatientCreate,
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Create a new patient"""
    service = get_production_patient_service()
    
    try:
        dob = None
        if patient_data.date_of_birth:
            try:
                dob = datetime.fromisoformat(patient_data.date_of_birth)
            except:
                pass
        
        patient = service.create_patient(
            db=db,
            patient_uid=patient_data.patient_uid,
            first_name=patient_data.first_name,
            last_name=patient_data.last_name,
            date_of_birth=dob,
            gender=patient_data.gender,
            phone=patient_data.phone,
            email=patient_data.email,
            address=patient_data.address,
            allergies=patient_data.allergies,
            conditions=patient_data.conditions,
            user=current_user,
            request=request
        )
        
        return {
            "message": "Patient created successfully",
            "patient_id": patient.id,
            "patient_uid": patient.patient_uid
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/patients")
async def list_patients(
    request: Request,
    query: Optional[str] = Query(None, description="Search by name, ID, or phone"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Search and list patients"""
    service = get_production_patient_service()
    patients = service.search_patients(db, query=query, limit=limit, offset=offset)
    
    # Audit log for patient list access
    audit_service.log(
        db=db,
        action="view_list",
        resource_type="patient",
        description=f"Viewed patient list (query: {query})",
        user=current_user,
        request=request
    )
    
    return [
        {
            "id": p.id,
            "patient_uid": p.patient_uid,
            "name": p.full_name,
            "age": p.age,
            "gender": p.gender,
            "phone": p.phone,
            "allergies_count": len(p.allergies),
            "prescriptions_count": len(p.prescriptions),
            "last_visit": p.prescriptions[0].prescription_date.isoformat() if p.prescriptions else None
        }
        for p in patients
    ]


@router.get("/patients/{patient_uid}")
async def get_patient(
    request: Request,
    patient_uid: str,
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Get patient details"""
    service = get_production_patient_service()
    patient = service.get_patient_by_uid(db, patient_uid)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Audit log
    audit_service.log(
        db=db,
        action="view",
        resource_type="patient",
        resource_id=patient.id,
        description=f"Viewed patient {patient.full_name}",
        user=current_user,
        request=request
    )
    
    summary = service.get_patient_summary(db, patient.id)
    return summary


@router.get("/patients/{patient_uid}/timeline")
async def get_patient_timeline(
    request: Request,
    patient_uid: str,
    event_type: Optional[str] = None,
    limit: int = Query(100, le=500),
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Get patient medical timeline"""
    service = get_production_patient_service()
    patient = service.get_patient_by_uid(db, patient_uid)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    timeline = service.get_patient_timeline(db, patient.id, event_type=event_type, limit=limit)
    
    return {
        "patient_uid": patient_uid,
        "total_events": len(timeline),
        "timeline": timeline
    }


@router.post("/patients/{patient_uid}/allergies")
async def add_patient_allergy(
    request: Request,
    patient_uid: str,
    allergy: str = Form(...),
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Add allergy to patient"""
    service = get_production_patient_service()
    patient = service.get_patient_by_uid(db, patient_uid)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    service.add_patient_allergy(db, patient.id, allergy, user=current_user, request=request)
    
    return {"message": f"Allergy '{allergy}' added successfully"}


@router.post("/patients/{patient_uid}/conditions")
async def add_patient_condition(
    request: Request,
    patient_uid: str,
    condition: str = Form(...),
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Add condition to patient"""
    service = get_production_patient_service()
    patient = service.get_patient_by_uid(db, patient_uid)
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    service.add_patient_condition(db, patient.id, condition, user=current_user, request=request)
    
    return {"message": f"Condition '{condition}' added successfully"}


# ==================== Prescription Scanning ====================

@router.post("/prescriptions/scan")
async def scan_prescription(
    request: Request,
    file: UploadFile = File(...),
    patient_uid: str = Form(...),
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """
    Scan a prescription and automatically process it.
    - Extracts data using OCR + AI
    - Saves to patient record
    - Builds timeline
    - Runs safety analysis
    """
    service = get_production_patient_service()
    
    # Find patient
    patient = service.get_patient_by_uid(db, patient_uid)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_uid} not found")
    
    # Validate file type
    allowed_types = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = settings.UPLOAD_DIR / f"{file_id}{file_ext}"
    
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Get patient allergies for processing
    patient_allergies = [a.name for a in patient.allergies]
    
    try:
        # Process with OCR
        result = complete_processor.process(
            file_path=str(file_path),
            patient_allergies=patient_allergies,
            patient_id=patient_uid,
            save_to_db=False
        )
        
        prescription_data = result.to_dict()
        prescription_data['filename'] = file.filename
        prescription_data['file_path'] = str(file_path)
        
        # Add prescription to patient
        add_result = service.add_prescription(
            db=db,
            patient_id=patient.id,
            prescription_data=prescription_data,
            user=current_user,
            request=request
        )
        
        # Get updated summary
        summary = service.get_patient_summary(db, patient.id)
        timeline = service.get_patient_timeline(db, patient.id, limit=10)
        
        return JSONResponse(content={
            'success': True,
            'message': f"Prescription #{add_result['prescription_number']} processed successfully",
            'prescription_uid': add_result['prescription_uid'],
            'extracted_data': prescription_data,
            'analysis': {
                'changes_detected': add_result['changes_detected'],
                'safety_analysis': add_result['safety_analysis'],
                'needs_review': add_result['needs_review']
            },
            'patient_summary': summary,
            'recent_timeline': timeline
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/prescriptions/{prescription_uid}")
async def get_prescription(
    request: Request,
    prescription_uid: str,
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Get prescription details"""
    prescription = db.query(Prescription).filter(
        Prescription.prescription_uid == prescription_uid
    ).first()
    
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    # Audit log
    audit_service.log(
        db=db,
        action="view",
        resource_type="prescription",
        resource_id=prescription.id,
        description=f"Viewed prescription {prescription_uid}",
        user=current_user,
        request=request
    )
    
    return {
        "prescription_uid": prescription.prescription_uid,
        "patient_uid": prescription.patient.patient_uid,
        "patient_name": prescription.patient.full_name,
        "prescription_date": prescription.prescription_date.isoformat(),
        "doctor_name": prescription.doctor_name,
        "clinic_name": prescription.clinic_name,
        "diagnosis": prescription.diagnosis,
        "vitals": prescription.vitals,
        "medications": [
            {
                "name": m.name,
                "generic_name": m.generic_name,
                "dosage": m.dosage,
                "frequency": m.frequency,
                "duration": m.duration,
                "instructions": m.instructions,
                "has_interaction": m.has_interaction,
                "has_allergy_risk": m.has_allergy_risk
            }
            for m in prescription.medications
        ],
        "status": prescription.status.value,
        "needs_review": prescription.needs_review,
        "safety_score": prescription.safety_score,
        "safety_alerts": prescription.safety_alerts,
        "confidence": prescription.extraction_confidence
    }


@router.post("/prescriptions/{prescription_uid}/verify")
async def verify_prescription(
    request: Request,
    prescription_uid: str,
    notes: Optional[str] = Form(None),
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Verify/approve a prescription (Pharmacist or Doctor)"""
    if current_user.role not in [UserRole.PHARMACIST, UserRole.DOCTOR, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Only pharmacists or doctors can verify prescriptions")
    
    prescription = db.query(Prescription).filter(
        Prescription.prescription_uid == prescription_uid
    ).first()
    
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    prescription.status = PrescriptionStatus.VERIFIED
    prescription.verified_by = current_user.id
    prescription.verified_at = datetime.utcnow()
    prescription.processing_notes = notes
    
    db.commit()
    
    audit_service.log(
        db=db,
        action="verify",
        resource_type="prescription",
        resource_id=prescription.id,
        description=f"Verified prescription {prescription_uid}",
        user=current_user,
        request=request
    )
    
    return {"message": "Prescription verified successfully"}


# ==================== Safety Alerts ====================

@router.get("/alerts")
async def get_active_alerts(
    request: Request,
    patient_uid: Optional[str] = None,
    severity: Optional[str] = None,
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Get active safety alerts"""
    query = db.query(SafetyAlert).filter(SafetyAlert.is_resolved == False)
    
    if patient_uid:
        patient = db.query(Patient).filter(Patient.patient_uid == patient_uid).first()
        if patient:
            query = query.filter(SafetyAlert.patient_id == patient.id)
    
    alerts = query.order_by(SafetyAlert.created_at.desc()).limit(100).all()
    
    return [
        {
            "id": a.id,
            "patient_uid": a.patient_id,
            "alert_type": a.alert_type,
            "severity": a.severity.value,
            "title": a.title,
            "description": a.description,
            "recommendation": a.recommendation,
            "created_at": a.created_at.isoformat()
        }
        for a in alerts
    ]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    request: Request,
    alert_id: int,
    note: str = Form(...),
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Acknowledge a safety alert"""
    alert = db.query(SafetyAlert).filter(SafetyAlert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.is_acknowledged = True
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledgment_note = note
    
    db.commit()
    
    audit_service.log(
        db=db,
        action="acknowledge",
        resource_type="safety_alert",
        resource_id=alert.id,
        description=f"Acknowledged safety alert: {alert.title}",
        user=current_user,
        request=request
    )
    
    return {"message": "Alert acknowledged"}


# ==================== Dashboard Stats ====================

@router.get("/dashboard/stats")
async def get_dashboard_stats(
    current_user: User = Depends(require_clinical_staff),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics"""
    from sqlalchemy import func
    
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    
    total_patients = db.query(func.count(Patient.id)).scalar()
    total_prescriptions = db.query(func.count(Prescription.id)).scalar()
    
    prescriptions_today = db.query(func.count(Prescription.id)).filter(
        func.date(Prescription.created_at) == today
    ).scalar()
    
    prescriptions_this_week = db.query(func.count(Prescription.id)).filter(
        func.date(Prescription.created_at) >= week_ago
    ).scalar()
    
    pending_review = db.query(func.count(Prescription.id)).filter(
        Prescription.status == PrescriptionStatus.PENDING,
        Prescription.needs_review == True
    ).scalar()
    
    active_alerts = db.query(func.count(SafetyAlert.id)).filter(
        SafetyAlert.is_resolved == False
    ).scalar()
    
    critical_alerts = db.query(func.count(SafetyAlert.id)).filter(
        SafetyAlert.is_resolved == False,
        SafetyAlert.severity == 'critical'
    ).scalar()
    
    return {
        "total_patients": total_patients,
        "total_prescriptions": total_prescriptions,
        "prescriptions_today": prescriptions_today,
        "prescriptions_this_week": prescriptions_this_week,
        "pending_review": pending_review,
        "active_alerts": active_alerts,
        "critical_alerts": critical_alerts
    }
