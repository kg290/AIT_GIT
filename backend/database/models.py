"""
Production Database Models
SQLAlchemy models for hospital prescription management system
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, 
    ForeignKey, JSON, Enum as SQLEnum, Index, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class UserRole(enum.Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    PHARMACIST = "pharmacist"
    NURSE = "nurse"
    RECEPTIONIST = "receptionist"


class AlertSeverity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PrescriptionStatus(enum.Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    DISPENSED = "dispensed"
    FLAGGED = "flagged"
    CANCELLED = "cancelled"


# Association table for patient allergies
patient_allergies = Table(
    'patient_allergies',
    Base.metadata,
    Column('patient_id', Integer, ForeignKey('patients.id'), primary_key=True),
    Column('allergy_id', Integer, ForeignKey('allergies.id'), primary_key=True)
)

# Association table for patient conditions
patient_conditions = Table(
    'patient_conditions',
    Base.metadata,
    Column('patient_id', Integer, ForeignKey('patients.id'), primary_key=True),
    Column('condition_id', Integer, ForeignKey('conditions.id'), primary_key=True)
)


class User(Base):
    """Hospital staff users"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.RECEPTIONIST)
    department = Column(String(100))
    employee_id = Column(String(50), unique=True)
    phone = Column(String(20))
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    prescriptions_created = relationship("Prescription", back_populates="created_by_user", foreign_keys="Prescription.created_by")
    prescriptions_verified = relationship("Prescription", back_populates="verified_by_user", foreign_keys="Prescription.verified_by")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    __table_args__ = (
        Index('idx_user_role', 'role'),
        Index('idx_user_department', 'department'),
    )


class Patient(Base):
    """Patient records"""
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_uid = Column(String(50), unique=True, nullable=False, index=True)  # Hospital's patient ID
    
    # Demographics
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    date_of_birth = Column(DateTime)
    gender = Column(String(10))
    blood_group = Column(String(5))
    
    # Contact
    phone = Column(String(20))
    email = Column(String(100))
    address = Column(Text)
    emergency_contact_name = Column(String(100))
    emergency_contact_phone = Column(String(20))
    
    # Medical
    weight_kg = Column(Float)
    height_cm = Column(Float)
    notes = Column(Text)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    allergies = relationship("Allergy", secondary=patient_allergies, back_populates="patients")
    conditions = relationship("Condition", secondary=patient_conditions, back_populates="patients")
    prescriptions = relationship("Prescription", back_populates="patient", order_by="desc(Prescription.prescription_date)")
    medications = relationship("PatientMedication", back_populates="patient")
    timeline_events = relationship("TimelineEvent", back_populates="patient", order_by="desc(TimelineEvent.event_date)")
    
    __table_args__ = (
        Index('idx_patient_name', 'first_name', 'last_name'),
        Index('idx_patient_phone', 'phone'),
    )
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age(self):
        if self.date_of_birth:
            today = datetime.utcnow()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None


class Allergy(Base):
    """Allergy records"""
    __tablename__ = 'allergies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50))  # drug, food, environmental
    severity = Column(String(20))  # mild, moderate, severe
    
    patients = relationship("Patient", secondary=patient_allergies, back_populates="allergies")


class Condition(Base):
    """Chronic conditions"""
    __tablename__ = 'conditions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    icd_code = Column(String(20))  # ICD-10 code
    category = Column(String(50))
    
    patients = relationship("Patient", secondary=patient_conditions, back_populates="conditions")


class Prescription(Base):
    """Prescription records from scanned documents"""
    __tablename__ = 'prescriptions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prescription_uid = Column(String(50), unique=True, nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    
    # Prescription details
    prescription_date = Column(DateTime, nullable=False)
    doctor_name = Column(String(100))
    doctor_registration_no = Column(String(50))
    doctor_qualification = Column(String(100))
    clinic_name = Column(String(200))
    clinic_address = Column(Text)
    
    # Diagnosis
    diagnosis = Column(JSON)  # List of diagnoses
    chief_complaint = Column(Text)
    
    # Vitals at time of prescription
    vitals = Column(JSON)
    
    # Instructions
    advice = Column(JSON)
    follow_up_date = Column(DateTime)
    
    # Document info
    original_filename = Column(String(255))
    file_path = Column(String(500))
    raw_ocr_text = Column(Text)
    
    # Processing info
    ocr_confidence = Column(Float)
    extraction_confidence = Column(Float)
    processing_notes = Column(Text)
    
    # Status tracking
    status = Column(SQLEnum(PrescriptionStatus), default=PrescriptionStatus.PENDING)
    needs_review = Column(Boolean, default=False)
    review_reasons = Column(JSON)
    
    # Safety analysis
    safety_score = Column(Float)
    has_interactions = Column(Boolean, default=False)
    has_allergy_alerts = Column(Boolean, default=False)
    safety_alerts = Column(JSON)
    
    # Workflow
    created_by = Column(Integer, ForeignKey('users.id'))
    verified_by = Column(Integer, ForeignKey('users.id'))
    verified_at = Column(DateTime)
    dispensed_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="prescriptions")
    medications = relationship("PrescriptionMedication", back_populates="prescription", cascade="all, delete-orphan")
    created_by_user = relationship("User", back_populates="prescriptions_created", foreign_keys=[created_by])
    verified_by_user = relationship("User", back_populates="prescriptions_verified", foreign_keys=[verified_by])
    
    __table_args__ = (
        Index('idx_prescription_date', 'prescription_date'),
        Index('idx_prescription_status', 'status'),
        Index('idx_prescription_patient', 'patient_id'),
    )


class PrescriptionMedication(Base):
    """Medications in a prescription"""
    __tablename__ = 'prescription_medications'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prescription_id = Column(Integer, ForeignKey('prescriptions.id'), nullable=False)
    
    # Medication details
    name = Column(String(200), nullable=False)
    generic_name = Column(String(200))
    brand_name = Column(String(200))
    drug_class = Column(String(100))
    
    # Dosage
    dosage = Column(String(100))
    dosage_value = Column(Float)
    dosage_unit = Column(String(20))
    
    # Administration
    frequency = Column(String(100))
    frequency_per_day = Column(Integer)
    route = Column(String(50))  # oral, topical, injection, etc.
    timing = Column(String(100))  # before meals, after meals, etc.
    
    # Duration
    duration = Column(String(100))
    duration_days = Column(Integer)
    quantity = Column(Integer)
    refills = Column(Integer, default=0)
    
    # Instructions
    instructions = Column(Text)
    
    # Safety flags
    has_interaction = Column(Boolean, default=False)
    interaction_details = Column(JSON)
    has_allergy_risk = Column(Boolean, default=False)
    allergy_details = Column(JSON)
    
    # Status
    is_dispensed = Column(Boolean, default=False)
    dispensed_quantity = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prescription = relationship("Prescription", back_populates="medications")
    
    __table_args__ = (
        Index('idx_med_name', 'name'),
        Index('idx_med_generic', 'generic_name'),
    )


class PatientMedication(Base):
    """Patient's current and historical medications (for timeline tracking)"""
    __tablename__ = 'patient_medications'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    prescription_id = Column(Integer, ForeignKey('prescriptions.id'))
    
    # Medication
    name = Column(String(200), nullable=False)
    generic_name = Column(String(200))
    dosage = Column(String(100))
    frequency = Column(String(100))
    
    # Timeline
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Change tracking
    previous_dosage = Column(String(100))
    change_reason = Column(Text)
    prescriber = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="medications")
    
    __table_args__ = (
        Index('idx_patient_med_active', 'patient_id', 'is_active'),
    )


class TimelineEvent(Base):
    """Patient medical timeline events"""
    __tablename__ = 'timeline_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    prescription_id = Column(Integer, ForeignKey('prescriptions.id'))
    
    event_type = Column(String(50), nullable=False)  # prescription_added, medication_started, etc.
    event_date = Column(DateTime, nullable=False)
    description = Column(Text, nullable=False)
    details = Column(JSON)
    severity = Column(SQLEnum(AlertSeverity), default=AlertSeverity.INFO)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="timeline_events")
    
    __table_args__ = (
        Index('idx_timeline_patient_date', 'patient_id', 'event_date'),
        Index('idx_timeline_type', 'event_type'),
    )


class SafetyAlert(Base):
    """Active safety alerts for patients"""
    __tablename__ = 'safety_alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    prescription_id = Column(Integer, ForeignKey('prescriptions.id'))
    
    alert_type = Column(String(50), nullable=False)  # interaction, allergy, contraindication
    severity = Column(SQLEnum(AlertSeverity), nullable=False)
    
    drug1 = Column(String(200))
    drug2 = Column(String(200))
    
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    recommendation = Column(Text)
    
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(Integer, ForeignKey('users.id'))
    acknowledged_at = Column(DateTime)
    acknowledgment_note = Column(Text)
    
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_alert_patient', 'patient_id', 'is_resolved'),
        Index('idx_alert_severity', 'severity'),
    )


class AuditLog(Base):
    """HIPAA-compliant audit logging"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Who
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String(50))
    user_role = Column(String(50))
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    
    # What
    action = Column(String(100), nullable=False)  # view, create, update, delete, export, print
    resource_type = Column(String(50), nullable=False)  # patient, prescription, medication
    resource_id = Column(String(50))
    
    # Details
    description = Column(Text)
    old_values = Column(JSON)
    new_values = Column(JSON)
    
    # When
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_action', 'action'),
    )


class SystemSetting(Base):
    """System configuration"""
    __tablename__ = 'system_settings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    description = Column(Text)
    updated_by = Column(Integer, ForeignKey('users.id'))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DrugDatabase(Base):
    """Local drug database for normalization"""
    __tablename__ = 'drug_database'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    generic_name = Column(String(200), nullable=False, index=True)
    brand_names = Column(JSON)  # List of brand names
    drug_class = Column(String(100))
    category = Column(String(100))
    common_dosages = Column(JSON)
    routes = Column(JSON)
    contraindications = Column(JSON)
    interactions = Column(JSON)
    pregnancy_category = Column(String(5))
    is_controlled = Column(Boolean, default=False)
    schedule = Column(String(10))  # Schedule II, III, IV, V
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
