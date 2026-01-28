"""
Database Models for Medical AI Gateway
Complete schema for hospital-ready prescription management
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON, Table, Enum as SQLEnum
from sqlalchemy.orm import relationship, declarative_base
from enum import Enum

Base = declarative_base()


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class InteractionSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"


# Association table for patient allergies
patient_allergies = Table(
    'patient_allergies',
    Base.metadata,
    Column('patient_id', Integer, ForeignKey('patients.id'), primary_key=True),
    Column('allergy_id', Integer, ForeignKey('allergies.id'), primary_key=True)
)


class Patient(Base):
    """Patient master table"""
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Basic Info
    mrn = Column(String(50), unique=True, index=True, comment="Medical Record Number")
    external_id = Column(String(100), index=True, comment="Hospital/Clinic Patient ID")
    
    # Demographics
    first_name = Column(String(100))
    last_name = Column(String(100))
    full_name = Column(String(200), index=True)
    date_of_birth = Column(DateTime)
    gender = Column(String(20))
    blood_group = Column(String(10))
    
    # Contact
    phone = Column(String(20))
    email = Column(String(100))
    address = Column(Text)
    city = Column(String(100))
    state = Column(String(100))
    pincode = Column(String(20))
    
    # Emergency Contact
    emergency_contact_name = Column(String(100))
    emergency_contact_phone = Column(String(20))
    emergency_contact_relation = Column(String(50))
    
    # Medical Info
    chronic_conditions = Column(JSON, default=list)  # List of conditions
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    allergies = relationship("Allergy", secondary=patient_allergies, back_populates="patients")
    prescriptions = relationship("Prescription", back_populates="patient")
    
    def __repr__(self):
        return f"<Patient {self.full_name} ({self.mrn})>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'mrn': self.mrn,
            'external_id': self.external_id,
            'full_name': self.full_name,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'gender': self.gender,
            'blood_group': self.blood_group,
            'phone': self.phone,
            'email': self.email,
            'address': self.address,
            'city': self.city,
            'chronic_conditions': self.chronic_conditions or [],
            'allergies': [a.name for a in self.allergies],
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Allergy(Base):
    """Patient allergies table"""
    __tablename__ = 'allergies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, index=True)
    category = Column(String(50))  # drug, food, environmental
    severity = Column(String(20))  # mild, moderate, severe
    description = Column(Text)
    
    patients = relationship("Patient", secondary=patient_allergies, back_populates="allergies")


class Doctor(Base):
    """Doctor/Physician table"""
    __tablename__ = 'doctors'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Basic Info
    name = Column(String(200), index=True)
    qualification = Column(String(200))
    specialization = Column(String(100))
    registration_number = Column(String(100), unique=True)
    
    # Contact
    phone = Column(String(20))
    email = Column(String(100))
    
    # Clinic/Hospital
    clinic_name = Column(String(200))
    clinic_address = Column(Text)
    clinic_phone = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prescriptions = relationship("Prescription", back_populates="doctor")
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'qualification': self.qualification,
            'specialization': self.specialization,
            'registration_number': self.registration_number,
            'clinic_name': self.clinic_name
        }


class Prescription(Base):
    """Prescription/Document table"""
    __tablename__ = 'prescriptions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(100), unique=True, index=True)  # UUID
    
    # References
    patient_id = Column(Integer, ForeignKey('patients.id'), index=True)
    doctor_id = Column(Integer, ForeignKey('doctors.id'), index=True)
    
    # Document Info
    file_name = Column(String(255))
    file_path = Column(String(500))
    file_type = Column(String(50))
    file_size = Column(Integer)
    
    # Prescription Details
    prescription_date = Column(DateTime, index=True)
    
    # Extracted Data (JSON for flexibility)
    patient_info_extracted = Column(JSON)  # Name, age, gender from OCR
    diagnosis = Column(JSON, default=list)
    chief_complaints = Column(JSON, default=list)
    vitals = Column(JSON, default=dict)
    advice = Column(JSON, default=list)
    follow_up_date = Column(DateTime)
    investigations = Column(JSON, default=list)
    
    # OCR Data
    raw_ocr_text = Column(Text)
    ocr_confidence = Column(Float, default=0.0)
    
    # Processing
    processing_status = Column(String(50), default=ProcessingStatus.PENDING.value)
    extraction_confidence = Column(Float, default=0.0)
    extraction_method = Column(String(50))  # ai, regex, manual
    needs_review = Column(Boolean, default=False)
    review_reasons = Column(JSON, default=list)
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    
    # Safety
    has_interactions = Column(Boolean, default=False)
    has_allergy_alerts = Column(Boolean, default=False)
    safety_checked = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="prescriptions")
    doctor = relationship("Doctor", back_populates="prescriptions")
    medications = relationship("PrescriptionMedication", back_populates="prescription", cascade="all, delete-orphan")
    interactions = relationship("DrugInteractionRecord", back_populates="prescription", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'document_id': self.document_id,
            'patient_id': self.patient_id,
            'patient_name': self.patient_info_extracted.get('name') if self.patient_info_extracted else None,
            'doctor_id': self.doctor_id,
            'prescription_date': self.prescription_date.isoformat() if self.prescription_date else None,
            'diagnosis': self.diagnosis or [],
            'vitals': self.vitals or {},
            'advice': self.advice or [],
            'follow_up_date': self.follow_up_date.isoformat() if self.follow_up_date else None,
            'medications': [m.to_dict() for m in self.medications],
            'ocr_confidence': self.ocr_confidence,
            'extraction_confidence': self.extraction_confidence,
            'needs_review': self.needs_review,
            'has_interactions': self.has_interactions,
            'has_allergy_alerts': self.has_allergy_alerts,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class PrescriptionMedication(Base):
    """Medications in a prescription"""
    __tablename__ = 'prescription_medications'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prescription_id = Column(Integer, ForeignKey('prescriptions.id', ondelete='CASCADE'), index=True)
    
    # Drug Info
    name = Column(String(200), index=True)
    generic_name = Column(String(200))
    brand_name = Column(String(200))
    
    # Details
    dosage = Column(String(100))
    dosage_value = Column(Float)
    dosage_unit = Column(String(20))
    form = Column(String(50))  # tablet, capsule, syrup
    route = Column(String(50), default='oral')
    
    # Schedule
    frequency = Column(String(100))
    frequency_per_day = Column(Integer)
    timing = Column(String(100))  # before food, after food
    duration = Column(String(100))
    duration_days = Column(Integer)
    
    # Quantity
    quantity = Column(Integer)
    refills = Column(Integer, default=0)
    
    # Instructions
    instructions = Column(Text)
    
    # Status
    is_active = Column(Boolean, default=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    
    # Extraction confidence
    confidence = Column(Float, default=0.8)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prescription = relationship("Prescription", back_populates="medications")
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'generic_name': self.generic_name,
            'brand_name': self.brand_name,
            'dosage': self.dosage,
            'form': self.form,
            'route': self.route,
            'frequency': self.frequency,
            'timing': self.timing,
            'duration': self.duration,
            'quantity': self.quantity,
            'instructions': self.instructions,
            'is_active': self.is_active
        }


class DrugInteractionRecord(Base):
    """Detected drug interactions"""
    __tablename__ = 'drug_interactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prescription_id = Column(Integer, ForeignKey('prescriptions.id', ondelete='CASCADE'), index=True)
    
    drug1 = Column(String(200))
    drug2 = Column(String(200))
    severity = Column(String(50))  # minor, moderate, major, contraindicated
    description = Column(Text)
    mechanism = Column(Text)
    management = Column(Text)
    
    # Review
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    override_reason = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    prescription = relationship("Prescription", back_populates="interactions")
    
    def to_dict(self):
        return {
            'id': self.id,
            'drug1': self.drug1,
            'drug2': self.drug2,
            'severity': self.severity,
            'description': self.description,
            'mechanism': self.mechanism,
            'management': self.management,
            'acknowledged': self.acknowledged
        }


class AllergyAlert(Base):
    """Allergy alerts detected"""
    __tablename__ = 'allergy_alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prescription_id = Column(Integer, ForeignKey('prescriptions.id', ondelete='CASCADE'), index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), index=True)
    
    drug_name = Column(String(200))
    allergen = Column(String(100))
    cross_reactivity = Column(Boolean, default=False)
    alternatives = Column(JSON, default=list)
    
    # Review
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    override_reason = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """Audit trail for all actions"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Action Info
    action_type = Column(String(50), index=True)  # create, read, update, delete, process, review
    entity_type = Column(String(50), index=True)  # patient, prescription, medication
    entity_id = Column(String(100), index=True)
    
    # User
    user_id = Column(String(100), index=True)
    user_name = Column(String(200))
    user_role = Column(String(50))
    
    # Details
    action_detail = Column(Text)
    old_value = Column(JSON)
    new_value = Column(JSON)
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'action_type': self.action_type,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'user_name': self.user_name,
            'action_detail': self.action_detail,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class Analytics(Base):
    """Analytics and statistics cache"""
    __tablename__ = 'analytics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    metric_name = Column(String(100), index=True)
    metric_value = Column(Float)
    metric_data = Column(JSON)
    
    period = Column(String(20))  # daily, weekly, monthly
    period_start = Column(DateTime, index=True)
    period_end = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Drug master table for reference
class DrugMaster(Base):
    """Master drug database"""
    __tablename__ = 'drug_master'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Names
    generic_name = Column(String(200), index=True)
    brand_names = Column(JSON, default=list)  # List of brand names
    
    # Classification
    drug_class = Column(String(100))
    sub_class = Column(String(100))
    category = Column(String(100))
    
    # Details
    forms = Column(JSON, default=list)  # Available forms
    strengths = Column(JSON, default=list)  # Available strengths
    route = Column(String(50))
    
    # Safety
    pregnancy_category = Column(String(10))
    controlled_substance = Column(Boolean, default=False)
    schedule = Column(String(10))
    
    # Interactions and contraindications
    interaction_classes = Column(JSON, default=list)
    contraindications = Column(JSON, default=list)
    warnings = Column(JSON, default=list)
    
    # Usage
    indications = Column(JSON, default=list)
    dosing_info = Column(JSON)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
