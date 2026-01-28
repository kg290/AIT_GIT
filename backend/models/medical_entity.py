"""
Medical Entity Models - Symptoms, Diagnoses, Vitals, etc.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Enum as SQLEnum, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class EntityType(enum.Enum):
    MEDICATION = "medication"
    DOSAGE = "dosage"
    FREQUENCY = "frequency"
    DURATION = "duration"
    ROUTE = "route"
    SYMPTOM = "symptom"
    DIAGNOSIS = "diagnosis"
    VITAL = "vital"
    LAB_VALUE = "lab_value"
    DATE = "date"
    PROCEDURE = "procedure"
    ALLERGY = "allergy"
    OTHER = "other"


class MedicalEntity(Base):
    """Extracted medical entities from documents"""
    __tablename__ = "medical_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Source
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    
    # Entity Info
    entity_type = Column(SQLEnum(EntityType), nullable=False)
    entity_text = Column(String(500), nullable=False)  # Raw text
    normalized_text = Column(String(500), nullable=True)  # Standardized form
    
    # Confidence
    confidence = Column(Float, nullable=True)
    extraction_method = Column(String(50), nullable=True)  # regex, ner, llm, etc.
    
    # Location in document
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    bounding_box = Column(JSON, nullable=True)  # {"x": 0, "y": 0, "width": 0, "height": 0}
    page_number = Column(Integer, nullable=True)
    
    # Additional data
    attributes = Column(JSON, default=dict)  # Entity-specific attributes
    
    # Flags
    is_uncertain = Column(Boolean, default=False)
    needs_review = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    source_document = relationship("Document", back_populates="entities")
    
    def __repr__(self):
        return f"<MedicalEntity {self.entity_type.value}: {self.entity_text}>"


from sqlalchemy import Boolean


class Symptom(Base):
    """Patient symptoms tracking"""
    __tablename__ = "symptoms"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Symptom Info
    symptom_name = Column(String(200), nullable=False)
    normalized_name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    
    # Severity
    severity = Column(String(50), nullable=True)  # mild, moderate, severe
    
    # Timing
    onset_date = Column(DateTime, nullable=True)
    resolution_date = Column(DateTime, nullable=True)
    duration = Column(String(100), nullable=True)
    is_ongoing = Column(Boolean, default=True)
    
    # Body location
    body_location = Column(String(100), nullable=True)
    
    # Confidence
    confidence = Column(Float, nullable=True)
    source_text = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="symptoms")
    
    def to_dict(self):
        return {
            "id": self.id,
            "symptom_name": self.symptom_name,
            "severity": self.severity,
            "onset_date": self.onset_date.isoformat() if self.onset_date else None,
            "is_ongoing": self.is_ongoing,
            "confidence": self.confidence
        }


class Diagnosis(Base):
    """Patient diagnoses tracking"""
    __tablename__ = "diagnoses"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Diagnosis Info
    diagnosis_name = Column(String(300), nullable=False)
    normalized_name = Column(String(300), nullable=True)
    
    # Codes
    icd10_code = Column(String(20), nullable=True)
    snomed_code = Column(String(50), nullable=True)
    
    # Classification
    diagnosis_type = Column(String(50), nullable=True)  # primary, secondary, differential
    status = Column(String(50), nullable=True)  # confirmed, suspected, ruled_out
    
    # Timing
    diagnosis_date = Column(DateTime, nullable=True)
    resolution_date = Column(DateTime, nullable=True)
    is_chronic = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Source
    diagnosed_by = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    source_text = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="diagnoses")
    
    def to_dict(self):
        return {
            "id": self.id,
            "diagnosis_name": self.diagnosis_name,
            "icd10_code": self.icd10_code,
            "diagnosis_type": self.diagnosis_type,
            "status": self.status,
            "diagnosis_date": self.diagnosis_date.isoformat() if self.diagnosis_date else None,
            "is_chronic": self.is_chronic,
            "is_active": self.is_active
        }


class Vital(Base):
    """Patient vital signs"""
    __tablename__ = "vitals"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Vital Info
    vital_type = Column(String(50), nullable=False)  # bp, heart_rate, temperature, etc.
    
    # Values
    value = Column(String(100), nullable=False)  # Raw value
    numeric_value = Column(Float, nullable=True)
    numeric_value_2 = Column(Float, nullable=True)  # For BP systolic/diastolic
    unit = Column(String(20), nullable=True)
    
    # Interpretation
    interpretation = Column(String(50), nullable=True)  # normal, high, low, critical
    reference_range = Column(String(100), nullable=True)
    
    # Timing
    recorded_at = Column(DateTime, nullable=True)
    
    # Source
    confidence = Column(Float, nullable=True)
    source_text = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="vitals")
    
    def to_dict(self):
        return {
            "id": self.id,
            "vital_type": self.vital_type,
            "value": self.value,
            "unit": self.unit,
            "interpretation": self.interpretation,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None
        }
