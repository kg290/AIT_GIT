"""
Prescription Model - Structured prescription data
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Boolean, Date
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Prescription(Base):
    """Structured prescription extracted from documents"""
    __tablename__ = "prescriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(String(50), unique=True, index=True, nullable=False)
    
    # Source
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    
    # Prescription Info
    prescriber_name = Column(String(200), nullable=True)
    prescriber_license = Column(String(100), nullable=True)
    clinic_name = Column(String(200), nullable=True)
    
    # Dates
    prescription_date = Column(Date, nullable=True)
    valid_until = Column(Date, nullable=True)
    
    # Diagnosis/Reason
    diagnosis = Column(Text, nullable=True)
    chief_complaint = Column(Text, nullable=True)
    
    # Raw text
    raw_prescription_text = Column(Text, nullable=True)
    
    # Processing
    is_structured = Column(Boolean, default=False)
    structure_confidence = Column(Float, nullable=True)
    has_ambiguity = Column(Boolean, default=False)
    ambiguity_notes = Column(JSON, default=list)
    
    # Flags
    needs_review = Column(Boolean, default=False)
    reviewed = Column(Boolean, default=False)
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_document = relationship("Document", back_populates="prescriptions")
    patient = relationship("Patient", back_populates="prescriptions")
    items = relationship("PrescriptionItem", back_populates="prescription", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Prescription {self.prescription_id}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "prescription_id": self.prescription_id,
            "patient_id": self.patient_id,
            "prescriber_name": self.prescriber_name,
            "prescription_date": str(self.prescription_date) if self.prescription_date else None,
            "diagnosis": self.diagnosis,
            "is_structured": self.is_structured,
            "structure_confidence": self.structure_confidence,
            "needs_review": self.needs_review,
            "items": [item.to_dict() for item in self.items] if self.items else [],
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PrescriptionItem(Base):
    """Individual medication item in a prescription"""
    __tablename__ = "prescription_items"
    
    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=False)
    
    # Medication Details
    medication_name = Column(String(200), nullable=False)
    generic_name = Column(String(200), nullable=True)
    brand_name = Column(String(200), nullable=True)
    
    # Dosage
    dosage = Column(String(100), nullable=True)  # e.g., "500mg"
    dosage_value = Column(Float, nullable=True)  # 500
    dosage_unit = Column(String(20), nullable=True)  # mg
    
    # Frequency
    frequency = Column(String(100), nullable=True)  # e.g., "1-0-1" or "twice daily"
    frequency_parsed = Column(JSON, nullable=True)  # {"morning": 1, "afternoon": 0, "night": 1}
    times_per_day = Column(Integer, nullable=True)
    
    # Route
    route = Column(String(50), nullable=True)  # oral, topical, injection, etc.
    
    # Duration
    duration = Column(String(100), nullable=True)  # e.g., "7 days"
    duration_days = Column(Integer, nullable=True)  # 7
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    
    # Quantity
    quantity = Column(Integer, nullable=True)
    refills = Column(Integer, default=0)
    
    # Instructions
    instructions = Column(Text, nullable=True)  # "Take with food"
    warnings = Column(Text, nullable=True)
    
    # Confidence & Extraction
    extraction_confidence = Column(Float, nullable=True)
    source_text = Column(Text, nullable=True)  # Original text this was extracted from
    source_region = Column(JSON, nullable=True)  # Bounding box in document
    
    # Flags
    is_uncertain = Column(Boolean, default=False)
    uncertainty_reason = Column(String(200), nullable=True)
    
    # Sequence
    sequence_order = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    prescription = relationship("Prescription", back_populates="items")
    
    def __repr__(self):
        return f"<PrescriptionItem {self.medication_name} {self.dosage}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "medication_name": self.medication_name,
            "generic_name": self.generic_name,
            "brand_name": self.brand_name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "frequency_parsed": self.frequency_parsed,
            "route": self.route,
            "duration": self.duration,
            "duration_days": self.duration_days,
            "instructions": self.instructions,
            "extraction_confidence": self.extraction_confidence,
            "is_uncertain": self.is_uncertain,
            "uncertainty_reason": self.uncertainty_reason
        }
