"""
Medication Models - Drug tracking, history, and interactions
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Boolean, Date, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class MedicationStatus(enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"
    ON_HOLD = "on_hold"


class InteractionSeverity(enum.Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"


class Medication(Base):
    """Master medication reference table"""
    __tablename__ = "medications"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Names
    generic_name = Column(String(200), nullable=False, index=True)
    brand_names = Column(JSON, default=list)  # List of brand names
    
    # Classification
    drug_class = Column(String(100), nullable=True)
    therapeutic_category = Column(String(100), nullable=True)
    
    # Details
    description = Column(Text, nullable=True)
    common_dosages = Column(JSON, default=list)
    routes = Column(JSON, default=list)  # oral, IV, topical, etc.
    
    # Safety
    contraindications = Column(JSON, default=list)
    common_side_effects = Column(JSON, default=list)
    serious_side_effects = Column(JSON, default=list)
    
    # External IDs
    rxcui = Column(String(50), nullable=True)  # RxNorm ID
    atc_code = Column(String(20), nullable=True)  # ATC Classification
    ndc_codes = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Medication {self.generic_name}>"


class MedicationHistory(Base):
    """Patient's medication history over time"""
    __tablename__ = "medication_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    medication_id = Column(Integer, ForeignKey("medications.id"), nullable=True)
    prescription_item_id = Column(Integer, ForeignKey("prescription_items.id"), nullable=True)
    
    # Medication Info (denormalized for history)
    medication_name = Column(String(200), nullable=False)
    generic_name = Column(String(200), nullable=True)
    dosage = Column(String(100), nullable=True)
    frequency = Column(String(100), nullable=True)
    route = Column(String(50), nullable=True)
    
    # Timeline
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    status = Column(SQLEnum(MedicationStatus), default=MedicationStatus.ACTIVE)
    
    # Reason
    reason = Column(Text, nullable=True)  # Why prescribed
    discontinuation_reason = Column(Text, nullable=True)  # Why stopped
    
    # Source
    prescriber = Column(String(200), nullable=True)
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="medication_history")
    
    def __repr__(self):
        return f"<MedicationHistory {self.medication_name} for patient {self.patient_id}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "medication_name": self.medication_name,
            "generic_name": self.generic_name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "start_date": str(self.start_date) if self.start_date else None,
            "end_date": str(self.end_date) if self.end_date else None,
            "status": self.status.value if self.status else None,
            "reason": self.reason,
            "prescriber": self.prescriber
        }


class DrugInteraction(Base):
    """Drug-drug interaction database"""
    __tablename__ = "drug_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Drugs involved
    drug1_name = Column(String(200), nullable=False, index=True)
    drug2_name = Column(String(200), nullable=False, index=True)
    drug1_id = Column(Integer, ForeignKey("medications.id"), nullable=True)
    drug2_id = Column(Integer, ForeignKey("medications.id"), nullable=True)
    
    # Interaction Details
    severity = Column(SQLEnum(InteractionSeverity), nullable=False)
    description = Column(Text, nullable=False)
    mechanism = Column(Text, nullable=True)
    clinical_effects = Column(Text, nullable=True)
    management = Column(Text, nullable=True)  # How to handle
    
    # Documentation
    evidence_level = Column(String(50), nullable=True)  # established, probable, suspected
    references = Column(JSON, default=list)
    
    # Flags
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<DrugInteraction {self.drug1_name} + {self.drug2_name}: {self.severity.value}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "drug1_name": self.drug1_name,
            "drug2_name": self.drug2_name,
            "severity": self.severity.value if self.severity else None,
            "description": self.description,
            "mechanism": self.mechanism,
            "clinical_effects": self.clinical_effects,
            "management": self.management
        }


class PatientDrugInteraction(Base):
    """Detected interactions for a specific patient"""
    __tablename__ = "patient_drug_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    interaction_id = Column(Integer, ForeignKey("drug_interactions.id"), nullable=True)
    
    # Medications involved
    medication1_name = Column(String(200), nullable=False)
    medication2_name = Column(String(200), nullable=False)
    
    # Severity (may be adjusted based on patient context)
    base_severity = Column(SQLEnum(InteractionSeverity), nullable=False)
    adjusted_severity = Column(SQLEnum(InteractionSeverity), nullable=True)
    severity_adjustment_reason = Column(Text, nullable=True)
    
    # Status
    is_dismissed = Column(Boolean, default=False)
    dismissed_by = Column(String(100), nullable=True)
    dismissed_at = Column(DateTime, nullable=True)
    dismissal_reason = Column(Text, nullable=True)
    
    # Detection
    detected_at = Column(DateTime, default=datetime.utcnow)
    source_prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=True)
    
    # Allergy-related
    is_allergy_based = Column(Boolean, default=False)
    allergy_info = Column(JSON, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "medication1_name": self.medication1_name,
            "medication2_name": self.medication2_name,
            "base_severity": self.base_severity.value if self.base_severity else None,
            "adjusted_severity": self.adjusted_severity.value if self.adjusted_severity else None,
            "is_dismissed": self.is_dismissed,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None
        }
