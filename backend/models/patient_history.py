"""
Patient History Models - Longitudinal patient data tracking
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Date, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class HistoryStatus(enum.Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ONGOING = "ongoing"
    DISCONTINUED = "discontinued"


class SeverityLevel(enum.Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class PatientMedicationHistory(Base):
    """Comprehensive medication history for longitudinal tracking"""
    __tablename__ = "patient_medication_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    
    # Medication Details
    medication_name = Column(String(200), nullable=False)
    generic_name = Column(String(200), nullable=True)
    dosage = Column(String(100), nullable=True)
    frequency = Column(String(100), nullable=True)
    route = Column(String(50), nullable=True)
    
    # Status and Timeline
    status = Column(SQLEnum(HistoryStatus), default=HistoryStatus.ACTIVE)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    
    # Prescribing Information
    prescribing_doctor = Column(String(200), nullable=True)
    prescription_id = Column(String(100), nullable=True)
    
    # Additional Context
    indication = Column(Text, nullable=True)  # Why prescribed
    notes = Column(Text, nullable=True)
    side_effects_reported = Column(JSON, default=list)
    effectiveness = Column(String(50), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<PatientMedicationHistory {self.medication_name} - {self.status.value}>"


class PatientConditionHistory(Base):
    """Patient diagnosis/condition history for longitudinal tracking"""
    __tablename__ = "patient_condition_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    
    # Diagnosis Details
    condition_name = Column(String(300), nullable=False)
    icd10_code = Column(String(20), nullable=True)
    diagnosis_type = Column(String(50), nullable=True)  # primary, secondary, differential
    
    # Status and Timeline
    status = Column(SQLEnum(HistoryStatus), default=HistoryStatus.ACTIVE)
    diagnosed_date = Column(Date, nullable=True)
    resolved_date = Column(Date, nullable=True)
    
    # Clinical Information
    severity = Column(SQLEnum(SeverityLevel), nullable=True)
    confidence = Column(Float, nullable=True)  # Diagnosis confidence 0-1
    diagnosing_doctor = Column(String(200), nullable=True)
    
    # Supporting Information
    investigations = Column(JSON, default=list)  # Tests, scans, etc.
    notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<PatientConditionHistory {self.condition_name} - {self.status.value}>"


class PatientSymptomHistory(Base):
    """Patient symptom tracking over time"""
    __tablename__ = "patient_symptom_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    
    # Symptom Details
    symptom_name = Column(String(200), nullable=False)
    body_location = Column(String(100), nullable=True)
    
    # Severity and Status
    severity = Column(SQLEnum(SeverityLevel), nullable=True)
    status = Column(SQLEnum(HistoryStatus), default=HistoryStatus.ACTIVE)
    progression = Column(String(50), nullable=True)  # improving, stable, worsening
    
    # Timeline
    onset_date = Column(Date, nullable=True)
    resolved_date = Column(Date, nullable=True)
    duration = Column(String(100), nullable=True)
    
    # Clinical Context
    associated_conditions = Column(JSON, default=list)
    triggers = Column(JSON, default=list)
    notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<PatientSymptomHistory {self.symptom_name} - {self.severity}>"


class VisitSummary(Base):
    """Summary of patient visits for quick reference"""
    __tablename__ = "visit_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    
    # Visit Information
    visit_id = Column(String(50), unique=True, index=True, nullable=False)
    visit_date = Column(DateTime, nullable=False)
    visit_type = Column(String(50), nullable=True)  # OPD, Emergency, Follow-up, etc.
    
    # Medical Staff
    attending_doctor = Column(String(200), nullable=True)
    department = Column(String(100), nullable=True)
    
    # Clinical Summary
    chief_complaint = Column(Text, nullable=True)
    examination_findings = Column(Text, nullable=True)
    diagnosis = Column(JSON, default=list)
    treatment_prescribed = Column(JSON, default=list)
    
    # Vitals (snapshot)
    vitals = Column(JSON, default=dict)  # BP, pulse, temp, weight, etc.
    
    # Follow-up
    follow_up_date = Column(Date, nullable=True)
    follow_up_notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<VisitSummary {self.visit_id} - {self.visit_date}>"
