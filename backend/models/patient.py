"""
Patient Model - Core entity for patient management
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Date, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class Gender(enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class Patient(Base):
    """Patient entity with full medical profile"""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Basic Information
    patient_id = Column(String(50), unique=True, index=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(SQLEnum(Gender), default=Gender.UNKNOWN)
    
    # Contact
    phone = Column(String(20), nullable=True)
    email = Column(String(100), nullable=True)
    address = Column(Text, nullable=True)
    
    # Medical Profile
    blood_group = Column(String(10), nullable=True)
    allergies = Column(JSON, default=list)  # List of known allergies
    chronic_conditions = Column(JSON, default=list)  # List of chronic conditions
    emergency_contact = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data = Column(JSON, default=dict)
    
    # Relationships
    documents = relationship("Document", back_populates="patient", cascade="all, delete-orphan")
    prescriptions = relationship("Prescription", back_populates="patient", cascade="all, delete-orphan")
    medication_history = relationship("MedicationHistory", back_populates="patient", cascade="all, delete-orphan")
    symptoms = relationship("Symptom", back_populates="patient", cascade="all, delete-orphan")
    diagnoses = relationship("Diagnosis", back_populates="patient", cascade="all, delete-orphan")
    vitals = relationship("Vital", back_populates="patient", cascade="all, delete-orphan")
    timeline_events = relationship("TimelineEvent", back_populates="patient", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Patient {self.patient_id}: {self.first_name} {self.last_name}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "date_of_birth": str(self.date_of_birth) if self.date_of_birth else None,
            "gender": self.gender.value if self.gender else None,
            "phone": self.phone,
            "email": self.email,
            "allergies": self.allergies,
            "chronic_conditions": self.chronic_conditions,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
