"""
Timeline Event Model - For temporal medical reasoning
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Boolean, Date, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class EventType(enum.Enum):
    PRESCRIPTION = "prescription"
    MEDICATION_START = "medication_start"
    MEDICATION_END = "medication_end"
    MEDICATION_CHANGE = "medication_change"
    DIAGNOSIS = "diagnosis"
    SYMPTOM_ONSET = "symptom_onset"
    SYMPTOM_RESOLUTION = "symptom_resolution"
    LAB_RESULT = "lab_result"
    VITAL_MEASUREMENT = "vital_measurement"
    VISIT = "visit"
    PROCEDURE = "procedure"
    HOSPITALIZATION = "hospitalization"
    ALLERGY_IDENTIFIED = "allergy_identified"
    ADVERSE_EVENT = "adverse_event"
    OTHER = "other"


class TimelineEvent(Base):
    """Chronological events for patient timeline"""
    __tablename__ = "timeline_events"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    
    # Event Info
    event_type = Column(SQLEnum(EventType), nullable=False)
    event_date = Column(DateTime, nullable=False)
    event_end_date = Column(DateTime, nullable=True)  # For events with duration
    
    # Description
    title = Column(String(300), nullable=False)
    description = Column(Text, nullable=True)
    
    # Related Entities
    related_entity_type = Column(String(50), nullable=True)
    related_entity_id = Column(Integer, nullable=True)
    related_data = Column(JSON, default=dict)  # Denormalized key data
    
    # Source Document
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    source_text = Column(Text, nullable=True)
    
    # Confidence
    confidence = Column(Float, nullable=True)
    date_precision = Column(String(20), nullable=True)  # exact, day, month, year, approximate
    
    # Flags
    is_inferred = Column(Boolean, default=False)  # True if date was inferred
    needs_review = Column(Boolean, default=False)
    
    # Extra Data
    extra_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="timeline_events")
    
    def __repr__(self):
        return f"<TimelineEvent {self.event_type.value}: {self.title} on {self.event_date}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "event_type": self.event_type.value if self.event_type else None,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "event_end_date": self.event_end_date.isoformat() if self.event_end_date else None,
            "title": self.title,
            "description": self.description,
            "related_data": self.related_data,
            "confidence": self.confidence,
            "is_inferred": self.is_inferred
        }
