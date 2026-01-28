"""
Audit and Correction Models - For compliance and tracking
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Enum as SQLEnum, Float, Boolean
from datetime import datetime
import enum
from .database import Base


class AuditAction(enum.Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    UPLOAD = "upload"
    PROCESS = "process"
    REVIEW = "review"
    CORRECT = "correct"
    DISMISS = "dismiss"
    EXPORT = "export"
    QUERY = "query"


class AuditLog(Base):
    """Immutable audit log for compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Action
    action = Column(SQLEnum(AuditAction), nullable=False)
    action_detail = Column(String(200), nullable=True)
    
    # Target
    entity_type = Column(String(50), nullable=False)  # patient, document, prescription, etc.
    entity_id = Column(Integer, nullable=True)
    entity_identifier = Column(String(100), nullable=True)  # e.g., patient_id, document_id
    
    # Actor
    user_id = Column(String(100), nullable=True)
    user_name = Column(String(200), nullable=True)
    user_role = Column(String(50), nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Data
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    changes = Column(JSON, nullable=True)
    
    # Context
    session_id = Column(String(100), nullable=True)
    request_id = Column(String(100), nullable=True)
    
    # Timestamp (immutable)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Extra Data
    extra_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<AuditLog {self.action.value} on {self.entity_type} {self.entity_id}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "action": self.action.value if self.action else None,
            "action_detail": self.action_detail,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "user_name": self.user_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "changes": self.changes
        }


class CorrectionType(enum.Enum):
    OCR_TEXT = "ocr_text"
    ENTITY = "entity"
    MEDICATION = "medication"
    DOSAGE = "dosage"
    FREQUENCY = "frequency"
    DIAGNOSIS = "diagnosis"
    INTERACTION_DISMISS = "interaction_dismiss"
    OTHER = "other"


class Correction(Base):
    """Human corrections to system outputs"""
    __tablename__ = "corrections"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # What was corrected
    correction_type = Column(SQLEnum(CorrectionType), nullable=False)
    
    # Source
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    entity_id = Column(Integer, nullable=True)  # Generic entity reference
    entity_type = Column(String(50), nullable=True)
    
    # Correction Details
    field_name = Column(String(100), nullable=True)
    original_value = Column(Text, nullable=True)
    corrected_value = Column(Text, nullable=True)
    
    # Context
    reason = Column(Text, nullable=True)
    confidence_before = Column(Float, nullable=True)
    
    # Correction Source Region (for OCR corrections)
    source_region = Column(JSON, nullable=True)
    
    # Who
    corrected_by = Column(String(100), nullable=False)
    corrected_at = Column(DateTime, default=datetime.utcnow)
    
    # Verification
    verified = Column(Boolean, default=False)
    verified_by = Column(String(100), nullable=True)
    verified_at = Column(DateTime, nullable=True)
    
    # Learning flag
    used_for_training = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Correction {self.correction_type.value}: {self.original_value} -> {self.corrected_value}>"


from sqlalchemy import Boolean, Float
