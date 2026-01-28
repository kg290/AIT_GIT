"""
Document Model - For managing uploaded medical documents
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Enum as SQLEnum, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class DocumentType(enum.Enum):
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    SCAN_REPORT = "scan_report"
    DISCHARGE_SUMMARY = "discharge_summary"
    CONSULTATION_NOTE = "consultation_note"
    OTHER = "other"


class DocumentStatus(enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    OCR_COMPLETE = "ocr_complete"
    EXTRACTED = "extracted"
    REVIEWED = "reviewed"
    FAILED = "failed"


class ContentType(enum.Enum):
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Document(Base):
    """Medical document with OCR and extraction results"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Document Info
    document_id = Column(String(50), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, jpg, png, etc.
    file_size = Column(Integer, nullable=True)
    
    # Classification
    document_type = Column(SQLEnum(DocumentType), default=DocumentType.OTHER)
    content_type = Column(SQLEnum(ContentType), default=ContentType.UNKNOWN)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED)
    
    # Source Tracking
    upload_source = Column(String(100), nullable=True)  # web, api, mobile, etc.
    uploaded_by = Column(String(100), nullable=True)
    upload_ip = Column(String(50), nullable=True)
    
    # OCR Results
    raw_ocr_text = Column(Text, nullable=True)  # Original OCR output
    cleaned_text = Column(Text, nullable=True)  # Processed/cleaned text
    ocr_confidence = Column(Float, nullable=True)  # Overall confidence
    ocr_details = Column(JSON, nullable=True)  # Bounding boxes, per-word confidence
    
    # Extracted Data
    extracted_entities = Column(JSON, default=dict)
    extraction_confidence = Column(Float, nullable=True)
    
    # Processing Info
    processing_time_ms = Column(Integer, nullable=True)
    processing_errors = Column(JSON, default=list)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # Relationships
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    patient = relationship("Patient", back_populates="documents")
    
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    prescriptions = relationship("Prescription", back_populates="source_document")
    entities = relationship("MedicalEntity", back_populates="source_document")
    
    def __repr__(self):
        return f"<Document {self.document_id}: {self.original_filename}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "filename": self.original_filename,
            "file_type": self.file_type,
            "document_type": self.document_type.value if self.document_type else None,
            "content_type": self.content_type.value if self.content_type else None,
            "status": self.status.value if self.status else None,
            "ocr_confidence": self.ocr_confidence,
            "patient_id": self.patient_id,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }


class DocumentVersion(Base):
    """Version history for documents - maintains immutability"""
    __tablename__ = "document_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    
    # Stored content at this version
    raw_ocr_text = Column(Text, nullable=True)
    cleaned_text = Column(Text, nullable=True)
    extracted_entities = Column(JSON, default=dict)
    
    # Change tracking
    change_type = Column(String(50), nullable=True)  # ocr_correction, entity_edit, etc.
    change_description = Column(Text, nullable=True)
    changed_by = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="versions")
    
    def __repr__(self):
        return f"<DocumentVersion {self.document_id} v{self.version_number}>"
