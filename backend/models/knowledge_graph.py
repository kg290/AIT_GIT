"""
Knowledge Graph Models - For relationship tracking
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class RelationshipType(enum.Enum):
    # Patient relationships
    PATIENT_HAS_MEDICATION = "patient_has_medication"
    PATIENT_HAS_CONDITION = "patient_has_condition"
    PATIENT_HAS_SYMPTOM = "patient_has_symptom"
    PATIENT_HAS_ALLERGY = "patient_has_allergy"
    PATIENT_VISITS_DOCTOR = "patient_visits_doctor"
    
    # Medication relationships
    MEDICATION_TREATS_CONDITION = "medication_treats_condition"
    MEDICATION_CAUSES_SYMPTOM = "medication_causes_symptom"
    MEDICATION_INTERACTS_WITH = "medication_interacts_with"
    MEDICATION_CONTRAINDICATED_FOR = "medication_contraindicated_for"
    
    # Condition relationships
    CONDITION_HAS_SYMPTOM = "condition_has_symptom"
    CONDITION_REQUIRES_MEDICATION = "condition_requires_medication"
    
    # Temporal relationships
    PRECEDED_BY = "preceded_by"
    FOLLOWED_BY = "followed_by"
    CONCURRENT_WITH = "concurrent_with"
    
    # Causal relationships
    CAUSED_BY = "caused_by"
    RESULTS_IN = "results_in"


class NodeType(enum.Enum):
    PATIENT = "patient"
    MEDICATION = "medication"
    CONDITION = "condition"
    SYMPTOM = "symptom"
    DOCTOR = "doctor"
    VISIT = "visit"
    PRESCRIPTION = "prescription"
    LAB_RESULT = "lab_result"
    VITAL = "vital"


class KnowledgeNode(Base):
    """Node in the knowledge graph"""
    __tablename__ = "knowledge_nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Node identification
    node_type = Column(SQLEnum(NodeType), nullable=False, index=True)
    external_id = Column(String(100), nullable=True, index=True)  # e.g., patient_id, drug code
    name = Column(String(200), nullable=False)
    normalized_name = Column(String(200), nullable=True, index=True)
    
    # Properties
    properties = Column(JSON, default=dict)
    
    # Source tracking
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    source_text = Column(Text, nullable=True)
    extraction_confidence = Column(Float, default=1.0)
    
    # Temporal
    valid_from = Column(DateTime, nullable=True)
    valid_until = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    outgoing_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.source_node_id", back_populates="source_node")
    incoming_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.target_node_id", back_populates="target_node")
    
    def to_dict(self):
        return {
            "id": self.id,
            "node_type": self.node_type.value if self.node_type else None,
            "external_id": self.external_id,
            "name": self.name,
            "properties": self.properties,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class KnowledgeEdge(Base):
    """Edge (relationship) in the knowledge graph"""
    __tablename__ = "knowledge_edges"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Relationship
    source_node_id = Column(Integer, ForeignKey("knowledge_nodes.id"), nullable=False, index=True)
    target_node_id = Column(Integer, ForeignKey("knowledge_nodes.id"), nullable=False, index=True)
    relationship_type = Column(SQLEnum(RelationshipType), nullable=False, index=True)
    
    # Properties
    properties = Column(JSON, default=dict)
    weight = Column(Float, default=1.0)
    confidence = Column(Float, default=1.0)
    
    # Source tracking
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    evidence_text = Column(Text, nullable=True)
    
    # Temporal validity
    valid_from = Column(DateTime, nullable=True)
    valid_until = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Version tracking for history
    version = Column(Integer, default=1)
    previous_version_id = Column(Integer, ForeignKey("knowledge_edges.id"), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_node = relationship("KnowledgeNode", foreign_keys=[source_node_id], back_populates="outgoing_edges")
    target_node = relationship("KnowledgeNode", foreign_keys=[target_node_id], back_populates="incoming_edges")
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relationship_type": self.relationship_type.value if self.relationship_type else None,
            "properties": self.properties,
            "confidence": self.confidence,
            "is_active": self.is_active,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None
        }


class PatientMedicationHistory(Base):
    """Detailed medication history for a patient"""
    __tablename__ = "patient_medication_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    
    # Medication details
    medication_name = Column(String(200), nullable=False)
    generic_name = Column(String(200), nullable=True)
    dosage = Column(String(100), nullable=True)
    frequency = Column(String(100), nullable=True)
    route = Column(String(50), nullable=True)
    
    # Prescription info
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=True)
    prescribing_doctor = Column(String(200), nullable=True)
    
    # Temporal
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Status
    status = Column(String(50), default="active")  # active, completed, discontinued, changed
    discontinuation_reason = Column(Text, nullable=True)
    
    # Source
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "medication_name": self.medication_name,
            "generic_name": self.generic_name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_active": self.is_active,
            "status": self.status
        }


class PatientConditionHistory(Base):
    """Diagnosis/condition history for a patient"""
    __tablename__ = "patient_condition_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    
    # Condition details
    condition_name = Column(String(200), nullable=False)
    icd_code = Column(String(20), nullable=True)
    severity = Column(String(50), nullable=True)
    
    # Temporal
    onset_date = Column(DateTime, nullable=True)
    resolution_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    is_chronic = Column(Boolean, default=False)
    
    # Source
    diagnosed_by = Column(String(200), nullable=True)
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "condition_name": self.condition_name,
            "icd_code": self.icd_code,
            "onset_date": self.onset_date.isoformat() if self.onset_date else None,
            "is_active": self.is_active,
            "is_chronic": self.is_chronic
        }


class PatientSymptomHistory(Base):
    """Symptom tracking history"""
    __tablename__ = "patient_symptom_history"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    
    # Symptom details
    symptom_name = Column(String(200), nullable=False)
    severity = Column(String(50), nullable=True)  # mild, moderate, severe
    frequency = Column(String(100), nullable=True)
    
    # Temporal
    reported_date = Column(DateTime, nullable=True)
    resolution_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Association
    associated_condition_id = Column(Integer, ForeignKey("patient_condition_history.id"), nullable=True)
    associated_medication_id = Column(Integer, ForeignKey("patient_medication_history.id"), nullable=True)
    
    # Source
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "symptom_name": self.symptom_name,
            "severity": self.severity,
            "reported_date": self.reported_date.isoformat() if self.reported_date else None,
            "is_active": self.is_active
        }


class VisitSummary(Base):
    """Visit-wise summary preserving historical state"""
    __tablename__ = "visit_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    
    # Visit info
    visit_date = Column(DateTime, nullable=False)
    visit_type = Column(String(50), nullable=True)  # regular, emergency, follow-up
    doctor_name = Column(String(200), nullable=True)
    clinic_name = Column(String(200), nullable=True)
    
    # Snapshot of patient state at visit time
    chief_complaints = Column(JSON, default=list)
    diagnoses = Column(JSON, default=list)
    medications_prescribed = Column(JSON, default=list)
    medications_continued = Column(JSON, default=list)
    medications_stopped = Column(JSON, default=list)
    vitals = Column(JSON, default=dict)
    investigations_ordered = Column(JSON, default=list)
    advice = Column(JSON, default=list)
    follow_up = Column(String(200), nullable=True)
    
    # Full summary
    summary_text = Column(Text, nullable=True)
    
    # Source
    source_document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "visit_date": self.visit_date.isoformat() if self.visit_date else None,
            "visit_type": self.visit_type,
            "doctor_name": self.doctor_name,
            "chief_complaints": self.chief_complaints,
            "diagnoses": self.diagnoses,
            "medications_prescribed": self.medications_prescribed,
            "summary_text": self.summary_text
        }
