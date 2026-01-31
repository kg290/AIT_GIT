"""
Knowledge Graph Models - SQLAlchemy models for graph storage
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class NodeType(enum.Enum):
    """Types of nodes in the knowledge graph"""
    PATIENT = "patient"
    MEDICATION = "medication"
    CONDITION = "condition"
    SYMPTOM = "symptom"
    PRESCRIPTION = "prescription"
    DOCUMENT = "document"
    DOCTOR = "doctor"
    LAB_TEST = "lab_test"


class RelationshipType(enum.Enum):
    """Types of relationships in the knowledge graph"""
    PATIENT_HAS_MEDICATION = "patient_has_medication"
    PATIENT_HAS_CONDITION = "patient_has_condition"
    PATIENT_HAS_SYMPTOM = "patient_has_symptom"
    PATIENT_VISITS_DOCTOR = "patient_visits_doctor"
    MEDICATION_TREATS = "medication_treats"
    MEDICATION_INTERACTS_WITH = "medication_interacts_with"
    CONDITION_CAUSES_SYMPTOM = "condition_causes_symptom"
    PRESCRIPTION_CONTAINS = "prescription_contains"
    EXTRACTED_FROM = "extracted_from"
    PRESCRIBED_BY = "prescribed_by"


class KnowledgeNode(Base):
    """Node in the knowledge graph"""
    __tablename__ = "knowledge_nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Node identification
    node_type = Column(SQLEnum(NodeType), nullable=False, index=True)
    name = Column(String(300), nullable=False, index=True)
    external_id = Column(String(100), nullable=True, index=True)  # Links to actual entity ID
    
    # Node properties (flexible JSON storage)
    properties = Column(JSON, default=dict)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    outgoing_edges = relationship(
        "KnowledgeEdge",
        foreign_keys="KnowledgeEdge.source_node_id",
        back_populates="source_node",
        cascade="all, delete-orphan"
    )
    incoming_edges = relationship(
        "KnowledgeEdge",
        foreign_keys="KnowledgeEdge.target_node_id",
        back_populates="target_node",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<KnowledgeNode {self.node_type.value}: {self.name}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "external_id": self.external_id,
            "properties": self.properties,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class KnowledgeEdge(Base):
    """Edge/Relationship in the knowledge graph"""
    __tablename__ = "knowledge_edges"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Source and target nodes
    source_node_id = Column(Integer, ForeignKey("knowledge_nodes.id"), nullable=False, index=True)
    target_node_id = Column(Integer, ForeignKey("knowledge_nodes.id"), nullable=False, index=True)
    
    # Relationship type
    relationship_type = Column(SQLEnum(RelationshipType), nullable=False, index=True)
    
    # Edge properties (flexible JSON storage)
    properties = Column(JSON, default=dict)
    
    # Temporal information
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    is_current = Column(Boolean, default=True)
    
    # Confidence/weight
    weight = Column(Float, default=1.0)
    confidence = Column(Float, default=1.0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_node = relationship(
        "KnowledgeNode",
        foreign_keys=[source_node_id],
        back_populates="outgoing_edges"
    )
    target_node = relationship(
        "KnowledgeNode",
        foreign_keys=[target_node_id],
        back_populates="incoming_edges"
    )
    
    def __repr__(self):
        return f"<KnowledgeEdge {self.source_node_id} --{self.relationship_type.value}--> {self.target_node_id}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relationship_type": self.relationship_type.value,
            "properties": self.properties,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_current": self.is_current,
            "weight": self.weight,
            "confidence": self.confidence
        }
