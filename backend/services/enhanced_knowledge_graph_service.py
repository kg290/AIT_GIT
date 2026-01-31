"""
Enhanced Knowledge Graph Service - SQLAlchemy-based implementation
Provides patient-medication-condition relationship management
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from backend.models.knowledge_graph import (
    KnowledgeNode, KnowledgeEdge, NodeType, RelationshipType
)

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """
    Knowledge Graph Service using SQLAlchemy for persistence
    
    Features:
    - Patient ↔ Medication links
    - Medication ↔ Condition links
    - Condition ↔ Symptom links
    - Temporal relationships
    - Graph traversal queries
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== Node Operations ====================
    
    def create_or_get_node(
        self,
        node_type: NodeType,
        name: str,
        external_id: str = None,
        properties: Dict = None
    ) -> KnowledgeNode:
        """Create a new node or get existing one"""
        # Try to find existing node
        query = self.db.query(KnowledgeNode).filter(
            KnowledgeNode.node_type == node_type,
            KnowledgeNode.name == name
        )
        
        if external_id:
            query = query.filter(KnowledgeNode.external_id == external_id)
        
        existing = query.first()
        
        if existing:
            # Update properties if provided
            if properties:
                existing.properties = {**(existing.properties or {}), **properties}
                existing.updated_at = datetime.utcnow()
                self.db.commit()
            return existing
        
        # Create new node
        node = KnowledgeNode(
            node_type=node_type,
            name=name,
            external_id=external_id,
            properties=properties or {}
        )
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        
        logger.info(f"Created node: {node_type.value} - {name}")
        return node
    
    def get_node_by_id(self, node_id: int) -> Optional[KnowledgeNode]:
        """Get node by ID"""
        return self.db.query(KnowledgeNode).filter(KnowledgeNode.id == node_id).first()
    
    def get_node_by_external_id(self, node_type: NodeType, external_id: str) -> Optional[KnowledgeNode]:
        """Get node by external ID"""
        return self.db.query(KnowledgeNode).filter(
            and_(
                KnowledgeNode.node_type == node_type,
                KnowledgeNode.external_id == external_id
            )
        ).first()
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Get all nodes of a type"""
        return self.db.query(KnowledgeNode).filter(
            KnowledgeNode.node_type == node_type
        ).all()
    
    # ==================== Relationship Operations ====================
    
    def create_relationship(
        self,
        source_node_id: int,
        target_node_id: int,
        relationship_type: RelationshipType,
        properties: Dict = None,
        start_date: datetime = None,
        end_date: datetime = None,
        weight: float = 1.0,
        confidence: float = 1.0
    ) -> KnowledgeEdge:
        """Create a relationship between two nodes"""
        # Check if relationship already exists
        existing = self.db.query(KnowledgeEdge).filter(
            and_(
                KnowledgeEdge.source_node_id == source_node_id,
                KnowledgeEdge.target_node_id == target_node_id,
                KnowledgeEdge.relationship_type == relationship_type
            )
        ).first()
        
        if existing:
            # Update existing relationship
            if properties:
                existing.properties = {**(existing.properties or {}), **properties}
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing
        
        # Create new edge
        edge = KnowledgeEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            properties=properties or {},
            start_date=start_date,
            end_date=end_date,
            is_current=end_date is None,
            weight=weight,
            confidence=confidence
        )
        self.db.add(edge)
        self.db.commit()
        self.db.refresh(edge)
        
        logger.info(f"Created edge: {source_node_id} --{relationship_type.value}--> {target_node_id}")
        return edge
    
    # ==================== Convenience Methods ====================
    
    def link_patient_medication(
        self,
        patient_id: int,
        medication_name: str,
        relationship_type: str = "is_taking"
    ) -> Optional[KnowledgeEdge]:
        """Link a patient to a medication"""
        # Get or create patient node
        patient_node = self.get_node_by_external_id(NodeType.PATIENT, str(patient_id))
        if not patient_node:
            patient_node = self.create_or_get_node(
                node_type=NodeType.PATIENT,
                name=f"Patient_{patient_id}",
                external_id=str(patient_id)
            )
        
        # Get or create medication node
        med_node = self.create_or_get_node(
            node_type=NodeType.MEDICATION,
            name=medication_name
        )
        
        # Create relationship
        return self.create_relationship(
            source_node_id=patient_node.id,
            target_node_id=med_node.id,
            relationship_type=RelationshipType.PATIENT_HAS_MEDICATION,
            properties={"relationship_subtype": relationship_type}
        )
    
    def link_patient_condition(
        self,
        patient_id: int,
        condition_name: str
    ) -> Optional[KnowledgeEdge]:
        """Link a patient to a condition"""
        # Get or create patient node
        patient_node = self.get_node_by_external_id(NodeType.PATIENT, str(patient_id))
        if not patient_node:
            patient_node = self.create_or_get_node(
                node_type=NodeType.PATIENT,
                name=f"Patient_{patient_id}",
                external_id=str(patient_id)
            )
        
        # Get or create condition node
        condition_node = self.create_or_get_node(
            node_type=NodeType.CONDITION,
            name=condition_name
        )
        
        # Create relationship
        return self.create_relationship(
            source_node_id=patient_node.id,
            target_node_id=condition_node.id,
            relationship_type=RelationshipType.PATIENT_HAS_CONDITION
        )
    
    # ==================== Query Operations ====================
    
    def get_patient_graph(self, patient_node_id: int, max_depth: int = 2) -> Dict:
        """Get complete knowledge graph for a patient"""
        nodes = []
        edges = []
        visited_nodes = set()
        
        def traverse(node_id: int, depth: int):
            if depth > max_depth or node_id in visited_nodes:
                return
            visited_nodes.add(node_id)
            
            node = self.get_node_by_id(node_id)
            if node:
                nodes.append(node.to_dict())
                
                # Get outgoing edges
                for edge in node.outgoing_edges:
                    edges.append(edge.to_dict())
                    traverse(edge.target_node_id, depth + 1)
                
                # Get incoming edges
                for edge in node.incoming_edges:
                    edges.append(edge.to_dict())
                    traverse(edge.source_node_id, depth + 1)
        
        traverse(patient_node_id, 0)
        
        # Remove duplicate edges
        seen_edges = set()
        unique_edges = []
        for edge in edges:
            edge_key = edge['id']
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": unique_edges,
            "patient_node_id": patient_node_id
        }
    
    def get_patient_medications(self, patient_id: int, current_only: bool = False) -> List[Dict]:
        """Get all medications for a patient"""
        patient_node = self.get_node_by_external_id(NodeType.PATIENT, str(patient_id))
        if not patient_node:
            return []
        
        query = self.db.query(KnowledgeEdge).filter(
            and_(
                KnowledgeEdge.source_node_id == patient_node.id,
                KnowledgeEdge.relationship_type == RelationshipType.PATIENT_HAS_MEDICATION
            )
        )
        
        if current_only:
            query = query.filter(KnowledgeEdge.is_current == True)
        
        medications = []
        for edge in query.all():
            med_node = self.get_node_by_id(edge.target_node_id)
            if med_node:
                medications.append({
                    "medication": med_node.name,
                    "properties": edge.properties,
                    "start_date": edge.start_date.isoformat() if edge.start_date else None,
                    "end_date": edge.end_date.isoformat() if edge.end_date else None,
                    "is_current": edge.is_current
                })
        
        return medications
    
    def get_patient_conditions(self, patient_id: int) -> List[Dict]:
        """Get all conditions for a patient"""
        patient_node = self.get_node_by_external_id(NodeType.PATIENT, str(patient_id))
        if not patient_node:
            return []
        
        conditions = []
        edges = self.db.query(KnowledgeEdge).filter(
            and_(
                KnowledgeEdge.source_node_id == patient_node.id,
                KnowledgeEdge.relationship_type == RelationshipType.PATIENT_HAS_CONDITION
            )
        ).all()
        
        for edge in edges:
            condition_node = self.get_node_by_id(edge.target_node_id)
            if condition_node:
                conditions.append({
                    "condition": condition_node.name,
                    "properties": edge.properties,
                    "diagnosis_date": edge.start_date.isoformat() if edge.start_date else None,
                    "is_current": edge.is_current
                })
        
        return conditions
    
    def get_graph_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        total_nodes = self.db.query(KnowledgeNode).count()
        total_edges = self.db.query(KnowledgeEdge).count()
        
        # Count by node type
        nodes_by_type = {}
        for node_type in NodeType:
            count = self.db.query(KnowledgeNode).filter(
                KnowledgeNode.node_type == node_type
            ).count()
            if count > 0:
                nodes_by_type[node_type.value] = count
        
        # Count by relationship type
        edges_by_type = {}
        for rel_type in RelationshipType:
            count = self.db.query(KnowledgeEdge).filter(
                KnowledgeEdge.relationship_type == rel_type
            ).count()
            if count > 0:
                edges_by_type[rel_type.value] = count
        
        return {
            "total_nodes": total_nodes,
            "total_relationships": total_edges,
            "nodes_by_type": nodes_by_type,
            "relationships_by_type": edges_by_type
        }
    
    def find_related_entities(
        self,
        node_id: int,
        relationship_type: RelationshipType = None,
        max_depth: int = 2
    ) -> List[Dict]:
        """Find entities related to a given node"""
        related = []
        visited = set()
        
        def traverse(current_id: int, depth: int):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)
            
            # Get outgoing edges
            query = self.db.query(KnowledgeEdge).filter(
                KnowledgeEdge.source_node_id == current_id
            )
            if relationship_type:
                query = query.filter(KnowledgeEdge.relationship_type == relationship_type)
            
            for edge in query.all():
                node = self.get_node_by_id(edge.target_node_id)
                if node and node.id not in visited:
                    related.append({
                        "node_id": node.id,
                        "type": node.node_type.value,
                        "name": node.name,
                        "depth": depth,
                        "relationship": edge.relationship_type.value
                    })
                    traverse(node.id, depth + 1)
            
            # Get incoming edges
            query = self.db.query(KnowledgeEdge).filter(
                KnowledgeEdge.target_node_id == current_id
            )
            if relationship_type:
                query = query.filter(KnowledgeEdge.relationship_type == relationship_type)
            
            for edge in query.all():
                node = self.get_node_by_id(edge.source_node_id)
                if node and node.id not in visited:
                    related.append({
                        "node_id": node.id,
                        "type": node.node_type.value,
                        "name": node.name,
                        "depth": depth,
                        "relationship": edge.relationship_type.value
                    })
                    traverse(node.id, depth + 1)
        
        traverse(node_id, 1)
        return related
    
    def update_relationship_status(
        self,
        edge_id: int,
        end_date: datetime = None,
        is_current: bool = False
    ):
        """Update relationship status (e.g., mark medication as stopped)"""
        edge = self.db.query(KnowledgeEdge).filter(KnowledgeEdge.id == edge_id).first()
        if edge:
            edge.end_date = end_date or datetime.utcnow()
            edge.is_current = is_current
            edge.updated_at = datetime.utcnow()
            self.db.commit()
