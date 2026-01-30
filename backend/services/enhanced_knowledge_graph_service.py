"""
Enhanced Knowledge Graph Service - Relationship management
Links patients, medications, conditions, symptoms across time
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from backend.models.knowledge_graph import (
    KnowledgeNode, KnowledgeEdge, NodeType, RelationshipType
)

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """
    Knowledge Graph Construction and Management
    
    Features:
    - Link patient ↔ medications
    - Link medications ↔ conditions
    - Link conditions ↔ symptoms
    - Link events across time
    - Update relationships as new data arrives
    - Preserve historical relationships
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== Node Management ====================
    
    def create_or_get_node(
        self,
        node_type: NodeType,
        name: str,
        external_id: str = None,
        properties: Dict = None,
        source_document_id: int = None,
        source_text: str = None,
        confidence: float = 1.0
    ) -> KnowledgeNode:
        """Create a new node or return existing one"""
        
        normalized = name.lower().strip()
        
        # Check for existing node
        existing = self.db.query(KnowledgeNode).filter(
            and_(
                KnowledgeNode.node_type == node_type,
                KnowledgeNode.normalized_name == normalized
            )
        ).first()
        
        if existing:
            # Update properties if needed
            if properties:
                existing.properties = {**existing.properties, **properties}
                existing.updated_at = datetime.utcnow()
                self.db.commit()
            return existing
        
        # Create new node
        node = KnowledgeNode(
            node_type=node_type,
            name=name,
            normalized_name=normalized,
            external_id=external_id,
            properties=properties or {},
            source_document_id=source_document_id,
            source_text=source_text,
            extraction_confidence=confidence
        )
        
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        
        logger.info(f"Created node: {node_type.value} - {name}")
        return node
    
    def get_node(self, node_id: int) -> Optional[KnowledgeNode]:
        """Get a node by ID"""
        return self.db.query(KnowledgeNode).filter(KnowledgeNode.id == node_id).first()
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Get all nodes of a specific type"""
        return self.db.query(KnowledgeNode).filter(
            and_(
                KnowledgeNode.node_type == node_type,
                KnowledgeNode.is_active == True
            )
        ).all()
    
    def search_nodes(self, query: str, node_type: NodeType = None) -> List[KnowledgeNode]:
        """Search for nodes by name"""
        q = self.db.query(KnowledgeNode).filter(
            KnowledgeNode.name.ilike(f"%{query}%")
        )
        
        if node_type:
            q = q.filter(KnowledgeNode.node_type == node_type)
        
        return q.limit(20).all()
    
    # ==================== Edge/Relationship Management ====================
    
    def create_relationship(
        self,
        source_node_id: int,
        target_node_id: int,
        relationship_type: RelationshipType,
        properties: Dict = None,
        confidence: float = 1.0,
        evidence_text: str = None,
        source_document_id: int = None,
        valid_from: datetime = None,
        valid_until: datetime = None
    ) -> KnowledgeEdge:
        """Create a relationship between two nodes"""
        
        # Check if relationship already exists
        existing = self.db.query(KnowledgeEdge).filter(
            and_(
                KnowledgeEdge.source_node_id == source_node_id,
                KnowledgeEdge.target_node_id == target_node_id,
                KnowledgeEdge.relationship_type == relationship_type,
                KnowledgeEdge.is_active == True
            )
        ).first()
        
        if existing:
            # Update existing relationship
            existing.properties = {**existing.properties, **(properties or {})}
            existing.confidence = max(existing.confidence, confidence)
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing
        
        edge = KnowledgeEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            properties=properties or {},
            confidence=confidence,
            evidence_text=evidence_text,
            source_document_id=source_document_id,
            valid_from=valid_from or datetime.utcnow(),
            valid_until=valid_until
        )
        
        self.db.add(edge)
        self.db.commit()
        self.db.refresh(edge)
        
        logger.info(f"Created relationship: {source_node_id} --[{relationship_type.value}]--> {target_node_id}")
        return edge
    
    def end_relationship(
        self,
        edge_id: int,
        end_date: datetime = None
    ) -> Optional[KnowledgeEdge]:
        """Mark a relationship as ended (historical preservation)"""
        
        edge = self.db.query(KnowledgeEdge).filter(KnowledgeEdge.id == edge_id).first()
        
        if edge:
            edge.valid_until = end_date or datetime.utcnow()
            edge.is_active = False
            edge.updated_at = datetime.utcnow()
            self.db.commit()
            return edge
        
        return None
    
    def get_relationships(
        self,
        node_id: int,
        direction: str = "both",
        relationship_type: RelationshipType = None,
        include_inactive: bool = False
    ) -> List[KnowledgeEdge]:
        """Get relationships for a node"""
        
        if direction == "outgoing":
            query = self.db.query(KnowledgeEdge).filter(
                KnowledgeEdge.source_node_id == node_id
            )
        elif direction == "incoming":
            query = self.db.query(KnowledgeEdge).filter(
                KnowledgeEdge.target_node_id == node_id
            )
        else:  # both
            query = self.db.query(KnowledgeEdge).filter(
                or_(
                    KnowledgeEdge.source_node_id == node_id,
                    KnowledgeEdge.target_node_id == node_id
                )
            )
        
        if relationship_type:
            query = query.filter(KnowledgeEdge.relationship_type == relationship_type)
        
        if not include_inactive:
            query = query.filter(KnowledgeEdge.is_active == True)
        
        return query.all()
    
    # ==================== High-Level Operations ====================
    
    def link_patient_medication(
        self,
        patient_node_id: int,
        medication_name: str,
        start_date: datetime = None,
        end_date: datetime = None,
        dosage: str = None,
        frequency: str = None,
        source_document_id: int = None
    ) -> Tuple[KnowledgeNode, KnowledgeEdge]:
        """Link a patient to a medication"""
        
        # Create or get medication node
        med_node = self.create_or_get_node(
            node_type=NodeType.MEDICATION,
            name=medication_name,
            properties={"dosage": dosage, "frequency": frequency},
            source_document_id=source_document_id
        )
        
        # Create relationship
        edge = self.create_relationship(
            source_node_id=patient_node_id,
            target_node_id=med_node.id,
            relationship_type=RelationshipType.PATIENT_HAS_MEDICATION,
            properties={"dosage": dosage, "frequency": frequency},
            valid_from=start_date,
            valid_until=end_date,
            source_document_id=source_document_id
        )
        
        return med_node, edge
    
    def link_patient_condition(
        self,
        patient_node_id: int,
        condition_name: str,
        onset_date: datetime = None,
        resolution_date: datetime = None,
        source_document_id: int = None
    ) -> Tuple[KnowledgeNode, KnowledgeEdge]:
        """Link a patient to a condition/diagnosis"""
        
        condition_node = self.create_or_get_node(
            node_type=NodeType.CONDITION,
            name=condition_name,
            source_document_id=source_document_id
        )
        
        edge = self.create_relationship(
            source_node_id=patient_node_id,
            target_node_id=condition_node.id,
            relationship_type=RelationshipType.PATIENT_HAS_CONDITION,
            valid_from=onset_date,
            valid_until=resolution_date,
            source_document_id=source_document_id
        )
        
        return condition_node, edge
    
    def link_medication_condition(
        self,
        medication_name: str,
        condition_name: str,
        relationship: str = "treats"  # treats, causes, etc.
    ) -> Tuple[KnowledgeNode, KnowledgeNode, KnowledgeEdge]:
        """Link a medication to a condition"""
        
        med_node = self.create_or_get_node(NodeType.MEDICATION, medication_name)
        condition_node = self.create_or_get_node(NodeType.CONDITION, condition_name)
        
        rel_type = RelationshipType.MEDICATION_TREATS_CONDITION if relationship == "treats" else RelationshipType.MEDICATION_CONTRAINDICATED_FOR
        
        edge = self.create_relationship(
            source_node_id=med_node.id,
            target_node_id=condition_node.id,
            relationship_type=rel_type
        )
        
        return med_node, condition_node, edge
    
    def link_condition_symptom(
        self,
        condition_name: str,
        symptom_name: str,
        source_document_id: int = None
    ) -> Tuple[KnowledgeNode, KnowledgeNode, KnowledgeEdge]:
        """Link a condition to a symptom"""
        
        condition_node = self.create_or_get_node(NodeType.CONDITION, condition_name)
        symptom_node = self.create_or_get_node(NodeType.SYMPTOM, symptom_name)
        
        edge = self.create_relationship(
            source_node_id=condition_node.id,
            target_node_id=symptom_node.id,
            relationship_type=RelationshipType.CONDITION_HAS_SYMPTOM,
            source_document_id=source_document_id
        )
        
        return condition_node, symptom_node, edge
    
    def link_drug_interaction(
        self,
        drug1_name: str,
        drug2_name: str,
        severity: str = None,
        description: str = None
    ) -> Tuple[KnowledgeNode, KnowledgeNode, KnowledgeEdge]:
        """Link two drugs that interact"""
        
        drug1_node = self.create_or_get_node(NodeType.MEDICATION, drug1_name)
        drug2_node = self.create_or_get_node(NodeType.MEDICATION, drug2_name)
        
        edge = self.create_relationship(
            source_node_id=drug1_node.id,
            target_node_id=drug2_node.id,
            relationship_type=RelationshipType.MEDICATION_INTERACTS_WITH,
            properties={"severity": severity, "description": description}
        )
        
        return drug1_node, drug2_node, edge
    
    # ==================== Graph Queries ====================
    
    def get_patient_graph(self, patient_node_id: int) -> Dict[str, Any]:
        """Get complete graph data for a patient"""
        
        patient_node = self.get_node(patient_node_id)
        if not patient_node:
            return {"error": "Patient not found"}
        
        relationships = self.get_relationships(patient_node_id, include_inactive=True)
        
        nodes = {patient_node_id: patient_node.to_dict()}
        edges = []
        
        for edge in relationships:
            edges.append(edge.to_dict())
            
            # Add connected nodes
            if edge.source_node_id not in nodes:
                node = self.get_node(edge.source_node_id)
                if node:
                    nodes[edge.source_node_id] = node.to_dict()
            
            if edge.target_node_id not in nodes:
                node = self.get_node(edge.target_node_id)
                if node:
                    nodes[edge.target_node_id] = node.to_dict()
        
        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }
    
    def get_medication_relationships(self, medication_name: str) -> Dict[str, Any]:
        """Get all relationships for a medication"""
        
        med_node = self.db.query(KnowledgeNode).filter(
            and_(
                KnowledgeNode.node_type == NodeType.MEDICATION,
                KnowledgeNode.normalized_name == medication_name.lower().strip()
            )
        ).first()
        
        if not med_node:
            return {"error": "Medication not found", "relationships": []}
        
        relationships = self.get_relationships(med_node.id, include_inactive=True)
        
        result = {
            "medication": med_node.to_dict(),
            "patients": [],
            "conditions_treated": [],
            "interactions": [],
            "contraindications": []
        }
        
        for edge in relationships:
            other_node_id = edge.target_node_id if edge.source_node_id == med_node.id else edge.source_node_id
            other_node = self.get_node(other_node_id)
            
            if not other_node:
                continue
            
            if edge.relationship_type == RelationshipType.PATIENT_HAS_MEDICATION:
                result["patients"].append(other_node.to_dict())
            elif edge.relationship_type == RelationshipType.MEDICATION_TREATS_CONDITION:
                result["conditions_treated"].append(other_node.to_dict())
            elif edge.relationship_type == RelationshipType.MEDICATION_INTERACTS_WITH:
                result["interactions"].append({
                    "drug": other_node.to_dict(),
                    "severity": edge.properties.get("severity"),
                    "description": edge.properties.get("description")
                })
            elif edge.relationship_type == RelationshipType.MEDICATION_CONTRAINDICATED_FOR:
                result["contraindications"].append(other_node.to_dict())
        
        return result
    
    def find_path(
        self,
        start_node_id: int,
        end_node_id: int,
        max_depth: int = 5
    ) -> List[Dict]:
        """Find path between two nodes (BFS)"""
        
        if start_node_id == end_node_id:
            return [{"node_id": start_node_id}]
        
        visited = {start_node_id}
        queue = [(start_node_id, [start_node_id])]
        
        while queue and len(visited) < 1000:
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            relationships = self.get_relationships(current_id)
            
            for edge in relationships:
                next_id = edge.target_node_id if edge.source_node_id == current_id else edge.source_node_id
                
                if next_id == end_node_id:
                    return path + [next_id]
                
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))
        
        return []  # No path found
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        
        node_counts = {}
        for node_type in NodeType:
            count = self.db.query(KnowledgeNode).filter(
                KnowledgeNode.node_type == node_type
            ).count()
            node_counts[node_type.value] = count
        
        edge_counts = {}
        for rel_type in RelationshipType:
            count = self.db.query(KnowledgeEdge).filter(
                KnowledgeEdge.relationship_type == rel_type
            ).count()
            edge_counts[rel_type.value] = count
        
        total_nodes = self.db.query(KnowledgeNode).count()
        total_edges = self.db.query(KnowledgeEdge).count()
        active_edges = self.db.query(KnowledgeEdge).filter(KnowledgeEdge.is_active == True).count()
        
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "active_edges": active_edges,
            "historical_edges": total_edges - active_edges,
            "nodes_by_type": node_counts,
            "edges_by_type": edge_counts
        }
