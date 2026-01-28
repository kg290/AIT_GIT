"""
Knowledge Graph Service - Patient-medication-condition relationships
Can use Neo4j for production or in-memory for development
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date
from collections import defaultdict

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class KGNode:
    """Knowledge graph node"""
    node_id: str
    node_type: str  # patient, medication, condition, symptom, prescription, document
    properties: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class KGRelationship:
    """Knowledge graph relationship"""
    rel_id: str
    source_id: str
    target_id: str
    rel_type: str  # takes, has_condition, treats, causes, prescribed_in, extracted_from
    properties: Dict[str, Any]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_current: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


class KnowledgeGraphService:
    """
    Knowledge Graph Service for medical relationships
    
    Features:
    - Patient ↔ Medication links
    - Medication ↔ Condition links
    - Condition ↔ Symptom links
    - Temporal relationships
    - Historical relationship preservation
    """
    
    def __init__(self, use_neo4j: bool = None):
        self.use_neo4j = use_neo4j if use_neo4j is not None else settings.USE_NEO4J
        
        if self.use_neo4j:
            self._init_neo4j()
        else:
            self._init_memory_graph()
    
    def _init_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Using in-memory graph.")
            self.use_neo4j = False
            self._init_memory_graph()
    
    def _init_memory_graph(self):
        """Initialize in-memory graph structure"""
        self.nodes: Dict[str, KGNode] = {}
        self.relationships: Dict[str, KGRelationship] = {}
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # type -> node_ids
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # node_id -> related_node_ids
        self.rel_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (source, type) -> rel_ids
    
    # ==================== Node Operations ====================
    
    def create_patient_node(self, patient_id: str, properties: Dict) -> str:
        """Create or update patient node"""
        node_id = f"patient_{patient_id}"
        return self._create_node(node_id, 'patient', properties)
    
    def create_medication_node(self, medication_name: str, properties: Dict = None) -> str:
        """Create or update medication node"""
        node_id = f"medication_{medication_name.lower().replace(' ', '_')}"
        props = properties or {}
        props['name'] = medication_name
        return self._create_node(node_id, 'medication', props)
    
    def create_condition_node(self, condition_name: str, properties: Dict = None) -> str:
        """Create or update condition node"""
        node_id = f"condition_{condition_name.lower().replace(' ', '_')}"
        props = properties or {}
        props['name'] = condition_name
        return self._create_node(node_id, 'condition', props)
    
    def create_symptom_node(self, symptom_name: str, properties: Dict = None) -> str:
        """Create or update symptom node"""
        node_id = f"symptom_{symptom_name.lower().replace(' ', '_')}"
        props = properties or {}
        props['name'] = symptom_name
        return self._create_node(node_id, 'symptom', props)
    
    def create_document_node(self, document_id: str, properties: Dict = None) -> str:
        """Create document node"""
        node_id = f"document_{document_id}"
        return self._create_node(node_id, 'document', properties or {})
    
    def create_prescription_node(self, prescription_id: str, properties: Dict = None) -> str:
        """Create prescription node"""
        node_id = f"prescription_{prescription_id}"
        return self._create_node(node_id, 'prescription', properties or {})
    
    def _create_node(self, node_id: str, node_type: str, properties: Dict) -> str:
        """Create or update a node"""
        if self.use_neo4j:
            return self._neo4j_create_node(node_id, node_type, properties)
        
        if node_id in self.nodes:
            # Update existing node
            self.nodes[node_id].properties.update(properties)
        else:
            # Create new node
            self.nodes[node_id] = KGNode(
                node_id=node_id,
                node_type=node_type,
                properties=properties
            )
            self.node_index[node_type].add(node_id)
        
        return node_id
    
    def _neo4j_create_node(self, node_id: str, node_type: str, properties: Dict) -> str:
        """Create node in Neo4j"""
        with self.driver.session() as session:
            query = f"""
            MERGE (n:{node_type.capitalize()} {{node_id: $node_id}})
            SET n += $properties
            RETURN n.node_id
            """
            result = session.run(query, node_id=node_id, properties=properties)
            return result.single()[0]
    
    # ==================== Relationship Operations ====================
    
    def link_patient_medication(self, patient_id: str, medication_name: str,
                                properties: Dict = None,
                                start_date: date = None,
                                end_date: date = None) -> str:
        """Link patient to medication (TAKES relationship)"""
        patient_node = f"patient_{patient_id}"
        med_node = self.create_medication_node(medication_name)
        
        rel_id = f"{patient_node}_takes_{med_node}_{start_date or 'current'}"
        
        return self._create_relationship(
            rel_id=rel_id,
            source_id=patient_node,
            target_id=med_node,
            rel_type='TAKES',
            properties=properties or {},
            start_date=start_date,
            end_date=end_date
        )
    
    def link_patient_condition(self, patient_id: str, condition_name: str,
                               properties: Dict = None,
                               diagnosis_date: date = None) -> str:
        """Link patient to condition (HAS_CONDITION relationship)"""
        patient_node = f"patient_{patient_id}"
        condition_node = self.create_condition_node(condition_name)
        
        rel_id = f"{patient_node}_has_{condition_node}"
        
        return self._create_relationship(
            rel_id=rel_id,
            source_id=patient_node,
            target_id=condition_node,
            rel_type='HAS_CONDITION',
            properties=properties or {},
            start_date=diagnosis_date
        )
    
    def link_medication_condition(self, medication_name: str, condition_name: str,
                                  rel_type: str = 'TREATS',
                                  properties: Dict = None) -> str:
        """Link medication to condition (TREATS or INDICATED_FOR)"""
        med_node = self.create_medication_node(medication_name)
        condition_node = self.create_condition_node(condition_name)
        
        rel_id = f"{med_node}_{rel_type.lower()}_{condition_node}"
        
        return self._create_relationship(
            rel_id=rel_id,
            source_id=med_node,
            target_id=condition_node,
            rel_type=rel_type,
            properties=properties or {}
        )
    
    def link_condition_symptom(self, condition_name: str, symptom_name: str,
                               properties: Dict = None) -> str:
        """Link condition to symptom (CAUSES or PRESENTS_WITH)"""
        condition_node = self.create_condition_node(condition_name)
        symptom_node = self.create_symptom_node(symptom_name)
        
        rel_id = f"{condition_node}_causes_{symptom_node}"
        
        return self._create_relationship(
            rel_id=rel_id,
            source_id=condition_node,
            target_id=symptom_node,
            rel_type='CAUSES',
            properties=properties or {}
        )
    
    def link_prescription_document(self, prescription_id: str, document_id: str,
                                   properties: Dict = None) -> str:
        """Link prescription to source document"""
        rx_node = f"prescription_{prescription_id}"
        doc_node = f"document_{document_id}"
        
        return self._create_relationship(
            rel_id=f"{rx_node}_extracted_from_{doc_node}",
            source_id=rx_node,
            target_id=doc_node,
            rel_type='EXTRACTED_FROM',
            properties=properties or {}
        )
    
    def link_patient_symptom(self, patient_id: str, symptom_name: str,
                             properties: Dict = None,
                             onset_date: date = None,
                             resolution_date: date = None) -> str:
        """Link patient to symptom"""
        patient_node = f"patient_{patient_id}"
        symptom_node = self.create_symptom_node(symptom_name)
        
        rel_id = f"{patient_node}_experiences_{symptom_node}_{onset_date or 'current'}"
        
        return self._create_relationship(
            rel_id=rel_id,
            source_id=patient_node,
            target_id=symptom_node,
            rel_type='EXPERIENCES',
            properties=properties or {},
            start_date=onset_date,
            end_date=resolution_date
        )
    
    def _create_relationship(self, rel_id: str, source_id: str, target_id: str,
                            rel_type: str, properties: Dict,
                            start_date: date = None, end_date: date = None) -> str:
        """Create or update a relationship"""
        if self.use_neo4j:
            return self._neo4j_create_relationship(
                rel_id, source_id, target_id, rel_type, properties, start_date, end_date
            )
        
        is_current = end_date is None or end_date >= date.today()
        
        self.relationships[rel_id] = KGRelationship(
            rel_id=rel_id,
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            properties=properties,
            start_date=start_date,
            end_date=end_date,
            is_current=is_current
        )
        
        # Update indices
        self.adjacency[source_id].add(target_id)
        self.adjacency[target_id].add(source_id)
        self.rel_index[(source_id, rel_type)].add(rel_id)
        
        return rel_id
    
    def _neo4j_create_relationship(self, rel_id: str, source_id: str, target_id: str,
                                   rel_type: str, properties: Dict,
                                   start_date: date, end_date: date) -> str:
        """Create relationship in Neo4j"""
        with self.driver.session() as session:
            props = {**properties}
            if start_date:
                props['start_date'] = str(start_date)
            if end_date:
                props['end_date'] = str(end_date)
            props['is_current'] = end_date is None or end_date >= date.today()
            
            query = f"""
            MATCH (a {{node_id: $source_id}})
            MATCH (b {{node_id: $target_id}})
            MERGE (a)-[r:{rel_type} {{rel_id: $rel_id}}]->(b)
            SET r += $properties
            RETURN r.rel_id
            """
            result = session.run(
                query, 
                source_id=source_id, 
                target_id=target_id,
                rel_id=rel_id,
                properties=props
            )
            return result.single()[0]
    
    # ==================== Query Operations ====================
    
    def get_patient_medications(self, patient_id: str, 
                                current_only: bool = False) -> List[Dict]:
        """Get all medications for a patient"""
        patient_node = f"patient_{patient_id}"
        
        if self.use_neo4j:
            return self._neo4j_query_medications(patient_node, current_only)
        
        medications = []
        for rel_id in self.rel_index.get((patient_node, 'TAKES'), set()):
            rel = self.relationships[rel_id]
            if current_only and not rel.is_current:
                continue
            
            med_node = self.nodes.get(rel.target_id)
            if med_node:
                medications.append({
                    'medication': med_node.properties.get('name', rel.target_id),
                    'properties': rel.properties,
                    'start_date': str(rel.start_date) if rel.start_date else None,
                    'end_date': str(rel.end_date) if rel.end_date else None,
                    'is_current': rel.is_current
                })
        
        return medications
    
    def get_patient_conditions(self, patient_id: str) -> List[Dict]:
        """Get all conditions for a patient"""
        patient_node = f"patient_{patient_id}"
        
        if self.use_neo4j:
            return self._neo4j_query_conditions(patient_node)
        
        conditions = []
        for rel_id in self.rel_index.get((patient_node, 'HAS_CONDITION'), set()):
            rel = self.relationships[rel_id]
            condition_node = self.nodes.get(rel.target_id)
            if condition_node:
                conditions.append({
                    'condition': condition_node.properties.get('name', rel.target_id),
                    'properties': rel.properties,
                    'diagnosis_date': str(rel.start_date) if rel.start_date else None,
                    'is_current': rel.is_current
                })
        
        return conditions
    
    def get_medication_indications(self, medication_name: str) -> List[Dict]:
        """Get conditions treated by a medication"""
        med_node = f"medication_{medication_name.lower().replace(' ', '_')}"
        
        indications = []
        for rel_id in self.rel_index.get((med_node, 'TREATS'), set()):
            rel = self.relationships[rel_id]
            condition_node = self.nodes.get(rel.target_id)
            if condition_node:
                indications.append({
                    'condition': condition_node.properties.get('name', rel.target_id),
                    'properties': rel.properties
                })
        
        return indications
    
    def get_patient_graph(self, patient_id: str) -> Dict:
        """Get complete knowledge graph for a patient"""
        patient_node = f"patient_{patient_id}"
        
        nodes = []
        edges = []
        visited_nodes = set()
        
        # BFS to collect connected subgraph
        queue = [patient_node]
        while queue:
            current = queue.pop(0)
            if current in visited_nodes:
                continue
            visited_nodes.add(current)
            
            node = self.nodes.get(current)
            if node:
                nodes.append({
                    'id': node.node_id,
                    'type': node.node_type,
                    'label': node.properties.get('name', node.node_id),
                    'properties': node.properties
                })
            
            # Get relationships
            for adj_node in self.adjacency.get(current, set()):
                if adj_node not in visited_nodes:
                    queue.append(adj_node)
        
        # Collect edges
        for rel_id, rel in self.relationships.items():
            if rel.source_id in visited_nodes and rel.target_id in visited_nodes:
                edges.append({
                    'id': rel.rel_id,
                    'source': rel.source_id,
                    'target': rel.target_id,
                    'type': rel.rel_type,
                    'properties': rel.properties,
                    'is_current': rel.is_current
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'patient_id': patient_id
        }
    
    def find_related_entities(self, entity_id: str, rel_type: str = None,
                             max_depth: int = 2) -> List[Dict]:
        """Find entities related to given entity"""
        if self.use_neo4j:
            return self._neo4j_find_related(entity_id, rel_type, max_depth)
        
        related = []
        visited = set()
        
        def traverse(node_id: str, depth: int):
            if depth > max_depth or node_id in visited:
                return
            visited.add(node_id)
            
            for adj in self.adjacency.get(node_id, set()):
                node = self.nodes.get(adj)
                if node:
                    related.append({
                        'node_id': adj,
                        'type': node.node_type,
                        'name': node.properties.get('name', adj),
                        'depth': depth
                    })
                traverse(adj, depth + 1)
        
        traverse(entity_id, 1)
        return related
    
    def _neo4j_query_medications(self, patient_node: str, current_only: bool) -> List[Dict]:
        """Query medications from Neo4j"""
        with self.driver.session() as session:
            if current_only:
                query = """
                MATCH (p {node_id: $patient_node})-[r:TAKES]->(m:Medication)
                WHERE r.is_current = true
                RETURN m.name as medication, r.start_date as start_date, 
                       r.end_date as end_date, r.is_current as is_current
                """
            else:
                query = """
                MATCH (p {node_id: $patient_node})-[r:TAKES]->(m:Medication)
                RETURN m.name as medication, r.start_date as start_date, 
                       r.end_date as end_date, r.is_current as is_current
                """
            result = session.run(query, patient_node=patient_node)
            return [dict(record) for record in result]
    
    def _neo4j_query_conditions(self, patient_node: str) -> List[Dict]:
        """Query conditions from Neo4j"""
        with self.driver.session() as session:
            query = """
            MATCH (p {node_id: $patient_node})-[r:HAS_CONDITION]->(c:Condition)
            RETURN c.name as condition, r.start_date as diagnosis_date, r.is_current as is_current
            """
            result = session.run(query, patient_node=patient_node)
            return [dict(record) for record in result]
    
    def _neo4j_find_related(self, entity_id: str, rel_type: str, max_depth: int) -> List[Dict]:
        """Find related entities in Neo4j"""
        with self.driver.session() as session:
            if rel_type:
                query = f"""
                MATCH (n {{node_id: $entity_id}})-[r:{rel_type}*1..{max_depth}]-(related)
                RETURN DISTINCT related.node_id as node_id, labels(related)[0] as type, 
                       related.name as name
                """
            else:
                query = f"""
                MATCH (n {{node_id: $entity_id}})-[*1..{max_depth}]-(related)
                RETURN DISTINCT related.node_id as node_id, labels(related)[0] as type,
                       related.name as name
                """
            result = session.run(query, entity_id=entity_id)
            return [dict(record) for record in result]
    
    # ==================== Update/Maintenance ====================
    
    def update_relationship_status(self, rel_id: str, end_date: date = None):
        """Update relationship status (e.g., mark medication as stopped)"""
        if self.use_neo4j:
            with self.driver.session() as session:
                query = """
                MATCH ()-[r {rel_id: $rel_id}]->()
                SET r.end_date = $end_date, r.is_current = false
                RETURN r.rel_id
                """
                session.run(query, rel_id=rel_id, end_date=str(end_date or date.today()))
        else:
            if rel_id in self.relationships:
                rel = self.relationships[rel_id]
                rel.end_date = end_date or date.today()
                rel.is_current = False
    
    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        if self.use_neo4j:
            return self._neo4j_statistics()
        
        return {
            'total_nodes': len(self.nodes),
            'total_relationships': len(self.relationships),
            'nodes_by_type': {
                node_type: len(node_ids) 
                for node_type, node_ids in self.node_index.items()
            },
            'relationships_by_type': defaultdict(int)
        }
    
    def _neo4j_statistics(self) -> Dict:
        """Get statistics from Neo4j"""
        with self.driver.session() as session:
            # Node counts
            node_query = """
            MATCH (n)
            RETURN labels(n)[0] as type, count(*) as count
            """
            nodes = {record['type']: record['count'] for record in session.run(node_query)}
            
            # Relationship counts
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            """
            rels = {record['type']: record['count'] for record in session.run(rel_query)}
            
            return {
                'total_nodes': sum(nodes.values()),
                'total_relationships': sum(rels.values()),
                'nodes_by_type': nodes,
                'relationships_by_type': rels
            }
    
    def close(self):
        """Close database connections"""
        if self.use_neo4j and hasattr(self, 'driver'):
            self.driver.close()
