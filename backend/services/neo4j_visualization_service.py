"""
Neo4j Visualization Service - Graph visualization for patient data
Provides visualization data structure and Cypher export
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class Neo4jVisualizationService:
    """
    Service for building graph visualization data and Neo4j export
    """
    
    def __init__(self):
        self._patient_graphs: Dict[str, Dict] = {}
    
    def build_patient_graph(
        self,
        patient_uid: str,
        patient_data: Dict,
        medications: List[Dict] = None,
        conditions: List[str] = None,
        allergies: List[str] = None,
        interactions: List[Dict] = None,
        prescriptions: List[Dict] = None,
        include_drug_relationships: bool = True
    ) -> Dict:
        """
        Build a graph visualization structure for a patient
        
        Returns a structure suitable for visualization libraries like D3.js or vis.js
        """
        nodes = []
        edges = []
        node_id_counter = 1
        node_map = {}  # name -> id mapping
        
        # Patient node (central)
        patient_node_id = f"patient_{patient_uid}"
        nodes.append({
            "id": patient_node_id,
            "label": patient_data.get("name", "Patient"),
            "type": "patient",
            "group": "patient",
            "properties": {
                "uid": patient_uid,
                "age": patient_data.get("age"),
                "gender": patient_data.get("gender"),
                "blood_group": patient_data.get("blood_group")
            },
            "size": 40,
            "color": "#4F46E5"  # Indigo for patient
        })
        node_map["patient"] = patient_node_id
        
        # Medication nodes
        for med in (medications or []):
            med_name = med.get("name") or med.get("drug_name", "Unknown")
            med_id = f"med_{med_name.lower().replace(' ', '_')}"
            
            if med_id not in node_map:
                nodes.append({
                    "id": med_id,
                    "label": med_name,
                    "type": "medication",
                    "group": "medication",
                    "properties": {
                        "dosage": med.get("dosage"),
                        "frequency": med.get("frequency"),
                        "route": med.get("route")
                    },
                    "size": 25,
                    "color": "#10B981"  # Green for medications
                })
                node_map[med_id] = med_id
            
            # Edge: Patient -> Medication (TAKES)
            edges.append({
                "id": f"edge_{patient_node_id}_{med_id}",
                "source": patient_node_id,
                "target": med_id,
                "label": "TAKES",
                "type": "takes",
                "properties": {
                    "dosage": med.get("dosage"),
                    "frequency": med.get("frequency")
                }
            })
        
        # Condition nodes
        for condition in (conditions or []):
            if not condition:
                continue
            cond_id = f"cond_{condition.lower().replace(' ', '_')}"
            
            if cond_id not in node_map:
                nodes.append({
                    "id": cond_id,
                    "label": condition,
                    "type": "condition",
                    "group": "condition",
                    "properties": {},
                    "size": 25,
                    "color": "#F59E0B"  # Amber for conditions
                })
                node_map[cond_id] = cond_id
            
            # Edge: Patient -> Condition (HAS_CONDITION)
            edges.append({
                "id": f"edge_{patient_node_id}_{cond_id}",
                "source": patient_node_id,
                "target": cond_id,
                "label": "HAS_CONDITION",
                "type": "has_condition"
            })
        
        # Allergy nodes
        for allergy in (allergies or []):
            if not allergy:
                continue
            allergy_id = f"allergy_{allergy.lower().replace(' ', '_')}"
            
            if allergy_id not in node_map:
                nodes.append({
                    "id": allergy_id,
                    "label": allergy,
                    "type": "allergy",
                    "group": "allergy",
                    "properties": {},
                    "size": 20,
                    "color": "#EF4444"  # Red for allergies
                })
                node_map[allergy_id] = allergy_id
            
            # Edge: Patient -> Allergy (ALLERGIC_TO)
            edges.append({
                "id": f"edge_{patient_node_id}_{allergy_id}",
                "source": patient_node_id,
                "target": allergy_id,
                "label": "ALLERGIC_TO",
                "type": "allergic_to"
            })
        
        # Interaction edges (between medications)
        for interaction in (interactions or []):
            drug1 = interaction.get("drug1", "")
            drug2 = interaction.get("drug2", "")
            severity = interaction.get("severity", "unknown")
            
            drug1_id = f"med_{drug1.lower().replace(' ', '_')}"
            drug2_id = f"med_{drug2.lower().replace(' ', '_')}"
            
            if drug1_id in node_map and drug2_id in node_map:
                edges.append({
                    "id": f"edge_interaction_{drug1_id}_{drug2_id}",
                    "source": drug1_id,
                    "target": drug2_id,
                    "label": "INTERACTS_WITH",
                    "type": "interaction",
                    "properties": {
                        "severity": severity,
                        "description": interaction.get("description")
                    },
                    "color": "#EF4444" if severity in ["major", "severe"] else "#F59E0B"
                })
        
        # Prescription nodes (optional)
        for rx in (prescriptions or []):
            rx_id = rx.get("prescription_id") or rx.get("id")
            if rx_id:
                rx_node_id = f"rx_{rx_id}"
                nodes.append({
                    "id": rx_node_id,
                    "label": f"Rx #{rx_id}",
                    "type": "prescription",
                    "group": "prescription",
                    "properties": {
                        "date": rx.get("date"),
                        "doctor": rx.get("doctor")
                    },
                    "size": 20,
                    "color": "#8B5CF6"  # Purple for prescriptions
                })
                
                edges.append({
                    "id": f"edge_{patient_node_id}_{rx_node_id}",
                    "source": patient_node_id,
                    "target": rx_node_id,
                    "label": "HAS_PRESCRIPTION",
                    "type": "has_prescription"
                })
        
        # Build drug-condition relationships if requested
        if include_drug_relationships:
            drug_condition_map = self._get_drug_condition_relationships()
            for med in (medications or []):
                med_name = (med.get("name") or med.get("drug_name", "")).lower()
                med_id = f"med_{med_name.replace(' ', '_')}"
                
                for cond_name, treating_drugs in drug_condition_map.items():
                    if med_name in [d.lower() for d in treating_drugs]:
                        cond_id = f"cond_{cond_name.lower().replace(' ', '_')}"
                        if cond_id in node_map:
                            edges.append({
                                "id": f"edge_treats_{med_id}_{cond_id}",
                                "source": med_id,
                                "target": cond_id,
                                "label": "TREATS",
                                "type": "treats",
                                "color": "#10B981"
                            })
        
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "patient_uid": patient_uid,
            "patient_name": patient_data.get("name"),
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "medications": len([n for n in nodes if n["type"] == "medication"]),
                "conditions": len([n for n in nodes if n["type"] == "condition"]),
                "allergies": len([n for n in nodes if n["type"] == "allergy"]),
                "interactions": len([e for e in edges if e["type"] == "interaction"])
            }
        }
        
        # Cache the graph
        self._patient_graphs[patient_uid] = graph_data
        
        return graph_data
    
    def _get_drug_condition_relationships(self) -> Dict[str, List[str]]:
        """Get mapping of conditions to drugs that treat them"""
        return {
            "type 2 diabetes": ["metformin", "glipizide", "sitagliptin", "insulin"],
            "hypertension": ["lisinopril", "amlodipine", "losartan", "metoprolol", "hydrochlorothiazide"],
            "hyperlipidemia": ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin"],
            "coronary artery disease": ["aspirin", "clopidogrel", "atorvastatin", "metoprolol"],
            "atrial fibrillation": ["warfarin", "apixaban", "rivaroxaban", "metoprolol"],
            "heart failure": ["lisinopril", "carvedilol", "furosemide", "spironolactone"],
            "copd": ["albuterol", "tiotropium", "fluticasone", "budesonide"],
            "asthma": ["albuterol", "fluticasone", "montelukast", "budesonide"],
            "depression": ["sertraline", "fluoxetine", "escitalopram", "bupropion"],
            "anxiety": ["sertraline", "buspirone", "escitalopram", "lorazepam"],
            "hypothyroidism": ["levothyroxine"],
            "gerd": ["omeprazole", "pantoprazole", "famotidine", "ranitidine"],
            "osteoporosis": ["alendronate", "risedronate", "calcium", "vitamin d"],
            "rheumatoid arthritis": ["methotrexate", "prednisone", "adalimumab", "etanercept"],
            "chronic kidney disease": ["lisinopril", "losartan", "sodium bicarbonate"],
        }
    
    def export_cypher(self, patient_uid: str) -> str:
        """
        Export the patient graph as Cypher statements for Neo4j import
        """
        graph = self._patient_graphs.get(patient_uid)
        if not graph:
            return "// No graph data available for this patient"
        
        cypher_lines = [
            "// Neo4j Cypher Export",
            f"// Patient: {graph.get('patient_name', patient_uid)}",
            f"// Generated: {datetime.now().isoformat()}",
            "",
            "// Create nodes"
        ]
        
        # Create node statements
        for node in graph.get("nodes", []):
            node_type = node["type"].capitalize()
            props = {
                "id": node["id"],
                "name": node["label"],
                **node.get("properties", {})
            }
            # Filter out None values
            props = {k: v for k, v in props.items() if v is not None}
            props_str = ", ".join([f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}' 
                                   for k, v in props.items()])
            cypher_lines.append(f"CREATE (:{node_type} {{{props_str}}})")
        
        cypher_lines.append("")
        cypher_lines.append("// Create relationships")
        
        # Create relationship statements
        for edge in graph.get("edges", []):
            rel_type = edge["label"].upper().replace(" ", "_")
            cypher_lines.append(
                f'MATCH (a {{id: "{edge["source"]}"}}), (b {{id: "{edge["target"]}"}}) '
                f'CREATE (a)-[:{rel_type}]->(b)'
            )
        
        return "\n".join(cypher_lines)
    
    def get_graph(self, patient_uid: str) -> Optional[Dict]:
        """Get cached graph for a patient"""
        return self._patient_graphs.get(patient_uid)
    
    def clear_cache(self, patient_uid: str = None):
        """Clear cached graphs"""
        if patient_uid:
            self._patient_graphs.pop(patient_uid, None)
        else:
            self._patient_graphs.clear()


# Singleton instance
_viz_service: Optional[Neo4jVisualizationService] = None


def get_neo4j_visualization_service() -> Neo4jVisualizationService:
    """Get or create the visualization service singleton"""
    global _viz_service
    if _viz_service is None:
        _viz_service = Neo4jVisualizationService()
    return _viz_service
