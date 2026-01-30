"""
Neo4j-Style Knowledge Graph Visualization Service
Builds interactive graph data for patient medical relationships
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node in the knowledge graph"""
    id: str
    label: str
    group: str  # patient, medication, condition, symptom, doctor, interaction
    title: str  # Tooltip on hover
    properties: Dict[str, Any] = field(default_factory=dict)
    size: int = 25
    
    def to_vis_format(self) -> Dict:
        """Convert to vis.js node format"""
        colors = {
            'patient': {'background': '#3b82f6', 'border': '#1d4ed8', 'highlight': '#60a5fa'},
            'medication': {'background': '#10b981', 'border': '#047857', 'highlight': '#34d399'},
            'condition': {'background': '#f59e0b', 'border': '#b45309', 'highlight': '#fbbf24'},
            'symptom': {'background': '#ef4444', 'border': '#b91c1c', 'highlight': '#f87171'},
            'doctor': {'background': '#8b5cf6', 'border': '#6d28d9', 'highlight': '#a78bfa'},
            'interaction': {'background': '#ec4899', 'border': '#be185d', 'highlight': '#f472b6'},
            'allergy': {'background': '#dc2626', 'border': '#991b1b', 'highlight': '#f87171'},
            'lab_result': {'background': '#06b6d4', 'border': '#0891b2', 'highlight': '#22d3ee'},
            'prescription': {'background': '#84cc16', 'border': '#4d7c0f', 'highlight': '#a3e635'},
        }
        
        icons = {
            'patient': '👤',
            'medication': '💊',
            'condition': '🏥',
            'symptom': '🩺',
            'doctor': '👨‍⚕️',
            'interaction': '⚠️',
            'allergy': '🚫',
            'lab_result': '🔬',
            'prescription': '📋',
        }
        
        color = colors.get(self.group, {'background': '#6b7280', 'border': '#4b5563', 'highlight': '#9ca3af'})
        icon = icons.get(self.group, '📌')
        
        return {
            'id': self.id,
            'label': f"{icon}\n{self.label}",
            'group': self.group,
            'title': self.title,
            'color': color,
            'size': self.size,
            'font': {'size': 12, 'color': '#1f2937', 'face': 'Inter, sans-serif'},
            'shape': 'dot',
            'borderWidth': 3,
            'shadow': True,
        }


@dataclass
class GraphEdge:
    """Edge/relationship in the knowledge graph"""
    id: str
    from_node: str
    to_node: str
    label: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    width: int = 2
    dashes: bool = False
    
    def to_vis_format(self) -> Dict:
        """Convert to vis.js edge format"""
        colors = {
            'takes': '#10b981',
            'has_condition': '#f59e0b',
            'treats': '#3b82f6',
            'interacts_with': '#ef4444',
            'prescribed_by': '#8b5cf6',
            'allergic_to': '#dc2626',
            'has_symptom': '#ec4899',
            'causes': '#f97316',
            'monitors': '#06b6d4',
            'contraindicated': '#be123c',
        }
        
        return {
            'id': self.id,
            'from': self.from_node,
            'to': self.to_node,
            'label': self.label,
            'color': {'color': colors.get(self.relationship_type, '#6b7280'), 'highlight': '#1f2937'},
            'width': self.width,
            'dashes': self.dashes,
            'font': {'size': 10, 'align': 'middle', 'color': '#4b5563'},
            'arrows': {'to': {'enabled': True, 'scaleFactor': 0.5}},
            'smooth': {'type': 'curvedCW', 'roundness': 0.2},
        }


class Neo4jVisualizationService:
    """
    Service to build Neo4j-style knowledge graph visualizations
    Works with in-memory data or can connect to actual Neo4j
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
    
    def _generate_id(self, prefix: str, name: str) -> str:
        """Generate unique ID for nodes"""
        hash_val = hashlib.md5(name.lower().encode()).hexdigest()[:8]
        return f"{prefix}_{hash_val}"
    
    def build_patient_graph(
        self,
        patient_uid: str,
        patient_data: Dict,
        medications: List[Dict],
        conditions: List[str] = None,
        allergies: List[str] = None,
        interactions: List[Dict] = None,
        prescriptions: List[Dict] = None,
        include_drug_relationships: bool = True
    ) -> Dict:
        """
        Build complete knowledge graph for a patient
        
        Args:
            patient_uid: Patient unique ID
            patient_data: Patient demographics
            medications: List of medication dicts
            conditions: List of condition names
            allergies: List of allergy names
            interactions: Drug interaction data
            prescriptions: List of prescription dicts with dates
            include_drug_relationships: Include drug-condition relationships
        
        Returns:
            Dict with nodes and edges in vis.js format
        """
        self.nodes = {}
        self.edges = {}
        conditions = conditions or []
        allergies = allergies or []
        interactions = interactions or []
        prescriptions = prescriptions or []
        
        # 1. Create patient node (center)
        patient_name = patient_data.get('name', 'Unknown Patient')
        patient_node = GraphNode(
            id=f"patient_{patient_uid}",
            label=patient_name,
            group='patient',
            title=f"<b>{patient_name}</b><br>UID: {patient_uid}<br>Age: {patient_data.get('age', 'N/A')}<br>Gender: {patient_data.get('gender', 'N/A')}",
            size=40,
            properties=patient_data
        )
        self.nodes[patient_node.id] = patient_node
        
        # 2. Create medication nodes
        for med in medications:
            med_name = med.get('name') or med.get('drug_name') or med.get('medication', 'Unknown')
            if not med_name or med_name == 'Unknown':
                continue
                
            med_id = self._generate_id('med', med_name)
            
            # Build tooltip with dates
            dosage = med.get('dosage', '')
            frequency = med.get('frequency', '')
            start_date = med.get('start_date', '')
            end_date = med.get('end_date', '')
            is_active = med.get('is_active', True)
            prescriber = med.get('prescriber', '')
            
            tooltip = f"<b>💊 {med_name}</b>"
            if dosage:
                tooltip += f"<br><b>Dosage:</b> {dosage}"
            if frequency:
                tooltip += f"<br><b>Frequency:</b> {frequency}"
            if start_date:
                # Format date nicely
                try:
                    if isinstance(start_date, str):
                        from dateutil import parser
                        start_dt = parser.parse(start_date)
                        tooltip += f"<br><b>Started:</b> {start_dt.strftime('%d %b %Y')}"
                    else:
                        tooltip += f"<br><b>Started:</b> {start_date.strftime('%d %b %Y')}"
                except:
                    tooltip += f"<br><b>Started:</b> {start_date}"
            if end_date:
                try:
                    if isinstance(end_date, str):
                        from dateutil import parser
                        end_dt = parser.parse(end_date)
                        tooltip += f"<br><b>Ended:</b> {end_dt.strftime('%d %b %Y')}"
                    else:
                        tooltip += f"<br><b>Ended:</b> {end_date.strftime('%d %b %Y')}"
                except:
                    tooltip += f"<br><b>Ended:</b> {end_date}"
            if prescriber:
                tooltip += f"<br><b>Prescribed by:</b> Dr. {prescriber}"
            
            # Status indicator
            status = "✅ Active" if is_active else "⏹️ Stopped"
            tooltip += f"<br><b>Status:</b> {status}"
            
            med_node = GraphNode(
                id=med_id,
                label=med_name[:20],
                group='medication',
                title=tooltip,
                size=30,
                properties=med
            )
            self.nodes[med_id] = med_node
            
            # Create edge: patient TAKES medication
            edge_id = f"edge_takes_{patient_uid}_{med_id}"
            self.edges[edge_id] = GraphEdge(
                id=edge_id,
                from_node=patient_node.id,
                to_node=med_id,
                label='takes',
                relationship_type='takes',
                width=3
            )
        
        # 3. Create condition nodes
        for condition in conditions:
            if not condition:
                continue
            cond_id = self._generate_id('cond', condition)
            
            cond_node = GraphNode(
                id=cond_id,
                label=condition[:25],
                group='condition',
                title=f"<b>Condition</b><br>{condition}",
                size=28
            )
            self.nodes[cond_id] = cond_node
            
            # Edge: patient HAS_CONDITION condition
            edge_id = f"edge_has_{patient_uid}_{cond_id}"
            self.edges[edge_id] = GraphEdge(
                id=edge_id,
                from_node=patient_node.id,
                to_node=cond_id,
                label='has',
                relationship_type='has_condition',
                width=2
            )
            
            # 4. Link medications that treat this condition
            if include_drug_relationships:
                for med_id, med_node in list(self.nodes.items()):
                    if med_node.group == 'medication':
                        med_name = med_node.properties.get('name', '').lower()
                        if self._medication_treats_condition(med_name, condition.lower()):
                            treat_edge_id = f"edge_treats_{med_id}_{cond_id}"
                            if treat_edge_id not in self.edges:
                                self.edges[treat_edge_id] = GraphEdge(
                                    id=treat_edge_id,
                                    from_node=med_id,
                                    to_node=cond_id,
                                    label='treats',
                                    relationship_type='treats',
                                    width=2,
                                    dashes=True
                                )
        
        # 5. Create allergy nodes
        for allergy in allergies:
            if not allergy:
                continue
            allergy_id = self._generate_id('allergy', allergy)
            
            allergy_node = GraphNode(
                id=allergy_id,
                label=allergy[:20],
                group='allergy',
                title=f"<b>⚠️ Allergy</b><br>{allergy}",
                size=25
            )
            self.nodes[allergy_id] = allergy_node
            
            # Edge: patient ALLERGIC_TO
            edge_id = f"edge_allergic_{patient_uid}_{allergy_id}"
            self.edges[edge_id] = GraphEdge(
                id=edge_id,
                from_node=patient_node.id,
                to_node=allergy_id,
                label='allergic',
                relationship_type='allergic_to',
                width=3,
                dashes=True
            )
        
        # 6. Create drug interaction nodes and edges
        for interaction in interactions:
            if not interaction:
                continue
            
            drug1 = interaction.get('drug1', '')
            drug2 = interaction.get('drug2', '')
            severity = interaction.get('severity', 'moderate')
            description = interaction.get('description', '')
            
            if not drug1 or not drug2:
                continue
            
            # Find corresponding medication nodes
            drug1_id = self._generate_id('med', drug1)
            drug2_id = self._generate_id('med', drug2)
            
            # Create interaction node
            interaction_id = self._generate_id('interaction', f"{drug1}_{drug2}")
            interaction_node = GraphNode(
                id=interaction_id,
                label=f"⚠️ {severity.upper()}",
                group='interaction',
                title=f"<b>Drug Interaction</b><br>{drug1} ↔ {drug2}<br>Severity: {severity}<br>{description}",
                size=22,
                properties=interaction
            )
            self.nodes[interaction_id] = interaction_node
            
            # Connect both drugs to interaction
            if drug1_id in self.nodes:
                self.edges[f"edge_int_{drug1_id}_{interaction_id}"] = GraphEdge(
                    id=f"edge_int_{drug1_id}_{interaction_id}",
                    from_node=drug1_id,
                    to_node=interaction_id,
                    label='interacts',
                    relationship_type='interacts_with',
                    width=3
                )
            if drug2_id in self.nodes:
                self.edges[f"edge_int_{drug2_id}_{interaction_id}"] = GraphEdge(
                    id=f"edge_int_{drug2_id}_{interaction_id}",
                    from_node=drug2_id,
                    to_node=interaction_id,
                    label='interacts',
                    relationship_type='interacts_with',
                    width=3
                )
        
        # 7. Create prescription nodes (timeline)
        for presc in prescriptions:
            if not presc:
                continue
                
            presc_uid = presc.get('prescription_uid', '')
            presc_date = presc.get('prescription_date', '')
            doctor_name = presc.get('doctor_name', 'Unknown Doctor')
            clinic = presc.get('clinic_name', '')
            presc_meds = presc.get('medications', [])
            
            if not presc_uid:
                continue
            
            presc_id = f"presc_{presc_uid}"
            
            # Format date for display
            date_display = 'Unknown Date'
            try:
                if presc_date:
                    if isinstance(presc_date, str):
                        from dateutil import parser
                        presc_dt = parser.parse(presc_date)
                        date_display = presc_dt.strftime('%d %b %Y')
                    else:
                        date_display = presc_date.strftime('%d %b %Y')
            except:
                date_display = str(presc_date)[:10] if presc_date else 'Unknown Date'
            
            # Build tooltip
            tooltip = f"<b>📋 Prescription</b><br><b>ID:</b> {presc_uid}"
            tooltip += f"<br><b>Date:</b> {date_display}"
            tooltip += f"<br><b>Doctor:</b> Dr. {doctor_name}"
            if clinic:
                tooltip += f"<br><b>Clinic:</b> {clinic}"
            if presc_meds:
                tooltip += f"<br><b>Medications:</b> {len(presc_meds)}"
                for pm in presc_meds[:3]:  # Show first 3 meds
                    pm_name = pm.get('name', '')
                    pm_dosage = pm.get('dosage', '')
                    if pm_name:
                        tooltip += f"<br>  • {pm_name}"
                        if pm_dosage:
                            tooltip += f" ({pm_dosage})"
                if len(presc_meds) > 3:
                    tooltip += f"<br>  ... and {len(presc_meds) - 3} more"
            
            presc_node = GraphNode(
                id=presc_id,
                label=f"Rx {date_display}",
                group='prescription',
                title=tooltip,
                size=28,
                properties=presc
            )
            self.nodes[presc_id] = presc_node
            
            # Edge: patient HAS prescription
            edge_id = f"edge_has_presc_{patient_uid}_{presc_id}"
            self.edges[edge_id] = GraphEdge(
                id=edge_id,
                from_node=patient_node.id,
                to_node=presc_id,
                label='received',
                relationship_type='has_prescription',
                width=2
            )
            
            # Link prescription to medications it contains
            for pm in presc_meds:
                pm_name = pm.get('name', '')
                if pm_name:
                    pm_id = self._generate_id('med', pm_name)
                    if pm_id in self.nodes:
                        edge_id = f"edge_presc_med_{presc_id}_{pm_id}"
                        if edge_id not in self.edges:
                            self.edges[edge_id] = GraphEdge(
                                id=edge_id,
                                from_node=presc_id,
                                to_node=pm_id,
                                label='includes',
                                relationship_type='includes',
                                width=2,
                                dashes=True
                            )
        
        # Convert to vis.js format
        return self._to_vis_format()
    
    def _medication_treats_condition(self, medication: str, condition: str) -> bool:
        """Check if medication commonly treats condition"""
        treatment_map = {
            # Diabetes
            'diabetes': ['metformin', 'glipizide', 'glyburide', 'insulin', 'glimepiride', 'sitagliptin', 
                        'empagliflozin', 'jardiance', 'ozempic', 'semaglutide', 'trulicity'],
            'hypertension': ['lisinopril', 'amlodipine', 'losartan', 'metoprolol', 'hydrochlorothiazide',
                           'valsartan', 'atenolol', 'carvedilol', 'olmesartan', 'telmisartan'],
            'high blood pressure': ['lisinopril', 'amlodipine', 'losartan', 'metoprolol'],
            'heart failure': ['carvedilol', 'metoprolol', 'lisinopril', 'spironolactone', 'entresto',
                             'sacubitril', 'furosemide', 'digoxin'],
            'atrial fibrillation': ['warfarin', 'apixaban', 'rivaroxaban', 'eliquis', 'xarelto',
                                   'metoprolol', 'diltiazem', 'amiodarone', 'digoxin'],
            'cholesterol': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'lipitor', 'crestor',
                          'pravastatin', 'ezetimibe'],
            'hyperlipidemia': ['atorvastatin', 'simvastatin', 'rosuvastatin'],
            'pain': ['ibuprofen', 'acetaminophen', 'naproxen', 'tramadol', 'morphine', 'oxycodone',
                    'hydrocodone', 'gabapentin', 'pregabalin'],
            'depression': ['sertraline', 'fluoxetine', 'escitalopram', 'citalopram', 'venlafaxine',
                          'duloxetine', 'bupropion', 'lexapro', 'zoloft'],
            'anxiety': ['sertraline', 'escitalopram', 'buspirone', 'lorazepam', 'alprazolam'],
            'asthma': ['albuterol', 'fluticasone', 'montelukast', 'budesonide', 'salmeterol'],
            'copd': ['tiotropium', 'spiriva', 'albuterol', 'fluticasone', 'symbicort'],
            'gerd': ['omeprazole', 'pantoprazole', 'esomeprazole', 'ranitidine', 'famotidine'],
            'acid reflux': ['omeprazole', 'pantoprazole', 'esomeprazole'],
            'infection': ['amoxicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline', 'cephalexin'],
            'thyroid': ['levothyroxine', 'synthroid', 'methimazole'],
        }
        
        for cond_keyword, meds in treatment_map.items():
            if cond_keyword in condition:
                for med in meds:
                    if med in medication:
                        return True
        return False
    
    def _to_vis_format(self) -> Dict:
        """Convert internal representation to vis.js format"""
        return {
            'nodes': [node.to_vis_format() for node in self.nodes.values()],
            'edges': [edge.to_vis_format() for edge in self.edges.values()],
            'statistics': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'node_types': self._count_by_group(),
                'generated_at': datetime.utcnow().isoformat()
            }
        }
    
    def _count_by_group(self) -> Dict[str, int]:
        """Count nodes by group type"""
        counts = {}
        for node in self.nodes.values():
            counts[node.group] = counts.get(node.group, 0) + 1
        return counts
    
    def export_cypher(self, patient_uid: str) -> str:
        """
        Export graph as Neo4j Cypher statements
        Can be used to import into actual Neo4j database
        """
        statements = ["// Neo4j Cypher Import Script", f"// Patient: {patient_uid}", ""]
        
        # Create nodes
        for node in self.nodes.values():
            props = ', '.join([f'{k}: "{v}"' for k, v in node.properties.items() if isinstance(v, str)])
            statements.append(
                f"CREATE (n:{node.group.capitalize()} {{id: '{node.id}', label: '{node.label}', {props}}});"
            )
        
        statements.append("")
        
        # Create relationships
        for edge in self.edges.values():
            statements.append(
                f"MATCH (a {{id: '{edge.from_node}'}}), (b {{id: '{edge.to_node}'}}) "
                f"CREATE (a)-[:{edge.relationship_type.upper()} {{label: '{edge.label}'}}]->(b);"
            )
        
        return '\n'.join(statements)


# Singleton instance
_visualization_service = None

def get_neo4j_visualization_service() -> Neo4jVisualizationService:
    """Get or create visualization service singleton"""
    global _visualization_service
    if _visualization_service is None:
        _visualization_service = Neo4jVisualizationService()
    return _visualization_service
