"""
Drug Interaction Service - Safety checks and interaction analysis
"""
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .drug_normalization_service import DrugNormalizationService

logger = logging.getLogger(__name__)


class InteractionSeverity(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"


@dataclass
class DrugInteraction:
    """Drug interaction details"""
    drug1: str
    drug2: str
    severity: InteractionSeverity
    description: str
    mechanism: str
    clinical_effects: str
    management: str
    evidence_level: str
    confidence: float


@dataclass
class AllergyAlert:
    """Allergy-based risk"""
    drug: str
    allergen: str
    risk_type: str
    severity: InteractionSeverity
    description: str
    alternatives: List[str]


@dataclass
class DuplicateTherapyAlert:
    """Duplicate therapy warning"""
    drugs: List[str]
    drug_class: str
    description: str
    recommendation: str


@dataclass
class SafetyAnalysisResult:
    """Complete safety analysis result"""
    interactions: List[DrugInteraction]
    allergy_alerts: List[AllergyAlert]
    duplicate_therapies: List[DuplicateTherapyAlert]
    contraindications: List[Dict]
    overall_risk_level: str
    high_priority_alerts: List[Dict]
    suppressed_alerts: List[Dict]


class DrugInteractionService:
    """
    Drug safety and interaction analysis service
    
    Features:
    - Drug-drug interaction detection
    - Duplicate therapy detection
    - Allergy-based risk analysis
    - Contraindication checking
    - Severity ranking
    - Alert suppression for low-value alerts
    """
    
    def __init__(self):
        self.drug_normalizer = DrugNormalizationService()
        self.interactions_db = self._load_interactions_database()
        self.class_interactions = self._load_class_interactions()
        self.contraindications = self._load_contraindications()
    
    def _load_interactions_database(self) -> Dict:
        """Load drug-drug interactions database"""
        # Key: tuple of sorted generic names
        # This is a simplified database - in production, use DrugBank, RxNav, etc.
        return {
            # Anticoagulants
            ('aspirin', 'warfarin'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Increased risk of bleeding when aspirin is combined with warfarin',
                'mechanism': 'Both drugs affect hemostasis through different mechanisms',
                'clinical_effects': 'Increased risk of serious bleeding, including GI and intracranial hemorrhage',
                'management': 'Avoid combination if possible. If necessary, use lowest effective aspirin dose and monitor closely',
                'evidence': 'established'
            },
            ('clopidogrel', 'warfarin'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Increased bleeding risk with dual anticoagulant/antiplatelet therapy',
                'mechanism': 'Combined anticoagulant and antiplatelet effects',
                'clinical_effects': 'Significantly increased risk of bleeding',
                'management': 'Monitor closely for signs of bleeding. Consider PPI for GI protection',
                'evidence': 'established'
            },
            ('aspirin', 'clopidogrel'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'Dual antiplatelet therapy increases bleeding risk',
                'mechanism': 'Additive antiplatelet effects',
                'clinical_effects': 'Increased bleeding risk, but often used intentionally for cardiac protection',
                'management': 'Often prescribed together intentionally. Monitor for bleeding',
                'evidence': 'established'
            },
            
            # NSAIDs
            ('ibuprofen', 'aspirin'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'Ibuprofen may reduce cardioprotective effect of low-dose aspirin',
                'mechanism': 'Competitive inhibition of COX-1 platelet binding site',
                'clinical_effects': 'Reduced antiplatelet effect of aspirin; increased GI bleeding risk',
                'management': 'Take aspirin 30 minutes before ibuprofen or use alternative analgesic',
                'evidence': 'established'
            },
            ('ibuprofen', 'warfarin'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'NSAIDs increase bleeding risk with warfarin',
                'mechanism': 'NSAIDs inhibit platelet function and may increase warfarin levels',
                'clinical_effects': 'Increased risk of GI and other bleeding',
                'management': 'Avoid if possible. Use acetaminophen for pain relief',
                'evidence': 'established'
            },
            ('diclofenac', 'warfarin'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'NSAIDs increase bleeding risk with warfarin',
                'mechanism': 'NSAIDs inhibit platelet function and may increase warfarin levels',
                'clinical_effects': 'Increased risk of GI and other bleeding',
                'management': 'Avoid if possible. Use acetaminophen for pain relief',
                'evidence': 'established'
            },
            
            # ACE Inhibitors & Potassium
            ('lisinopril', 'spironolactone'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Risk of hyperkalemia',
                'mechanism': 'Both drugs increase potassium levels',
                'clinical_effects': 'Potentially life-threatening hyperkalemia',
                'management': 'Monitor potassium levels closely. Consider alternative',
                'evidence': 'established'
            },
            ('losartan', 'spironolactone'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Risk of hyperkalemia',
                'mechanism': 'Both drugs increase potassium levels',
                'clinical_effects': 'Potentially life-threatening hyperkalemia',
                'management': 'Monitor potassium levels closely',
                'evidence': 'established'
            },
            
            # Metformin interactions
            ('metformin', 'alcohol'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Increased risk of lactic acidosis',
                'mechanism': 'Alcohol increases lactate production and impairs gluconeogenesis',
                'clinical_effects': 'Rare but potentially fatal lactic acidosis',
                'management': 'Limit alcohol intake significantly',
                'evidence': 'established'
            },
            
            # CNS Depression
            ('alprazolam', 'tramadol'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'CNS and respiratory depression risk',
                'mechanism': 'Additive CNS depressant effects',
                'clinical_effects': 'Sedation, respiratory depression, coma, death',
                'management': 'Avoid combination. If necessary, use lowest doses and monitor',
                'evidence': 'established'
            },
            ('alprazolam', 'codeine'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'CNS and respiratory depression risk',
                'mechanism': 'Additive CNS depressant effects',
                'clinical_effects': 'Sedation, respiratory depression, coma, death',
                'management': 'Avoid combination if possible',
                'evidence': 'established'
            },
            ('clonazepam', 'tramadol'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'CNS and respiratory depression risk',
                'mechanism': 'Additive CNS depressant effects',
                'clinical_effects': 'Sedation, respiratory depression',
                'management': 'Avoid or use with extreme caution',
                'evidence': 'established'
            },
            
            # Serotonin Syndrome
            ('sertraline', 'tramadol'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Risk of serotonin syndrome',
                'mechanism': 'Both drugs increase serotonin levels',
                'clinical_effects': 'Agitation, hyperthermia, tachycardia, neuromuscular abnormalities',
                'management': 'Use alternative analgesic. Monitor for serotonin syndrome symptoms',
                'evidence': 'established'
            },
            ('escitalopram', 'tramadol'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Risk of serotonin syndrome',
                'mechanism': 'Both drugs increase serotonin levels',
                'clinical_effects': 'Agitation, hyperthermia, neuromuscular symptoms',
                'management': 'Avoid combination if possible',
                'evidence': 'established'
            },
            
            # Statin interactions
            ('atorvastatin', 'clarithromycin'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Increased statin levels and myopathy risk',
                'mechanism': 'CYP3A4 inhibition increases statin concentration',
                'clinical_effects': 'Increased risk of rhabdomyolysis',
                'management': 'Use alternative antibiotic or temporarily hold statin',
                'evidence': 'established'
            },
            ('simvastatin', 'clarithromycin'): {
                'severity': InteractionSeverity.CONTRAINDICATED,
                'description': 'Contraindicated - severe myopathy risk',
                'mechanism': 'CYP3A4 inhibition dramatically increases simvastatin levels',
                'clinical_effects': 'High risk of rhabdomyolysis',
                'management': 'Do not use together. Use alternative antibiotic',
                'evidence': 'established'
            },
            
            # Thyroid
            ('levothyroxine', 'calcium'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'Calcium reduces levothyroxine absorption',
                'mechanism': 'Calcium binds levothyroxine in GI tract',
                'clinical_effects': 'Reduced thyroid hormone levels, hypothyroid symptoms',
                'management': 'Separate administration by 4 hours',
                'evidence': 'established'
            },
            ('levothyroxine', 'omeprazole'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'PPIs may reduce levothyroxine absorption',
                'mechanism': 'Altered gastric pH affects absorption',
                'clinical_effects': 'Reduced levothyroxine effectiveness',
                'management': 'Monitor thyroid levels. May need dose adjustment',
                'evidence': 'probable'
            },
            
            # QT Prolongation
            ('azithromycin', 'ondansetron'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'Additive QT prolongation risk',
                'mechanism': 'Both drugs can prolong QT interval',
                'clinical_effects': 'Risk of cardiac arrhythmias including Torsades de Pointes',
                'management': 'Monitor ECG in high-risk patients',
                'evidence': 'probable'
            },
            ('ciprofloxacin', 'ondansetron'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'Additive QT prolongation risk',
                'mechanism': 'Both drugs can prolong QT interval',
                'clinical_effects': 'Risk of cardiac arrhythmias',
                'management': 'Use with caution, especially in elderly',
                'evidence': 'probable'
            },
            
            # Diabetes
            ('metformin', 'furosemide'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'Furosemide may increase metformin levels',
                'mechanism': 'Competition for renal tubular transport',
                'clinical_effects': 'Increased metformin concentration and effect',
                'management': 'Monitor blood glucose and for metformin side effects',
                'evidence': 'probable'
            },
            ('glimepiride', 'fluconazole'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'Increased hypoglycemia risk',
                'mechanism': 'Fluconazole inhibits sulfonylurea metabolism',
                'clinical_effects': 'Enhanced hypoglycemic effect',
                'management': 'Monitor blood glucose closely. May need dose reduction',
                'evidence': 'established'
            },
        }
    
    def _load_class_interactions(self) -> Dict:
        """Load drug class-level interactions"""
        return {
            ('nsaid', 'anticoagulant'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'NSAIDs increase bleeding risk with anticoagulants',
                'management': 'Avoid NSAIDs with anticoagulants when possible'
            },
            ('nsaid', 'ace_inhibitor'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'NSAIDs may reduce antihypertensive effect and worsen renal function',
                'management': 'Monitor blood pressure and renal function'
            },
            ('nsaid', 'arb'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'NSAIDs may reduce antihypertensive effect and worsen renal function',
                'management': 'Monitor blood pressure and renal function'
            },
            ('benzodiazepine', 'opioid_analgesic'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Combined CNS depression risk',
                'management': 'Avoid combination. Black box warning exists'
            },
            ('ssri', 'opioid_analgesic'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Risk of serotonin syndrome with certain opioids',
                'management': 'Monitor for serotonin syndrome symptoms'
            },
            ('statin', 'fibrate'): {
                'severity': InteractionSeverity.MAJOR,
                'description': 'Increased myopathy risk',
                'management': 'Use lowest effective statin dose. Monitor for muscle symptoms'
            },
            ('ppi', 'clopidogrel'): {
                'severity': InteractionSeverity.MODERATE,
                'description': 'PPIs may reduce clopidogrel effectiveness',
                'management': 'Use pantoprazole if PPI needed (least interaction)'
            },
        }
    
    def _load_contraindications(self) -> Dict:
        """Load condition-based contraindications"""
        return {
            'renal_impairment': {
                'contraindicated': ['metformin'],
                'use_caution': ['nsaid', 'ace_inhibitor', 'arb', 'digoxin'],
                'dose_adjust': ['gabapentin', 'pregabalin', 'metformin']
            },
            'liver_disease': {
                'contraindicated': ['methotrexate'],
                'use_caution': ['statin', 'paracetamol'],
                'dose_adjust': ['atorvastatin', 'simvastatin']
            },
            'heart_failure': {
                'contraindicated': ['nsaid', 'thiazolidinedione'],
                'use_caution': ['calcium_channel_blocker', 'beta_blocker'],
            },
            'asthma': {
                'contraindicated': ['beta_blocker'],
                'use_caution': ['aspirin', 'nsaid'],
            },
            'pregnancy': {
                'contraindicated': ['warfarin', 'statin', 'ace_inhibitor', 'arb', 'methotrexate'],
                'use_caution': ['nsaid'],
            },
            'gi_ulcer': {
                'contraindicated': [],
                'use_caution': ['nsaid', 'aspirin', 'corticosteroid'],
            }
        }
    
    def analyze_safety(self, medications: List[str], 
                       patient_allergies: List[str] = None,
                       patient_conditions: List[str] = None,
                       suppress_low_value: bool = True) -> SafetyAnalysisResult:
        """
        Perform complete safety analysis on a medication list
        
        Args:
            medications: List of medication names
            patient_allergies: Known patient allergies
            patient_conditions: Patient's medical conditions
            suppress_low_value: Whether to suppress minor/noisy alerts
            
        Returns:
            SafetyAnalysisResult with all findings
        """
        interactions = []
        allergy_alerts = []
        duplicate_therapies = []
        contraindications = []
        suppressed = []
        
        # Normalize all medication names
        normalized_meds = [
            (med, self.drug_normalizer.normalize(med))
            for med in medications
        ]
        
        # Check drug-drug interactions
        for i, (med1, norm1) in enumerate(normalized_meds):
            for j, (med2, norm2) in enumerate(normalized_meds[i+1:], i+1):
                interaction = self._check_interaction(
                    norm1.generic_name, 
                    norm2.generic_name,
                    norm1.drug_class,
                    norm2.drug_class
                )
                if interaction:
                    if suppress_low_value and interaction.severity == InteractionSeverity.MINOR:
                        suppressed.append({
                            'type': 'interaction',
                            'drugs': [med1, med2],
                            'reason': 'Minor severity'
                        })
                    else:
                        interactions.append(interaction)
        
        # Check for duplicate therapies
        duplicates = self._check_duplicate_therapies(normalized_meds)
        duplicate_therapies.extend(duplicates)
        
        # Check allergies
        if patient_allergies:
            for med, norm in normalized_meds:
                allergy = self._check_allergy(med, norm, patient_allergies)
                if allergy:
                    allergy_alerts.append(allergy)
        
        # Check contraindications
        if patient_conditions:
            for med, norm in normalized_meds:
                contra = self._check_contraindications(med, norm, patient_conditions)
                contraindications.extend(contra)
        
        # Determine overall risk level
        overall_risk = self._calculate_overall_risk(
            interactions, allergy_alerts, duplicate_therapies, contraindications
        )
        
        # Identify high priority alerts
        high_priority = self._identify_high_priority(
            interactions, allergy_alerts, contraindications
        )
        
        return SafetyAnalysisResult(
            interactions=interactions,
            allergy_alerts=allergy_alerts,
            duplicate_therapies=duplicate_therapies,
            contraindications=contraindications,
            overall_risk_level=overall_risk,
            high_priority_alerts=high_priority,
            suppressed_alerts=suppressed
        )
    
    def _check_interaction(self, drug1: str, drug2: str, 
                          class1: str, class2: str) -> Optional[DrugInteraction]:
        """Check for interaction between two drugs"""
        # Normalize names
        d1 = drug1.lower()
        d2 = drug2.lower()
        
        # Check specific drug interaction
        key = tuple(sorted([d1, d2]))
        if key in self.interactions_db:
            data = self.interactions_db[key]
            return DrugInteraction(
                drug1=drug1,
                drug2=drug2,
                severity=data['severity'],
                description=data['description'],
                mechanism=data.get('mechanism', ''),
                clinical_effects=data.get('clinical_effects', ''),
                management=data.get('management', ''),
                evidence_level=data.get('evidence', 'unknown'),
                confidence=0.95
            )
        
        # Check class-level interaction
        c1 = class1.lower() if class1 else ''
        c2 = class2.lower() if class2 else ''
        
        class_key = tuple(sorted([c1, c2]))
        if class_key in self.class_interactions:
            data = self.class_interactions[class_key]
            return DrugInteraction(
                drug1=drug1,
                drug2=drug2,
                severity=data['severity'],
                description=data['description'],
                mechanism='Class-level interaction',
                clinical_effects='',
                management=data.get('management', ''),
                evidence_level='class-based',
                confidence=0.8
            )
        
        # Check partial class matches
        for (c1_key, c2_key), data in self.class_interactions.items():
            if (c1_key in c1 or c1 in c1_key) and (c2_key in c2 or c2 in c2_key):
                return DrugInteraction(
                    drug1=drug1,
                    drug2=drug2,
                    severity=data['severity'],
                    description=data['description'],
                    mechanism='Class-level interaction',
                    clinical_effects='',
                    management=data.get('management', ''),
                    evidence_level='class-based',
                    confidence=0.7
                )
        
        return None
    
    def _check_duplicate_therapies(self, normalized_meds: List) -> List[DuplicateTherapyAlert]:
        """Check for duplicate therapies (same drug class)"""
        alerts = []
        
        # Group by drug class
        class_groups = {}
        for med, norm in normalized_meds:
            drug_class = norm.drug_class
            if drug_class and drug_class != 'unknown':
                if drug_class not in class_groups:
                    class_groups[drug_class] = []
                class_groups[drug_class].append(med)
        
        # Flag duplicates
        for drug_class, meds in class_groups.items():
            if len(meds) > 1:
                # Some combinations are intentional
                if drug_class in ['vitamin', 'mineral', 'supplement']:
                    continue
                
                alerts.append(DuplicateTherapyAlert(
                    drugs=meds,
                    drug_class=drug_class,
                    description=f"Multiple {drug_class} medications prescribed",
                    recommendation="Review if therapeutic duplication is intended"
                ))
        
        # Check for same generic drug
        generic_groups = {}
        for med, norm in normalized_meds:
            generic = norm.generic_name.lower()
            if generic not in generic_groups:
                generic_groups[generic] = []
            generic_groups[generic].append(med)
        
        for generic, meds in generic_groups.items():
            if len(meds) > 1:
                alerts.append(DuplicateTherapyAlert(
                    drugs=meds,
                    drug_class=generic,
                    description=f"Same medication ({generic}) prescribed multiple times",
                    recommendation="Consolidate to single prescription or verify intentional"
                ))
        
        return alerts
    
    def _check_allergy(self, med: str, norm, allergies: List[str]) -> Optional[AllergyAlert]:
        """Check if medication conflicts with known allergies"""
        # Cross-reactivity patterns
        cross_reactivity = {
            'penicillin': ['amoxicillin', 'ampicillin', 'piperacillin'],
            'sulfa': ['sulfamethoxazole', 'sulfasalazine', 'celecoxib'],
            'aspirin': ['ibuprofen', 'diclofenac', 'naproxen'],  # Cross-sensitivity in some patients
            'cephalosporin': ['cephalexin', 'cefixime', 'ceftriaxone'],
        }
        
        med_lower = med.lower()
        generic_lower = norm.generic_name.lower()
        
        for allergy in allergies:
            allergy_lower = allergy.lower()
            
            # Direct match
            if allergy_lower in med_lower or allergy_lower in generic_lower:
                return AllergyAlert(
                    drug=med,
                    allergen=allergy,
                    risk_type='direct',
                    severity=InteractionSeverity.CONTRAINDICATED,
                    description=f"Patient is allergic to {allergy}",
                    alternatives=self.drug_normalizer.get_therapeutic_alternatives(med)
                )
            
            # Cross-reactivity
            for allergen_class, related_drugs in cross_reactivity.items():
                if allergy_lower in allergen_class or allergen_class in allergy_lower:
                    if generic_lower in related_drugs or any(d in generic_lower for d in related_drugs):
                        return AllergyAlert(
                            drug=med,
                            allergen=allergy,
                            risk_type='cross_reactivity',
                            severity=InteractionSeverity.MAJOR,
                            description=f"Potential cross-reactivity with {allergy} allergy",
                            alternatives=[]
                        )
        
        return None
    
    def _check_contraindications(self, med: str, norm, conditions: List[str]) -> List[Dict]:
        """Check contraindications based on patient conditions"""
        results = []
        
        generic_lower = norm.generic_name.lower()
        class_lower = norm.drug_class.lower() if norm.drug_class else ''
        
        for condition in conditions:
            condition_lower = condition.lower().replace(' ', '_')
            
            if condition_lower in self.contraindications:
                data = self.contraindications[condition_lower]
                
                # Check contraindicated
                for contra in data.get('contraindicated', []):
                    if contra in generic_lower or contra in class_lower:
                        results.append({
                            'drug': med,
                            'condition': condition,
                            'level': 'contraindicated',
                            'description': f"{med} is contraindicated in {condition}"
                        })
                
                # Check caution
                for caution in data.get('use_caution', []):
                    if caution in generic_lower or caution in class_lower:
                        results.append({
                            'drug': med,
                            'condition': condition,
                            'level': 'caution',
                            'description': f"Use {med} with caution in {condition}"
                        })
        
        return results
    
    def _calculate_overall_risk(self, interactions, allergies, duplicates, contraindications) -> str:
        """Calculate overall risk level"""
        # Count severity levels
        contraindicated_count = sum(
            1 for i in interactions if i.severity == InteractionSeverity.CONTRAINDICATED
        ) + len([a for a in allergies if a.severity == InteractionSeverity.CONTRAINDICATED])
        
        major_count = sum(
            1 for i in interactions if i.severity == InteractionSeverity.MAJOR
        ) + len([a for a in allergies if a.severity == InteractionSeverity.MAJOR])
        
        if contraindicated_count > 0:
            return "CRITICAL"
        elif major_count >= 2:
            return "HIGH"
        elif major_count == 1:
            return "MODERATE"
        elif len(interactions) > 0 or len(duplicates) > 0:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _identify_high_priority(self, interactions, allergies, contraindications) -> List[Dict]:
        """Identify alerts requiring immediate attention"""
        high_priority = []
        
        for interaction in interactions:
            if interaction.severity in [InteractionSeverity.MAJOR, InteractionSeverity.CONTRAINDICATED]:
                high_priority.append({
                    'type': 'interaction',
                    'severity': interaction.severity.value,
                    'drugs': [interaction.drug1, interaction.drug2],
                    'description': interaction.description,
                    'action_required': interaction.management
                })
        
        for allergy in allergies:
            high_priority.append({
                'type': 'allergy',
                'severity': allergy.severity.value,
                'drug': allergy.drug,
                'allergen': allergy.allergen,
                'description': allergy.description,
                'action_required': 'Consider alternative medication'
            })
        
        for contra in contraindications:
            if contra['level'] == 'contraindicated':
                high_priority.append({
                    'type': 'contraindication',
                    'severity': 'contraindicated',
                    'drug': contra['drug'],
                    'condition': contra['condition'],
                    'description': contra['description'],
                    'action_required': 'Do not prescribe'
                })
        
        return high_priority
    
    def to_dict(self, result: SafetyAnalysisResult) -> Dict:
        """Convert SafetyAnalysisResult to dictionary"""
        return {
            'interactions': [
                {
                    'drug1': i.drug1,
                    'drug2': i.drug2,
                    'severity': i.severity.value,
                    'description': i.description,
                    'mechanism': i.mechanism,
                    'clinical_effects': i.clinical_effects,
                    'management': i.management,
                    'evidence_level': i.evidence_level
                }
                for i in result.interactions
            ],
            'allergy_alerts': [
                {
                    'drug': a.drug,
                    'allergen': a.allergen,
                    'risk_type': a.risk_type,
                    'severity': a.severity.value,
                    'description': a.description,
                    'alternatives': a.alternatives
                }
                for a in result.allergy_alerts
            ],
            'duplicate_therapies': [
                {
                    'drugs': d.drugs,
                    'drug_class': d.drug_class,
                    'description': d.description,
                    'recommendation': d.recommendation
                }
                for d in result.duplicate_therapies
            ],
            'contraindications': result.contraindications,
            'overall_risk_level': result.overall_risk_level,
            'high_priority_alerts': result.high_priority_alerts,
            'suppressed_alerts_count': len(result.suppressed_alerts)
        }
