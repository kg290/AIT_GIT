"""
Enhanced API Routes
API endpoints for all new services
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

from backend.database import get_db
from backend.services.patient_history_service import PatientHistoryService
from backend.services.enhanced_knowledge_graph_service import KnowledgeGraphService
from backend.services.enhanced_temporal_reasoning_service import EnhancedTemporalReasoningService
from backend.services.explainability_service import ExplainabilityService
from backend.services.uncertainty_service import UncertaintyService
from backend.services.human_review_service import HumanReviewService
from backend.services.conversational_query_service import ConversationalQueryService
from backend.services.compliance_service import ComplianceService
from backend.services.drug_normalization_service import DrugNormalizationService
from backend.services.drug_interaction_service import DrugInteractionService
from backend.services.temporal_reasoning_service import TemporalReasoningService

router = APIRouter(prefix="/api/v2", tags=["Enhanced API"])

# Initialize singleton service instances
_drug_normalizer = None
_drug_interaction_service = None
_temporal_service = None


def get_drug_normalizer() -> DrugNormalizationService:
    """Get or create drug normalization service singleton"""
    global _drug_normalizer
    if _drug_normalizer is None:
        _drug_normalizer = DrugNormalizationService()
    return _drug_normalizer


def get_drug_interaction_service() -> DrugInteractionService:
    """Get or create drug interaction service singleton"""
    global _drug_interaction_service
    if _drug_interaction_service is None:
        _drug_interaction_service = DrugInteractionService()
    return _drug_interaction_service


def get_temporal_service() -> TemporalReasoningService:
    """Get or create temporal reasoning service singleton"""
    global _temporal_service
    if _temporal_service is None:
        _temporal_service = TemporalReasoningService()
    return _temporal_service


# ==================== Request/Response Models ====================

class MedicationInput(BaseModel):
    medication_name: str
    dosage: str
    frequency: str
    prescribing_doctor: str = None
    indication: str = None
    start_date: datetime = None


class ConditionInput(BaseModel):
    condition_name: str
    icd_code: str = None
    severity: str = "moderate"
    diagnosed_by: str = None
    diagnosed_date: datetime = None


# Drug Normalization Models
class DrugNormalizeInput(BaseModel):
    drug_name: str


class DrugNormalizeBatchInput(BaseModel):
    drug_names: List[str]


class DuplicateCheckInput(BaseModel):
    medications: List[str]


class DrugCompareInput(BaseModel):
    drug1: str
    drug2: str


# Drug Safety Models
class SafetyAnalysisInput(BaseModel):
    medications: List[str]
    patient_allergies: List[str] = []
    patient_conditions: List[str] = []
    suppress_low_value: bool = True


class InteractionCheckInput(BaseModel):
    medications: List[str]


class AllergyCheckInput(BaseModel):
    medications: List[str]
    allergies: List[str]


class ContraindicationCheckInput(BaseModel):
    medications: List[str]
    conditions: List[str]


# Temporal Reasoning Models
class PrescriptionInput(BaseModel):
    date: str
    prescriber: str = None
    diagnosis: str = None
    document_id: int = None
    medications: List[Dict[str, Any]]


class TimelineBuildInput(BaseModel):
    prescriptions: List[PrescriptionInput]
    visits: List[Dict[str, Any]] = []
    diagnoses: List[Dict[str, Any]] = []
    vitals: List[Dict[str, Any]] = []


class PrescriptionCompareInput(BaseModel):
    prescription1: Dict[str, Any]
    prescription2: Dict[str, Any]


class CorrectionInput(BaseModel):
    entity_type: str
    entity_id: int
    field_name: str
    original_value: str
    corrected_value: str
    reason: str = None
    corrected_by: str


class DismissAlertInput(BaseModel):
    drug1: str
    drug2: str
    reason: str
    dismissed_by: str
    patient_id: int = None


class QueryInput(BaseModel):
    query: str
    patient_id: int = None


class AuditQueryParams(BaseModel):
    entity_type: str = None
    entity_id: int = None
    action: str = None
    user_name: str = None
    start_date: datetime = None
    end_date: datetime = None
    limit: int = 100


# ==================== Drug Normalization Endpoints ====================

@router.post("/drugs/normalize", tags=["Drug Normalization"])
async def normalize_drug(input_data: DrugNormalizeInput):
    """
    Normalize a drug name to generic form
    
    - Converts brand names to generic names
    - Identifies drug class
    - Returns confidence score
    """
    service = get_drug_normalizer()
    result = service.normalize(input_data.drug_name)
    return {
        "original_name": result.original_name,
        "generic_name": result.generic_name,
        "brand_names": result.brand_names,
        "drug_class": result.drug_class,
        "confidence": result.confidence,
        "is_brand_name": result.is_brand,
        "common_dosages": result.common_dosages
    }


@router.post("/drugs/normalize/batch", tags=["Drug Normalization"])
async def normalize_drugs_batch(input_data: DrugNormalizeBatchInput):
    """
    Normalize multiple drug names in a single request
    
    - Batch processing for efficiency
    - Returns normalized info for each drug
    """
    service = get_drug_normalizer()
    results = []
    for drug_name in input_data.drug_names:
        result = service.normalize(drug_name)
        results.append({
            "original_name": result.original_name,
            "generic_name": result.generic_name,
            "brand_names": result.brand_names,
            "drug_class": result.drug_class,
            "confidence": result.confidence,
            "is_brand_name": result.is_brand,
            "common_dosages": result.common_dosages
        })
    return {"drugs": results, "count": len(results)}


@router.post("/drugs/duplicates", tags=["Drug Normalization"])
async def detect_duplicate_medications(input_data: DuplicateCheckInput):
    """
    Detect duplicate medications (same drug under different names)
    
    - Finds generic equivalents
    - Groups by active ingredient
    - Warns about therapeutic duplication
    """
    service = get_drug_normalizer()
    duplicates = service.detect_duplicates(input_data.medications)
    return {
        "duplicates": duplicates,
        "has_duplicates": len(duplicates) > 0,
        "duplicate_count": len(duplicates)
    }


@router.post("/drugs/compare", tags=["Drug Normalization"])
async def compare_drugs(input_data: DrugCompareInput):
    """
    Check if two drug names refer to the same medication
    
    - Compares generic equivalents
    - Returns match confidence
    """
    service = get_drug_normalizer()
    is_same, confidence = service.are_same_drug(input_data.drug1, input_data.drug2)
    
    norm1 = service.normalize(input_data.drug1)
    norm2 = service.normalize(input_data.drug2)
    
    return {
        "drug1": {
            "original": input_data.drug1,
            "generic": norm1.generic_name,
            "class": norm1.drug_class
        },
        "drug2": {
            "original": input_data.drug2,
            "generic": norm2.generic_name,
            "class": norm2.drug_class
        },
        "is_same_drug": is_same,
        "confidence": confidence,
        "same_class": norm1.drug_class == norm2.drug_class
    }


@router.get("/drugs/alternatives/{drug_name}", tags=["Drug Normalization"])
async def get_therapeutic_alternatives(drug_name: str):
    """
    Get therapeutic alternatives for a drug
    
    - Returns drugs in same therapeutic class
    - Useful for substitution recommendations
    """
    service = get_drug_normalizer()
    alternatives = service.get_therapeutic_alternatives(drug_name)
    normalized = service.normalize(drug_name)
    
    return {
        "drug": drug_name,
        "generic_name": normalized.generic_name,
        "drug_class": normalized.drug_class,
        "alternatives": alternatives,
        "alternatives_count": len(alternatives)
    }


@router.get("/drugs/class/{drug_name}", tags=["Drug Normalization"])
async def get_drug_class(drug_name: str):
    """
    Get the drug class for a medication
    
    - Returns therapeutic classification
    - Useful for categorization
    """
    service = get_drug_normalizer()
    drug_class = service.get_drug_class(drug_name)
    normalized = service.normalize(drug_name)
    
    return {
        "drug": drug_name,
        "generic_name": normalized.generic_name,
        "drug_class": drug_class,
        "therapeutic_class": normalized.drug_class,
        "confidence": normalized.confidence
    }


@router.get("/drugs/standardize/{drug_name}", tags=["Drug Normalization"])
async def standardize_drug_name(drug_name: str):
    """
    Get standardized (generic) name for a drug
    
    - Returns preferred generic name
    - Useful for data normalization
    """
    service = get_drug_normalizer()
    standardized = service.standardize_name(drug_name)
    normalized = service.normalize(drug_name)
    
    return {
        "original_name": drug_name,
        "standardized_name": standardized,
        "is_brand_name": normalized.is_brand,
        "confidence": normalized.confidence
    }


# ==================== Drug Safety Analysis Endpoints ====================

@router.post("/drugs/safety/analyze", tags=["Drug Safety"])
async def analyze_drug_safety(input_data: SafetyAnalysisInput):
    """
    Complete drug safety analysis
    
    - Checks drug-drug interactions
    - Checks patient allergies
    - Checks contraindications against conditions
    - Detects duplicate therapies
    - Returns severity classification
    """
    service = get_drug_interaction_service()
    result = service.analyze_safety(
        medications=input_data.medications,
        patient_allergies=input_data.patient_allergies,
        patient_conditions=input_data.patient_conditions,
        suppress_low_value=input_data.suppress_low_value
    )
    return service.to_dict(result)


@router.post("/drugs/interactions/check", tags=["Drug Safety"])
async def check_drug_interactions(input_data: InteractionCheckInput):
    """
    Check for drug-drug interactions only
    
    - Specific drug pair interactions (30+ pairs)
    - Class-level interactions (NSAIDs + Anticoagulants, etc.)
    - Severity classification (Minor/Moderate/Major/Contraindicated)
    """
    service = get_drug_interaction_service()
    result = service.analyze_safety(
        medications=input_data.medications,
        patient_allergies=[],
        patient_conditions=[],
        suppress_low_value=False
    )
    
    interactions = [
        {
            "drug1": i.drug1,
            "drug2": i.drug2,
            "severity": i.severity.value,
            "description": i.description,
            "mechanism": i.mechanism,
            "clinical_effects": i.clinical_effects,
            "management": i.management,
            "evidence_level": i.evidence_level
        }
        for i in result.interactions
    ]
    
    return {
        "interactions": interactions,
        "interaction_count": len(interactions),
        "has_major_interactions": any(i["severity"] in ["major", "contraindicated"] for i in interactions),
        "has_contraindicated": any(i["severity"] == "contraindicated" for i in interactions)
    }


@router.post("/drugs/allergies/check", tags=["Drug Safety"])
async def check_allergy_risks(input_data: AllergyCheckInput):
    """
    Check medication-allergy conflicts
    
    - Direct allergen matching
    - Cross-reactivity detection (e.g., penicillin cross-sensitivity)
    - Suggests alternatives
    """
    service = get_drug_interaction_service()
    result = service.analyze_safety(
        medications=input_data.medications,
        patient_allergies=input_data.allergies,
        patient_conditions=[],
        suppress_low_value=False
    )
    
    allergy_alerts = [
        {
            "drug": a.drug,
            "allergen": a.allergen,
            "risk_type": a.risk_type,
            "severity": a.severity.value,
            "description": a.description,
            "alternatives": a.alternatives
        }
        for a in result.allergy_alerts
    ]
    
    return {
        "allergy_alerts": allergy_alerts,
        "alert_count": len(allergy_alerts),
        "has_contraindicated": any(a["severity"] == "contraindicated" for a in allergy_alerts)
    }


@router.post("/drugs/contraindications/check", tags=["Drug Safety"])
async def check_contraindications(input_data: ContraindicationCheckInput):
    """
    Check drug-condition contraindications
    
    - Checks against patient conditions
    - Returns contraindicated and caution-level findings
    """
    service = get_drug_interaction_service()
    result = service.analyze_safety(
        medications=input_data.medications,
        patient_allergies=[],
        patient_conditions=input_data.conditions,
        suppress_low_value=False
    )
    
    return {
        "contraindications": result.contraindications,
        "contraindication_count": len(result.contraindications),
        "has_absolute_contraindications": any(c["level"] == "contraindicated" for c in result.contraindications)
    }


@router.post("/drugs/duplicates/therapy", tags=["Drug Safety"])
async def check_duplicate_therapy(input_data: InteractionCheckInput):
    """
    Check for duplicate therapy (same class medications)
    
    - Detects multiple drugs in same therapeutic class
    - Identifies same generic drug under different names
    """
    service = get_drug_interaction_service()
    result = service.analyze_safety(
        medications=input_data.medications,
        patient_allergies=[],
        patient_conditions=[],
        suppress_low_value=False
    )
    
    duplicate_therapies = [
        {
            "drugs": d.drugs,
            "drug_class": d.drug_class,
            "description": d.description,
            "recommendation": d.recommendation
        }
        for d in result.duplicate_therapies
    ]
    
    return {
        "duplicate_therapies": duplicate_therapies,
        "duplicate_count": len(duplicate_therapies),
        "has_duplicates": len(duplicate_therapies) > 0
    }


@router.get("/drugs/safety/risk-levels", tags=["Drug Safety"])
async def get_risk_level_info():
    """
    Get information about risk levels and severity classifications
    """
    return {
        "severity_levels": [
            {
                "level": "minor",
                "description": "Minimal clinical significance, monitor if needed",
                "action": "Be aware, usually no action required"
            },
            {
                "level": "moderate", 
                "description": "May require dosage adjustment or monitoring",
                "action": "Monitor patient, consider alternatives"
            },
            {
                "level": "major",
                "description": "Significant clinical consequences possible",
                "action": "Avoid combination or use extreme caution with close monitoring"
            },
            {
                "level": "contraindicated",
                "description": "Life-threatening or absolutely contraindicated",
                "action": "Do not use together, find alternatives"
            }
        ],
        "overall_risk_levels": [
            {"level": "MINIMAL", "description": "No significant concerns identified"},
            {"level": "LOW", "description": "Minor interactions or duplicates detected"},
            {"level": "MODERATE", "description": "One major interaction identified"},
            {"level": "HIGH", "description": "Multiple major interactions identified"},
            {"level": "CRITICAL", "description": "Contraindicated combination detected"}
        ]
    }


# ==================== Temporal Reasoning Endpoints ====================

@router.post("/temporal/build-timeline", tags=["Temporal Reasoning"])
async def build_medication_timeline(input_data: TimelineBuildInput):
    """
    Build comprehensive medical timeline from prescriptions
    
    - Creates chronological event timeline
    - Calculates medication periods
    - Detects medication changes
    - Identifies overlapping medications
    """
    service = get_temporal_service()
    
    # Convert Pydantic models to dicts
    prescriptions = [p.model_dump() for p in input_data.prescriptions]
    
    result = service.build_timeline(
        prescriptions=prescriptions,
        visits=input_data.visits,
        diagnoses=input_data.diagnoses,
        vitals=input_data.vitals
    )
    
    return service.to_dict(result)


@router.post("/temporal/compare-prescriptions", tags=["Temporal Reasoning"])
async def compare_prescriptions(input_data: PrescriptionCompareInput):
    """
    Compare two prescriptions to identify changes
    
    - Finds new medications
    - Finds discontinued medications
    - Finds dosage/frequency changes
    """
    service = get_temporal_service()
    comparison = service.compare_prescriptions(
        prescription1=input_data.prescription1,
        prescription2=input_data.prescription2
    )
    return comparison


@router.post("/temporal/medication-changes", tags=["Temporal Reasoning"])
async def get_medication_changes(input_data: TimelineBuildInput):
    """
    Get detected medication changes over time
    
    - Started medications
    - Stopped medications
    - Dosage changes
    - Frequency changes
    """
    service = get_temporal_service()
    prescriptions = [p.model_dump() for p in input_data.prescriptions]
    
    result = service.build_timeline(prescriptions=prescriptions)
    
    changes = [
        {
            "medication_name": c.medication_name,
            "change_type": c.change_type,
            "old_value": c.old_value,
            "new_value": c.new_value,
            "change_date": str(c.change_date),
            "confidence": c.confidence
        }
        for c in result.medication_changes
    ]
    
    # Group by change type
    grouped = {
        "started": [c for c in changes if c["change_type"] == "started"],
        "stopped": [c for c in changes if c["change_type"] == "stopped"],
        "dose_changed": [c for c in changes if c["change_type"] == "dose_changed"],
        "frequency_changed": [c for c in changes if c["change_type"] == "frequency_changed"]
    }
    
    return {
        "all_changes": changes,
        "grouped_changes": grouped,
        "total_changes": len(changes)
    }


@router.post("/temporal/overlaps", tags=["Temporal Reasoning"])
async def get_medication_overlaps(input_data: TimelineBuildInput):
    """
    Find overlapping medication periods
    
    - Identifies concurrent medications
    - Calculates overlap duration
    - Flags significant overlaps
    """
    service = get_temporal_service()
    prescriptions = [p.model_dump() for p in input_data.prescriptions]
    
    result = service.build_timeline(prescriptions=prescriptions)
    
    overlaps = [
        {
            "medication1": o.medication1,
            "medication2": o.medication2,
            "overlap_start": str(o.overlap_start),
            "overlap_end": str(o.overlap_end),
            "duration_days": o.duration_days,
            "is_significant": o.is_significant,
            "notes": o.notes
        }
        for o in result.overlapping_medications
    ]
    
    return {
        "overlaps": overlaps,
        "overlap_count": len(overlaps),
        "significant_overlaps": [o for o in overlaps if o["is_significant"]]
    }


@router.post("/temporal/current-medications", tags=["Temporal Reasoning"])
async def get_current_medications(input_data: TimelineBuildInput):
    """
    Get currently active medications vs historical
    
    - Lists active medications
    - Lists historical (discontinued) medications
    """
    service = get_temporal_service()
    prescriptions = [p.model_dump() for p in input_data.prescriptions]
    
    result = service.build_timeline(prescriptions=prescriptions)
    
    return {
        "current_medications": result.current_medications,
        "historical_medications": result.historical_medications,
        "current_count": len(result.current_medications),
        "historical_count": len(result.historical_medications)
    }


@router.post("/temporal/medication-periods", tags=["Temporal Reasoning"])
async def get_medication_periods(input_data: TimelineBuildInput):
    """
    Get medication periods with start/end dates
    
    - Calculates medication duration
    - Identifies ongoing medications
    """
    service = get_temporal_service()
    prescriptions = [p.model_dump() for p in input_data.prescriptions]
    
    result = service.build_timeline(prescriptions=prescriptions)
    
    periods = [
        {
            "medication_name": p.medication_name,
            "generic_name": p.generic_name,
            "start_date": str(p.start_date),
            "end_date": str(p.end_date) if p.end_date else None,
            "dosage": p.dosage,
            "frequency": p.frequency,
            "is_ongoing": p.is_ongoing,
            "confidence": p.confidence
        }
        for p in result.medication_periods
    ]
    
    return {
        "medication_periods": periods,
        "total_periods": len(periods),
        "ongoing_count": len([p for p in periods if p["is_ongoing"]])
    }


# ==================== Combined Analysis Endpoints ====================

@router.post("/drugs/comprehensive-analysis", tags=["Combined Analysis"])
async def comprehensive_drug_analysis(
    medications: List[str],
    patient_allergies: List[str] = [],
    patient_conditions: List[str] = []
):
    """
    Comprehensive drug analysis combining normalization, safety, and duplicates
    
    - Normalizes all drug names
    - Checks all interactions
    - Checks allergies and contraindications
    - Detects duplicates
    - Returns complete analysis
    """
    normalizer = get_drug_normalizer()
    interaction_service = get_drug_interaction_service()
    
    # Normalize all medications
    normalized = []
    for med in medications:
        norm = normalizer.normalize(med)
        normalized.append({
            "original": med,
            "generic": norm.generic_name,
            "drug_class": norm.drug_class,
            "confidence": norm.confidence,
            "is_brand": norm.is_brand
        })
    
    # Check duplicates
    duplicates = normalizer.detect_duplicates(medications)
    
    # Full safety analysis
    safety = interaction_service.analyze_safety(
        medications=medications,
        patient_allergies=patient_allergies,
        patient_conditions=patient_conditions
    )
    safety_dict = interaction_service.to_dict(safety)
    
    return {
        "normalized_medications": normalized,
        "duplicates": duplicates,
        "safety_analysis": safety_dict,
        "summary": {
            "total_medications": len(medications),
            "unique_generics": len(set(n["generic"] for n in normalized)),
            "interaction_count": len(safety.interactions),
            "allergy_alerts": len(safety.allergy_alerts),
            "contraindications": len(safety.contraindications),
            "duplicate_therapies": len(safety.duplicate_therapies),
            "overall_risk": safety.overall_risk_level
        }
    }


# ==================== Patient History Endpoints ====================

@router.post("/patients/{patient_id}/medications")
async def add_patient_medication(
    patient_id: int,
    medication: MedicationInput,
    db: Session = Depends(get_db)
):
    """Add medication to patient history"""
    service = PatientHistoryService(db)
    result = service.add_medication(
        patient_id=patient_id,
        medication_name=medication.medication_name,
        dosage=medication.dosage,
        frequency=medication.frequency,
        prescribing_doctor=medication.prescribing_doctor,
        indication=medication.indication,
        start_date=medication.start_date
    )
    return {"status": "success", "medication_id": result.id}


@router.post("/patients/{patient_id}/medications/{medication_id}/stop")
async def stop_patient_medication(
    patient_id: int,
    medication_id: int,
    reason: str = None,
    db: Session = Depends(get_db)
):
    """Stop a medication"""
    service = PatientHistoryService(db)
    result = service.stop_medication(medication_id, reason)
    if result:
        return {"status": "success", "message": "Medication stopped"}
    raise HTTPException(status_code=404, detail="Medication not found")


@router.get("/patients/{patient_id}/medications")
async def get_patient_medications(
    patient_id: int,
    active_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get patient medications"""
    service = PatientHistoryService(db)
    medications = service.get_medication_history(patient_id, active_only=active_only)
    return {
        "patient_id": patient_id,
        "medications": [
            {
                "id": m.id,
                "name": m.medication_name,
                "dosage": m.dosage,
                "frequency": m.frequency,
                "active": m.active,
                "start_date": str(m.start_date) if m.start_date else None,
                "end_date": str(m.end_date) if m.end_date else None
            }
            for m in medications
        ]
    }


@router.post("/patients/{patient_id}/conditions")
async def add_patient_condition(
    patient_id: int,
    condition: ConditionInput,
    db: Session = Depends(get_db)
):
    """Add condition to patient history"""
    service = PatientHistoryService(db)
    result = service.add_condition(
        patient_id=patient_id,
        condition_name=condition.condition_name,
        icd_code=condition.icd_code,
        severity=condition.severity,
        diagnosed_by=condition.diagnosed_by,
        diagnosed_date=condition.diagnosed_date
    )
    return {"status": "success", "condition_id": result.id}


@router.get("/patients/{patient_id}/history")
async def get_patient_history(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get complete patient history"""
    service = PatientHistoryService(db)
    history = service.get_complete_patient_history(patient_id)
    return history


# ==================== Knowledge Graph Endpoints ====================

@router.get("/knowledge-graph/patient/{patient_id}")
async def get_patient_graph(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get knowledge graph for a patient"""
    service = KnowledgeGraphService(db)
    
    # First, find the patient node by external_id (which stores the actual patient ID)
    from backend.models.knowledge_graph import KnowledgeNode, NodeType
    from sqlalchemy import and_
    
    patient_node = db.query(KnowledgeNode).filter(
        and_(
            KnowledgeNode.node_type == NodeType.PATIENT,
            KnowledgeNode.external_id == str(patient_id)
        )
    ).first()
    
    if not patient_node:
        # Return empty graph if no data yet
        return {"nodes": [], "edges": [], "message": "No knowledge graph data for this patient yet"}
    
    graph = service.get_patient_graph(patient_node.id)
    return graph


@router.get("/knowledge-graph/stats")
async def get_graph_statistics(db: Session = Depends(get_db)):
    """Get knowledge graph statistics"""
    service = KnowledgeGraphService(db)
    stats = service.get_graph_statistics()
    return stats


@router.post("/knowledge-graph/link/medication")
async def link_patient_medication(
    patient_id: int,
    medication_name: str,
    relationship_type: str = "is_taking",
    db: Session = Depends(get_db)
):
    """Link patient to medication in graph"""
    service = KnowledgeGraphService(db)
    edge = service.link_patient_medication(patient_id, medication_name, relationship_type)
    return {"status": "success", "edge_id": edge.id if edge else None}


@router.post("/knowledge-graph/link/condition")
async def link_patient_condition(
    patient_id: int,
    condition_name: str,
    db: Session = Depends(get_db)
):
    """Link patient to condition in graph"""
    service = KnowledgeGraphService(db)
    edge = service.link_patient_condition(patient_id, condition_name)
    return {"status": "success", "edge_id": edge.id if edge else None}


# ==================== Temporal Reasoning Endpoints ====================

@router.get("/temporal/patient/{patient_id}/timeline")
async def get_patient_timeline(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get patient timeline"""
    service = EnhancedTemporalReasoningService(db)
    timeline = service.build_patient_timeline(patient_id)
    return {
        "patient_id": patient_id,
        "events": [
            {
                "date": str(e.event_date),
                "type": e.event_type,
                "description": e.description,
                "entity_id": e.entity_id
            }
            for e in timeline
        ]
    }


@router.get("/temporal/patient/{patient_id}/medication-timeline")
async def get_medication_timeline(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get medication-specific timeline"""
    service = EnhancedTemporalReasoningService(db)
    timeline = service.get_medication_timeline(patient_id)
    return {
        "patient_id": patient_id,
        "medications": timeline
    }


@router.get("/temporal/patient/{patient_id}/overlaps")
async def get_medication_overlaps(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get medication overlap periods"""
    service = EnhancedTemporalReasoningService(db)
    overlaps = service.detect_medication_overlaps(patient_id)
    return {
        "patient_id": patient_id,
        "overlaps": [
            {
                "medications": o.medications,
                "start": str(o.start_date),
                "end": str(o.end_date) if o.end_date else "ongoing",
                "duration_days": o.duration_days
            }
            for o in overlaps
        ]
    }


@router.get("/temporal/patient/{patient_id}/gantt")
async def get_gantt_chart_data(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get Gantt chart data for visualization"""
    service = EnhancedTemporalReasoningService(db)
    gantt_data = service.get_gantt_chart_data(patient_id)
    return gantt_data


@router.get("/temporal/patient/{patient_id}/summary")
async def get_timeline_summary(
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get timeline summary"""
    service = EnhancedTemporalReasoningService(db)
    summary = service.get_timeline_summary(patient_id)
    return summary


# ==================== Explainability Endpoints ====================

@router.get("/explain/extraction/{document_id}")
async def explain_extraction(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get explanation for document extraction"""
    service = ExplainabilityService(db)
    explanation = service.explain_extraction(document_id)
    return explanation


@router.get("/explain/interaction")
async def explain_drug_interaction(
    drug1: str,
    drug2: str,
    db: Session = Depends(get_db)
):
    """Get explanation for drug interaction"""
    service = ExplainabilityService(db)
    explanation = service.explain_drug_interaction(drug1, drug2)
    return explanation


@router.get("/explain/extraction/{document_id}/summary")
async def get_extraction_summary(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get extraction summary with confidence"""
    service = ExplainabilityService(db)
    summary = service.get_extraction_summary(document_id)
    return summary


# ==================== Uncertainty & Risk Endpoints ====================

@router.get("/risk/extraction/{document_id}")
async def get_extraction_risk(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get risk assessment for extraction"""
    service = UncertaintyService(db)
    # This would need integration with stored extraction data
    assessment = service.assess_risk()
    return service.to_dict()


# ==================== Human Review Endpoints ====================

@router.get("/review/queue")
async def get_review_queue(
    confidence_threshold: float = 0.7,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get items pending human review"""
    service = HumanReviewService(db)
    items = service.get_items_for_review(confidence_threshold, limit)
    return {"items": items, "count": len(items)}


@router.post("/review/correct")
async def submit_correction(
    correction: CorrectionInput,
    db: Session = Depends(get_db)
):
    """Submit a correction"""
    service = HumanReviewService(db)
    result = service.correct_entity(
        entity_type=correction.entity_type,
        entity_id=correction.entity_id,
        field_name=correction.field_name,
        original_value=correction.original_value,
        corrected_value=correction.corrected_value,
        corrected_by=correction.corrected_by,
        reason=correction.reason
    )
    return {"status": "success", "correction_id": result.id}


@router.post("/review/dismiss-alert")
async def dismiss_alert(
    alert: DismissAlertInput,
    db: Session = Depends(get_db)
):
    """Dismiss a drug interaction alert"""
    service = HumanReviewService(db)
    result = service.dismiss_interaction_alert(
        drug1=alert.drug1,
        drug2=alert.drug2,
        dismissed_by=alert.dismissed_by,
        reason=alert.reason,
        patient_id=alert.patient_id
    )
    return {"status": "success", "dismissal_id": result.id}


@router.get("/review/corrections")
async def get_corrections(
    document_id: int = None,
    user: str = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get correction history"""
    service = HumanReviewService(db)
    
    if document_id:
        corrections = service.get_corrections_for_document(document_id)
    elif user:
        corrections = service.get_corrections_by_user(user, limit)
    else:
        corrections = service.get_unverified_corrections(limit)
    
    return {
        "corrections": [
            {
                "id": c.id,
                "type": c.correction_type.value if c.correction_type else None,
                "field": c.field_name,
                "original": c.original_value,
                "corrected": c.corrected_value,
                "by": c.corrected_by,
                "at": str(c.corrected_at),
                "verified": c.verified
            }
            for c in corrections
        ]
    }


@router.get("/review/statistics")
async def get_correction_statistics(db: Session = Depends(get_db)):
    """Get correction statistics"""
    service = HumanReviewService(db)
    stats = service.get_correction_statistics()
    return stats


# ==================== Conversational Query Endpoints ====================

@router.post("/query")
async def process_query(
    query_input: QueryInput,
    db: Session = Depends(get_db)
):
    """Process a natural language query"""
    service = ConversationalQueryService(db)
    response = service.process_query(
        query=query_input.query,
        patient_id=query_input.patient_id
    )
    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "evidence": response.evidence,
        "related_queries": response.related_queries,
        "visualization": response.visualization_data
    }


@router.post("/query/batch")
async def batch_query(
    queries: List[str],
    patient_id: int = None,
    db: Session = Depends(get_db)
):
    """Process multiple queries"""
    service = ConversationalQueryService(db)
    responses = service.batch_query(queries, patient_id)
    return {
        "responses": [
            {
                "query": q,
                "answer": r.answer,
                "confidence": r.confidence
            }
            for q, r in zip(queries, responses)
        ]
    }


# ==================== Compliance & Audit Endpoints ====================

@router.get("/audit/trail")
async def get_audit_trail(
    entity_type: str = None,
    entity_id: int = None,
    action: str = None,
    user_name: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get audit trail with filters"""
    from backend.models.audit import AuditAction
    
    service = ComplianceService(db)
    
    action_enum = None
    if action:
        try:
            action_enum = AuditAction(action)
        except ValueError:
            pass
    
    logs = service.get_audit_trail(
        entity_type=entity_type,
        entity_id=entity_id,
        action=action_enum,
        user_name=user_name,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    
    return {
        "logs": [
            {
                "id": log.id,
                "timestamp": str(log.timestamp),
                "action": log.action.value if log.action else None,
                "action_detail": log.action_detail,
                "entity_type": log.entity_type,
                "entity_id": log.entity_id,
                "user": log.user_name,
                "changes": log.changes
            }
            for log in logs
        ],
        "count": len(logs)
    }


@router.get("/audit/document/{document_id}")
async def get_document_audit_trail(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get audit trail for a specific document"""
    service = ComplianceService(db)
    logs = service.get_document_audit_trail(document_id)
    return {
        "document_id": document_id,
        "audit_trail": [
            {
                "timestamp": str(log.timestamp),
                "action": log.action.value if log.action else None,
                "action_detail": log.action_detail,
                "user": log.user_name
            }
            for log in logs
        ]
    }


@router.get("/audit/user/{user_name}")
async def get_user_activity(
    user_name: str,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get user activity for compliance review"""
    service = ComplianceService(db)
    logs = service.get_user_activity(user_name, days)
    return {
        "user": user_name,
        "period_days": days,
        "activity": [
            {
                "timestamp": str(log.timestamp),
                "action": log.action.value if log.action else None,
                "entity_type": log.entity_type,
                "entity_id": log.entity_id
            }
            for log in logs
        ],
        "total_actions": len(logs)
    }


@router.get("/compliance/report")
async def get_compliance_report(
    start_date: datetime = None,
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Generate compliance report"""
    service = ComplianceService(db)
    report = service.generate_compliance_report(start_date, end_date)
    return report


@router.get("/compliance/access-report")
async def get_access_report(
    document_id: int = None,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Generate access report"""
    service = ComplianceService(db)
    report = service.generate_access_report(document_id, days)
    return report


@router.get("/compliance/integrity")
async def verify_audit_integrity(
    limit: int = 1000,
    db: Session = Depends(get_db)
):
    """Verify audit chain integrity"""
    service = ComplianceService(db)
    result = service.verify_audit_chain(limit)
    return result


@router.get("/compliance/export")
async def export_audit_trail(
    start_date: datetime = None,
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Export audit trail for external audit"""
    service = ComplianceService(db)
    export = service.export_audit_trail(start_date, end_date, format="list")
    return {"audit_records": export, "count": len(export)}


# ==================== Document Version Endpoints ====================

@router.get("/documents/{document_id}/versions")
async def get_document_versions(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get all versions of a document"""
    service = ComplianceService(db)
    versions = service.get_document_versions(document_id)
    return {
        "document_id": document_id,
        "versions": [
            {
                "version": v.version_number,
                "created_at": str(v.created_at),
                "created_by": v.created_by,
                "content_hash": v.content_hash,
                "content_type": v.content_type
            }
            for v in versions
        ]
    }


@router.get("/documents/{document_id}/verify")
async def verify_document_integrity(
    document_id: int,
    version: int = None,
    db: Session = Depends(get_db)
):
    """Verify document integrity"""
    service = ComplianceService(db)
    result = service.verify_document_integrity(document_id, version)
    return result
