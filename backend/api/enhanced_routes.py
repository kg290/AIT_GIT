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

router = APIRouter(prefix="/api/v2", tags=["Enhanced API"])


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
