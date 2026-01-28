"""
Analytics API Routes - Dashboard and visualization data
"""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict

from backend.api.patients import patient_store
from backend.services import KnowledgeGraphService, AuditService

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/dashboard")
async def get_dashboard_stats():
    """
    Get overview statistics for the dashboard
    """
    total_patients = len(patient_store)
    total_medications = 0
    total_interactions = 0
    total_allergies = 0
    
    severity_counts = defaultdict(int)
    medication_counts = defaultdict(int)
    
    for patient_id, patient in patient_store.items():
        # Count medications
        current_meds = patient.get('current_medications', [])
        total_medications += len(current_meds)
        
        for med in current_meds:
            med_name = med.get('medication_name', 'Unknown')
            medication_counts[med_name] += 1
        
        # Count allergies
        total_allergies += len(patient.get('allergies', []))
    
    # Top medications
    top_medications = sorted(
        medication_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    return {
        'overview': {
            'total_patients': total_patients,
            'total_medications': total_medications,
            'total_allergies': total_allergies,
            'avg_medications_per_patient': total_medications / max(total_patients, 1)
        },
        'top_medications': [
            {'name': name, 'count': count} 
            for name, count in top_medications
        ],
        'processing_stats': {
            'documents_processed_today': 0,
            'avg_processing_time_ms': 0,
            'success_rate': 1.0
        },
        'safety_alerts': {
            'total_interactions_detected': total_interactions,
            'by_severity': dict(severity_counts)
        }
    }


@router.get("/medications/frequency")
async def get_medication_frequency():
    """
    Get frequency distribution of medications across all patients
    """
    medication_counts = defaultdict(int)
    drug_classes = defaultdict(int)
    
    for patient in patient_store.values():
        for med in patient.get('current_medications', []):
            name = med.get('medication_name', 'Unknown')
            medication_counts[name] += 1
    
    return {
        'medications': sorted(
            [{'name': k, 'count': v} for k, v in medication_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        ),
        'total_unique': len(medication_counts)
    }


@router.get("/interactions/summary")
async def get_interactions_summary():
    """
    Get summary of drug interactions across the system
    """
    from backend.services import DrugInteractionService
    
    interaction_service = DrugInteractionService()
    
    all_interactions = []
    severity_counts = defaultdict(int)
    
    for patient_id, patient in patient_store.items():
        current_meds = patient.get('current_medications', [])
        allergies = patient.get('allergies', [])
        
        if current_meds:
            med_names = [m.get('medication_name') for m in current_meds if m.get('medication_name')]
            result = interaction_service.analyze_medications(med_names, allergies)
            
            for interaction in result.interactions:
                severity_counts[interaction.severity.value] += 1
                all_interactions.append({
                    'patient_id': patient_id,
                    'drug1': interaction.drug1,
                    'drug2': interaction.drug2,
                    'severity': interaction.severity.value
                })
    
    return {
        'total_interactions': len(all_interactions),
        'by_severity': dict(severity_counts),
        'recent_interactions': all_interactions[:20]
    }


@router.get("/timeline/activity")
async def get_activity_timeline(
    days: int = Query(30, description="Number of days to look back")
):
    """
    Get activity timeline for the system
    """
    # This would typically come from the audit log
    audit = AuditService()
    
    # Generate sample activity data
    activity = []
    for i in range(days):
        date = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
        activity.append({
            'date': date,
            'documents_processed': 0,
            'patients_created': 0,
            'corrections_made': 0
        })
    
    return {
        'period_days': days,
        'activity': activity
    }


@router.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats():
    """
    Get statistics about the knowledge graph
    """
    kg = KnowledgeGraphService()
    stats = kg.get_statistics()
    
    return {
        'graph_stats': stats,
        'node_types': [
            'patient', 'medication', 'condition', 
            'symptom', 'prescription', 'document'
        ],
        'relationship_types': [
            'TAKES', 'HAS_CONDITION', 'TREATS', 
            'CAUSES', 'EXTRACTED_FROM', 'EXPERIENCES'
        ]
    }


@router.get("/confidence/distribution")
async def get_confidence_distribution():
    """
    Get distribution of confidence scores across processed entities
    """
    # This would aggregate from actual processing results
    return {
        'ocr_confidence': {
            'average': 0.85,
            'distribution': [
                {'range': '0-50%', 'count': 5},
                {'range': '50-70%', 'count': 15},
                {'range': '70-90%', 'count': 45},
                {'range': '90-100%', 'count': 35}
            ]
        },
        'entity_extraction_confidence': {
            'average': 0.82,
            'by_entity_type': {
                'medication': 0.88,
                'dosage': 0.75,
                'diagnosis': 0.80
            }
        }
    }


@router.get("/review-queue")
async def get_review_queue():
    """
    Get items flagged for human review
    """
    # This would come from actual flagged items
    return {
        'pending_review': 0,
        'items': [],
        'by_reason': {
            'low_ocr_confidence': 0,
            'drug_interactions': 0,
            'handwritten_content': 0,
            'manual_request': 0
        }
    }


@router.get("/export/summary")
async def export_summary(
    format: str = Query("json", description="Export format: json or csv")
):
    """
    Export summary data for reporting
    """
    summary = {
        'export_date': datetime.utcnow().isoformat(),
        'total_patients': len(patient_store),
        'patients': []
    }
    
    for patient_id, patient in patient_store.items():
        summary['patients'].append({
            'patient_id': patient_id,
            'medication_count': len(patient.get('current_medications', [])),
            'allergy_count': len(patient.get('allergies', [])),
            'condition_count': len(patient.get('chronic_conditions', []))
        })
    
    return summary
