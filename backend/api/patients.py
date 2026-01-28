"""
Patient API Routes
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime

from backend.services import (
    KnowledgeGraphService,
    TemporalReasoningService,
    DrugInteractionService
)

router = APIRouter(prefix="/patients", tags=["Patients"])


# In-memory patient store (would be database in production)
patient_store = {}


@router.post("/")
async def create_patient(
    patient_id: str,
    name: Optional[str] = None,
    date_of_birth: Optional[str] = None,
    allergies: Optional[List[str]] = None,
    chronic_conditions: Optional[List[str]] = None
):
    """
    Create or update a patient record
    """
    patient_data = {
        'patient_id': patient_id,
        'name': name,
        'date_of_birth': date_of_birth,
        'allergies': allergies or [],
        'chronic_conditions': chronic_conditions or [],
        'current_medications': [],
        'historical_medications': [],
        'prescriptions': [],
        'diagnoses': [],
        'symptoms': [],
        'vitals': [],
        'timeline': [],
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    }
    
    if patient_id in patient_store:
        # Update existing
        existing = patient_store[patient_id]
        existing.update({k: v for k, v in patient_data.items() if v is not None})
        existing['updated_at'] = datetime.utcnow().isoformat()
        patient_store[patient_id] = existing
    else:
        # Create new
        patient_store[patient_id] = patient_data
    
    # Update knowledge graph
    kg = KnowledgeGraphService()
    kg.create_patient_node(patient_id, {'name': name})
    
    if chronic_conditions:
        for condition in chronic_conditions:
            kg.link_patient_condition(patient_id, condition)
    
    return patient_store[patient_id]


@router.get("/{patient_id}")
async def get_patient(patient_id: str):
    """
    Get patient information and medical summary
    """
    if patient_id not in patient_store:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient_store[patient_id]


@router.get("/{patient_id}/medications")
async def get_patient_medications(
    patient_id: str,
    current_only: bool = Query(True, description="Only show current medications")
):
    """
    Get patient's medication list
    """
    if patient_id not in patient_store:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = patient_store[patient_id]
    
    if current_only:
        return {
            'patient_id': patient_id,
            'current_medications': patient.get('current_medications', [])
        }
    
    return {
        'patient_id': patient_id,
        'current_medications': patient.get('current_medications', []),
        'historical_medications': patient.get('historical_medications', [])
    }


@router.post("/{patient_id}/medications")
async def add_patient_medication(
    patient_id: str,
    medication_name: str,
    dosage: Optional[str] = None,
    frequency: Optional[str] = None,
    route: Optional[str] = None,
    start_date: Optional[str] = None,
    prescriber: Optional[str] = None
):
    """
    Add a medication to patient's record
    """
    if patient_id not in patient_store:
        patient_store[patient_id] = {
            'patient_id': patient_id,
            'current_medications': [],
            'historical_medications': [],
            'allergies': [],
            'chronic_conditions': [],
            'prescriptions': [],
            'diagnoses': [],
            'timeline': []
        }
    
    medication = {
        'medication_name': medication_name,
        'dosage': dosage,
        'frequency': frequency,
        'route': route,
        'start_date': start_date or datetime.utcnow().date().isoformat(),
        'prescriber': prescriber,
        'is_current': True
    }
    
    patient_store[patient_id]['current_medications'].append(medication)
    patient_store[patient_id]['updated_at'] = datetime.utcnow().isoformat()
    
    # Update knowledge graph
    kg = KnowledgeGraphService()
    kg.link_patient_medication(patient_id, medication_name)
    
    return {'success': True, 'medication': medication}


@router.delete("/{patient_id}/medications/{medication_name}")
async def stop_patient_medication(
    patient_id: str,
    medication_name: str,
    reason: Optional[str] = None
):
    """
    Stop/discontinue a medication
    """
    if patient_id not in patient_store:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = patient_store[patient_id]
    current_meds = patient.get('current_medications', [])
    
    # Find and move medication to historical
    found = False
    for med in current_meds:
        if med.get('medication_name', '').lower() == medication_name.lower():
            med['is_current'] = False
            med['end_date'] = datetime.utcnow().date().isoformat()
            med['discontinuation_reason'] = reason
            
            patient.setdefault('historical_medications', []).append(med)
            current_meds.remove(med)
            found = True
            break
    
    if not found:
        raise HTTPException(status_code=404, detail="Medication not found in current list")
    
    patient_store[patient_id]['updated_at'] = datetime.utcnow().isoformat()
    
    return {'success': True, 'message': f'{medication_name} discontinued'}


@router.get("/{patient_id}/interactions")
async def check_patient_interactions(patient_id: str):
    """
    Check drug interactions for patient's current medications
    """
    if patient_id not in patient_store:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = patient_store[patient_id]
    current_meds = patient.get('current_medications', [])
    allergies = patient.get('allergies', [])
    
    if not current_meds:
        return {
            'patient_id': patient_id,
            'interactions': [],
            'allergy_alerts': [],
            'safety_score': 1.0,
            'message': 'No current medications to check'
        }
    
    # Get medication names
    med_names = [m.get('medication_name') for m in current_meds if m.get('medication_name')]
    
    # Check interactions
    interaction_service = DrugInteractionService()
    result = interaction_service.analyze_medications(med_names, allergies)
    
    return {
        'patient_id': patient_id,
        'medications_checked': med_names,
        'interactions': [
            {
                'drug1': i.drug1,
                'drug2': i.drug2,
                'severity': i.severity.value,
                'description': i.description,
                'management': i.management
            }
            for i in result.interactions
        ],
        'allergy_alerts': [
            {
                'drug': a.drug,
                'allergen': a.allergen,
                'severity': a.severity.value,
                'description': a.description
            }
            for a in result.allergy_alerts
        ],
        'duplicate_therapies': [
            {
                'drug1': d.drug1,
                'drug2': d.drug2,
                'reason': d.reason
            }
            for d in result.duplicate_therapies
        ],
        'safety_score': result.overall_safety_score
    }


@router.get("/{patient_id}/timeline")
async def get_patient_timeline(
    patient_id: str,
    limit: int = Query(50, description="Maximum events to return")
):
    """
    Get patient's medical timeline
    """
    if patient_id not in patient_store:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = patient_store[patient_id]
    timeline = patient.get('timeline', [])
    
    # Sort by date, most recent first
    sorted_timeline = sorted(
        timeline, 
        key=lambda x: x.get('event_date', ''), 
        reverse=True
    )[:limit]
    
    return {
        'patient_id': patient_id,
        'timeline': sorted_timeline,
        'total_events': len(timeline)
    }


@router.get("/{patient_id}/graph")
async def get_patient_knowledge_graph(patient_id: str):
    """
    Get patient's knowledge graph (nodes and relationships)
    """
    kg = KnowledgeGraphService()
    graph = kg.get_patient_graph(patient_id)
    
    return graph


@router.post("/{patient_id}/allergies")
async def add_patient_allergy(
    patient_id: str,
    allergen: str,
    reaction_type: Optional[str] = None,
    severity: Optional[str] = None
):
    """
    Add an allergy to patient's record
    """
    if patient_id not in patient_store:
        patient_store[patient_id] = {
            'patient_id': patient_id,
            'allergies': [],
            'current_medications': [],
            'historical_medications': []
        }
    
    allergy_entry = {
        'allergen': allergen,
        'reaction_type': reaction_type,
        'severity': severity,
        'documented_date': datetime.utcnow().isoformat()
    }
    
    # Store as string for simple allergies list
    patient_store[patient_id].setdefault('allergies', []).append(allergen)
    patient_store[patient_id].setdefault('allergy_details', []).append(allergy_entry)
    
    return {'success': True, 'allergy': allergy_entry}


@router.get("/{patient_id}/summary")
async def get_patient_summary(patient_id: str):
    """
    Get a comprehensive patient summary
    """
    if patient_id not in patient_store:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient = patient_store[patient_id]
    
    # Get interaction check
    current_meds = patient.get('current_medications', [])
    med_names = [m.get('medication_name') for m in current_meds if m.get('medication_name')]
    
    interaction_service = DrugInteractionService()
    safety = interaction_service.analyze_medications(
        med_names, 
        patient.get('allergies', [])
    )
    
    return {
        'patient_id': patient_id,
        'name': patient.get('name'),
        'summary': {
            'total_current_medications': len(current_meds),
            'total_allergies': len(patient.get('allergies', [])),
            'total_conditions': len(patient.get('chronic_conditions', [])),
            'active_interactions': len(safety.interactions),
            'safety_score': safety.overall_safety_score
        },
        'current_medications': current_meds,
        'allergies': patient.get('allergies', []),
        'conditions': patient.get('chronic_conditions', []),
        'recent_prescriptions': patient.get('prescriptions', [])[-5:],
        'safety_alerts': {
            'interactions': [
                {'drugs': [i.drug1, i.drug2], 'severity': i.severity.value}
                for i in safety.interactions
            ],
            'allergy_alerts': [
                {'drug': a.drug, 'allergen': a.allergen}
                for a in safety.allergy_alerts
            ]
        }
    }
