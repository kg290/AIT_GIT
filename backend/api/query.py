"""
Query API Routes - Conversational medical querying
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.services import QueryService
from backend.api.patients import patient_store

router = APIRouter(prefix="/query", tags=["Query"])


@router.get("/")
async def query_medical_data(
    question: str,
    patient_id: Optional[str] = Query(None, description="Patient ID for context")
):
    """
    Ask natural language questions about medical data
    
    Examples:
    - "What medications is the patient taking?"
    - "Are there any drug interactions?"
    - "What are the patient's allergies?"
    - "Show the medication history"
    - "When did the patient start metformin?"
    """
    # Get patient data if patient_id provided
    patient_data = None
    if patient_id:
        patient_data = patient_store.get(patient_id)
        if not patient_data:
            return {
                'question': question,
                'answer': f"Patient with ID '{patient_id}' not found. Please create the patient first or provide a valid patient ID.",
                'evidence': [],
                'confidence': 0.0,
                'sources': [],
                'related_queries': ["Create a patient", "List all patients"]
            }
    
    # Process query
    query_service = QueryService()
    result = query_service.query(question, patient_id, patient_data)
    
    return query_service.to_dict(result)


@router.post("/chat")
async def chat(
    message: str,
    patient_id: Optional[str] = None,
    conversation_history: Optional[list] = None
):
    """
    Chat-style interface for medical queries
    
    Supports follow-up questions using conversation history
    """
    patient_data = patient_store.get(patient_id) if patient_id else None
    
    query_service = QueryService()
    result = query_service.query(message, patient_id, patient_data)
    
    response = query_service.to_dict(result)
    response['conversation_id'] = patient_id or 'anonymous'
    
    return response


@router.get("/suggestions")
async def get_query_suggestions(patient_id: Optional[str] = None):
    """
    Get suggested queries based on patient context
    """
    if patient_id and patient_id in patient_store:
        patient = patient_store[patient_id]
        
        suggestions = ["What medications is the patient taking?"]
        
        if patient.get('current_medications'):
            suggestions.append("Are there any drug interactions?")
            suggestions.append("Show medication history")
        
        if patient.get('allergies'):
            suggestions.append("What are the patient's allergies?")
        
        if patient.get('chronic_conditions'):
            suggestions.append("What conditions does the patient have?")
        
        suggestions.extend([
            "When was the last prescription?",
            "What changes have been made to medications?",
            "Show the patient timeline"
        ])
        
        return {'suggestions': suggestions}
    
    return {
        'suggestions': [
            "What medications is the patient taking?",
            "Are there any drug interactions?",
            "What are the patient's allergies?",
            "Show the medication history",
            "List all diagnoses"
        ]
    }


@router.get("/explain/{entity_type}/{entity_id}")
async def explain_entity(entity_type: str, entity_id: str):
    """
    Get explanation for a specific entity (medication, interaction, etc.)
    """
    explanations = {
        'medication': {
            'what': 'This is a medication extracted from the document',
            'confidence': 'Confidence score indicates how certain the system is about this extraction',
            'review': 'Low confidence items should be reviewed by a human'
        },
        'interaction': {
            'what': 'A potential drug-drug interaction was detected',
            'severity': 'Severity ranges from minor to contraindicated',
            'action': 'Review and consult clinical guidelines'
        },
        'allergy': {
            'what': 'A potential allergen match was detected',
            'action': 'Verify allergy status before prescribing'
        }
    }
    
    if entity_type in explanations:
        return {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'explanation': explanations[entity_type]
        }
    
    raise HTTPException(status_code=404, detail=f"Unknown entity type: {entity_type}")
