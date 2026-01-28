"""
Query Service - Conversational medical querying with evidence
"""
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a medical query"""
    answer: str
    evidence: List[Dict]
    confidence: float
    sources: List[str]
    reasoning_steps: List[str]
    related_queries: List[str]


class QueryService:
    """
    Conversational medical querying service
    
    Features:
    - Natural language question answering
    - Time-aware queries
    - Medication-specific queries
    - Evidence-based responses
    - Source references
    """
    
    def __init__(self, db: Session = None):
        self.db = db
        
        # Query patterns for rule-based answering
        self.query_patterns = {
            'current_medications': [
                r'what\s+(?:medications?|drugs?|medicines?)\s+(?:is|are)\s+(?:the\s+)?patient\s+(?:currently\s+)?(?:taking|on)',
                r'current\s+(?:medications?|drugs?|medicines?)',
                r'(?:list|show)\s+(?:all\s+)?(?:current\s+)?(?:medications?|drugs?)',
            ],
            'medication_history': [
                r'(?:medication|drug)\s+history',
                r'(?:past|previous|historical)\s+(?:medications?|drugs?)',
                r'what\s+(?:medications?|drugs?)\s+(?:has|have)\s+(?:been|the\s+patient)\s+(?:taken|prescribed)',
            ],
            'allergies': [
                r'(?:what\s+are\s+)?(?:the\s+)?(?:patient[\'s]?\s+)?allergies',
                r'(?:is|are)\s+(?:the\s+)?patient\s+allergic',
                r'(?:drug|medication)\s+allergies',
            ],
            'diagnoses': [
                r'(?:what\s+are\s+)?(?:the\s+)?(?:patient[\'s]?\s+)?(?:diagnoses|conditions)',
                r'(?:medical\s+)?conditions?',
                r'diagnosed\s+with',
            ],
            'interactions': [
                r'(?:drug|medication)\s+interactions?',
                r'(?:any|are\s+there)\s+interactions?',
                r'(?:can|should)\s+.+\s+(?:be\s+taken\s+)?(?:with|together)',
            ],
            'dosage': [
                r'(?:what\s+is\s+)?(?:the\s+)?dosage\s+(?:of|for)\s+(.+)',
                r'how\s+much\s+(.+)\s+(?:should|to)\s+(?:be\s+)?(?:taken|take)',
                r'(.+)\s+(?:dose|dosage)',
            ],
            'timeline': [
                r'(?:when|what\s+date)\s+(?:did|was)\s+(?:the\s+)?patient\s+(?:start|begin|prescribed)',
                r'how\s+long\s+(?:has|have)\s+(?:the\s+)?patient\s+(?:been\s+)?(?:taking|on)',
                r'(?:medication|prescription)\s+(?:timeline|history|dates)',
            ],
            'last_visit': [
                r'(?:when\s+was\s+)?(?:the\s+)?(?:last|most\s+recent)\s+(?:visit|appointment|prescription)',
                r'(?:latest|recent)\s+(?:visit|prescription)',
            ],
            'compare': [
                r'(?:compare|difference|changes?)\s+(?:between|from)\s+(?:prescriptions?|visits?)',
                r'what\s+(?:changed|is\s+different)',
                r'(?:any|are\s+there)\s+changes?',
            ],
        }
    
    def query(self, question: str, patient_id: str = None,
              patient_data: Dict = None) -> QueryResult:
        """
        Answer a natural language medical query
        
        Args:
            question: Natural language question
            patient_id: Optional patient context
            patient_data: Pre-loaded patient data
            
        Returns:
            QueryResult with answer and evidence
        """
        question_lower = question.lower().strip()
        
        # Identify query type
        query_type = self._identify_query_type(question_lower)
        
        # Route to appropriate handler
        if query_type == 'current_medications':
            return self._handle_current_medications(patient_data)
        elif query_type == 'medication_history':
            return self._handle_medication_history(patient_data)
        elif query_type == 'allergies':
            return self._handle_allergies(patient_data)
        elif query_type == 'diagnoses':
            return self._handle_diagnoses(patient_data)
        elif query_type == 'interactions':
            return self._handle_interactions(question_lower, patient_data)
        elif query_type == 'dosage':
            return self._handle_dosage(question_lower, patient_data)
        elif query_type == 'timeline':
            return self._handle_timeline(question_lower, patient_data)
        elif query_type == 'last_visit':
            return self._handle_last_visit(patient_data)
        elif query_type == 'compare':
            return self._handle_comparison(patient_data)
        else:
            return self._handle_general_query(question_lower, patient_data)
    
    def _identify_query_type(self, question: str) -> Optional[str]:
        """Identify the type of query"""
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return query_type
        return None
    
    def _handle_current_medications(self, patient_data: Dict) -> QueryResult:
        """Handle current medications query"""
        if not patient_data:
            return self._no_data_response("current medications")
        
        medications = patient_data.get('current_medications', [])
        
        if not medications:
            return QueryResult(
                answer="The patient is not currently taking any medications according to available records.",
                evidence=[],
                confidence=0.9,
                sources=[],
                reasoning_steps=["Checked current medication list", "No active medications found"],
                related_queries=["What medications has the patient taken in the past?", "When was the last prescription?"]
            )
        
        # Build answer
        med_list = []
        evidence = []
        for med in medications:
            med_str = med.get('medication_name', 'Unknown')
            if med.get('dosage'):
                med_str += f" {med['dosage']}"
            if med.get('frequency'):
                med_str += f" ({med['frequency']})"
            med_list.append(f"• {med_str}")
            evidence.append({
                'type': 'medication',
                'text': med_str,
                'source': med.get('source_document_id')
            })
        
        answer = f"The patient is currently taking {len(medications)} medication(s):\n\n" + "\n".join(med_list)
        
        return QueryResult(
            answer=answer,
            evidence=evidence,
            confidence=0.95,
            sources=[f"Document {med.get('source_document_id')}" for med in medications if med.get('source_document_id')],
            reasoning_steps=[
                "Retrieved current medication list",
                f"Found {len(medications)} active medications",
                "Compiled medication details with dosage and frequency"
            ],
            related_queries=[
                "Are there any drug interactions?",
                "What are the side effects of these medications?",
                "When did the patient start these medications?"
            ]
        )
    
    def _handle_medication_history(self, patient_data: Dict) -> QueryResult:
        """Handle medication history query"""
        if not patient_data:
            return self._no_data_response("medication history")
        
        current = patient_data.get('current_medications', [])
        historical = patient_data.get('historical_medications', [])
        
        all_meds = current + historical
        
        if not all_meds:
            return QueryResult(
                answer="No medication history found for this patient.",
                evidence=[],
                confidence=0.9,
                sources=[],
                reasoning_steps=["Checked medication history", "No records found"],
                related_queries=["Upload a prescription to start tracking medications"]
            )
        
        # Build timeline
        answer = f"Medication History ({len(all_meds)} medications found):\n\n"
        answer += f"**Currently Taking ({len(current)}):**\n"
        for med in current:
            answer += f"• {med.get('medication_name')} - started {med.get('start_date', 'unknown date')}\n"
        
        if historical:
            answer += f"\n**Previously Taken ({len(historical)}):**\n"
            for med in historical:
                answer += f"• {med.get('medication_name')} ({med.get('start_date')} to {med.get('end_date')})\n"
        
        return QueryResult(
            answer=answer,
            evidence=[{'type': 'medication', 'text': med.get('medication_name')} for med in all_meds],
            confidence=0.9,
            sources=[],
            reasoning_steps=[
                "Retrieved complete medication history",
                f"Found {len(current)} current and {len(historical)} past medications",
                "Organized by current vs historical"
            ],
            related_queries=[
                "What changes have been made to medications?",
                "Why was a medication stopped?"
            ]
        )
    
    def _handle_allergies(self, patient_data: Dict) -> QueryResult:
        """Handle allergies query"""
        if not patient_data:
            return self._no_data_response("allergy information")
        
        allergies = patient_data.get('allergies', [])
        
        if not allergies:
            return QueryResult(
                answer="No known allergies recorded for this patient (NKDA).",
                evidence=[],
                confidence=0.8,
                sources=[],
                reasoning_steps=["Checked allergy records", "No allergies documented"],
                related_queries=["Should the patient be tested for allergies?"]
            )
        
        allergy_list = "\n".join([f"• {a}" for a in allergies])
        answer = f"The patient has {len(allergies)} documented allergy(ies):\n\n{allergy_list}\n\n"
        answer += "⚠️ Always verify allergies before prescribing."
        
        return QueryResult(
            answer=answer,
            evidence=[{'type': 'allergy', 'text': a} for a in allergies],
            confidence=0.95,
            sources=[],
            reasoning_steps=["Retrieved allergy list", f"Found {len(allergies)} allergies"],
            related_queries=["Are any current medications contraindicated?"]
        )
    
    def _handle_diagnoses(self, patient_data: Dict) -> QueryResult:
        """Handle diagnoses query"""
        if not patient_data:
            return self._no_data_response("diagnosis information")
        
        diagnoses = patient_data.get('diagnoses', [])
        conditions = patient_data.get('chronic_conditions', [])
        
        all_conditions = diagnoses + [{'diagnosis_name': c} for c in conditions]
        
        if not all_conditions:
            return QueryResult(
                answer="No diagnoses or conditions recorded for this patient.",
                evidence=[],
                confidence=0.8,
                sources=[],
                reasoning_steps=["Checked diagnosis records", "No conditions found"],
                related_queries=["What symptoms has the patient reported?"]
            )
        
        answer = f"The patient has {len(all_conditions)} documented condition(s):\n\n"
        for dx in all_conditions:
            name = dx.get('diagnosis_name') or dx.get('name', 'Unknown')
            answer += f"• {name}"
            if dx.get('diagnosis_date'):
                answer += f" (diagnosed: {dx['diagnosis_date']})"
            if dx.get('is_chronic'):
                answer += " [Chronic]"
            answer += "\n"
        
        return QueryResult(
            answer=answer,
            evidence=[{'type': 'diagnosis', 'text': dx.get('diagnosis_name', '')} for dx in all_conditions],
            confidence=0.9,
            sources=[],
            reasoning_steps=["Retrieved diagnosis list", f"Found {len(all_conditions)} conditions"],
            related_queries=["What medications are prescribed for these conditions?"]
        )
    
    def _handle_interactions(self, question: str, patient_data: Dict) -> QueryResult:
        """Handle drug interaction queries"""
        if not patient_data:
            return self._no_data_response("drug interaction check")
        
        interactions = patient_data.get('drug_interactions', [])
        
        if not interactions:
            return QueryResult(
                answer="No significant drug interactions detected with current medications.",
                evidence=[],
                confidence=0.85,
                sources=["Drug interaction database"],
                reasoning_steps=[
                    "Retrieved current medication list",
                    "Checked against drug interaction database",
                    "No significant interactions found"
                ],
                related_queries=["What are the side effects of current medications?"]
            )
        
        # Sort by severity
        major = [i for i in interactions if i.get('severity') in ['major', 'contraindicated']]
        moderate = [i for i in interactions if i.get('severity') == 'moderate']
        minor = [i for i in interactions if i.get('severity') == 'minor']
        
        answer = f"⚠️ Found {len(interactions)} drug interaction(s):\n\n"
        
        if major:
            answer += "**MAJOR/CONTRAINDICATED:**\n"
            for i in major:
                answer += f"• {i.get('drug1')} + {i.get('drug2')}: {i.get('description')}\n"
        
        if moderate:
            answer += "\n**MODERATE:**\n"
            for i in moderate:
                answer += f"• {i.get('drug1')} + {i.get('drug2')}: {i.get('description')}\n"
        
        return QueryResult(
            answer=answer,
            evidence=[{'type': 'interaction', 'drugs': [i.get('drug1'), i.get('drug2')]} for i in interactions],
            confidence=0.9,
            sources=["Drug interaction database"],
            reasoning_steps=[
                "Retrieved current medications",
                "Checked pairwise interactions",
                f"Found {len(major)} major, {len(moderate)} moderate, {len(minor)} minor"
            ],
            related_queries=["How should these interactions be managed?", "Are there alternatives?"]
        )
    
    def _handle_dosage(self, question: str, patient_data: Dict) -> QueryResult:
        """Handle dosage queries"""
        # Extract medication name from question
        med_match = re.search(r'dosage\s+(?:of|for)\s+([a-zA-Z]+)', question)
        if not med_match:
            med_match = re.search(r'([a-zA-Z]+)\s+(?:dose|dosage)', question)
        
        if med_match and patient_data:
            med_name = med_match.group(1).lower()
            current_meds = patient_data.get('current_medications', [])
            
            for med in current_meds:
                if med_name in med.get('medication_name', '').lower():
                    answer = f"Current dosage of {med.get('medication_name')}:\n"
                    answer += f"• Dose: {med.get('dosage', 'Not specified')}\n"
                    answer += f"• Frequency: {med.get('frequency', 'Not specified')}\n"
                    answer += f"• Route: {med.get('route', 'Not specified')}"
                    
                    return QueryResult(
                        answer=answer,
                        evidence=[med],
                        confidence=0.95,
                        sources=[],
                        reasoning_steps=[f"Found {med.get('medication_name')} in current medications"],
                        related_queries=["Is this the correct dosage?", "When should the dose be taken?"]
                    )
        
        return QueryResult(
            answer="Could not find dosage information for the specified medication. Please specify the medication name.",
            evidence=[],
            confidence=0.5,
            sources=[],
            reasoning_steps=["Attempted to find medication in records", "Medication not found"],
            related_queries=["What medications is the patient taking?"]
        )
    
    def _handle_timeline(self, question: str, patient_data: Dict) -> QueryResult:
        """Handle timeline queries"""
        if not patient_data:
            return self._no_data_response("medication timeline")
        
        timeline = patient_data.get('timeline', [])
        
        if not timeline:
            return QueryResult(
                answer="No timeline events found for this patient.",
                evidence=[],
                confidence=0.8,
                sources=[],
                reasoning_steps=["Checked timeline events", "No events found"],
                related_queries=["Upload documents to build patient timeline"]
            )
        
        # Sort by date
        sorted_timeline = sorted(timeline, key=lambda x: x.get('event_date', ''), reverse=True)
        
        answer = "Patient Medical Timeline:\n\n"
        for event in sorted_timeline[:10]:  # Show last 10 events
            date = event.get('event_date', 'Unknown date')
            if isinstance(date, str) and len(date) > 10:
                date = date[:10]
            answer += f"📅 **{date}**: {event.get('title', 'Event')}\n"
            if event.get('description'):
                answer += f"   {event['description']}\n"
        
        return QueryResult(
            answer=answer,
            evidence=sorted_timeline[:10],
            confidence=0.9,
            sources=[],
            reasoning_steps=[f"Retrieved {len(timeline)} timeline events", "Sorted chronologically"],
            related_queries=["What changed between visits?", "When was a specific medication started?"]
        )
    
    def _handle_last_visit(self, patient_data: Dict) -> QueryResult:
        """Handle last visit query"""
        if not patient_data:
            return self._no_data_response("visit information")
        
        prescriptions = patient_data.get('prescriptions', [])
        
        if not prescriptions:
            return QueryResult(
                answer="No prescription records found.",
                evidence=[],
                confidence=0.8,
                sources=[],
                reasoning_steps=["No prescriptions found"],
                related_queries=["Upload a prescription to add records"]
            )
        
        # Get most recent
        sorted_rx = sorted(prescriptions, key=lambda x: x.get('prescription_date', ''), reverse=True)
        last_rx = sorted_rx[0]
        
        answer = f"Most Recent Prescription:\n\n"
        answer += f"📅 Date: {last_rx.get('prescription_date', 'Unknown')}\n"
        answer += f"👨‍⚕️ Prescriber: {last_rx.get('prescriber_name', 'Unknown')}\n"
        if last_rx.get('diagnosis'):
            answer += f"🔍 Diagnosis: {last_rx['diagnosis']}\n"
        answer += f"\nMedications prescribed: {len(last_rx.get('items', []))}"
        
        return QueryResult(
            answer=answer,
            evidence=[last_rx],
            confidence=0.95,
            sources=[last_rx.get('source_document_id')],
            reasoning_steps=["Retrieved prescription history", "Found most recent prescription"],
            related_queries=["What medications were prescribed?", "Compare with previous visit"]
        )
    
    def _handle_comparison(self, patient_data: Dict) -> QueryResult:
        """Handle prescription comparison queries"""
        if not patient_data:
            return self._no_data_response("prescription comparison")
        
        changes = patient_data.get('medication_changes', [])
        
        if not changes:
            return QueryResult(
                answer="No medication changes detected in available records.",
                evidence=[],
                confidence=0.8,
                sources=[],
                reasoning_steps=["Analyzed prescription history", "No changes found"],
                related_queries=["What is the current medication list?"]
            )
        
        answer = "Medication Changes Detected:\n\n"
        for change in changes:
            if change.get('change_type') == 'started':
                answer += f"➕ Started: {change.get('medication_name')} ({change.get('change_date')})\n"
            elif change.get('change_type') == 'stopped':
                answer += f"➖ Stopped: {change.get('medication_name')} ({change.get('change_date')})\n"
            elif change.get('change_type') == 'dose_changed':
                answer += f"📊 Dose changed: {change.get('medication_name')} ({change.get('old_value')} → {change.get('new_value')})\n"
        
        return QueryResult(
            answer=answer,
            evidence=changes,
            confidence=0.9,
            sources=[],
            reasoning_steps=["Compared prescriptions over time", f"Found {len(changes)} changes"],
            related_queries=["Why was a medication changed?", "Current medication list"]
        )
    
    def _handle_general_query(self, question: str, patient_data: Dict) -> QueryResult:
        """Handle general queries that don't match specific patterns"""
        # Build a summary response
        answer = "I can help you with questions about:\n\n"
        answer += "• Current medications\n"
        answer += "• Medication history\n"
        answer += "• Drug interactions\n"
        answer += "• Allergies\n"
        answer += "• Diagnoses and conditions\n"
        answer += "• Prescription timeline\n"
        answer += "• Medication changes\n\n"
        answer += "Please try asking a more specific question, such as:\n"
        answer += "- 'What medications is the patient currently taking?'\n"
        answer += "- 'Are there any drug interactions?'\n"
        answer += "- 'Show the medication history'"
        
        return QueryResult(
            answer=answer,
            evidence=[],
            confidence=0.5,
            sources=[],
            reasoning_steps=["Could not match query to specific pattern", "Provided guidance"],
            related_queries=[
                "What medications is the patient taking?",
                "Are there any drug interactions?",
                "What is the patient's diagnosis?"
            ]
        )
    
    def _no_data_response(self, data_type: str) -> QueryResult:
        """Response when no patient data available"""
        return QueryResult(
            answer=f"No patient data available to answer questions about {data_type}. Please select a patient or upload documents first.",
            evidence=[],
            confidence=0.0,
            sources=[],
            reasoning_steps=["No patient context provided"],
            related_queries=["Select a patient to continue"]
        )
    
    def to_dict(self, result: QueryResult) -> Dict:
        """Convert QueryResult to dictionary"""
        return {
            'answer': result.answer,
            'evidence': result.evidence,
            'confidence': result.confidence,
            'sources': result.sources,
            'reasoning_steps': result.reasoning_steps,
            'related_queries': result.related_queries
        }
