"""
Conversational Medical Querying Service
Natural language questions over patient history with evidence
"""
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Parsed query intent"""
    intent_type: str  # medication, history, interaction, comparison, timeline, summary
    entities: Dict[str, Any]
    time_range: Optional[Tuple[datetime, datetime]]
    patient_id: Optional[int]


@dataclass
class QueryResponse:
    """Response to a natural language query"""
    answer: str
    evidence: List[Dict]
    confidence: float
    related_queries: List[str]
    visualization_data: Dict = None


class ConversationalQueryService:
    """
    Conversational Medical Querying Service
    
    Features:
    - Parse natural language medical questions
    - Search patient history
    - Provide evidence-backed answers
    - Suggest related queries
    """
    
    # Intent patterns
    INTENT_PATTERNS = {
        "medication_current": [
            r"what (medications?|drugs?|meds?) (is|are) .+ (taking|on|using)",
            r"current (medications?|drugs?|prescriptions?)",
            r"(list|show|get) .+ (medications?|drugs?|prescriptions?)"
        ],
        "medication_history": [
            r"(history|past|previous) (medications?|drugs?|prescriptions?)",
            r"(has|did|was) .+ (taken?|prescribed|used?) .+",
            r"when (did|was) .+ (start|stop|begin|end)"
        ],
        "interaction_check": [
            r"(interaction|conflict|combine|mix|safe).*(between|with)",
            r"can .+ take .+ (with|and|together)",
            r"(is|are) .+ (safe|compatible|ok) (with|together)"
        ],
        "dosage_query": [
            r"(dosage|dose|how much) .+ (of|for)?",
            r"what (dose|dosage|amount) .+ (taking|prescribed|on)"
        ],
        "allergy_check": [
            r"(allerg|sensitive|react).*(to|with)",
            r"(has|does|is) .+ (allergic|sensitive|intolerant)"
        ],
        "timeline_query": [
            r"(timeline|history|when|dates?) .* (medications?|treatment|conditions?)",
            r"(show|display|get) .* (timeline|history|overview)"
        ],
        "comparison_query": [
            r"(compare|difference|changed?|vs|versus)",
            r"(before|after|since|between).*(visit|appointment|date)"
        ],
        "summary_query": [
            r"(summary|summarize|overview|report)",
            r"(tell|give|show) me about .+ (health|medical|history)"
        ],
        "condition_query": [
            r"(condition|diagnosis|disease|illness|problem)",
            r"(what|any) .* (conditions?|diagnos|disease|illness)"
        ]
    }
    
    # Time expression patterns
    TIME_PATTERNS = {
        "today": lambda: (datetime.now().replace(hour=0, minute=0), datetime.now()),
        "yesterday": lambda: (
            (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0),
            (datetime.now() - timedelta(days=1)).replace(hour=23, minute=59)
        ),
        "last week": lambda: (datetime.now() - timedelta(weeks=1), datetime.now()),
        "last month": lambda: (datetime.now() - timedelta(days=30), datetime.now()),
        "last year": lambda: (datetime.now() - timedelta(days=365), datetime.now()),
        "last 3 months": lambda: (datetime.now() - timedelta(days=90), datetime.now()),
        "last 6 months": lambda: (datetime.now() - timedelta(days=180), datetime.now())
    }
    
    def __init__(self, db: Session):
        self.db = db
        self._load_services()
    
    def _load_services(self):
        """Lazy load other services"""
        try:
            from backend.services.patient_history_service import PatientHistoryService
            from backend.services.drug_interaction_service import DrugInteractionService
            from backend.services.enhanced_temporal_reasoning_service import EnhancedTemporalReasoningService
            
            self.history_service = PatientHistoryService(self.db)
            self.interaction_service = DrugInteractionService(self.db)
            self.temporal_service = EnhancedTemporalReasoningService(self.db)
        except ImportError as e:
            logger.warning(f"Could not load all services: {e}")
            self.history_service = None
            self.interaction_service = None
            self.temporal_service = None
    
    # ==================== Query Parsing ====================
    
    def parse_query(self, query: str, patient_id: int = None) -> QueryIntent:
        """Parse natural language query into structured intent"""
        
        query_lower = query.lower().strip()
        
        # Detect intent type
        intent_type = self._detect_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Extract time range
        time_range = self._extract_time_range(query_lower)
        
        # Extract patient reference if mentioned
        if patient_id is None:
            patient_id = self._extract_patient_id(query_lower)
        
        return QueryIntent(
            intent_type=intent_type,
            entities=entities,
            time_range=time_range,
            patient_id=patient_id
        )
    
    def _detect_intent(self, query: str) -> str:
        """Detect the type of query"""
        
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent_type
        
        return "general"
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract medical entities from query"""
        
        entities = {
            "medications": [],
            "conditions": [],
            "patient_name": None
        }
        
        # Extract medication names (simplified - would use NER in production)
        medication_indicators = [
            "aspirin", "ibuprofen", "metformin", "lisinopril", "atorvastatin",
            "omeprazole", "amlodipine", "metoprolol", "losartan", "gabapentin",
            "warfarin", "paracetamol", "acetaminophen", "prednisone", "albuterol"
        ]
        
        words = query.lower().split()
        for med in medication_indicators:
            if med in query.lower():
                entities["medications"].append(med)
        
        # Extract condition names (simplified)
        condition_indicators = [
            "diabetes", "hypertension", "asthma", "arthritis", "depression",
            "anxiety", "cholesterol", "heart disease", "copd", "thyroid"
        ]
        
        for cond in condition_indicators:
            if cond in query.lower():
                entities["conditions"].append(cond)
        
        return entities
    
    def _extract_time_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract time range from query"""
        
        for pattern, time_func in self.TIME_PATTERNS.items():
            if pattern in query.lower():
                return time_func()
        
        # Try to extract specific dates (simplified)
        date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", query)
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                return (date, date + timedelta(days=1))
            except ValueError:
                pass
        
        return None
    
    def _extract_patient_id(self, query: str) -> Optional[int]:
        """Extract patient ID from query"""
        
        # Look for patient ID patterns
        match = re.search(r"patient[#\s]*(\d+)", query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None
    
    # ==================== Query Processing ====================
    
    def process_query(
        self,
        query: str,
        patient_id: int = None,
        context: Dict = None
    ) -> QueryResponse:
        """Process a natural language query and return response"""
        
        # Parse the query
        intent = self.parse_query(query, patient_id)
        
        # Route to appropriate handler
        handlers = {
            "medication_current": self._handle_current_medications,
            "medication_history": self._handle_medication_history,
            "interaction_check": self._handle_interaction_check,
            "dosage_query": self._handle_dosage_query,
            "allergy_check": self._handle_allergy_check,
            "timeline_query": self._handle_timeline_query,
            "comparison_query": self._handle_comparison_query,
            "summary_query": self._handle_summary_query,
            "condition_query": self._handle_condition_query,
            "general": self._handle_general_query
        }
        
        handler = handlers.get(intent.intent_type, self._handle_general_query)
        
        try:
            response = handler(intent, context)
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            response = QueryResponse(
                answer=f"I encountered an error processing your question. Please try rephrasing.",
                evidence=[],
                confidence=0.0,
                related_queries=self._suggest_related_queries(intent)
            )
        
        return response
    
    def _handle_current_medications(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle current medication queries"""
        
        if not intent.patient_id:
            return QueryResponse(
                answer="Please specify a patient ID to view their current medications.",
                evidence=[],
                confidence=0.5,
                related_queries=["Show medications for patient #123", "List all patients"]
            )
        
        if self.history_service:
            history = self.history_service.get_medication_history(intent.patient_id, active_only=True)
            
            if history:
                med_list = []
                evidence = []
                for med in history:
                    med_list.append(f"• {med.medication_name} - {med.dosage} ({med.frequency})")
                    evidence.append({
                        "type": "medication",
                        "name": med.medication_name,
                        "dosage": med.dosage,
                        "started": str(med.start_date),
                        "source": "patient_history"
                    })
                
                answer = f"Patient #{intent.patient_id} is currently taking {len(med_list)} medication(s):\n\n" + "\n".join(med_list)
            else:
                answer = f"No current medications found for patient #{intent.patient_id}."
                evidence = []
        else:
            answer = "Patient history service is not available."
            evidence = []
        
        return QueryResponse(
            answer=answer,
            evidence=evidence,
            confidence=0.9,
            related_queries=[
                f"Show medication history for patient #{intent.patient_id}",
                f"Check interactions for patient #{intent.patient_id}",
                f"Show timeline for patient #{intent.patient_id}"
            ]
        )
    
    def _handle_medication_history(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle medication history queries"""
        
        if not intent.patient_id:
            return QueryResponse(
                answer="Please specify a patient ID.",
                evidence=[],
                confidence=0.5,
                related_queries=[]
            )
        
        if self.history_service:
            history = self.history_service.get_complete_patient_history(intent.patient_id)
            
            meds = history.get("medications", [])
            if meds:
                answer_parts = [f"Medication history for patient #{intent.patient_id}:\n"]
                evidence = []
                
                for med in meds:
                    status = "Active" if med.get("active") else f"Stopped {med.get('end_date', 'N/A')}"
                    answer_parts.append(f"• {med['name']} - {med.get('dosage', 'N/A')} ({status})")
                    evidence.append({"type": "medication_history", **med})
                
                answer = "\n".join(answer_parts)
            else:
                answer = f"No medication history found for patient #{intent.patient_id}."
                evidence = []
        else:
            answer = "History service unavailable."
            evidence = []
        
        return QueryResponse(
            answer=answer,
            evidence=evidence,
            confidence=0.85,
            related_queries=[
                f"Current medications for patient #{intent.patient_id}",
                f"Timeline for patient #{intent.patient_id}"
            ]
        )
    
    def _handle_interaction_check(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle drug interaction queries"""
        
        meds = intent.entities.get("medications", [])
        
        if len(meds) < 2:
            return QueryResponse(
                answer="Please specify at least two medications to check for interactions.",
                evidence=[],
                confidence=0.5,
                related_queries=[
                    "Can I take aspirin with ibuprofen?",
                    "Check interactions between warfarin and aspirin"
                ]
            )
        
        if self.interaction_service:
            interactions = []
            evidence = []
            
            for i, drug1 in enumerate(meds):
                for drug2 in meds[i+1:]:
                    result = self.interaction_service.check_interaction(drug1, drug2)
                    if result and result.get("has_interaction"):
                        interactions.append(result)
                        evidence.append({
                            "type": "interaction",
                            "drugs": [drug1, drug2],
                            "severity": result.get("severity"),
                            "description": result.get("description")
                        })
            
            if interactions:
                answer_parts = [f"⚠️ Found {len(interactions)} interaction(s):\n"]
                for inter in interactions:
                    answer_parts.append(f"• {inter.get('description', 'Unknown interaction')}")
                    answer_parts.append(f"  Severity: {inter.get('severity', 'Unknown')}")
                answer = "\n".join(answer_parts)
            else:
                answer = f"✅ No known interactions found between {', '.join(meds)}."
        else:
            answer = "Interaction checking service is not available."
            evidence = []
        
        return QueryResponse(
            answer=answer,
            evidence=evidence,
            confidence=0.9,
            related_queries=[]
        )
    
    def _handle_dosage_query(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle dosage queries"""
        
        meds = intent.entities.get("medications", [])
        
        if not meds:
            return QueryResponse(
                answer="Please specify a medication to get dosage information.",
                evidence=[],
                confidence=0.5,
                related_queries=["What is the dosage for metformin?"]
            )
        
        # This would query the drug database in production
        answer = f"I can look up dosage information for: {', '.join(meds)}. However, please consult with a healthcare provider for specific dosing instructions."
        
        return QueryResponse(
            answer=answer,
            evidence=[],
            confidence=0.7,
            related_queries=[f"Check interactions for {meds[0]}" if meds else ""]
        )
    
    def _handle_allergy_check(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle allergy queries"""
        
        if not intent.patient_id:
            return QueryResponse(
                answer="Please specify a patient ID to check allergies.",
                evidence=[],
                confidence=0.5,
                related_queries=[]
            )
        
        # Query patient allergies from history
        answer = f"Allergy information for patient #{intent.patient_id}: Please check patient records for allergy data."
        
        return QueryResponse(
            answer=answer,
            evidence=[],
            confidence=0.7,
            related_queries=[f"Show medical history for patient #{intent.patient_id}"]
        )
    
    def _handle_timeline_query(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle timeline queries"""
        
        if not intent.patient_id:
            return QueryResponse(
                answer="Please specify a patient ID to view their timeline.",
                evidence=[],
                confidence=0.5,
                related_queries=[]
            )
        
        if self.temporal_service:
            timeline = self.temporal_service.build_patient_timeline(intent.patient_id)
            gantt_data = self.temporal_service.get_gantt_chart_data(intent.patient_id)
            
            answer_parts = [f"Timeline for patient #{intent.patient_id}:\n"]
            
            for event in timeline[:10]:  # Show last 10 events
                answer_parts.append(f"• {event.event_date.strftime('%Y-%m-%d')}: {event.event_type} - {event.description}")
            
            answer = "\n".join(answer_parts)
            evidence = [{"type": "timeline_event", **e.__dict__} for e in timeline[:10]]
        else:
            answer = "Timeline service is not available."
            evidence = []
        
        return QueryResponse(
            answer=answer,
            evidence=evidence,
            confidence=0.85,
            related_queries=[
                f"Show medication overlaps for patient #{intent.patient_id}",
                f"Compare visits for patient #{intent.patient_id}"
            ],
            visualization_data=gantt_data if self.temporal_service else None
        )
    
    def _handle_comparison_query(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle comparison queries"""
        
        answer = "Comparison queries require two dates or visits to compare. Please specify: 'Compare medications before and after [date]' or 'Compare visit 1 vs visit 2 for patient #123'."
        
        return QueryResponse(
            answer=answer,
            evidence=[],
            confidence=0.5,
            related_queries=["Show medication changes for patient #123"]
        )
    
    def _handle_summary_query(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle summary queries"""
        
        if not intent.patient_id:
            return QueryResponse(
                answer="Please specify a patient ID for their summary.",
                evidence=[],
                confidence=0.5,
                related_queries=[]
            )
        
        if self.history_service and self.temporal_service:
            history = self.history_service.get_complete_patient_history(intent.patient_id)
            summary = self.temporal_service.get_timeline_summary(intent.patient_id)
            
            answer_parts = [f"📋 Summary for Patient #{intent.patient_id}\n"]
            answer_parts.append(f"Total visits: {history.get('total_visits', 'N/A')}")
            answer_parts.append(f"Current medications: {len(history.get('medications', []))}")
            answer_parts.append(f"Conditions: {len(history.get('conditions', []))}")
            
            if summary:
                answer_parts.append(f"\n{summary.get('summary_text', '')}")
            
            answer = "\n".join(answer_parts)
            evidence = [{"type": "summary", "data": history}]
        else:
            answer = "Summary service is not available."
            evidence = []
        
        return QueryResponse(
            answer=answer,
            evidence=evidence,
            confidence=0.8,
            related_queries=[
                f"Show timeline for patient #{intent.patient_id}",
                f"Current medications for patient #{intent.patient_id}"
            ]
        )
    
    def _handle_condition_query(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle condition/diagnosis queries"""
        
        if not intent.patient_id:
            return QueryResponse(
                answer="Please specify a patient ID to view their conditions.",
                evidence=[],
                confidence=0.5,
                related_queries=[]
            )
        
        if self.history_service:
            history = self.history_service.get_complete_patient_history(intent.patient_id)
            conditions = history.get("conditions", [])
            
            if conditions:
                answer_parts = [f"Conditions for patient #{intent.patient_id}:\n"]
                for cond in conditions:
                    status = "Active" if cond.get("active") else "Resolved"
                    answer_parts.append(f"• {cond['name']} ({status})")
                answer = "\n".join(answer_parts)
                evidence = conditions
            else:
                answer = f"No conditions recorded for patient #{intent.patient_id}."
                evidence = []
        else:
            answer = "History service unavailable."
            evidence = []
        
        return QueryResponse(
            answer=answer,
            evidence=evidence,
            confidence=0.85,
            related_queries=[f"Medications for patient #{intent.patient_id}"]
        )
    
    def _handle_general_query(self, intent: QueryIntent, context: Dict) -> QueryResponse:
        """Handle general/unrecognized queries"""
        
        suggested_queries = [
            "What medications is patient #123 taking?",
            "Check interactions between aspirin and warfarin",
            "Show medication history for patient #456",
            "Show timeline for patient #789"
        ]
        
        answer = """I can help you with the following types of questions:

• **Current medications**: "What medications is patient #123 taking?"
• **Medication history**: "Show medication history for patient #123"
• **Drug interactions**: "Check interactions between aspirin and ibuprofen"
• **Timeline**: "Show timeline for patient #123"
• **Conditions**: "What conditions does patient #123 have?"
• **Summary**: "Give me a summary for patient #123"

Please try asking a specific question about a patient."""
        
        return QueryResponse(
            answer=answer,
            evidence=[],
            confidence=0.3,
            related_queries=suggested_queries
        )
    
    def _suggest_related_queries(self, intent: QueryIntent) -> List[str]:
        """Suggest related queries based on current intent"""
        
        suggestions = []
        
        if intent.patient_id:
            suggestions.extend([
                f"Current medications for patient #{intent.patient_id}",
                f"Show timeline for patient #{intent.patient_id}",
                f"Summary for patient #{intent.patient_id}"
            ])
        
        if intent.entities.get("medications"):
            meds = intent.entities["medications"]
            if len(meds) >= 1:
                suggestions.append(f"Check interactions for {meds[0]}")
        
        return suggestions[:5]
    
    # ==================== Batch Querying ====================
    
    def batch_query(
        self,
        queries: List[str],
        patient_id: int = None
    ) -> List[QueryResponse]:
        """Process multiple queries"""
        
        return [self.process_query(q, patient_id) for q in queries]
