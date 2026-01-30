"""
Medical AI Assistant Service
A comprehensive AI assistant for doctors that integrates all medical features:
- Natural language patient queries
- Drug normalization and analysis
- Allergy and interaction checking
- Medication timeline analysis
- Intelligent recommendations
"""
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the AI can handle"""
    PATIENT_INFO = "patient_info"
    MEDICATION_QUERY = "medication_query"
    ALLERGY_CHECK = "allergy_check"
    DRUG_INTERACTION = "drug_interaction"
    DRUG_NORMALIZE = "drug_normalize"
    SAFETY_ANALYSIS = "safety_analysis"
    TIMELINE_QUERY = "timeline_query"
    PRESCRIPTION_HISTORY = "prescription_history"
    GENERAL_MEDICAL = "general_medical"
    RECOMMENDATION = "recommendation"


@dataclass
class AIResponse:
    """Structured response from AI Assistant"""
    success: bool
    query_type: str
    answer: str
    data: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    processing_time_ms: float = 0


class MedicalAIAssistant:
    """
    Comprehensive Medical AI Assistant for Doctors
    
    Features:
    - Natural language understanding for medical queries
    - Automatic intent detection
    - Integration with all medical services
    - Context-aware responses
    - Safety-first approach
    """
    
    # Query patterns for intent detection
    INTENT_PATTERNS = {
        QueryType.MEDICATION_QUERY: [
            r"(what|which|list|show|get).*(medication|medicine|drug|prescription)s?",
            r"(current|last|recent|previous).*(medication|medicine|drug|prescription)s?",
            r"(is|are|was).*(taking|prescribed|on).*(medication|medicine|drug)?",
            r"(give|show|tell).*(medication|medicine|drug|prescription)s?",
        ],
        QueryType.ALLERGY_CHECK: [
            r"(allerg|sensitive|react|intoleran)",
            r"(is|are|does|can).*(allergic|sensitive)",
            r"(check|verify|test).*(allergy|allergies)",
            r"(what|which|any|list).*(allergy|allergies)",
        ],
        QueryType.DRUG_INTERACTION: [
            r"(interact|conflict|combine|mix|safe)",
            r"(can|should).*(take|use).*(together|with|and)",
            r"(drug|medication).*(interaction|conflict)",
            r"(is|are).*(safe|compatible|ok).*(together|with)",
        ],
        QueryType.DRUG_NORMALIZE: [
            r"(generic|brand|normalize|standardize)",
            r"(what is|convert|same as)",
            r"(equivalent|alternative|substitute)",
        ],
        QueryType.SAFETY_ANALYSIS: [
            r"(safety|safe|risk|danger|warning|concern)",
            r"(analyze|check|verify|assess).*(safety|risk|medication)",
            r"(comprehensive|full|complete).*(analysis|check|review)",
        ],
        QueryType.TIMELINE_QUERY: [
            r"(timeline|history|when|how long)",
            r"(started|stopped|changed|began|ended)",
            r"(duration|period|time)",
        ],
        QueryType.PRESCRIPTION_HISTORY: [
            r"(prescription|rx).*(history|past|previous)",
            r"(previous|past|old|last).*(prescription|rx)",
            r"(show|list|get).*(prescription|rx).*(history)?",
        ],
        QueryType.PATIENT_INFO: [
            r"(patient|user|person).*(info|information|details|data)",
            r"(who is|about|tell me).*(patient|user)",
            r"(age|gender|dob|name).*(patient|of)?",
        ],
        QueryType.RECOMMENDATION: [
            r"(recommend|suggest|advise|should)",
            r"(what|which).*(best|better|prefer)",
            r"(alternative|option|choice)",
        ],
    }
    
    # Common medication abbreviation expansions
    MED_ABBREVIATIONS = {
        "od": "once daily",
        "bd": "twice daily",
        "bid": "twice daily",
        "tds": "three times daily",
        "tid": "three times daily",
        "qid": "four times daily",
        "prn": "as needed",
        "ac": "before meals",
        "pc": "after meals",
        "hs": "at bedtime",
        "sos": "if required",
        "stat": "immediately",
        "po": "by mouth",
        "iv": "intravenous",
        "im": "intramuscular",
        "sc": "subcutaneous",
        "tab": "tablet",
        "cap": "capsule",
        "syp": "syrup",
        "inj": "injection",
    }
    
    def __init__(self):
        """Initialize the Medical AI Assistant with all required services"""
        self.drug_normalizer = None
        self.drug_interaction_service = None
        self.temporal_service = None
        self.gemini_service = None
        self.unified_patient_service = None
        self._load_services()
    
    def _load_services(self):
        """Lazy load all required services"""
        try:
            from backend.services.drug_normalization_service import DrugNormalizationService
            self.drug_normalizer = DrugNormalizationService()
            logger.info("Drug normalization service loaded")
        except Exception as e:
            logger.warning(f"Could not load drug normalization service: {e}")
        
        try:
            from backend.services.drug_interaction_service import DrugInteractionService
            self.drug_interaction_service = DrugInteractionService()
            logger.info("Drug interaction service loaded")
        except Exception as e:
            logger.warning(f"Could not load drug interaction service: {e}")
        
        try:
            from backend.services.temporal_reasoning_service import TemporalReasoningService
            self.temporal_service = TemporalReasoningService()
            logger.info("Temporal reasoning service loaded")
        except Exception as e:
            logger.warning(f"Could not load temporal reasoning service: {e}")
        
        try:
            from backend.services.gemini_service import GeminiService
            self.gemini_service = GeminiService()
            logger.info("Gemini service loaded")
        except Exception as e:
            logger.warning(f"Could not load Gemini service: {e}")
        
        # Load the unified patient service for database access
        try:
            from backend.services.unified_patient_service import get_unified_patient_service
            self.unified_patient_service = get_unified_patient_service()
            logger.info("Unified patient service loaded - AI now queries from database")
        except Exception as e:
            logger.warning(f"Could not load unified patient service: {e}")
    
    # ==================== Main Query Handler ====================
    
    async def process_query(
        self,
        query: str,
        patient_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> AIResponse:
        """
        Main entry point for processing doctor queries
        
        Args:
            query: Natural language query from doctor
            patient_id: Optional patient context
            context: Additional context (allergies, current meds, etc.)
        
        Returns:
            AIResponse with answer, data, and recommendations
        """
        import time
        start_time = time.time()
        
        try:
            # Detect query intent
            query_type = self._detect_intent(query)
            
            # Get patient context if available
            patient_data = self._get_patient_context(patient_id, context)
            
            # Route to appropriate handler
            if query_type == QueryType.MEDICATION_QUERY:
                response = await self._handle_medication_query(query, patient_data)
            elif query_type == QueryType.ALLERGY_CHECK:
                response = await self._handle_allergy_check(query, patient_data)
            elif query_type == QueryType.DRUG_INTERACTION:
                response = await self._handle_interaction_check(query, patient_data)
            elif query_type == QueryType.DRUG_NORMALIZE:
                response = await self._handle_drug_normalize(query)
            elif query_type == QueryType.SAFETY_ANALYSIS:
                response = await self._handle_safety_analysis(query, patient_data)
            elif query_type == QueryType.TIMELINE_QUERY:
                response = await self._handle_timeline_query(query, patient_data)
            elif query_type == QueryType.PRESCRIPTION_HISTORY:
                response = await self._handle_prescription_history(query, patient_data)
            elif query_type == QueryType.PATIENT_INFO:
                response = await self._handle_patient_info(query, patient_data)
            elif query_type == QueryType.RECOMMENDATION:
                response = await self._handle_recommendation(query, patient_data)
            else:
                response = await self._handle_general_query(query, patient_data)
            
            # Add processing time
            response.processing_time_ms = (time.time() - start_time) * 1000
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return AIResponse(
                success=False,
                query_type=QueryType.GENERAL_MEDICAL.value,
                answer=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    # ==================== Intent Detection ====================
    
    def _detect_intent(self, query: str) -> QueryType:
        """Detect the type of query from natural language"""
        query_lower = query.lower().strip()
        
        # Score each intent type
        intent_scores = {}
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            if score > 0:
                intent_scores[intent_type] = score
        
        # Return highest scoring intent
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        return QueryType.GENERAL_MEDICAL
    
    def _get_patient_context(self, patient_id: Optional[str], context: Optional[Dict]) -> Dict:
        """Get patient context from the unified database service"""
        patient_data = {}
        
        if patient_id and self.unified_patient_service:
            try:
                # Get comprehensive patient summary from database
                summary = self.unified_patient_service.get_patient_summary(patient_id)
                
                if summary and 'error' not in summary:
                    patient_info = summary.get('patient', {})
                    patient_data = {
                        "patient_id": patient_info.get('patient_uid', patient_id),
                        "name": patient_info.get('name', ''),
                        "age": patient_info.get('age'),
                        "gender": patient_info.get('gender'),
                        "phone": patient_info.get('phone'),
                        "address": patient_info.get('address'),
                        "allergies": summary.get('allergies', []),
                        "chronic_conditions": summary.get('conditions', []),
                        "medications": summary.get('current_medications', []),
                        "all_diagnoses": summary.get('all_diagnoses', []),
                        "treating_doctors": summary.get('treating_doctors', []),
                        "statistics": summary.get('statistics', {}),
                        "timeline": summary.get('recent_timeline', [])
                    }
                    
                    # Also get prescriptions
                    prescriptions = self.unified_patient_service.get_patient_prescriptions(patient_id)
                    patient_data["prescriptions"] = prescriptions
                    
                    # Get historical medications
                    all_meds = self.unified_patient_service.get_patient_medications(patient_id, active_only=False)
                    patient_data["historical_medications"] = [m for m in all_meds if not m.get('is_active', True)]
                    
                    logger.info(f"Loaded patient context from database: {patient_id} with {len(patient_data['medications'])} active meds")
                else:
                    logger.warning(f"Patient {patient_id} not found in database")
                    patient_data = {"patient_id": patient_id}
                    
            except Exception as e:
                logger.error(f"Error loading patient context: {e}")
                patient_data = {"patient_id": patient_id}
        elif patient_id:
            patient_data = {"patient_id": patient_id}
        
        # Merge with provided context (context can add additional data)
        if context:
            for key, value in context.items():
                if key == "medications" and "medications" in patient_data:
                    # Merge medications
                    existing_names = {m.get("name", "").lower() for m in patient_data["medications"] if isinstance(m, dict)}
                    for med in value:
                        if isinstance(med, dict):
                            if med.get("name", "").lower() not in existing_names:
                                patient_data["medications"].append(med)
                        else:
                            patient_data["medications"].append({"name": med})
                elif key == "allergies" and "allergies" in patient_data:
                    for allergy in value:
                        if allergy not in patient_data["allergies"]:
                            patient_data["allergies"].append(allergy)
                else:
                    patient_data[key] = value
        
        return patient_data
    
    # ==================== Query Handlers ====================
    
    async def _handle_medication_query(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle medication-related queries"""
        query_lower = query.lower()
        
        # Check if asking for last medication
        if any(word in query_lower for word in ["last", "recent", "latest"]):
            meds = patient_data.get("medications", [])
            if meds:
                last_med = meds[-1] if isinstance(meds, list) else meds
                if isinstance(last_med, dict):
                    med_info = f"**{last_med.get('name', 'Unknown')}**\n"
                    med_info += f"- Dosage: {last_med.get('dosage', 'N/A')}\n"
                    med_info += f"- Frequency: {last_med.get('frequency', 'N/A')}\n"
                    if last_med.get('prescribed_date'):
                        med_info += f"- Prescribed: {last_med.get('prescribed_date')}"
                    
                    return AIResponse(
                        success=True,
                        query_type=QueryType.MEDICATION_QUERY.value,
                        answer=f"The last prescribed medication is:\n\n{med_info}",
                        data={"last_medication": last_med},
                        confidence=0.95,
                        sources=["Patient medication records"],
                        follow_up_questions=[
                            "Are there any interactions with other medications?",
                            "What are the side effects of this medication?",
                            "Show complete medication history"
                        ]
                    )
                else:
                    return AIResponse(
                        success=True,
                        query_type=QueryType.MEDICATION_QUERY.value,
                        answer=f"The last prescribed medication is: **{last_med}**",
                        data={"last_medication": last_med},
                        confidence=0.9
                    )
            else:
                return AIResponse(
                    success=True,
                    query_type=QueryType.MEDICATION_QUERY.value,
                    answer="No medication records found for this patient.",
                    confidence=0.8,
                    follow_up_questions=["Would you like to add a new prescription?"]
                )
        
        # List all current medications
        if any(word in query_lower for word in ["current", "all", "list", "show", "what"]):
            meds = patient_data.get("medications", [])
            if meds:
                med_list = []
                for i, med in enumerate(meds, 1):
                    if isinstance(med, dict):
                        med_list.append(f"{i}. **{med.get('name', 'Unknown')}** - {med.get('dosage', '')} {med.get('frequency', '')}")
                    else:
                        med_list.append(f"{i}. {med}")
                
                answer = f"📋 **Current Medications** ({len(meds)} total):\n\n" + "\n".join(med_list)
                
                # Check for potential interactions
                warnings = []
                if self.drug_interaction_service and len(meds) >= 2:
                    med_names = [m.get('name', m) if isinstance(m, dict) else m for m in meds]
                    try:
                        safety_result = self.drug_interaction_service.analyze_safety(
                            medications=med_names,
                            patient_allergies=patient_data.get("allergies", []),
                            patient_conditions=patient_data.get("conditions", [])
                        )
                        if hasattr(safety_result, 'interactions') and safety_result.interactions:
                            for interaction in safety_result.interactions:
                                warnings.append({
                                    "type": "interaction",
                                    "severity": getattr(interaction.severity, 'value', 'unknown'),
                                    "message": f"Interaction between {interaction.drug1} and {interaction.drug2}: {interaction.description}"
                                })
                    except Exception as e:
                        logger.warning(f"Could not check interactions: {e}")
                
                return AIResponse(
                    success=True,
                    query_type=QueryType.MEDICATION_QUERY.value,
                    answer=answer,
                    data={"medications": meds, "count": len(meds)},
                    warnings=warnings,
                    confidence=0.95,
                    sources=["Patient medication records"],
                    follow_up_questions=[
                        "Check for drug interactions",
                        "Show medication timeline",
                        "Any allergies to check?"
                    ]
                )
            else:
                return AIResponse(
                    success=True,
                    query_type=QueryType.MEDICATION_QUERY.value,
                    answer="No medications currently recorded for this patient.",
                    confidence=0.8
                )
        
        return AIResponse(
            success=True,
            query_type=QueryType.MEDICATION_QUERY.value,
            answer="I can help you with medication information. Please specify what you'd like to know:\n- Current medications\n- Last prescribed medication\n- Medication history\n- Check for interactions",
            confidence=0.7,
            follow_up_questions=[
                "What are the current medications?",
                "What was the last medication prescribed?",
                "Check for drug interactions"
            ]
        )
    
    async def _handle_allergy_check(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle allergy-related queries and checks"""
        query_lower = query.lower()
        allergies = patient_data.get("allergies", [])
        medications = patient_data.get("medications", [])
        
        # Extract drug name from query if checking specific drug
        drug_match = re.search(r"(?:allergic to|allergy to|check|for)\s+(\w+(?:\s+\w+)?)", query_lower)
        specific_drug = drug_match.group(1) if drug_match else None
        
        # If just listing allergies
        if any(word in query_lower for word in ["what", "list", "show", "any"]) and not specific_drug:
            if allergies:
                allergy_list = ", ".join(f"**{a}**" for a in allergies)
                return AIResponse(
                    success=True,
                    query_type=QueryType.ALLERGY_CHECK.value,
                    answer=f"🚨 **Patient Allergies**:\n\n{allergy_list}",
                    data={"allergies": allergies, "count": len(allergies)},
                    warnings=[{"type": "allergy", "severity": "high", "message": f"Patient has {len(allergies)} known allergies"}],
                    confidence=0.95,
                    sources=["Patient allergy records"],
                    follow_up_questions=[
                        "Check medications against allergies",
                        "Are any current medications contraindicated?"
                    ]
                )
            else:
                return AIResponse(
                    success=True,
                    query_type=QueryType.ALLERGY_CHECK.value,
                    answer="✅ No known allergies recorded for this patient.",
                    data={"allergies": [], "count": 0},
                    confidence=0.8,
                    follow_up_questions=["Would you like to add an allergy?"]
                )
        
        # Check specific drug against allergies
        if specific_drug and self.drug_interaction_service:
            try:
                result = self.drug_interaction_service.analyze_safety(
                    medications=[specific_drug],
                    patient_allergies=allergies,
                    patient_conditions=patient_data.get("conditions", [])
                )
                
                allergy_risks = []
                if hasattr(result, 'allergy_risks'):
                    allergy_risks = result.allergy_risks
                
                if allergy_risks:
                    warnings = []
                    risk_messages = []
                    for risk in allergy_risks:
                        risk_msg = f"⚠️ **{risk.medication}** - Risk due to {risk.allergen} allergy"
                        if hasattr(risk, 'cross_reactivity') and risk.cross_reactivity:
                            risk_msg += f" (Cross-reactivity)"
                        risk_messages.append(risk_msg)
                        warnings.append({
                            "type": "allergy_risk",
                            "severity": "high",
                            "message": risk_msg
                        })
                    
                    return AIResponse(
                        success=True,
                        query_type=QueryType.ALLERGY_CHECK.value,
                        answer=f"🚫 **ALLERGY ALERT**: {specific_drug} may pose a risk!\n\n" + "\n".join(risk_messages),
                        data={"drug": specific_drug, "risks": [{"allergen": r.allergen, "medication": r.medication} for r in allergy_risks]},
                        warnings=warnings,
                        recommendations=[
                            f"Consider alternative to {specific_drug}",
                            "Review patient allergy history",
                            "Consult with allergist if needed"
                        ],
                        confidence=0.95
                    )
                else:
                    return AIResponse(
                        success=True,
                        query_type=QueryType.ALLERGY_CHECK.value,
                        answer=f"✅ No allergy concerns found for **{specific_drug}** based on patient's known allergies.",
                        data={"drug": specific_drug, "safe": True},
                        confidence=0.9
                    )
            except Exception as e:
                logger.error(f"Error checking allergies: {e}")
        
        # Check all medications against allergies
        if medications and allergies and self.drug_interaction_service:
            med_names = [m.get('name', m) if isinstance(m, dict) else m for m in medications]
            try:
                result = self.drug_interaction_service.analyze_safety(
                    medications=med_names,
                    patient_allergies=allergies,
                    patient_conditions=patient_data.get("conditions", [])
                )
                
                if hasattr(result, 'allergy_risks') and result.allergy_risks:
                    warnings = []
                    for risk in result.allergy_risks:
                        warnings.append({
                            "type": "allergy_risk",
                            "severity": "high",
                            "medication": risk.medication,
                            "allergen": risk.allergen
                        })
                    
                    return AIResponse(
                        success=True,
                        query_type=QueryType.ALLERGY_CHECK.value,
                        answer=f"🚨 **{len(result.allergy_risks)} ALLERGY RISKS DETECTED**\n\nReview patient medications immediately!",
                        data={"risks": warnings},
                        warnings=warnings,
                        confidence=0.95
                    )
                else:
                    return AIResponse(
                        success=True,
                        query_type=QueryType.ALLERGY_CHECK.value,
                        answer="✅ All current medications are safe based on known allergies.",
                        data={"checked_medications": med_names, "allergies": allergies},
                        confidence=0.9
                    )
            except Exception as e:
                logger.error(f"Error in allergy check: {e}")
        
        return AIResponse(
            success=True,
            query_type=QueryType.ALLERGY_CHECK.value,
            answer="I can check for allergy risks. Please provide:\n- Patient allergies\n- Medication to check\n\nOr ask me to check current medications against known allergies.",
            confidence=0.7
        )
    
    async def _handle_interaction_check(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle drug interaction queries"""
        medications = patient_data.get("medications", [])
        
        # Extract drug names from query
        drug_pattern = r"(?:between|with|and|take)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)"
        drugs_from_query = re.findall(drug_pattern, query.lower())
        
        # Use medications from query or patient data
        drugs_to_check = drugs_from_query if drugs_from_query else [
            m.get('name', m) if isinstance(m, dict) else m for m in medications
        ]
        
        if len(drugs_to_check) < 2:
            return AIResponse(
                success=True,
                query_type=QueryType.DRUG_INTERACTION.value,
                answer="Please provide at least 2 medications to check for interactions.\n\nExample: 'Check interactions between Aspirin and Warfarin'",
                confidence=0.7,
                follow_up_questions=[
                    "Check interactions between Aspirin and Warfarin",
                    "Are current medications safe together?"
                ]
            )
        
        if self.drug_interaction_service:
            try:
                result = self.drug_interaction_service.analyze_safety(
                    medications=drugs_to_check,
                    patient_allergies=patient_data.get("allergies", []),
                    patient_conditions=patient_data.get("conditions", [])
                )
                
                interactions = []
                warnings = []
                
                if hasattr(result, 'interactions') and result.interactions:
                    for interaction in result.interactions:
                        severity = getattr(interaction.severity, 'value', 'unknown')
                        severity_emoji = {"minor": "🟡", "moderate": "🟠", "major": "🔴", "contraindicated": "⛔"}.get(severity, "⚪")
                        
                        interaction_info = {
                            "drugs": [interaction.drug1, interaction.drug2],
                            "severity": severity,
                            "description": interaction.description,
                            "mechanism": getattr(interaction, 'mechanism', ''),
                            "management": getattr(interaction, 'management', '')
                        }
                        interactions.append(interaction_info)
                        
                        warnings.append({
                            "type": "interaction",
                            "severity": severity,
                            "message": f"{interaction.drug1} + {interaction.drug2}: {interaction.description}"
                        })
                    
                    # Format response
                    response_lines = [f"⚠️ **{len(interactions)} Drug Interactions Found**\n"]
                    for i, interaction in enumerate(interactions, 1):
                        severity_emoji = {"minor": "🟡", "moderate": "🟠", "major": "🔴", "contraindicated": "⛔"}.get(interaction['severity'], "⚪")
                        response_lines.append(f"\n**{i}. {interaction['drugs'][0]} + {interaction['drugs'][1]}** {severity_emoji}")
                        response_lines.append(f"   Severity: **{interaction['severity'].upper()}**")
                        response_lines.append(f"   {interaction['description']}")
                        if interaction['management']:
                            response_lines.append(f"   💊 Management: {interaction['management']}")
                    
                    return AIResponse(
                        success=True,
                        query_type=QueryType.DRUG_INTERACTION.value,
                        answer="\n".join(response_lines),
                        data={"interactions": interactions, "drugs_checked": drugs_to_check},
                        warnings=warnings,
                        recommendations=[
                            "Review medication timing",
                            "Consider alternatives for major interactions",
                            "Monitor patient closely"
                        ],
                        confidence=0.95
                    )
                else:
                    return AIResponse(
                        success=True,
                        query_type=QueryType.DRUG_INTERACTION.value,
                        answer=f"✅ **No significant interactions found** between:\n\n" + ", ".join(f"**{d}**" for d in drugs_to_check),
                        data={"interactions": [], "drugs_checked": drugs_to_check},
                        confidence=0.9
                    )
            except Exception as e:
                logger.error(f"Error checking interactions: {e}")
        
        return AIResponse(
            success=False,
            query_type=QueryType.DRUG_INTERACTION.value,
            answer="Unable to check drug interactions at this time. Please try again.",
            confidence=0.5
        )
    
    async def _handle_drug_normalize(self, query: str) -> AIResponse:
        """Handle drug normalization queries"""
        # Extract drug name from query
        drug_pattern = r"(?:what is|normalize|generic|brand|convert|standardize)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)"
        match = re.search(drug_pattern, query.lower())
        
        if not match:
            # Try to find any capitalized word that might be a drug
            words = query.split()
            drug_name = None
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    drug_name = word
                    break
            if not drug_name:
                return AIResponse(
                    success=True,
                    query_type=QueryType.DRUG_NORMALIZE.value,
                    answer="Please specify a drug name to normalize.\n\nExample: 'What is the generic for Tylenol?'",
                    confidence=0.7
                )
        else:
            drug_name = match.group(1)
        
        if self.drug_normalizer:
            try:
                result = self.drug_normalizer.normalize(drug_name)
                
                response = f"💊 **Drug Information: {drug_name}**\n\n"
                response += f"- **Generic Name**: {result.generic_name}\n"
                response += f"- **Drug Class**: {result.drug_class}\n"
                response += f"- **Is Brand Name**: {'Yes' if result.is_brand else 'No'}\n"
                
                if result.brand_names:
                    response += f"- **Brand Names**: {', '.join(result.brand_names[:5])}\n"
                
                if result.common_dosages:
                    response += f"- **Common Dosages**: {', '.join(result.common_dosages[:3])}\n"
                
                return AIResponse(
                    success=True,
                    query_type=QueryType.DRUG_NORMALIZE.value,
                    answer=response,
                    data={
                        "original_name": result.original_name,
                        "generic_name": result.generic_name,
                        "drug_class": result.drug_class,
                        "brand_names": result.brand_names,
                        "is_brand": result.is_brand,
                        "common_dosages": result.common_dosages
                    },
                    confidence=result.confidence,
                    follow_up_questions=[
                        f"What are alternatives to {result.generic_name}?",
                        f"Check interactions with {result.generic_name}"
                    ]
                )
            except Exception as e:
                logger.error(f"Error normalizing drug: {e}")
        
        return AIResponse(
            success=False,
            query_type=QueryType.DRUG_NORMALIZE.value,
            answer=f"Unable to find information for '{drug_name}'",
            confidence=0.5
        )
    
    async def _handle_safety_analysis(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle comprehensive safety analysis"""
        medications = patient_data.get("medications", [])
        allergies = patient_data.get("allergies", [])
        conditions = patient_data.get("conditions", [])
        
        if not medications:
            return AIResponse(
                success=True,
                query_type=QueryType.SAFETY_ANALYSIS.value,
                answer="No medications found to analyze. Please provide patient medications first.",
                confidence=0.8
            )
        
        med_names = [m.get('name', m) if isinstance(m, dict) else m for m in medications]
        
        if self.drug_interaction_service:
            try:
                result = self.drug_interaction_service.analyze_safety(
                    medications=med_names,
                    patient_allergies=allergies,
                    patient_conditions=conditions,
                    suppress_low_value=True
                )
                
                # Build comprehensive response
                response = "🔬 **Comprehensive Safety Analysis**\n\n"
                response += f"📋 **Medications Analyzed**: {len(med_names)}\n"
                response += f"🚨 **Known Allergies**: {len(allergies) if allergies else 'None'}\n"
                response += f"🏥 **Conditions**: {len(conditions) if conditions else 'None'}\n\n"
                
                warnings = []
                recommendations = []
                
                # Drug Interactions
                if hasattr(result, 'interactions') and result.interactions:
                    response += f"⚠️ **Drug Interactions**: {len(result.interactions)} found\n"
                    for interaction in result.interactions[:3]:  # Top 3
                        severity = getattr(interaction.severity, 'value', 'unknown')
                        response += f"   • {interaction.drug1} + {interaction.drug2} ({severity})\n"
                        warnings.append({
                            "type": "interaction",
                            "severity": severity,
                            "drugs": [interaction.drug1, interaction.drug2]
                        })
                else:
                    response += "✅ **Drug Interactions**: None detected\n"
                
                # Allergy Risks
                if hasattr(result, 'allergy_risks') and result.allergy_risks:
                    response += f"\n🚫 **Allergy Risks**: {len(result.allergy_risks)} found\n"
                    for risk in result.allergy_risks:
                        response += f"   • {risk.medication} - {risk.allergen}\n"
                        warnings.append({
                            "type": "allergy",
                            "severity": "high",
                            "medication": risk.medication
                        })
                    recommendations.append("Review allergy-related medications immediately")
                else:
                    response += "\n✅ **Allergy Risks**: None detected\n"
                
                # Contraindications
                if hasattr(result, 'contraindications') and result.contraindications:
                    response += f"\n⛔ **Contraindications**: {len(result.contraindications)} found\n"
                    for contra in result.contraindications:
                        response += f"   • {contra.medication} - {contra.condition}\n"
                        warnings.append({
                            "type": "contraindication",
                            "severity": "high",
                            "medication": contra.medication
                        })
                else:
                    response += "\n✅ **Contraindications**: None detected\n"
                
                # Duplicates
                if hasattr(result, 'duplicate_therapies') and result.duplicate_therapies:
                    response += f"\n🔄 **Duplicate Therapies**: {len(result.duplicate_therapies)} found\n"
                    recommendations.append("Review duplicate therapy medications")
                
                # Overall safety score
                safety_score = getattr(result, 'safety_score', 0.8)
                if len(warnings) == 0:
                    response += f"\n\n✅ **Overall Status**: SAFE (Score: {safety_score:.0%})"
                elif any(w['severity'] in ['high', 'major', 'contraindicated'] for w in warnings):
                    response += f"\n\n🔴 **Overall Status**: REVIEW REQUIRED (Score: {safety_score:.0%})"
                    recommendations.append("Clinical pharmacist review recommended")
                else:
                    response += f"\n\n🟡 **Overall Status**: CAUTION (Score: {safety_score:.0%})"
                
                return AIResponse(
                    success=True,
                    query_type=QueryType.SAFETY_ANALYSIS.value,
                    answer=response,
                    data={
                        "medications_checked": med_names,
                        "allergies_checked": allergies,
                        "conditions_checked": conditions,
                        "interaction_count": len(result.interactions) if hasattr(result, 'interactions') else 0,
                        "safety_score": safety_score
                    },
                    warnings=warnings,
                    recommendations=recommendations,
                    confidence=0.95,
                    sources=["Drug interaction database", "Allergy database"]
                )
            except Exception as e:
                logger.error(f"Error in safety analysis: {e}")
        
        return AIResponse(
            success=False,
            query_type=QueryType.SAFETY_ANALYSIS.value,
            answer="Unable to perform safety analysis at this time.",
            confidence=0.5
        )
    
    async def _handle_timeline_query(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle medication timeline queries"""
        medications = patient_data.get("medications", [])
        prescriptions = patient_data.get("prescriptions", [])
        
        if not medications and not prescriptions:
            return AIResponse(
                success=True,
                query_type=QueryType.TIMELINE_QUERY.value,
                answer="No medication history found for this patient.",
                confidence=0.8
            )
        
        # Build timeline from available data
        timeline_events = []
        for med in medications:
            if isinstance(med, dict):
                event = {
                    "medication": med.get('name', 'Unknown'),
                    "date": med.get('prescribed_date', med.get('start_date', 'Unknown')),
                    "action": "prescribed",
                    "dosage": med.get('dosage', ''),
                    "prescriber": med.get('prescriber', '')
                }
                timeline_events.append(event)
        
        response = "📅 **Medication Timeline**\n\n"
        if timeline_events:
            for event in sorted(timeline_events, key=lambda x: x.get('date', ''), reverse=True)[:10]:
                response += f"• **{event['date']}**: {event['medication']} {event['dosage']}\n"
                if event['prescriber']:
                    response += f"  _Prescribed by: {event['prescriber']}_\n"
        else:
            response += "No dated medication records available."
        
        return AIResponse(
            success=True,
            query_type=QueryType.TIMELINE_QUERY.value,
            answer=response,
            data={"timeline": timeline_events},
            confidence=0.85,
            follow_up_questions=[
                "Any medication changes in the last month?",
                "Show current medications"
            ]
        )
    
    async def _handle_prescription_history(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle prescription history queries"""
        prescriptions = patient_data.get("prescriptions", [])
        medications = patient_data.get("medications", [])
        
        if not prescriptions and not medications:
            return AIResponse(
                success=True,
                query_type=QueryType.PRESCRIPTION_HISTORY.value,
                answer="No prescription history found for this patient.",
                confidence=0.8
            )
        
        response = "📋 **Prescription History**\n\n"
        
        if prescriptions:
            for i, rx in enumerate(prescriptions[:5], 1):
                if isinstance(rx, dict):
                    response += f"**Prescription #{i}**\n"
                    response += f"• Date: {rx.get('date', 'Unknown')}\n"
                    response += f"• Doctor: {rx.get('prescriber', 'Unknown')}\n"
                    if rx.get('medications'):
                        meds = rx.get('medications', [])
                        response += f"• Medications: {len(meds)} items\n"
                    response += "\n"
        elif medications:
            response += f"Found **{len(medications)}** medications in history:\n\n"
            for med in medications[:10]:
                if isinstance(med, dict):
                    response += f"• {med.get('name', 'Unknown')} - {med.get('dosage', '')}\n"
                else:
                    response += f"• {med}\n"
        
        return AIResponse(
            success=True,
            query_type=QueryType.PRESCRIPTION_HISTORY.value,
            answer=response,
            data={"prescriptions": prescriptions, "medications": medications},
            confidence=0.85
        )
    
    async def _handle_patient_info(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle patient information queries"""
        if not patient_data or patient_data.get('patient_id') is None:
            return AIResponse(
                success=True,
                query_type=QueryType.PATIENT_INFO.value,
                answer="No patient selected. Please specify a patient ID or load patient data.",
                confidence=0.8
            )
        
        response = "👤 **Patient Information**\n\n"
        response += f"• **Patient ID**: {patient_data.get('patient_id', 'Unknown')}\n"
        
        if patient_data.get('name'):
            response += f"• **Name**: {patient_data.get('name')}\n"
        if patient_data.get('age'):
            response += f"• **Age**: {patient_data.get('age')}\n"
        if patient_data.get('gender'):
            response += f"• **Gender**: {patient_data.get('gender')}\n"
        
        if patient_data.get('allergies'):
            response += f"• **Allergies**: {', '.join(patient_data.get('allergies'))}\n"
        else:
            response += "• **Allergies**: None recorded\n"
        
        if patient_data.get('conditions'):
            response += f"• **Conditions**: {', '.join(patient_data.get('conditions'))}\n"
        
        med_count = len(patient_data.get('medications', []))
        response += f"• **Current Medications**: {med_count}\n"
        
        return AIResponse(
            success=True,
            query_type=QueryType.PATIENT_INFO.value,
            answer=response,
            data=patient_data,
            confidence=0.95,
            follow_up_questions=[
                "What medications is the patient taking?",
                "Check for drug interactions",
                "Run safety analysis"
            ]
        )
    
    async def _handle_recommendation(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle recommendation queries"""
        query_lower = query.lower()
        
        # Extract drug name if asking for alternatives
        drug_match = re.search(r"(?:alternative|substitute|replace|instead of)\s+([A-Za-z]+)", query_lower)
        
        if drug_match and self.drug_normalizer:
            drug_name = drug_match.group(1)
            try:
                alternatives = self.drug_normalizer.get_therapeutic_alternatives(drug_name)
                normalized = self.drug_normalizer.normalize(drug_name)
                
                if alternatives:
                    response = f"💊 **Alternatives to {drug_name}**\n\n"
                    response += f"Drug Class: {normalized.drug_class}\n\n"
                    response += "**Therapeutic Alternatives:**\n"
                    for alt in alternatives[:5]:
                        response += f"• {alt}\n"
                    
                    return AIResponse(
                        success=True,
                        query_type=QueryType.RECOMMENDATION.value,
                        answer=response,
                        data={"drug": drug_name, "alternatives": alternatives},
                        recommendations=[
                            "Consider patient-specific factors when selecting alternative",
                            "Check for interactions with current medications"
                        ],
                        confidence=0.9
                    )
            except Exception as e:
                logger.error(f"Error getting alternatives: {e}")
        
        return AIResponse(
            success=True,
            query_type=QueryType.RECOMMENDATION.value,
            answer="I can help with recommendations. Please specify:\n- What medication needs an alternative\n- Any specific requirements or constraints",
            confidence=0.7
        )
    
    async def _handle_general_query(self, query: str, patient_data: Dict) -> AIResponse:
        """Handle general medical queries"""
        # Try to use Gemini for general questions if available
        if self.gemini_service and self.gemini_service.initialized:
            try:
                # Simple Gemini query for general medical info
                prompt = f"""You are a helpful medical assistant for doctors. Answer this question concisely and professionally:

Question: {query}

Patient Context:
- Current Medications: {len(patient_data.get('medications', []))} medications
- Known Allergies: {', '.join(patient_data.get('allergies', [])) or 'None'}
- Conditions: {', '.join(patient_data.get('conditions', [])) or 'None'}

Provide a helpful, medically accurate response. Keep it concise."""
                
                # Use Gemini to generate response
                response = self.gemini_service.model.generate_content(prompt)
                
                return AIResponse(
                    success=True,
                    query_type=QueryType.GENERAL_MEDICAL.value,
                    answer=response.text,
                    confidence=0.8,
                    sources=["AI Medical Assistant"],
                    follow_up_questions=[
                        "Would you like more details?",
                        "Should I check for drug interactions?",
                        "Run a safety analysis?"
                    ]
                )
            except Exception as e:
                logger.warning(f"Gemini query failed: {e}")
        
        # Fallback response
        return AIResponse(
            success=True,
            query_type=QueryType.GENERAL_MEDICAL.value,
            answer=f"""I understand you're asking: "{query}"

I can help you with:
• **Patient medications** - "What medications is the patient taking?"
• **Allergy checks** - "Does the patient have any allergies?"
• **Drug interactions** - "Check interactions between medications"
• **Drug information** - "What is the generic for Lipitor?"
• **Safety analysis** - "Run comprehensive safety analysis"
• **Timeline** - "Show medication history"

Please try rephrasing your question or choose one of the options above.""",
            confidence=0.6,
            follow_up_questions=[
                "What medications is the patient taking?",
                "Run safety analysis",
                "Check for drug interactions"
            ]
        )
    
    # ==================== Patient Data Management ====================
    
    def set_patient_data(self, patient_id: str, data: Dict):
        """Store patient data for context - uses unified database service"""
        if not self.unified_patient_service:
            logger.warning("Unified patient service not available")
            return
        
        try:
            self.unified_patient_service.get_or_create_patient(
                patient_uid=patient_id,
                name=data.get('name'),
                age=data.get('age'),
                gender=data.get('gender'),
                phone=data.get('phone'),
                address=data.get('address'),
                allergies=data.get('allergies', []),
                conditions=data.get('conditions', [])
            )
            logger.info(f"Patient data saved to database: {patient_id}")
        except Exception as e:
            logger.error(f"Could not save patient data: {e}")
    
    def add_medication(self, patient_id: str, medication: Dict):
        """Add a medication to patient's record via database"""
        # Medications are added via prescriptions, not directly
        logger.info(f"Medication add request for {patient_id}: {medication.get('name')}")
    
    def add_allergy(self, patient_id: str, allergy: str):
        """Add an allergy to patient's record"""
        if not self.unified_patient_service:
            logger.warning("Unified patient service not available")
            return
        
        try:
            # Get current allergies and add new one
            self.unified_patient_service.get_or_create_patient(
                patient_uid=patient_id,
                allergies=[allergy]
            )
            logger.info(f"Allergy added for {patient_id}: {allergy}")
        except Exception as e:
            logger.error(f"Could not add allergy: {e}")
    
    def get_patient_data(self, patient_id: str) -> Dict:
        """Get patient data from database"""
        if not self.unified_patient_service:
            return {"patient_id": patient_id}
        
        try:
            summary = self.unified_patient_service.get_patient_summary(patient_id)
            if summary and 'error' not in summary:
                return summary
        except Exception as e:
            logger.error(f"Could not get patient data: {e}")
        
        return {"patient_id": patient_id}


# Singleton instance
_ai_assistant = None


def get_ai_assistant() -> MedicalAIAssistant:
    """Get or create AI Assistant singleton"""
    global _ai_assistant
    if _ai_assistant is None:
        _ai_assistant = MedicalAIAssistant()
    return _ai_assistant
