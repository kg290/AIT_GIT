"""
Document Processor - Orchestrates the complete document processing pipeline
Uses Google Vision OCR + Gemini AI for robust prescription understanding
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import uuid
import time
import os

from sqlalchemy.orm import Session

from backend.config import settings
from backend.services.ocr_service import OCRService
from backend.services.gemini_service import GeminiService
from backend.services.text_cleaning_service import TextCleaningService
from backend.services.entity_extraction_service import EntityExtractionService
from backend.services.drug_normalization_service import DrugNormalizationService
from backend.services.prescription_structuring_service import PrescriptionStructuringService
from backend.services.drug_interaction_service import DrugInteractionService
from backend.services.temporal_reasoning_service import TemporalReasoningService
from backend.services.knowledge_graph_service import KnowledgeGraphService
from backend.services.audit_service import AuditService

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Complete result of document processing"""
    document_id: str
    success: bool
    
    # OCR Results
    ocr_text: str
    ocr_confidence: float
    is_handwritten: bool
    has_mixed_content: bool
    
    # Cleaned text
    cleaned_text: str
    cleaning_confidence: float
    corrections_made: List[Dict]
    
    # Extracted entities
    medications: List[Dict]
    dosages: List[Dict]
    frequencies: List[Dict]
    diagnoses: List[Dict]
    symptoms: List[Dict]
    vitals: List[Dict]
    dates: List[Dict]
    
    # Normalized medications
    normalized_medications: List[Dict]
    duplicates_found: List[Dict]
    
    # Structured prescription
    prescription: Dict
    prescription_confidence: float
    
    # Safety analysis
    drug_interactions: List[Dict]
    allergy_alerts: List[Dict]
    duplicate_therapies: List[Dict]
    safety_score: float
    
    # Timeline
    timeline_entries: List[Dict]
    
    # Explainability
    reasoning_steps: List[str]
    confidence_explanation: Dict
    
    # Processing metadata
    processing_time_ms: int
    flags_for_review: List[str]
    
    # Errors/warnings
    errors: List[str]
    warnings: List[str]


class DocumentProcessor:
    """
    Main document processing orchestrator
    
    Coordinates all services to process a medical document through:
    1. OCR extraction (Google Vision)
    2. Gemini AI analysis (for robust prescription understanding)
    3. Text cleaning
    4. Entity extraction
    5. Drug normalization
    6. Prescription structuring
    7. Drug safety analysis
    8. Timeline integration
    9. Knowledge graph update
    """
    
    def __init__(self, db: Session = None):
        self.db = db
        
        # Initialize all services
        self.ocr_service = OCRService()
        self.gemini_service = GeminiService()  # Gemini for AI-powered prescription understanding
        self.text_cleaning = TextCleaningService()
        self.entity_extraction = EntityExtractionService()
        self.drug_normalization = DrugNormalizationService()
        self.prescription_structuring = PrescriptionStructuringService()
        self.drug_interaction = DrugInteractionService()
        self.temporal_reasoning = TemporalReasoningService()
        self.knowledge_graph = KnowledgeGraphService()
        self.audit = AuditService(db)
    
    def process_document(self, file_path: str, patient_id: str = None,
                        patient_allergies: List[str] = None,
                        current_medications: List[Dict] = None,
                        force_review: bool = False) -> ProcessingResult:
        """
        Process a medical document through the complete pipeline
        
        Args:
            file_path: Path to the document file
            patient_id: Optional patient ID for context
            patient_allergies: List of known patient allergies
            current_medications: List of current medications
            force_review: Always flag for human review
            
        Returns:
            ProcessingResult with all extracted and analyzed data
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        reasoning_steps = []
        errors = []
        warnings = []
        flags_for_review = []
        
        if force_review:
            flags_for_review.append("Manual review requested")
        
        # Log document upload
        self.audit.log_document_upload(
            document_id=document_id,
            filename=Path(file_path).name,
            patient_id=patient_id
        )
        
        reasoning_steps.append(f"Started processing document: {Path(file_path).name}")
        
        # ==================== Step 1: OCR ====================
        reasoning_steps.append("Step 1: Performing OCR text extraction")
        
        try:
            ocr_result = self.ocr_service.extract_text(file_path)
            ocr_text = ocr_result.full_text
            ocr_confidence = ocr_result.confidence
            is_handwritten = ocr_result.is_handwritten
            has_mixed_content = ocr_result.has_mixed_content
            
            reasoning_steps.append(f"OCR completed with {ocr_confidence:.2%} confidence")
            
            if ocr_confidence < settings.OCR_CONFIDENCE_THRESHOLD:
                flags_for_review.append(f"Low OCR confidence: {ocr_confidence:.2%}")
                warnings.append("OCR quality may be poor - manual verification recommended")
            
            if is_handwritten:
                flags_for_review.append("Handwritten content detected")
                reasoning_steps.append("⚠️ Document contains handwritten text")
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            errors.append(f"OCR extraction failed: {str(e)}")
            ocr_text = ""
            ocr_confidence = 0.0
            is_handwritten = False
            has_mixed_content = False
        
        # ==================== Step 1.5: Gemini AI Analysis ====================
        # Use Gemini for intelligent prescription understanding
        gemini_result = None
        gemini_medications = []
        
        reasoning_steps.append("Step 1.5: Using Gemini AI for prescription analysis")
        
        try:
            # First try direct image analysis with Gemini (works best for any format)
            gemini_result = self.gemini_service.analyze_image(file_path)
            
            if gemini_result.success:
                reasoning_steps.append(f"✓ Gemini AI analysis successful (confidence: {gemini_result.confidence:.0%})")
                gemini_medications = gemini_result.medications
                
                # If OCR failed but Gemini succeeded, use Gemini's raw response as text
                if not ocr_text and gemini_result.raw_response:
                    reasoning_steps.append("Using Gemini analysis to supplement missing OCR text")
                
                # Log notes from Gemini
                for note in gemini_result.notes:
                    warnings.append(f"Gemini note: {note}")
            else:
                # Fallback: If image analysis failed, try with OCR text
                if ocr_text:
                    reasoning_steps.append("Gemini image analysis failed, trying with OCR text...")
                    gemini_result = self.gemini_service.analyze_text(ocr_text)
                    if gemini_result.success:
                        reasoning_steps.append(f"✓ Gemini text analysis successful")
                        gemini_medications = gemini_result.medications
                    else:
                        warnings.append(f"Gemini analysis warning: {gemini_result.error}")
                else:
                    warnings.append(f"Gemini analysis failed: {gemini_result.error}")
                    
        except Exception as e:
            logger.error(f"Gemini AI analysis failed: {e}")
            warnings.append(f"Gemini AI analysis failed: {str(e)}")
        
        # ==================== Step 2: Text Cleaning ====================
        reasoning_steps.append("Step 2: Cleaning and normalizing text")
        
        try:
            # If OCR text exists, use Gemini to enhance it
            if ocr_text:
                enhanced_text = self.gemini_service.enhance_ocr_text(ocr_text)
                cleaning_result = self.text_cleaning.clean_text(enhanced_text)
            else:
                # No OCR text - create summary from Gemini if available
                if gemini_result and gemini_result.success:
                    # Create text summary from Gemini extraction
                    text_parts = []
                    if gemini_result.patient_name:
                        text_parts.append(f"Patient: {gemini_result.patient_name}")
                    if gemini_result.prescription_date:
                        text_parts.append(f"Date: {gemini_result.prescription_date}")
                    if gemini_result.prescriber_name:
                        text_parts.append(f"Doctor: {gemini_result.prescriber_name}")
                    if gemini_result.diagnoses:
                        text_parts.append(f"Diagnosis: {', '.join(gemini_result.diagnoses)}")
                    for med in gemini_medications:
                        med_str = f"Rx: {med.get('medication_name', '')} {med.get('dosage', '')} {med.get('frequency', '')} x {med.get('duration', '')}"
                        text_parts.append(med_str)
                    ocr_text = "\n".join(text_parts)
                    ocr_confidence = gemini_result.confidence
                    cleaning_result = self.text_cleaning.clean_text(ocr_text)
                else:
                    # No text at all - return error
                    errors.append("No text could be extracted from the document")
                    return self._error_result(document_id, errors, time.time() - start_time)
            
            cleaned_text = cleaning_result.cleaned_text
            cleaning_confidence = cleaning_result.confidence
            corrections_made = [
                {'original': c.original, 'corrected': c.corrected, 'type': c.correction_type.value}
                for c in cleaning_result.corrections
            ]
            
            if corrections_made:
                reasoning_steps.append(f"Made {len(corrections_made)} text corrections")
            
            # Check for low confidence after cleaning
            if cleaning_confidence < 0.7:
                flags_for_review.append("Low text confidence after cleaning")
                
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            warnings.append(f"Text cleaning failed: {str(e)}")
            cleaned_text = ocr_text
            cleaning_confidence = ocr_confidence
            corrections_made = []
        
        # ==================== Step 3: Entity Extraction ====================
        reasoning_steps.append("Step 3: Extracting medical entities")
        
        try:
            # Use Gemini results if available and successful
            if gemini_result and gemini_result.success and gemini_medications:
                # Use Gemini-extracted medications
                medications = [
                    {
                        'text': m.get('medication_name', ''),
                        'brand_name': m.get('brand_name', ''),
                        'dosage': m.get('dosage', ''),
                        'form': m.get('form', ''),
                        'frequency': m.get('frequency', ''),
                        'timing': m.get('timing', ''),
                        'duration': m.get('duration', ''),
                        'route': m.get('route', 'oral'),
                        'special_instructions': m.get('special_instructions', ''),
                        'confidence': gemini_result.confidence,
                        'source': 'gemini'
                    }
                    for m in gemini_medications if m.get('medication_name')
                ]
                
                # Extract dosages and frequencies from Gemini medications
                dosages = [
                    {'text': m.get('dosage', ''), 'confidence': gemini_result.confidence}
                    for m in gemini_medications if m.get('dosage')
                ]
                frequencies = [
                    {'text': m.get('frequency', ''), 'confidence': gemini_result.confidence}
                    for m in gemini_medications if m.get('frequency')
                ]
                
                # Use Gemini diagnoses
                diagnoses = [
                    {'text': d, 'confidence': gemini_result.confidence, 'source': 'gemini'}
                    for d in gemini_result.diagnoses if d
                ]
                
                # Use Gemini symptoms
                symptoms = [
                    {'text': s, 'confidence': gemini_result.confidence, 'source': 'gemini'}
                    for s in gemini_result.symptoms if s
                ]
                
                # Use Gemini vitals
                vitals = [
                    {'type': k, 'value': v, 'confidence': gemini_result.confidence}
                    for k, v in (gemini_result.vitals or {}).items() if v
                ]
                
                # Extract dates if available
                dates = []
                if gemini_result.prescription_date:
                    dates.append({'text': gemini_result.prescription_date, 'type': 'prescription_date'})
                if gemini_result.follow_up:
                    dates.append({'text': gemini_result.follow_up, 'type': 'follow_up'})
                
                reasoning_steps.append(
                    f"✓ Extracted via Gemini AI: {len(medications)} medications, {len(diagnoses)} diagnoses, "
                    f"{len(symptoms)} symptoms, {len(vitals)} vitals"
                )
            else:
                # Fallback to regex-based extraction
                extraction_result = self.entity_extraction.extract_entities(cleaned_text)
                
                medications = [m.to_dict() for m in extraction_result.medications]
                dosages = [d.to_dict() for d in extraction_result.dosages]
                frequencies = [f.to_dict() for f in extraction_result.frequencies]
                diagnoses = [d.to_dict() for d in extraction_result.diagnoses]
                symptoms = [s.to_dict() for s in extraction_result.symptoms]
                vitals = [v.to_dict() for v in extraction_result.vitals]
                dates = [d.to_dict() for d in extraction_result.dates]
                
                reasoning_steps.append(
                    f"Extracted via regex: {len(medications)} medications, {len(diagnoses)} diagnoses, "
                    f"{len(symptoms)} symptoms, {len(vitals)} vitals"
                )
            
            # Check for low confidence entities
            low_confidence_meds = [m for m in medications if m.get('confidence', 1.0) < 0.7]
            if low_confidence_meds:
                flags_for_review.append(f"{len(low_confidence_meds)} medications with low confidence")
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            errors.append(f"Entity extraction failed: {str(e)}")
            medications = dosages = frequencies = diagnoses = symptoms = vitals = dates = []
        
        # ==================== Step 4: Drug Normalization ====================
        reasoning_steps.append("Step 4: Normalizing medication names")
        
        try:
            normalized_medications = []
            duplicates_found = []
            
            for med in medications:
                norm_result = self.drug_normalization.normalize(med['text'])
                normalized_medications.append({
                    'original': med['text'],
                    'generic_name': norm_result.generic_name,
                    'brand_names': norm_result.brand_names,
                    'drug_class': norm_result.drug_class,
                    'normalized_form': norm_result.normalized_form,
                    'confidence': norm_result.confidence
                })
            
            # Check for duplicates
            if normalized_medications:
                dup_result = self.drug_normalization.find_duplicates(
                    [m['original'] for m in medications]
                )
                duplicates_found = [
                    {'drug1': d[0], 'drug2': d[1], 'reason': d[2]} 
                    for d in dup_result
                ]
                
                if duplicates_found:
                    flags_for_review.append(f"Found {len(duplicates_found)} duplicate medications")
                    reasoning_steps.append(f"⚠️ {len(duplicates_found)} duplicate medications detected")
                    
        except Exception as e:
            logger.error(f"Drug normalization failed: {e}")
            warnings.append(f"Drug normalization failed: {str(e)}")
            normalized_medications = []
            duplicates_found = []
        
        # ==================== Step 5: Prescription Structuring ====================
        reasoning_steps.append("Step 5: Structuring prescription data")
        
        try:
            prescription_result = self.prescription_structuring.structure_prescription(cleaned_text)
            
            prescription = {
                'header': prescription_result.header.to_dict() if prescription_result.header else None,
                'medications': [item.to_dict() for item in prescription_result.medication_items],
                'overall_confidence': prescription_result.overall_confidence,
                'missing_fields': prescription_result.missing_fields,
                'warnings': prescription_result.warnings
            }
            prescription_confidence = prescription_result.overall_confidence
            
            if prescription_result.warnings:
                for w in prescription_result.warnings:
                    warnings.append(w)
                    
            if prescription_confidence < 0.7:
                flags_for_review.append("Low prescription structuring confidence")
                
        except Exception as e:
            logger.error(f"Prescription structuring failed: {e}")
            warnings.append(f"Prescription structuring failed: {str(e)}")
            prescription = {}
            prescription_confidence = 0.0
        
        # ==================== Step 6: Drug Safety Analysis ====================
        reasoning_steps.append("Step 6: Performing drug safety analysis")
        
        try:
            # Get medication names for safety check
            med_names = [m.get('generic_name') or m.get('original', '') for m in normalized_medications]
            
            # Add current medications if provided
            if current_medications:
                for med in current_medications:
                    name = med.get('generic_name') or med.get('medication_name', '')
                    if name and name not in med_names:
                        med_names.append(name)
            
            safety_result = self.drug_interaction.analyze_medications(
                medications=med_names,
                patient_allergies=patient_allergies or []
            )
            
            drug_interactions = [
                {
                    'drug1': i.drug1,
                    'drug2': i.drug2,
                    'severity': i.severity.value,
                    'description': i.description,
                    'mechanism': i.mechanism,
                    'management': i.management
                }
                for i in safety_result.interactions
            ]
            
            allergy_alerts = [
                {
                    'drug': a.drug,
                    'allergen': a.allergen,
                    'severity': a.severity.value,
                    'description': a.description
                }
                for a in safety_result.allergy_alerts
            ]
            
            duplicate_therapies = [
                {
                    'drug1': d.drug1,
                    'drug2': d.drug2,
                    'reason': d.reason
                }
                for d in safety_result.duplicate_therapies
            ]
            
            safety_score = safety_result.overall_safety_score
            
            # Flag for review if safety issues
            if safety_result.interactions:
                major_interactions = [i for i in safety_result.interactions 
                                     if i.severity.value in ['major', 'contraindicated']]
                if major_interactions:
                    flags_for_review.append(f"{len(major_interactions)} MAJOR drug interactions")
                    reasoning_steps.append(f"⚠️ CRITICAL: {len(major_interactions)} major drug interactions found")
            
            if allergy_alerts:
                flags_for_review.append(f"{len(allergy_alerts)} allergy alerts")
                reasoning_steps.append(f"⚠️ {len(allergy_alerts)} potential allergy concerns")
                
        except Exception as e:
            logger.error(f"Drug safety analysis failed: {e}")
            warnings.append(f"Drug safety analysis failed: {str(e)}")
            drug_interactions = []
            allergy_alerts = []
            duplicate_therapies = []
            safety_score = 0.0
        
        # ==================== Step 7: Timeline Integration ====================
        reasoning_steps.append("Step 7: Building temporal timeline")
        
        try:
            # Build medication list for temporal analysis
            temporal_meds = []
            for med in prescription.get('medications', []):
                temporal_meds.append({
                    'medication_name': med.get('medication_name', ''),
                    'dosage': med.get('dosage', ''),
                    'frequency': med.get('frequency', ''),
                    'duration': med.get('duration', '')
                })
            
            # Get prescription date if available
            prescription_date = None
            if dates:
                prescription_date = dates[0].get('normalized_date')
            
            temporal_result = self.temporal_reasoning.analyze_prescription_timeline(
                medications=temporal_meds,
                prescription_date=prescription_date
            )
            
            timeline_entries = [
                {
                    'event_date': e.event_date.isoformat() if e.event_date else None,
                    'event_type': e.event_type,
                    'title': e.title,
                    'description': e.description,
                    'medications': e.medications,
                    'source_document': document_id
                }
                for e in temporal_result.timeline
            ]
            
        except Exception as e:
            logger.error(f"Timeline integration failed: {e}")
            warnings.append(f"Timeline integration failed: {str(e)}")
            timeline_entries = []
        
        # ==================== Step 8: Knowledge Graph Update ====================
        if patient_id:
            reasoning_steps.append("Step 8: Updating knowledge graph")
            
            try:
                # Create patient node if needed
                self.knowledge_graph.create_patient_node(patient_id, {
                    'last_updated': datetime.utcnow().isoformat()
                })
                
                # Link medications
                for med in normalized_medications:
                    self.knowledge_graph.link_patient_medication(
                        patient_id=patient_id,
                        medication_name=med.get('generic_name') or med.get('original'),
                        properties={'source_document': document_id}
                    )
                
                # Link diagnoses
                for dx in diagnoses:
                    self.knowledge_graph.link_patient_condition(
                        patient_id=patient_id,
                        condition_name=dx.get('text', '')
                    )
                    
            except Exception as e:
                logger.error(f"Knowledge graph update failed: {e}")
                warnings.append(f"Knowledge graph update failed: {str(e)}")
        
        # ==================== Build Confidence Explanation ====================
        confidence_explanation = {
            'ocr_confidence': {
                'score': ocr_confidence,
                'factors': [
                    'Image quality' if ocr_confidence > 0.8 else 'Poor image quality detected',
                    'Handwritten text' if is_handwritten else 'Printed text'
                ]
            },
            'extraction_confidence': {
                'score': cleaning_confidence,
                'factors': [f"{len(corrections_made)} corrections applied"]
            },
            'prescription_confidence': {
                'score': prescription_confidence,
                'factors': prescription.get('warnings', [])
            },
            'overall_confidence': {
                'score': (ocr_confidence + cleaning_confidence + prescription_confidence) / 3,
                'recommendation': 'Review recommended' if flags_for_review else 'Auto-processing suitable'
            }
        }
        
        # ==================== Calculate Processing Time ====================
        processing_time_ms = int((time.time() - start_time) * 1000)
        reasoning_steps.append(f"Processing completed in {processing_time_ms}ms")
        
        # Log processing completion
        self.audit.log_document_processing(
            document_id=document_id,
            processing_type='full_pipeline',
            result_summary={
                'medications_found': len(medications),
                'interactions_found': len(drug_interactions),
                'flags': len(flags_for_review)
            },
            processing_time_ms=processing_time_ms
        )
        
        return ProcessingResult(
            document_id=document_id,
            success=len(errors) == 0,
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            is_handwritten=is_handwritten,
            has_mixed_content=has_mixed_content,
            cleaned_text=cleaned_text,
            cleaning_confidence=cleaning_confidence,
            corrections_made=corrections_made,
            medications=medications,
            dosages=dosages,
            frequencies=frequencies,
            diagnoses=diagnoses,
            symptoms=symptoms,
            vitals=vitals,
            dates=dates,
            normalized_medications=normalized_medications,
            duplicates_found=duplicates_found,
            prescription=prescription,
            prescription_confidence=prescription_confidence,
            drug_interactions=drug_interactions,
            allergy_alerts=allergy_alerts,
            duplicate_therapies=duplicate_therapies,
            safety_score=safety_score,
            timeline_entries=timeline_entries,
            reasoning_steps=reasoning_steps,
            confidence_explanation=confidence_explanation,
            processing_time_ms=processing_time_ms,
            flags_for_review=flags_for_review,
            errors=errors,
            warnings=warnings
        )
    
    def _error_result(self, document_id: str, errors: List[str], 
                      processing_time: float) -> ProcessingResult:
        """Create an error result"""
        return ProcessingResult(
            document_id=document_id,
            success=False,
            ocr_text="",
            ocr_confidence=0.0,
            is_handwritten=False,
            has_mixed_content=False,
            cleaned_text="",
            cleaning_confidence=0.0,
            corrections_made=[],
            medications=[],
            dosages=[],
            frequencies=[],
            diagnoses=[],
            symptoms=[],
            vitals=[],
            dates=[],
            normalized_medications=[],
            duplicates_found=[],
            prescription={},
            prescription_confidence=0.0,
            drug_interactions=[],
            allergy_alerts=[],
            duplicate_therapies=[],
            safety_score=0.0,
            timeline_entries=[],
            reasoning_steps=["Processing failed due to errors"],
            confidence_explanation={},
            processing_time_ms=int(processing_time * 1000),
            flags_for_review=["Processing failed - manual review required"],
            errors=errors,
            warnings=[]
        )
    
    def to_dict(self, result: ProcessingResult) -> Dict:
        """Convert ProcessingResult to dictionary"""
        return {
            'document_id': result.document_id,
            'success': result.success,
            'ocr': {
                'text': result.ocr_text,
                'confidence': result.ocr_confidence,
                'is_handwritten': result.is_handwritten,
                'has_mixed_content': result.has_mixed_content
            },
            'cleaned': {
                'text': result.cleaned_text,
                'confidence': result.cleaning_confidence,
                'corrections': result.corrections_made
            },
            'entities': {
                'medications': result.medications,
                'dosages': result.dosages,
                'frequencies': result.frequencies,
                'diagnoses': result.diagnoses,
                'symptoms': result.symptoms,
                'vitals': result.vitals,
                'dates': result.dates
            },
            'normalized_medications': result.normalized_medications,
            'duplicates': result.duplicates_found,
            'prescription': result.prescription,
            'safety': {
                'drug_interactions': result.drug_interactions,
                'allergy_alerts': result.allergy_alerts,
                'duplicate_therapies': result.duplicate_therapies,
                'safety_score': result.safety_score
            },
            'timeline': result.timeline_entries,
            'explainability': {
                'reasoning_steps': result.reasoning_steps,
                'confidence_explanation': result.confidence_explanation
            },
            'metadata': {
                'processing_time_ms': result.processing_time_ms,
                'flags_for_review': result.flags_for_review,
                'errors': result.errors,
                'warnings': result.warnings
            }
        }
