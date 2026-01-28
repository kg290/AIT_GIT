"""
Simple Document Processor - Production-ready prescription processing
Focused on reliable extraction for hospital use
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
import uuid
import time

from backend.config import settings
from backend.services.ocr_service import OCRService
from backend.services.prescription_extractor import PrescriptionExtractor, ExtractedPrescription
from backend.services.drug_interaction_service import DrugInteractionService

logger = logging.getLogger(__name__)


@dataclass
class SimpleProcessingResult:
    """Simplified processing result for hospital use"""
    # Identifiers
    document_id: str
    success: bool
    
    # Patient Information
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    patient_id: Optional[str] = None
    
    # Prescription Info
    prescription_date: Optional[str] = None
    
    # Doctor Information
    doctor_name: Optional[str] = None
    doctor_qualification: Optional[str] = None
    clinic_name: Optional[str] = None
    clinic_phone: Optional[str] = None
    
    # Clinical Data
    diagnosis: List[str] = field(default_factory=list)
    vitals: Dict[str, str] = field(default_factory=dict)
    
    # Medications (main focus)
    medications: List[Dict] = field(default_factory=list)
    
    # Additional Info
    advice: List[str] = field(default_factory=list)
    follow_up: Optional[str] = None
    
    # Safety
    drug_interactions: List[Dict] = field(default_factory=list)
    safety_alerts: List[str] = field(default_factory=list)
    
    # Raw Data
    raw_ocr_text: str = ""
    
    # Metadata
    confidence: float = 0.0
    processing_time_ms: int = 0
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SimpleDocumentProcessor:
    """
    Streamlined document processor for hospital use
    
    Pipeline:
    1. OCR extraction (Google Vision)
    2. Prescription parsing (regex-based, reliable)
    3. Drug safety check
    4. Return structured result
    """
    
    def __init__(self):
        self.ocr_service = OCRService()
        self.extractor = PrescriptionExtractor()
        self.drug_service = DrugInteractionService()
    
    def process(self, file_path: str, patient_allergies: List[str] = None) -> SimpleProcessingResult:
        """
        Process a prescription document
        
        Args:
            file_path: Path to the prescription image/PDF
            patient_allergies: Known patient allergies for safety check
            
        Returns:
            SimpleProcessingResult with extracted data
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        result = SimpleProcessingResult(
            document_id=document_id,
            success=False
        )
        
        try:
            # Step 1: OCR
            logger.info(f"Processing document: {file_path}")
            ocr_result = self.ocr_service.extract_text(file_path)
            
            if not ocr_result.full_text:
                result.errors.append("OCR failed - no text extracted")
                result.needs_review = True
                result.review_reasons.append("OCR extraction failed")
                return result
            
            result.raw_ocr_text = ocr_result.full_text
            
            if ocr_result.confidence < 0.5:
                result.warnings.append(f"Low OCR confidence: {ocr_result.confidence:.0%}")
                result.needs_review = True
                result.review_reasons.append("Low OCR quality")
            
            # Step 2: Extract prescription data
            logger.info("Extracting prescription data...")
            extracted = self.extractor.extract(ocr_result.full_text)
            
            # Populate result
            result.patient_name = extracted.patient_name
            result.patient_age = extracted.patient_age
            result.patient_gender = extracted.patient_gender
            result.patient_id = extracted.patient_id
            
            result.prescription_date = extracted.prescription_date
            
            result.doctor_name = extracted.doctor_name
            result.doctor_qualification = extracted.doctor_qualification
            result.clinic_name = extracted.clinic_name
            result.clinic_phone = extracted.clinic_phone
            
            result.diagnosis = extracted.diagnosis
            result.vitals = extracted.vitals
            
            # Convert medications to dicts
            result.medications = [med.to_dict() for med in extracted.medications]
            
            result.advice = extracted.advice
            result.follow_up = extracted.follow_up_date
            
            result.confidence = extracted.extraction_confidence
            result.warnings.extend(extracted.warnings)
            
            # Step 3: Drug safety check
            if result.medications:
                med_names = [m['name'] for m in result.medications if m.get('name')]
                
                try:
                    safety = self.drug_service.analyze_safety(
                        medications=med_names,
                        patient_allergies=patient_allergies or []
                    )
                    
                    # Extract interactions
                    for interaction in safety.interactions:
                        result.drug_interactions.append({
                            'drug1': interaction.drug1,
                            'drug2': interaction.drug2,
                            'severity': interaction.severity.value,
                            'description': interaction.description,
                            'management': interaction.management
                        })
                        
                        if interaction.severity.value in ['major', 'contraindicated']:
                            result.safety_alerts.append(
                                f"⚠️ {interaction.severity.value.upper()}: {interaction.drug1} + {interaction.drug2}"
                            )
                            result.needs_review = True
                            result.review_reasons.append(f"Drug interaction: {interaction.drug1} + {interaction.drug2}")
                    
                    # Check for allergy alerts
                    for alert in safety.allergy_alerts:
                        result.safety_alerts.append(
                            f"🚨 ALLERGY: {alert.drug} may cause reaction (allergen: {alert.allergen})"
                        )
                        result.needs_review = True
                        result.review_reasons.append(f"Allergy risk: {alert.drug}")
                        
                except Exception as e:
                    logger.warning(f"Drug safety check failed: {e}")
                    result.warnings.append("Drug safety check incomplete")
            
            # Check if key fields are missing
            if not result.patient_name:
                result.warnings.append("Patient name not found")
            if not result.medications:
                result.warnings.append("No medications extracted")
                result.needs_review = True
                result.review_reasons.append("No medications found")
            if not result.prescription_date:
                result.warnings.append("Prescription date not found")
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            result.errors.append(str(e))
            result.needs_review = True
            result.review_reasons.append(f"Processing error: {str(e)}")
        
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Processing complete: {len(result.medications)} medications, confidence: {result.confidence:.0%}")
        
        return result


# Create singleton instance
simple_processor = SimpleDocumentProcessor()
