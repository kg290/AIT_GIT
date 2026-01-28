"""
Complete Document Processor - Full pipeline for prescription processing
Integrates OCR, AI extraction, drug safety, and database persistence
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

from backend.config import settings
from backend.services.ocr_service import OCRService
from backend.services.ai_extractor import AIExtractor, PrescriptionData, MedicationData
from backend.services.drug_database import (
    find_all_interactions, find_allergy_alerts, 
    DrugInteraction, Severity
)
from backend.services.database_service import db_service

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Complete processing result"""
    # Status
    success: bool
    document_id: str
    
    # Patient Info
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    patient_id: Optional[str] = None
    patient_address: Optional[str] = None
    patient_phone: Optional[str] = None
    
    # Doctor Info
    doctor_name: Optional[str] = None
    doctor_qualification: Optional[str] = None
    clinic_name: Optional[str] = None
    doctor_reg_no: Optional[str] = None
    
    # Prescription Info
    prescription_date: Optional[str] = None
    
    # Clinical Data
    diagnosis: List[str] = field(default_factory=list)
    chief_complaints: List[str] = field(default_factory=list)
    vitals: Dict[str, str] = field(default_factory=dict)
    
    # Medications
    medications: List[Dict] = field(default_factory=list)
    
    # Additional
    advice: List[str] = field(default_factory=list)
    follow_up: Optional[str] = None
    investigations: List[str] = field(default_factory=list)
    
    # Safety
    drug_interactions: List[Dict] = field(default_factory=list)
    allergy_alerts: List[Dict] = field(default_factory=list)
    safety_alerts: List[str] = field(default_factory=list)
    
    # Raw Data
    raw_ocr_text: str = ""
    
    # Metadata
    confidence: float = 0.0
    ocr_confidence: float = 0.0
    extraction_method: str = "unknown"
    processing_time_ms: int = 0
    
    # Review
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    
    # Errors/Warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CompleteDocumentProcessor:
    """
    Full prescription processing pipeline:
    1. OCR text extraction (Google Vision)
    2. AI-powered data extraction (Gemini or regex fallback)
    3. Drug interaction checking
    4. Allergy alerts
    5. Database persistence
    6. Audit logging
    """
    
    def __init__(self, gemini_api_key: str = None):
        self.ocr_service = OCRService()
        self.ai_extractor = AIExtractor(api_key=gemini_api_key)
        
    def process(
        self,
        file_path: str,
        patient_allergies: List[str] = None,
        patient_id: int = None,
        save_to_db: bool = True
    ) -> ProcessingResult:
        """
        Process a prescription document completely
        
        Args:
            file_path: Path to the prescription image/PDF
            patient_allergies: Known patient allergies
            patient_id: Database patient ID for linking
            save_to_db: Whether to save to database
            
        Returns:
            ProcessingResult with all extracted data and safety checks
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        result = ProcessingResult(
            success=False,
            document_id=document_id
        )
        
        try:
            # ============ STEP 1: OCR ============
            logger.info(f"[{document_id}] Starting OCR extraction...")
            
            ocr_result = self.ocr_service.extract_text(file_path)
            
            if not ocr_result.full_text:
                result.errors.append("OCR failed - no text extracted from document")
                result.needs_review = True
                result.review_reasons.append("OCR extraction failed")
                return result
            
            result.raw_ocr_text = ocr_result.full_text
            result.ocr_confidence = ocr_result.confidence
            
            logger.info(f"[{document_id}] OCR complete: {len(ocr_result.full_text)} chars, {ocr_result.confidence:.0%} confidence")
            
            if ocr_result.confidence < 0.5:
                result.warnings.append(f"Low OCR quality: {ocr_result.confidence:.0%} confidence")
                result.needs_review = True
                result.review_reasons.append("Low OCR quality")
            
            # ============ STEP 2: AI EXTRACTION ============
            logger.info(f"[{document_id}] Starting AI extraction...")
            
            extracted = self.ai_extractor.extract(ocr_result.full_text)
            
            # Populate result from extraction - Patient Info
            result.patient_name = extracted.patient_name
            result.patient_age = extracted.patient_age
            result.patient_gender = extracted.patient_gender
            result.patient_id = extracted.patient_id
            result.patient_address = getattr(extracted, 'patient_address', None)
            result.patient_phone = getattr(extracted, 'patient_phone', None)
            
            # Doctor Info
            result.doctor_name = extracted.doctor_name
            result.doctor_qualification = extracted.doctor_qualification
            result.clinic_name = extracted.clinic_name
            result.doctor_reg_no = getattr(extracted, 'doctor_reg_no', None)
            
            result.prescription_date = extracted.prescription_date
            
            result.diagnosis = extracted.diagnosis
            result.chief_complaints = extracted.chief_complaints
            result.vitals = extracted.vitals
            
            # Convert medications to dicts
            result.medications = [
                med.to_dict() if hasattr(med, 'to_dict') else med 
                for med in extracted.medications
            ]
            
            result.advice = extracted.advice
            result.follow_up = extracted.follow_up
            result.investigations = extracted.investigations
            
            result.confidence = extracted.confidence
            result.extraction_method = extracted.extraction_method
            
            logger.info(f"[{document_id}] Extraction complete: {len(result.medications)} medications, {result.confidence:.0%} confidence")
            
            # Check for missing critical data
            if not result.patient_name:
                result.warnings.append("Patient name not found")
            if not result.medications:
                result.warnings.append("No medications extracted")
                result.needs_review = True
                result.review_reasons.append("No medications found")
            if not result.prescription_date:
                result.warnings.append("Prescription date not found")
            
            # ============ STEP 3: DRUG SAFETY ============
            if result.medications:
                logger.info(f"[{document_id}] Checking drug safety...")
                
                med_names = [m.get('name', '') for m in result.medications if m.get('name')]
                
                # Check interactions
                interactions = find_all_interactions(med_names)
                for interaction in interactions:
                    result.drug_interactions.append({
                        'drug1': interaction.drug1,
                        'drug2': interaction.drug2,
                        'severity': interaction.severity.value,
                        'description': interaction.description,
                        'mechanism': interaction.mechanism,
                        'management': interaction.management
                    })
                    
                    if interaction.severity in [Severity.MAJOR, Severity.CONTRAINDICATED]:
                        result.safety_alerts.append(
                            f"⚠️ {interaction.severity.value.upper()}: {interaction.drug1} + {interaction.drug2} - {interaction.description}"
                        )
                        result.needs_review = True
                        result.review_reasons.append(f"Drug interaction: {interaction.drug1} + {interaction.drug2}")
                
                # Check allergies
                if patient_allergies:
                    allergy_alerts = find_allergy_alerts(med_names, patient_allergies)
                    for alert in allergy_alerts:
                        result.allergy_alerts.append({
                            'drug': alert.drug,
                            'allergen': alert.allergen,
                            'cross_reactivity': alert.cross_reactivity,
                            'alternatives': alert.alternatives
                        })
                        result.safety_alerts.append(
                            f"🚨 ALLERGY ALERT: {alert.drug} - patient allergic to {alert.allergen}"
                        )
                        result.needs_review = True
                        result.review_reasons.append(f"Allergy risk: {alert.drug}")
                
                logger.info(f"[{document_id}] Safety check: {len(interactions)} interactions, {len(result.allergy_alerts)} allergy alerts")
            
            # ============ STEP 4: DATABASE PERSISTENCE ============
            if save_to_db:
                try:
                    self._save_to_database(document_id, file_path, result, patient_id)
                    logger.info(f"[{document_id}] Saved to database")
                except Exception as e:
                    logger.error(f"[{document_id}] Database save failed: {e}")
                    result.warnings.append("Failed to save to database")
            
            result.success = True
            
        except Exception as e:
            logger.error(f"[{document_id}] Processing failed: {e}", exc_info=True)
            result.errors.append(f"Processing failed: {str(e)}")
            result.needs_review = True
            result.review_reasons.append(f"Processing error")
        
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"[{document_id}] Processing complete in {result.processing_time_ms}ms")
        
        return result
    
    def _save_to_database(self, document_id: str, file_path: str, 
                         result: ProcessingResult, patient_id: int = None):
        """Save processing result to database"""
        
        # Parse prescription date
        prescription_date = None
        if result.prescription_date:
            try:
                from dateutil import parser
                prescription_date = parser.parse(result.prescription_date, dayfirst=True)
            except:
                pass
        
        # Create prescription record
        prescription = db_service.create_prescription(
            document_id=document_id,
            patient_id=patient_id,
            file_name=Path(file_path).name,
            file_path=file_path,
            prescription_date=prescription_date,
            patient_info_extracted={
                'name': result.patient_name,
                'age': result.patient_age,
                'gender': result.patient_gender,
                'id': result.patient_id
            },
            diagnosis=result.diagnosis,
            chief_complaints=result.chief_complaints,
            vitals=result.vitals,
            advice=result.advice,
            raw_ocr_text=result.raw_ocr_text,
            ocr_confidence=result.ocr_confidence,
            extraction_confidence=result.confidence,
            extraction_method=result.extraction_method,
            needs_review=result.needs_review,
            review_reasons=result.review_reasons,
            has_interactions=len(result.drug_interactions) > 0,
            has_allergy_alerts=len(result.allergy_alerts) > 0,
            safety_checked=True,
            processing_status='completed' if result.success else 'failed'
        )
        
        prescription_id = prescription.get('id')
        
        # Add medications
        for med in result.medications:
            db_service.add_prescription_medication(
                prescription_id=prescription_id,
                name=med.get('name'),
                dosage=med.get('dosage'),
                form=med.get('form'),
                frequency=med.get('frequency'),
                timing=med.get('timing'),
                duration=med.get('duration'),
                quantity=med.get('quantity'),
                instructions=med.get('instructions'),
                route=med.get('route', 'oral')
            )
        
        # Add drug interactions
        for interaction in result.drug_interactions:
            db_service.add_drug_interaction(
                prescription_id=prescription_id,
                drug1=interaction['drug1'],
                drug2=interaction['drug2'],
                severity=interaction['severity'],
                description=interaction['description'],
                mechanism=interaction.get('mechanism'),
                management=interaction.get('management')
            )


# Create singleton
complete_processor = CompleteDocumentProcessor()
