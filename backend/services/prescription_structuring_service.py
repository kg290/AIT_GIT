"""
Prescription Structuring Service - Convert free-text to structured format
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date

from .entity_extraction_service import EntityExtractionService, ExtractionResult
from .drug_normalization_service import DrugNormalizationService
from .text_cleaning_service import TextCleaningService

logger = logging.getLogger(__name__)


@dataclass
class StructuredMedication:
    """Structured medication item"""
    medication_name: str
    generic_name: Optional[str]
    brand_name: Optional[str]
    dosage: Optional[str]
    dosage_value: Optional[float]
    dosage_unit: Optional[str]
    frequency: Optional[str]
    frequency_parsed: Optional[Dict]
    times_per_day: Optional[int]
    route: Optional[str]
    duration: Optional[str]
    duration_days: Optional[int]
    instructions: Optional[str]
    quantity: Optional[int]
    confidence: float
    is_uncertain: bool
    uncertainty_reasons: List[str]
    source_text: str


@dataclass
class StructuredPrescription:
    """Complete structured prescription"""
    medications: List[StructuredMedication]
    prescriber: Optional[str]
    prescription_date: Optional[date]
    diagnosis: Optional[str]
    overall_confidence: float
    has_ambiguity: bool
    ambiguity_notes: List[str]
    raw_text: str
    warnings: List[str]


class PrescriptionStructuringService:
    """
    Service to convert free-text prescriptions into structured format
    
    Features:
    - Separates multiple medications
    - Attaches dosage, frequency, duration to correct drug
    - Handles missing/ambiguous fields
    - Preserves uncertainty
    """
    
    def __init__(self):
        self.entity_extractor = EntityExtractionService()
        self.drug_normalizer = DrugNormalizationService()
        self.text_cleaner = TextCleaningService()
        
        # Medication separator patterns
        self.separator_patterns = [
            r'\n\s*\d+[\.\)]\s*',  # 1. or 1)
            r'\n\s*[-•]\s*',       # Bullet points
            r'\n\s*Tab\.?\s+',     # Tab Paracetamol
            r'\n\s*Cap\.?\s+',     # Cap Amoxicillin
            r'\n\s*Inj\.?\s+',     # Inj Insulin
            r'\n\s*Syr\.?\s+',     # Syr Benadryl
            r'\n{2,}',              # Double line breaks
        ]
        
        # Instruction patterns
        self.instruction_patterns = [
            (r'take\s+with\s+(food|meals?|water|milk)', 'take_with'),
            (r'(before|after)\s+(breakfast|lunch|dinner|meals?|food)', 'meal_timing'),
            (r'(empty|full)\s+stomach', 'stomach_condition'),
            (r'(avoid|do not take)\s+(alcohol|driving|sunlight)', 'avoidance'),
            (r'(swallow|chew)\s+(whole|thoroughly)', 'administration'),
            (r'apply\s+(to|on)\s+(affected\s+area|skin)', 'topical_instruction'),
        ]
    
    def structure_prescription(self, text: str, prescription_date: Optional[date] = None) -> StructuredPrescription:
        """
        Convert free-text prescription to structured format
        
        Args:
            text: Raw prescription text (usually from OCR)
            prescription_date: Optional prescription date
            
        Returns:
            StructuredPrescription with all extracted data
        """
        # Clean the text first
        cleaning_result = self.text_cleaner.clean_text(text)
        cleaned_text = cleaning_result.cleaned_text
        
        # Extract all entities
        extraction = self.entity_extractor.extract_entities(cleaned_text)
        
        # Split into medication blocks
        med_blocks = self._split_into_medication_blocks(cleaned_text, extraction)
        
        # Structure each medication
        structured_meds = []
        ambiguity_notes = []
        warnings = []
        
        for block in med_blocks:
            structured_med, notes, block_warnings = self._structure_medication_block(block, extraction)
            if structured_med:
                structured_meds.append(structured_med)
                ambiguity_notes.extend(notes)
                warnings.extend(block_warnings)
        
        # Extract prescriber info
        prescriber = self._extract_prescriber(text)
        
        # Extract diagnosis
        diagnosis = self._extract_diagnosis(text, extraction)
        
        # Calculate overall confidence
        if structured_meds:
            overall_confidence = sum(m.confidence for m in structured_meds) / len(structured_meds)
        else:
            overall_confidence = 0.3
        
        has_ambiguity = len(ambiguity_notes) > 0 or any(m.is_uncertain for m in structured_meds)
        
        return StructuredPrescription(
            medications=structured_meds,
            prescriber=prescriber,
            prescription_date=prescription_date,
            diagnosis=diagnosis,
            overall_confidence=overall_confidence,
            has_ambiguity=has_ambiguity,
            ambiguity_notes=ambiguity_notes,
            raw_text=text,
            warnings=warnings
        )
    
    def _split_into_medication_blocks(self, text: str, extraction: ExtractionResult) -> List[Dict]:
        """Split prescription text into individual medication blocks"""
        blocks = []
        
        # Method 1: Split by separator patterns
        lines = text.split('\n')
        current_block = []
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_text:
                    blocks.append({'text': current_text.strip(), 'lines': current_block.copy()})
                    current_block = []
                    current_text = ""
                continue
            
            # Check if this line starts a new medication
            is_new_med = False
            
            # Check for numbered items
            if re.match(r'^\d+[\.\)]\s*', line):
                is_new_med = True
            # Check for bullet points
            elif re.match(r'^[-•]\s*', line):
                is_new_med = True
            # Check for medication form prefixes
            elif re.match(r'^(Tab|Cap|Inj|Syr|Susp|Oint|Drops?)[\.\s]', line, re.IGNORECASE):
                is_new_med = True
            # Check if line starts with a known medication
            else:
                for med in extraction.medications:
                    if line.lower().startswith(med['text'].lower()):
                        is_new_med = True
                        break
            
            if is_new_med and current_text:
                blocks.append({'text': current_text.strip(), 'lines': current_block.copy()})
                current_block = []
                current_text = ""
            
            current_block.append(line)
            current_text += " " + line
        
        # Don't forget the last block
        if current_text:
            blocks.append({'text': current_text.strip(), 'lines': current_block})
        
        # If no blocks found, treat entire text as one block
        if not blocks:
            blocks = [{'text': text, 'lines': [text]}]
        
        return blocks
    
    def _structure_medication_block(self, block: Dict, extraction: ExtractionResult) -> Tuple[Optional[StructuredMedication], List[str], List[str]]:
        """Structure a single medication block"""
        text = block['text']
        notes = []
        warnings = []
        
        # Find medication in this block
        medication = self._find_medication_in_block(text, extraction)
        if not medication:
            return None, [], []
        
        # Normalize medication name
        normalized = self.drug_normalizer.normalize(medication['text'])
        
        # Find dosage
        dosage = self._find_dosage_in_block(text, extraction)
        
        # Find frequency
        frequency = self._find_frequency_in_block(text, extraction)
        
        # Find duration
        duration = self._find_duration_in_block(text, extraction)
        
        # Find route
        route = self._find_route_in_block(text, extraction)
        
        # Find instructions
        instructions = self._extract_instructions(text)
        
        # Calculate confidence and determine uncertainty
        confidence, is_uncertain, uncertainty_reasons = self._calculate_confidence(
            medication, dosage, frequency, duration
        )
        
        if is_uncertain:
            notes.append(f"Uncertainty in '{medication['text']}': {', '.join(uncertainty_reasons)}")
        
        # Check for common issues
        if dosage and normalized.common_dosages:
            if dosage.get('normalized') not in normalized.common_dosages:
                warnings.append(f"Unusual dosage for {normalized.generic_name}: {dosage.get('normalized')}")
        
        return StructuredMedication(
            medication_name=medication['text'],
            generic_name=normalized.generic_name if normalized.confidence > 0.7 else None,
            brand_name=normalized.original_name if normalized.is_brand else None,
            dosage=dosage.get('text') if dosage else None,
            dosage_value=dosage.get('value') if dosage else None,
            dosage_unit=dosage.get('unit') if dosage else None,
            frequency=frequency.get('text') if frequency else None,
            frequency_parsed=frequency.get('parsed') if frequency else None,
            times_per_day=frequency.get('times_per_day') if frequency else None,
            route=route.get('route') if route else None,
            duration=duration.get('text') if duration else None,
            duration_days=duration.get('days') if duration else None,
            instructions=instructions,
            quantity=None,  # Would need specific extraction
            confidence=confidence,
            is_uncertain=is_uncertain,
            uncertainty_reasons=uncertainty_reasons,
            source_text=text
        ), notes, warnings
    
    def _find_medication_in_block(self, text: str, extraction: ExtractionResult) -> Optional[Dict]:
        """Find medication name in a text block"""
        text_lower = text.lower()
        
        for med in extraction.medications:
            if med['text'].lower() in text_lower:
                return med
        
        # Try to find medication forms (Tab, Cap, etc.) followed by a word
        form_match = re.search(r'(Tab|Cap|Inj|Syr|Susp|Oint|Drops?)[\.\s]+([A-Za-z]+(?:\s+[A-Za-z]+)?)', text, re.IGNORECASE)
        if form_match:
            med_name = form_match.group(2)
            return {
                'text': med_name,
                'start': form_match.start(2),
                'end': form_match.end(2),
                'confidence': 0.7
            }
        
        return None
    
    def _find_dosage_in_block(self, text: str, extraction: ExtractionResult) -> Optional[Dict]:
        """Find dosage in text block"""
        text_lower = text.lower()
        
        for dosage in extraction.dosages:
            if dosage['text'].lower() in text_lower:
                return dosage
        
        # Try direct pattern matching
        match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)', text, re.IGNORECASE)
        if match:
            return {
                'text': match.group(0),
                'value': float(match.group(1)),
                'unit': match.group(2).lower(),
                'normalized': f"{match.group(1)}{match.group(2).lower()}"
            }
        
        return None
    
    def _find_frequency_in_block(self, text: str, extraction: ExtractionResult) -> Optional[Dict]:
        """Find frequency in text block"""
        text_lower = text.lower()
        
        for freq in extraction.frequencies:
            if freq['text'].lower() in text_lower:
                return freq
        
        # Common frequency patterns not in extraction
        freq_patterns = [
            (r'(\d+)\s*x\s*(\d+)', lambda m: {'text': m.group(0), 'times_per_day': int(m.group(1)) * int(m.group(2))}),
            (r'daily', lambda m: {'text': m.group(0), 'times_per_day': 1, 'normalized': 'once daily'}),
            (r'morning', lambda m: {'text': m.group(0), 'times_per_day': 1, 'normalized': 'in the morning'}),
        ]
        
        for pattern, parser in freq_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return parser(match)
        
        return None
    
    def _find_duration_in_block(self, text: str, extraction: ExtractionResult) -> Optional[Dict]:
        """Find duration in text block"""
        text_lower = text.lower()
        
        for dur in extraction.durations:
            if dur['text'].lower() in text_lower:
                return dur
        
        return None
    
    def _find_route_in_block(self, text: str, extraction: ExtractionResult) -> Optional[Dict]:
        """Find route in text block"""
        # Check extraction
        for route in extraction.routes:
            if route['text'].lower() in text.lower():
                return route
        
        # Infer from context
        text_lower = text.lower()
        if any(word in text_lower for word in ['tab', 'tablet', 'cap', 'capsule', 'syr', 'syrup']):
            return {'route': 'oral', 'confidence': 0.9}
        elif any(word in text_lower for word in ['inj', 'injection']):
            return {'route': 'injection', 'confidence': 0.9}
        elif any(word in text_lower for word in ['oint', 'ointment', 'cream', 'gel', 'apply']):
            return {'route': 'topical', 'confidence': 0.9}
        elif any(word in text_lower for word in ['drop', 'eye']):
            return {'route': 'ophthalmic', 'confidence': 0.8}
        elif any(word in text_lower for word in ['inhaler', 'puff', 'nebulizer']):
            return {'route': 'inhaled', 'confidence': 0.9}
        
        return None
    
    def _extract_instructions(self, text: str) -> Optional[str]:
        """Extract special instructions from text"""
        instructions = []
        
        for pattern, inst_type in self.instruction_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                instructions.append(match.group(0))
        
        return '; '.join(instructions) if instructions else None
    
    def _extract_prescriber(self, text: str) -> Optional[str]:
        """Extract prescriber name from text"""
        patterns = [
            r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Prescriber:\s*([A-Za-z\s]+)',
            r'Physician:\s*([A-Za-z\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_diagnosis(self, text: str, extraction: ExtractionResult) -> Optional[str]:
        """Extract diagnosis from text"""
        # Check extraction results
        if extraction.diagnoses:
            return extraction.diagnoses[0]['text']
        
        # Pattern matching
        patterns = [
            r'(?:Diagnosis|Dx|D/x):\s*([^\n]+)',
            r'(?:For|Indication):\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _calculate_confidence(self, medication: Dict, dosage: Optional[Dict], 
                             frequency: Optional[Dict], duration: Optional[Dict]) -> Tuple[float, bool, List[str]]:
        """Calculate confidence and determine if uncertain"""
        confidence = 1.0
        reasons = []
        
        # Medication confidence
        med_conf = medication.get('confidence', 0.8)
        confidence *= med_conf
        if med_conf < 0.8:
            reasons.append('Low medication name confidence')
        
        # Penalize missing fields
        if not dosage:
            confidence *= 0.8
            reasons.append('Missing dosage')
        elif dosage.get('confidence', 0.9) < 0.8:
            confidence *= 0.9
            reasons.append('Low dosage confidence')
        
        if not frequency:
            confidence *= 0.8
            reasons.append('Missing frequency')
        
        if not duration:
            confidence *= 0.9  # Duration often omitted
        
        is_uncertain = confidence < 0.7 or len(reasons) > 1
        
        return confidence, is_uncertain, reasons
    
    def to_dict(self, prescription: StructuredPrescription) -> Dict:
        """Convert StructuredPrescription to dictionary"""
        return {
            'medications': [
                {
                    'medication_name': m.medication_name,
                    'generic_name': m.generic_name,
                    'brand_name': m.brand_name,
                    'dosage': m.dosage,
                    'dosage_value': m.dosage_value,
                    'dosage_unit': m.dosage_unit,
                    'frequency': m.frequency,
                    'frequency_parsed': m.frequency_parsed,
                    'times_per_day': m.times_per_day,
                    'route': m.route,
                    'duration': m.duration,
                    'duration_days': m.duration_days,
                    'instructions': m.instructions,
                    'confidence': m.confidence,
                    'is_uncertain': m.is_uncertain,
                    'uncertainty_reasons': m.uncertainty_reasons
                }
                for m in prescription.medications
            ],
            'prescriber': prescription.prescriber,
            'prescription_date': str(prescription.prescription_date) if prescription.prescription_date else None,
            'diagnosis': prescription.diagnosis,
            'overall_confidence': prescription.overall_confidence,
            'has_ambiguity': prescription.has_ambiguity,
            'ambiguity_notes': prescription.ambiguity_notes,
            'warnings': prescription.warnings
        }
