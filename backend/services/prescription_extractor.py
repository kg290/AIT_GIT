"""
Prescription Extractor - Extracts key medical information from OCR text
Designed for real-world hospital use with Indian and international prescription formats
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMedication:
    """Extracted medication with all details"""
    name: str
    dosage: Optional[str] = None
    form: Optional[str] = None  # tablet, capsule, syrup, etc.
    frequency: Optional[str] = None  # once daily, twice daily, etc.
    timing: Optional[str] = None  # before food, after food, etc.
    duration: Optional[str] = None  # 5 days, 1 week, etc.
    quantity: Optional[int] = None
    route: str = "oral"
    instructions: Optional[str] = None
    confidence: float = 0.8
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExtractedPrescription:
    """Complete extracted prescription data"""
    # Patient Info
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    patient_id: Optional[str] = None
    patient_address: Optional[str] = None
    patient_phone: Optional[str] = None
    
    # Prescription Info
    prescription_date: Optional[str] = None
    prescription_id: Optional[str] = None
    
    # Doctor Info
    doctor_name: Optional[str] = None
    doctor_qualification: Optional[str] = None
    doctor_registration: Optional[str] = None
    clinic_name: Optional[str] = None
    clinic_address: Optional[str] = None
    clinic_phone: Optional[str] = None
    
    # Clinical Info
    diagnosis: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    vitals: Dict[str, str] = field(default_factory=dict)
    
    # Medications
    medications: List[ExtractedMedication] = field(default_factory=list)
    
    # Additional
    advice: List[str] = field(default_factory=list)
    follow_up_date: Optional[str] = None
    investigations: List[str] = field(default_factory=list)
    
    # Metadata
    extraction_confidence: float = 0.0
    raw_text: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['medications'] = [m.to_dict() if hasattr(m, 'to_dict') else m for m in self.medications]
        return data


class PrescriptionExtractor:
    """
    Extracts structured prescription data from OCR text
    
    Supports:
    - Indian prescription formats
    - International formats
    - Handwritten prescriptions (with OCR)
    - Multiple medication entries
    - Various date formats
    """
    
    def __init__(self):
        # Common medication forms
        self.medication_forms = {
            'tab': 'Tablet', 'tab.': 'Tablet', 'tablet': 'Tablet', 'tablets': 'Tablet',
            'cap': 'Capsule', 'cap.': 'Capsule', 'capsule': 'Capsule', 'capsules': 'Capsule',
            'syr': 'Syrup', 'syr.': 'Syrup', 'syrup': 'Syrup', 'syp': 'Syrup',
            'inj': 'Injection', 'inj.': 'Injection', 'injection': 'Injection',
            'cream': 'Cream', 'oint': 'Ointment', 'ointment': 'Ointment',
            'drops': 'Drops', 'drop': 'Drops', 'gel': 'Gel',
            'susp': 'Suspension', 'suspension': 'Suspension',
            'powder': 'Powder', 'sachet': 'Sachet',
            'inhaler': 'Inhaler', 'spray': 'Spray', 'nasal': 'Nasal Spray',
            'lotion': 'Lotion', 'solution': 'Solution', 'sol': 'Solution'
        }
        
        # Frequency patterns (Indian and international)
        self.frequency_patterns = {
            r'\b(od|o\.d\.|once\s*daily|once\s*a\s*day|1\s*x\s*1)\b': 'Once daily',
            r'\b(bd|b\.d\.|bid|b\.i\.d\.|twice\s*daily|twice\s*a\s*day|2\s*times|1-0-1)\b': 'Twice daily',
            r'\b(tds|t\.d\.s\.|tid|t\.i\.d\.|thrice\s*daily|three\s*times|3\s*times|1-1-1)\b': 'Three times daily',
            r'\b(qid|q\.i\.d\.|four\s*times|4\s*times|1-1-1-1)\b': 'Four times daily',
            r'\b(hs|h\.s\.|at\s*bedtime|at\s*night|0-0-1)\b': 'At bedtime',
            r'\b(sos|s\.o\.s\.|as\s*needed|when\s*required|prn|p\.r\.n\.)\b': 'As needed',
            r'\b(stat|immediately)\b': 'Immediately',
            r'\b(weekly|once\s*a\s*week)\b': 'Once weekly',
            r'\b(alternate\s*days?|every\s*other\s*day)\b': 'Alternate days',
            r'\b(1\s*morning|morning\s*only|0-0-0-1)\b': 'Morning only',
            r'\b(1\s*night|night\s*only|0-0-1)\b': 'Night only',
        }
        
        # Timing patterns
        self.timing_patterns = {
            r'\b(ac|a\.c\.|before\s*(food|meal|eating)|empty\s*stomach)\b': 'Before food',
            r'\b(pc|p\.c\.|after\s*(food|meal|eating)|with\s*food)\b': 'After food',
            r'\b(with\s*milk)\b': 'With milk',
            r'\b(with\s*water|plenty\s*of\s*water)\b': 'With water',
        }
        
        # Duration patterns
        self.duration_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?',
            r'x\s*(\d+)\s*days?',
            r'for\s*(\d+)\s*days?',
        ]
        
        # Dosage patterns
        self.dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)',
            r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)/(\d+(?:\.\d+)?)\s*(ml)',
            r'(\d+/\d+)',  # Fractions like 1/2
        ]
        
        # Common Indian qualifications
        self.qualifications = [
            'MBBS', 'MD', 'MS', 'M.D.', 'M.S.', 'FRCS', 'MRCP', 'DNB',
            'DM', 'MCh', 'DCH', 'DTCD', 'DA', 'FCPS', 'DOMS', 'DGO',
            'BDS', 'MDS', 'BAMS', 'BHMS', 'BUMS', 'BYNS'
        ]
    
    def extract(self, ocr_text: str) -> ExtractedPrescription:
        """
        Extract all prescription information from OCR text
        
        Args:
            ocr_text: Raw text from OCR
            
        Returns:
            ExtractedPrescription with all extracted data
        """
        result = ExtractedPrescription(raw_text=ocr_text)
        
        if not ocr_text or not ocr_text.strip():
            result.warnings.append("Empty OCR text")
            return result
        
        # Clean text
        text = self._clean_text(ocr_text)
        lines = text.split('\n')
        
        # Extract each component
        result.patient_name = self._extract_patient_name(text, lines)
        result.patient_age, result.patient_gender = self._extract_age_gender(text)
        result.patient_id = self._extract_patient_id(text)
        
        result.prescription_date = self._extract_date(text)
        
        result.doctor_name = self._extract_doctor_name(text, lines)
        result.doctor_qualification = self._extract_qualifications(text)
        result.doctor_registration = self._extract_registration(text)
        result.clinic_name = self._extract_clinic_name(text, lines)
        result.clinic_phone = self._extract_phone(text)
        
        result.diagnosis = self._extract_diagnosis(text)
        result.vitals = self._extract_vitals(text)
        
        result.medications = self._extract_medications(text, lines)
        
        result.advice = self._extract_advice(text)
        result.follow_up_date = self._extract_follow_up(text)
        
        # Calculate confidence
        result.extraction_confidence = self._calculate_confidence(result)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize OCR text"""
        # Fix common OCR errors
        replacements = {
            '|': 'I',
            '0': 'O',  # Will be context-dependent
            '1': 'l',  # Will be context-dependent
            '\t': ' ',
            '  +': ' ',
        }
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def _extract_patient_name(self, text: str, lines: List[str]) -> Optional[str]:
        """Extract patient name"""
        patterns = [
            r'(?:patient|pt|name|patient\s*name)\s*[:\-]?\s*([A-Za-z\s\.]+?)(?:\s*(?:\d|age|male|female|m/|f/|address|\n))',
            r'(?:mr\.|mrs\.|ms\.|master|baby)\s+([A-Za-z\s\.]+?)(?:\s*(?:\d|age|,|\n))',
            r'ID:\s*\d+\s*[-–]\s*([A-Za-z\s]+?)(?:\s*\()',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up name
                name = re.sub(r'\s+', ' ', name)
                if len(name) > 2 and len(name) < 100:
                    return name.title()
        
        return None
    
    def _extract_age_gender(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract patient age and gender"""
        age = None
        gender = None
        
        # Age patterns
        age_patterns = [
            r'(\d{1,3})\s*(?:yrs?|years?|y/o|yo)',
            r'age\s*[:\-]?\s*(\d{1,3})',
            r'\((\d{1,3})\s*(?:yrs?|y)\)',
            r'(\d{1,3})\s*(?:months?|m)\s*old',
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age_val = int(match.group(1))
                if 0 < age_val < 150:
                    age = f"{age_val} years"
                    break
        
        # Gender patterns
        gender_patterns = [
            (r'\b(male|m)\s*/?\s*(\d)', 'Male'),
            (r'\b(female|f)\s*/?\s*(\d)', 'Female'),
            (r'\((m|male)\)', 'Male'),
            (r'\((f|female)\)', 'Female'),
            (r'gender\s*[:\-]?\s*(male|female|m|f)', None),
            (r'\b(mr\.|master)\s+', 'Male'),
            (r'\b(mrs\.|ms\.|miss)\s+', 'Female'),
        ]
        
        for pattern, default_gender in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if default_gender:
                    gender = default_gender
                else:
                    g = match.group(1).upper()
                    gender = 'Male' if g in ['M', 'MALE'] else 'Female'
                break
        
        return age, gender
    
    def _extract_patient_id(self, text: str) -> Optional[str]:
        """Extract patient ID"""
        patterns = [
            r'(?:id|patient\s*id|reg\.?\s*no|registration)\s*[:\-]?\s*(\w+[-/]?\d+)',
            r'ID:\s*(\d+)',
            r'MRN\s*[:\-]?\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract prescription date"""
        patterns = [
            r'date\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'date\s*[:\-]?\s*(\d{1,2}[-/\s]+[A-Za-z]+[-/\s]+\d{2,4})',
            r'(\d{1,2}[-/]\w{3}[-/]\d{4})',
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                try:
                    parsed = date_parser.parse(date_str, dayfirst=True)
                    return parsed.strftime('%Y-%m-%d')
                except:
                    return date_str
        
        return None
    
    def _extract_doctor_name(self, text: str, lines: List[str]) -> Optional[str]:
        """Extract doctor name"""
        patterns = [
            r'(?:dr\.?|doctor)\s+([A-Za-z\s\.]+?)(?:\s*(?:' + '|'.join(self.qualifications) + r'))',
            r'^(?:dr\.?|doctor)\s+([A-Za-z\s\.]+)',
            r'(?:consulting|treated\s*by)\s*[:\-]?\s*(?:dr\.?)?\s*([A-Za-z\s\.]+)',
        ]
        
        # Check first few and last few lines for doctor name
        check_lines = lines[:5] + lines[-5:]
        
        for line in check_lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    name = re.sub(r'\s+', ' ', name)
                    if 2 < len(name) < 50:
                        return f"Dr. {name.title()}"
        
        return None
    
    def _extract_qualifications(self, text: str) -> Optional[str]:
        """Extract doctor qualifications"""
        found = []
        for qual in self.qualifications:
            if re.search(rf'\b{qual}\b', text, re.IGNORECASE):
                found.append(qual)
        
        return ', '.join(found) if found else None
    
    def _extract_registration(self, text: str) -> Optional[str]:
        """Extract medical registration number"""
        patterns = [
            r'(?:reg\.?|registration)\s*(?:no\.?|number)?\s*[:\-]?\s*(\d+)',
            r'(?:mc|mci|state\s*medical\s*council)\s*(?:no\.?|reg\.?)?\s*[:\-]?\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_clinic_name(self, text: str, lines: List[str]) -> Optional[str]:
        """Extract clinic/hospital name"""
        patterns = [
            r'([A-Za-z\s]+(?:clinic|hospital|medical\s*centre?|health\s*centre?))',
            r'([A-Za-z\s]+(?:nursing\s*home|polyclinic|dispensary))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number"""
        patterns = [
            r'(?:ph|phone|tel|mob|mobile|contact)\s*[:\-]?\s*([\d\s\-]+)',
            r'\b(\d{10})\b',
            r'\b(\d{3}[-\s]\d{3}[-\s]\d{4})\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phone = re.sub(r'[^\d]', '', match.group(1))
                if 10 <= len(phone) <= 12:
                    return phone
        
        return None
    
    def _extract_diagnosis(self, text: str) -> List[str]:
        """Extract diagnosis/chief complaints"""
        diagnoses = []
        
        patterns = [
            r'(?:diagnosis|dx|c/c|chief\s*complaint|complaint)\s*[:\-]?\s*([^\n]+)',
            r'(?:provisional\s*diagnosis|final\s*diagnosis)\s*[:\-]?\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split by comma or semicolon
                parts = re.split(r'[,;]', match)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 2:
                        diagnoses.append(part)
        
        return diagnoses[:5]  # Limit to 5
    
    def _extract_vitals(self, text: str) -> Dict[str, str]:
        """Extract vital signs"""
        vitals = {}
        
        # Blood Pressure
        bp_match = re.search(r'(?:bp|blood\s*pressure)\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})', text, re.IGNORECASE)
        if bp_match:
            vitals['blood_pressure'] = f"{bp_match.group(1)}/{bp_match.group(2)} mmHg"
        
        # Temperature
        temp_match = re.search(r'(?:temp|temperature)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?)\s*(?:°?[fc]?)', text, re.IGNORECASE)
        if temp_match:
            vitals['temperature'] = f"{temp_match.group(1)}°F"
        
        # Pulse
        pulse_match = re.search(r'(?:pulse|pr|heart\s*rate)\s*[:\-]?\s*(\d{2,3})', text, re.IGNORECASE)
        if pulse_match:
            vitals['pulse'] = f"{pulse_match.group(1)} bpm"
        
        # SpO2
        spo2_match = re.search(r'(?:spo2|oxygen|o2\s*sat)\s*[:\-]?\s*(\d{2,3})\s*%?', text, re.IGNORECASE)
        if spo2_match:
            vitals['spo2'] = f"{spo2_match.group(1)}%"
        
        # Weight
        weight_match = re.search(r'(?:weight|wt)\s*[:\-]?\s*(\d{1,3}(?:\.\d)?)\s*(?:kg)?', text, re.IGNORECASE)
        if weight_match:
            vitals['weight'] = f"{weight_match.group(1)} kg"
        
        return vitals
    
    def _extract_medications(self, text: str, lines: List[str]) -> List[ExtractedMedication]:
        """Extract medications from prescription"""
        medications = []
        
        # Find Rx section
        rx_start = -1
        for i, line in enumerate(lines):
            if re.search(r'^R[xX]|\bR\s*/\s*[xX]|℞', line):
                rx_start = i
                break
        
        # Patterns for medication lines
        med_patterns = [
            # Tab. MEDICINE NAME 500mg 1-0-1 x 5 days
            r'(?:\d+\)?\s*)?(?:tab\.?|cap\.?|syr\.?|inj\.?|cream|oint\.?)\s+([A-Za-z0-9\s\-]+?)(?:\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)))?(?:\s+([\d\-\/]+|od|bd|tds|qid|hs|sos))?(?:\s+(?:x\s*)?(\d+\s*(?:days?|weeks?)))?',
            # 1) MEDICINE NAME
            r'^\s*\d+\)\s*([A-Za-z][A-Za-z0-9\s\-]+)',
            # MEDICINE 500mg
            r'^([A-Z][A-Za-z0-9\s\-]+?)\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml))',
        ]
        
        # Process lines
        process_lines = lines[rx_start+1:] if rx_start >= 0 else lines
        
        for line in process_lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Skip non-medication lines
            if re.search(r'advice|follow|next|review|investigation|diagnosis', line, re.IGNORECASE):
                continue
            
            # Try to extract medication
            med = self._parse_medication_line(line)
            if med and med.name:
                medications.append(med)
        
        return medications
    
    def _parse_medication_line(self, line: str) -> Optional[ExtractedMedication]:
        """Parse a single medication line"""
        med = ExtractedMedication(name="")
        
        # Remove numbering
        line = re.sub(r'^\s*\d+\)\s*', '', line)
        line = re.sub(r'^\s*[-•]\s*', '', line)
        
        # Extract form
        form = None
        for abbr, full_form in self.medication_forms.items():
            if re.search(rf'\b{abbr}\b', line, re.IGNORECASE):
                form = full_form
                line = re.sub(rf'\b{abbr}\.?\s*', '', line, flags=re.IGNORECASE)
                break
        med.form = form
        
        # Extract dosage
        dosage_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)', line, re.IGNORECASE)
        if dosage_match:
            med.dosage = f"{dosage_match.group(1)} {dosage_match.group(2).lower()}"
            line = line[:dosage_match.start()] + line[dosage_match.end():]
        
        # Extract frequency
        for pattern, freq_text in self.frequency_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                med.frequency = freq_text
                line = re.sub(pattern, '', line, flags=re.IGNORECASE)
                break
        
        # Extract timing
        for pattern, timing_text in self.timing_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                med.timing = timing_text
                line = re.sub(pattern, '', line, flags=re.IGNORECASE)
                break
        
        # Extract duration
        for pattern in self.duration_patterns:
            dur_match = re.search(pattern, line, re.IGNORECASE)
            if dur_match:
                med.duration = dur_match.group(0)
                line = line[:dur_match.start()] + line[dur_match.end():]
                break
        
        # Extract quantity (Tot: X)
        qty_match = re.search(r'(?:tot|total|qty)[:\s]*(\d+)', line, re.IGNORECASE)
        if qty_match:
            med.quantity = int(qty_match.group(1))
            line = line[:qty_match.start()] + line[qty_match.end():]
        
        # Clean up remaining text as medication name
        name = re.sub(r'[^\w\s\-]', '', line).strip()
        name = re.sub(r'\s+', ' ', name)
        
        # Remove common non-drug words
        noise_words = ['morning', 'night', 'evening', 'afternoon', 'days', 'weeks', 'months', 'before', 'after', 'food', 'meal']
        for word in noise_words:
            name = re.sub(rf'\b{word}\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name).strip()
        
        if name and len(name) > 1:
            med.name = name.title()
            return med
        
        return None
    
    def _extract_advice(self, text: str) -> List[str]:
        """Extract medical advice"""
        advice = []
        
        patterns = [
            r'(?:advice|instructions?|notes?)\s*(?:given)?\s*[:\-]?\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                parts = re.split(r'[,;•]', match)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 3:
                        advice.append(part)
        
        # Look for common advice phrases
        common_advice = [
            r'(avoid\s+[^\n,;]+)',
            r'(take\s+rest)',
            r'(drink\s+plenty\s+of\s+[^\n,;]+)',
            r'(complete\s+the\s+course)',
        ]
        
        for pattern in common_advice:
            matches = re.findall(pattern, text, re.IGNORECASE)
            advice.extend(matches)
        
        return list(set(advice))[:5]
    
    def _extract_follow_up(self, text: str) -> Optional[str]:
        """Extract follow-up date"""
        patterns = [
            r'(?:follow\s*up|next\s*visit|review)\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:follow\s*up|next\s*visit|review)\s*[:\-]?\s*(\d{1,2}[-/\s]+[A-Za-z]+[-/\s]+\d{2,4})',
            r'(?:follow\s*up|review)\s*(?:after|in)\s*(\d+\s*(?:days?|weeks?))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _calculate_confidence(self, result: ExtractedPrescription) -> float:
        """Calculate overall extraction confidence"""
        score = 0.0
        max_score = 100.0
        
        # Essential fields
        if result.patient_name:
            score += 15
        if result.prescription_date:
            score += 15
        if result.doctor_name:
            score += 10
        if result.medications:
            score += 30
            # Bonus for complete medication info
            for med in result.medications[:5]:
                if med.dosage:
                    score += 2
                if med.frequency:
                    score += 2
                if med.duration:
                    score += 1
        
        # Additional fields
        if result.patient_age:
            score += 5
        if result.diagnosis:
            score += 5
        if result.vitals:
            score += 5
        if result.clinic_name:
            score += 3
        if result.doctor_qualification:
            score += 2
        
        return min(score / max_score, 1.0)
