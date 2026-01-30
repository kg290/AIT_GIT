"""
AI-Powered Prescription Extractor using Google Generative AI
Uses service account credentials via Vertex AI
Falls back to intelligent parsing when AI unavailable
"""
import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CREDENTIALS_FILE = BASE_DIR / "kg-hackathon-e3f03b59d928.json"
PROJECT_ID = "kg-hackathon"
LOCATION = "us-central1"

# Try to import Google GenAI (new unified SDK)
GENAI_AVAILABLE = False
genai_client = None
try:
    from google import genai
    GENAI_AVAILABLE = True
    logger.info("google-genai package loaded")
except ImportError:
    logger.warning("google-genai not installed")


@dataclass
class MedicationData:
    """Medication with full details"""
    name: str
    dosage: Optional[str] = None
    form: Optional[str] = None
    frequency: Optional[str] = None
    timing: Optional[str] = None
    duration: Optional[str] = None
    quantity: Optional[int] = None
    instructions: Optional[str] = None
    route: str = "oral"
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass 
class PrescriptionData:
    """Complete prescription data"""
    # Patient
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    patient_id: Optional[str] = None
    patient_address: Optional[str] = None
    patient_phone: Optional[str] = None
    
    # Doctor
    doctor_name: Optional[str] = None
    doctor_qualification: Optional[str] = None
    clinic_name: Optional[str] = None
    clinic_phone: Optional[str] = None
    clinic_address: Optional[str] = None
    doctor_reg_no: Optional[str] = None
    
    # Prescription
    prescription_date: Optional[str] = None
    
    # Clinical
    diagnosis: List[str] = field(default_factory=list)
    chief_complaints: List[str] = field(default_factory=list)
    vitals: Dict[str, str] = field(default_factory=dict)
    
    # Medications
    medications: List[MedicationData] = field(default_factory=list)
    
    # Additional
    advice: List[str] = field(default_factory=list)
    follow_up: Optional[str] = None
    investigations: List[str] = field(default_factory=list)
    
    # Meta
    confidence: float = 0.0
    raw_text: str = ""
    extraction_method: str = "unknown"
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['medications'] = [m.to_dict() if hasattr(m, 'to_dict') else m for m in self.medications]
        return d


class AIExtractor:
    """
    Intelligent prescription extractor 
    Uses Gemini AI via Generative Language API or Vertex AI for intelligent extraction with regex fallback
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        self.initialized = False
        self.model_name = "gemini-2.0-flash-001"
        
        if not GENAI_AVAILABLE:
            logger.warning("Google GenAI not available - will use regex parser only")
            return
        
        # Set credentials environment variable for service account
        if CREDENTIALS_FILE.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE)
        
        # Load .env file if exists
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            logger.info("Loaded environment from .env file")
        
        # Try API key first (simpler, no IAM permissions needed)
        api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        
        if api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.initialized = True
                logger.info(f"✓ Google GenAI Client initialized with API key")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize with API key: {e}")
        
        # Try Vertex AI with service account (requires aiplatform.endpoints.predict permission)
        try:
            self.client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location=LOCATION
            )
            self.initialized = True
            logger.info(f"✓ Google GenAI Client initialized with Vertex AI (project: {PROJECT_ID}, location: {LOCATION})")
            return
        except Exception as e:
            logger.warning(f"Failed to initialize with Vertex AI: {e}")
        
        if not self.initialized:
            logger.warning("AI not initialized - will use regex parser only. Either set GEMINI_API_KEY or add 'Vertex AI User' role to service account.")

    
    def extract(self, ocr_text: str) -> PrescriptionData:
        """
        Extract prescription data using AI as primary method
        Falls back to regex parser only if AI fails completely
        """
        if not ocr_text or not ocr_text.strip():
            return PrescriptionData(raw_text="", extraction_method="empty")
        
        # Try AI extraction first - this is the primary method
        if self.initialized:
            try:
                logger.info("Using Gemini AI for intelligent extraction...")
                ai_result = self._extract_with_ai(ocr_text)
                if ai_result:
                    # AI extraction succeeded - always prefer this
                    ai_result.extraction_method = "gemini_ai"
                    logger.info(f"AI extracted {len(ai_result.medications)} medications")
                    logger.info(f"Patient: {ai_result.patient_name}, Doctor: {ai_result.doctor_name}")
                    return ai_result
            except Exception as e:
                logger.warning(f"AI extraction failed, falling back to regex: {e}")
        else:
            logger.warning("Gemini AI not initialized - using regex fallback")
        
        # Fallback to regex parsing only if AI unavailable/failed
        logger.info("Using regex parser fallback...")
        return self._extract_with_parser(ocr_text)
    
    def _extract_with_ai(self, text: str) -> Optional[PrescriptionData]:
        """Extract using Gemini AI with comprehensive prompt"""
        prompt = f"""You are a medical prescription data extraction AI. Analyze this OCR text from a prescription and extract ONLY the correct information for each field.

CRITICAL RULES:
1. PATIENT NAME: Only the actual patient's name (e.g., "John Doe", "Maria Santos"). NOT addresses, NOT ages, NOT doctor names.
2. PATIENT ADDRESS: Only the actual address/location of the patient (e.g., "123 Main St, City").
3. PATIENT AGE: Only numeric age (e.g., "29", "45 years").
4. PATIENT GENDER: Only "Male" or "Female" or "M" or "F".
5. DOCTOR NAME: Only the prescribing physician's name, usually prefixed with "Dr." or followed by medical qualifications.
6. DOCTOR REGISTRATION: License number, PTR number, PRC number, Lac No, or any official registration number of the doctor.
7. CLINIC NAME: The hospital or clinic name where the prescription was issued.

8. MEDICATIONS - MOST IMPORTANT:
   - Only include ACTUAL DRUG/MEDICINE names (e.g., "Amoxicillin", "Paracetamol", "Hinox", "Metformin", "Losartan")
   - NEVER include patient names, addresses, ages, or any personal information as medications
   - NEVER include "Name", "Address", "Age", "Sex", "Physicians Sig", "Lic No", "PTR No" as medications
   - NEVER include random text fragments as medications
   - Common medication suffixes: -cillin, -mycin, -pril, -sartan, -olol, -pine, -zole, -ine, -ide
   - Parse dosage (mg, ml, g), frequency (1x, 2x, 3x daily), and duration from sig/instructions

Return this exact JSON structure:

{{
    "patient_name": "string or null - ONLY the person's name receiving treatment",
    "patient_age": "string or null - ONLY age number",
    "patient_gender": "Male/Female or null",
    "patient_id": "string or null",
    "patient_address": "string or null - ONLY address/location",
    "patient_phone": "string or null",
    "doctor_name": "string or null - ONLY doctor's name",
    "doctor_qualification": "string or null - degrees like MBBS, MD, etc",
    "doctor_reg_no": "string or null - license/registration numbers",
    "clinic_name": "string or null - hospital/clinic name",
    "clinic_address": "string or null",
    "prescription_date": "YYYY-MM-DD or original format or null",
    "diagnosis": ["array of medical diagnoses ONLY - not names or addresses"],
    "vitals": {{"bp": "120/80", "temp": "98.6F", etc}},
    "medications": [
        {{
            "name": "ACTUAL DRUG NAME ONLY (e.g., Amoxicillin, Paracetamol)",
            "dosage": "dose amount (e.g., 500mg, 250ml)",
            "form": "Tablet/Capsule/Syrup/Injection/etc",
            "frequency": "how often (e.g., 3 times a day, once daily)",
            "timing": "when to take (e.g., ~ter meals, before bed)",
            "duration": "how long (e.g., 7 days, 2 weeks)",
            "quantity": "number of units if mentioned",
            "instructions": "any special instructions"
        }}
    ],
    "advice": ["medical advice given"],
    "follow_up": "follow-up date/instruction or null"
}}

OCR TEXT TO ANALYZE:
---
{text}
---

REMEMBER: 
- Medications are DRUG NAMES ONLY, not patient info, not administrative data
- If unsure whether something is a medication, exclude it
- Return ONLY valid JSON, no explanation or markdown"""

        try:
            # Use the new google-genai client API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={'automatic_function_calling': {'disable': True}}
            )
            json_str = response.text.strip()
            
            # Clean up response
            if json_str.startswith('```'):
                json_str = re.sub(r'^```json?\n?', '', json_str)
                json_str = re.sub(r'\n?```$', '', json_str)
            
            data = json.loads(json_str)
            
            # Convert to PrescriptionData with all fields
            result = PrescriptionData(
                patient_name=data.get('patient_name'),
                patient_age=data.get('patient_age'),
                patient_gender=data.get('patient_gender'),
                patient_id=data.get('patient_id'),
                patient_address=data.get('patient_address'),
                patient_phone=data.get('patient_phone'),
                doctor_name=data.get('doctor_name'),
                doctor_qualification=data.get('doctor_qualification'),
                doctor_reg_no=data.get('doctor_reg_no'),
                clinic_name=data.get('clinic_name'),
                clinic_address=data.get('clinic_address'),
                prescription_date=data.get('prescription_date'),
                diagnosis=data.get('diagnosis', []),
                vitals=data.get('vitals', {}),
                advice=data.get('advice', []),
                follow_up=data.get('follow_up'),
                raw_text=text,
                confidence=0.90
            )
            
            # Parse medications
            for med_data in data.get('medications', []):
                if med_data.get('name'):
                    med = MedicationData(
                        name=med_data['name'],
                        dosage=med_data.get('dosage'),
                        form=med_data.get('form'),
                        frequency=med_data.get('frequency'),
                        timing=med_data.get('timing'),
                        duration=med_data.get('duration'),
                        instructions=med_data.get('instructions')
                    )
                    result.medications.append(med)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"AI extraction error: {e}")
            return None
    
    def _extract_with_parser(self, text: str) -> PrescriptionData:
        """Enhanced regex-based extraction"""
        result = PrescriptionData(raw_text=text, extraction_method="regex_parser")
        
        # Normalize text
        text_lower = text.lower()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Extract patient info
        result.patient_name = self._find_patient_name(text, lines)
        result.patient_age, result.patient_gender = self._find_age_gender(text)
        result.patient_id = self._find_pattern(text, [
            r'(?:patient\s*)?(?:id|reg\.?\s*no)\s*[:\-]?\s*(\d+)',
            r'ID:\s*(\d+)',
        ])
        result.patient_address = self._find_patient_address(text, lines)
        result.patient_phone = self._find_pattern(text, [
            r'(?:phone|mobile|contact|tel)\s*[:\-]?\s*(\+?\d[\d\s\-]{8,15})',
        ])
        
        # Extract doctor info
        result.doctor_name = self._find_doctor_name(text, lines)
        result.doctor_qualification = self._find_qualifications(text)
        result.clinic_name = self._find_clinic_name(text, lines)
        result.doctor_reg_no = self._find_pattern(text, [
            r'(?:reg\.?\s*no|license|lic\.?\s*no|ptr\.?\s*no|lac\.?\s*no)\s*[:\-]?\s*(\d+)',
        ])
        
        # Extract date
        result.prescription_date = self._find_date(text)
        
        # Extract vitals
        result.vitals = self._find_vitals(text)
        
        # Extract diagnosis
        result.diagnosis = self._find_diagnosis(text)
        
        # Extract medications - most important part
        result.medications = self._find_medications(text, lines)
        
        # Extract advice and follow-up
        result.advice = self._find_advice(text)
        result.follow_up = self._find_follow_up(text)
        
        # Calculate confidence
        result.confidence = self._calc_confidence(result)
        
        return result
    
    def _find_clinic_name(self, text: str, lines: List[str]) -> Optional[str]:
        """Find clinic/hospital name"""
        patterns = [
            r'([A-Za-z][A-Za-z\s\.]+(?:clinic|hospital|medical\s*(?:center|centre)|health\s*(?:center|centre)|nursing\s*home))',
            r'((?:clinic|hospital)\s*[:\-]?\s*[A-Za-z][A-Za-z\s\.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                if 3 < len(name) < 80:
                    return name.title()
        return None
    
    def _find_patient_name(self, text: str, lines: List[str]) -> Optional[str]:
        """Find patient name with improved accuracy"""
        patterns = [
            # "Name: John Doe" or "Patient Name: John Doe"
            r'(?:patient\s*)?name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,35})(?=\s*(?:\d|age|male|female|m/|f/|\(|,|\n|address|$))',
            # "Mr./Mrs./Ms. John Doe"
            r'(?:mr\.?|mrs\.?|ms\.?|master|baby|miss)\s+([A-Za-z][A-Za-z\s\.]{2,30})',
            # "ID: 123 - John Doe"
            r'ID:\s*\d+\s*[-–]\s*([A-Za-z][A-Za-z\s]+?)(?:\s*\()',
            # Line starting with Name followed by name
            r'^name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]+?)$',
        ]
        
        # First check lines for explicit name markers
        for line in lines:
            line_clean = line.strip()
            if re.match(r'^(?:patient\s*)?name\s*[:\-]', line_clean, re.I):
                # Extract name from this line
                name_match = re.search(r'name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]+)', line_clean, re.I)
                if name_match:
                    name = name_match.group(1).strip()
                    name = re.sub(r'\s+', ' ', name)
                    # Remove any trailing numbers or keywords
                    name = re.split(r'\s+(?:age|sex|male|female|address|\d)', name, flags=re.I)[0].strip()
                    if 2 < len(name) < 50 and not re.search(r'address|city|street', name, re.I):
                        return name.title()
        
        # Try general patterns
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                # Filter out common non-name words
                if not re.search(r'^(date|doctor|dr|clinic|hospital|age|sex|address|city)', name, re.I):
                    if 2 < len(name) < 50:
                        return name.title()
        return None
    
    def _find_patient_address(self, text: str, lines: List[str]) -> Optional[str]:
        """Find patient address"""
        patterns = [
            r'(?:address)\s*[:\-]?\s*([A-Za-z0-9\s,\.\-]+?)(?=\s*(?:phone|contact|age|sex|date|\n\n))',
            r'(?:address)\s*[:\-]?\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                address = re.sub(r'\s+', ' ', address)
                if 5 < len(address) < 150:
                    return address.title()
        return None
    
    def _find_age_gender(self, text: str):
        """Find age and gender with improved patterns"""
        age = None
        gender = None
        
        # Age patterns
        age_patterns = [
            r'age\s*[:\-]?\s*(\d{1,3})\s*(?:yrs?|years?|y\.?)?',
            r'(\d{1,3})\s*(?:yrs?|years?|y/?o)\b',
            r'age\s*(\d{1,3})',
        ]
        
        for pattern in age_patterns:
            age_match = re.search(pattern, text, re.IGNORECASE)
            if age_match:
                val = int(age_match.group(1))
                if 0 < val < 150:
                    age = f"{val} years"
                    break
        
        # Gender patterns
        if re.search(r'sex\s*[:\-]?\s*m(?:ale)?|\(M\)|\bM\s*/|/\s*M\b|\bmale\b', text, re.IGNORECASE):
            gender = "Male"
        elif re.search(r'sex\s*[:\-]?\s*f(?:emale)?|\(F\)|\bF\s*/|/\s*F\b|\bfemale\b', text, re.IGNORECASE):
            gender = "Female"
        
        return age, gender
    
    def _find_doctor_name(self, text: str, lines: List[str]) -> Optional[str]:
        """Find doctor name"""
        # Check first and last lines (usually where doctor info appears)
        search_lines = lines[:7] + lines[-7:] if len(lines) > 10 else lines
        
        for line in search_lines:
            # Look for "Dr. Name" pattern
            match = re.search(r'(?:Dr\.?|Doctor)\s+([A-Za-z][A-Za-z\s\.]{2,30})', line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                # Clean qualifications from name
                name = re.split(r'\s*(?:MBBS|MD|MS|FRCS|DNB|MRCP|DM)', name)[0].strip()
                if 2 < len(name) < 40 and not re.search(r'clinic|hospital|medical', name, re.I):
                    return f"Dr. {name.title()}"
        return None
    
    def _find_qualifications(self, text: str) -> Optional[str]:
        """Find medical qualifications"""
        quals = []
        qual_list = ['MBBS', 'MD', 'MS', 'FRCS', 'MRCP', 'DNB', 'DM', 'MCh', 'BDS', 'MDS', 'BAMS', 'BHMS']
        for q in qual_list:
            if re.search(rf'\b{q}\b', text, re.IGNORECASE):
                quals.append(q)
        return ', '.join(quals) if quals else None
    
    def _find_date(self, text: str) -> Optional[str]:
        """Find prescription date"""
        patterns = [
            r'(?:date)\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:date)\s*[:\-]?\s*(\d{1,2}[-/\s]*[A-Za-z]{3,9}[-/\s]*\d{2,4})',
            r'(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _find_vitals(self, text: str) -> Dict[str, str]:
        """Find vital signs"""
        vitals = {}
        
        bp = re.search(r'(?:bp|b\.p\.?)\s*[:\-]?\s*(\d{2,3}\s*/\s*\d{2,3})', text, re.I)
        if bp:
            vitals['blood_pressure'] = bp.group(1).replace(' ', '') + ' mmHg'
        
        temp = re.search(r'(?:temp)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?)', text, re.I)
        if temp:
            vitals['temperature'] = temp.group(1) + '°F'
        
        pulse = re.search(r'(?:pulse|pr)\s*[:\-]?\s*(\d{2,3})', text, re.I)
        if pulse:
            vitals['pulse'] = pulse.group(1) + ' bpm'
        
        spo2 = re.search(r'(?:spo2|o2)\s*[:\-]?\s*(\d{2,3})', text, re.I)
        if spo2:
            vitals['spo2'] = spo2.group(1) + '%'
        
        return vitals
    
    def _find_diagnosis(self, text: str) -> List[str]:
        """Find diagnosis"""
        diagnoses = []
        pattern = r'(?:diagnosis|dx|c/c|chief\s*complaint|complaint)\s*[:\-]?\s*([^\n]{3,50})'
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            parts = re.split(r'[,;]', m)
            for p in parts:
                p = p.strip()
                if p and len(p) > 2:
                    diagnoses.append(p.capitalize())
        return diagnoses[:5]
    
    def _find_medications(self, text: str, lines: List[str]) -> List[MedicationData]:
        """Find all medications with intelligent filtering"""
        medications = []
        
        # Known medication names database (common drugs)
        KNOWN_MEDICATIONS = {
            # Antibiotics
            'amoxicillin', 'amoxyclav', 'azithromycin', 'ciprofloxacin', 'cephalexin',
            'cefixime', 'ceftriaxone', 'doxycycline', 'metronidazole', 'norfloxacin',
            'ofloxacin', 'levofloxacin', 'penicillin', 'erythromycin', 'clindamycin',
            'augmentin', 'cefpodoxime', 'moxifloxacin', 'clarithromycin',
            # Pain/Fever
            'paracetamol', 'acetaminophen', 'ibuprofen', 'diclofenac', 'aspirin',
            'naproxen', 'aceclofenac', 'piroxicam', 'tramadol', 'dolo', 'crocin',
            'combiflam', 'brufen', 'voveran', 'zerodol', 'hifenac',
            # Antacids/GI
            'omeprazole', 'pantoprazole', 'ranitidine', 'famotidine', 'esomeprazole',
            'rabeprazole', 'domperidone', 'ondansetron', 'metoclopramide',
            'sucralfate', 'antacid', 'gelusil', 'digene', 'pan', 'rantac',
            # Antihistamines
            'cetirizine', 'loratadine', 'fexofenadine', 'levocetirizine', 'chlorpheniramine',
            'allegra', 'zyrtec', 'montair', 'montelukast', 'bilastine',
            # Cardiac
            'amlodipine', 'atenolol', 'metoprolol', 'losartan', 'telmisartan',
            'ramipril', 'enalapril', 'aspirin', 'clopidogrel', 'atorvastatin',
            'rosuvastatin', 'ecosprin', 'cardace', 'telma',
            # Diabetes
            'metformin', 'glimepiride', 'glipizide', 'sitagliptin', 'insulin',
            'voglibose', 'pioglitazone', 'januvia', 'glycomet',
            # Respiratory
            'salbutamol', 'theophylline', 'montelukast', 'budesonide', 'levosalbutamol',
            'deriphyllin', 'asthalin', 'foracort', 'seroflo', 'budecort',
            # Vitamins/Supplements
            'vitamin', 'multivitamin', 'calcium', 'iron', 'folic', 'zinc',
            'b12', 'b-complex', 'bcosules', 'shelcal', 'calcirol', 'neurobion',
            'folvite', 'feronia', 'livogen', 'zincovit',
            # Steroids
            'prednisolone', 'prednisone', 'dexamethasone', 'hydrocortisone', 'betamethasone',
            'wysolone', 'decdan', 'deflazacort',
            # Others common
            'hinox', 'norflox', 'taxim', 'monocef', 'injection', 'syrup',
            'tablet', 'capsule', 'cream', 'ointment', 'drops', 'gel',
            'suspension', 'powder', 'inhaler', 'spray'
        }
        
        # Words that indicate NON-medication content (filter these out)
        NON_MED_INDICATORS = [
            # Patient info
            r'^\s*name\b', r'\bpatient\b', r'\baddress\b', r'\bcontact\b', r'\bphone\b',
            r'\bsex\b', r'\bgender\b', r'\b(male|female)\b', r'\bage\s*\d', r'\byears?\s*old\b',
            r'\bmr\.?\b', r'\bmrs\.?\b', r'\bms\.?\b', r'\bmaster\b', r'\bbaby\b',
            # Doctor/Clinic info
            r'\bdoctor\b', r'\bdr\.?\s+[a-z]', r'\bphysician', r'\bclinic\b', r'\bhospital\b',
            r'\bmbbs\b', r'\bmd\b', r'\bms\b', r'\bfrcs\b', r'\breg\.?\s*no', r'\blic\.?\s*no',
            r'\bptr\.?\s*no', r'\blac\.?\s*no', r'\blicense\b', r'\bregistration\b',
            # Section headers
            r'^r[xX]$', r'^\s*rx\s*$', r'\bprescription\b', r'\bsignature\b', r'\bsig\s*$',
            r'\bphysicians?\s*sig', r'\badvice\b', r'\bfollow\s*up', r'\bnext\s*visit',
            r'\binvestigation', r'\bdiagnosis\b', r'\bcomplaint', r'\bhistory\b',
            # Administrative
            r'\bdate\b', r'\btime\b', r'\bopd\b', r'\bipd\b', r'\bward\b', r'\bbed\b',
            r'\bbill\b', r'\breceipt\b', r'\btoken\b', r'\bslip\b',
            # Location
            r'\bcity\b', r'\bstreet\b', r'\broad\b', r'\bdistrict\b', r'\bstate\b',
            r'\bpin\s*code', r'\bzip\b', r'\barea\b', r'\bfloor\b', r'\bbuilding\b',
            # Random noise
            r'^\d+$', r'^not?\b', r'^\s*$', r'^[-_=]+$',
        ]
        
        # Find Rx section start
        rx_idx = -1
        rx_end_idx = len(lines)
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Find start of Rx section
            if re.search(r'^r[xX]$|^\s*rx\s*:|^rx\b|℞', line, re.I) and rx_idx < 0:
                rx_idx = i
            # Find end markers (advice, follow-up, signature)
            if rx_idx >= 0 and re.search(r'\badvice\b|\bfollow\s*up|\bnext\s*visit|\bsignature|\bphysician.*sig', line, re.I):
                rx_end_idx = i
                break
        
        # Process lines in Rx section (or all lines if no Rx found)
        if rx_idx >= 0:
            med_lines = lines[rx_idx+1:rx_end_idx]
        else:
            med_lines = lines
        
        for line in med_lines:
            line = line.strip()
            if not line or len(line) < 2:
                continue
            
            line_lower = line.lower()
            
            # Skip if matches any non-medication indicator
            is_non_med = False
            for pattern in NON_MED_INDICATORS:
                if re.search(pattern, line_lower):
                    is_non_med = True
                    break
            
            if is_non_med:
                continue
            
            # Check if line contains a known medication or medication form
            contains_known_med = False
            for med in KNOWN_MEDICATIONS:
                if med in line_lower:
                    contains_known_med = True
                    break
            
            # Also check for medication patterns (dosage, frequency)
            has_med_pattern = bool(re.search(
                r'\d+\s*(?:mg|mcg|ml|g|iu|units?)\b|'  # Dosage
                r'\b(?:od|bd|tds|qid|hs|sos|prn)\b|'  # Frequency
                r'\b(?:tab|cap|syr|inj|cream|oint|drops?|gel)\b|'  # Form
                r'\d+-\d+-\d+|'  # Dosing pattern like 1-0-1
                r'\b(?:before|after)\s*(?:food|meal)\b|'  # Timing
                r'\bx\s*\d+\s*(?:days?|weeks?)\b',  # Duration
                line_lower
            ))
            
            # Only process if it looks like medication info
            if contains_known_med or has_med_pattern:
                med = self._parse_med_line(line)
                if med and self._validate_medication(med):
                    medications.append(med)
        
        return medications
    
    def _validate_medication(self, med: MedicationData) -> bool:
        """Validate that the extracted medication is valid"""
        if not med.name:
            return False
        
        name_lower = med.name.lower().strip()
        
        # Reject if name is too short or too long
        if len(name_lower) < 2 or len(name_lower) > 50:
            return False
        
        # Reject common non-medication words
        reject_words = [
            'name', 'patient', 'address', 'age', 'sex', 'male', 'female',
            'doctor', 'physician', 'clinic', 'hospital', 'date', 'time',
            'signature', 'advice', 'follow', 'next', 'visit', 'ptr', 'lac',
            'license', 'registration', 'city', 'street', 'road', 'not',
            'day', 'for', 'sig', 'physicians', 'rx', 'prescription',
            'contact', 'phone', 'email', 'mobile', 'building', 'floor'
        ]
        
        # Check if name is just a reject word
        for word in reject_words:
            if name_lower == word or name_lower.startswith(word + ' '):
                return False
        
        # Reject if name contains numbers at the start (like "29 Sex M")
        if re.match(r'^\d+\s', med.name):
            return False
        
        # Reject if it's just administrative info
        if re.search(r'\b(?:no|number|id)\s*\d+', name_lower):
            return False
        
        return True
    
    def _parse_med_line(self, line: str) -> Optional[MedicationData]:
        """Parse a single medication line"""
        original = line
        
        # Remove numbering
        line = re.sub(r'^\s*\d+[.\)]\s*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        
        # Initialize
        med_name = ""
        dosage = None
        form = None
        frequency = None
        timing = None
        duration = None
        
        # Extract form
        forms_map = {
            'tab': 'Tablet', 'cap': 'Capsule', 'syr': 'Syrup', 'syp': 'Syrup',
            'inj': 'Injection', 'cream': 'Cream', 'oint': 'Ointment',
            'drops': 'Drops', 'gel': 'Gel', 'susp': 'Suspension',
            'powder': 'Powder', 'sachet': 'Sachet', 'inhaler': 'Inhaler'
        }
        
        for abbr, full in forms_map.items():
            if re.search(rf'\b{abbr}\.?\b', line, re.I):
                form = full
                line = re.sub(rf'\b{abbr}\.?\s*', '', line, flags=re.I)
                break
        
        # Extract dosage
        dos_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)', line, re.I)
        if dos_match:
            dosage = f"{dos_match.group(1)} {dos_match.group(2).lower()}"
            line = line[:dos_match.start()] + ' ' + line[dos_match.end():]
        
        # Extract frequency
        freq_map = {
            r'\b(OD|O\.D\.?|once\s*daily)\b': 'Once daily',
            r'\b(BD|B\.D\.?|BID|twice\s*daily|1-0-1)\b': 'Twice daily',
            r'\b(TDS|T\.D\.S\.?|TID|thrice\s*daily|1-1-1)\b': 'Three times daily',
            r'\b(QID|four\s*times|1-1-1-1)\b': 'Four times daily',
            r'\b(HS|H\.S\.?|at\s*(?:bed)?night|0-0-1)\b': 'At bedtime',
            r'\b(SOS|S\.O\.S\.?|as\s*needed|PRN)\b': 'As needed',
        }
        
        for pattern, freq_text in freq_map.items():
            if re.search(pattern, line, re.I):
                frequency = freq_text
                line = re.sub(pattern, '', line, flags=re.I)
                break
        
        # Extract timing
        timing_map = {
            r'\b(AC|A\.C\.?|before\s*(?:food|meal)|empty\s*stomach)\b': 'Before food',
            r'\b(PC|P\.C\.?|after\s*(?:food|meal)|with\s*food)\b': 'After food',
        }
        
        for pattern, timing_text in timing_map.items():
            if re.search(pattern, line, re.I):
                timing = timing_text
                line = re.sub(pattern, '', line, flags=re.I)
                break
        
        # Extract duration
        dur_match = re.search(r'(?:x\s*)?(\d+)\s*(days?|weeks?|D|W)', line, re.I)
        if dur_match:
            num = dur_match.group(1)
            unit = dur_match.group(2).lower()
            if unit.startswith('d'):
                duration = f"{num} days"
            else:
                duration = f"{num} weeks"
            line = line[:dur_match.start()] + ' ' + line[dur_match.end():]
        
        # Extract quantity (Tot: X)
        qty_match = re.search(r'(?:tot|qty|total)[:\s]*(\d+)', line, re.I)
        quantity = int(qty_match.group(1)) if qty_match else None
        if qty_match:
            line = line[:qty_match.start()] + ' ' + line[qty_match.end():]
        
        # Clean remaining text as drug name
        med_name = re.sub(r'[^A-Za-z0-9\s\-]', '', line).strip()
        med_name = re.sub(r'\s+', ' ', med_name)
        
        # Remove noise words
        noise = ['morning', 'evening', 'night', 'afternoon', 'days', 'weeks', 
                 'before', 'after', 'food', 'meal', 'daily', 'total', 'qty']
        for word in noise:
            med_name = re.sub(rf'\b{word}\b', '', med_name, flags=re.I)
        med_name = re.sub(r'\s+', ' ', med_name).strip()
        
        # Validate
        if med_name and len(med_name) > 1 and not med_name.isdigit():
            return MedicationData(
                name=med_name.title(),
                dosage=dosage,
                form=form,
                frequency=frequency,
                timing=timing,
                duration=duration,
                quantity=quantity
            )
        
        return None
    
    def _find_advice(self, text: str) -> List[str]:
        """Find medical advice"""
        advice = []
        pattern = r'(?:advice|instruction)\s*[:\-]?\s*([^\n]+)'
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            parts = re.split(r'[,;•]', m)
            advice.extend([p.strip() for p in parts if p.strip() and len(p.strip()) > 3])
        
        # Common advice phrases
        common = [
            r'(avoid\s+[^\n,;]{3,30})',
            r'(take\s+rest)',
            r'(drink\s+plenty\s+[^\n,;]{3,20})',
            r'(complete\s+(?:the\s+)?course)',
        ]
        for p in common:
            matches = re.findall(p, text, re.I)
            advice.extend(matches)
        
        return list(set(advice))[:5]
    
    def _find_follow_up(self, text: str) -> Optional[str]:
        """Find follow-up date"""
        patterns = [
            r'(?:follow\s*up|next\s*visit|review)\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:follow\s*up|next\s*visit|review)\s*[:\-]?\s*(\d{1,2}[-/\s]*[A-Za-z]{3,9}[-/\s]*\d{2,4})',
            r'(?:follow\s*up|review)\s*(?:after|in)\s*(\d+\s*(?:days?|weeks?))',
        ]
        for p in patterns:
            match = re.search(p, text, re.I)
            if match:
                return match.group(1).strip()
        return None
    
    def _find_pattern(self, text: str, patterns: List[str]) -> Optional[str]:
        """Find first matching pattern"""
        for p in patterns:
            match = re.search(p, text, re.I)
            if match:
                return match.group(1).strip()
        return None
    
    def _calc_confidence(self, result: PrescriptionData) -> float:
        """Calculate extraction confidence"""
        score = 0
        if result.patient_name:
            score += 15
        if result.prescription_date:
            score += 15
        if result.doctor_name:
            score += 10
        if result.medications:
            score += 30
            for med in result.medications[:5]:
                if med.dosage:
                    score += 3
                if med.frequency:
                    score += 2
        if result.vitals:
            score += 10
        if result.diagnosis:
            score += 10
        
        return min(score / 100, 1.0)


# Create singleton
ai_extractor = AIExtractor()
