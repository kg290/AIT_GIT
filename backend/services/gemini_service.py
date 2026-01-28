"""
Gemini AI Service - For understanding and structuring medical prescriptions
Uses Google Cloud Vertex AI with service account authentication
"""
import logging
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import base64

import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, HarmCategory, HarmBlockThreshold
from google.oauth2 import service_account

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GeminiExtractionResult:
    """Result from Gemini prescription analysis"""
    success: bool
    raw_response: str
    
    # Prescription Header
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    prescription_date: Optional[str] = None
    prescriber_name: Optional[str] = None
    prescriber_qualification: Optional[str] = None
    clinic_name: Optional[str] = None
    clinic_address: Optional[str] = None
    
    # Medications
    medications: List[Dict] = field(default_factory=list)
    
    # Diagnosis & Symptoms
    diagnoses: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    
    # Vitals
    vitals: Dict[str, str] = field(default_factory=dict)
    
    # Additional Instructions
    instructions: List[str] = field(default_factory=list)
    follow_up: Optional[str] = None
    
    # Confidence
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)
    
    # Errors
    error: Optional[str] = None


class GeminiService:
    """
    Google Gemini AI Service via Vertex AI for medical document understanding
    Uses service account credentials for authentication (no API key needed)
    
    Features:
    - Direct image analysis (Vision + Language)
    - OCR text enhancement
    - Prescription structuring
    - Multi-language support
    - Handwriting interpretation
    """
    
    def __init__(self):
        self.project_id = None
        self.location = "us-central1"  # Default Vertex AI location
        self.model = None
        self.initialized = False
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Vertex AI with service account credentials"""
        try:
            # Load credentials from service account JSON
            credentials_path = settings.GOOGLE_APPLICATION_CREDENTIALS
            
            if not os.path.exists(credentials_path):
                logger.error(f"Service account file not found: {credentials_path}")
                return
            
            # Load credentials and get project ID
            with open(credentials_path, 'r') as f:
                creds_data = json.load(f)
                self.project_id = creds_data.get('project_id')
            
            if not self.project_id:
                logger.error("No project_id found in service account JSON")
                return
            
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize Vertex AI
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials
            )
            
            # Initialize Gemini model
            self.model = GenerativeModel("gemini-1.5-flash-001")
            self.initialized = True
            
            logger.info(f"Gemini initialized successfully with project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini via Vertex AI: {e}")
            self.initialized = False
        
        # Safety settings - allow medical content
        self.safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.OFF
            ),
        ]
        
        # Prescription extraction prompt
        self.extraction_prompt = """You are a medical prescription analysis expert. Analyze the following prescription image or text and extract ALL information in a structured JSON format.

IMPORTANT INSTRUCTIONS:
1. Extract EVERY medication with complete details
2. Handle handwritten text carefully - make your best interpretation
3. Recognize common medical abbreviations (OD=once daily, BD/BID=twice daily, TDS/TID=three times daily, QID=four times daily, PRN=as needed, AC=before meals, PC=after meals, HS=at bedtime, SOS=if needed)
4. Convert abbreviations to full forms in output
5. If information is unclear, provide your best interpretation with a note
6. Handle multiple formats: Indian, US, international prescriptions

Extract the following in JSON format:
{
    "patient": {
        "name": "Patient's full name",
        "age": "Age with unit (e.g., '45 years' or '6 months')",
        "gender": "Male/Female/Other",
        "weight": "Weight if mentioned",
        "contact": "Phone/address if present"
    },
    "prescriber": {
        "name": "Doctor's name",
        "qualification": "MD, MBBS, etc.",
        "registration_number": "Medical license number if present",
        "clinic_name": "Clinic/Hospital name",
        "clinic_address": "Address",
        "contact": "Phone number"
    },
    "prescription_date": "Date in YYYY-MM-DD format",
    "diagnosis": ["List of diagnoses or chief complaints"],
    "symptoms": ["List of symptoms mentioned"],
    "vitals": {
        "blood_pressure": "BP reading",
        "pulse": "Heart rate",
        "temperature": "Body temperature",
        "weight": "Weight",
        "spo2": "Oxygen saturation",
        "other": "Any other vitals"
    },
    "medications": [
        {
            "name": "Medication name (generic preferred)",
            "brand_name": "Brand name if different from generic",
            "dosage": "Strength (e.g., 500mg, 10mg/5ml)",
            "form": "Tablet/Capsule/Syrup/Injection/Cream/etc.",
            "frequency": "How often (e.g., 'twice daily', 'every 8 hours')",
            "timing": "When to take (before meals, after meals, at bedtime, etc.)",
            "duration": "How long to take (e.g., '7 days', '2 weeks', 'ongoing')",
            "quantity": "Total quantity prescribed",
            "route": "Oral/Topical/IV/IM/etc.",
            "special_instructions": "Any specific instructions"
        }
    ],
    "instructions": ["General instructions for the patient"],
    "follow_up": "Follow-up date or instructions",
    "investigations": ["Any lab tests or investigations ordered"],
    "confidence": 0.0 to 1.0,
    "notes": ["Any notes about unclear or ambiguous information"]
}

If any field is not present or unclear, use null for that field.
Respond ONLY with the JSON object, no other text."""

    def analyze_image(self, image_path: str) -> GeminiExtractionResult:
        """
        Analyze a prescription image directly using Gemini Vision via Vertex AI
        
        Args:
            image_path: Path to the image file
            
        Returns:
            GeminiExtractionResult with extracted data
        """
        if not self.initialized or not self.model:
            return GeminiExtractionResult(
                success=False,
                raw_response="",
                error="Gemini model not initialized. Check service account credentials and Vertex AI API enablement."
            )
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Determine MIME type
            ext = image_path.lower().split('.')[-1]
            mime_types = {
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp',
                'pdf': 'application/pdf'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # Create image part for Vertex AI
            image_part = Part.from_data(data=image_data, mime_type=mime_type)
            
            # Generate content with image
            response = self.model.generate_content(
                [self.extraction_prompt, image_part],
                safety_settings=self.safety_settings,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "max_output_tokens": 4096
                }
            )
            
            return self._parse_response(response.text)
            
        except Exception as e:
            logger.error(f"Gemini image analysis failed: {e}")
            return GeminiExtractionResult(
                success=False,
                raw_response="",
                error=str(e)
            )
    
    def analyze_text(self, ocr_text: str) -> GeminiExtractionResult:
        """
        Analyze OCR-extracted text using Gemini via Vertex AI
        
        Args:
            ocr_text: Text extracted from prescription via OCR
            
        Returns:
            GeminiExtractionResult with extracted data
        """
        if not self.initialized or not self.model:
            return GeminiExtractionResult(
                success=False,
                raw_response="",
                error="Gemini model not initialized. Check service account credentials."
            )
        
        if not ocr_text or not ocr_text.strip():
            return GeminiExtractionResult(
                success=False,
                raw_response="",
                error="No text provided for analysis"
            )
        
        try:
            # Create prompt with OCR text
            prompt = f"""{self.extraction_prompt}

Here is the prescription text (extracted via OCR, may contain errors):

---
{ocr_text}
---

Analyze this prescription and extract all information in JSON format. 
Handle any OCR errors intelligently (e.g., 'Paracetam0l' should be interpreted as 'Paracetamol')."""

            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "max_output_tokens": 4096
                }
            )
            
            return self._parse_response(response.text)
            
        except Exception as e:
            logger.error(f"Gemini text analysis failed: {e}")
            return GeminiExtractionResult(
                success=False,
                raw_response="",
                error=str(e)
            )
    
    def enhance_ocr_text(self, raw_ocr_text: str) -> str:
        """
        Use Gemini to clean and enhance OCR text
        
        Args:
            raw_ocr_text: Raw OCR output with potential errors
            
        Returns:
            Cleaned and enhanced text
        """
        if not self.initialized or not self.model:
            return raw_ocr_text
        
        try:
            prompt = f"""You are an OCR post-processing expert for medical documents. 
Clean and correct the following OCR text from a medical prescription:

1. Fix common OCR errors (0/O confusion, l/1 confusion, etc.)
2. Correct misspelled medical terms and drug names
3. Fix broken words and line breaks
4. Preserve the original structure
5. Do NOT add information that isn't there

Original OCR text:
---
{raw_ocr_text}
---

Provide the cleaned text only, no explanations:"""

            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2048
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini OCR enhancement failed: {e}")
            return raw_ocr_text
    
    def _parse_response(self, response_text: str) -> GeminiExtractionResult:
        """Parse Gemini response into structured result"""
        try:
            # Clean response - remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Parse JSON
            data = json.loads(cleaned)
            
            # Extract patient info
            patient = data.get('patient', {}) or {}
            prescriber = data.get('prescriber', {}) or {}
            vitals = data.get('vitals', {}) or {}
            
            # Build medications list
            medications = []
            for med in data.get('medications', []):
                if med:
                    medications.append({
                        'medication_name': med.get('name', ''),
                        'brand_name': med.get('brand_name'),
                        'dosage': med.get('dosage', ''),
                        'form': med.get('form', ''),
                        'frequency': med.get('frequency', ''),
                        'timing': med.get('timing', ''),
                        'duration': med.get('duration', ''),
                        'quantity': med.get('quantity'),
                        'route': med.get('route', 'oral'),
                        'special_instructions': med.get('special_instructions', '')
                    })
            
            return GeminiExtractionResult(
                success=True,
                raw_response=response_text,
                patient_name=patient.get('name'),
                patient_age=patient.get('age'),
                patient_gender=patient.get('gender'),
                prescription_date=data.get('prescription_date'),
                prescriber_name=prescriber.get('name'),
                prescriber_qualification=prescriber.get('qualification'),
                clinic_name=prescriber.get('clinic_name'),
                clinic_address=prescriber.get('clinic_address'),
                medications=medications,
                diagnoses=data.get('diagnosis', []) or [],
                symptoms=data.get('symptoms', []) or [],
                vitals=vitals,
                instructions=data.get('instructions', []) or [],
                follow_up=data.get('follow_up'),
                confidence=data.get('confidence', 0.8),
                notes=data.get('notes', []) or []
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            return GeminiExtractionResult(
                success=False,
                raw_response=response_text,
                error=f"JSON parse error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return GeminiExtractionResult(
                success=False,
                raw_response=response_text,
                error=str(e)
            )
    
    def to_dict(self, result: GeminiExtractionResult) -> Dict:
        """Convert GeminiExtractionResult to dictionary"""
        return {
            'success': result.success,
            'patient': {
                'name': result.patient_name,
                'age': result.patient_age,
                'gender': result.patient_gender
            },
            'prescriber': {
                'name': result.prescriber_name,
                'qualification': result.prescriber_qualification,
                'clinic_name': result.clinic_name,
                'clinic_address': result.clinic_address
            },
            'prescription_date': result.prescription_date,
            'medications': result.medications,
            'diagnoses': result.diagnoses,
            'symptoms': result.symptoms,
            'vitals': result.vitals,
            'instructions': result.instructions,
            'follow_up': result.follow_up,
            'confidence': result.confidence,
            'notes': result.notes,
            'error': result.error
        }
