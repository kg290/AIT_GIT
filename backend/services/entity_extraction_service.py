"""
Entity Extraction Service - Medical NER and entity detection
Handles: Medications, dosages, frequencies, symptoms, diagnoses, vitals, dates
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Extracted medical entity"""
    entity_type: str
    text: str
    normalized_text: Optional[str]
    confidence: float
    start_pos: int
    end_pos: int
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""


@dataclass
class ExtractionResult:
    """Complete extraction result"""
    entities: List[ExtractedEntity]
    medications: List[Dict]
    dosages: List[Dict]
    frequencies: List[Dict]
    routes: List[Dict]
    durations: List[Dict]
    symptoms: List[Dict]
    diagnoses: List[Dict]
    vitals: List[Dict]
    dates: List[Dict]
    confidence: float


class EntityExtractionService:
    """
    Medical Named Entity Recognition Service
    
    Uses pattern matching and medical dictionaries for entity extraction.
    Can be enhanced with ML models for better accuracy.
    """
    
    def __init__(self):
        # Load medication database
        self.medications = self._load_medication_database()
        
        # Load symptom database
        self.symptoms_db = self._load_symptom_database()
        
        # Load diagnosis database
        self.diagnoses_db = self._load_diagnosis_database()
        
        # Route patterns
        self.route_patterns = {
            'oral': ['oral', 'po', 'p.o.', 'by mouth', 'orally'],
            'intravenous': ['iv', 'i.v.', 'intravenous', 'intravenously'],
            'intramuscular': ['im', 'i.m.', 'intramuscular'],
            'subcutaneous': ['sc', 's.c.', 'subcutaneous', 'subcut'],
            'topical': ['topical', 'top', 'externally', 'apply'],
            'inhaled': ['inhaled', 'inhalation', 'inh', 'nebulizer'],
            'sublingual': ['sublingual', 'sl', 's.l.', 'under tongue'],
            'rectal': ['rectal', 'pr', 'p.r.', 'per rectum'],
            'ophthalmic': ['ophthalmic', 'eye drops', 'od', 'os', 'ou'],
            'otic': ['otic', 'ear drops'],
            'nasal': ['nasal', 'intranasal'],
            'transdermal': ['transdermal', 'patch'],
        }
        
        # Vital sign patterns
        self.vital_patterns = {
            'blood_pressure': [
                r'(?:bp|blood\s*pressure)[:\s]*(\d{2,3})[/\\](\d{2,3})',
                r'(\d{2,3})[/\\](\d{2,3})\s*(?:mmhg|mm\s*hg)?',
            ],
            'heart_rate': [
                r'(?:hr|heart\s*rate|pulse)[:\s]*(\d{2,3})\s*(?:bpm|/min)?',
                r'(\d{2,3})\s*bpm',
            ],
            'temperature': [
                r'(?:temp|temperature)[:\s]*(\d{2,3}(?:\.\d)?)\s*(?:°?[fc])?',
                r'(\d{2,3}\.\d)\s*(?:°?[fc])',
            ],
            'respiratory_rate': [
                r'(?:rr|resp(?:iratory)?\s*rate)[:\s]*(\d{1,2})\s*(?:/min)?',
            ],
            'oxygen_saturation': [
                r'(?:spo2|o2\s*sat|oxygen\s*sat)[:\s]*(\d{2,3})\s*%?',
                r'(\d{2,3})\s*%\s*(?:on\s*(?:ra|room\s*air))?',
            ],
            'blood_sugar': [
                r'(?:bs|blood\s*sugar|glucose|fbs|rbs|ppbs)[:\s]*(\d{2,3})\s*(?:mg/dl)?',
            ],
            'weight': [
                r'(?:wt|weight)[:\s]*(\d{2,3}(?:\.\d)?)\s*(?:kg|lb)?',
            ],
            'height': [
                r'(?:ht|height)[:\s]*(\d{2,3}(?:\.\d)?)\s*(?:cm|m|ft|in)?',
            ],
        }
        
        # Frequency patterns
        self.frequency_patterns = [
            (r'\b(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\b', 'dose_pattern'),  # 1-0-1
            (r'\b(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\b', 'dose_pattern_4'),  # 1-0-1-1
            (r'\b(?:once|1x|one\s*time)\s*(?:daily|a\s*day|per\s*day)\b', 'once_daily'),
            (r'\b(?:twice|2x|two\s*times?|bd|bid)\s*(?:daily|a\s*day|per\s*day)?\b', 'twice_daily'),
            (r'\b(?:thrice|3x|three\s*times?|tds|tid)\s*(?:daily|a\s*day|per\s*day)?\b', 'thrice_daily'),
            (r'\b(?:four\s*times?|4x|qds|qid)\s*(?:daily|a\s*day|per\s*day)?\b', 'four_daily'),
            (r'\bevery\s*(\d+)\s*hours?\b', 'every_n_hours'),
            (r'\bq(\d+)h\b', 'q_hours'),  # q4h, q6h, etc.
            (r'\b(?:at\s*)?(?:bed\s*time|night|hs|h\.s\.)\b', 'at_bedtime'),
            (r'\b(?:in\s*the\s*)?morning\b', 'morning'),
            (r'\b(?:prn|as\s*needed|sos|when\s*needed)\b', 'as_needed'),
            (r'\b(?:before|ac|a\.c\.)\s*(?:meals?|food)\b', 'before_meals'),
            (r'\b(?:after|pc|p\.c\.)\s*(?:meals?|food)\b', 'after_meals'),
            (r'\b(?:with|during)\s*(?:meals?|food)\b', 'with_meals'),
        ]
        
        # Duration patterns
        self.duration_patterns = [
            (r'\b(?:for\s*)?(\d+)\s*(?:days?|d)\b', 'days'),
            (r'\b(?:for\s*)?(\d+)\s*(?:weeks?|wk)\b', 'weeks'),
            (r'\b(?:for\s*)?(\d+)\s*(?:months?|mon)\b', 'months'),
            (r'\bx\s*(\d+)\s*(?:days?|d)\b', 'days'),
            (r'\b(\d+)\s*(?:/|per)\s*(?:7|week)\b', 'per_week'),
            (r'\bcontinue\s*(?:for\s*)?(\d+)', 'continue_days'),
            (r'\b(?:till|until)\s*(\w+\s*\d+)', 'until_date'),
        ]
        
        # Dosage patterns
        self.dosage_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|l|iu|units?|meq|%)', 'standard'),
            (r'(\d+(?:\.\d+)?)\s*(tablet|tab|capsule|cap|pill)s?', 'count'),
            (r'(\d+(?:\.\d+)?)\s*(drop|gtt|puff|spray|scoop)s?', 'count'),
            (r'(\d+)\s*/\s*(\d+)\s*(mg|ml)', 'concentration'),  # 250/5 ml
        ]
    
    def _load_medication_database(self) -> Dict[str, Dict]:
        """Load medication database with brand/generic mappings"""
        # Comprehensive medication database
        medications = {
            # Generic: {brand_names, drug_class}
            'paracetamol': {
                'brand_names': ['tylenol', 'crocin', 'dolo', 'calpol', 'panadol', 'acetaminophen'],
                'drug_class': 'analgesic',
                'common_dosages': ['325mg', '500mg', '650mg', '1000mg']
            },
            'ibuprofen': {
                'brand_names': ['advil', 'motrin', 'brufen', 'nurofen'],
                'drug_class': 'nsaid',
                'common_dosages': ['200mg', '400mg', '600mg', '800mg']
            },
            'amoxicillin': {
                'brand_names': ['amoxil', 'mox', 'novamox', 'trimox'],
                'drug_class': 'antibiotic',
                'common_dosages': ['250mg', '500mg', '875mg']
            },
            'azithromycin': {
                'brand_names': ['zithromax', 'azithral', 'zmax', 'azee'],
                'drug_class': 'antibiotic',
                'common_dosages': ['250mg', '500mg']
            },
            'ciprofloxacin': {
                'brand_names': ['cipro', 'ciplox', 'cifran'],
                'drug_class': 'antibiotic',
                'common_dosages': ['250mg', '500mg', '750mg']
            },
            'metformin': {
                'brand_names': ['glucophage', 'glycomet', 'glumet', 'fortamet'],
                'drug_class': 'antidiabetic',
                'common_dosages': ['500mg', '850mg', '1000mg']
            },
            'atorvastatin': {
                'brand_names': ['lipitor', 'atorva', 'storvas'],
                'drug_class': 'statin',
                'common_dosages': ['10mg', '20mg', '40mg', '80mg']
            },
            'amlodipine': {
                'brand_names': ['norvasc', 'amlong', 'amlip'],
                'drug_class': 'calcium_channel_blocker',
                'common_dosages': ['2.5mg', '5mg', '10mg']
            },
            'losartan': {
                'brand_names': ['cozaar', 'losacar', 'losar'],
                'drug_class': 'arb',
                'common_dosages': ['25mg', '50mg', '100mg']
            },
            'omeprazole': {
                'brand_names': ['prilosec', 'omez', 'losec'],
                'drug_class': 'ppi',
                'common_dosages': ['20mg', '40mg']
            },
            'pantoprazole': {
                'brand_names': ['protonix', 'pan', 'pantop'],
                'drug_class': 'ppi',
                'common_dosages': ['20mg', '40mg']
            },
            'metoprolol': {
                'brand_names': ['lopressor', 'toprol', 'betaloc'],
                'drug_class': 'beta_blocker',
                'common_dosages': ['25mg', '50mg', '100mg']
            },
            'lisinopril': {
                'brand_names': ['zestril', 'prinivil', 'listril'],
                'drug_class': 'ace_inhibitor',
                'common_dosages': ['5mg', '10mg', '20mg', '40mg']
            },
            'hydrochlorothiazide': {
                'brand_names': ['microzide', 'aquazide', 'hctz'],
                'drug_class': 'diuretic',
                'common_dosages': ['12.5mg', '25mg', '50mg']
            },
            'furosemide': {
                'brand_names': ['lasix', 'frusenex'],
                'drug_class': 'diuretic',
                'common_dosages': ['20mg', '40mg', '80mg']
            },
            'prednisone': {
                'brand_names': ['deltasone', 'predone', 'rayos'],
                'drug_class': 'corticosteroid',
                'common_dosages': ['5mg', '10mg', '20mg']
            },
            'cetirizine': {
                'brand_names': ['zyrtec', 'cetzine', 'incid'],
                'drug_class': 'antihistamine',
                'common_dosages': ['5mg', '10mg']
            },
            'loratadine': {
                'brand_names': ['claritin', 'lorfast', 'alavert'],
                'drug_class': 'antihistamine',
                'common_dosages': ['10mg']
            },
            'montelukast': {
                'brand_names': ['singulair', 'montair', 'montek'],
                'drug_class': 'leukotriene_inhibitor',
                'common_dosages': ['4mg', '5mg', '10mg']
            },
            'salbutamol': {
                'brand_names': ['ventolin', 'asthalin', 'proventil', 'albuterol'],
                'drug_class': 'bronchodilator',
                'common_dosages': ['2mg', '4mg', '100mcg']
            },
            'levothyroxine': {
                'brand_names': ['synthroid', 'thyronorm', 'eltroxin', 'levoxyl'],
                'drug_class': 'thyroid_hormone',
                'common_dosages': ['25mcg', '50mcg', '75mcg', '100mcg', '125mcg']
            },
            'sertraline': {
                'brand_names': ['zoloft', 'serlift', 'lustral'],
                'drug_class': 'ssri',
                'common_dosages': ['25mg', '50mg', '100mg']
            },
            'escitalopram': {
                'brand_names': ['lexapro', 'cipralex', 'nexito'],
                'drug_class': 'ssri',
                'common_dosages': ['5mg', '10mg', '20mg']
            },
            'alprazolam': {
                'brand_names': ['xanax', 'alprax', 'restyl'],
                'drug_class': 'benzodiazepine',
                'common_dosages': ['0.25mg', '0.5mg', '1mg', '2mg']
            },
            'clonazepam': {
                'brand_names': ['klonopin', 'rivotril', 'clonotril'],
                'drug_class': 'benzodiazepine',
                'common_dosages': ['0.25mg', '0.5mg', '1mg', '2mg']
            },
            'gabapentin': {
                'brand_names': ['neurontin', 'gabantin', 'gralise'],
                'drug_class': 'anticonvulsant',
                'common_dosages': ['100mg', '300mg', '400mg', '600mg']
            },
            'tramadol': {
                'brand_names': ['ultram', 'tramazac', 'contramal'],
                'drug_class': 'opioid_analgesic',
                'common_dosages': ['50mg', '100mg']
            },
            'diclofenac': {
                'brand_names': ['voltaren', 'voveran', 'cataflam'],
                'drug_class': 'nsaid',
                'common_dosages': ['25mg', '50mg', '75mg', '100mg']
            },
            'ranitidine': {
                'brand_names': ['zantac', 'aciloc', 'rantac'],
                'drug_class': 'h2_blocker',
                'common_dosages': ['150mg', '300mg']
            },
            'domperidone': {
                'brand_names': ['motilium', 'domstal', 'vomistop'],
                'drug_class': 'antiemetic',
                'common_dosages': ['10mg']
            },
            'ondansetron': {
                'brand_names': ['zofran', 'ondem', 'emeset'],
                'drug_class': 'antiemetic',
                'common_dosages': ['4mg', '8mg']
            },
            'insulin': {
                'brand_names': ['humulin', 'novolin', 'lantus', 'humalog', 'novolog'],
                'drug_class': 'insulin',
                'common_dosages': ['units']
            },
            'aspirin': {
                'brand_names': ['ecosprin', 'disprin', 'bayer'],
                'drug_class': 'antiplatelet',
                'common_dosages': ['75mg', '81mg', '150mg', '325mg']
            },
            'clopidogrel': {
                'brand_names': ['plavix', 'clopilet', 'deplatt'],
                'drug_class': 'antiplatelet',
                'common_dosages': ['75mg']
            },
            'warfarin': {
                'brand_names': ['coumadin', 'warf', 'acitrom'],
                'drug_class': 'anticoagulant',
                'common_dosages': ['1mg', '2mg', '2.5mg', '5mg']
            },
        }
        
        # Create reverse lookup for brand names
        self.brand_to_generic = {}
        for generic, data in medications.items():
            for brand in data.get('brand_names', []):
                self.brand_to_generic[brand.lower()] = generic
            self.brand_to_generic[generic.lower()] = generic
        
        return medications
    
    def _load_symptom_database(self) -> set:
        """Load common symptoms"""
        return {
            'fever', 'cough', 'cold', 'headache', 'migraine', 'pain', 'ache',
            'nausea', 'vomiting', 'diarrhea', 'constipation', 'fatigue', 'weakness',
            'dizziness', 'vertigo', 'breathlessness', 'shortness of breath', 'dyspnea',
            'chest pain', 'palpitations', 'swelling', 'edema', 'rash', 'itching',
            'burning', 'numbness', 'tingling', 'insomnia', 'anxiety', 'depression',
            'loss of appetite', 'weight loss', 'weight gain', 'joint pain', 'muscle pain',
            'back pain', 'abdominal pain', 'stomach pain', 'bloating', 'acidity',
            'heartburn', 'sore throat', 'runny nose', 'nasal congestion', 'sneezing',
            'wheezing', 'chills', 'sweating', 'night sweats', 'blurred vision',
            'eye pain', 'ear pain', 'hearing loss', 'tinnitus', 'difficulty swallowing',
            'frequent urination', 'painful urination', 'blood in urine', 'blood in stool',
        }
    
    def _load_diagnosis_database(self) -> set:
        """Load common diagnoses"""
        return {
            'hypertension', 'diabetes mellitus', 'type 2 diabetes', 'type 1 diabetes',
            'hyperlipidemia', 'dyslipidemia', 'coronary artery disease', 'heart failure',
            'atrial fibrillation', 'stroke', 'tia', 'asthma', 'copd', 'bronchitis',
            'pneumonia', 'tuberculosis', 'anemia', 'thyroid disorder', 'hypothyroidism',
            'hyperthyroidism', 'gerd', 'gastritis', 'peptic ulcer', 'ibs', 'crohn disease',
            'ulcerative colitis', 'hepatitis', 'cirrhosis', 'fatty liver', 'kidney disease',
            'chronic kidney disease', 'uti', 'urinary tract infection', 'arthritis',
            'osteoarthritis', 'rheumatoid arthritis', 'gout', 'osteoporosis', 'fracture',
            'depression', 'anxiety disorder', 'bipolar disorder', 'schizophrenia',
            'dementia', 'alzheimer', 'parkinson', 'epilepsy', 'migraine', 'neuropathy',
            'cancer', 'malignancy', 'tumor', 'infection', 'sepsis', 'cellulitis',
            'abscess', 'dermatitis', 'eczema', 'psoriasis', 'allergy', 'allergic rhinitis',
        }
    
    def extract_entities(self, text: str, document_id: Optional[int] = None) -> ExtractionResult:
        """
        Extract all medical entities from text
        
        Args:
            text: Input text (cleaned OCR output)
            document_id: Optional source document ID
            
        Returns:
            ExtractionResult with all extracted entities
        """
        entities = []
        
        # Extract medications
        medications = self._extract_medications(text)
        entities.extend([
            ExtractedEntity(
                entity_type='medication',
                text=m['text'],
                normalized_text=m.get('generic_name'),
                confidence=m.get('confidence', 0.8),
                start_pos=m.get('start', 0),
                end_pos=m.get('end', 0),
                attributes=m,
                source_text=text
            )
            for m in medications
        ])
        
        # Extract dosages
        dosages = self._extract_dosages(text)
        entities.extend([
            ExtractedEntity(
                entity_type='dosage',
                text=d['text'],
                normalized_text=d.get('normalized'),
                confidence=d.get('confidence', 0.9),
                start_pos=d.get('start', 0),
                end_pos=d.get('end', 0),
                attributes=d,
                source_text=text
            )
            for d in dosages
        ])
        
        # Extract frequencies
        frequencies = self._extract_frequencies(text)
        entities.extend([
            ExtractedEntity(
                entity_type='frequency',
                text=f['text'],
                normalized_text=f.get('normalized'),
                confidence=f.get('confidence', 0.85),
                start_pos=f.get('start', 0),
                end_pos=f.get('end', 0),
                attributes=f,
                source_text=text
            )
            for f in frequencies
        ])
        
        # Extract routes
        routes = self._extract_routes(text)
        
        # Extract durations
        durations = self._extract_durations(text)
        
        # Extract symptoms
        symptoms = self._extract_symptoms(text)
        entities.extend([
            ExtractedEntity(
                entity_type='symptom',
                text=s['text'],
                normalized_text=s.get('normalized'),
                confidence=s.get('confidence', 0.75),
                start_pos=s.get('start', 0),
                end_pos=s.get('end', 0),
                attributes=s,
                source_text=text
            )
            for s in symptoms
        ])
        
        # Extract diagnoses
        diagnoses = self._extract_diagnoses(text)
        entities.extend([
            ExtractedEntity(
                entity_type='diagnosis',
                text=d['text'],
                normalized_text=d.get('normalized'),
                confidence=d.get('confidence', 0.7),
                start_pos=d.get('start', 0),
                end_pos=d.get('end', 0),
                attributes=d,
                source_text=text
            )
            for d in diagnoses
        ])
        
        # Extract vitals
        vitals = self._extract_vitals(text)
        entities.extend([
            ExtractedEntity(
                entity_type='vital',
                text=v['text'],
                normalized_text=v.get('normalized'),
                confidence=v.get('confidence', 0.9),
                start_pos=v.get('start', 0),
                end_pos=v.get('end', 0),
                attributes=v,
                source_text=text
            )
            for v in vitals
        ])
        
        # Extract dates
        dates = self._extract_dates(text)
        
        # Calculate overall confidence
        if entities:
            overall_confidence = sum(e.confidence for e in entities) / len(entities)
        else:
            overall_confidence = 0.5
        
        return ExtractionResult(
            entities=entities,
            medications=medications,
            dosages=dosages,
            frequencies=frequencies,
            routes=routes,
            durations=durations,
            symptoms=symptoms,
            diagnoses=diagnoses,
            vitals=vitals,
            dates=dates,
            confidence=overall_confidence
        )
    
    def _extract_medications(self, text: str) -> List[Dict]:
        """Extract medication names from text"""
        medications = []
        text_lower = text.lower()
        
        # Search for known medications
        for generic, data in self.medications.items():
            # Check generic name
            pattern = r'\b' + re.escape(generic) + r'\b'
            for match in re.finditer(pattern, text_lower):
                medications.append({
                    'text': text[match.start():match.end()],
                    'generic_name': generic,
                    'brand_name': None,
                    'drug_class': data.get('drug_class'),
                    'confidence': 0.95,
                    'start': match.start(),
                    'end': match.end()
                })
            
            # Check brand names
            for brand in data.get('brand_names', []):
                pattern = r'\b' + re.escape(brand) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    medications.append({
                        'text': text[match.start():match.end()],
                        'generic_name': generic,
                        'brand_name': brand,
                        'drug_class': data.get('drug_class'),
                        'confidence': 0.9,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Remove duplicates (keep highest confidence)
        seen_positions = set()
        unique_medications = []
        for med in sorted(medications, key=lambda x: x['confidence'], reverse=True):
            pos_key = (med['start'], med['end'])
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_medications.append(med)
        
        return unique_medications
    
    def _extract_dosages(self, text: str) -> List[Dict]:
        """Extract dosage information from text"""
        dosages = []
        
        for pattern, pattern_type in self.dosage_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if pattern_type == 'standard':
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    dosages.append({
                        'text': match.group(0),
                        'value': value,
                        'unit': unit,
                        'normalized': f"{value}{unit}",
                        'confidence': 0.95,
                        'start': match.start(),
                        'end': match.end()
                    })
                elif pattern_type == 'count':
                    count = int(match.group(1))
                    form = match.group(2).lower()
                    dosages.append({
                        'text': match.group(0),
                        'count': count,
                        'form': form,
                        'normalized': f"{count} {form}",
                        'confidence': 0.9,
                        'start': match.start(),
                        'end': match.end()
                    })
                elif pattern_type == 'concentration':
                    amount = int(match.group(1))
                    per = int(match.group(2))
                    unit = match.group(3).lower()
                    dosages.append({
                        'text': match.group(0),
                        'concentration': f"{amount}/{per}",
                        'unit': unit,
                        'normalized': f"{amount}/{per}{unit}",
                        'confidence': 0.9,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return dosages
    
    def _extract_frequencies(self, text: str) -> List[Dict]:
        """Extract frequency patterns from text"""
        frequencies = []
        
        for pattern, freq_type in self.frequency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                freq_data = {
                    'text': match.group(0),
                    'type': freq_type,
                    'confidence': 0.9,
                    'start': match.start(),
                    'end': match.end()
                }
                
                if freq_type == 'dose_pattern':
                    morning = int(match.group(1))
                    afternoon = int(match.group(2))
                    night = int(match.group(3))
                    freq_data['parsed'] = {
                        'morning': morning,
                        'afternoon': afternoon,
                        'night': night
                    }
                    freq_data['times_per_day'] = morning + afternoon + night
                    freq_data['normalized'] = f"{morning}-{afternoon}-{night}"
                elif freq_type == 'dose_pattern_4':
                    m = int(match.group(1))
                    a = int(match.group(2))
                    e = int(match.group(3))
                    n = int(match.group(4))
                    freq_data['parsed'] = {
                        'morning': m, 'afternoon': a, 'evening': e, 'night': n
                    }
                    freq_data['times_per_day'] = m + a + e + n
                elif freq_type == 'every_n_hours':
                    hours = int(match.group(1))
                    freq_data['hours_interval'] = hours
                    freq_data['times_per_day'] = 24 // hours
                    freq_data['normalized'] = f"every {hours} hours"
                elif freq_type == 'q_hours':
                    hours = int(match.group(1))
                    freq_data['hours_interval'] = hours
                    freq_data['times_per_day'] = 24 // hours
                    freq_data['normalized'] = f"every {hours} hours"
                elif freq_type == 'once_daily':
                    freq_data['times_per_day'] = 1
                    freq_data['normalized'] = "once daily"
                elif freq_type == 'twice_daily':
                    freq_data['times_per_day'] = 2
                    freq_data['normalized'] = "twice daily"
                elif freq_type == 'thrice_daily':
                    freq_data['times_per_day'] = 3
                    freq_data['normalized'] = "three times daily"
                elif freq_type == 'four_daily':
                    freq_data['times_per_day'] = 4
                    freq_data['normalized'] = "four times daily"
                else:
                    freq_data['normalized'] = freq_type.replace('_', ' ')
                
                frequencies.append(freq_data)
        
        return frequencies
    
    def _extract_routes(self, text: str) -> List[Dict]:
        """Extract administration routes"""
        routes = []
        text_lower = text.lower()
        
        for route_name, patterns in self.route_patterns.items():
            for pattern in patterns:
                regex = r'\b' + re.escape(pattern) + r'\b'
                for match in re.finditer(regex, text_lower):
                    routes.append({
                        'text': text[match.start():match.end()],
                        'route': route_name,
                        'confidence': 0.9,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return routes
    
    def _extract_durations(self, text: str) -> List[Dict]:
        """Extract treatment durations"""
        durations = []
        
        for pattern, dur_type in self.duration_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dur_data = {
                    'text': match.group(0),
                    'type': dur_type,
                    'confidence': 0.85,
                    'start': match.start(),
                    'end': match.end()
                }
                
                if dur_type == 'days':
                    days = int(match.group(1))
                    dur_data['days'] = days
                    dur_data['normalized'] = f"{days} days"
                elif dur_type == 'weeks':
                    weeks = int(match.group(1))
                    dur_data['days'] = weeks * 7
                    dur_data['weeks'] = weeks
                    dur_data['normalized'] = f"{weeks} weeks"
                elif dur_type == 'months':
                    months = int(match.group(1))
                    dur_data['days'] = months * 30
                    dur_data['months'] = months
                    dur_data['normalized'] = f"{months} months"
                
                durations.append(dur_data)
        
        return durations
    
    def _extract_symptoms(self, text: str) -> List[Dict]:
        """Extract symptoms from text"""
        symptoms = []
        text_lower = text.lower()
        
        for symptom in self.symptoms_db:
            pattern = r'\b' + re.escape(symptom) + r'\b'
            for match in re.finditer(pattern, text_lower):
                symptoms.append({
                    'text': text[match.start():match.end()],
                    'normalized': symptom,
                    'confidence': 0.8,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return symptoms
    
    def _extract_diagnoses(self, text: str) -> List[Dict]:
        """Extract diagnoses from text"""
        diagnoses = []
        text_lower = text.lower()
        
        for diagnosis in self.diagnoses_db:
            pattern = r'\b' + re.escape(diagnosis) + r'\b'
            for match in re.finditer(pattern, text_lower):
                diagnoses.append({
                    'text': text[match.start():match.end()],
                    'normalized': diagnosis,
                    'confidence': 0.75,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return diagnoses
    
    def _extract_vitals(self, text: str) -> List[Dict]:
        """Extract vital signs from text"""
        vitals = []
        
        for vital_type, patterns in self.vital_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    vital_data = {
                        'text': match.group(0),
                        'vital_type': vital_type,
                        'confidence': 0.9,
                        'start': match.start(),
                        'end': match.end()
                    }
                    
                    if vital_type == 'blood_pressure':
                        vital_data['systolic'] = int(match.group(1))
                        vital_data['diastolic'] = int(match.group(2))
                        vital_data['value'] = f"{match.group(1)}/{match.group(2)}"
                        vital_data['unit'] = 'mmHg'
                        # Interpretation
                        sys = int(match.group(1))
                        if sys < 90:
                            vital_data['interpretation'] = 'low'
                        elif sys < 120:
                            vital_data['interpretation'] = 'normal'
                        elif sys < 140:
                            vital_data['interpretation'] = 'elevated'
                        else:
                            vital_data['interpretation'] = 'high'
                    else:
                        vital_data['value'] = match.group(1)
                        vital_data['numeric_value'] = float(match.group(1))
                    
                    vitals.append(vital_data)
        
        return vitals
    
    def _extract_dates(self, text: str) -> List[Dict]:
        """Extract dates from text"""
        dates = []
        
        # Common date patterns
        date_patterns = [
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',  # DD/MM/YYYY or MM/DD/YYYY
            r'\b(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{2,4})?\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{1,2}),?\s*(\d{2,4})?\b',
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    parsed_date = date_parser.parse(match.group(0), fuzzy=True)
                    dates.append({
                        'text': match.group(0),
                        'parsed': parsed_date.isoformat(),
                        'date': str(parsed_date.date()),
                        'confidence': 0.85,
                        'start': match.start(),
                        'end': match.end()
                    })
                except:
                    dates.append({
                        'text': match.group(0),
                        'parsed': None,
                        'confidence': 0.5,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return dates
