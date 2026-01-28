"""
Text Cleaning Service - OCR post-processing and normalization
Handles: Spelling correction, abbreviation expansion, medical text normalization
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """Result of text cleaning operation"""
    original_text: str
    cleaned_text: str
    corrections: List[Dict]  # List of corrections made
    unreadable_segments: List[Dict]  # Flagged unreadable parts
    confidence: float


class TextCleaningService:
    """
    Text cleaning and normalization service for medical OCR text
    
    Features:
    - OCR error correction
    - Medical abbreviation normalization
    - Preserves original alongside corrected
    - Flags unreadable segments
    """
    
    def __init__(self):
        # Common OCR error patterns
        self.ocr_corrections = {
            # Number/letter confusion
            '0': {'O', 'o'},
            '1': {'l', 'I', 'i', '|'},
            '5': {'S', 's'},
            '8': {'B'},
            # Common OCR mistakes
            'rn': {'m'},
            'vv': {'w'},
            'cl': {'d'},
            'li': {'h'},
            # Medical specific
            'rnl': {'ml'},
            'rng': {'mg'},
            'iu': {'IU'},
        }
        
        # Medical abbreviations dictionary
        self.medical_abbreviations = {
            # Frequency
            'od': 'once daily',
            'o.d.': 'once daily',
            'bd': 'twice daily',
            'b.d.': 'twice daily',
            'bid': 'twice daily',
            'b.i.d.': 'twice daily',
            'tid': 'three times daily',
            't.i.d.': 'three times daily',
            'tds': 'three times daily',
            't.d.s.': 'three times daily',
            'qid': 'four times daily',
            'q.i.d.': 'four times daily',
            'qds': 'four times daily',
            'q.d.s.': 'four times daily',
            'prn': 'as needed',
            'p.r.n.': 'as needed',
            'sos': 'if needed',
            's.o.s.': 'if needed',
            'stat': 'immediately',
            'hs': 'at bedtime',
            'h.s.': 'at bedtime',
            'ac': 'before meals',
            'a.c.': 'before meals',
            'pc': 'after meals',
            'p.c.': 'after meals',
            'qh': 'every hour',
            'q2h': 'every 2 hours',
            'q4h': 'every 4 hours',
            'q6h': 'every 6 hours',
            'q8h': 'every 8 hours',
            'q12h': 'every 12 hours',
            
            # Route
            'po': 'by mouth',
            'p.o.': 'by mouth',
            'iv': 'intravenous',
            'i.v.': 'intravenous',
            'im': 'intramuscular',
            'i.m.': 'intramuscular',
            'sc': 'subcutaneous',
            's.c.': 'subcutaneous',
            'sl': 'sublingual',
            's.l.': 'sublingual',
            'pr': 'per rectum',
            'p.r.': 'per rectum',
            'inh': 'inhaled',
            'top': 'topical',
            'od (eye)': 'right eye',
            'os': 'left eye',
            'ou': 'both eyes',
            
            # Units
            'mg': 'milligram',
            'mcg': 'microgram',
            'g': 'gram',
            'kg': 'kilogram',
            'ml': 'milliliter',
            'l': 'liter',
            'cc': 'cubic centimeter',
            'iu': 'international units',
            'u': 'units',
            'meq': 'milliequivalent',
            
            # Forms
            'tab': 'tablet',
            'tabs': 'tablets',
            'cap': 'capsule',
            'caps': 'capsules',
            'inj': 'injection',
            'syr': 'syrup',
            'susp': 'suspension',
            'oint': 'ointment',
            'cr': 'cream',
            'sol': 'solution',
            'drops': 'drops',
            'gtt': 'drops',
            'supp': 'suppository',
            'puff': 'puff',
            'spray': 'spray',
            
            # Common medical
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'wt': 'weight',
            'ht': 'height',
            'bmi': 'body mass index',
            'rx': 'prescription',
            'dx': 'diagnosis',
            'hx': 'history',
            'tx': 'treatment',
            'sx': 'symptoms',
            'fx': 'fracture',
            'nkda': 'no known drug allergies',
            'nka': 'no known allergies',
            'c/o': 'complaining of',
            'h/o': 'history of',
            's/p': 'status post',
            'w/': 'with',
            'w/o': 'without',
            'y/o': 'years old',
        }
        
        # Common medical terms for spell checking
        self.medical_terms = self._load_medical_terms()
        
        # Patterns for unreadable text
        self.unreadable_patterns = [
            r'[^\w\s]{3,}',  # Multiple special characters
            r'\b[a-z]{1}[A-Z]{2,}[a-z]+\b',  # Weird capitalization
            r'\b\d+[a-zA-Z]+\d+[a-zA-Z]+\b',  # Mixed numbers and letters oddly
        ]
    
    def _load_medical_terms(self) -> Set[str]:
        """Load common medical terms for spell checking"""
        # Common drug names and medical terms
        terms = {
            # Common medications
            'paracetamol', 'acetaminophen', 'ibuprofen', 'aspirin', 'amoxicillin',
            'metformin', 'lisinopril', 'atorvastatin', 'omeprazole', 'amlodipine',
            'metoprolol', 'losartan', 'gabapentin', 'hydrochlorothiazide', 'sertraline',
            'simvastatin', 'montelukast', 'escitalopram', 'pantoprazole', 'bupropion',
            'furosemide', 'alprazolam', 'prednisone', 'tramadol', 'tamsulosin',
            'clopidogrel', 'carvedilol', 'trazodone', 'pravastatin', 'fluticasone',
            'cetirizine', 'loratadine', 'ranitidine', 'famotidine', 'azithromycin',
            'ciprofloxacin', 'doxycycline', 'clindamycin', 'fluconazole', 'valacyclovir',
            'insulin', 'glipizide', 'glyburide', 'pioglitazone', 'sitagliptin',
            'warfarin', 'rivaroxaban', 'apixaban', 'enoxaparin', 'heparin',
            'oxycodone', 'hydrocodone', 'morphine', 'fentanyl', 'codeine',
            'diazepam', 'lorazepam', 'clonazepam', 'zolpidem', 'eszopiclone',
            'duloxetine', 'venlafaxine', 'fluoxetine', 'paroxetine', 'citalopram',
            'quetiapine', 'risperidone', 'olanzapine', 'aripiprazole', 'haloperidol',
            'levothyroxine', 'liothyronine', 'methimazole', 'propylthiouracil',
            'albuterol', 'salmeterol', 'tiotropium', 'budesonide', 'ipratropium',
            'cyclobenzaprine', 'methocarbamol', 'baclofen', 'tizanidine',
            
            # Common medical terms
            'hypertension', 'diabetes', 'mellitus', 'hyperlipidemia', 'arthritis',
            'asthma', 'copd', 'pneumonia', 'bronchitis', 'sinusitis',
            'gastritis', 'colitis', 'hepatitis', 'pancreatitis', 'appendicitis',
            'migraine', 'headache', 'vertigo', 'nausea', 'vomiting',
            'diarrhea', 'constipation', 'dyspepsia', 'reflux', 'heartburn',
            'infection', 'inflammation', 'fever', 'fatigue', 'weakness',
            'edema', 'swelling', 'pain', 'ache', 'tenderness',
            'prescription', 'medication', 'dosage', 'frequency', 'duration',
            'diagnosis', 'prognosis', 'treatment', 'therapy', 'prophylaxis',
            'systolic', 'diastolic', 'cholesterol', 'triglycerides', 'glucose',
            'hemoglobin', 'creatinine', 'bilirubin', 'albumin', 'potassium',
        }
        return terms
    
    def clean_text(self, text: str, preserve_original: bool = True) -> CleaningResult:
        """
        Clean and normalize OCR text
        
        Args:
            text: Raw OCR text
            preserve_original: Keep original text in result
            
        Returns:
            CleaningResult with cleaned text and corrections
        """
        if not text:
            return CleaningResult(
                original_text="",
                cleaned_text="",
                corrections=[],
                unreadable_segments=[],
                confidence=1.0
            )
        
        original = text
        corrections = []
        
        # Step 1: Basic cleanup
        cleaned = self._basic_cleanup(text)
        
        # Step 2: Fix common OCR errors
        cleaned, ocr_corrections = self._fix_ocr_errors(cleaned)
        corrections.extend(ocr_corrections)
        
        # Step 3: Fix medical spelling errors
        cleaned, spelling_corrections = self._fix_medical_spelling(cleaned)
        corrections.extend(spelling_corrections)
        
        # Step 4: Normalize medical abbreviations (but keep both forms)
        cleaned, abbrev_info = self._annotate_abbreviations(cleaned)
        
        # Step 5: Detect unreadable segments
        unreadable = self._detect_unreadable_segments(cleaned)
        
        # Calculate confidence based on corrections needed
        correction_ratio = len(corrections) / max(len(text.split()), 1)
        confidence = max(0.5, 1.0 - correction_ratio * 0.5)
        
        return CleaningResult(
            original_text=original if preserve_original else "",
            cleaned_text=cleaned,
            corrections=corrections,
            unreadable_segments=unreadable,
            confidence=confidence
        )
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleanup"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation
        
        # Fix quotation marks
        text = text.replace('``', '"').replace("''", '"')
        
        # Normalize dashes
        text = re.sub(r'[-–—]{2,}', '—', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _fix_ocr_errors(self, text: str) -> Tuple[str, List[Dict]]:
        """Fix common OCR errors"""
        corrections = []
        
        # Fix common number/unit patterns
        patterns = [
            # mg, ml corrections
            (r'\b(\d+)\s*(rng|rnl)\b', r'\1 mg', 'OCR error: rng/rnl → mg'),
            (r'\b(\d+)\s*rnl\b', r'\1 ml', 'OCR error: rnl → ml'),
            
            # Common word corrections
            (r'\brnorning\b', 'morning', 'OCR error: rnorning → morning'),
            (r'\baftemoon\b', 'afternoon', 'OCR error: aftemoon → afternoon'),
            (r'\btabIet\b', 'tablet', 'OCR error: tabIet → tablet'),
            (r'\bcapsuIe\b', 'capsule', 'OCR error: capsuIe → capsule'),
            
            # Number confusion in context
            (r'\b([0O])nce\b', 'once', 'OCR error: 0nce → once'),
            (r'\bda1ly\b', 'daily', 'OCR error: da1ly → daily'),
            (r'\btwlce\b', 'twice', 'OCR error: twlce → twice'),
        ]
        
        for pattern, replacement, reason in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                old_text = text
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if old_text != text:
                    corrections.append({
                        'type': 'ocr_error',
                        'original': str(matches),
                        'corrected': replacement,
                        'reason': reason
                    })
        
        return text, corrections
    
    def _fix_medical_spelling(self, text: str) -> Tuple[str, List[Dict]]:
        """Fix medical term spelling errors"""
        corrections = []
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip short words and numbers
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) < 4 or clean_word.isdigit():
                corrected_words.append(word)
                continue
            
            # Check if word might be a misspelled medical term
            if clean_word not in self.medical_terms:
                best_match = self._find_best_match(clean_word, self.medical_terms)
                if best_match:
                    # Preserve original casing pattern
                    if word[0].isupper():
                        corrected = best_match.capitalize()
                    else:
                        corrected = best_match
                    
                    # Preserve trailing punctuation
                    trailing = ''
                    for i in range(len(word) - 1, -1, -1):
                        if not word[i].isalnum():
                            trailing = word[i] + trailing
                        else:
                            break
                    
                    corrected_words.append(corrected + trailing)
                    corrections.append({
                        'type': 'spelling',
                        'original': word,
                        'corrected': corrected + trailing,
                        'reason': f'Possible spelling error: {word} → {corrected}'
                    })
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words), corrections
    
    def _find_best_match(self, word: str, terms: Set[str], threshold: float = 0.85) -> Optional[str]:
        """Find best matching medical term"""
        best_match = None
        best_ratio = 0
        
        for term in terms:
            # Only compare similar length words
            if abs(len(term) - len(word)) > 2:
                continue
            
            ratio = SequenceMatcher(None, word.lower(), term.lower()).ratio()
            if ratio > threshold and ratio > best_ratio:
                best_ratio = ratio
                best_match = term
        
        return best_match
    
    def _annotate_abbreviations(self, text: str) -> Tuple[str, List[Dict]]:
        """Add annotations for medical abbreviations"""
        annotations = []
        
        for abbrev, expansion in self.medical_abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                annotations.append({
                    'abbreviation': abbrev,
                    'expansion': expansion
                })
        
        return text, annotations
    
    def _detect_unreadable_segments(self, text: str) -> List[Dict]:
        """Detect potentially unreadable segments"""
        unreadable = []
        
        for pattern in self.unreadable_patterns:
            for match in re.finditer(pattern, text):
                unreadable.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'reason': 'Pattern suggests OCR failure'
                })
        
        # Also flag very short isolated characters
        for match in re.finditer(r'\s([a-zA-Z])\s', text):
            # Exclude valid single letters
            if match.group(1).lower() not in ['a', 'i']:
                unreadable.append({
                    'text': match.group(1),
                    'start': match.start() + 1,
                    'end': match.end() - 1,
                    'reason': 'Isolated character'
                })
        
        return unreadable
    
    def expand_abbreviation(self, abbrev: str) -> Optional[str]:
        """Get expansion for a medical abbreviation"""
        return self.medical_abbreviations.get(abbrev.lower())
    
    def normalize_dosage(self, dosage_text: str) -> Dict:
        """Normalize dosage text to standard format"""
        # Extract numeric value and unit
        match = re.match(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', dosage_text.strip())
        
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            # Normalize unit
            unit_map = {
                'mg': 'mg', 'milligram': 'mg', 'milligrams': 'mg',
                'mcg': 'mcg', 'microgram': 'mcg', 'micrograms': 'mcg', 'ug': 'mcg',
                'g': 'g', 'gram': 'g', 'grams': 'g', 'gm': 'g',
                'ml': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
                'l': 'L', 'liter': 'L', 'liters': 'L',
                'iu': 'IU', 'units': 'units', 'u': 'units',
            }
            
            normalized_unit = unit_map.get(unit, unit)
            
            return {
                'original': dosage_text,
                'value': value,
                'unit': normalized_unit,
                'normalized': f"{value} {normalized_unit}"
            }
        
        return {
            'original': dosage_text,
            'value': None,
            'unit': None,
            'normalized': dosage_text
        }
    
    def normalize_frequency(self, freq_text: str) -> Dict:
        """Normalize frequency text to standard format"""
        freq_lower = freq_text.lower().strip()
        
        # Pattern-based frequency parsing
        patterns = {
            # 1-0-1 format
            r'^(\d+)-(\d+)-(\d+)$': lambda m: {
                'morning': int(m.group(1)),
                'afternoon': int(m.group(2)),
                'night': int(m.group(3)),
                'times_per_day': int(m.group(1)) + int(m.group(2)) + int(m.group(3))
            },
            # 1-1-1 format
            r'^(\d+)-(\d+)-(\d+)-(\d+)$': lambda m: {
                'morning': int(m.group(1)),
                'afternoon': int(m.group(2)),
                'evening': int(m.group(3)),
                'night': int(m.group(4)),
                'times_per_day': int(m.group(1)) + int(m.group(2)) + int(m.group(3)) + int(m.group(4))
            },
            # Once daily
            r'(once|1|one)\s*(time[s]?)?\s*(daily|a\s*day|per\s*day)': lambda m: {
                'times_per_day': 1,
                'schedule': 'once daily'
            },
            # Twice daily
            r'(twice|2|two)\s*(times?)?\s*(daily|a\s*day|per\s*day)': lambda m: {
                'times_per_day': 2,
                'schedule': 'twice daily'
            },
            # Three times daily
            r'(thrice|3|three)\s*(times?)?\s*(daily|a\s*day|per\s*day)': lambda m: {
                'times_per_day': 3,
                'schedule': 'three times daily'
            },
            # Four times daily
            r'(4|four)\s*(times?)?\s*(daily|a\s*day|per\s*day)': lambda m: {
                'times_per_day': 4,
                'schedule': 'four times daily'
            },
            # Every N hours
            r'every\s*(\d+)\s*hour': lambda m: {
                'hours_interval': int(m.group(1)),
                'times_per_day': 24 // int(m.group(1))
            },
        }
        
        for pattern, parser in patterns.items():
            match = re.search(pattern, freq_lower)
            if match:
                result = parser(match)
                result['original'] = freq_text
                return result
        
        # Check abbreviations
        expansion = self.medical_abbreviations.get(freq_lower)
        if expansion:
            return {
                'original': freq_text,
                'abbreviation': freq_lower,
                'expansion': expansion,
                'parsed': self.normalize_frequency(expansion) if expansion != freq_lower else None
            }
        
        return {
            'original': freq_text,
            'parsed': None,
            'uncertain': True
        }
