"""
Handwriting Enhancement Service
Improves OCR accuracy for handwritten prescriptions through:
- Image preprocessing (contrast, sharpening, denoising)
- Multi-pass OCR with different settings
- Handwriting-specific error correction
- Medical context validation
"""
import io
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhancementResult:
    """Result of image enhancement"""
    original_bytes: bytes
    enhanced_bytes: bytes
    enhancement_applied: List[str]
    estimated_improvement: float


@dataclass 
class HandwritingOCRResult:
    """Enhanced OCR result for handwritten text"""
    text: str
    confidence: float
    corrections_made: List[Dict]
    enhancement_method: str
    word_confidences: Dict[str, float] = field(default_factory=dict)


class HandwritingEnhancer:
    """
    Enhances handwritten prescription images for better OCR accuracy
    
    Techniques:
    1. Adaptive contrast enhancement
    2. Noise reduction
    3. Sharpening for faded text
    4. Binarization for consistent ink
    5. Deskewing for tilted scans
    """
    
    def __init__(self):
        # Common handwriting OCR errors (what OCR reads -> what it likely should be)
        self.handwriting_corrections = {
            # Letters that look similar in handwriting
            'rn': 'm', 'nn': 'm', 'vv': 'w', 'cl': 'd', 'cL': 'd',
            'li': 'h', 'lI': 'h', 'ln': 'h', 'ii': 'u', 'll': 'u',
            'ri': 'n', 'ni': 'n', 'iv': 'w', 'iu': 'w',
            
            # Numbers vs letters
            '0': 'O', 'l': '1', 'I': '1', 'O': '0', 'S': '5',
            'B': '8', 'Z': '2', 'G': '6', 'q': '9', 'g': '9',
            
            # Common medication misspellings from handwriting
            'rnorning': 'morning', 'aftemoon': 'afternoon', 'evemng': 'evening',
            'tabIet': 'tablet', 'capsuIe': 'capsule', 'symp': 'syrup',
            'rnl': 'ml', 'rng': 'mg', 'mcq': 'mcg', 'lU': 'IU',
            'daIly': 'daily', 'daity': 'daily', 'daiIy': 'daily',
            'twlce': 'twice', 'twicc': 'twice', 'oncc': 'once',
            'tirncs': 'times', 'tirnes': 'times',
            'beforo': 'before', 'aftcr': 'after', 'meaIs': 'meals',
            
            # Common drug name OCR errors
            'Paracclamol': 'Paracetamol', 'Paracetarnol': 'Paracetamol',
            'Arnoxicillin': 'Amoxicillin', 'Amoxiclllin': 'Amoxicillin',
            'lbuprofen': 'Ibuprofen', 'Ibuprofcn': 'Ibuprofen',
            'Mctformin': 'Metformin', 'Metfonnin': 'Metformin',
            'Orneprazole': 'Omeprazole', 'Omeprazo1e': 'Omeprazole',
            'Azlthromycin': 'Azithromycin', 'Azithrornyc1n': 'Azithromycin',
            'Ciprofloxacln': 'Ciprofloxacin', 'Clprofloxacin': 'Ciprofloxacin',
            'Arnlodipine': 'Amlodipine', 'AmlodIpine': 'Amlodipine',
            'Atorvastatln': 'Atorvastatin', 'Atorvastat1n': 'Atorvastatin',
            'Losarlan': 'Losartan', 'Losattan': 'Losartan',
            'Te1misartan': 'Telmisartan', 'Telmisattan': 'Telmisartan',
            'Pantoprazo1e': 'Pantoprazole', 'Pantoprazolc': 'Pantoprazole',
            'Montelukast': 'Montelukast', 'Monte1ukast': 'Montelukast',
            'Cetirlzine': 'Cetirizine', 'Cctlrizine': 'Cetirizine',
            'Levocetirlzine': 'Levocetirizine',
            'Prednisonc': 'Prednisone', 'Predmsonc': 'Prednisone',
            'Doxycycllne': 'Doxycycline', 'Doxyeycline': 'Doxycycline',
            'Clindarnycin': 'Clindamycin', 'Cl1ndamycin': 'Clindamycin',
            'Gabapentln': 'Gabapentin', 'Gabapent1n': 'Gabapentin',
            'Pregabalin': 'Pregabalin', 'Pregaba1in': 'Pregabalin',
        }
        
        # Known drug names for validation (expanded list)
        self.known_drugs = self._build_drug_database()
        
        # Common dosage patterns
        self.dosage_patterns = [
            r'\d+\s*(?:mg|ml|mcg|g|iu|units?)',
            r'\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu|units?)',
        ]
        
        # Common frequency patterns
        self.frequency_patterns = [
            r'\d+-\d+-\d+',  # 1-0-1 format
            r'(?:once|twice|thrice)\s+(?:daily|a\s*day)',
            r'\d+\s*(?:times?)\s*(?:daily|a\s*day|per\s*day)',
            r'(?:bd|tid|qid|od|hs|prn|sos)',
        ]
    
    def _build_drug_database(self) -> set:
        """Build comprehensive drug name database"""
        drugs = {
            # Analgesics
            'paracetamol', 'acetaminophen', 'ibuprofen', 'aspirin', 'diclofenac',
            'naproxen', 'tramadol', 'codeine', 'morphine', 'fentanyl',
            'oxycodone', 'hydrocodone', 'ketorolac', 'piroxicam', 'meloxicam',
            'celecoxib', 'etoricoxib', 'aceclofenac', 'nimesulide', 'mefenamic',
            
            # Antibiotics
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline',
            'metronidazole', 'cephalexin', 'cefixime', 'ceftriaxone',
            'clindamycin', 'clarithromycin', 'erythromycin', 'levofloxacin',
            'ofloxacin', 'norfloxacin', 'amoxyclav', 'augmentin',
            'cefuroxime', 'cefpodoxime', 'cotrimoxazole', 'nitrofurantoin',
            
            # Cardiovascular
            'amlodipine', 'atorvastatin', 'rosuvastatin', 'metoprolol',
            'atenolol', 'lisinopril', 'losartan', 'telmisartan', 'olmesartan',
            'ramipril', 'enalapril', 'clopidogrel', 'aspirin', 'warfarin',
            'rivaroxaban', 'apixaban', 'furosemide', 'hydrochlorothiazide',
            'spironolactone', 'diltiazem', 'verapamil', 'digoxin',
            'carvedilol', 'bisoprolol', 'nebivolol', 'prazosin',
            
            # Diabetes
            'metformin', 'glimepiride', 'glipizide', 'glyburide', 'sitagliptin',
            'vildagliptin', 'linagliptin', 'empagliflozin', 'dapagliflozin',
            'pioglitazone', 'insulin', 'glargine', 'lispro', 'aspart',
            
            # GI
            'omeprazole', 'pantoprazole', 'rabeprazole', 'esomeprazole',
            'ranitidine', 'famotidine', 'domperidone', 'ondansetron',
            'metoclopramide', 'sucralfate', 'antacid', 'lactulose',
            'bisacodyl', 'loperamide', 'mesalamine',
            
            # Respiratory
            'salbutamol', 'albuterol', 'budesonide', 'fluticasone',
            'montelukast', 'theophylline', 'ipratropium', 'tiotropium',
            'cetirizine', 'levocetirizine', 'fexofenadine', 'loratadine',
            'chlorpheniramine', 'diphenhydramine', 'dextromethorphan',
            'guaifenesin', 'ambroxol', 'bromhexine',
            
            # CNS
            'alprazolam', 'diazepam', 'lorazepam', 'clonazepam',
            'escitalopram', 'sertraline', 'fluoxetine', 'paroxetine',
            'duloxetine', 'venlafaxine', 'amitriptyline', 'gabapentin',
            'pregabalin', 'carbamazepine', 'phenytoin', 'valproate',
            'levetiracetam', 'lamotrigine', 'topiramate', 'zolpidem',
            
            # Others
            'levothyroxine', 'prednisone', 'prednisolone', 'dexamethasone',
            'methylprednisolone', 'hydroxychloroquine', 'methotrexate',
            'folic', 'folate', 'calcium', 'vitamin', 'iron', 'zinc',
            'multivitamin', 'b12', 'thiamine', 'pyridoxine',
            
            # Indian brand names (common)
            'crocin', 'dolo', 'calpol', 'combiflam', 'voveran', 'brufen',
            'taxim', 'cifran', 'ciplox', 'azee', 'zifi', 'monocef',
            'telma', 'amlong', 'stamlo', 'atorva', 'rozavel', 'betaloc',
            'glycomet', 'amaryl', 'januvia', 'pan', 'pantop', 'nexpro',
            'montair', 'alex', 'allegra', 'zyrtec', 'atarax',
            'restyl', 'nexito', 'stalopam', 'gabapin', 'pregabalin',
            'thyronorm', 'eltroxin', 'omnacortil', 'wysolone',
            'shelcal', 'calcimax', 'neurobion', 'becosules',
            'flexura', 'hifenac', 'zerodol', 'ultracet', 'nucoxia',
        }
        return drugs
    
    def enhance_image(self, image_bytes: bytes) -> EnhancementResult:
        """
        Apply multiple enhancement techniques to improve OCR readability
        """
        enhancements_applied = []
        
        try:
            # Open image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_img = img.copy()
            
            # 1. Auto-contrast
            img = ImageOps.autocontrast(img, cutoff=2)
            enhancements_applied.append("auto_contrast")
            
            # 2. Increase contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            enhancements_applied.append("contrast_boost_1.5x")
            
            # 3. Sharpen for clearer edges
            img = img.filter(ImageFilter.SHARPEN)
            enhancements_applied.append("sharpen")
            
            # 4. Slight denoise
            img = img.filter(ImageFilter.MedianFilter(size=3))
            enhancements_applied.append("median_denoise")
            
            # 5. Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            enhancements_applied.append("brightness_1.1x")
            
            # Convert back to bytes
            output = io.BytesIO()
            img.save(output, format='PNG', quality=95)
            enhanced_bytes = output.getvalue()
            
            # Estimate improvement based on histogram analysis
            estimated_improvement = self._estimate_improvement(original_img, img)
            
            return EnhancementResult(
                original_bytes=image_bytes,
                enhanced_bytes=enhanced_bytes,
                enhancement_applied=enhancements_applied,
                estimated_improvement=estimated_improvement
            )
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return EnhancementResult(
                original_bytes=image_bytes,
                enhanced_bytes=image_bytes,
                enhancement_applied=["none_failed"],
                estimated_improvement=0.0
            )
    
    def enhance_for_handwriting(self, image_bytes: bytes) -> List[bytes]:
        """
        Generate multiple enhanced versions optimized for handwriting OCR
        Returns list of enhanced images for multi-pass OCR
        """
        versions = []
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Version 1: High contrast for faded handwriting
            v1 = ImageOps.autocontrast(img, cutoff=5)
            v1 = ImageEnhance.Contrast(v1).enhance(2.0)
            v1 = v1.filter(ImageFilter.SHARPEN)
            versions.append(self._img_to_bytes(v1))
            
            # Version 2: Binarization for clear ink
            v2 = img.convert('L')  # Grayscale
            threshold = 128
            v2 = v2.point(lambda x: 255 if x > threshold else 0, mode='1')
            v2 = v2.convert('RGB')
            versions.append(self._img_to_bytes(v2))
            
            # Version 3: Adaptive enhancement
            v3 = ImageOps.autocontrast(img, cutoff=1)
            v3 = ImageEnhance.Sharpness(v3).enhance(2.0)
            v3 = v3.filter(ImageFilter.EDGE_ENHANCE)
            versions.append(self._img_to_bytes(v3))
            
            # Version 4: Original with just denoise
            v4 = img.filter(ImageFilter.MedianFilter(size=3))
            versions.append(self._img_to_bytes(v4))
            
        except Exception as e:
            logger.error(f"Multi-version enhancement failed: {e}")
            versions.append(image_bytes)
        
        return versions
    
    def _img_to_bytes(self, img: Image.Image) -> bytes:
        """Convert PIL Image to bytes"""
        output = io.BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()
    
    def _estimate_improvement(self, original: Image.Image, enhanced: Image.Image) -> float:
        """Estimate improvement in image quality for OCR"""
        try:
            # Compare contrast ranges
            orig_gray = original.convert('L')
            enh_gray = enhanced.convert('L')
            
            orig_hist = orig_gray.histogram()
            enh_hist = enh_gray.histogram()
            
            # Calculate spread of histogram (higher spread = better contrast)
            def calc_spread(hist):
                total = sum(hist)
                if total == 0:
                    return 0
                mean = sum(i * h for i, h in enumerate(hist)) / total
                variance = sum((i - mean) ** 2 * h for i, h in enumerate(hist)) / total
                return variance ** 0.5
            
            orig_spread = calc_spread(orig_hist)
            enh_spread = calc_spread(enh_hist)
            
            improvement = (enh_spread - orig_spread) / max(orig_spread, 1) * 100
            return min(max(improvement, 0), 50)  # Cap at 50% improvement
            
        except:
            return 0.0
    
    def correct_handwriting_errors(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Apply handwriting-specific OCR error corrections
        """
        corrections = []
        corrected_text = text
        
        # Apply character-level corrections
        for error, fix in self.handwriting_corrections.items():
            if error in corrected_text:
                old_text = corrected_text
                corrected_text = corrected_text.replace(error, fix)
                if old_text != corrected_text:
                    corrections.append({
                        'type': 'handwriting_char',
                        'original': error,
                        'corrected': fix,
                        'reason': f'Common handwriting OCR error: {error} → {fix}'
                    })
        
        # Apply word-level corrections using drug database
        words = corrected_text.split()
        corrected_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word).lower()
            
            # Skip short words, numbers, and known words
            if len(clean_word) < 4 or clean_word.isdigit():
                corrected_words.append(word)
                continue
            
            # Check if word might be a misspelled drug
            if clean_word not in self.known_drugs:
                best_match = self._find_drug_match(clean_word)
                if best_match:
                    # Preserve punctuation and casing
                    leading = ''
                    trailing = ''
                    for i, c in enumerate(word):
                        if c.isalnum():
                            break
                        leading += c
                    for i in range(len(word) - 1, -1, -1):
                        if word[i].isalnum():
                            break
                        trailing = word[i] + trailing
                    
                    # Preserve original casing pattern
                    if word[len(leading):len(leading)+1].isupper():
                        best_match = best_match.capitalize()
                    
                    corrected_word = leading + best_match + trailing
                    corrected_words.append(corrected_word)
                    
                    if corrected_word.lower() != word.lower():
                        corrections.append({
                            'type': 'drug_spelling',
                            'original': word,
                            'corrected': corrected_word,
                            'reason': f'Matched to known drug: {best_match}',
                            'confidence': 0.85
                        })
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        # Fix common patterns
        corrected_text, pattern_corrections = self._fix_common_patterns(corrected_text)
        corrections.extend(pattern_corrections)
        
        return corrected_text, corrections
    
    def _find_drug_match(self, word: str, threshold: float = 0.80) -> Optional[str]:
        """Find best matching drug name"""
        word_lower = word.lower()
        best_match = None
        best_score = 0
        
        for drug in self.known_drugs:
            # Skip if length difference is too large
            if abs(len(drug) - len(word_lower)) > 3:
                continue
            
            score = SequenceMatcher(None, word_lower, drug).ratio()
            if score > threshold and score > best_score:
                best_score = score
                best_match = drug
        
        return best_match
    
    def _fix_common_patterns(self, text: str) -> Tuple[str, List[Dict]]:
        """Fix common OCR errors in medical patterns"""
        corrections = []
        
        patterns = [
            # Dosage fixes
            (r'(\d+)\s*(rng|rnq)', r'\1 mg', 'Dosage unit: rng → mg'),
            (r'(\d+)\s*(rnl)', r'\1 ml', 'Dosage unit: rnl → ml'),
            (r'(\d+)\s*(rncg)', r'\1 mcg', 'Dosage unit: rncg → mcg'),
            (r'(\d+)\s*lU\b', r'\1 IU', 'Dosage unit: lU → IU'),
            
            # Frequency fixes
            (r'\b(\d+)\s*x\s*(\d+)\b', r'\1-\2', 'Frequency format'),
            (r'(\d+)-0-(\d+)', r'\1-0-\2', 'Frequency format'),
            (r'\bOD\b', 'once daily', 'Frequency: OD → once daily'),
            (r'\bBD\b', 'twice daily', 'Frequency: BD → twice daily'),
            (r'\bTDS\b', 'three times daily', 'Frequency: TDS → three times daily'),
            
            # Time fixes
            (r'\brnorning\b', 'morning', 'Time: rnorning → morning'),
            (r'\baftemoon\b', 'afternoon', 'Time: aftemoon → afternoon'),
            (r'\bevemng\b', 'evening', 'Time: evemng → evening'),
            (r'\bnlght\b', 'night', 'Time: nlght → night'),
            
            # Form fixes
            (r'\btabIet[s]?\b', 'tablet', 'Form: tabIet → tablet'),
            (r'\bcapsuIe[s]?\b', 'capsule', 'Form: capsuIe → capsule'),
            (r'\bsymp\b', 'syrup', 'Form: symp → syrup'),
            (r'\binj[.]?\b', 'injection', 'Form: inj → injection'),
            
            # Instruction fixes
            (r'\bafter\s+meaIs\b', 'after meals', 'Instruction: meaIs → meals'),
            (r'\bbefore\s+meaIs\b', 'before meals', 'Instruction: meaIs → meals'),
            (r'\bwith\s+foocl\b', 'with food', 'Instruction: foocl → food'),
        ]
        
        for pattern, replacement, reason in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                old_text = text
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if old_text != text:
                    corrections.append({
                        'type': 'pattern_fix',
                        'pattern': pattern,
                        'corrected': replacement,
                        'reason': reason
                    })
        
        return text, corrections
    
    def validate_medications(self, medications: List[Dict]) -> List[Dict]:
        """
        Validate and correct medication names against drug database
        """
        validated = []
        
        for med in medications:
            name = med.get('name', '')
            if not name:
                continue
            
            clean_name = re.sub(r'[^\w\s]', '', name).strip().lower()
            
            # Check if valid drug
            if clean_name in self.known_drugs:
                med['validated'] = True
                med['confidence'] = 0.95
            else:
                # Try to find match
                match = self._find_drug_match(clean_name, threshold=0.75)
                if match:
                    med['original_name'] = name
                    med['name'] = match.capitalize()
                    med['validated'] = True
                    med['confidence'] = 0.85
                    med['correction_applied'] = f'{name} → {match}'
                else:
                    med['validated'] = False
                    med['confidence'] = 0.5
                    med['warning'] = 'Could not validate against drug database'
            
            validated.append(med)
        
        return validated
    
    def get_confidence_factors(self, text: str, corrections: List[Dict]) -> Dict[str, Any]:
        """
        Calculate confidence factors based on corrections made
        """
        total_words = len(text.split())
        correction_count = len(corrections)
        
        # Calculate correction ratio
        correction_ratio = correction_count / max(total_words, 1)
        
        # Base confidence
        base_confidence = 0.90
        
        # Reduce confidence based on corrections needed
        confidence_penalty = correction_ratio * 0.3
        
        # Check for drug matches
        drug_matches = sum(1 for c in corrections if c.get('type') == 'drug_spelling')
        drug_match_bonus = min(drug_matches * 0.02, 0.1)  # Up to 10% bonus for successful matches
        
        final_confidence = max(0.5, base_confidence - confidence_penalty + drug_match_bonus)
        
        return {
            'final_confidence': round(final_confidence, 2),
            'base_confidence': base_confidence,
            'correction_ratio': round(correction_ratio, 3),
            'total_corrections': correction_count,
            'drug_matches': drug_matches,
            'factors': [
                {'name': 'Base OCR confidence', 'impact': base_confidence},
                {'name': 'Corrections needed', 'impact': -confidence_penalty},
                {'name': 'Drug name validations', 'impact': drug_match_bonus},
            ]
        }


# Create singleton instance
handwriting_enhancer = HandwritingEnhancer()
