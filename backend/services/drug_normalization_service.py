"""
Drug Normalization Service - Brand to generic mapping and standardization
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class NormalizedDrug:
    """Normalized drug information"""
    original_name: str
    generic_name: str
    brand_names: List[str]
    drug_class: str
    confidence: float
    is_brand: bool
    common_dosages: List[str]


class DrugNormalizationService:
    """
    Drug name normalization service
    
    Features:
    - Brand to generic name conversion
    - Drug class identification
    - Duplicate detection
    - Standardized naming
    """
    
    def __init__(self):
        # Comprehensive drug database
        self.drug_database = self._build_drug_database()
        
        # Build lookup indices
        self._build_indices()
    
    def _build_drug_database(self) -> Dict[str, Dict]:
        """Build comprehensive drug database"""
        return {
            # Analgesics/Antipyretics
            'paracetamol': {
                'brand_names': ['tylenol', 'crocin', 'dolo', 'calpol', 'panadol', 'acetaminophen', 
                               'dolo 650', 'crocin 650', 'p-650', 'pacimol', 'metacin'],
                'drug_class': 'analgesic_antipyretic',
                'therapeutic_class': 'Pain Relief',
                'common_dosages': ['325mg', '500mg', '650mg', '1000mg'],
                'max_daily_dose': '4000mg'
            },
            'ibuprofen': {
                'brand_names': ['advil', 'motrin', 'brufen', 'nurofen', 'ibugesic', 'combiflam'],
                'drug_class': 'nsaid',
                'therapeutic_class': 'Pain Relief',
                'common_dosages': ['200mg', '400mg', '600mg', '800mg'],
                'max_daily_dose': '3200mg'
            },
            'diclofenac': {
                'brand_names': ['voltaren', 'voveran', 'cataflam', 'diclogesic', 'reactin'],
                'drug_class': 'nsaid',
                'therapeutic_class': 'Pain Relief',
                'common_dosages': ['25mg', '50mg', '75mg', '100mg'],
                'max_daily_dose': '150mg'
            },
            'aspirin': {
                'brand_names': ['ecosprin', 'disprin', 'aspro', 'bayer aspirin', 'aspilet'],
                'drug_class': 'nsaid_antiplatelet',
                'therapeutic_class': 'Pain Relief / Cardiac',
                'common_dosages': ['75mg', '81mg', '150mg', '325mg'],
            },
            'tramadol': {
                'brand_names': ['ultram', 'tramazac', 'contramal', 'domadol', 'trambax'],
                'drug_class': 'opioid_analgesic',
                'therapeutic_class': 'Pain Relief',
                'common_dosages': ['50mg', '100mg'],
            },
            
            # Antibiotics
            'amoxicillin': {
                'brand_names': ['amoxil', 'mox', 'novamox', 'trimox', 'wymox', 'amoxyclav'],
                'drug_class': 'antibiotic_penicillin',
                'therapeutic_class': 'Antibiotic',
                'common_dosages': ['250mg', '500mg', '875mg'],
            },
            'azithromycin': {
                'brand_names': ['zithromax', 'azithral', 'zmax', 'azee', 'azicip', 'azibact'],
                'drug_class': 'antibiotic_macrolide',
                'therapeutic_class': 'Antibiotic',
                'common_dosages': ['250mg', '500mg'],
            },
            'ciprofloxacin': {
                'brand_names': ['cipro', 'ciplox', 'cifran', 'ciprolet'],
                'drug_class': 'antibiotic_fluoroquinolone',
                'therapeutic_class': 'Antibiotic',
                'common_dosages': ['250mg', '500mg', '750mg'],
            },
            'doxycycline': {
                'brand_names': ['vibramycin', 'doxy', 'doxt', 'oracea'],
                'drug_class': 'antibiotic_tetracycline',
                'therapeutic_class': 'Antibiotic',
                'common_dosages': ['50mg', '100mg'],
            },
            'metronidazole': {
                'brand_names': ['flagyl', 'metrogyl', 'rozex', 'metron'],
                'drug_class': 'antibiotic_antiprotozoal',
                'therapeutic_class': 'Antibiotic',
                'common_dosages': ['200mg', '400mg', '500mg'],
            },
            'cephalexin': {
                'brand_names': ['keflex', 'ceff', 'sporidex', 'ceporex'],
                'drug_class': 'antibiotic_cephalosporin',
                'therapeutic_class': 'Antibiotic',
                'common_dosages': ['250mg', '500mg'],
            },
            'cefixime': {
                'brand_names': ['suprax', 'taxim', 'cefspan', 'zifi'],
                'drug_class': 'antibiotic_cephalosporin',
                'therapeutic_class': 'Antibiotic',
                'common_dosages': ['100mg', '200mg', '400mg'],
            },
            
            # Cardiovascular
            'amlodipine': {
                'brand_names': ['norvasc', 'amlong', 'amlip', 'amlopin', 'amtas'],
                'drug_class': 'calcium_channel_blocker',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['2.5mg', '5mg', '10mg'],
            },
            'atorvastatin': {
                'brand_names': ['lipitor', 'atorva', 'storvas', 'tonact', 'atorlip'],
                'drug_class': 'statin',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['10mg', '20mg', '40mg', '80mg'],
            },
            'rosuvastatin': {
                'brand_names': ['crestor', 'rosuvas', 'rozavel', 'rosulip'],
                'drug_class': 'statin',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['5mg', '10mg', '20mg', '40mg'],
            },
            'metoprolol': {
                'brand_names': ['lopressor', 'toprol', 'betaloc', 'metolar'],
                'drug_class': 'beta_blocker',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['25mg', '50mg', '100mg'],
            },
            'atenolol': {
                'brand_names': ['tenormin', 'betacard', 'aten', 'tenolol'],
                'drug_class': 'beta_blocker',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['25mg', '50mg', '100mg'],
            },
            'lisinopril': {
                'brand_names': ['zestril', 'prinivil', 'listril', 'lipril'],
                'drug_class': 'ace_inhibitor',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['5mg', '10mg', '20mg', '40mg'],
            },
            'losartan': {
                'brand_names': ['cozaar', 'losacar', 'losar', 'repace'],
                'drug_class': 'arb',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['25mg', '50mg', '100mg'],
            },
            'telmisartan': {
                'brand_names': ['micardis', 'telma', 'telmikind', 'telsar'],
                'drug_class': 'arb',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['20mg', '40mg', '80mg'],
            },
            'clopidogrel': {
                'brand_names': ['plavix', 'clopilet', 'deplatt', 'clavix'],
                'drug_class': 'antiplatelet',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['75mg'],
            },
            'warfarin': {
                'brand_names': ['coumadin', 'warf', 'acitrom', 'warfarin'],
                'drug_class': 'anticoagulant',
                'therapeutic_class': 'Cardiovascular',
                'common_dosages': ['1mg', '2mg', '2.5mg', '5mg'],
            },
            
            # Diabetes
            'metformin': {
                'brand_names': ['glucophage', 'glycomet', 'glumet', 'fortamet', 'glyciphage'],
                'drug_class': 'biguanide',
                'therapeutic_class': 'Antidiabetic',
                'common_dosages': ['500mg', '850mg', '1000mg'],
            },
            'glimepiride': {
                'brand_names': ['amaryl', 'glimestar', 'glimy', 'glimisave'],
                'drug_class': 'sulfonylurea',
                'therapeutic_class': 'Antidiabetic',
                'common_dosages': ['1mg', '2mg', '4mg'],
            },
            'sitagliptin': {
                'brand_names': ['januvia', 'zita', 'istavel'],
                'drug_class': 'dpp4_inhibitor',
                'therapeutic_class': 'Antidiabetic',
                'common_dosages': ['25mg', '50mg', '100mg'],
            },
            
            # Gastrointestinal
            'omeprazole': {
                'brand_names': ['prilosec', 'omez', 'losec', 'ocid', 'omecip'],
                'drug_class': 'ppi',
                'therapeutic_class': 'Gastrointestinal',
                'common_dosages': ['20mg', '40mg'],
            },
            'pantoprazole': {
                'brand_names': ['protonix', 'pan', 'pantop', 'pantocar', 'pantocid'],
                'drug_class': 'ppi',
                'therapeutic_class': 'Gastrointestinal',
                'common_dosages': ['20mg', '40mg'],
            },
            'ranitidine': {
                'brand_names': ['zantac', 'aciloc', 'rantac', 'zinetac'],
                'drug_class': 'h2_blocker',
                'therapeutic_class': 'Gastrointestinal',
                'common_dosages': ['150mg', '300mg'],
            },
            'domperidone': {
                'brand_names': ['motilium', 'domstal', 'vomistop', 'domperi'],
                'drug_class': 'prokinetic',
                'therapeutic_class': 'Gastrointestinal',
                'common_dosages': ['10mg'],
            },
            'ondansetron': {
                'brand_names': ['zofran', 'ondem', 'emeset', 'vomikind'],
                'drug_class': 'antiemetic',
                'therapeutic_class': 'Gastrointestinal',
                'common_dosages': ['4mg', '8mg'],
            },
            
            # Respiratory
            'salbutamol': {
                'brand_names': ['ventolin', 'asthalin', 'proventil', 'albuterol', 'derihaler'],
                'drug_class': 'beta2_agonist',
                'therapeutic_class': 'Respiratory',
                'common_dosages': ['2mg', '4mg', '100mcg'],
            },
            'montelukast': {
                'brand_names': ['singulair', 'montair', 'montek', 'telekast'],
                'drug_class': 'leukotriene_antagonist',
                'therapeutic_class': 'Respiratory',
                'common_dosages': ['4mg', '5mg', '10mg'],
            },
            'cetirizine': {
                'brand_names': ['zyrtec', 'cetzine', 'incid', 'okacet'],
                'drug_class': 'antihistamine',
                'therapeutic_class': 'Allergy',
                'common_dosages': ['5mg', '10mg'],
            },
            'loratadine': {
                'brand_names': ['claritin', 'lorfast', 'alavert', 'clarityn'],
                'drug_class': 'antihistamine',
                'therapeutic_class': 'Allergy',
                'common_dosages': ['10mg'],
            },
            'fexofenadine': {
                'brand_names': ['allegra', 'fexova', 'altiva'],
                'drug_class': 'antihistamine',
                'therapeutic_class': 'Allergy',
                'common_dosages': ['120mg', '180mg'],
            },
            
            # Psychiatric/CNS
            'sertraline': {
                'brand_names': ['zoloft', 'serlift', 'lustral', 'daxid'],
                'drug_class': 'ssri',
                'therapeutic_class': 'Psychiatric',
                'common_dosages': ['25mg', '50mg', '100mg'],
            },
            'escitalopram': {
                'brand_names': ['lexapro', 'cipralex', 'nexito', 'stalopam'],
                'drug_class': 'ssri',
                'therapeutic_class': 'Psychiatric',
                'common_dosages': ['5mg', '10mg', '20mg'],
            },
            'alprazolam': {
                'brand_names': ['xanax', 'alprax', 'restyl', 'trika'],
                'drug_class': 'benzodiazepine',
                'therapeutic_class': 'Psychiatric',
                'common_dosages': ['0.25mg', '0.5mg', '1mg'],
            },
            'clonazepam': {
                'brand_names': ['klonopin', 'rivotril', 'clonotril', 'zapiz'],
                'drug_class': 'benzodiazepine',
                'therapeutic_class': 'Psychiatric',
                'common_dosages': ['0.25mg', '0.5mg', '1mg', '2mg'],
            },
            'gabapentin': {
                'brand_names': ['neurontin', 'gabantin', 'gralise', 'gabapin'],
                'drug_class': 'anticonvulsant',
                'therapeutic_class': 'Neurological',
                'common_dosages': ['100mg', '300mg', '400mg', '600mg'],
            },
            
            # Thyroid
            'levothyroxine': {
                'brand_names': ['synthroid', 'thyronorm', 'eltroxin', 'levoxyl', 'thyrox'],
                'drug_class': 'thyroid_hormone',
                'therapeutic_class': 'Endocrine',
                'common_dosages': ['25mcg', '50mcg', '75mcg', '100mcg', '125mcg'],
            },
            
            # Vitamins/Supplements
            'vitamin_d': {
                'brand_names': ['calcirol', 'd3 must', 'arachitol', 'cholecalciferol'],
                'drug_class': 'vitamin',
                'therapeutic_class': 'Supplement',
                'common_dosages': ['1000IU', '2000IU', '60000IU'],
            },
            'vitamin_b12': {
                'brand_names': ['neurobion', 'methylcobal', 'mecobalamin', 'cobadex'],
                'drug_class': 'vitamin',
                'therapeutic_class': 'Supplement',
                'common_dosages': ['500mcg', '1000mcg', '1500mcg'],
            },
            'calcium': {
                'brand_names': ['shelcal', 'calcimax', 'gemcal', 'ccm'],
                'drug_class': 'mineral',
                'therapeutic_class': 'Supplement',
                'common_dosages': ['500mg', '1000mg'],
            },
            'iron': {
                'brand_names': ['orofer', 'autrin', 'fefol', 'ferrous sulfate'],
                'drug_class': 'mineral',
                'therapeutic_class': 'Supplement',
                'common_dosages': ['60mg', '100mg'],
            },
        }
    
    def _build_indices(self):
        """Build lookup indices for fast matching"""
        self.brand_to_generic = {}
        self.all_names = set()
        self.name_to_data = {}
        
        for generic, data in self.drug_database.items():
            # Add generic name
            self.brand_to_generic[generic.lower()] = generic
            self.all_names.add(generic.lower())
            self.name_to_data[generic.lower()] = {'generic': generic, 'is_brand': False, **data}
            
            # Add brand names
            for brand in data.get('brand_names', []):
                brand_lower = brand.lower()
                self.brand_to_generic[brand_lower] = generic
                self.all_names.add(brand_lower)
                self.name_to_data[brand_lower] = {'generic': generic, 'is_brand': True, **data}
    
    def normalize(self, drug_name: str) -> NormalizedDrug:
        """
        Normalize a drug name to generic form
        
        Args:
            drug_name: Drug name (brand or generic)
            
        Returns:
            NormalizedDrug with normalized information
        """
        drug_lower = drug_name.lower().strip()
        
        # Direct lookup
        if drug_lower in self.name_to_data:
            data = self.name_to_data[drug_lower]
            return NormalizedDrug(
                original_name=drug_name,
                generic_name=data['generic'],
                brand_names=data.get('brand_names', []),
                drug_class=data.get('drug_class', 'unknown'),
                confidence=0.95,
                is_brand=data.get('is_brand', False),
                common_dosages=data.get('common_dosages', [])
            )
        
        # Fuzzy matching
        best_match = self._fuzzy_match(drug_lower)
        if best_match:
            data = self.name_to_data[best_match['name']]
            return NormalizedDrug(
                original_name=drug_name,
                generic_name=data['generic'],
                brand_names=data.get('brand_names', []),
                drug_class=data.get('drug_class', 'unknown'),
                confidence=best_match['confidence'],
                is_brand=data.get('is_brand', False),
                common_dosages=data.get('common_dosages', [])
            )
        
        # Not found - return as unknown
        return NormalizedDrug(
            original_name=drug_name,
            generic_name=drug_name,
            brand_names=[],
            drug_class='unknown',
            confidence=0.3,
            is_brand=False,
            common_dosages=[]
        )
    
    def _fuzzy_match(self, name: str, threshold: float = 0.8) -> Optional[Dict]:
        """Find best fuzzy match for a drug name"""
        best_match = None
        best_score = 0
        
        for known_name in self.all_names:
            # Skip if length difference is too large
            if abs(len(known_name) - len(name)) > 3:
                continue
            
            score = SequenceMatcher(None, name, known_name).ratio()
            if score > threshold and score > best_score:
                best_score = score
                best_match = {'name': known_name, 'confidence': score}
        
        return best_match
    
    def detect_duplicates(self, medications: List[str]) -> List[Dict]:
        """
        Detect duplicate medications (same generic drug, different names)
        
        Args:
            medications: List of medication names
            
        Returns:
            List of duplicate groups
        """
        # Normalize all medications
        normalized = [self.normalize(med) for med in medications]
        
        # Group by generic name
        generic_groups = {}
        for i, norm in enumerate(normalized):
            generic = norm.generic_name.lower()
            if generic not in generic_groups:
                generic_groups[generic] = []
            generic_groups[generic].append({
                'original': medications[i],
                'normalized': norm
            })
        
        # Find duplicates
        duplicates = []
        for generic, group in generic_groups.items():
            if len(group) > 1:
                duplicates.append({
                    'generic_name': generic,
                    'count': len(group),
                    'medications': [g['original'] for g in group],
                    'is_duplicate': True,
                    'warning': f"Multiple forms of {generic} detected"
                })
        
        return duplicates
    
    def get_drug_class(self, drug_name: str) -> Optional[str]:
        """Get the drug class for a medication"""
        normalized = self.normalize(drug_name)
        return normalized.drug_class if normalized.confidence > 0.5 else None
    
    def are_same_drug(self, drug1: str, drug2: str) -> Tuple[bool, float]:
        """
        Check if two drug names refer to the same medication
        
        Returns:
            Tuple of (is_same, confidence)
        """
        norm1 = self.normalize(drug1)
        norm2 = self.normalize(drug2)
        
        if norm1.generic_name.lower() == norm2.generic_name.lower():
            confidence = min(norm1.confidence, norm2.confidence)
            return (True, confidence)
        
        return (False, 0.0)
    
    def get_therapeutic_alternatives(self, drug_name: str) -> List[str]:
        """Get other drugs in the same therapeutic class"""
        normalized = self.normalize(drug_name)
        drug_class = normalized.drug_class
        
        if drug_class == 'unknown':
            return []
        
        alternatives = []
        for generic, data in self.drug_database.items():
            if data.get('drug_class') == drug_class and generic != normalized.generic_name:
                alternatives.append(generic)
        
        return alternatives
    
    def standardize_name(self, drug_name: str) -> str:
        """Get standardized (generic) name for a drug"""
        normalized = self.normalize(drug_name)
        if normalized.confidence > 0.7:
            return normalized.generic_name
        return drug_name
