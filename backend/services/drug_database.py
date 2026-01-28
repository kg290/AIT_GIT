"""
Drug Interaction Database - Comprehensive drug safety data
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"


@dataclass
class DrugInteraction:
    drug1: str
    drug2: str
    severity: Severity
    description: str
    mechanism: str
    management: str
    

@dataclass
class AllergyAlert:
    drug: str
    allergen: str
    cross_reactivity: bool
    alternatives: List[str]


# Drug class mappings
DRUG_CLASSES = {
    # NSAIDs
    'ibuprofen': ['nsaid', 'pain_reliever'],
    'aspirin': ['nsaid', 'pain_reliever', 'antiplatelet'],
    'naproxen': ['nsaid', 'pain_reliever'],
    'diclofenac': ['nsaid', 'pain_reliever'],
    'piroxicam': ['nsaid', 'pain_reliever'],
    'indomethacin': ['nsaid', 'pain_reliever'],
    'celecoxib': ['nsaid', 'cox2_inhibitor'],
    'etoricoxib': ['nsaid', 'cox2_inhibitor'],
    'aceclofenac': ['nsaid', 'pain_reliever'],
    'nimesulide': ['nsaid', 'pain_reliever'],
    
    # Antibiotics - Penicillins
    'amoxicillin': ['antibiotic', 'penicillin'],
    'ampicillin': ['antibiotic', 'penicillin'],
    'penicillin': ['antibiotic', 'penicillin'],
    'amoxyclav': ['antibiotic', 'penicillin'],
    'augmentin': ['antibiotic', 'penicillin'],
    'piperacillin': ['antibiotic', 'penicillin'],
    
    # Antibiotics - Cephalosporins
    'cefixime': ['antibiotic', 'cephalosporin'],
    'ceftriaxone': ['antibiotic', 'cephalosporin'],
    'cefpodoxime': ['antibiotic', 'cephalosporin'],
    'cephalexin': ['antibiotic', 'cephalosporin'],
    'cefuroxime': ['antibiotic', 'cephalosporin'],
    
    # Antibiotics - Fluoroquinolones
    'ciprofloxacin': ['antibiotic', 'fluoroquinolone'],
    'levofloxacin': ['antibiotic', 'fluoroquinolone'],
    'ofloxacin': ['antibiotic', 'fluoroquinolone'],
    'norfloxacin': ['antibiotic', 'fluoroquinolone'],
    'moxifloxacin': ['antibiotic', 'fluoroquinolone'],
    
    # Antibiotics - Macrolides
    'azithromycin': ['antibiotic', 'macrolide'],
    'erythromycin': ['antibiotic', 'macrolide'],
    'clarithromycin': ['antibiotic', 'macrolide'],
    
    # Antibiotics - Others
    'metronidazole': ['antibiotic', 'nitroimidazole'],
    'doxycycline': ['antibiotic', 'tetracycline'],
    'clindamycin': ['antibiotic', 'lincosamide'],
    
    # Blood Thinners
    'warfarin': ['anticoagulant', 'blood_thinner'],
    'heparin': ['anticoagulant', 'blood_thinner'],
    'enoxaparin': ['anticoagulant', 'blood_thinner', 'lmwh'],
    'rivaroxaban': ['anticoagulant', 'blood_thinner', 'doac'],
    'apixaban': ['anticoagulant', 'blood_thinner', 'doac'],
    'dabigatran': ['anticoagulant', 'blood_thinner', 'doac'],
    'clopidogrel': ['antiplatelet', 'blood_thinner'],
    
    # Diabetes
    'metformin': ['antidiabetic', 'biguanide'],
    'glimepiride': ['antidiabetic', 'sulfonylurea'],
    'gliclazide': ['antidiabetic', 'sulfonylurea'],
    'glipizide': ['antidiabetic', 'sulfonylurea'],
    'pioglitazone': ['antidiabetic', 'thiazolidinedione'],
    'sitagliptin': ['antidiabetic', 'dpp4_inhibitor'],
    'vildagliptin': ['antidiabetic', 'dpp4_inhibitor'],
    'empagliflozin': ['antidiabetic', 'sglt2_inhibitor'],
    'dapagliflozin': ['antidiabetic', 'sglt2_inhibitor'],
    'insulin': ['antidiabetic', 'insulin'],
    
    # Blood Pressure
    'amlodipine': ['antihypertensive', 'calcium_channel_blocker'],
    'nifedipine': ['antihypertensive', 'calcium_channel_blocker'],
    'diltiazem': ['antihypertensive', 'calcium_channel_blocker'],
    'verapamil': ['antihypertensive', 'calcium_channel_blocker'],
    'losartan': ['antihypertensive', 'arb'],
    'telmisartan': ['antihypertensive', 'arb'],
    'olmesartan': ['antihypertensive', 'arb'],
    'valsartan': ['antihypertensive', 'arb'],
    'enalapril': ['antihypertensive', 'ace_inhibitor'],
    'ramipril': ['antihypertensive', 'ace_inhibitor'],
    'lisinopril': ['antihypertensive', 'ace_inhibitor'],
    'atenolol': ['antihypertensive', 'beta_blocker'],
    'metoprolol': ['antihypertensive', 'beta_blocker'],
    'propranolol': ['antihypertensive', 'beta_blocker'],
    'carvedilol': ['antihypertensive', 'beta_blocker'],
    'nebivolol': ['antihypertensive', 'beta_blocker'],
    'hydrochlorothiazide': ['antihypertensive', 'diuretic', 'thiazide'],
    'furosemide': ['antihypertensive', 'diuretic', 'loop_diuretic'],
    'spironolactone': ['antihypertensive', 'diuretic', 'potassium_sparing'],
    
    # Cholesterol
    'atorvastatin': ['statin', 'cholesterol'],
    'rosuvastatin': ['statin', 'cholesterol'],
    'simvastatin': ['statin', 'cholesterol'],
    'pravastatin': ['statin', 'cholesterol'],
    'fenofibrate': ['fibrate', 'cholesterol'],
    
    # Acid Reducers
    'omeprazole': ['ppi', 'acid_reducer'],
    'pantoprazole': ['ppi', 'acid_reducer'],
    'rabeprazole': ['ppi', 'acid_reducer'],
    'esomeprazole': ['ppi', 'acid_reducer'],
    'lansoprazole': ['ppi', 'acid_reducer'],
    'ranitidine': ['h2_blocker', 'acid_reducer'],
    'famotidine': ['h2_blocker', 'acid_reducer'],
    
    # Pain/Analgesics
    'paracetamol': ['analgesic', 'antipyretic'],
    'acetaminophen': ['analgesic', 'antipyretic'],
    'tramadol': ['opioid', 'analgesic'],
    'morphine': ['opioid', 'analgesic'],
    'codeine': ['opioid', 'analgesic'],
    'fentanyl': ['opioid', 'analgesic'],
    
    # Antihistamines
    'cetirizine': ['antihistamine', 'h1_blocker'],
    'levocetirizine': ['antihistamine', 'h1_blocker'],
    'fexofenadine': ['antihistamine', 'h1_blocker'],
    'loratadine': ['antihistamine', 'h1_blocker'],
    'chlorpheniramine': ['antihistamine', 'h1_blocker', 'sedating'],
    'diphenhydramine': ['antihistamine', 'h1_blocker', 'sedating'],
    'promethazine': ['antihistamine', 'h1_blocker', 'sedating'],
    
    # Steroids
    'prednisolone': ['corticosteroid', 'steroid'],
    'prednisone': ['corticosteroid', 'steroid'],
    'methylprednisolone': ['corticosteroid', 'steroid'],
    'dexamethasone': ['corticosteroid', 'steroid'],
    'hydrocortisone': ['corticosteroid', 'steroid'],
    'betamethasone': ['corticosteroid', 'steroid'],
    
    # Thyroid
    'levothyroxine': ['thyroid', 't4'],
    'thyroxine': ['thyroid', 't4'],
    'carbimazole': ['antithyroid'],
    'methimazole': ['antithyroid'],
    
    # Psychiatric
    'alprazolam': ['benzodiazepine', 'anxiolytic'],
    'diazepam': ['benzodiazepine', 'anxiolytic'],
    'lorazepam': ['benzodiazepine', 'anxiolytic'],
    'clonazepam': ['benzodiazepine', 'anxiolytic'],
    'escitalopram': ['antidepressant', 'ssri'],
    'sertraline': ['antidepressant', 'ssri'],
    'fluoxetine': ['antidepressant', 'ssri'],
    'paroxetine': ['antidepressant', 'ssri'],
    'amitriptyline': ['antidepressant', 'tca'],
    'quetiapine': ['antipsychotic', 'atypical'],
    'olanzapine': ['antipsychotic', 'atypical'],
    'risperidone': ['antipsychotic', 'atypical'],
    
    # Respiratory
    'salbutamol': ['bronchodilator', 'beta2_agonist'],
    'ipratropium': ['bronchodilator', 'anticholinergic'],
    'montelukast': ['leukotriene_antagonist', 'asthma'],
    'budesonide': ['inhaled_corticosteroid', 'asthma'],
    'fluticasone': ['inhaled_corticosteroid', 'asthma'],
    'theophylline': ['bronchodilator', 'methylxanthine'],
    
    # GI
    'domperidone': ['prokinetic', 'antiemetic'],
    'metoclopramide': ['prokinetic', 'antiemetic'],
    'ondansetron': ['antiemetic', '5ht3_antagonist'],
    'loperamide': ['antidiarrheal'],
    'lactulose': ['laxative', 'osmotic'],
    'bisacodyl': ['laxative', 'stimulant'],
}


# Known drug interactions
INTERACTIONS_DATABASE = [
    # Warfarin interactions
    DrugInteraction("warfarin", "aspirin", Severity.MAJOR,
        "Increased bleeding risk",
        "Both drugs affect clotting through different mechanisms",
        "Avoid combination if possible. If necessary, monitor INR closely and watch for bleeding signs."),
    DrugInteraction("warfarin", "ibuprofen", Severity.MAJOR,
        "Increased bleeding risk and INR elevation",
        "NSAIDs inhibit platelet function and can increase warfarin levels",
        "Use acetaminophen instead. If NSAID needed, use lowest dose for shortest time."),
    DrugInteraction("warfarin", "metronidazole", Severity.MAJOR,
        "Increased anticoagulant effect",
        "Metronidazole inhibits warfarin metabolism",
        "Reduce warfarin dose by 25-30% and monitor INR."),
    DrugInteraction("warfarin", "azithromycin", Severity.MODERATE,
        "May increase anticoagulant effect",
        "Azithromycin may inhibit warfarin metabolism",
        "Monitor INR more frequently during antibiotic course."),
    
    # ACE inhibitor + Potassium sparing diuretic
    DrugInteraction("enalapril", "spironolactone", Severity.MAJOR,
        "Risk of severe hyperkalemia",
        "Both drugs increase potassium retention",
        "Monitor potassium levels closely. Avoid in renal impairment."),
    DrugInteraction("ramipril", "spironolactone", Severity.MAJOR,
        "Risk of severe hyperkalemia",
        "Both drugs increase potassium retention",
        "Monitor potassium levels closely. Avoid in renal impairment."),
    DrugInteraction("lisinopril", "spironolactone", Severity.MAJOR,
        "Risk of severe hyperkalemia",
        "Both drugs increase potassium retention",
        "Monitor potassium levels closely. Avoid in renal impairment."),
    
    # Metformin + Contrast dye consideration
    DrugInteraction("metformin", "iodinated contrast", Severity.MAJOR,
        "Risk of lactic acidosis",
        "Contrast media can cause acute kidney injury affecting metformin clearance",
        "Hold metformin 48 hours before and after contrast procedures."),
    
    # SSRI + MAO inhibitors (serotonin syndrome)
    DrugInteraction("escitalopram", "tramadol", Severity.MAJOR,
        "Risk of serotonin syndrome",
        "Both drugs increase serotonin levels",
        "Use with caution. Monitor for confusion, rapid heart rate, high BP."),
    DrugInteraction("sertraline", "tramadol", Severity.MAJOR,
        "Risk of serotonin syndrome",
        "Both drugs increase serotonin levels",
        "Use with caution. Monitor for confusion, rapid heart rate, high BP."),
    
    # NSAIDs + ACE inhibitors
    DrugInteraction("ibuprofen", "enalapril", Severity.MODERATE,
        "Reduced antihypertensive effect, increased renal risk",
        "NSAIDs reduce prostaglandin-mediated renal blood flow",
        "Monitor blood pressure and renal function."),
    DrugInteraction("ibuprofen", "ramipril", Severity.MODERATE,
        "Reduced antihypertensive effect, increased renal risk",
        "NSAIDs reduce prostaglandin-mediated renal blood flow",
        "Monitor blood pressure and renal function."),
    DrugInteraction("ibuprofen", "lisinopril", Severity.MODERATE,
        "Reduced antihypertensive effect, increased renal risk",
        "NSAIDs reduce prostaglandin-mediated renal blood flow",
        "Monitor blood pressure and renal function."),
    
    # Statin + Macrolide
    DrugInteraction("simvastatin", "clarithromycin", Severity.MAJOR,
        "Increased risk of rhabdomyolysis",
        "Clarithromycin inhibits CYP3A4 which metabolizes simvastatin",
        "Use azithromycin instead, or temporarily stop statin."),
    DrugInteraction("simvastatin", "erythromycin", Severity.MAJOR,
        "Increased risk of rhabdomyolysis",
        "Erythromycin inhibits CYP3A4 which metabolizes simvastatin",
        "Use azithromycin instead, or temporarily stop statin."),
    DrugInteraction("atorvastatin", "clarithromycin", Severity.MODERATE,
        "Increased statin levels",
        "Clarithromycin inhibits CYP3A4",
        "Use lower statin dose or choose azithromycin."),
    
    # Clopidogrel + PPI
    DrugInteraction("clopidogrel", "omeprazole", Severity.MODERATE,
        "Reduced antiplatelet effect",
        "Omeprazole inhibits CYP2C19 needed to activate clopidogrel",
        "Use pantoprazole or H2 blocker instead."),
    DrugInteraction("clopidogrel", "esomeprazole", Severity.MODERATE,
        "Reduced antiplatelet effect",
        "Esomeprazole inhibits CYP2C19 needed to activate clopidogrel",
        "Use pantoprazole or H2 blocker instead."),
    
    # Fluoroquinolone + Steroids
    DrugInteraction("ciprofloxacin", "prednisolone", Severity.MODERATE,
        "Increased risk of tendon rupture",
        "Both drugs independently increase tendon damage risk",
        "Monitor for tendon pain. Consider alternative antibiotic."),
    DrugInteraction("levofloxacin", "prednisolone", Severity.MODERATE,
        "Increased risk of tendon rupture",
        "Both drugs independently increase tendon damage risk",
        "Monitor for tendon pain. Consider alternative antibiotic."),
    
    # Digoxin interactions
    DrugInteraction("digoxin", "amiodarone", Severity.MAJOR,
        "Increased digoxin toxicity",
        "Amiodarone decreases digoxin clearance",
        "Reduce digoxin dose by 50% and monitor levels."),
    DrugInteraction("digoxin", "verapamil", Severity.MAJOR,
        "Increased digoxin levels and bradycardia",
        "Verapamil decreases digoxin clearance and adds to AV block",
        "Reduce digoxin dose and monitor heart rate."),
    
    # Methotrexate + NSAIDs
    DrugInteraction("methotrexate", "ibuprofen", Severity.MAJOR,
        "Increased methotrexate toxicity",
        "NSAIDs reduce methotrexate renal clearance",
        "Avoid NSAIDs or reduce methotrexate dose with close monitoring."),
    DrugInteraction("methotrexate", "aspirin", Severity.MAJOR,
        "Increased methotrexate toxicity",
        "Aspirin displaces methotrexate from protein binding",
        "Avoid combination or monitor methotrexate levels closely."),
    
    # Theophylline interactions
    DrugInteraction("theophylline", "ciprofloxacin", Severity.MAJOR,
        "Increased theophylline toxicity",
        "Ciprofloxacin inhibits theophylline metabolism",
        "Monitor theophylline levels and reduce dose if needed."),
    DrugInteraction("theophylline", "erythromycin", Severity.MODERATE,
        "Increased theophylline levels",
        "Erythromycin inhibits theophylline metabolism",
        "Monitor for toxicity symptoms: nausea, palpitations."),
    
    # Lithium interactions
    DrugInteraction("lithium", "ibuprofen", Severity.MAJOR,
        "Increased lithium toxicity",
        "NSAIDs reduce lithium renal clearance",
        "Avoid NSAIDs or monitor lithium levels closely."),
    DrugInteraction("lithium", "hydrochlorothiazide", Severity.MAJOR,
        "Increased lithium levels",
        "Thiazides reduce lithium clearance",
        "Monitor lithium levels and adjust dose."),
    
    # QT prolongation combinations
    DrugInteraction("azithromycin", "domperidone", Severity.MAJOR,
        "Risk of dangerous heart rhythm (QT prolongation)",
        "Both drugs prolong QT interval",
        "Avoid combination. Use alternative antiemetic."),
    DrugInteraction("ciprofloxacin", "domperidone", Severity.MAJOR,
        "Risk of QT prolongation",
        "Both drugs prolong QT interval",
        "Avoid combination if possible."),
    
    # Metronidazole + Alcohol (disulfiram reaction)
    DrugInteraction("metronidazole", "alcohol", Severity.MAJOR,
        "Severe nausea, vomiting, flushing (disulfiram-like reaction)",
        "Metronidazole inhibits aldehyde dehydrogenase",
        "Strictly avoid alcohol during and 48 hours after treatment."),
    
    # Sildenafil + Nitrates
    DrugInteraction("sildenafil", "isosorbide", Severity.CONTRAINDICATED,
        "Life-threatening hypotension",
        "Synergistic vasodilation",
        "Combination is absolutely contraindicated."),
    DrugInteraction("tadalafil", "isosorbide", Severity.CONTRAINDICATED,
        "Life-threatening hypotension",
        "Synergistic vasodilation",
        "Combination is absolutely contraindicated."),
    
    # Allopurinol + Azathioprine
    DrugInteraction("allopurinol", "azathioprine", Severity.MAJOR,
        "Increased azathioprine toxicity",
        "Allopurinol inhibits xanthine oxidase which metabolizes azathioprine",
        "Reduce azathioprine dose by 75% if combination necessary."),
    
    # Calcium + Thyroid
    DrugInteraction("levothyroxine", "calcium", Severity.MODERATE,
        "Reduced levothyroxine absorption",
        "Calcium forms complexes with levothyroxine in GI tract",
        "Take levothyroxine 4 hours apart from calcium supplements."),
    DrugInteraction("levothyroxine", "omeprazole", Severity.MODERATE,
        "Reduced levothyroxine absorption",
        "Reduced gastric acid affects levothyroxine absorption",
        "Monitor TSH levels; may need levothyroxine dose adjustment."),
    
    # Antacids reducing absorption
    DrugInteraction("ciprofloxacin", "antacid", Severity.MODERATE,
        "Reduced ciprofloxacin absorption",
        "Cations in antacids chelate fluoroquinolones",
        "Take ciprofloxacin 2 hours before or 6 hours after antacids."),
    DrugInteraction("tetracycline", "antacid", Severity.MODERATE,
        "Reduced tetracycline absorption",
        "Cations chelate tetracyclines",
        "Separate administration by 2-3 hours."),
    DrugInteraction("doxycycline", "antacid", Severity.MODERATE,
        "Reduced doxycycline absorption",
        "Cations chelate tetracyclines",
        "Separate administration by 2-3 hours."),
]


# Allergy cross-reactivity
ALLERGY_DATABASE = {
    'penicillin': AllergyAlert(
        drug='penicillin',
        allergen='penicillin',
        cross_reactivity=True,
        alternatives=['azithromycin', 'fluoroquinolones', 'doxycycline']
    ),
    'amoxicillin': AllergyAlert(
        drug='amoxicillin',
        allergen='penicillin',
        cross_reactivity=True,
        alternatives=['azithromycin', 'fluoroquinolones', 'doxycycline']
    ),
    'ampicillin': AllergyAlert(
        drug='ampicillin',
        allergen='penicillin',
        cross_reactivity=True,
        alternatives=['azithromycin', 'fluoroquinolones', 'doxycycline']
    ),
    'cephalosporin': AllergyAlert(
        drug='cephalosporin',
        allergen='penicillin',
        cross_reactivity=True,  # ~1-2% cross-reactivity
        alternatives=['azithromycin', 'fluoroquinolones']
    ),
    'sulfa': AllergyAlert(
        drug='sulfamethoxazole',
        allergen='sulfa',
        cross_reactivity=False,
        alternatives=['amoxicillin', 'azithromycin', 'fluoroquinolones']
    ),
    'aspirin': AllergyAlert(
        drug='aspirin',
        allergen='nsaid',
        cross_reactivity=True,  # Cross-reactive with other NSAIDs
        alternatives=['acetaminophen', 'celecoxib (with caution)']
    ),
    'ibuprofen': AllergyAlert(
        drug='ibuprofen',
        allergen='nsaid',
        cross_reactivity=True,
        alternatives=['acetaminophen']
    ),
}


def normalize_drug_name(name: str) -> str:
    """Normalize drug name for matching"""
    name = name.lower().strip()
    # Remove common suffixes
    name = name.replace(' tablet', '').replace(' capsule', '').replace(' syrup', '')
    name = name.replace(' injection', '').replace(' cream', '').replace(' ointment', '')
    # Remove dosage info
    import re
    name = re.sub(r'\s*\d+\s*(?:mg|mcg|g|ml|iu|%)\s*', '', name)
    return name.strip()


def get_drug_classes(drug_name: str) -> Set[str]:
    """Get drug classes for a medication"""
    normalized = normalize_drug_name(drug_name)
    
    # Direct match
    if normalized in DRUG_CLASSES:
        return set(DRUG_CLASSES[normalized])
    
    # Partial match
    for drug, classes in DRUG_CLASSES.items():
        if drug in normalized or normalized in drug:
            return set(classes)
    
    return set()


def check_interactions(drug1: str, drug2: str) -> Optional[DrugInteraction]:
    """Check for interaction between two drugs"""
    d1 = normalize_drug_name(drug1)
    d2 = normalize_drug_name(drug2)
    
    for interaction in INTERACTIONS_DATABASE:
        i1 = normalize_drug_name(interaction.drug1)
        i2 = normalize_drug_name(interaction.drug2)
        
        # Direct match
        if (d1 == i1 and d2 == i2) or (d1 == i2 and d2 == i1):
            return interaction
        
        # Partial match
        if ((i1 in d1 or d1 in i1) and (i2 in d2 or d2 in i2)) or \
           ((i1 in d2 or d2 in i1) and (i2 in d1 or d1 in i2)):
            return interaction
    
    return None


def check_allergy(drug_name: str, allergies: List[str]) -> Optional[AllergyAlert]:
    """Check if drug may cause allergic reaction"""
    normalized = normalize_drug_name(drug_name)
    drug_classes = get_drug_classes(drug_name)
    
    for allergy in allergies:
        allergy_lower = allergy.lower().strip()
        
        # Direct drug allergy
        if allergy_lower in normalized or normalized in allergy_lower:
            if normalized in ALLERGY_DATABASE:
                return ALLERGY_DATABASE[normalized]
            return AllergyAlert(drug_name, allergy, False, [])
        
        # Check class allergy
        if allergy_lower in ALLERGY_DATABASE:
            alert_info = ALLERGY_DATABASE[allergy_lower]
            
            # Check if prescribed drug belongs to allergen class
            if alert_info.cross_reactivity:
                # Check drug classes
                if allergy_lower == 'penicillin':
                    if 'penicillin' in drug_classes or 'cephalosporin' in drug_classes:
                        return AllergyAlert(drug_name, allergy, True, alert_info.alternatives)
                elif allergy_lower in ['sulfa', 'nsaid']:
                    if allergy_lower in drug_classes:
                        return AllergyAlert(drug_name, allergy, True, alert_info.alternatives)
    
    return None


def find_all_interactions(medications: List[str]) -> List[DrugInteraction]:
    """Find all interactions in a list of medications"""
    interactions = []
    
    for i in range(len(medications)):
        for j in range(i + 1, len(medications)):
            interaction = check_interactions(medications[i], medications[j])
            if interaction:
                interactions.append(interaction)
    
    # Sort by severity
    severity_order = {
        Severity.CONTRAINDICATED: 0,
        Severity.MAJOR: 1,
        Severity.MODERATE: 2,
        Severity.MINOR: 3
    }
    interactions.sort(key=lambda x: severity_order[x.severity])
    
    return interactions


def find_allergy_alerts(medications: List[str], allergies: List[str]) -> List[AllergyAlert]:
    """Find all allergy alerts for medications"""
    alerts = []
    
    for med in medications:
        alert = check_allergy(med, allergies)
        if alert:
            alerts.append(alert)
    
    return alerts
