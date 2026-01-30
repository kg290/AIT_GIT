"""
Clinical Decision Support Service - AI-powered clinical intelligence for doctors

Features:
1. Evidence-based treatment alternatives
2. Medical guideline compliance scoring
3. Pharmacogenomics alerts
4. Patient-specific treatment optimization
5. Contraindication-based alternatives
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class GuidelineSource(Enum):
    """Medical guideline sources"""
    AHA = "American Heart Association"
    ACC = "American College of Cardiology"
    ADA = "American Diabetes Association"
    WHO = "World Health Organization"
    NICE = "National Institute for Health and Care Excellence"
    ESC = "European Society of Cardiology"
    IDSA = "Infectious Diseases Society of America"
    GOLD = "Global Initiative for Chronic Obstructive Lung Disease"


class EvidenceLevel(Enum):
    """Evidence levels for recommendations"""
    LEVEL_A = "A"  # Multiple RCTs or meta-analyses
    LEVEL_B = "B"  # Single RCT or large observational studies
    LEVEL_C = "C"  # Expert consensus or small studies
    LEVEL_D = "D"  # Expert opinion only


class RecommendationStrength(Enum):
    """Recommendation strength classes"""
    CLASS_I = "I"      # Benefit >>> Risk - Should be performed
    CLASS_IIA = "IIa"  # Benefit >> Risk - Reasonable to perform
    CLASS_IIB = "IIb"  # Benefit >= Risk - May be considered
    CLASS_III = "III"  # Risk >= Benefit - Should not be performed


@dataclass
class TreatmentAlternative:
    """Evidence-based treatment alternative"""
    current_drug: str
    alternative_drug: str
    reason: str
    evidence_level: str
    guideline_source: str
    benefit_summary: str
    considerations: List[str] = field(default_factory=list)
    patient_criteria: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    cost_comparison: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GuidelineCompliance:
    """Guideline compliance assessment"""
    overall_score: float  # 0-100
    guideline_source: str
    guideline_version: str
    compliant_items: List[Dict] = field(default_factory=list)
    non_compliant_items: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    gaps: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PharmacogenomicAlert:
    """Pharmacogenomic consideration"""
    drug: str
    gene: str
    phenotype: str
    clinical_implication: str
    recommendation: str
    alternative_drugs: List[str] = field(default_factory=list)
    dosing_adjustment: Optional[str] = None
    evidence_level: str = "B"
    source: str = "CPIC Guidelines"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ClinicalDecisionReport:
    """Complete clinical decision support report"""
    patient_id: str
    generated_at: str
    alternatives: List[TreatmentAlternative] = field(default_factory=list)
    guideline_compliance: Optional[GuidelineCompliance] = None
    pharmacogenomic_alerts: List[PharmacogenomicAlert] = field(default_factory=list)
    optimization_suggestions: List[Dict] = field(default_factory=list)
    risk_benefit_analysis: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.guideline_compliance:
            result['guideline_compliance'] = self.guideline_compliance.to_dict()
        return result


class ClinicalDecisionSupportService:
    """
    AI-powered Clinical Decision Support System
    
    Provides:
    - Evidence-based treatment alternatives
    - Guideline compliance scoring
    - Pharmacogenomic alerts
    - Treatment optimization suggestions
    """
    
    def __init__(self):
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load clinical knowledge bases"""
        
        # ============ TREATMENT ALTERNATIVES DATABASE ============
        # Format: current_drug -> list of alternatives with conditions
        self.treatment_alternatives = {
            # Diabetes - More specific alternatives based on patient profile
            "metformin": [
                {
                    "alternative": "Empagliflozin (SGLT2i)",
                    "reason": "Superior cardiovascular and renal outcomes",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ADA.value,
                    "benefit": "Reduces cardiovascular death by 38%, heart failure hospitalization by 35%",
                    "criteria": ["established cardiovascular disease", "heart failure", "CKD stage 2-3"],
                    "contraindications": ["eGFR < 20", "recurrent UTI", "DKA history"]
                },
                {
                    "alternative": "Liraglutide (GLP-1 RA)",
                    "reason": "Weight reduction + cardiovascular benefit",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ADA.value,
                    "benefit": "13% reduction in MACE, significant weight loss (5-10%)",
                    "criteria": ["obesity BMI > 30", "atherosclerotic CVD", "high cardiovascular risk"],
                    "contraindications": ["MTC history", "MEN2 syndrome", "pancreatitis history"]
                },
                {
                    "alternative": "Dapagliflozin (SGLT2i)",
                    "reason": "Heart failure and renal protection",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ESC.value,
                    "benefit": "Reduces HF hospitalization by 30%, slows CKD progression",
                    "criteria": ["heart failure HFrEF", "CKD with albuminuria"],
                    "contraindications": ["eGFR < 25", "type 1 diabetes"]
                }
            ],
            
            # Hypertension alternatives
            "amlodipine": [
                {
                    "alternative": "Sacubitril/Valsartan",
                    "reason": "Superior outcomes in HFrEF patients",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "20% reduction in cardiovascular death and HF hospitalization",
                    "criteria": ["heart failure with reduced EF < 40%", "NYHA Class II-IV"],
                    "contraindications": ["angioedema history", "pregnancy", "bilateral renal artery stenosis"]
                },
                {
                    "alternative": "Chlorthalidone",
                    "reason": "Better 24-hour BP control, outcome data",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": GuidelineSource.AHA.value,
                    "benefit": "More potent diuretic effect, proven stroke reduction",
                    "criteria": ["resistant hypertension", "volume overload", "elderly patients"],
                    "contraindications": ["severe hypokalemia", "symptomatic hyperuricemia"]
                }
            ],
            
            "lisinopril": [
                {
                    "alternative": "Sacubitril/Valsartan (Entresto)",
                    "reason": "Superior outcomes in heart failure",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "20% additional reduction in CV death vs ACE inhibitor",
                    "criteria": ["HFrEF EF < 40%", "stable on ACE/ARB"],
                    "contraindications": ["angioedema history", "concomitant ACEi"]
                },
                {
                    "alternative": "Losartan",
                    "reason": "ARB alternative if ACE intolerant (cough)",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.AHA.value,
                    "benefit": "Similar efficacy without bradykinin-mediated cough",
                    "criteria": ["ACE inhibitor cough", "angioedema with ACEi"],
                    "contraindications": ["pregnancy", "hyperkalemia"]
                }
            ],
            
            # Antiplatelet alternatives
            "clopidogrel": [
                {
                    "alternative": "Ticagrelor",
                    "reason": "Faster onset, more consistent platelet inhibition",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "16% reduction in CV death/MI/stroke in ACS",
                    "criteria": ["acute coronary syndrome", "PCI planned/performed"],
                    "contraindications": ["prior ICH", "active bleeding", "severe hepatic impairment"]
                },
                {
                    "alternative": "Prasugrel",
                    "reason": "More potent in CYP2C19 poor metabolizers",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "19% reduction in CV events in ACS with PCI",
                    "criteria": ["PCI for ACS", "CYP2C19 poor metabolizer", "stent thrombosis risk"],
                    "contraindications": ["age > 75", "weight < 60kg", "prior TIA/stroke"]
                }
            ],
            
            # Statins
            "atorvastatin": [
                {
                    "alternative": "Rosuvastatin",
                    "reason": "More potent LDL reduction per mg",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "5-10% additional LDL lowering at equivalent doses",
                    "criteria": ["LDL not at goal on atorvastatin", "high-intensity statin needed"],
                    "contraindications": ["Asian ethnicity (start lower dose)", "severe renal impairment"]
                },
                {
                    "alternative": "Atorvastatin + Ezetimibe",
                    "reason": "Additional LDL lowering without increasing statin dose",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "Additional 15-20% LDL reduction, reduced CV events",
                    "criteria": ["LDL not at goal", "statin intolerance at higher doses"],
                    "contraindications": ["active liver disease"]
                },
                {
                    "alternative": "Add PCSK9 inhibitor (Evolocumab)",
                    "reason": "Dramatic LDL reduction for very high risk",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "60% additional LDL reduction, proven CV benefit",
                    "criteria": ["familial hypercholesterolemia", "ASCVD with LDL > 70 on max statin"],
                    "contraindications": ["cost/access limitations"]
                }
            ],
            
            # Antibiotics
            "amoxicillin": [
                {
                    "alternative": "Amoxicillin-Clavulanate",
                    "reason": "Broader coverage including beta-lactamase producers",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": GuidelineSource.IDSA.value,
                    "benefit": "Covers H. influenzae and M. catarrhalis in respiratory infections",
                    "criteria": ["treatment failure with amoxicillin", "sinusitis", "animal bites"],
                    "contraindications": ["penicillin allergy", "cholestatic jaundice history"]
                },
                {
                    "alternative": "Azithromycin",
                    "reason": "Atypical coverage, shorter course",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": GuidelineSource.IDSA.value,
                    "benefit": "Covers atypicals (Mycoplasma, Chlamydia), 3-5 day course",
                    "criteria": ["community-acquired pneumonia", "penicillin allergy"],
                    "contraindications": ["QT prolongation", "macrolide allergy"]
                }
            ],
            
            # PPIs
            "omeprazole": [
                {
                    "alternative": "Esomeprazole",
                    "reason": "S-isomer with more predictable metabolism",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": GuidelineSource.WHO.value,
                    "benefit": "Fewer CYP2C19 drug interactions",
                    "criteria": ["on clopidogrel", "CYP2C19 poor metabolizer"],
                    "contraindications": ["same as omeprazole"]
                },
                {
                    "alternative": "H2 blocker (Famotidine)",
                    "reason": "Step-down therapy, fewer long-term risks",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": GuidelineSource.WHO.value,
                    "benefit": "Lower risk of C. diff, fractures, B12 deficiency",
                    "criteria": ["mild GERD", "long-term PPI use > 8 weeks", "CDI history"],
                    "contraindications": ["severe erosive esophagitis"]
                }
            ],
            
            # Opioids
            "tramadol": [
                {
                    "alternative": "Duloxetine",
                    "reason": "Non-opioid chronic pain management",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.WHO.value,
                    "benefit": "Effective for neuropathic pain without opioid risks",
                    "criteria": ["diabetic neuropathy", "fibromyalgia", "chronic musculoskeletal pain"],
                    "contraindications": ["MAO inhibitor use", "uncontrolled glaucoma"]
                },
                {
                    "alternative": "Gabapentin",
                    "reason": "Neuropathic pain without addiction risk",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.NICE.value,
                    "benefit": "First-line for neuropathic pain per guidelines",
                    "criteria": ["postherpetic neuralgia", "diabetic neuropathy"],
                    "contraindications": ["respiratory depression risk", "renal impairment (adjust dose)"]
                }
            ],
            
            # Anticoagulants
            "warfarin": [
                {
                    "alternative": "Apixaban",
                    "reason": "Safer profile, no INR monitoring needed",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "31% reduction in stroke, 69% reduction in ICH vs warfarin",
                    "criteria": ["non-valvular AF", "VTE treatment", "patient preference"],
                    "contraindications": ["mechanical heart valve", "severe renal impairment CrCl < 25"]
                },
                {
                    "alternative": "Rivaroxaban",
                    "reason": "Once daily dosing, no monitoring",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": GuidelineSource.ACC.value,
                    "benefit": "Convenient dosing, similar efficacy to warfarin",
                    "criteria": ["AF", "VTE", "preference for once daily"],
                    "contraindications": ["CrCl < 15", "moderate-severe hepatic impairment"]
                }
            ]
        }
        
        # ============ MEDICAL GUIDELINES DATABASE ============
        self.guidelines = {
            "diabetes_type2": {
                "source": GuidelineSource.ADA.value,
                "version": "ADA Standards of Care 2025",
                "requirements": [
                    {"item": "Metformin first-line unless contraindicated", "weight": 15},
                    {"item": "SGLT2i or GLP-1 RA if ASCVD/HF/CKD", "weight": 20},
                    {"item": "A1C target individualized (typically < 7%)", "weight": 10},
                    {"item": "Annual nephropathy screening (uACR)", "weight": 10},
                    {"item": "Annual retinopathy screening", "weight": 10},
                    {"item": "Statin therapy for ages 40-75", "weight": 15},
                    {"item": "Blood pressure target < 130/80", "weight": 10},
                    {"item": "Aspirin if high CV risk", "weight": 10}
                ]
            },
            "hypertension": {
                "source": GuidelineSource.AHA.value,
                "version": "AHA/ACC 2024 Guidelines",
                "requirements": [
                    {"item": "ACEi/ARB for diabetes or CKD", "weight": 20},
                    {"item": "Thiazide or CCB for uncomplicated HTN", "weight": 15},
                    {"item": "Beta-blocker if prior MI or HFrEF", "weight": 15},
                    {"item": "BP target < 130/80 for high risk", "weight": 20},
                    {"item": "Home BP monitoring recommended", "weight": 10},
                    {"item": "Lifestyle modifications counseled", "weight": 10},
                    {"item": "Assess for secondary causes if resistant", "weight": 10}
                ]
            },
            "heart_failure_hfref": {
                "source": GuidelineSource.ACC.value,
                "version": "ACC/AHA 2024 HF Guidelines",
                "requirements": [
                    {"item": "ACEi/ARB/ARNI (Entresto preferred)", "weight": 20},
                    {"item": "Beta-blocker (carvedilol/metoprolol/bisoprolol)", "weight": 20},
                    {"item": "MRA (spironolactone/eplerenone)", "weight": 15},
                    {"item": "SGLT2i (dapagliflozin/empagliflozin)", "weight": 15},
                    {"item": "Loop diuretic for volume management", "weight": 10},
                    {"item": "ICD if EF ≤ 35% after 3 months GDMT", "weight": 10},
                    {"item": "Cardiac rehab referral", "weight": 10}
                ]
            },
            "atrial_fibrillation": {
                "source": GuidelineSource.ACC.value,
                "version": "ACC/AHA 2024 AF Guidelines",
                "requirements": [
                    {"item": "CHA2DS2-VASc assessment", "weight": 15},
                    {"item": "Anticoagulation if score ≥ 2 (men) or ≥ 3 (women)", "weight": 25},
                    {"item": "DOAC preferred over warfarin", "weight": 15},
                    {"item": "Rate control target < 110 bpm at rest", "weight": 15},
                    {"item": "Assess for modifiable risk factors", "weight": 10},
                    {"item": "Discuss rhythm vs rate control strategy", "weight": 10},
                    {"item": "HAS-BLED bleeding risk assessment", "weight": 10}
                ]
            },
            "copd": {
                "source": GuidelineSource.GOLD.value,
                "version": "GOLD 2025 Report",
                "requirements": [
                    {"item": "Inhaled bronchodilator (LAMA or LABA)", "weight": 20},
                    {"item": "Add ICS if frequent exacerbations + eosinophils > 300", "weight": 15},
                    {"item": "Smoking cessation counseling", "weight": 20},
                    {"item": "Pulmonary rehabilitation referral", "weight": 15},
                    {"item": "Annual influenza vaccination", "weight": 10},
                    {"item": "Pneumococcal vaccination", "weight": 10},
                    {"item": "Rescue SABA prescribed", "weight": 10}
                ]
            },
            "acute_coronary_syndrome": {
                "source": GuidelineSource.ACC.value,
                "version": "ACC/AHA 2024 NSTE-ACS Guidelines",
                "requirements": [
                    {"item": "Dual antiplatelet therapy (DAPT)", "weight": 20},
                    {"item": "High-intensity statin", "weight": 15},
                    {"item": "ACEi/ARB if EF < 40% or HTN", "weight": 15},
                    {"item": "Beta-blocker within 24 hours", "weight": 15},
                    {"item": "P2Y12 inhibitor (ticagrelor preferred for ACS)", "weight": 15},
                    {"item": "Anticoagulation during hospitalization", "weight": 10},
                    {"item": "Cardiac rehab referral at discharge", "weight": 10}
                ]
            }
        }
        
        # ============ PHARMACOGENOMICS DATABASE ============
        self.pharmacogenomics = {
            "clopidogrel": [
                {
                    "gene": "CYP2C19",
                    "phenotype": "Poor Metabolizer (*2/*2, *2/*3, *3/*3)",
                    "implication": "Reduced conversion to active metabolite, diminished antiplatelet effect",
                    "recommendation": "Consider prasugrel or ticagrelor as alternatives",
                    "alternatives": ["Prasugrel", "Ticagrelor"],
                    "dosing": None,
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2022"
                },
                {
                    "gene": "CYP2C19",
                    "phenotype": "Intermediate Metabolizer (*1/*2, *1/*3)",
                    "implication": "Reduced active metabolite formation",
                    "recommendation": "Alternative antiplatelet therapy if high risk (ACS, PCI)",
                    "alternatives": ["Prasugrel", "Ticagrelor"],
                    "dosing": None,
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": "CPIC Guideline 2022"
                }
            ],
            "warfarin": [
                {
                    "gene": "CYP2C9",
                    "phenotype": "Poor Metabolizer (*2/*2, *2/*3, *3/*3)",
                    "implication": "Reduced warfarin metabolism, higher bleeding risk",
                    "recommendation": "Reduce initial dose by 50-80%, more frequent INR monitoring",
                    "alternatives": ["Apixaban", "Rivaroxaban"],
                    "dosing": "Start 2-3 mg daily instead of 5 mg",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2022"
                },
                {
                    "gene": "VKORC1",
                    "phenotype": "-1639 G>A (AA genotype)",
                    "implication": "Increased warfarin sensitivity",
                    "recommendation": "Reduce dose by 25-50%",
                    "alternatives": ["Consider DOAC if appropriate"],
                    "dosing": "Start 2-3 mg daily",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2022"
                }
            ],
            "codeine": [
                {
                    "gene": "CYP2D6",
                    "phenotype": "Ultra-rapid Metabolizer (gene duplications)",
                    "implication": "Rapid conversion to morphine, risk of toxicity",
                    "recommendation": "AVOID codeine - use non-tramadol alternatives",
                    "alternatives": ["Morphine (adjusted dose)", "Hydromorphone", "Non-opioid analgesics"],
                    "dosing": "Contraindicated",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2021"
                },
                {
                    "gene": "CYP2D6",
                    "phenotype": "Poor Metabolizer (*4/*4, *5/*5)",
                    "implication": "Insufficient morphine formation, lack of efficacy",
                    "recommendation": "Use alternative analgesic",
                    "alternatives": ["Morphine", "Hydromorphone", "Non-opioid analgesics"],
                    "dosing": "Avoid - will be ineffective",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2021"
                }
            ],
            "tramadol": [
                {
                    "gene": "CYP2D6",
                    "phenotype": "Ultra-rapid Metabolizer",
                    "implication": "Increased active metabolite, toxicity risk",
                    "recommendation": "Avoid tramadol, use alternatives",
                    "alternatives": ["Non-opioid analgesics", "Gabapentin for neuropathic pain"],
                    "dosing": "Contraindicated",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": "CPIC Guideline 2021"
                }
            ],
            "simvastatin": [
                {
                    "gene": "SLCO1B1",
                    "phenotype": "Poor Function (521 CC genotype)",
                    "implication": "17-fold increased myopathy risk at 80mg dose",
                    "recommendation": "Avoid simvastatin 80mg, consider alternative statin",
                    "alternatives": ["Atorvastatin", "Rosuvastatin", "Pravastatin"],
                    "dosing": "Maximum 20mg daily if simvastatin used",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2022"
                }
            ],
            "omeprazole": [
                {
                    "gene": "CYP2C19",
                    "phenotype": "Ultra-rapid Metabolizer (*17/*17)",
                    "implication": "Reduced efficacy for H. pylori eradication",
                    "recommendation": "Increase dose or use alternative PPI",
                    "alternatives": ["Esomeprazole", "Rabeprazole"],
                    "dosing": "Double standard dose",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": "CPIC Guideline 2020"
                }
            ],
            "fluorouracil": [
                {
                    "gene": "DPYD",
                    "phenotype": "Poor Metabolizer (*2A, *13)",
                    "implication": "Severe/fatal toxicity risk",
                    "recommendation": "Reduce dose by 50% or use alternative",
                    "alternatives": ["Consider alternative chemotherapy regimen"],
                    "dosing": "50% dose reduction mandatory",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2023"
                }
            ],
            "carbamazepine": [
                {
                    "gene": "HLA-B*15:02",
                    "phenotype": "Positive carrier (common in Asian ancestry)",
                    "implication": "High risk of Stevens-Johnson syndrome/TEN",
                    "recommendation": "AVOID carbamazepine - use alternative anticonvulsant",
                    "alternatives": ["Levetiracetam", "Valproic acid", "Lamotrigine (with caution)"],
                    "dosing": "Contraindicated",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2022"
                }
            ],
            "allopurinol": [
                {
                    "gene": "HLA-B*58:01",
                    "phenotype": "Positive carrier (higher in Asian, African ancestry)",
                    "implication": "High risk of severe cutaneous adverse reactions",
                    "recommendation": "Consider febuxostat as alternative",
                    "alternatives": ["Febuxostat", "Probenecid"],
                    "dosing": "Start very low dose (50mg) if used",
                    "evidence": EvidenceLevel.LEVEL_A.value,
                    "source": "CPIC Guideline 2022"
                }
            ],
            "metoprolol": [
                {
                    "gene": "CYP2D6",
                    "phenotype": "Ultra-rapid Metabolizer",
                    "implication": "Reduced drug levels, may need higher doses",
                    "recommendation": "Consider alternative beta-blocker or higher doses",
                    "alternatives": ["Bisoprolol", "Carvedilol"],
                    "dosing": "May need doses above typical maximum",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": "DPWG Guideline 2022"
                },
                {
                    "gene": "CYP2D6",
                    "phenotype": "Poor Metabolizer (*4/*4)",
                    "implication": "Increased drug levels, risk of bradycardia/hypotension",
                    "recommendation": "Reduce dose by 50% or use alternative",
                    "alternatives": ["Bisoprolol", "Atenolol"],
                    "dosing": "50% of standard dose",
                    "evidence": EvidenceLevel.LEVEL_B.value,
                    "source": "DPWG Guideline 2022"
                }
            ]
        }
        
        # ============ CONDITION TO GUIDELINE MAPPING ============
        self.condition_guideline_map = {
            "diabetes": "diabetes_type2",
            "type 2 diabetes": "diabetes_type2",
            "dm2": "diabetes_type2",
            "t2dm": "diabetes_type2",
            "hypertension": "hypertension",
            "htn": "hypertension",
            "high blood pressure": "hypertension",
            "heart failure": "heart_failure_hfref",
            "hf": "heart_failure_hfref",
            "chf": "heart_failure_hfref",
            "hfref": "heart_failure_hfref",
            "atrial fibrillation": "atrial_fibrillation",
            "afib": "atrial_fibrillation",
            "af": "atrial_fibrillation",
            "copd": "copd",
            "acs": "acute_coronary_syndrome",
            "nstemi": "acute_coronary_syndrome",
            "stemi": "acute_coronary_syndrome",
            "unstable angina": "acute_coronary_syndrome",
            "myocardial infarction": "acute_coronary_syndrome",
            "mi": "acute_coronary_syndrome"
        }
        
        # ============ DRUG TO CLASS MAPPING ============
        self.drug_classes = {
            "metformin": "biguanide",
            "empagliflozin": "sglt2i",
            "dapagliflozin": "sglt2i",
            "canagliflozin": "sglt2i",
            "liraglutide": "glp1ra",
            "semaglutide": "glp1ra",
            "dulaglutide": "glp1ra",
            "sitagliptin": "dpp4i",
            "linagliptin": "dpp4i",
            "glipizide": "sulfonylurea",
            "glimepiride": "sulfonylurea",
            "pioglitazone": "tzd",
            "lisinopril": "acei",
            "enalapril": "acei",
            "ramipril": "acei",
            "losartan": "arb",
            "valsartan": "arb",
            "telmisartan": "arb",
            "sacubitril/valsartan": "arni",
            "amlodipine": "ccb",
            "nifedipine": "ccb",
            "diltiazem": "ccb",
            "metoprolol": "bb",
            "carvedilol": "bb",
            "bisoprolol": "bb",
            "atenolol": "bb",
            "hydrochlorothiazide": "thiazide",
            "chlorthalidone": "thiazide",
            "furosemide": "loop",
            "spironolactone": "mra",
            "eplerenone": "mra",
            "atorvastatin": "statin",
            "rosuvastatin": "statin",
            "simvastatin": "statin",
            "pravastatin": "statin",
            "aspirin": "antiplatelet",
            "clopidogrel": "antiplatelet",
            "ticagrelor": "antiplatelet",
            "prasugrel": "antiplatelet",
            "warfarin": "vka",
            "apixaban": "doac",
            "rivaroxaban": "doac",
            "dabigatran": "doac",
            "edoxaban": "doac"
        }
    
    def get_treatment_alternatives(
        self,
        medications: List[str],
        conditions: List[str] = None,
        patient_profile: Dict = None
    ) -> List[TreatmentAlternative]:
        """
        Get evidence-based treatment alternatives
        
        Args:
            medications: Current medications
            conditions: Patient conditions for context
            patient_profile: Additional patient info (age, comorbidities, etc.)
            
        Returns:
            List of TreatmentAlternative objects
        """
        alternatives = []
        conditions = conditions or []
        patient_profile = patient_profile or {}
        
        for med in medications:
            med_lower = med.lower().strip()
            
            # Check if we have alternatives for this drug
            for drug_key, alt_list in self.treatment_alternatives.items():
                if drug_key in med_lower or med_lower in drug_key:
                    for alt in alt_list:
                        # Check if patient meets criteria
                        criteria_met = self._check_patient_criteria(
                            alt.get('criteria', []),
                            conditions,
                            patient_profile
                        )
                        
                        # Check contraindications
                        has_contraindication = self._check_contraindications(
                            alt.get('contraindications', []),
                            conditions,
                            patient_profile
                        )
                        
                        if criteria_met and not has_contraindication:
                            alternatives.append(TreatmentAlternative(
                                current_drug=med,
                                alternative_drug=alt['alternative'],
                                reason=alt['reason'],
                                evidence_level=alt['evidence'],
                                guideline_source=alt['source'],
                                benefit_summary=alt['benefit'],
                                considerations=alt.get('criteria', []),
                                patient_criteria=alt.get('criteria', []),
                                contraindications=alt.get('contraindications', []),
                                cost_comparison=alt.get('cost_comparison')
                            ))
        
        return alternatives
    
    def _check_patient_criteria(
        self,
        criteria: List[str],
        conditions: List[str],
        profile: Dict
    ) -> bool:
        """Check if patient meets criteria for alternative"""
        if not criteria:
            return True  # No specific criteria
        
        conditions_lower = [c.lower() for c in conditions]
        
        for criterion in criteria:
            criterion_lower = criterion.lower()
            
            # Check conditions
            for cond in conditions_lower:
                if any(keyword in cond for keyword in criterion_lower.split()):
                    return True
            
            # Check profile
            if 'age' in criterion_lower and 'age' in profile:
                return True
            if 'bmi' in criterion_lower and (profile.get('bmi') or 0) > 30:
                return True
        
        return len(criteria) == 0
    
    def _check_contraindications(
        self,
        contraindications: List[str],
        conditions: List[str],
        profile: Dict
    ) -> bool:
        """Check if patient has contraindications"""
        if not contraindications:
            return False
        
        conditions_lower = [c.lower() for c in conditions]
        
        for contra in contraindications:
            contra_lower = contra.lower()
            for cond in conditions_lower:
                if any(keyword in cond for keyword in contra_lower.split()):
                    return True
        
        # Check pregnancy
        if 'pregnancy' in [c.lower() for c in contraindications]:
            if profile.get('pregnant') or profile.get('gender', '').lower() == 'female' and (profile.get('age') or 50) < 50:
                pass  # Could be relevant but don't auto-exclude
        
        return False
    
    def assess_guideline_compliance(
        self,
        medications: List[str],
        conditions: List[str],
        patient_profile: Dict = None
    ) -> List[GuidelineCompliance]:
        """
        Assess prescription compliance with medical guidelines
        
        Args:
            medications: Current medications
            conditions: Patient conditions
            patient_profile: Additional patient info
            
        Returns:
            List of GuidelineCompliance assessments
        """
        assessments = []
        patient_profile = patient_profile or {}
        
        # Map conditions to guidelines
        for condition in conditions:
            condition_lower = condition.lower().strip()
            
            # Find matching guideline
            guideline_key = None
            for key, value in self.condition_guideline_map.items():
                if key in condition_lower:
                    guideline_key = value
                    break
            
            if guideline_key and guideline_key in self.guidelines:
                guideline = self.guidelines[guideline_key]
                assessment = self._assess_single_guideline(
                    guideline_key,
                    guideline,
                    medications,
                    patient_profile
                )
                assessments.append(assessment)
        
        return assessments
    
    def _assess_single_guideline(
        self,
        guideline_key: str,
        guideline: Dict,
        medications: List[str],
        profile: Dict
    ) -> GuidelineCompliance:
        """Assess compliance with a single guideline"""
        compliant = []
        non_compliant = []
        recommendations = []
        total_weight = 0
        achieved_weight = 0
        
        med_classes = set()
        for med in medications:
            med_lower = med.lower()
            for drug, drug_class in self.drug_classes.items():
                if drug in med_lower:
                    med_classes.add(drug_class)
        
        for req in guideline['requirements']:
            item = req['item']
            weight = req['weight']
            total_weight += weight
            
            met = self._check_requirement_met(item, medications, med_classes, profile)
            
            if met:
                compliant.append({
                    'item': item,
                    'weight': weight,
                    'status': 'compliant'
                })
                achieved_weight += weight
            else:
                non_compliant.append({
                    'item': item,
                    'weight': weight,
                    'status': 'non-compliant'
                })
                recommendations.append(f"Consider: {item}")
        
        score = (achieved_weight / total_weight * 100) if total_weight > 0 else 0
        
        return GuidelineCompliance(
            overall_score=round(score, 1),
            guideline_source=guideline['source'],
            guideline_version=guideline['version'],
            compliant_items=compliant,
            non_compliant_items=non_compliant,
            recommendations=recommendations,
            gaps=[{'item': nc['item'], 'priority': 'high' if nc['weight'] >= 15 else 'medium'} 
                  for nc in non_compliant]
        )
    
    def _check_requirement_met(
        self,
        requirement: str,
        medications: List[str],
        med_classes: set,
        profile: Dict
    ) -> bool:
        """Check if a guideline requirement is met"""
        req_lower = requirement.lower()
        
        # Drug class checks
        class_checks = {
            'metformin': 'biguanide' in med_classes or any('metformin' in m.lower() for m in medications),
            'sglt2': 'sglt2i' in med_classes,
            'glp-1': 'glp1ra' in med_classes,
            'acei': 'acei' in med_classes,
            'arb': 'arb' in med_classes or 'arni' in med_classes,
            'arni': 'arni' in med_classes,
            'beta-blocker': 'bb' in med_classes,
            'statin': 'statin' in med_classes,
            'mra': 'mra' in med_classes,
            'spironolactone': 'mra' in med_classes or any('spironolactone' in m.lower() for m in medications),
            'anticoagula': 'doac' in med_classes or 'vka' in med_classes,
            'doac': 'doac' in med_classes,
            'aspirin': any('aspirin' in m.lower() for m in medications),
            'antiplatelet': 'antiplatelet' in med_classes,
            'dapt': len([c for c in ['antiplatelet'] if c in med_classes]) >= 1 and any('aspirin' in m.lower() for m in medications),
            'thiazide': 'thiazide' in med_classes,
            'ccb': 'ccb' in med_classes,
            'diuretic': 'loop' in med_classes or 'thiazide' in med_classes
        }
        
        for check_term, result in class_checks.items():
            if check_term in req_lower and result:
                return True
        
        return False
    
    def get_pharmacogenomic_alerts(
        self,
        medications: List[str],
        genetic_data: Dict = None
    ) -> List[PharmacogenomicAlert]:
        """
        Get pharmacogenomic alerts for medications
        
        Args:
            medications: Current medications
            genetic_data: Known genetic data (gene: phenotype)
            
        Returns:
            List of PharmacogenomicAlert objects
        """
        alerts = []
        genetic_data = genetic_data or {}
        
        for med in medications:
            med_lower = med.lower().strip()
            
            # Check each drug in pharmacogenomics database
            for drug_key, pgx_list in self.pharmacogenomics.items():
                if drug_key in med_lower:
                    for pgx in pgx_list:
                        # If we have genetic data, check if patient has this variant
                        gene = pgx['gene']
                        if gene in genetic_data:
                            patient_phenotype = genetic_data[gene]
                            if pgx['phenotype'].lower() in patient_phenotype.lower():
                                alerts.append(PharmacogenomicAlert(
                                    drug=med,
                                    gene=gene,
                                    phenotype=pgx['phenotype'],
                                    clinical_implication=pgx['implication'],
                                    recommendation=pgx['recommendation'],
                                    alternative_drugs=pgx['alternatives'],
                                    dosing_adjustment=pgx.get('dosing'),
                                    evidence_level=pgx['evidence'],
                                    source=pgx['source']
                                ))
                        else:
                            # No genetic data - still provide as informational
                            alerts.append(PharmacogenomicAlert(
                                drug=med,
                                gene=gene,
                                phenotype=f"Consider testing: {pgx['phenotype']}",
                                clinical_implication=pgx['implication'],
                                recommendation=f"If {pgx['phenotype']}: {pgx['recommendation']}",
                                alternative_drugs=pgx['alternatives'],
                                dosing_adjustment=pgx.get('dosing'),
                                evidence_level=pgx['evidence'],
                                source=pgx['source']
                            ))
                    break  # Only process each drug once
        
        return alerts
    
    def get_optimization_suggestions(
        self,
        medications: List[str],
        conditions: List[str],
        patient_profile: Dict = None
    ) -> List[Dict]:
        """
        Get treatment optimization suggestions
        """
        suggestions = []
        patient_profile = patient_profile or {}
        
        med_classes = set()
        for med in medications:
            med_lower = med.lower()
            for drug, drug_class in self.drug_classes.items():
                if drug in med_lower:
                    med_classes.add(drug_class)
        
        conditions_lower = [c.lower() for c in conditions]
        
        # Check for optimization opportunities
        
        # 1. Diabetes with CVD but no SGLT2i/GLP-1 RA
        if any('diabet' in c for c in conditions_lower):
            has_cvd = any(term in ' '.join(conditions_lower) for term in 
                        ['cardiovascular', 'heart', 'coronary', 'stroke', 'pad', 'atheroscler'])
            has_sglt2_or_glp1 = 'sglt2i' in med_classes or 'glp1ra' in med_classes
            
            if has_cvd and not has_sglt2_or_glp1:
                suggestions.append({
                    'type': 'evidence_based_addition',
                    'priority': 'high',
                    'title': 'Add Cardioprotective Agent',
                    'description': 'Patient has diabetes with cardiovascular disease but is not on SGLT2i or GLP-1 RA',
                    'recommendation': 'Consider adding empagliflozin or liraglutide for cardiovascular protection',
                    'evidence': 'ADA 2025 Standards: SGLT2i or GLP-1 RA recommended in T2DM with ASCVD (Class I, Level A)'
                })
        
        # 2. HF with reduced EF but not on optimal therapy
        if any('heart failure' in c or 'hf' in c for c in conditions_lower):
            gdmt_classes = {'acei', 'arb', 'arni', 'bb', 'mra', 'sglt2i'}
            on_gdmt = len(med_classes & gdmt_classes)
            
            if on_gdmt < 4:
                missing = gdmt_classes - med_classes
                suggestions.append({
                    'type': 'guideline_optimization',
                    'priority': 'high',
                    'title': 'Optimize Heart Failure GDMT',
                    'description': f'Patient on {on_gdmt}/4 foundational HF therapies',
                    'recommendation': f'Consider adding: {", ".join(missing)}',
                    'evidence': 'ACC/AHA 2024: Quadruple therapy (ARNI + BB + MRA + SGLT2i) recommended for HFrEF'
                })
        
        # 3. AF with anticoagulation check
        if any('fibrillation' in c or 'afib' in c or ' af' in c for c in conditions_lower):
            on_anticoag = 'doac' in med_classes or 'vka' in med_classes
            
            if not on_anticoag:
                suggestions.append({
                    'type': 'safety_gap',
                    'priority': 'high',
                    'title': 'Anticoagulation Assessment Needed',
                    'description': 'Patient has atrial fibrillation without apparent anticoagulation',
                    'recommendation': 'Calculate CHA₂DS₂-VASc score and consider anticoagulation if ≥2 (men) or ≥3 (women)',
                    'evidence': 'ACC/AHA 2024 AF Guidelines'
                })
            elif 'vka' in med_classes and 'doac' not in med_classes:
                suggestions.append({
                    'type': 'therapy_modernization',
                    'priority': 'medium',
                    'title': 'Consider DOAC Over Warfarin',
                    'description': 'Patient on warfarin for non-valvular AF',
                    'recommendation': 'Consider switching to apixaban (preferred) - better safety profile, no INR monitoring',
                    'evidence': 'DOACs preferred over warfarin for non-valvular AF (Class I, Level A)'
                })
        
        # 4. High-risk patient not on statin
        age = patient_profile.get('age') or 50
        if age >= 40:
            has_statin = 'statin' in med_classes
            high_risk = any(term in ' '.join(conditions_lower) for term in 
                          ['diabet', 'coronary', 'stroke', 'pad', 'mi', 'heart'])
            
            if high_risk and not has_statin:
                suggestions.append({
                    'type': 'prevention_gap',
                    'priority': 'high',
                    'title': 'Statin Therapy Recommended',
                    'description': 'High-risk patient (diabetes/ASCVD) not on statin therapy',
                    'recommendation': 'Initiate high-intensity statin (atorvastatin 40-80mg or rosuvastatin 20-40mg)',
                    'evidence': 'ACC/AHA 2019: High-intensity statin for clinical ASCVD or diabetes age 40-75 (Class I)'
                })
        
        return suggestions
    
    def generate_full_report(
        self,
        patient_id: str,
        medications: List[str],
        conditions: List[str] = None,
        patient_profile: Dict = None,
        genetic_data: Dict = None
    ) -> ClinicalDecisionReport:
        """
        Generate complete clinical decision support report
        """
        conditions = conditions or []
        patient_profile = patient_profile or {}
        
        # Get all components
        alternatives = self.get_treatment_alternatives(medications, conditions, patient_profile)
        guideline_assessments = self.assess_guideline_compliance(medications, conditions, patient_profile)
        pgx_alerts = self.get_pharmacogenomic_alerts(medications, genetic_data)
        optimizations = self.get_optimization_suggestions(medications, conditions, patient_profile)
        
        # Combine guideline assessments
        combined_compliance = None
        if guideline_assessments:
            avg_score = sum(g.overall_score for g in guideline_assessments) / len(guideline_assessments)
            all_compliant = []
            all_non_compliant = []
            all_recommendations = []
            
            for g in guideline_assessments:
                all_compliant.extend(g.compliant_items)
                all_non_compliant.extend(g.non_compliant_items)
                all_recommendations.extend(g.recommendations)
            
            combined_compliance = GuidelineCompliance(
                overall_score=round(avg_score, 1),
                guideline_source="Multiple Guidelines",
                guideline_version="2024-2025",
                compliant_items=all_compliant,
                non_compliant_items=all_non_compliant,
                recommendations=list(set(all_recommendations))
            )
        
        return ClinicalDecisionReport(
            patient_id=patient_id,
            generated_at=datetime.now().isoformat(),
            alternatives=alternatives,
            guideline_compliance=combined_compliance,
            pharmacogenomic_alerts=pgx_alerts,
            optimization_suggestions=optimizations
        )


# Create singleton
clinical_decision_support = ClinicalDecisionSupportService()
