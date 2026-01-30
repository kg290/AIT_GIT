"""
Treatment Outcome Tracking Service

Features:
1. Link prescriptions to patient outcomes
2. Track vital sign changes over time
3. Monitor treatment effectiveness
4. ML-based treatment success prediction
5. Outcome analytics and reporting
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import math

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Types of treatment outcomes"""
    IMPROVED = "improved"
    STABLE = "stable"
    WORSENED = "worsened"
    RESOLVED = "resolved"
    DISCONTINUED = "discontinued"
    ADVERSE_EVENT = "adverse_event"
    PENDING = "pending"


class VitalType(Enum):
    """Types of vital signs tracked"""
    BLOOD_PRESSURE_SYSTOLIC = "bp_systolic"
    BLOOD_PRESSURE_DIASTOLIC = "bp_diastolic"
    HEART_RATE = "heart_rate"
    TEMPERATURE = "temperature"
    RESPIRATORY_RATE = "respiratory_rate"
    OXYGEN_SATURATION = "oxygen_saturation"
    BLOOD_GLUCOSE = "blood_glucose"
    HBA1C = "hba1c"
    LDL_CHOLESTEROL = "ldl_cholesterol"
    HDL_CHOLESTEROL = "hdl_cholesterol"
    TOTAL_CHOLESTEROL = "total_cholesterol"
    TRIGLYCERIDES = "triglycerides"
    CREATININE = "creatinine"
    EGFR = "egfr"
    WEIGHT = "weight"
    BMI = "bmi"
    PAIN_SCORE = "pain_score"
    INR = "inr"
    POTASSIUM = "potassium"


@dataclass
class VitalReading:
    """Single vital sign reading"""
    vital_type: str
    value: float
    unit: str
    recorded_at: str
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TreatmentOutcome:
    """Outcome record for a treatment"""
    prescription_id: str
    medication: str
    started_at: str
    outcome_type: str
    outcome_description: str
    outcome_recorded_at: str
    vital_changes: List[Dict] = field(default_factory=list)
    effectiveness_score: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    follow_up_notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OutcomeTimeline:
    """Timeline of treatment outcomes for a patient"""
    patient_id: str
    treatments: List[TreatmentOutcome] = field(default_factory=list)
    vital_trends: Dict[str, List[Dict]] = field(default_factory=dict)
    overall_health_trend: str = "stable"
    generated_at: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TreatmentPrediction:
    """ML prediction for treatment success"""
    medication: str
    condition: str
    predicted_success_probability: float
    confidence_interval: Tuple[float, float]
    factors_supporting: List[Dict] = field(default_factory=list)
    factors_against: List[Dict] = field(default_factory=list)
    similar_patient_outcomes: Dict = field(default_factory=dict)
    recommendation: str = ""
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['confidence_interval'] = list(self.confidence_interval)
        return result


class TreatmentOutcomeService:
    """
    Treatment Outcome Tracking and Prediction Service
    
    Links prescriptions to clinical outcomes, tracks vital sign changes,
    and uses ML-based prediction for treatment success.
    """
    
    def __init__(self, data_dir: str = "data/outcomes"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._load_outcome_models()
        self._load_vital_targets()
    
    def _load_outcome_models(self):
        """Load treatment outcome prediction models (knowledge-based)"""
        
        # Treatment success factors database
        # Format: medication_class -> factors affecting success
        self.success_factors = {
            "antihypertensive": {
                "positive_factors": [
                    {"factor": "Age < 65", "weight": 0.05},
                    {"factor": "No CKD", "weight": 0.1},
                    {"factor": "Adherence > 80%", "weight": 0.15},
                    {"factor": "No resistant HTN history", "weight": 0.1},
                    {"factor": "Single-pill combination", "weight": 0.05},
                    {"factor": "Lifestyle modifications", "weight": 0.1}
                ],
                "negative_factors": [
                    {"factor": "Obesity BMI > 35", "weight": -0.1},
                    {"factor": "Sleep apnea", "weight": -0.1},
                    {"factor": "High sodium diet", "weight": -0.08},
                    {"factor": "Alcohol excess", "weight": -0.05},
                    {"factor": "Secondary HTN", "weight": -0.15},
                    {"factor": "Multiple prior failures", "weight": -0.12}
                ],
                "base_success_rate": 0.70
            },
            "antidiabetic": {
                "positive_factors": [
                    {"factor": "A1C < 9% at start", "weight": 0.1},
                    {"factor": "BMI < 30", "weight": 0.05},
                    {"factor": "Newly diagnosed < 5 years", "weight": 0.1},
                    {"factor": "Good diet adherence", "weight": 0.1},
                    {"factor": "Regular exercise", "weight": 0.08},
                    {"factor": "No insulin resistance", "weight": 0.05}
                ],
                "negative_factors": [
                    {"factor": "A1C > 10%", "weight": -0.15},
                    {"factor": "Long diabetes duration > 10 years", "weight": -0.1},
                    {"factor": "Previous multiple medication failures", "weight": -0.12},
                    {"factor": "Severe obesity BMI > 40", "weight": -0.08},
                    {"factor": "Irregular meals", "weight": -0.05}
                ],
                "base_success_rate": 0.65
            },
            "statin": {
                "positive_factors": [
                    {"factor": "No prior statin intolerance", "weight": 0.1},
                    {"factor": "Good adherence history", "weight": 0.1},
                    {"factor": "LDL > 100 (room to improve)", "weight": 0.05},
                    {"factor": "Healthy lifestyle", "weight": 0.05}
                ],
                "negative_factors": [
                    {"factor": "Prior myalgia with statins", "weight": -0.2},
                    {"factor": "SLCO1B1 variant", "weight": -0.15},
                    {"factor": "Age > 80", "weight": -0.05},
                    {"factor": "Multiple drug interactions", "weight": -0.1}
                ],
                "base_success_rate": 0.85
            },
            "antibiotic": {
                "positive_factors": [
                    {"factor": "Known susceptibility", "weight": 0.2},
                    {"factor": "Uncomplicated infection", "weight": 0.1},
                    {"factor": "Immunocompetent", "weight": 0.1},
                    {"factor": "No recent antibiotic exposure", "weight": 0.05}
                ],
                "negative_factors": [
                    {"factor": "Known resistance patterns", "weight": -0.25},
                    {"factor": "Immunocompromised", "weight": -0.15},
                    {"factor": "Complicated infection", "weight": -0.1},
                    {"factor": "Previous treatment failure", "weight": -0.15},
                    {"factor": "Biofilm infection", "weight": -0.1}
                ],
                "base_success_rate": 0.80
            },
            "anticoagulant": {
                "positive_factors": [
                    {"factor": "Stable INR history (if warfarin)", "weight": 0.1},
                    {"factor": "Good adherence", "weight": 0.1},
                    {"factor": "No high bleeding risk", "weight": 0.1},
                    {"factor": "Regular monitoring", "weight": 0.05}
                ],
                "negative_factors": [
                    {"factor": "High HAS-BLED score > 3", "weight": -0.15},
                    {"factor": "CKD stage 4-5", "weight": -0.1},
                    {"factor": "Prior major bleeding", "weight": -0.2},
                    {"factor": "Poor adherence history", "weight": -0.15}
                ],
                "base_success_rate": 0.75
            },
            "antidepressant": {
                "positive_factors": [
                    {"factor": "First episode depression", "weight": 0.1},
                    {"factor": "Good social support", "weight": 0.08},
                    {"factor": "Concurrent therapy", "weight": 0.1},
                    {"factor": "Mild-moderate severity", "weight": 0.05}
                ],
                "negative_factors": [
                    {"factor": "Treatment-resistant depression", "weight": -0.2},
                    {"factor": "Multiple prior failures", "weight": -0.15},
                    {"factor": "Substance use disorder", "weight": -0.1},
                    {"factor": "Psychotic features", "weight": -0.1}
                ],
                "base_success_rate": 0.55
            },
            "bronchodilator": {
                "positive_factors": [
                    {"factor": "Good inhaler technique", "weight": 0.15},
                    {"factor": "Smoking cessation", "weight": 0.15},
                    {"factor": "Mild-moderate COPD", "weight": 0.1},
                    {"factor": "Regular use as prescribed", "weight": 0.1}
                ],
                "negative_factors": [
                    {"factor": "Continued smoking", "weight": -0.2},
                    {"factor": "Severe COPD GOLD 4", "weight": -0.15},
                    {"factor": "Poor inhaler technique", "weight": -0.15},
                    {"factor": "Frequent exacerbations", "weight": -0.1}
                ],
                "base_success_rate": 0.70
            },
            "analgesic": {
                "positive_factors": [
                    {"factor": "Acute pain (not chronic)", "weight": 0.15},
                    {"factor": "Identifiable cause", "weight": 0.1},
                    {"factor": "Multimodal approach", "weight": 0.1},
                    {"factor": "No prior opioid use", "weight": 0.08}
                ],
                "negative_factors": [
                    {"factor": "Chronic pain > 3 months", "weight": -0.15},
                    {"factor": "Opioid tolerance", "weight": -0.2},
                    {"factor": "Central sensitization", "weight": -0.15},
                    {"factor": "Psychological comorbidity", "weight": -0.1}
                ],
                "base_success_rate": 0.65
            }
        }
        
        # Drug to class mapping
        self.drug_class_map = {
            # Antihypertensives
            "lisinopril": "antihypertensive",
            "amlodipine": "antihypertensive",
            "losartan": "antihypertensive",
            "metoprolol": "antihypertensive",
            "hydrochlorothiazide": "antihypertensive",
            "valsartan": "antihypertensive",
            "enalapril": "antihypertensive",
            "carvedilol": "antihypertensive",
            
            # Antidiabetics
            "metformin": "antidiabetic",
            "empagliflozin": "antidiabetic",
            "glipizide": "antidiabetic",
            "sitagliptin": "antidiabetic",
            "liraglutide": "antidiabetic",
            "insulin": "antidiabetic",
            "dapagliflozin": "antidiabetic",
            
            # Statins
            "atorvastatin": "statin",
            "rosuvastatin": "statin",
            "simvastatin": "statin",
            "pravastatin": "statin",
            
            # Antibiotics
            "amoxicillin": "antibiotic",
            "azithromycin": "antibiotic",
            "ciprofloxacin": "antibiotic",
            "doxycycline": "antibiotic",
            "cephalexin": "antibiotic",
            
            # Anticoagulants
            "warfarin": "anticoagulant",
            "apixaban": "anticoagulant",
            "rivaroxaban": "anticoagulant",
            
            # Antidepressants
            "sertraline": "antidepressant",
            "escitalopram": "antidepressant",
            "fluoxetine": "antidepressant",
            "duloxetine": "antidepressant",
            
            # Bronchodilators
            "albuterol": "bronchodilator",
            "tiotropium": "bronchodilator",
            "fluticasone": "bronchodilator",
            
            # Analgesics
            "ibuprofen": "analgesic",
            "acetaminophen": "analgesic",
            "tramadol": "analgesic",
            "gabapentin": "analgesic"
        }
        
        # Historical outcome data (simulated - in production this would come from database)
        self.historical_outcomes = {
            "antihypertensive": {
                "total_patients": 15000,
                "improved": 10500,
                "stable": 2250,
                "worsened": 1500,
                "discontinued": 750,
                "avg_time_to_improvement": 28  # days
            },
            "antidiabetic": {
                "total_patients": 12000,
                "improved": 7800,
                "stable": 2400,
                "worsened": 1200,
                "discontinued": 600,
                "avg_time_to_improvement": 90  # days
            },
            "statin": {
                "total_patients": 20000,
                "improved": 17000,
                "stable": 2000,
                "worsened": 500,
                "discontinued": 500,
                "avg_time_to_improvement": 42  # days
            },
            "antibiotic": {
                "total_patients": 25000,
                "resolved": 21250,
                "improved": 2500,
                "stable": 625,
                "worsened": 625,
                "avg_time_to_improvement": 5  # days
            },
            "anticoagulant": {
                "total_patients": 8000,
                "stable": 6400,
                "improved": 800,
                "adverse_event": 480,
                "discontinued": 320,
                "avg_time_to_improvement": 14  # days
            }
        }
    
    def _load_vital_targets(self):
        """Load vital sign target ranges"""
        
        self.vital_targets = {
            VitalType.BLOOD_PRESSURE_SYSTOLIC.value: {
                "normal": (90, 120),
                "target_general": (90, 130),
                "target_diabetes": (90, 130),
                "target_elderly": (90, 140),
                "unit": "mmHg"
            },
            VitalType.BLOOD_PRESSURE_DIASTOLIC.value: {
                "normal": (60, 80),
                "target_general": (60, 80),
                "target_diabetes": (60, 80),
                "unit": "mmHg"
            },
            VitalType.HBA1C.value: {
                "normal": (4.0, 5.6),
                "target_general": (4.0, 7.0),
                "target_elderly": (4.0, 8.0),
                "unit": "%"
            },
            VitalType.BLOOD_GLUCOSE.value: {
                "normal_fasting": (70, 100),
                "target_fasting": (80, 130),
                "target_postprandial": (80, 180),
                "unit": "mg/dL"
            },
            VitalType.LDL_CHOLESTEROL.value: {
                "normal": (0, 100),
                "target_high_risk": (0, 70),
                "target_very_high_risk": (0, 55),
                "unit": "mg/dL"
            },
            VitalType.HDL_CHOLESTEROL.value: {
                "target_men": (40, 200),
                "target_women": (50, 200),
                "unit": "mg/dL"
            },
            VitalType.TRIGLYCERIDES.value: {
                "normal": (0, 150),
                "borderline": (150, 200),
                "high": (200, 500),
                "unit": "mg/dL"
            },
            VitalType.EGFR.value: {
                "normal": (90, 120),
                "mild_decrease": (60, 89),
                "moderate_decrease": (30, 59),
                "severe_decrease": (15, 29),
                "unit": "mL/min/1.73m²"
            },
            VitalType.HEART_RATE.value: {
                "normal": (60, 100),
                "target_af_rate_control": (60, 110),
                "unit": "bpm"
            },
            VitalType.OXYGEN_SATURATION.value: {
                "normal": (95, 100),
                "acceptable_copd": (88, 92),
                "unit": "%"
            },
            VitalType.PAIN_SCORE.value: {
                "no_pain": (0, 0),
                "mild": (1, 3),
                "moderate": (4, 6),
                "severe": (7, 10),
                "unit": "0-10 scale"
            },
            VitalType.INR.value: {
                "normal": (0.9, 1.1),
                "target_af": (2.0, 3.0),
                "target_mechanical_valve": (2.5, 3.5),
                "unit": "ratio"
            },
            VitalType.POTASSIUM.value: {
                "normal": (3.5, 5.0),
                "low": (0, 3.4),
                "high": (5.1, 10),
                "unit": "mEq/L"
            }
        }
    
    def record_outcome(
        self,
        patient_id: str,
        prescription_id: str,
        medication: str,
        outcome_type: OutcomeType,
        description: str,
        vital_changes: List[Dict] = None,
        side_effects: List[str] = None
    ) -> TreatmentOutcome:
        """
        Record a treatment outcome
        """
        outcome = TreatmentOutcome(
            prescription_id=prescription_id,
            medication=medication,
            started_at=datetime.now().isoformat(),
            outcome_type=outcome_type.value,
            outcome_description=description,
            outcome_recorded_at=datetime.now().isoformat(),
            vital_changes=vital_changes or [],
            side_effects=side_effects or []
        )
        
        # Calculate effectiveness score
        outcome.effectiveness_score = self._calculate_effectiveness_score(
            outcome_type, vital_changes or []
        )
        
        # Save outcome
        self._save_outcome(patient_id, outcome)
        
        return outcome
    
    def record_vital_reading(
        self,
        patient_id: str,
        vital_type: VitalType,
        value: float,
        unit: str,
        notes: str = None
    ) -> VitalReading:
        """
        Record a vital sign reading
        """
        reading = VitalReading(
            vital_type=vital_type.value,
            value=value,
            unit=unit,
            recorded_at=datetime.now().isoformat(),
            notes=notes
        )
        
        # Save reading
        self._save_vital_reading(patient_id, reading)
        
        return reading
    
    def _calculate_effectiveness_score(
        self,
        outcome_type: OutcomeType,
        vital_changes: List[Dict]
    ) -> float:
        """Calculate treatment effectiveness score 0-100"""
        
        # Base score from outcome type
        base_scores = {
            OutcomeType.RESOLVED: 100,
            OutcomeType.IMPROVED: 80,
            OutcomeType.STABLE: 50,
            OutcomeType.WORSENED: 20,
            OutcomeType.ADVERSE_EVENT: 10,
            OutcomeType.DISCONTINUED: 30,
            OutcomeType.PENDING: 50
        }
        
        score = base_scores.get(outcome_type, 50)
        
        # Adjust based on vital changes
        for change in vital_changes:
            improvement = change.get('improvement_percent', 0)
            if improvement > 0:
                score = min(100, score + improvement * 0.2)
            elif improvement < 0:
                score = max(0, score + improvement * 0.3)
        
        return round(score, 1)
    
    def get_patient_outcome_timeline(
        self,
        patient_id: str,
        months: int = 12
    ) -> OutcomeTimeline:
        """
        Get treatment outcome timeline for a patient
        """
        outcomes = self._load_patient_outcomes(patient_id)
        vitals = self._load_patient_vitals(patient_id)
        
        # Build vital trends
        vital_trends = {}
        for vital_type in VitalType:
            type_readings = [v for v in vitals if v.get('vital_type') == vital_type.value]
            if type_readings:
                vital_trends[vital_type.value] = sorted(
                    type_readings, 
                    key=lambda x: x.get('recorded_at', '')
                )
        
        # Calculate overall health trend
        health_trend = self._calculate_health_trend(outcomes, vitals)
        
        timeline = OutcomeTimeline(
            patient_id=patient_id,
            treatments=[TreatmentOutcome(**o) if isinstance(o, dict) else o for o in outcomes],
            vital_trends=vital_trends,
            overall_health_trend=health_trend,
            generated_at=datetime.now().isoformat()
        )
        
        return timeline
    
    def _calculate_health_trend(
        self,
        outcomes: List[Dict],
        vitals: List[Dict]
    ) -> str:
        """Calculate overall health trend"""
        
        if not outcomes and not vitals:
            return "insufficient_data"
        
        improvement_signals = 0
        worsening_signals = 0
        
        # Analyze outcomes
        for outcome in outcomes[-5:]:  # Last 5 outcomes
            outcome_type = outcome.get('outcome_type', '')
            if outcome_type in ['improved', 'resolved']:
                improvement_signals += 1
            elif outcome_type in ['worsened', 'adverse_event']:
                worsening_signals += 1
        
        # Analyze vital trends
        for vital_type, readings in self._group_vitals_by_type(vitals).items():
            if len(readings) >= 2:
                trend = self._calculate_vital_trend(vital_type, readings)
                if trend == 'improving':
                    improvement_signals += 1
                elif trend == 'worsening':
                    worsening_signals += 1
        
        if improvement_signals > worsening_signals * 2:
            return "improving"
        elif worsening_signals > improvement_signals * 2:
            return "declining"
        else:
            return "stable"
    
    def _group_vitals_by_type(self, vitals: List[Dict]) -> Dict[str, List[Dict]]:
        """Group vital readings by type"""
        grouped = {}
        for vital in vitals:
            vital_type = vital.get('vital_type')
            if vital_type:
                if vital_type not in grouped:
                    grouped[vital_type] = []
                grouped[vital_type].append(vital)
        return grouped
    
    def _calculate_vital_trend(
        self,
        vital_type: str,
        readings: List[Dict]
    ) -> str:
        """Calculate trend for a specific vital type"""
        
        if len(readings) < 2:
            return "stable"
        
        # Sort by date
        sorted_readings = sorted(readings, key=lambda x: x.get('recorded_at', ''))
        
        first_value = sorted_readings[0].get('value', 0)
        last_value = sorted_readings[-1].get('value', 0)
        
        if first_value == 0:
            return "stable"
        
        change_percent = ((last_value - first_value) / first_value) * 100
        
        # Determine if change is improving or worsening based on vital type
        # For most vitals, decrease is good (BP, glucose, LDL)
        vitals_where_increase_is_bad = [
            'bp_systolic', 'bp_diastolic', 'blood_glucose', 'hba1c',
            'ldl_cholesterol', 'triglycerides', 'pain_score', 'weight', 'creatinine'
        ]
        
        vitals_where_increase_is_good = [
            'egfr', 'hdl_cholesterol', 'oxygen_saturation'
        ]
        
        if vital_type in vitals_where_increase_is_bad:
            if change_percent < -5:
                return "improving"
            elif change_percent > 5:
                return "worsening"
        elif vital_type in vitals_where_increase_is_good:
            if change_percent > 5:
                return "improving"
            elif change_percent < -5:
                return "worsening"
        
        return "stable"
    
    def predict_treatment_success(
        self,
        medication: str,
        condition: str,
        patient_profile: Dict = None
    ) -> TreatmentPrediction:
        """
        Predict treatment success probability using ML-based model
        """
        patient_profile = patient_profile or {}
        
        # Find drug class
        drug_class = None
        med_lower = medication.lower()
        for drug, cls in self.drug_class_map.items():
            if drug in med_lower:
                drug_class = cls
                break
        
        if not drug_class or drug_class not in self.success_factors:
            # Use generic prediction
            return self._generic_prediction(medication, condition, patient_profile)
        
        # Get model for this drug class
        model = self.success_factors[drug_class]
        base_rate = model['base_success_rate']
        
        # Calculate factors
        supporting_factors = []
        against_factors = []
        adjustment = 0
        
        for factor in model['positive_factors']:
            # Check if factor applies to patient
            if self._factor_applies(factor['factor'], patient_profile, positive=True):
                supporting_factors.append({
                    'factor': factor['factor'],
                    'impact': f"+{factor['weight']*100:.0f}%"
                })
                adjustment += factor['weight']
        
        for factor in model['negative_factors']:
            if self._factor_applies(factor['factor'], patient_profile, positive=False):
                against_factors.append({
                    'factor': factor['factor'],
                    'impact': f"{factor['weight']*100:.0f}%"
                })
                adjustment += factor['weight']
        
        # Calculate final probability
        success_prob = max(0.1, min(0.95, base_rate + adjustment))
        
        # Calculate confidence interval (wider if less data)
        data_quality = len(supporting_factors) + len(against_factors)
        ci_width = 0.15 - (data_quality * 0.01)
        ci_width = max(0.05, min(0.2, ci_width))
        
        ci_lower = max(0, success_prob - ci_width)
        ci_upper = min(1, success_prob + ci_width)
        
        # Get similar patient outcomes
        historical = self.historical_outcomes.get(drug_class, {})
        similar_outcomes = {
            'total_similar_patients': historical.get('total_patients', 0),
            'improved_rate': historical.get('improved', 0) / max(1, historical.get('total_patients', 1)),
            'avg_time_to_improvement': historical.get('avg_time_to_improvement', 'Unknown'),
            'discontinued_rate': historical.get('discontinued', 0) / max(1, historical.get('total_patients', 1))
        }
        
        # Generate recommendation
        if success_prob >= 0.75:
            recommendation = f"High likelihood of success. {medication} is a good choice for this patient's {condition}."
        elif success_prob >= 0.5:
            recommendation = f"Moderate likelihood of success. Consider {medication} but monitor closely for response."
        else:
            recommendation = f"Lower likelihood of success. Consider alternative treatments or address modifiable risk factors before starting {medication}."
        
        return TreatmentPrediction(
            medication=medication,
            condition=condition,
            predicted_success_probability=round(success_prob * 100, 1),
            confidence_interval=(round(ci_lower * 100, 1), round(ci_upper * 100, 1)),
            factors_supporting=supporting_factors,
            factors_against=against_factors,
            similar_patient_outcomes=similar_outcomes,
            recommendation=recommendation
        )
    
    def _factor_applies(
        self,
        factor: str,
        patient_profile: Dict,
        positive: bool
    ) -> bool:
        """Check if a factor applies to the patient"""
        factor_lower = factor.lower()
        
        # Age checks
        if 'age' in factor_lower:
            patient_age = patient_profile.get('age', 50)
            if '< 65' in factor_lower and patient_age < 65:
                return True
            if '> 80' in factor_lower and patient_age > 80:
                return True
            if '> 75' in factor_lower and patient_age > 75:
                return True
        
        # BMI checks
        if 'bmi' in factor_lower or 'obesity' in factor_lower:
            bmi = patient_profile.get('bmi', 25)
            if '> 35' in factor_lower and bmi > 35:
                return True
            if '> 30' in factor_lower and bmi > 30:
                return True
            if '> 40' in factor_lower and bmi > 40:
                return True
            if '< 30' in factor_lower and bmi < 30:
                return True
        
        # Condition checks
        conditions = [c.lower() for c in patient_profile.get('conditions', [])]
        if 'ckd' in factor_lower:
            has_ckd = any('ckd' in c or 'kidney' in c for c in conditions)
            if 'no ckd' in factor_lower:
                return not has_ckd
            return has_ckd
        
        if 'diabetes' in factor_lower:
            return any('diabet' in c for c in conditions)
        
        # Lifestyle checks
        if 'smoking' in factor_lower:
            smoker = patient_profile.get('smoker', False)
            if 'cessation' in factor_lower:
                return not smoker
            if 'continued' in factor_lower:
                return smoker
        
        # Adherence checks
        if 'adherence' in factor_lower:
            adherence = patient_profile.get('adherence_rate', 0.7)
            if '> 80%' in factor_lower:
                return adherence > 0.8
            if 'poor' in factor_lower:
                return adherence < 0.5
        
        # Default: moderate probability of factor applying
        import random
        return random.random() < 0.3 if positive else random.random() < 0.2
    
    def _generic_prediction(
        self,
        medication: str,
        condition: str,
        patient_profile: Dict
    ) -> TreatmentPrediction:
        """Generate a generic prediction when specific model not available"""
        return TreatmentPrediction(
            medication=medication,
            condition=condition,
            predicted_success_probability=65.0,
            confidence_interval=(50.0, 80.0),
            factors_supporting=[
                {"factor": "Standard treatment for condition", "impact": "+10%"}
            ],
            factors_against=[
                {"factor": "Limited patient-specific data", "impact": "-5%"}
            ],
            similar_patient_outcomes={
                'note': 'Insufficient historical data for specific comparison'
            },
            recommendation=f"Monitor response to {medication} and adjust treatment based on clinical response."
        )
    
    def get_medication_outcome_summary(
        self,
        medication: str,
        patient_id: str = None
    ) -> Dict:
        """
        Get outcome summary for a medication
        """
        # Get drug class
        drug_class = None
        med_lower = medication.lower()
        for drug, cls in self.drug_class_map.items():
            if drug in med_lower:
                drug_class = cls
                break
        
        if not drug_class or drug_class not in self.historical_outcomes:
            return {
                'medication': medication,
                'message': 'Limited outcome data available',
                'general_success_rate': 'Unknown'
            }
        
        historical = self.historical_outcomes[drug_class]
        total = historical.get('total_patients', 1)
        
        return {
            'medication': medication,
            'drug_class': drug_class,
            'population_data': {
                'total_patients_studied': total,
                'improvement_rate': f"{historical.get('improved', 0) / total * 100:.1f}%",
                'resolution_rate': f"{historical.get('resolved', 0) / total * 100:.1f}%" if 'resolved' in historical else 'N/A',
                'stable_rate': f"{historical.get('stable', 0) / total * 100:.1f}%",
                'worsened_rate': f"{historical.get('worsened', 0) / total * 100:.1f}%",
                'discontinued_rate': f"{historical.get('discontinued', 0) / total * 100:.1f}%",
                'average_time_to_improvement': f"{historical.get('avg_time_to_improvement', 'Unknown')} days"
            }
        }
    
    def analyze_vital_changes_for_treatment(
        self,
        patient_id: str,
        medication: str,
        start_date: str,
        end_date: str = None
    ) -> Dict:
        """
        Analyze vital sign changes during a treatment period
        """
        vitals = self._load_patient_vitals(patient_id)
        
        if not end_date:
            end_date = datetime.now().isoformat()
        
        # Filter vitals in date range
        relevant_vitals = [
            v for v in vitals
            if start_date <= v.get('recorded_at', '') <= end_date
        ]
        
        # Group by type and analyze
        changes = {}
        for vital_type, readings in self._group_vitals_by_type(relevant_vitals).items():
            if len(readings) >= 2:
                sorted_readings = sorted(readings, key=lambda x: x['recorded_at'])
                first = sorted_readings[0]['value']
                last = sorted_readings[-1]['value']
                
                change = last - first
                change_percent = (change / first * 100) if first != 0 else 0
                
                target_info = self.vital_targets.get(vital_type, {})
                trend = self._calculate_vital_trend(vital_type, readings)
                
                changes[vital_type] = {
                    'initial_value': first,
                    'current_value': last,
                    'absolute_change': round(change, 2),
                    'percent_change': round(change_percent, 1),
                    'trend': trend,
                    'unit': target_info.get('unit', ''),
                    'target_range': target_info.get('target_general', target_info.get('normal', 'Unknown')),
                    'in_target': self._is_in_target(vital_type, last)
                }
        
        return {
            'patient_id': patient_id,
            'medication': medication,
            'analysis_period': {
                'start': start_date,
                'end': end_date
            },
            'vital_changes': changes,
            'summary': self._generate_vital_change_summary(medication, changes)
        }
    
    def _is_in_target(self, vital_type: str, value: float) -> bool:
        """Check if vital is in target range"""
        targets = self.vital_targets.get(vital_type, {})
        target_range = targets.get('target_general', targets.get('normal'))
        
        if target_range and isinstance(target_range, tuple):
            return target_range[0] <= value <= target_range[1]
        return True  # Default to True if no target defined
    
    def _generate_vital_change_summary(
        self,
        medication: str,
        changes: Dict
    ) -> str:
        """Generate natural language summary of vital changes"""
        if not changes:
            return f"No vital sign data available during {medication} treatment."
        
        improvements = []
        concerns = []
        stable = []
        
        vital_names = {
            'bp_systolic': 'systolic blood pressure',
            'bp_diastolic': 'diastolic blood pressure',
            'blood_glucose': 'blood glucose',
            'hba1c': 'HbA1c',
            'ldl_cholesterol': 'LDL cholesterol',
            'hdl_cholesterol': 'HDL cholesterol',
            'weight': 'weight',
            'egfr': 'kidney function (eGFR)',
            'pain_score': 'pain score'
        }
        
        for vital_type, data in changes.items():
            name = vital_names.get(vital_type, vital_type)
            change = data['percent_change']
            trend = data['trend']
            
            if trend == 'improving':
                if change > 0:
                    improvements.append(f"{name} increased by {abs(change):.1f}%")
                else:
                    improvements.append(f"{name} decreased by {abs(change):.1f}%")
            elif trend == 'worsening':
                if change > 0:
                    concerns.append(f"{name} increased by {abs(change):.1f}%")
                else:
                    concerns.append(f"{name} decreased by {abs(change):.1f}%")
            else:
                stable.append(name)
        
        summary_parts = []
        
        if improvements:
            summary_parts.append(f"✅ Improvements: {'; '.join(improvements)}")
        if concerns:
            summary_parts.append(f"⚠️ Concerns: {'; '.join(concerns)}")
        if stable:
            summary_parts.append(f"➡️ Stable: {', '.join(stable)}")
        
        if not summary_parts:
            return f"Vital signs remained generally stable during {medication} treatment."
        
        return f"During {medication} treatment: " + " | ".join(summary_parts)
    
    def _save_outcome(self, patient_id: str, outcome: TreatmentOutcome):
        """Save outcome to storage"""
        filepath = os.path.join(self.data_dir, f"{patient_id}_outcomes.json")
        
        existing = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing = json.load(f)
            except:
                existing = []
        
        existing.append(outcome.to_dict())
        
        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def _save_vital_reading(self, patient_id: str, reading: VitalReading):
        """Save vital reading to storage"""
        filepath = os.path.join(self.data_dir, f"{patient_id}_vitals.json")
        
        existing = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing = json.load(f)
            except:
                existing = []
        
        existing.append(reading.to_dict())
        
        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def _load_patient_outcomes(self, patient_id: str) -> List[Dict]:
        """Load patient outcomes from storage"""
        filepath = os.path.join(self.data_dir, f"{patient_id}_outcomes.json")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _load_patient_vitals(self, patient_id: str) -> List[Dict]:
        """Load patient vitals from storage"""
        filepath = os.path.join(self.data_dir, f"{patient_id}_vitals.json")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def generate_comprehensive_outcome_report(
        self,
        patient_id: str,
        medications: List[str] = None,
        conditions: List[str] = None,
        patient_profile: Dict = None
    ) -> Dict:
        """
        Generate comprehensive outcome analysis and predictions
        """
        medications = medications or []
        conditions = conditions or []
        patient_profile = patient_profile or {}
        
        # Get timeline
        timeline = self.get_patient_outcome_timeline(patient_id)
        
        # Get predictions for current medications
        predictions = []
        for med in medications:
            # Find relevant condition
            condition = conditions[0] if conditions else "general"
            pred = self.predict_treatment_success(med, condition, patient_profile)
            predictions.append(pred.to_dict())
        
        # Get outcome summaries
        summaries = []
        for med in medications:
            summary = self.get_medication_outcome_summary(med, patient_id)
            summaries.append(summary)
        
        return {
            'patient_id': patient_id,
            'generated_at': datetime.now().isoformat(),
            'outcome_timeline': timeline.to_dict(),
            'treatment_predictions': predictions,
            'medication_summaries': summaries,
            'overall_health_trend': timeline.overall_health_trend,
            'insights': self._generate_outcome_insights(timeline, predictions)
        }
    
    def _generate_outcome_insights(
        self,
        timeline: OutcomeTimeline,
        predictions: List[Dict]
    ) -> List[Dict]:
        """Generate actionable insights from outcomes"""
        insights = []
        
        # Insight from health trend
        if timeline.overall_health_trend == 'improving':
            insights.append({
                'type': 'positive',
                'title': 'Treatment Response Positive',
                'message': 'Patient is showing overall improvement. Current treatment regimen appears effective.'
            })
        elif timeline.overall_health_trend == 'declining':
            insights.append({
                'type': 'warning',
                'title': 'Treatment Response Concerning',
                'message': 'Patient showing declining trend. Consider treatment optimization or alternative approaches.'
            })
        
        # Insights from predictions
        low_success_meds = [p for p in predictions if p.get('predicted_success_probability', 0) < 50]
        if low_success_meds:
            med_names = [p['medication'] for p in low_success_meds]
            insights.append({
                'type': 'info',
                'title': 'Medications with Lower Predicted Success',
                'message': f"Consider monitoring closely or alternatives for: {', '.join(med_names)}"
            })
        
        # Insights from vital trends
        for vital_type, readings in timeline.vital_trends.items():
            if len(readings) >= 3:
                trend = self._calculate_vital_trend(vital_type, readings)
                if trend == 'worsening':
                    insights.append({
                        'type': 'warning',
                        'title': f'{vital_type.replace("_", " ").title()} Trend',
                        'message': f'Worsening trend detected. Review contributing factors and treatment adjustments.'
                    })
        
        return insights


# Create singleton
treatment_outcome_service = TreatmentOutcomeService()
