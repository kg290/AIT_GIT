"""
Uncertainty & Risk Handling Service
Track confidence, flag low-confidence extractions, escalate unclear cases
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk levels for medical decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UncertaintySource(str, Enum):
    """Sources of uncertainty"""
    OCR_QUALITY = "ocr_quality"
    EXTRACTION_CONFIDENCE = "extraction_confidence"
    ENTITY_AMBIGUITY = "entity_ambiguity"
    DATA_MISSING = "data_missing"
    CONFLICTING_INFO = "conflicting_info"
    DRUG_LOOKUP_FAILED = "drug_lookup_failed"
    DOSAGE_UNCLEAR = "dosage_unclear"


@dataclass
class UncertaintyFlag:
    """A flag indicating uncertainty in extraction"""
    source: UncertaintySource
    field: str
    confidence: float
    message: str
    suggested_review: bool = True
    severity: RiskLevel = RiskLevel.MEDIUM
    original_value: str = None
    alternatives: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Overall risk assessment for an extraction"""
    overall_risk: RiskLevel
    overall_confidence: float
    flags: List[UncertaintyFlag]
    requires_review: bool
    auto_approve: bool
    risk_summary: str


class UncertaintyService:
    """
    Uncertainty & Risk Handling Service
    
    Features:
    - Track confidence at every extraction stage
    - Flag low-confidence extractions
    - Calculate overall risk scores
    - Determine when human review is needed
    - Provide uncertainty explanations
    """
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    MEDIUM_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.5
    
    # Fields that require higher confidence
    CRITICAL_FIELDS = [
        "medication_name", "dosage", "frequency",
        "patient_allergies", "diagnosis"
    ]
    
    # Fields with lower risk
    LOW_RISK_FIELDS = [
        "doctor_address", "clinic_name", "prescription_date"
    ]
    
    def __init__(self, db: Session = None):
        self.db = db
        self.flags: List[UncertaintyFlag] = []
    
    def reset(self):
        """Reset flags for new extraction"""
        self.flags = []
    
    # ==================== Confidence Tracking ====================
    
    def assess_ocr_confidence(
        self,
        raw_text: str,
        word_confidences: List[float] = None
    ) -> Tuple[float, List[UncertaintyFlag]]:
        """Assess confidence of OCR extraction"""
        
        flags = []
        
        # Check text quality indicators
        if not raw_text or len(raw_text.strip()) < 20:
            flags.append(UncertaintyFlag(
                source=UncertaintySource.OCR_QUALITY,
                field="raw_text",
                confidence=0.3,
                message="OCR extracted very little text - image may be unclear",
                severity=RiskLevel.HIGH
            ))
            return 0.3, flags
        
        # Check for common OCR quality issues
        quality_issues = []
        
        # Too many special characters
        special_ratio = sum(1 for c in raw_text if not c.isalnum() and not c.isspace()) / len(raw_text)
        if special_ratio > 0.2:
            quality_issues.append("High ratio of special characters")
        
        # Very long words (likely merged)
        words = raw_text.split()
        long_words = [w for w in words if len(w) > 25]
        if len(long_words) > 3:
            quality_issues.append("Words appear merged together")
        
        # Calculate confidence
        if word_confidences:
            avg_confidence = sum(word_confidences) / len(word_confidences)
        else:
            avg_confidence = 0.85 if not quality_issues else 0.6
        
        if quality_issues:
            flags.append(UncertaintyFlag(
                source=UncertaintySource.OCR_QUALITY,
                field="raw_text",
                confidence=avg_confidence,
                message=f"OCR quality concerns: {'; '.join(quality_issues)}",
                severity=RiskLevel.MEDIUM
            ))
        
        self.flags.extend(flags)
        return avg_confidence, flags
    
    def assess_entity_confidence(
        self,
        entity_type: str,
        entity_value: str,
        extraction_method: str,
        raw_confidence: float,
        alternatives: List[str] = None
    ) -> UncertaintyFlag:
        """Assess confidence for a specific extracted entity"""
        
        # Adjust confidence based on field criticality
        adjusted_confidence = raw_confidence
        severity = RiskLevel.MEDIUM
        
        if entity_type in self.CRITICAL_FIELDS:
            # More strict for critical fields
            adjusted_confidence *= 0.95
            if adjusted_confidence < self.MEDIUM_CONFIDENCE_THRESHOLD:
                severity = RiskLevel.HIGH
        elif entity_type in self.LOW_RISK_FIELDS:
            # More lenient for low-risk fields
            adjusted_confidence = min(1.0, adjusted_confidence * 1.05)
            severity = RiskLevel.LOW
        
        # Create flag if confidence is concerning
        flag = None
        if adjusted_confidence < self.MEDIUM_CONFIDENCE_THRESHOLD:
            flag = UncertaintyFlag(
                source=UncertaintySource.EXTRACTION_CONFIDENCE,
                field=entity_type,
                confidence=adjusted_confidence,
                message=f"Low confidence extraction for {entity_type}",
                severity=severity,
                original_value=entity_value,
                alternatives=alternatives or []
            )
            self.flags.append(flag)
        
        return flag
    
    def flag_ambiguous_entity(
        self,
        entity_type: str,
        entity_value: str,
        possible_matches: List[str],
        context: str = None
    ) -> UncertaintyFlag:
        """Flag an ambiguous entity that could match multiple values"""
        
        flag = UncertaintyFlag(
            source=UncertaintySource.ENTITY_AMBIGUITY,
            field=entity_type,
            confidence=0.5,
            message=f"Ambiguous {entity_type}: '{entity_value}' could be: {', '.join(possible_matches[:5])}",
            severity=RiskLevel.HIGH if entity_type in self.CRITICAL_FIELDS else RiskLevel.MEDIUM,
            original_value=entity_value,
            alternatives=possible_matches
        )
        
        self.flags.append(flag)
        return flag
    
    def flag_missing_data(
        self,
        field: str,
        expected: bool = True,
        context: str = None
    ) -> UncertaintyFlag:
        """Flag missing required data"""
        
        severity = RiskLevel.HIGH if field in self.CRITICAL_FIELDS else RiskLevel.MEDIUM
        
        flag = UncertaintyFlag(
            source=UncertaintySource.DATA_MISSING,
            field=field,
            confidence=0.0,
            message=f"Required field '{field}' not found in extraction",
            severity=severity
        )
        
        self.flags.append(flag)
        return flag
    
    def flag_conflicting_info(
        self,
        field: str,
        values: List[str],
        context: str = None
    ) -> UncertaintyFlag:
        """Flag conflicting information found"""
        
        flag = UncertaintyFlag(
            source=UncertaintySource.CONFLICTING_INFO,
            field=field,
            confidence=0.4,
            message=f"Conflicting values found for {field}: {', '.join(values)}",
            severity=RiskLevel.HIGH,
            alternatives=values
        )
        
        self.flags.append(flag)
        return flag
    
    def flag_dosage_unclear(
        self,
        medication: str,
        dosage_text: str,
        issue: str
    ) -> UncertaintyFlag:
        """Flag unclear dosage information"""
        
        flag = UncertaintyFlag(
            source=UncertaintySource.DOSAGE_UNCLEAR,
            field="dosage",
            confidence=0.5,
            message=f"Dosage unclear for {medication}: {issue}",
            severity=RiskLevel.CRITICAL,  # Dosage issues are critical
            original_value=dosage_text
        )
        
        self.flags.append(flag)
        return flag
    
    def flag_drug_not_found(
        self,
        medication_name: str,
        closest_matches: List[str] = None
    ) -> UncertaintyFlag:
        """Flag medication not found in database"""
        
        flag = UncertaintyFlag(
            source=UncertaintySource.DRUG_LOOKUP_FAILED,
            field="medication_name",
            confidence=0.6,
            message=f"Medication '{medication_name}' not found in drug database",
            severity=RiskLevel.MEDIUM,
            original_value=medication_name,
            alternatives=closest_matches or []
        )
        
        self.flags.append(flag)
        return flag
    
    # ==================== Risk Assessment ====================
    
    def calculate_risk_score(self, flags: List[UncertaintyFlag] = None) -> float:
        """Calculate overall risk score (0-1, higher = more risky)"""
        
        if flags is None:
            flags = self.flags
        
        if not flags:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 1.0
        }
        
        total_weight = 0
        risk_sum = 0
        
        for flag in flags:
            weight = severity_weights.get(flag.severity, 0.3)
            total_weight += weight
            # Lower confidence = higher risk
            risk_sum += weight * (1 - flag.confidence)
        
        if total_weight == 0:
            return 0.0
        
        return min(1.0, risk_sum / total_weight)
    
    def calculate_overall_confidence(
        self,
        component_confidences: Dict[str, float]
    ) -> float:
        """Calculate weighted overall confidence"""
        
        # Weights for different components
        weights = {
            "ocr": 0.3,
            "patient": 0.2,
            "doctor": 0.1,
            "medications": 0.4
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for component, confidence in component_confidences.items():
            weight = weights.get(component, 0.2)
            total_weight += weight
            weighted_sum += weight * confidence
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def assess_risk(
        self,
        component_confidences: Dict[str, float] = None
    ) -> RiskAssessment:
        """Generate comprehensive risk assessment"""
        
        # Calculate scores
        if component_confidences:
            overall_confidence = self.calculate_overall_confidence(component_confidences)
        else:
            # Use flags to estimate
            if self.flags:
                overall_confidence = sum(f.confidence for f in self.flags) / len(self.flags)
            else:
                overall_confidence = 0.9
        
        risk_score = self.calculate_risk_score()
        
        # Determine risk level
        if risk_score >= 0.7:
            overall_risk = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            overall_risk = RiskLevel.HIGH
        elif risk_score >= 0.3:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        # Determine if review needed
        requires_review = (
            overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            overall_confidence < self.MEDIUM_CONFIDENCE_THRESHOLD or
            any(f.severity == RiskLevel.CRITICAL for f in self.flags)
        )
        
        # Auto-approve only if very confident and low risk
        auto_approve = (
            overall_confidence >= self.HIGH_CONFIDENCE_THRESHOLD and
            overall_risk == RiskLevel.LOW and
            len(self.flags) == 0
        )
        
        # Generate summary
        summary = self._generate_risk_summary(overall_risk, overall_confidence, self.flags)
        
        return RiskAssessment(
            overall_risk=overall_risk,
            overall_confidence=overall_confidence,
            flags=self.flags.copy(),
            requires_review=requires_review,
            auto_approve=auto_approve,
            risk_summary=summary
        )
    
    def _generate_risk_summary(
        self,
        risk: RiskLevel,
        confidence: float,
        flags: List[UncertaintyFlag]
    ) -> str:
        """Generate human-readable risk summary"""
        
        if not flags:
            return f"✅ Extraction complete with {confidence*100:.1f}% confidence. No issues detected."
        
        critical_flags = [f for f in flags if f.severity == RiskLevel.CRITICAL]
        high_flags = [f for f in flags if f.severity == RiskLevel.HIGH]
        
        parts = []
        
        if critical_flags:
            parts.append(f"⚠️ CRITICAL: {len(critical_flags)} critical issues requiring immediate review")
        if high_flags:
            parts.append(f"🔴 HIGH: {len(high_flags)} high-priority items need attention")
        
        parts.append(f"Overall confidence: {confidence*100:.1f}%")
        
        # List specific issues
        issues = []
        for flag in flags[:5]:  # Top 5 issues
            issues.append(f"  • {flag.message}")
        
        if issues:
            parts.append("Issues:\n" + "\n".join(issues))
        
        return "\n".join(parts)
    
    # ==================== Decision Support ====================
    
    def should_escalate(self, assessment: RiskAssessment = None) -> Tuple[bool, str]:
        """Determine if extraction should be escalated to supervisor"""
        
        if assessment is None:
            assessment = self.assess_risk()
        
        reasons = []
        
        if assessment.overall_risk == RiskLevel.CRITICAL:
            reasons.append("Critical risk level detected")
        
        if assessment.overall_confidence < self.LOW_CONFIDENCE_THRESHOLD:
            reasons.append("Overall confidence below minimum threshold")
        
        critical_flags = [f for f in assessment.flags if f.severity == RiskLevel.CRITICAL]
        if critical_flags:
            reasons.append(f"{len(critical_flags)} critical issues found")
        
        # Check for specific dangerous situations
        dosage_issues = [f for f in assessment.flags if f.source == UncertaintySource.DOSAGE_UNCLEAR]
        if dosage_issues:
            reasons.append("Unclear dosage detected - potential safety risk")
        
        should_escalate = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else "No escalation needed"
        
        return should_escalate, reason_text
    
    def get_review_priority(self, assessment: RiskAssessment = None) -> int:
        """Get review priority (1=highest, 5=lowest)"""
        
        if assessment is None:
            assessment = self.assess_risk()
        
        if assessment.overall_risk == RiskLevel.CRITICAL:
            return 1
        elif assessment.overall_risk == RiskLevel.HIGH:
            return 2
        elif assessment.overall_confidence < self.MEDIUM_CONFIDENCE_THRESHOLD:
            return 3
        elif assessment.overall_risk == RiskLevel.MEDIUM:
            return 4
        else:
            return 5
    
    # ==================== Reporting ====================
    
    def get_flags_by_severity(self) -> Dict[str, List[UncertaintyFlag]]:
        """Group flags by severity"""
        
        result = {level.value: [] for level in RiskLevel}
        
        for flag in self.flags:
            result[flag.severity.value].append(flag)
        
        return result
    
    def get_flags_by_source(self) -> Dict[str, List[UncertaintyFlag]]:
        """Group flags by source"""
        
        result = {source.value: [] for source in UncertaintySource}
        
        for flag in self.flags:
            result[flag.source.value].append(flag)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all uncertainty data"""
        
        assessment = self.assess_risk()
        
        return {
            "overall_risk": assessment.overall_risk.value,
            "overall_confidence": assessment.overall_confidence,
            "requires_review": assessment.requires_review,
            "auto_approve": assessment.auto_approve,
            "risk_summary": assessment.risk_summary,
            "flags": [
                {
                    "source": f.source.value,
                    "field": f.field,
                    "confidence": f.confidence,
                    "message": f.message,
                    "severity": f.severity.value,
                    "original_value": f.original_value,
                    "alternatives": f.alternatives
                }
                for f in self.flags
            ],
            "flags_by_severity": {
                k: len(v) for k, v in self.get_flags_by_severity().items()
            }
        }
