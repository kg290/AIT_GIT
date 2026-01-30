"""
Explainability & Evidence Service
Track sources, reasoning, and confidence for all extractions
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Float
from backend.models.database import Base

logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    OCR_EXTRACTION = "ocr_extraction"
    PATTERN_MATCH = "pattern_match"
    AI_INFERENCE = "ai_inference"
    USER_INPUT = "user_input"
    DATABASE_LOOKUP = "database_lookup"
    DRUG_DATABASE = "drug_database"


@dataclass
class Evidence:
    """Evidence supporting a fact or extraction"""
    evidence_type: str
    source_document_id: Optional[int]
    source_text: str
    text_region: Optional[Dict]  # bounding box or line numbers
    confidence: float
    extraction_method: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "evidence_type": self.evidence_type,
            "source_document_id": self.source_document_id,
            "source_text": self.source_text,
            "text_region": self.text_region,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    step_number: int
    action: str
    input_data: Any
    output_data: Any
    reasoning: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "step_number": self.step_number,
            "action": self.action,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


@dataclass
class ExplainableResult:
    """A result with full explainability"""
    result_type: str
    value: Any
    confidence: float
    evidence_list: List[Evidence]
    reasoning_steps: List[ReasoningStep]
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "result_type": self.result_type,
            "value": self.value,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence_list],
            "reasoning": [r.to_dict() for r in self.reasoning_steps],
            "warnings": self.warnings
        }


class ExtractionEvidence(Base):
    """Database model to store extraction evidence"""
    __tablename__ = "extraction_evidence"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # What was extracted
    entity_type = Column(String(50), nullable=False)  # medication, patient_name, diagnosis, etc.
    entity_value = Column(Text, nullable=False)
    
    # Source
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=True)
    
    # Evidence
    source_text = Column(Text, nullable=True)
    text_region = Column(JSON, nullable=True)  # {"start_line": 5, "end_line": 5, "start_char": 10, "end_char": 25}
    
    # Extraction details
    extraction_method = Column(String(50), nullable=False)  # regex, ai, pattern, user
    pattern_used = Column(String(200), nullable=True)
    confidence = Column(Float, default=1.0)
    
    # Timestamps
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "entity_value": self.entity_value,
            "source_text": self.source_text,
            "text_region": self.text_region,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None
        }


class InteractionExplanation(Base):
    """Explanation for why a drug interaction was flagged"""
    __tablename__ = "interaction_explanations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Interaction
    drug1 = Column(String(200), nullable=False)
    drug2 = Column(String(200), nullable=False)
    severity = Column(String(50), nullable=False)
    
    # Explanation
    mechanism = Column(Text, nullable=True)
    clinical_effects = Column(Text, nullable=True)
    evidence_source = Column(String(200), nullable=True)  # DrugBank, FDA, literature
    evidence_url = Column(String(500), nullable=True)
    
    # Recommendation
    management = Column(Text, nullable=True)
    alternatives = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "drug1": self.drug1,
            "drug2": self.drug2,
            "severity": self.severity,
            "mechanism": self.mechanism,
            "clinical_effects": self.clinical_effects,
            "evidence_source": self.evidence_source,
            "management": self.management,
            "alternatives": self.alternatives
        }


class ExplainabilityService:
    """
    Service to track and explain all extractions and decisions
    
    Features:
    - Show which document caused each extraction
    - Show text region supporting each fact
    - Explain why a drug interaction was flagged
    - Show reasoning steps for conclusions
    - Display confidence for every output
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.reasoning_log = []
    
    def start_reasoning_session(self):
        """Start a new reasoning session"""
        self.reasoning_log = []
    
    def log_reasoning_step(
        self,
        action: str,
        input_data: Any,
        output_data: Any,
        reasoning: str,
        confidence: float = 1.0
    ):
        """Log a reasoning step"""
        step = ReasoningStep(
            step_number=len(self.reasoning_log) + 1,
            action=action,
            input_data=input_data,
            output_data=output_data,
            reasoning=reasoning,
            confidence=confidence
        )
        self.reasoning_log.append(step)
        return step
    
    def get_reasoning_log(self) -> List[ReasoningStep]:
        """Get the current reasoning log"""
        return self.reasoning_log
    
    # ==================== Evidence Storage ====================
    
    def store_extraction_evidence(
        self,
        entity_type: str,
        entity_value: str,
        document_id: int = None,
        prescription_id: int = None,
        source_text: str = None,
        text_region: Dict = None,
        extraction_method: str = "unknown",
        pattern_used: str = None,
        confidence: float = 1.0
    ) -> ExtractionEvidence:
        """Store evidence for an extraction"""
        
        evidence = ExtractionEvidence(
            entity_type=entity_type,
            entity_value=entity_value,
            document_id=document_id,
            prescription_id=prescription_id,
            source_text=source_text,
            text_region=text_region,
            extraction_method=extraction_method,
            pattern_used=pattern_used,
            confidence=confidence
        )
        
        self.db.add(evidence)
        self.db.commit()
        self.db.refresh(evidence)
        
        return evidence
    
    def get_evidence_for_entity(
        self,
        entity_type: str,
        entity_value: str = None,
        document_id: int = None
    ) -> List[ExtractionEvidence]:
        """Get all evidence for an entity type"""
        
        query = self.db.query(ExtractionEvidence).filter(
            ExtractionEvidence.entity_type == entity_type
        )
        
        if entity_value:
            query = query.filter(ExtractionEvidence.entity_value.ilike(f"%{entity_value}%"))
        
        if document_id:
            query = query.filter(ExtractionEvidence.document_id == document_id)
        
        return query.all()
    
    # ==================== Interaction Explanations ====================
    
    def store_interaction_explanation(
        self,
        drug1: str,
        drug2: str,
        severity: str,
        mechanism: str = None,
        clinical_effects: str = None,
        evidence_source: str = None,
        management: str = None,
        alternatives: List[str] = None
    ) -> InteractionExplanation:
        """Store explanation for a drug interaction"""
        
        explanation = InteractionExplanation(
            drug1=drug1,
            drug2=drug2,
            severity=severity,
            mechanism=mechanism,
            clinical_effects=clinical_effects,
            evidence_source=evidence_source,
            management=management,
            alternatives=alternatives or []
        )
        
        self.db.add(explanation)
        self.db.commit()
        self.db.refresh(explanation)
        
        return explanation
    
    def get_interaction_explanation(
        self,
        drug1: str,
        drug2: str
    ) -> Optional[InteractionExplanation]:
        """Get explanation for a specific drug interaction"""
        
        # Try both orderings
        explanation = self.db.query(InteractionExplanation).filter(
            ((InteractionExplanation.drug1.ilike(f"%{drug1}%")) & 
             (InteractionExplanation.drug2.ilike(f"%{drug2}%"))) |
            ((InteractionExplanation.drug1.ilike(f"%{drug2}%")) & 
             (InteractionExplanation.drug2.ilike(f"%{drug1}%")))
        ).first()
        
        return explanation
    
    # ==================== Explainable Results ====================
    
    def create_explainable_result(
        self,
        result_type: str,
        value: Any,
        confidence: float,
        evidence_list: List[Evidence] = None,
        warnings: List[str] = None
    ) -> ExplainableResult:
        """Create an explainable result with all context"""
        
        return ExplainableResult(
            result_type=result_type,
            value=value,
            confidence=confidence,
            evidence_list=evidence_list or [],
            reasoning_steps=self.reasoning_log.copy(),
            warnings=warnings or []
        )
    
    def explain_extraction(
        self,
        entity_type: str,
        entity_value: str,
        source_text: str,
        confidence: float,
        method: str
    ) -> Dict[str, Any]:
        """Generate a human-readable explanation for an extraction"""
        
        explanation = {
            "what": f"Extracted {entity_type}: {entity_value}",
            "from": f"Source text: '{source_text[:100]}...'" if len(source_text) > 100 else f"Source text: '{source_text}'",
            "how": f"Method: {method}",
            "confidence": f"{confidence * 100:.1f}%",
            "interpretation": self._interpret_confidence(confidence)
        }
        
        return explanation
    
    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence level"""
        if confidence >= 0.9:
            return "High confidence - reliable extraction"
        elif confidence >= 0.7:
            return "Moderate confidence - may need verification"
        elif confidence >= 0.5:
            return "Low confidence - recommend human review"
        else:
            return "Very low confidence - likely needs correction"
    
    def explain_drug_interaction(
        self,
        drug1: str,
        drug2: str,
        severity: str,
        interaction_info: Dict
    ) -> Dict[str, Any]:
        """Generate explanation for a drug interaction flag"""
        
        explanation = {
            "alert": f"Potential interaction between {drug1} and {drug2}",
            "severity": severity,
            "why_flagged": interaction_info.get('description', 'Known drug-drug interaction'),
            "mechanism": interaction_info.get('mechanism', 'See medical literature'),
            "clinical_significance": interaction_info.get('clinical_effects', 'Consult prescribing information'),
            "recommendation": interaction_info.get('management', 'Consult with healthcare provider'),
            "evidence_level": interaction_info.get('evidence', 'established')
        }
        
        return explanation
    
    # ==================== Confidence Tracking ====================
    
    def aggregate_confidence(self, confidences: List[float], method: str = "average") -> float:
        """Aggregate multiple confidence scores"""
        
        if not confidences:
            return 0.0
        
        if method == "average":
            return sum(confidences) / len(confidences)
        elif method == "minimum":
            return min(confidences)
        elif method == "geometric":
            product = 1.0
            for c in confidences:
                product *= c
            return product ** (1.0 / len(confidences))
        else:
            return sum(confidences) / len(confidences)
    
    def should_flag_for_review(self, confidence: float, threshold: float = 0.7) -> bool:
        """Determine if extraction should be flagged for human review"""
        return confidence < threshold
    
    def get_extraction_summary(self, document_id: int) -> Dict[str, Any]:
        """Get summary of all extractions from a document with evidence"""
        
        evidence_list = self.db.query(ExtractionEvidence).filter(
            ExtractionEvidence.document_id == document_id
        ).all()
        
        summary = {
            "document_id": document_id,
            "total_extractions": len(evidence_list),
            "by_type": {},
            "confidence_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "needs_review": []
        }
        
        for ev in evidence_list:
            # Count by type
            if ev.entity_type not in summary["by_type"]:
                summary["by_type"][ev.entity_type] = []
            summary["by_type"][ev.entity_type].append(ev.to_dict())
            
            # Confidence distribution
            if ev.confidence >= 0.8:
                summary["confidence_distribution"]["high"] += 1
            elif ev.confidence >= 0.5:
                summary["confidence_distribution"]["medium"] += 1
            else:
                summary["confidence_distribution"]["low"] += 1
                summary["needs_review"].append(ev.to_dict())
        
        return summary
