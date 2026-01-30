"""
Human Review & Correction Service
Allow corrections, track edits, maintain version history
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from dataclasses import dataclass

from backend.models.audit import Correction, CorrectionType, AuditLog, AuditAction

logger = logging.getLogger(__name__)


@dataclass
class CorrectionRequest:
    """Request to correct an extraction"""
    correction_type: str
    document_id: Optional[int]
    entity_id: Optional[int]
    entity_type: Optional[str]
    field_name: str
    original_value: str
    corrected_value: str
    reason: Optional[str]
    corrected_by: str


@dataclass
class DismissedAlert:
    """A dismissed safety alert"""
    alert_type: str
    alert_id: Optional[int]
    drug1: Optional[str]
    drug2: Optional[str]
    reason: str
    dismissed_by: str
    dismissed_at: datetime


class HumanReviewService:
    """
    Human Review & Correction Service
    
    Features:
    - Allow correction of OCR text
    - Allow correction of extracted entities
    - Allow dismissal of incorrect alerts
    - Log all corrections
    - Version all edits
    - Learn from corrections without deleting history
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== OCR Corrections ====================
    
    def correct_ocr_text(
        self,
        document_id: int,
        original_text: str,
        corrected_text: str,
        corrected_by: str,
        region: Dict = None,
        reason: str = None
    ) -> Correction:
        """Correct OCR extracted text"""
        
        correction = Correction(
            correction_type=CorrectionType.OCR_TEXT,
            document_id=document_id,
            field_name="ocr_text",
            original_value=original_text,
            corrected_value=corrected_text,
            reason=reason,
            source_region=region,
            corrected_by=corrected_by
        )
        
        self.db.add(correction)
        self.db.commit()
        self.db.refresh(correction)
        
        # Log the correction
        self._log_correction(correction, corrected_by)
        
        logger.info(f"OCR correction by {corrected_by} on document {document_id}")
        return correction
    
    # ==================== Entity Corrections ====================
    
    def correct_entity(
        self,
        entity_type: str,
        entity_id: int,
        field_name: str,
        original_value: str,
        corrected_value: str,
        corrected_by: str,
        document_id: int = None,
        reason: str = None,
        confidence_before: float = None
    ) -> Correction:
        """Correct an extracted entity"""
        
        # Map entity type to correction type
        correction_type_map = {
            "medication": CorrectionType.MEDICATION,
            "dosage": CorrectionType.DOSAGE,
            "frequency": CorrectionType.FREQUENCY,
            "diagnosis": CorrectionType.DIAGNOSIS,
            "patient": CorrectionType.ENTITY,
            "doctor": CorrectionType.ENTITY
        }
        
        corr_type = correction_type_map.get(entity_type.lower(), CorrectionType.OTHER)
        
        correction = Correction(
            correction_type=corr_type,
            document_id=document_id,
            entity_id=entity_id,
            entity_type=entity_type,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            reason=reason,
            confidence_before=confidence_before,
            corrected_by=corrected_by
        )
        
        self.db.add(correction)
        self.db.commit()
        self.db.refresh(correction)
        
        self._log_correction(correction, corrected_by)
        
        logger.info(f"Entity correction: {entity_type}.{field_name} by {corrected_by}")
        return correction
    
    def correct_medication(
        self,
        prescription_id: int,
        medication_id: int,
        field: str,
        original: str,
        corrected: str,
        corrected_by: str,
        reason: str = None
    ) -> Correction:
        """Correct a medication extraction"""
        
        return self.correct_entity(
            entity_type="medication",
            entity_id=medication_id,
            field_name=field,
            original_value=original,
            corrected_value=corrected,
            corrected_by=corrected_by,
            reason=reason
        )
    
    # ==================== Alert Dismissal ====================
    
    def dismiss_interaction_alert(
        self,
        drug1: str,
        drug2: str,
        dismissed_by: str,
        reason: str,
        patient_id: int = None
    ) -> Correction:
        """Dismiss a drug interaction alert as not applicable"""
        
        correction = Correction(
            correction_type=CorrectionType.INTERACTION_DISMISS,
            field_name="drug_interaction",
            original_value=f"{drug1} + {drug2}",
            corrected_value="dismissed",
            reason=reason,
            corrected_by=dismissed_by
        )
        
        self.db.add(correction)
        self.db.commit()
        self.db.refresh(correction)
        
        # Log dismissal
        self._log_audit(
            action=AuditAction.DISMISS,
            action_detail=f"Dismissed interaction: {drug1} + {drug2}",
            entity_type="drug_interaction",
            user_name=dismissed_by,
            changes={"drug1": drug1, "drug2": drug2, "reason": reason}
        )
        
        logger.info(f"Alert dismissed: {drug1}+{drug2} by {dismissed_by}")
        return correction
    
    def is_alert_dismissed(
        self,
        drug1: str,
        drug2: str,
        patient_id: int = None
    ) -> bool:
        """Check if an interaction alert has been dismissed"""
        
        dismissed = self.db.query(Correction).filter(
            and_(
                Correction.correction_type == CorrectionType.INTERACTION_DISMISS,
                Correction.original_value.in_([
                    f"{drug1} + {drug2}",
                    f"{drug2} + {drug1}"
                ])
            )
        ).first()
        
        return dismissed is not None
    
    def get_dismissed_alerts(self, patient_id: int = None) -> List[Correction]:
        """Get all dismissed alerts"""
        
        query = self.db.query(Correction).filter(
            Correction.correction_type == CorrectionType.INTERACTION_DISMISS
        )
        
        return query.order_by(desc(Correction.corrected_at)).all()
    
    # ==================== Verification ====================
    
    def verify_correction(
        self,
        correction_id: int,
        verified_by: str
    ) -> Optional[Correction]:
        """Mark a correction as verified"""
        
        correction = self.db.query(Correction).filter(
            Correction.id == correction_id
        ).first()
        
        if correction:
            correction.verified = True
            correction.verified_by = verified_by
            correction.verified_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Correction {correction_id} verified by {verified_by}")
            return correction
        
        return None
    
    # ==================== Correction History ====================
    
    def get_corrections_for_document(self, document_id: int) -> List[Correction]:
        """Get all corrections made to a document"""
        
        return self.db.query(Correction).filter(
            Correction.document_id == document_id
        ).order_by(desc(Correction.corrected_at)).all()
    
    def get_corrections_by_user(
        self,
        user: str,
        limit: int = 50
    ) -> List[Correction]:
        """Get corrections made by a specific user"""
        
        return self.db.query(Correction).filter(
            Correction.corrected_by == user
        ).order_by(desc(Correction.corrected_at)).limit(limit).all()
    
    def get_unverified_corrections(self, limit: int = 100) -> List[Correction]:
        """Get corrections pending verification"""
        
        return self.db.query(Correction).filter(
            Correction.verified == False
        ).order_by(desc(Correction.corrected_at)).limit(limit).all()
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get statistics about corrections"""
        
        total = self.db.query(Correction).count()
        verified = self.db.query(Correction).filter(Correction.verified == True).count()
        
        by_type = {}
        for ct in CorrectionType:
            count = self.db.query(Correction).filter(
                Correction.correction_type == ct
            ).count()
            by_type[ct.value] = count
        
        return {
            "total_corrections": total,
            "verified": verified,
            "pending_verification": total - verified,
            "by_type": by_type
        }
    
    # ==================== Learning from Corrections ====================
    
    def mark_for_training(self, correction_id: int) -> Optional[Correction]:
        """Mark a correction to be used for model improvement"""
        
        correction = self.db.query(Correction).filter(
            Correction.id == correction_id
        ).first()
        
        if correction:
            correction.used_for_training = True
            self.db.commit()
            return correction
        
        return None
    
    def get_training_corrections(
        self,
        correction_type: CorrectionType = None
    ) -> List[Correction]:
        """Get corrections that can be used for training"""
        
        query = self.db.query(Correction).filter(
            and_(
                Correction.verified == True,
                Correction.used_for_training == False
            )
        )
        
        if correction_type:
            query = query.filter(Correction.correction_type == correction_type)
        
        return query.all()
    
    def get_common_corrections(self, limit: int = 20) -> List[Dict]:
        """Get most common corrections for pattern learning"""
        
        # This would be more sophisticated in production
        corrections = self.db.query(Correction).filter(
            Correction.verified == True
        ).all()
        
        # Group by original -> corrected
        patterns = {}
        for c in corrections:
            key = f"{c.original_value} -> {c.corrected_value}"
            if key not in patterns:
                patterns[key] = {
                    "original": c.original_value,
                    "corrected": c.corrected_value,
                    "type": c.correction_type.value if c.correction_type else None,
                    "count": 0
                }
            patterns[key]["count"] += 1
        
        # Sort by count
        sorted_patterns = sorted(patterns.values(), key=lambda x: x["count"], reverse=True)
        return sorted_patterns[:limit]
    
    # ==================== Audit Logging ====================
    
    def _log_correction(self, correction: Correction, user: str):
        """Log a correction to audit trail"""
        
        self._log_audit(
            action=AuditAction.CORRECT,
            action_detail=f"Corrected {correction.correction_type.value if correction.correction_type else 'unknown'}",
            entity_type=correction.entity_type or "document",
            entity_id=correction.document_id or correction.entity_id,
            user_name=user,
            old_value={"value": correction.original_value},
            new_value={"value": correction.corrected_value},
            changes={
                "field": correction.field_name,
                "reason": correction.reason
            }
        )
    
    def _log_audit(
        self,
        action: AuditAction,
        action_detail: str,
        entity_type: str,
        entity_id: int = None,
        user_name: str = None,
        old_value: Dict = None,
        new_value: Dict = None,
        changes: Dict = None
    ):
        """Create an audit log entry"""
        
        audit = AuditLog(
            action=action,
            action_detail=action_detail,
            entity_type=entity_type,
            entity_id=entity_id,
            user_name=user_name,
            old_value=old_value,
            new_value=new_value,
            changes=changes
        )
        
        self.db.add(audit)
        self.db.commit()
    
    # ==================== Review Queue ====================
    
    def get_items_for_review(
        self,
        confidence_threshold: float = 0.7,
        limit: int = 50
    ) -> List[Dict]:
        """Get items that need human review based on confidence"""
        
        from backend.services.explainability_service import ExtractionEvidence
        
        low_confidence_items = self.db.query(ExtractionEvidence).filter(
            ExtractionEvidence.confidence < confidence_threshold
        ).order_by(ExtractionEvidence.confidence).limit(limit).all()
        
        return [
            {
                "id": item.id,
                "entity_type": item.entity_type,
                "entity_value": item.entity_value,
                "source_text": item.source_text,
                "confidence": item.confidence,
                "document_id": item.document_id,
                "extraction_method": item.extraction_method
            }
            for item in low_confidence_items
        ]
    
    def submit_review(
        self,
        evidence_id: int,
        is_correct: bool,
        corrected_value: str = None,
        reviewer: str = None
    ) -> Optional[Correction]:
        """Submit a review for a low-confidence extraction"""
        
        from backend.services.explainability_service import ExtractionEvidence
        
        evidence = self.db.query(ExtractionEvidence).filter(
            ExtractionEvidence.id == evidence_id
        ).first()
        
        if not evidence:
            return None
        
        if is_correct:
            # Mark as verified (implicitly correct)
            logger.info(f"Extraction {evidence_id} confirmed correct by {reviewer}")
            return None
        else:
            # Create correction
            return self.correct_entity(
                entity_type=evidence.entity_type,
                entity_id=evidence.id,
                field_name="value",
                original_value=evidence.entity_value,
                corrected_value=corrected_value,
                corrected_by=reviewer or "reviewer",
                document_id=evidence.document_id,
                confidence_before=evidence.confidence
            )
