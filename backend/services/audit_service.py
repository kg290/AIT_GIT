"""
Audit Service - Logging, compliance, and version tracking
"""
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import hashlib
import uuid

from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.audit import AuditLog, AuditAction, Correction, CorrectionType

logger = logging.getLogger(__name__)


class AuditService:
    """
    Audit and compliance service
    
    Features:
    - Immutable audit logging
    - Version tracking
    - Correction logging
    - Compliance support
    - Change history
    """
    
    def __init__(self, db: Session = None):
        self.db = db
        self.log_path = settings.AUDIT_LOG_PATH
        self.log_path.mkdir(parents=True, exist_ok=True)
    
    def log_action(self, action: AuditAction, entity_type: str, 
                   entity_id: int = None, entity_identifier: str = None,
                   user_id: str = None, user_name: str = None,
                   old_value: Any = None, new_value: Any = None,
                   changes: Dict = None, metadata: Dict = None,
                   request_context: Dict = None, action_detail: str = None) -> AuditLog:
        """
        Create an immutable audit log entry
        
        Args:
            action: Type of action (create, update, delete, etc.)
            entity_type: Type of entity affected
            entity_id: Database ID of entity
            entity_identifier: Business identifier (e.g., patient_id)
            user_id: User performing action
            user_name: Display name of user
            old_value: Previous value (for updates)
            new_value: New value
            changes: Dictionary of field changes
            metadata: Additional context
            request_context: HTTP request info
            action_detail: Additional detail about the action
            
        Returns:
            Created AuditLog entry
        """
        # Create audit entry
        audit_entry = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_identifier=entity_identifier,
            user_id=user_id,
            user_name=user_name or "System",
            old_value=self._serialize_value(old_value),
            new_value=self._serialize_value(new_value),
            changes=changes,
            extra_data=metadata or {},
            action_detail=action_detail,
            timestamp=datetime.utcnow()
        )
        
        # Add request context if provided
        if request_context:
            audit_entry.ip_address = request_context.get('ip_address')
            audit_entry.user_agent = request_context.get('user_agent')
            audit_entry.session_id = request_context.get('session_id')
            audit_entry.request_id = request_context.get('request_id')
        
        # Save to database
        if self.db:
            self.db.add(audit_entry)
            self.db.commit()
            self.db.refresh(audit_entry)
        
        # Also write to file for redundancy
        self._write_to_file(audit_entry)
        
        return audit_entry
    
    def log_document_upload(self, document_id: str, filename: str,
                           patient_id: str = None, user_id: str = None,
                           upload_source: str = None) -> AuditLog:
        """Log document upload"""
        return self.log_action(
            action=AuditAction.UPLOAD,
            entity_type='document',
            entity_identifier=document_id,
            user_id=user_id,
            new_value={'filename': filename, 'patient_id': patient_id},
            metadata={'upload_source': upload_source}
        )
    
    def log_document_processing(self, document_id: str, 
                                processing_type: str,
                                result_summary: Dict,
                                processing_time_ms: int) -> AuditLog:
        """Log document processing (OCR, extraction, etc.)"""
        return self.log_action(
            action=AuditAction.PROCESS,
            action_detail=processing_type,
            entity_type='document',
            entity_identifier=document_id,
            new_value=result_summary,
            metadata={'processing_time_ms': processing_time_ms}
        )
    
    def log_correction(self, correction_type: CorrectionType,
                       document_id: int = None,
                       entity_type: str = None,
                       entity_id: int = None,
                       field_name: str = None,
                       original_value: str = None,
                       corrected_value: str = None,
                       reason: str = None,
                       corrected_by: str = None,
                       confidence_before: float = None) -> Correction:
        """
        Log a human correction to system output
        
        Args:
            correction_type: Type of correction
            document_id: Source document
            entity_type: Type of entity corrected
            entity_id: ID of entity
            field_name: Field that was corrected
            original_value: System's original value
            corrected_value: Human's corrected value
            reason: Reason for correction
            corrected_by: User making correction
            confidence_before: Original confidence score
            
        Returns:
            Created Correction entry
        """
        correction = Correction(
            correction_type=correction_type,
            document_id=document_id,
            entity_id=entity_id,
            entity_type=entity_type,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            reason=reason,
            corrected_by=corrected_by or "Unknown",
            confidence_before=confidence_before,
            corrected_at=datetime.utcnow()
        )
        
        if self.db:
            self.db.add(correction)
            self.db.commit()
            self.db.refresh(correction)
        
        # Also create audit log
        self.log_action(
            action=AuditAction.CORRECT,
            entity_type=entity_type or 'unknown',
            entity_id=entity_id,
            user_name=corrected_by,
            old_value=original_value,
            new_value=corrected_value,
            changes={
                'field': field_name,
                'correction_type': correction_type.value
            },
            metadata={'reason': reason}
        )
        
        return correction
    
    def log_ocr_correction(self, document_id: int,
                           original_text: str,
                           corrected_text: str,
                           corrected_by: str,
                           reason: str = None) -> Correction:
        """Log OCR text correction"""
        return self.log_correction(
            correction_type=CorrectionType.OCR_TEXT,
            document_id=document_id,
            field_name='ocr_text',
            original_value=original_text,
            corrected_value=corrected_text,
            corrected_by=corrected_by,
            reason=reason or "OCR error correction"
        )
    
    def log_entity_correction(self, document_id: int,
                              entity_type: str,
                              entity_id: int,
                              field_name: str,
                              original_value: str,
                              corrected_value: str,
                              corrected_by: str,
                              reason: str = None) -> Correction:
        """Log entity extraction correction"""
        return self.log_correction(
            correction_type=CorrectionType.ENTITY,
            document_id=document_id,
            entity_type=entity_type,
            entity_id=entity_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            corrected_by=corrected_by,
            reason=reason
        )
    
    def log_interaction_dismissal(self, patient_id: int,
                                  interaction_id: int,
                                  dismissed_by: str,
                                  reason: str) -> AuditLog:
        """Log dismissal of drug interaction alert"""
        self.log_correction(
            correction_type=CorrectionType.INTERACTION_DISMISS,
            entity_type='patient_drug_interaction',
            entity_id=interaction_id,
            original_value='active',
            corrected_value='dismissed',
            corrected_by=dismissed_by,
            reason=reason
        )
        
        return self.log_action(
            action=AuditAction.DISMISS,
            entity_type='patient_drug_interaction',
            entity_id=interaction_id,
            user_name=dismissed_by,
            changes={'status': 'dismissed'},
            metadata={'reason': reason, 'patient_id': patient_id}
        )
    
    def log_review(self, entity_type: str, entity_id: int,
                   reviewed_by: str, review_notes: str = None) -> AuditLog:
        """Log human review completion"""
        return self.log_action(
            action=AuditAction.REVIEW,
            entity_type=entity_type,
            entity_id=entity_id,
            user_name=reviewed_by,
            metadata={'review_notes': review_notes}
        )
    
    def get_audit_trail(self, entity_type: str, entity_id: int = None,
                       entity_identifier: str = None,
                       start_date: datetime = None,
                       end_date: datetime = None,
                       limit: int = 100) -> List[AuditLog]:
        """
        Get audit trail for an entity
        
        Args:
            entity_type: Type of entity
            entity_id: Database ID
            entity_identifier: Business identifier
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum entries to return
            
        Returns:
            List of AuditLog entries
        """
        if not self.db:
            return []
        
        query = self.db.query(AuditLog).filter(AuditLog.entity_type == entity_type)
        
        if entity_id:
            query = query.filter(AuditLog.entity_id == entity_id)
        if entity_identifier:
            query = query.filter(AuditLog.entity_identifier == entity_identifier)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    def get_corrections(self, document_id: int = None,
                       entity_type: str = None,
                       correction_type: CorrectionType = None,
                       limit: int = 100) -> List[Correction]:
        """Get corrections history"""
        if not self.db:
            return []
        
        query = self.db.query(Correction)
        
        if document_id:
            query = query.filter(Correction.document_id == document_id)
        if entity_type:
            query = query.filter(Correction.entity_type == entity_type)
        if correction_type:
            query = query.filter(Correction.correction_type == correction_type)
        
        return query.order_by(Correction.corrected_at.desc()).limit(limit).all()
    
    def get_change_history(self, entity_type: str, entity_id: int) -> List[Dict]:
        """Get complete change history for an entity"""
        audit_trail = self.get_audit_trail(entity_type, entity_id)
        
        history = []
        for entry in audit_trail:
            history.append({
                'timestamp': entry.timestamp.isoformat(),
                'action': entry.action.value,
                'user': entry.user_name,
                'changes': entry.changes,
                'old_value': entry.old_value,
                'new_value': entry.new_value
            })
        
        return history
    
    def compute_document_hash(self, content: bytes) -> str:
        """Compute hash of document content for integrity verification"""
        return hashlib.sha256(content).hexdigest()
    
    def verify_document_integrity(self, document_id: str, content: bytes) -> bool:
        """Verify document hasn't been modified"""
        # Get original hash from audit log
        trail = self.get_audit_trail('document', entity_identifier=document_id)
        for entry in trail:
            if entry.action == AuditAction.UPLOAD:
                original_hash = entry.extra_data.get('content_hash') if entry.extra_data else None
                if original_hash:
                    current_hash = self.compute_document_hash(content)
                    return original_hash == current_hash
        return False
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for storage"""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return value
        try:
            return str(value)
        except:
            return repr(value)
    
    def _write_to_file(self, audit_entry: AuditLog):
        """Write audit entry to file for redundancy"""
        try:
            date_str = audit_entry.timestamp.strftime('%Y-%m-%d')
            log_file = self.log_path / f"audit_{date_str}.jsonl"
            
            entry_dict = {
                'id': audit_entry.id,
                'timestamp': audit_entry.timestamp.isoformat(),
                'action': audit_entry.action.value if audit_entry.action else None,
                'entity_type': audit_entry.entity_type,
                'entity_id': audit_entry.entity_id,
                'entity_identifier': audit_entry.entity_identifier,
                'user_id': audit_entry.user_id,
                'user_name': audit_entry.user_name,
                'changes': audit_entry.changes,
                'extra_data': audit_entry.extra_data
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(entry_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit to file: {e}")
    
    def export_audit_report(self, start_date: datetime, end_date: datetime,
                           entity_types: List[str] = None) -> Dict:
        """Export audit report for compliance"""
        if not self.db:
            return {'error': 'No database connection'}
        
        query = self.db.query(AuditLog).filter(
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        )
        
        if entity_types:
            query = query.filter(AuditLog.entity_type.in_(entity_types))
        
        entries = query.all()
        
        # Aggregate statistics
        stats = {
            'total_entries': len(entries),
            'by_action': {},
            'by_entity_type': {},
            'by_user': {}
        }
        
        for entry in entries:
            action = entry.action.value if entry.action else 'unknown'
            stats['by_action'][action] = stats['by_action'].get(action, 0) + 1
            stats['by_entity_type'][entry.entity_type] = stats['by_entity_type'].get(entry.entity_type, 0) + 1
            user = entry.user_name or 'System'
            stats['by_user'][user] = stats['by_user'].get(user, 0) + 1
        
        return {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'statistics': stats,
            'entries': [entry.to_dict() for entry in entries]
        }
