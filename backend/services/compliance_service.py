"""
Compliance & Audit API
Expose audit log endpoints, immutable document storage
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
import hashlib
import json

from backend.models.audit import AuditLog, AuditAction, Correction
from backend.models.document import DocumentVersion

logger = logging.getLogger(__name__)


class ComplianceService:
    """
    Compliance & Audit Service
    
    Features:
    - Immutable audit logging
    - Document versioning
    - Access logging
    - Compliance reporting
    - Data integrity verification
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== Audit Logging ====================
    
    def log_action(
        self,
        action: AuditAction,
        action_detail: str,
        entity_type: str,
        entity_id: int = None,
        user_name: str = None,
        user_ip: str = None,
        user_agent: str = None,
        old_value: Dict = None,
        new_value: Dict = None,
        changes: Dict = None
    ) -> AuditLog:
        """Log an action to the audit trail"""
        
        # Calculate data hash for integrity
        data_hash = self._calculate_hash({
            "action": action.value,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "changes": changes,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        audit = AuditLog(
            action=action,
            action_detail=action_detail,
            entity_type=entity_type,
            entity_id=entity_id,
            user_name=user_name,
            user_ip=user_ip,
            user_agent=user_agent,
            old_value=old_value,
            new_value=new_value,
            changes=changes,
            data_hash=data_hash
        )
        
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(audit)
        
        logger.info(f"Audit: {action.value} on {entity_type}:{entity_id} by {user_name}")
        return audit
    
    def log_document_upload(
        self,
        document_id: int,
        filename: str,
        user_name: str,
        file_hash: str = None,
        user_ip: str = None
    ) -> AuditLog:
        """Log document upload"""
        
        return self.log_action(
            action=AuditAction.UPLOAD,
            action_detail=f"Uploaded document: {filename}",
            entity_type="document",
            entity_id=document_id,
            user_name=user_name,
            user_ip=user_ip,
            new_value={"filename": filename, "file_hash": file_hash}
        )
    
    def log_document_view(
        self,
        document_id: int,
        user_name: str,
        user_ip: str = None
    ) -> AuditLog:
        """Log document view"""
        
        return self.log_action(
            action=AuditAction.VIEW,
            action_detail=f"Viewed document #{document_id}",
            entity_type="document",
            entity_id=document_id,
            user_name=user_name,
            user_ip=user_ip
        )
    
    def log_extraction(
        self,
        document_id: int,
        extraction_result: Dict,
        user_name: str = "system"
    ) -> AuditLog:
        """Log extraction processing"""
        
        return self.log_action(
            action=AuditAction.EXTRACT,
            action_detail=f"Extracted data from document #{document_id}",
            entity_type="document",
            entity_id=document_id,
            user_name=user_name,
            new_value={
                "medications_count": len(extraction_result.get("medications", [])),
                "confidence": extraction_result.get("confidence")
            }
        )
    
    def log_correction(
        self,
        correction_id: int,
        document_id: int,
        field: str,
        old_value: str,
        new_value: str,
        user_name: str
    ) -> AuditLog:
        """Log a correction"""
        
        return self.log_action(
            action=AuditAction.CORRECT,
            action_detail=f"Corrected {field} in document #{document_id}",
            entity_type="correction",
            entity_id=correction_id,
            user_name=user_name,
            old_value={"value": old_value},
            new_value={"value": new_value},
            changes={"field": field}
        )
    
    def log_alert_dismiss(
        self,
        alert_type: str,
        drugs: List[str],
        reason: str,
        user_name: str
    ) -> AuditLog:
        """Log alert dismissal"""
        
        return self.log_action(
            action=AuditAction.DISMISS,
            action_detail=f"Dismissed {alert_type} alert: {', '.join(drugs)}",
            entity_type="alert",
            user_name=user_name,
            changes={"drugs": drugs, "reason": reason}
        )
    
    def log_export(
        self,
        export_type: str,
        record_count: int,
        user_name: str,
        user_ip: str = None
    ) -> AuditLog:
        """Log data export"""
        
        return self.log_action(
            action=AuditAction.EXPORT,
            action_detail=f"Exported {record_count} {export_type} records",
            entity_type="export",
            user_name=user_name,
            user_ip=user_ip,
            changes={"export_type": export_type, "record_count": record_count}
        )
    
    # ==================== Document Versioning ====================
    
    def create_document_version(
        self,
        document_id: int,
        content: bytes,
        content_type: str,
        version_number: int = None,
        created_by: str = None
    ) -> DocumentVersion:
        """Create an immutable document version"""
        
        # Calculate hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Get next version number if not provided
        if version_number is None:
            last_version = self.db.query(func.max(DocumentVersion.version_number)).filter(
                DocumentVersion.document_id == document_id
            ).scalar()
            version_number = (last_version or 0) + 1
        
        version = DocumentVersion(
            document_id=document_id,
            version_number=version_number,
            content=content,
            content_hash=content_hash,
            content_type=content_type,
            created_by=created_by
        )
        
        self.db.add(version)
        self.db.commit()
        self.db.refresh(version)
        
        logger.info(f"Created version {version_number} for document {document_id}")
        return version
    
    def get_document_versions(self, document_id: int) -> List[DocumentVersion]:
        """Get all versions of a document"""
        
        return self.db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).order_by(desc(DocumentVersion.version_number)).all()
    
    def get_document_version(
        self,
        document_id: int,
        version_number: int = None
    ) -> Optional[DocumentVersion]:
        """Get specific document version (latest if version not specified)"""
        
        query = self.db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        )
        
        if version_number:
            return query.filter(
                DocumentVersion.version_number == version_number
            ).first()
        else:
            return query.order_by(desc(DocumentVersion.version_number)).first()
    
    def verify_document_integrity(
        self,
        document_id: int,
        version_number: int = None
    ) -> Dict[str, Any]:
        """Verify document integrity using stored hash"""
        
        version = self.get_document_version(document_id, version_number)
        
        if not version:
            return {"verified": False, "error": "Document version not found"}
        
        # Recalculate hash
        current_hash = hashlib.sha256(version.content).hexdigest()
        
        return {
            "verified": current_hash == version.content_hash,
            "document_id": document_id,
            "version": version.version_number,
            "stored_hash": version.content_hash,
            "calculated_hash": current_hash
        }
    
    # ==================== Audit Trail Queries ====================
    
    def get_audit_trail(
        self,
        entity_type: str = None,
        entity_id: int = None,
        action: AuditAction = None,
        user_name: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLog]:
        """Query audit trail with filters"""
        
        query = self.db.query(AuditLog)
        
        if entity_type:
            query = query.filter(AuditLog.entity_type == entity_type)
        if entity_id:
            query = query.filter(AuditLog.entity_id == entity_id)
        if action:
            query = query.filter(AuditLog.action == action)
        if user_name:
            query = query.filter(AuditLog.user_name == user_name)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        return query.order_by(desc(AuditLog.timestamp)).offset(offset).limit(limit).all()
    
    def get_document_audit_trail(self, document_id: int) -> List[AuditLog]:
        """Get complete audit trail for a document"""
        
        return self.get_audit_trail(entity_type="document", entity_id=document_id)
    
    def get_user_activity(
        self,
        user_name: str,
        days: int = 30
    ) -> List[AuditLog]:
        """Get user activity for compliance review"""
        
        start_date = datetime.utcnow() - timedelta(days=days)
        return self.get_audit_trail(user_name=user_name, start_date=start_date)
    
    # ==================== Compliance Reports ====================
    
    def generate_compliance_report(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get all audit logs in period
        logs = self.get_audit_trail(start_date=start_date, end_date=end_date, limit=10000)
        
        # Aggregate statistics
        action_counts = {}
        user_activity = {}
        entity_counts = {}
        
        for log in logs:
            # Count by action
            action = log.action.value if log.action else "unknown"
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Count by user
            user = log.user_name or "anonymous"
            user_activity[user] = user_activity.get(user, 0) + 1
            
            # Count by entity type
            entity = log.entity_type or "unknown"
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        # Get corrections
        corrections = self.db.query(Correction).filter(
            and_(
                Correction.corrected_at >= start_date,
                Correction.corrected_at <= end_date
            )
        ).count()
        
        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(logs),
            "events_by_action": action_counts,
            "events_by_user": user_activity,
            "events_by_entity": entity_counts,
            "corrections_count": corrections,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def generate_access_report(
        self,
        document_id: int = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Generate access report for document(s)"""
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        query = self.db.query(AuditLog).filter(
            and_(
                AuditLog.action == AuditAction.VIEW,
                AuditLog.timestamp >= start_date
            )
        )
        
        if document_id:
            query = query.filter(
                and_(
                    AuditLog.entity_type == "document",
                    AuditLog.entity_id == document_id
                )
            )
        
        accesses = query.all()
        
        # Aggregate
        by_user = {}
        by_date = {}
        
        for access in accesses:
            user = access.user_name or "anonymous"
            by_user[user] = by_user.get(user, 0) + 1
            
            date = access.timestamp.strftime("%Y-%m-%d")
            by_date[date] = by_date.get(date, 0) + 1
        
        return {
            "total_accesses": len(accesses),
            "accesses_by_user": by_user,
            "accesses_by_date": by_date,
            "document_id": document_id,
            "period_days": days
        }
    
    # ==================== Data Integrity ====================
    
    def verify_audit_chain(self, limit: int = 1000) -> Dict[str, Any]:
        """Verify integrity of audit chain"""
        
        logs = self.db.query(AuditLog).order_by(AuditLog.id).limit(limit).all()
        
        issues = []
        verified = 0
        
        for log in logs:
            if log.data_hash:
                # Recalculate hash
                expected = self._calculate_hash({
                    "action": log.action.value if log.action else None,
                    "entity_type": log.entity_type,
                    "entity_id": log.entity_id,
                    "changes": log.changes,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None
                })
                
                if expected != log.data_hash:
                    issues.append({
                        "log_id": log.id,
                        "issue": "Hash mismatch",
                        "timestamp": log.timestamp.isoformat() if log.timestamp else None
                    })
                else:
                    verified += 1
            else:
                verified += 1  # No hash to verify
        
        return {
            "total_checked": len(logs),
            "verified": verified,
            "issues": issues,
            "integrity_score": verified / len(logs) if logs else 1.0
        }
    
    def _calculate_hash(self, data: Dict) -> str:
        """Calculate SHA-256 hash of data"""
        
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    # ==================== Export for Auditors ====================
    
    def export_audit_trail(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        format: str = "json"
    ) -> Any:
        """Export audit trail for external audit"""
        
        logs = self.get_audit_trail(
            start_date=start_date,
            end_date=end_date,
            limit=50000
        )
        
        export_data = []
        for log in logs:
            export_data.append({
                "id": log.id,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "action": log.action.value if log.action else None,
                "action_detail": log.action_detail,
                "entity_type": log.entity_type,
                "entity_id": log.entity_id,
                "user_name": log.user_name,
                "user_ip": log.user_ip,
                "old_value": log.old_value,
                "new_value": log.new_value,
                "changes": log.changes,
                "data_hash": log.data_hash
            })
        
        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            return export_data
