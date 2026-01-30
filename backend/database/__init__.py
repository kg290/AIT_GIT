"""
Database Package
Provides database models, connection management, and session handling
"""
from backend.database.connection import get_db, db_manager, init_database
from backend.database.models import (
    Base, User, UserRole, Patient, Prescription, PrescriptionStatus,
    PrescriptionMedication, PatientMedication, Allergy, Condition,
    TimelineEvent, SafetyAlert, AlertSeverity, AuditLog, SystemSetting, DrugDatabase
)


def init_db():
    """Initialize database - backwards compatible wrapper"""
    db_manager.init_db()


def get_session_local():
    """Get SessionLocal after initialization"""
    if db_manager.SessionLocal is None:
        db_manager.init_db()
    return db_manager.SessionLocal


# Lazy property for backwards compatibility
class _SessionLocalProxy:
    def __call__(self):
        return get_session_local()()
    
    def __getattr__(self, name):
        return getattr(get_session_local(), name)


SessionLocal = _SessionLocalProxy()


__all__ = [
    'get_db', 'db_manager', 'init_database', 'init_db', 'SessionLocal',
    'Base', 'User', 'UserRole', 'Patient', 'Prescription', 'PrescriptionStatus',
    'PrescriptionMedication', 'PatientMedication', 'Allergy', 'Condition',
    'TimelineEvent', 'SafetyAlert', 'AlertSeverity', 'AuditLog', 'SystemSetting', 'DrugDatabase'
]
