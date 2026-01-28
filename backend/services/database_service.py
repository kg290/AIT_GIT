"""
Database Service - Repository pattern for database operations
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, and_, or_, func, desc
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import uuid

from backend.models.schema import (
    Base, Patient, Doctor, Prescription, PrescriptionMedication,
    DrugInteractionRecord, AllergyAlert, AuditLog, Analytics, Allergy
)
from backend.config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database operations service"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    # ==================== PATIENT OPERATIONS ====================
    
    def create_patient(self, **data) -> Patient:
        """Create a new patient"""
        with self.get_session() as session:
            # Generate MRN if not provided
            if 'mrn' not in data:
                data['mrn'] = f"MRN{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"
            
            # Combine names
            if 'first_name' in data and 'last_name' in data:
                data['full_name'] = f"{data['first_name']} {data['last_name']}"
            
            patient = Patient(**data)
            session.add(patient)
            session.flush()
            
            self._log_audit(session, 'create', 'patient', str(patient.id), 
                          f"Created patient: {patient.full_name}")
            
            return patient.to_dict()
    
    def get_patient(self, patient_id: int) -> Optional[Dict]:
        """Get patient by ID"""
        with self.get_session() as session:
            patient = session.query(Patient).filter(Patient.id == patient_id).first()
            return patient.to_dict() if patient else None
    
    def get_patient_by_mrn(self, mrn: str) -> Optional[Dict]:
        """Get patient by MRN"""
        with self.get_session() as session:
            patient = session.query(Patient).filter(Patient.mrn == mrn).first()
            return patient.to_dict() if patient else None
    
    def search_patients(self, query: str, limit: int = 20) -> List[Dict]:
        """Search patients by name, MRN, or phone"""
        with self.get_session() as session:
            patients = session.query(Patient).filter(
                or_(
                    Patient.full_name.ilike(f"%{query}%"),
                    Patient.mrn.ilike(f"%{query}%"),
                    Patient.phone.ilike(f"%{query}%"),
                    Patient.external_id.ilike(f"%{query}%")
                )
            ).limit(limit).all()
            return [p.to_dict() for p in patients]
    
    def update_patient(self, patient_id: int, **data) -> Optional[Dict]:
        """Update patient"""
        with self.get_session() as session:
            patient = session.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                return None
            
            for key, value in data.items():
                if hasattr(patient, key):
                    setattr(patient, key, value)
            
            self._log_audit(session, 'update', 'patient', str(patient_id), 
                          f"Updated patient: {patient.full_name}")
            
            return patient.to_dict()
    
    def add_patient_allergy(self, patient_id: int, allergy_name: str, 
                           category: str = 'drug', severity: str = 'moderate') -> bool:
        """Add allergy to patient"""
        with self.get_session() as session:
            patient = session.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                return False
            
            # Find or create allergy
            allergy = session.query(Allergy).filter(Allergy.name == allergy_name).first()
            if not allergy:
                allergy = Allergy(name=allergy_name, category=category, severity=severity)
                session.add(allergy)
                session.flush()
            
            if allergy not in patient.allergies:
                patient.allergies.append(allergy)
            
            return True
    
    def get_patient_prescriptions(self, patient_id: int, limit: int = 50) -> List[Dict]:
        """Get patient's prescription history"""
        with self.get_session() as session:
            prescriptions = session.query(Prescription).filter(
                Prescription.patient_id == patient_id
            ).order_by(desc(Prescription.created_at)).limit(limit).all()
            return [p.to_dict() for p in prescriptions]
    
    def get_patient_medications(self, patient_id: int, active_only: bool = True) -> List[Dict]:
        """Get patient's current medications"""
        with self.get_session() as session:
            query = session.query(PrescriptionMedication).join(Prescription).filter(
                Prescription.patient_id == patient_id
            )
            
            if active_only:
                query = query.filter(PrescriptionMedication.is_active == True)
            
            meds = query.order_by(desc(PrescriptionMedication.created_at)).all()
            return [m.to_dict() for m in meds]
    
    # ==================== PRESCRIPTION OPERATIONS ====================
    
    def create_prescription(self, document_id: str, **data) -> Dict:
        """Create a new prescription record"""
        with self.get_session() as session:
            prescription = Prescription(document_id=document_id, **data)
            session.add(prescription)
            session.flush()
            
            self._log_audit(session, 'create', 'prescription', document_id,
                          f"Created prescription")
            
            return prescription.to_dict()
    
    def get_prescription(self, prescription_id: int = None, document_id: str = None) -> Optional[Dict]:
        """Get prescription by ID or document_id"""
        with self.get_session() as session:
            query = session.query(Prescription)
            if prescription_id:
                query = query.filter(Prescription.id == prescription_id)
            elif document_id:
                query = query.filter(Prescription.document_id == document_id)
            else:
                return None
            
            prescription = query.first()
            return prescription.to_dict() if prescription else None
    
    def update_prescription(self, document_id: str, **data) -> Optional[Dict]:
        """Update prescription"""
        with self.get_session() as session:
            prescription = session.query(Prescription).filter(
                Prescription.document_id == document_id
            ).first()
            
            if not prescription:
                return None
            
            for key, value in data.items():
                if hasattr(prescription, key):
                    setattr(prescription, key, value)
            
            return prescription.to_dict()
    
    def add_prescription_medication(self, prescription_id: int, **med_data) -> Dict:
        """Add medication to prescription"""
        with self.get_session() as session:
            medication = PrescriptionMedication(prescription_id=prescription_id, **med_data)
            session.add(medication)
            session.flush()
            return medication.to_dict()
    
    def add_drug_interaction(self, prescription_id: int, drug1: str, drug2: str,
                            severity: str, description: str, 
                            mechanism: str = None, management: str = None) -> Dict:
        """Record drug interaction"""
        with self.get_session() as session:
            interaction = DrugInteractionRecord(
                prescription_id=prescription_id,
                drug1=drug1,
                drug2=drug2,
                severity=severity,
                description=description,
                mechanism=mechanism,
                management=management
            )
            session.add(interaction)
            session.flush()
            
            # Update prescription flag
            session.query(Prescription).filter(
                Prescription.id == prescription_id
            ).update({'has_interactions': True})
            
            return interaction.to_dict()
    
    def search_prescriptions(self, query: str = None, patient_id: int = None,
                            start_date: datetime = None, end_date: datetime = None,
                            needs_review: bool = None, limit: int = 50) -> List[Dict]:
        """Search prescriptions with filters"""
        with self.get_session() as session:
            q = session.query(Prescription)
            
            if patient_id:
                q = q.filter(Prescription.patient_id == patient_id)
            
            if start_date:
                q = q.filter(Prescription.created_at >= start_date)
            
            if end_date:
                q = q.filter(Prescription.created_at <= end_date)
            
            if needs_review is not None:
                q = q.filter(Prescription.needs_review == needs_review)
            
            prescriptions = q.order_by(desc(Prescription.created_at)).limit(limit).all()
            return [p.to_dict() for p in prescriptions]
    
    def get_prescriptions_needing_review(self, limit: int = 50) -> List[Dict]:
        """Get prescriptions that need review"""
        with self.get_session() as session:
            prescriptions = session.query(Prescription).filter(
                Prescription.needs_review == True,
                Prescription.reviewed_at.is_(None)
            ).order_by(desc(Prescription.created_at)).limit(limit).all()
            return [p.to_dict() for p in prescriptions]
    
    def mark_prescription_reviewed(self, document_id: str, reviewer: str) -> bool:
        """Mark prescription as reviewed"""
        with self.get_session() as session:
            result = session.query(Prescription).filter(
                Prescription.document_id == document_id
            ).update({
                'reviewed_by': reviewer,
                'reviewed_at': datetime.utcnow(),
                'needs_review': False
            })
            
            if result:
                self._log_audit(session, 'review', 'prescription', document_id,
                              f"Prescription reviewed by {reviewer}")
            
            return result > 0
    
    # ==================== DOCTOR OPERATIONS ====================
    
    def create_doctor(self, **data) -> Dict:
        """Create doctor record"""
        with self.get_session() as session:
            doctor = Doctor(**data)
            session.add(doctor)
            session.flush()
            return doctor.to_dict()
    
    def find_or_create_doctor(self, name: str, qualification: str = None,
                             clinic_name: str = None) -> Dict:
        """Find existing doctor or create new"""
        with self.get_session() as session:
            doctor = session.query(Doctor).filter(
                Doctor.name.ilike(f"%{name}%")
            ).first()
            
            if not doctor:
                doctor = Doctor(
                    name=name,
                    qualification=qualification,
                    clinic_name=clinic_name
                )
                session.add(doctor)
                session.flush()
            
            return doctor.to_dict()
    
    # ==================== ANALYTICS ====================
    
    def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics"""
        with self.get_session() as session:
            today = datetime.utcnow().date()
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            stats = {
                'total_patients': session.query(Patient).count(),
                'total_prescriptions': session.query(Prescription).count(),
                'prescriptions_today': session.query(Prescription).filter(
                    func.date(Prescription.created_at) == today
                ).count(),
                'prescriptions_this_week': session.query(Prescription).filter(
                    Prescription.created_at >= week_ago
                ).count(),
                'prescriptions_this_month': session.query(Prescription).filter(
                    Prescription.created_at >= month_ago
                ).count(),
                'pending_reviews': session.query(Prescription).filter(
                    Prescription.needs_review == True,
                    Prescription.reviewed_at.is_(None)
                ).count(),
                'interactions_detected': session.query(DrugInteractionRecord).count(),
                'major_interactions': session.query(DrugInteractionRecord).filter(
                    DrugInteractionRecord.severity.in_(['major', 'contraindicated'])
                ).count(),
                'avg_confidence': session.query(
                    func.avg(Prescription.extraction_confidence)
                ).scalar() or 0
            }
            
            # Top medications
            top_meds = session.query(
                PrescriptionMedication.name,
                func.count(PrescriptionMedication.id).label('count')
            ).group_by(PrescriptionMedication.name).order_by(
                desc('count')
            ).limit(10).all()
            
            stats['top_medications'] = [{'name': m[0], 'count': m[1]} for m in top_meds]
            
            # Prescriptions by day (last 7 days)
            daily_counts = session.query(
                func.date(Prescription.created_at).label('date'),
                func.count(Prescription.id).label('count')
            ).filter(
                Prescription.created_at >= week_ago
            ).group_by(
                func.date(Prescription.created_at)
            ).all()
            
            stats['daily_prescriptions'] = [
                {'date': str(d[0]), 'count': d[1]} for d in daily_counts
            ]
            
            return stats
    
    def get_medication_analytics(self) -> Dict:
        """Get medication analytics"""
        with self.get_session() as session:
            # Most prescribed medications
            top_meds = session.query(
                PrescriptionMedication.name,
                func.count(PrescriptionMedication.id).label('count')
            ).group_by(PrescriptionMedication.name).order_by(
                desc('count')
            ).limit(20).all()
            
            # Common interactions
            top_interactions = session.query(
                DrugInteractionRecord.drug1,
                DrugInteractionRecord.drug2,
                DrugInteractionRecord.severity,
                func.count(DrugInteractionRecord.id).label('count')
            ).group_by(
                DrugInteractionRecord.drug1,
                DrugInteractionRecord.drug2,
                DrugInteractionRecord.severity
            ).order_by(desc('count')).limit(10).all()
            
            return {
                'top_medications': [{'name': m[0], 'count': m[1]} for m in top_meds],
                'top_interactions': [
                    {'drug1': i[0], 'drug2': i[1], 'severity': i[2], 'count': i[3]}
                    for i in top_interactions
                ]
            }
    
    # ==================== AUDIT ====================
    
    def _log_audit(self, session: Session, action_type: str, entity_type: str,
                  entity_id: str, action_detail: str, user_id: str = 'system',
                  user_name: str = 'System'):
        """Log audit entry"""
        log = AuditLog(
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
            action_detail=action_detail,
            user_id=user_id,
            user_name=user_name
        )
        session.add(log)
    
    def get_audit_logs(self, entity_type: str = None, entity_id: str = None,
                      limit: int = 100) -> List[Dict]:
        """Get audit logs"""
        with self.get_session() as session:
            query = session.query(AuditLog)
            
            if entity_type:
                query = query.filter(AuditLog.entity_type == entity_type)
            if entity_id:
                query = query.filter(AuditLog.entity_id == entity_id)
            
            logs = query.order_by(desc(AuditLog.timestamp)).limit(limit).all()
            return [l.to_dict() for l in logs]


# Create singleton instance
db_service = DatabaseService()
