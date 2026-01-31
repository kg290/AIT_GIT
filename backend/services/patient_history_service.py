"""
Patient History Service - Longitudinal patient data management
Maintains full medication, symptom, diagnosis history with visit summaries
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from backend.models.patient import Patient
from backend.models.patient_history import (
    PatientMedicationHistory, PatientConditionHistory,
    PatientSymptomHistory, VisitSummary
)

logger = logging.getLogger(__name__)


class PatientHistoryService:
    """
    Manages longitudinal patient history
    
    Features:
    - Full medication history tracking
    - Symptom progression history
    - Diagnosis history
    - Visit-wise summaries
    - Historical state preservation
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== Medication History ====================
    
    def add_medication(
        self,
        patient_id: int,
        medication_name: str,
        dosage: str = None,
        frequency: str = None,
        start_date: datetime = None,
        prescription_id: int = None,
        prescribing_doctor: str = None,
        source_document_id: int = None,
        generic_name: str = None,
        route: str = None
    ) -> PatientMedicationHistory:
        """Add a medication to patient history"""
        
        # Check if medication already active
        existing = self.db.query(PatientMedicationHistory).filter(
            and_(
                PatientMedicationHistory.patient_id == patient_id,
                PatientMedicationHistory.medication_name.ilike(f"%{medication_name}%"),
                PatientMedicationHistory.is_active == True
            )
        ).first()
        
        if existing:
            # Update existing record
            existing.dosage = dosage or existing.dosage
            existing.frequency = frequency or existing.frequency
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing
        
        # Create new record
        med_history = PatientMedicationHistory(
            patient_id=patient_id,
            medication_name=medication_name,
            generic_name=generic_name,
            dosage=dosage,
            frequency=frequency,
            route=route,
            prescription_id=prescription_id,
            prescribing_doctor=prescribing_doctor,
            start_date=start_date or datetime.utcnow(),
            is_active=True,
            status="active",
            source_document_id=source_document_id
        )
        
        self.db.add(med_history)
        self.db.commit()
        self.db.refresh(med_history)
        
        logger.info(f"Added medication {medication_name} for patient {patient_id}")
        return med_history
    
    def stop_medication(
        self,
        patient_id: int,
        medication_name: str,
        end_date: datetime = None,
        reason: str = None
    ) -> Optional[PatientMedicationHistory]:
        """Mark a medication as stopped"""
        
        med = self.db.query(PatientMedicationHistory).filter(
            and_(
                PatientMedicationHistory.patient_id == patient_id,
                PatientMedicationHistory.medication_name.ilike(f"%{medication_name}%"),
                PatientMedicationHistory.is_active == True
            )
        ).first()
        
        if med:
            med.is_active = False
            med.end_date = end_date or datetime.utcnow()
            med.status = "discontinued"
            med.discontinuation_reason = reason
            med.updated_at = datetime.utcnow()
            self.db.commit()
            logger.info(f"Stopped medication {medication_name} for patient {patient_id}")
            return med
        
        return None
    
    def get_active_medications(self, patient_id: int) -> List[PatientMedicationHistory]:
        """Get all active medications for a patient"""
        return self.db.query(PatientMedicationHistory).filter(
            and_(
                PatientMedicationHistory.patient_id == patient_id,
                PatientMedicationHistory.is_active == True
            )
        ).order_by(desc(PatientMedicationHistory.start_date)).all()
    
    def get_medication_history(
        self,
        patient_id: int,
        include_inactive: bool = True,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[PatientMedicationHistory]:
        """Get full medication history for a patient"""
        
        query = self.db.query(PatientMedicationHistory).filter(
            PatientMedicationHistory.patient_id == patient_id
        )
        
        if not include_inactive:
            query = query.filter(PatientMedicationHistory.is_active == True)
        
        if start_date:
            query = query.filter(PatientMedicationHistory.start_date >= start_date)
        
        if end_date:
            query = query.filter(
                or_(
                    PatientMedicationHistory.end_date <= end_date,
                    PatientMedicationHistory.end_date.is_(None)
                )
            )
        
        return query.order_by(desc(PatientMedicationHistory.start_date)).all()
    
    # ==================== Condition History ====================
    
    def add_condition(
        self,
        patient_id: int,
        condition_name: str,
        icd_code: str = None,
        severity: str = None,
        onset_date: datetime = None,
        is_chronic: bool = False,
        diagnosed_by: str = None,
        source_document_id: int = None
    ) -> PatientConditionHistory:
        """Add a condition/diagnosis to patient history"""
        
        # Check if condition already exists
        existing = self.db.query(PatientConditionHistory).filter(
            and_(
                PatientConditionHistory.patient_id == patient_id,
                PatientConditionHistory.condition_name.ilike(f"%{condition_name}%"),
                PatientConditionHistory.is_active == True
            )
        ).first()
        
        if existing:
            # Update existing
            existing.severity = severity or existing.severity
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing
        
        condition = PatientConditionHistory(
            patient_id=patient_id,
            condition_name=condition_name,
            icd_code=icd_code,
            severity=severity,
            onset_date=onset_date or datetime.utcnow(),
            is_active=True,
            is_chronic=is_chronic,
            diagnosed_by=diagnosed_by,
            source_document_id=source_document_id
        )
        
        self.db.add(condition)
        self.db.commit()
        self.db.refresh(condition)
        
        logger.info(f"Added condition {condition_name} for patient {patient_id}")
        return condition
    
    def resolve_condition(
        self,
        patient_id: int,
        condition_name: str,
        resolution_date: datetime = None
    ) -> Optional[PatientConditionHistory]:
        """Mark a condition as resolved"""
        
        condition = self.db.query(PatientConditionHistory).filter(
            and_(
                PatientConditionHistory.patient_id == patient_id,
                PatientConditionHistory.condition_name.ilike(f"%{condition_name}%"),
                PatientConditionHistory.is_active == True
            )
        ).first()
        
        if condition:
            condition.is_active = False
            condition.resolution_date = resolution_date or datetime.utcnow()
            condition.updated_at = datetime.utcnow()
            self.db.commit()
            return condition
        
        return None
    
    def get_active_conditions(self, patient_id: int) -> List[PatientConditionHistory]:
        """Get all active conditions for a patient"""
        return self.db.query(PatientConditionHistory).filter(
            and_(
                PatientConditionHistory.patient_id == patient_id,
                PatientConditionHistory.is_active == True
            )
        ).all()
    
    def get_chronic_conditions(self, patient_id: int) -> List[PatientConditionHistory]:
        """Get all chronic conditions for a patient"""
        return self.db.query(PatientConditionHistory).filter(
            and_(
                PatientConditionHistory.patient_id == patient_id,
                PatientConditionHistory.is_chronic == True
            )
        ).all()
    
    # ==================== Symptom History ====================
    
    def add_symptom(
        self,
        patient_id: int,
        symptom_name: str,
        severity: str = None,
        frequency: str = None,
        reported_date: datetime = None,
        associated_condition_id: int = None,
        associated_medication_id: int = None,
        source_document_id: int = None
    ) -> PatientSymptomHistory:
        """Add a symptom to patient history"""
        
        symptom = PatientSymptomHistory(
            patient_id=patient_id,
            symptom_name=symptom_name,
            severity=severity,
            frequency=frequency,
            reported_date=reported_date or datetime.utcnow(),
            is_active=True,
            associated_condition_id=associated_condition_id,
            associated_medication_id=associated_medication_id,
            source_document_id=source_document_id
        )
        
        self.db.add(symptom)
        self.db.commit()
        self.db.refresh(symptom)
        
        return symptom
    
    def resolve_symptom(
        self,
        symptom_id: int,
        resolution_date: datetime = None
    ) -> Optional[PatientSymptomHistory]:
        """Mark a symptom as resolved"""
        
        symptom = self.db.query(PatientSymptomHistory).filter(
            PatientSymptomHistory.id == symptom_id
        ).first()
        
        if symptom:
            symptom.is_active = False
            symptom.resolution_date = resolution_date or datetime.utcnow()
            self.db.commit()
            return symptom
        
        return None
    
    def get_active_symptoms(self, patient_id: int) -> List[PatientSymptomHistory]:
        """Get all active symptoms for a patient"""
        return self.db.query(PatientSymptomHistory).filter(
            and_(
                PatientSymptomHistory.patient_id == patient_id,
                PatientSymptomHistory.is_active == True
            )
        ).all()
    
    # ==================== Visit Summaries ====================
    
    def create_visit_summary(
        self,
        patient_id: int,
        visit_date: datetime,
        doctor_name: str = None,
        clinic_name: str = None,
        chief_complaints: List[str] = None,
        diagnoses: List[str] = None,
        medications_prescribed: List[Dict] = None,
        medications_continued: List[Dict] = None,
        medications_stopped: List[Dict] = None,
        vitals: Dict = None,
        investigations_ordered: List[str] = None,
        advice: List[str] = None,
        follow_up: str = None,
        source_document_id: int = None
    ) -> VisitSummary:
        """Create a visit summary preserving the patient state at that time"""
        
        summary = VisitSummary(
            patient_id=patient_id,
            visit_date=visit_date,
            visit_type="regular",
            doctor_name=doctor_name,
            clinic_name=clinic_name,
            chief_complaints=chief_complaints or [],
            diagnoses=diagnoses or [],
            medications_prescribed=medications_prescribed or [],
            medications_continued=medications_continued or [],
            medications_stopped=medications_stopped or [],
            vitals=vitals or {},
            investigations_ordered=investigations_ordered or [],
            advice=advice or [],
            follow_up=follow_up,
            source_document_id=source_document_id
        )
        
        # Generate summary text
        summary.summary_text = self._generate_summary_text(summary)
        
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        
        logger.info(f"Created visit summary for patient {patient_id} on {visit_date}")
        return summary
    
    def _generate_summary_text(self, summary: VisitSummary) -> str:
        """Generate human-readable summary text"""
        parts = []
        
        if summary.visit_date:
            parts.append(f"Visit Date: {summary.visit_date.strftime('%Y-%m-%d')}")
        
        if summary.doctor_name:
            parts.append(f"Doctor: {summary.doctor_name}")
        
        if summary.chief_complaints:
            parts.append(f"Chief Complaints: {', '.join(summary.chief_complaints)}")
        
        if summary.diagnoses:
            parts.append(f"Diagnoses: {', '.join(summary.diagnoses)}")
        
        if summary.medications_prescribed:
            med_list = [m.get('name', m) if isinstance(m, dict) else m for m in summary.medications_prescribed]
            parts.append(f"Medications Prescribed: {', '.join(med_list)}")
        
        if summary.advice:
            parts.append(f"Advice: {', '.join(summary.advice)}")
        
        if summary.follow_up:
            parts.append(f"Follow-up: {summary.follow_up}")
        
        return "\n".join(parts)
    
    def get_visit_history(
        self,
        patient_id: int,
        limit: int = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[VisitSummary]:
        """Get visit history for a patient"""
        
        query = self.db.query(VisitSummary).filter(
            VisitSummary.patient_id == patient_id
        )
        
        if start_date:
            query = query.filter(VisitSummary.visit_date >= start_date)
        
        if end_date:
            query = query.filter(VisitSummary.visit_date <= end_date)
        
        query = query.order_by(desc(VisitSummary.visit_date))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_last_visit(self, patient_id: int) -> Optional[VisitSummary]:
        """Get the most recent visit summary"""
        return self.db.query(VisitSummary).filter(
            VisitSummary.patient_id == patient_id
        ).order_by(desc(VisitSummary.visit_date)).first()
    
    # ==================== Comprehensive History ====================
    
    def get_complete_patient_history(self, patient_id: int) -> Dict[str, Any]:
        """Get complete longitudinal history for a patient"""
        
        return {
            "patient_id": patient_id,
            "medications": {
                "active": [m.to_dict() for m in self.get_active_medications(patient_id)],
                "history": [m.to_dict() for m in self.get_medication_history(patient_id)]
            },
            "conditions": {
                "active": [c.to_dict() for c in self.get_active_conditions(patient_id)],
                "chronic": [c.to_dict() for c in self.get_chronic_conditions(patient_id)]
            },
            "symptoms": {
                "active": [s.to_dict() for s in self.get_active_symptoms(patient_id)]
            },
            "visits": [v.to_dict() for v in self.get_visit_history(patient_id, limit=10)],
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def compare_visits(
        self,
        patient_id: int,
        visit1_id: int,
        visit2_id: int
    ) -> Dict[str, Any]:
        """Compare two visits to identify changes"""
        
        visit1 = self.db.query(VisitSummary).filter(VisitSummary.id == visit1_id).first()
        visit2 = self.db.query(VisitSummary).filter(VisitSummary.id == visit2_id).first()
        
        if not visit1 or not visit2:
            return {"error": "One or both visits not found"}
        
        # Compare medications
        meds1 = set(m.get('name', m) if isinstance(m, dict) else m for m in visit1.medications_prescribed)
        meds2 = set(m.get('name', m) if isinstance(m, dict) else m for m in visit2.medications_prescribed)
        
        return {
            "visit1_date": visit1.visit_date.isoformat() if visit1.visit_date else None,
            "visit2_date": visit2.visit_date.isoformat() if visit2.visit_date else None,
            "medications": {
                "added": list(meds2 - meds1),
                "removed": list(meds1 - meds2),
                "continued": list(meds1 & meds2)
            },
            "diagnoses": {
                "new": list(set(visit2.diagnoses or []) - set(visit1.diagnoses or [])),
                "resolved": list(set(visit1.diagnoses or []) - set(visit2.diagnoses or []))
            }
        }
