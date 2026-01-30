"""
Enhanced Temporal Medical Reasoning Service
Build timelines, track medication changes, detect overlaps
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from dataclasses import dataclass, asdict
from enum import Enum

from backend.models.knowledge_graph import PatientMedicationHistory, VisitSummary

logger = logging.getLogger(__name__)


class TimelineEventType(Enum):
    MEDICATION_START = "medication_start"
    MEDICATION_STOP = "medication_stop"
    MEDICATION_CHANGE = "medication_change"
    DIAGNOSIS_NEW = "diagnosis_new"
    DIAGNOSIS_RESOLVED = "diagnosis_resolved"
    SYMPTOM_REPORTED = "symptom_reported"
    SYMPTOM_RESOLVED = "symptom_resolved"
    VISIT = "visit"
    LAB_RESULT = "lab_result"
    VITAL_RECORDED = "vital_recorded"
    PRESCRIPTION = "prescription"


@dataclass
class TimelineEvent:
    """A single event in the patient timeline"""
    event_type: str
    timestamp: datetime
    title: str
    description: str
    data: Dict[str, Any]
    source_document_id: Optional[int] = None
    confidence: float = 1.0


@dataclass
class MedicationOverlap:
    """Detected overlap between medications"""
    medication1: str
    medication2: str
    overlap_start: datetime
    overlap_end: Optional[datetime]
    overlap_days: int
    risk_level: str  # low, moderate, high
    interaction_info: Optional[Dict] = None


@dataclass
class MedicationChange:
    """Change in medication between visits"""
    medication_name: str
    change_type: str  # added, removed, dosage_changed, frequency_changed
    old_value: Optional[str]
    new_value: Optional[str]
    change_date: datetime
    reason: Optional[str] = None


class EnhancedTemporalReasoningService:
    """
    Advanced temporal reasoning for medical data
    
    Features:
    - Build chronological timeline of prescriptions
    - Track when medications start and stop
    - Detect overlapping medications
    - Identify changes across visits
    - Compare current prescription with past history
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== Timeline Building ====================
    
    def build_patient_timeline(
        self,
        patient_id: int,
        start_date: datetime = None,
        end_date: datetime = None,
        event_types: List[TimelineEventType] = None
    ) -> List[TimelineEvent]:
        """Build a comprehensive chronological timeline for a patient"""
        
        events = []
        
        # Get medication history
        med_query = self.db.query(PatientMedicationHistory).filter(
            PatientMedicationHistory.patient_id == patient_id
        )
        
        if start_date:
            med_query = med_query.filter(
                or_(
                    PatientMedicationHistory.start_date >= start_date,
                    PatientMedicationHistory.end_date >= start_date
                )
            )
        
        medications = med_query.all()
        
        for med in medications:
            # Medication start event
            if med.start_date:
                events.append(TimelineEvent(
                    event_type=TimelineEventType.MEDICATION_START.value,
                    timestamp=med.start_date,
                    title=f"Started {med.medication_name}",
                    description=f"Dosage: {med.dosage or 'N/A'}, Frequency: {med.frequency or 'N/A'}",
                    data={
                        "medication_id": med.id,
                        "medication_name": med.medication_name,
                        "dosage": med.dosage,
                        "frequency": med.frequency,
                        "prescribing_doctor": med.prescribing_doctor
                    },
                    source_document_id=med.source_document_id
                ))
            
            # Medication stop event
            if med.end_date:
                events.append(TimelineEvent(
                    event_type=TimelineEventType.MEDICATION_STOP.value,
                    timestamp=med.end_date,
                    title=f"Stopped {med.medication_name}",
                    description=med.discontinuation_reason or "Medication discontinued",
                    data={
                        "medication_id": med.id,
                        "medication_name": med.medication_name,
                        "reason": med.discontinuation_reason,
                        "status": med.status
                    },
                    source_document_id=med.source_document_id
                ))
        
        # Get visit summaries
        visit_query = self.db.query(VisitSummary).filter(
            VisitSummary.patient_id == patient_id
        )
        
        if start_date:
            visit_query = visit_query.filter(VisitSummary.visit_date >= start_date)
        if end_date:
            visit_query = visit_query.filter(VisitSummary.visit_date <= end_date)
        
        visits = visit_query.all()
        
        for visit in visits:
            events.append(TimelineEvent(
                event_type=TimelineEventType.VISIT.value,
                timestamp=visit.visit_date,
                title=f"Visit - {visit.doctor_name or 'Doctor'}",
                description=visit.summary_text or "Clinical visit",
                data={
                    "visit_id": visit.id,
                    "doctor_name": visit.doctor_name,
                    "clinic_name": visit.clinic_name,
                    "diagnoses": visit.diagnoses,
                    "medications_prescribed": visit.medications_prescribed
                },
                source_document_id=visit.source_document_id
            ))
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        # Filter by event types if specified
        if event_types:
            type_values = [et.value for et in event_types]
            events = [e for e in events if e.event_type in type_values]
        
        return events
    
    # ==================== Medication Tracking ====================
    
    def get_medication_timeline(
        self,
        patient_id: int,
        medication_name: str = None
    ) -> List[Dict[str, Any]]:
        """Get timeline of a specific medication or all medications"""
        
        query = self.db.query(PatientMedicationHistory).filter(
            PatientMedicationHistory.patient_id == patient_id
        )
        
        if medication_name:
            query = query.filter(
                PatientMedicationHistory.medication_name.ilike(f"%{medication_name}%")
            )
        
        medications = query.order_by(PatientMedicationHistory.start_date).all()
        
        timeline = []
        for med in medications:
            timeline.append({
                "medication_name": med.medication_name,
                "generic_name": med.generic_name,
                "start_date": med.start_date.isoformat() if med.start_date else None,
                "end_date": med.end_date.isoformat() if med.end_date else None,
                "duration_days": self._calculate_duration(med.start_date, med.end_date),
                "is_active": med.is_active,
                "dosage": med.dosage,
                "frequency": med.frequency,
                "status": med.status,
                "prescribing_doctor": med.prescribing_doctor
            })
        
        return timeline
    
    def _calculate_duration(self, start: datetime, end: datetime) -> Optional[int]:
        """Calculate duration in days"""
        if not start:
            return None
        end_date = end or datetime.utcnow()
        return (end_date - start).days
    
    # ==================== Overlap Detection ====================
    
    def detect_medication_overlaps(
        self,
        patient_id: int,
        check_interactions: bool = True
    ) -> List[MedicationOverlap]:
        """Detect overlapping medications for a patient"""
        
        medications = self.db.query(PatientMedicationHistory).filter(
            PatientMedicationHistory.patient_id == patient_id
        ).all()
        
        overlaps = []
        
        # Compare each pair of medications
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                overlap = self._check_overlap(med1, med2)
                if overlap:
                    overlaps.append(overlap)
        
        return overlaps
    
    def _check_overlap(
        self,
        med1: PatientMedicationHistory,
        med2: PatientMedicationHistory
    ) -> Optional[MedicationOverlap]:
        """Check if two medications overlap in time"""
        
        if not med1.start_date or not med2.start_date:
            return None
        
        # Determine time ranges
        end1 = med1.end_date or datetime.utcnow()
        end2 = med2.end_date or datetime.utcnow()
        
        # Check for overlap
        overlap_start = max(med1.start_date, med2.start_date)
        overlap_end = min(end1, end2)
        
        if overlap_start < overlap_end:
            overlap_days = (overlap_end - overlap_start).days
            
            # Both currently active = potentially higher risk
            risk_level = "moderate" if med1.is_active and med2.is_active else "low"
            
            return MedicationOverlap(
                medication1=med1.medication_name,
                medication2=med2.medication_name,
                overlap_start=overlap_start,
                overlap_end=overlap_end if overlap_end != datetime.utcnow() else None,
                overlap_days=overlap_days,
                risk_level=risk_level
            )
        
        return None
    
    def get_concurrent_medications(
        self,
        patient_id: int,
        at_date: datetime = None
    ) -> List[PatientMedicationHistory]:
        """Get all medications active at a specific date"""
        
        check_date = at_date or datetime.utcnow()
        
        medications = self.db.query(PatientMedicationHistory).filter(
            and_(
                PatientMedicationHistory.patient_id == patient_id,
                PatientMedicationHistory.start_date <= check_date,
                or_(
                    PatientMedicationHistory.end_date.is_(None),
                    PatientMedicationHistory.end_date >= check_date
                )
            )
        ).all()
        
        return medications
    
    # ==================== Change Detection ====================
    
    def detect_medication_changes(
        self,
        patient_id: int,
        visit1_id: int,
        visit2_id: int
    ) -> List[MedicationChange]:
        """Detect medication changes between two visits"""
        
        visit1 = self.db.query(VisitSummary).filter(VisitSummary.id == visit1_id).first()
        visit2 = self.db.query(VisitSummary).filter(VisitSummary.id == visit2_id).first()
        
        if not visit1 or not visit2:
            return []
        
        changes = []
        
        # Get medications from each visit
        meds1 = {self._normalize_med_name(m): m for m in (visit1.medications_prescribed or [])}
        meds2 = {self._normalize_med_name(m): m for m in (visit2.medications_prescribed or [])}
        
        # Find added medications
        for name, med in meds2.items():
            if name not in meds1:
                changes.append(MedicationChange(
                    medication_name=med.get('name', med) if isinstance(med, dict) else med,
                    change_type="added",
                    old_value=None,
                    new_value=str(med),
                    change_date=visit2.visit_date
                ))
        
        # Find removed medications
        for name, med in meds1.items():
            if name not in meds2:
                changes.append(MedicationChange(
                    medication_name=med.get('name', med) if isinstance(med, dict) else med,
                    change_type="removed",
                    old_value=str(med),
                    new_value=None,
                    change_date=visit2.visit_date
                ))
        
        # Find dosage changes
        for name in meds1.keys() & meds2.keys():
            old_med = meds1[name]
            new_med = meds2[name]
            
            if isinstance(old_med, dict) and isinstance(new_med, dict):
                if old_med.get('dosage') != new_med.get('dosage'):
                    changes.append(MedicationChange(
                        medication_name=name,
                        change_type="dosage_changed",
                        old_value=old_med.get('dosage'),
                        new_value=new_med.get('dosage'),
                        change_date=visit2.visit_date
                    ))
                
                if old_med.get('frequency') != new_med.get('frequency'):
                    changes.append(MedicationChange(
                        medication_name=name,
                        change_type="frequency_changed",
                        old_value=old_med.get('frequency'),
                        new_value=new_med.get('frequency'),
                        change_date=visit2.visit_date
                    ))
        
        return changes
    
    def _normalize_med_name(self, med: Any) -> str:
        """Normalize medication name for comparison"""
        if isinstance(med, dict):
            name = med.get('name', '')
        else:
            name = str(med)
        return name.lower().strip()
    
    # ==================== History Comparison ====================
    
    def compare_with_history(
        self,
        patient_id: int,
        current_medications: List[Dict]
    ) -> Dict[str, Any]:
        """Compare current prescription with patient's medication history"""
        
        # Get historical medications
        history = self.db.query(PatientMedicationHistory).filter(
            PatientMedicationHistory.patient_id == patient_id
        ).all()
        
        historical_meds = {m.medication_name.lower(): m for m in history}
        
        comparison = {
            "new_medications": [],
            "previously_taken": [],
            "restarted": [],
            "dosage_changes": [],
            "potential_concerns": []
        }
        
        for med in current_medications:
            med_name = med.get('name', '').lower() if isinstance(med, dict) else str(med).lower()
            
            if med_name in historical_meds:
                hist_med = historical_meds[med_name]
                
                if hist_med.is_active:
                    comparison["previously_taken"].append({
                        "medication": med_name,
                        "status": "continuing",
                        "since": hist_med.start_date.isoformat() if hist_med.start_date else None
                    })
                else:
                    comparison["restarted"].append({
                        "medication": med_name,
                        "previous_end_date": hist_med.end_date.isoformat() if hist_med.end_date else None,
                        "discontinuation_reason": hist_med.discontinuation_reason
                    })
                    
                    # Flag if previously discontinued for adverse reason
                    if hist_med.discontinuation_reason and 'adverse' in hist_med.discontinuation_reason.lower():
                        comparison["potential_concerns"].append({
                            "medication": med_name,
                            "concern": "Previously discontinued due to adverse effects",
                            "details": hist_med.discontinuation_reason
                        })
            else:
                comparison["new_medications"].append({
                    "medication": med_name,
                    "status": "new_to_patient"
                })
        
        return comparison
    
    # ==================== Timeline Visualization Data ====================
    
    def get_gantt_chart_data(
        self,
        patient_id: int,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """Get medication data formatted for Gantt chart visualization"""
        
        query = self.db.query(PatientMedicationHistory).filter(
            PatientMedicationHistory.patient_id == patient_id
        )
        
        if start_date:
            query = query.filter(
                or_(
                    PatientMedicationHistory.start_date >= start_date,
                    PatientMedicationHistory.end_date >= start_date,
                    and_(
                        PatientMedicationHistory.end_date.is_(None),
                        PatientMedicationHistory.is_active == True
                    )
                )
            )
        
        medications = query.order_by(PatientMedicationHistory.start_date).all()
        
        gantt_data = []
        for med in medications:
            gantt_data.append({
                "task": med.medication_name,
                "start": med.start_date.isoformat() if med.start_date else None,
                "end": (med.end_date or datetime.utcnow()).isoformat(),
                "is_active": med.is_active,
                "status": med.status,
                "dosage": med.dosage,
                "category": "active" if med.is_active else "completed"
            })
        
        return gantt_data
    
    def get_timeline_summary(self, patient_id: int) -> Dict[str, Any]:
        """Get summary statistics for patient timeline"""
        
        medications = self.db.query(PatientMedicationHistory).filter(
            PatientMedicationHistory.patient_id == patient_id
        ).all()
        
        visits = self.db.query(VisitSummary).filter(
            VisitSummary.patient_id == patient_id
        ).all()
        
        active_meds = [m for m in medications if m.is_active]
        
        # Calculate average medication duration
        durations = []
        for med in medications:
            if med.start_date:
                end = med.end_date or datetime.utcnow()
                durations.append((end - med.start_date).days)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_medications_ever": len(medications),
            "active_medications": len(active_meds),
            "discontinued_medications": len(medications) - len(active_meds),
            "total_visits": len(visits),
            "average_medication_duration_days": round(avg_duration, 1),
            "first_visit": min(v.visit_date for v in visits).isoformat() if visits else None,
            "last_visit": max(v.visit_date for v in visits).isoformat() if visits else None,
            "overlapping_medications": len(self.detect_medication_overlaps(patient_id))
        }
