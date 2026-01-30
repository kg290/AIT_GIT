"""
Production Patient Service
Database-backed patient management with full audit trail
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from backend.database.models import (
    Patient, Prescription, PrescriptionMedication, PatientMedication,
    TimelineEvent, SafetyAlert, Allergy, Condition, User, AlertSeverity,
    PrescriptionStatus
)
from backend.services.drug_normalization_service import DrugNormalizationService
from backend.services.drug_interaction_service import DrugInteractionService
from backend.services.auth_service import audit_service

import uuid


class ProductionPatientService:
    """
    Production-ready patient service with database persistence
    """
    
    def __init__(self):
        self.drug_normalizer = DrugNormalizationService()
        self.drug_interaction_service = DrugInteractionService()
    
    # ==================== Patient CRUD ====================
    
    def create_patient(
        self,
        db: Session,
        patient_uid: str,
        first_name: str,
        last_name: str,
        date_of_birth: datetime = None,
        gender: str = None,
        phone: str = None,
        email: str = None,
        address: str = None,
        allergies: List[str] = None,
        conditions: List[str] = None,
        user: User = None,
        request = None
    ) -> Patient:
        """Create a new patient record"""
        
        # Check if patient UID already exists
        existing = db.query(Patient).filter(Patient.patient_uid == patient_uid).first()
        if existing:
            raise ValueError(f"Patient with ID {patient_uid} already exists")
        
        # Create patient
        patient = Patient(
            patient_uid=patient_uid,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            gender=gender,
            phone=phone,
            email=email,
            address=address
        )
        
        # Add allergies
        if allergies:
            for allergy_name in allergies:
                allergy = self._get_or_create_allergy(db, allergy_name)
                patient.allergies.append(allergy)
        
        # Add conditions
        if conditions:
            for condition_name in conditions:
                condition = self._get_or_create_condition(db, condition_name)
                patient.conditions.append(condition)
        
        db.add(patient)
        db.commit()
        db.refresh(patient)
        
        # Audit log
        audit_service.log(
            db=db,
            action="create",
            resource_type="patient",
            resource_id=patient.id,
            description=f"Created patient {patient.full_name} ({patient_uid})",
            new_values={"patient_uid": patient_uid, "name": patient.full_name},
            user=user,
            request=request
        )
        
        return patient
    
    def get_patient_by_uid(self, db: Session, patient_uid: str) -> Optional[Patient]:
        """Get patient by hospital UID"""
        return db.query(Patient).filter(Patient.patient_uid == patient_uid).first()
    
    def get_patient(self, db: Session, patient_id: int) -> Optional[Patient]:
        """Get patient by database ID"""
        return db.query(Patient).filter(Patient.id == patient_id).first()
    
    def search_patients(
        self,
        db: Session,
        query: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Patient]:
        """Search patients by name, UID, or phone"""
        q = db.query(Patient).filter(Patient.is_active == True)
        
        if query:
            search = f"%{query}%"
            q = q.filter(or_(
                Patient.patient_uid.ilike(search),
                Patient.first_name.ilike(search),
                Patient.last_name.ilike(search),
                Patient.phone.ilike(search)
            ))
        
        return q.order_by(Patient.updated_at.desc()).offset(offset).limit(limit).all()
    
    def update_patient(
        self,
        db: Session,
        patient_id: int,
        updates: Dict[str, Any],
        user: User = None,
        request = None
    ) -> Patient:
        """Update patient information"""
        patient = self.get_patient(db, patient_id)
        if not patient:
            raise ValueError("Patient not found")
        
        old_values = {
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "phone": patient.phone
        }
        
        # Update fields
        allowed_fields = [
            "first_name", "last_name", "date_of_birth", "gender",
            "phone", "email", "address", "weight_kg", "height_cm",
            "emergency_contact_name", "emergency_contact_phone", "notes"
        ]
        
        for field in allowed_fields:
            if field in updates:
                setattr(patient, field, updates[field])
        
        patient.updated_at = datetime.utcnow()
        db.commit()
        
        # Audit log
        audit_service.log(
            db=db,
            action="update",
            resource_type="patient",
            resource_id=patient.id,
            description=f"Updated patient {patient.full_name}",
            old_values=old_values,
            new_values=updates,
            user=user,
            request=request
        )
        
        return patient
    
    def add_patient_allergy(
        self,
        db: Session,
        patient_id: int,
        allergy_name: str,
        user: User = None,
        request = None
    ):
        """Add an allergy to patient"""
        patient = self.get_patient(db, patient_id)
        if not patient:
            raise ValueError("Patient not found")
        
        allergy = self._get_or_create_allergy(db, allergy_name)
        
        if allergy not in patient.allergies:
            patient.allergies.append(allergy)
            db.commit()
            
            # Re-run safety analysis for current medications
            self._recheck_safety(db, patient)
            
            audit_service.log(
                db=db,
                action="add_allergy",
                resource_type="patient",
                resource_id=patient.id,
                description=f"Added allergy {allergy_name} to patient {patient.full_name}",
                user=user,
                request=request
            )
    
    def add_patient_condition(
        self,
        db: Session,
        patient_id: int,
        condition_name: str,
        user: User = None,
        request = None
    ):
        """Add a condition to patient"""
        patient = self.get_patient(db, patient_id)
        if not patient:
            raise ValueError("Patient not found")
        
        condition = self._get_or_create_condition(db, condition_name)
        
        if condition not in patient.conditions:
            patient.conditions.append(condition)
            db.commit()
            
            # Re-run safety analysis for current medications
            self._recheck_safety(db, patient)
            
            audit_service.log(
                db=db,
                action="add_condition",
                resource_type="patient",
                resource_id=patient.id,
                description=f"Added condition {condition_name} to patient {patient.full_name}",
                user=user,
                request=request
            )
    
    # ==================== Prescription Processing ====================
    
    def add_prescription(
        self,
        db: Session,
        patient_id: int,
        prescription_data: Dict[str, Any],
        user: User = None,
        request = None
    ) -> Dict[str, Any]:
        """
        Add a new prescription and automatically:
        1. Save prescription record
        2. Track medication changes
        3. Build timeline
        4. Run safety analysis
        """
        patient = self.get_patient(db, patient_id)
        if not patient:
            raise ValueError("Patient not found")
        
        # Create prescription
        prescription_uid = str(uuid.uuid4())[:12].upper()
        prescription_date = prescription_data.get('prescription_date')
        if isinstance(prescription_date, str):
            try:
                prescription_date = datetime.fromisoformat(prescription_date.replace('Z', '+00:00'))
            except:
                prescription_date = datetime.utcnow()
        elif not prescription_date:
            prescription_date = datetime.utcnow()
        
        prescription = Prescription(
            prescription_uid=prescription_uid,
            patient_id=patient.id,
            prescription_date=prescription_date,
            doctor_name=prescription_data.get('doctor_name', 'Unknown'),
            doctor_registration_no=prescription_data.get('doctor_reg_no'),
            doctor_qualification=prescription_data.get('doctor_qualification'),
            clinic_name=prescription_data.get('clinic_name'),
            clinic_address=prescription_data.get('clinic_address'),
            diagnosis=prescription_data.get('diagnosis', []),
            vitals=prescription_data.get('vitals', {}),
            advice=prescription_data.get('advice', []),
            original_filename=prescription_data.get('filename'),
            file_path=prescription_data.get('file_path'),
            raw_ocr_text=prescription_data.get('raw_text'),
            ocr_confidence=prescription_data.get('ocr_confidence'),
            extraction_confidence=prescription_data.get('confidence'),
            status=PrescriptionStatus.PENDING,
            needs_review=prescription_data.get('needs_review', False),
            review_reasons=prescription_data.get('review_reasons'),
            created_by=user.id if user else None
        )
        
        db.add(prescription)
        db.flush()  # Get the ID
        
        # Add timeline event for prescription
        self._add_timeline_event(
            db=db,
            patient_id=patient.id,
            prescription_id=prescription.id,
            event_type='prescription_added',
            event_date=prescription_date,
            description=f"New prescription from Dr. {prescription.doctor_name}",
            details={
                'doctor': prescription.doctor_name,
                'clinic': prescription.clinic_name,
                'diagnosis': prescription.diagnosis
            },
            severity=AlertSeverity.INFO
        )
        
        # Process medications
        changes = self._process_medications(
            db=db,
            patient=patient,
            prescription=prescription,
            medications_data=prescription_data.get('medications', [])
        )
        
        # Run safety analysis
        safety_result = self._run_safety_analysis(db, patient, prescription)
        
        # Update prescription with safety info
        prescription.safety_score = safety_result.get('safety_score', 1.0)
        prescription.has_interactions = len(safety_result.get('interactions', [])) > 0
        prescription.has_allergy_alerts = len(safety_result.get('allergy_alerts', [])) > 0
        prescription.safety_alerts = safety_result.get('high_priority_alerts', [])
        
        # Mark for review if safety issues
        if prescription.has_interactions or prescription.has_allergy_alerts:
            prescription.needs_review = True
            prescription.review_reasons = prescription.review_reasons or []
            prescription.review_reasons.append("Safety alerts detected")
        
        db.commit()
        
        # Audit log
        audit_service.log(
            db=db,
            action="create",
            resource_type="prescription",
            resource_id=prescription.id,
            description=f"Added prescription {prescription_uid} for patient {patient.full_name}",
            new_values={
                "prescription_uid": prescription_uid,
                "medications_count": len(prescription.medications),
                "has_alerts": prescription.has_interactions or prescription.has_allergy_alerts
            },
            user=user,
            request=request
        )
        
        return {
            'success': True,
            'prescription_id': prescription.id,
            'prescription_uid': prescription_uid,
            'prescription_number': len(patient.prescriptions),
            'changes_detected': changes,
            'safety_analysis': safety_result,
            'needs_review': prescription.needs_review
        }
    
    def _process_medications(
        self,
        db: Session,
        patient: Patient,
        prescription: Prescription,
        medications_data: List[Dict]
    ) -> Dict[str, Any]:
        """Process medications and detect changes"""
        changes = {
            'new_medications': [],
            'continued_medications': [],
            'dose_changes': [],
            'stopped_medications': [],
            'restarted_medications': []
        }
        
        # Get current active medications
        current_meds = {
            self._normalize_med_name(m.name): m 
            for m in db.query(PatientMedication).filter(
                PatientMedication.patient_id == patient.id,
                PatientMedication.is_active == True
            ).all()
        }
        
        processed_med_names = set()
        
        for med_data in medications_data:
            med_name = med_data.get('name', '')
            if not med_name:
                continue
            
            # Normalize medication
            normalized = self.drug_normalizer.normalize(med_name)
            generic_name = normalized.generic_name
            normalized_key = self._normalize_med_name(med_name)
            processed_med_names.add(normalized_key)
            
            # Create prescription medication record
            rx_med = PrescriptionMedication(
                prescription_id=prescription.id,
                name=med_name,
                generic_name=generic_name,
                brand_name=med_name if normalized.is_brand_name else None,
                drug_class=normalized.drug_class,
                dosage=med_data.get('dosage'),
                frequency=med_data.get('frequency'),
                route=med_data.get('route', 'oral'),
                timing=med_data.get('timing'),
                duration=med_data.get('duration'),
                instructions=med_data.get('instructions')
            )
            db.add(rx_med)
            
            # Check if medication exists in patient's current list
            if normalized_key in current_meds:
                existing_med = current_meds[normalized_key]
                
                # Check for dose change
                if existing_med.dosage != med_data.get('dosage'):
                    changes['dose_changes'].append({
                        'medication': med_name,
                        'previous_dosage': existing_med.dosage,
                        'new_dosage': med_data.get('dosage')
                    })
                    
                    # Update existing medication
                    existing_med.previous_dosage = existing_med.dosage
                    existing_med.dosage = med_data.get('dosage')
                    existing_med.frequency = med_data.get('frequency')
                    existing_med.updated_at = datetime.utcnow()
                    
                    self._add_timeline_event(
                        db=db,
                        patient_id=patient.id,
                        prescription_id=prescription.id,
                        event_type='medication_changed',
                        event_date=prescription.prescription_date,
                        description=f"Dose change: {med_name} {existing_med.previous_dosage} → {med_data.get('dosage')}",
                        details={
                            'medication': med_name,
                            'previous_dosage': existing_med.previous_dosage,
                            'new_dosage': med_data.get('dosage')
                        },
                        severity=AlertSeverity.WARNING
                    )
                else:
                    changes['continued_medications'].append(med_name)
            else:
                # Check if previously discontinued
                was_discontinued = db.query(PatientMedication).filter(
                    PatientMedication.patient_id == patient.id,
                    func.lower(PatientMedication.generic_name) == generic_name.lower(),
                    PatientMedication.is_active == False
                ).first()
                
                if was_discontinued:
                    changes['restarted_medications'].append(med_name)
                    event_type = 'medication_restarted'
                else:
                    changes['new_medications'].append(med_name)
                    event_type = 'medication_started'
                
                # Add new patient medication
                patient_med = PatientMedication(
                    patient_id=patient.id,
                    prescription_id=prescription.id,
                    name=med_name,
                    generic_name=generic_name,
                    dosage=med_data.get('dosage'),
                    frequency=med_data.get('frequency'),
                    start_date=prescription.prescription_date,
                    is_active=True,
                    prescriber=prescription.doctor_name
                )
                db.add(patient_med)
                
                self._add_timeline_event(
                    db=db,
                    patient_id=patient.id,
                    prescription_id=prescription.id,
                    event_type=event_type,
                    event_date=prescription.prescription_date,
                    description=f"{'Restarted' if was_discontinued else 'New medication'}: {med_name} {med_data.get('dosage', '')}",
                    details={
                        'medication': med_name,
                        'generic_name': generic_name,
                        'dosage': med_data.get('dosage'),
                        'frequency': med_data.get('frequency')
                    },
                    severity=AlertSeverity.INFO
                )
        
        # Check for stopped medications
        for med_key, med in current_meds.items():
            if med_key not in processed_med_names:
                changes['stopped_medications'].append(med.name)
                
                # Mark as discontinued
                med.is_active = False
                med.end_date = prescription.prescription_date
                
                self._add_timeline_event(
                    db=db,
                    patient_id=patient.id,
                    prescription_id=prescription.id,
                    event_type='medication_stopped',
                    event_date=prescription.prescription_date,
                    description=f"Medication discontinued: {med.name}",
                    details={
                        'medication': med.name,
                        'was_dosage': med.dosage
                    },
                    severity=AlertSeverity.WARNING
                )
        
        return changes
    
    def _run_safety_analysis(
        self,
        db: Session,
        patient: Patient,
        prescription: Prescription
    ) -> Dict[str, Any]:
        """Run comprehensive safety analysis"""
        # Get current medication names
        med_names = [m.name for m in prescription.medications]
        
        if not med_names:
            return {'risk_level': 'NONE', 'interactions': [], 'allergy_alerts': []}
        
        # Get patient allergies and conditions
        allergies = [a.name for a in patient.allergies]
        conditions = [c.name for c in patient.conditions]
        
        try:
            result = self.drug_interaction_service.comprehensive_safety_check(
                medications=med_names,
                patient_allergies=allergies,
                patient_conditions=conditions
            )
            
            # Create safety alerts in database
            for alert in result.get('high_priority_alerts', []):
                severity = AlertSeverity.CRITICAL if alert.get('severity') in ['major', 'contraindicated'] else AlertSeverity.WARNING
                
                safety_alert = SafetyAlert(
                    patient_id=patient.id,
                    prescription_id=prescription.id,
                    alert_type=alert.get('type', 'interaction'),
                    severity=severity,
                    drug1=alert.get('drugs', [None, None])[0] if isinstance(alert.get('drugs'), list) else alert.get('drug'),
                    drug2=alert.get('drugs', [None, None])[1] if isinstance(alert.get('drugs'), list) and len(alert.get('drugs', [])) > 1 else None,
                    title=f"{alert.get('type', 'Alert').upper()}: {alert.get('description', '')[:100]}",
                    description=alert.get('description', ''),
                    recommendation=alert.get('action_required', '')
                )
                db.add(safety_alert)
                
                # Add timeline event for critical alerts
                if severity == AlertSeverity.CRITICAL:
                    self._add_timeline_event(
                        db=db,
                        patient_id=patient.id,
                        prescription_id=prescription.id,
                        event_type='safety_alert',
                        event_date=prescription.prescription_date,
                        description=f"⚠️ SAFETY ALERT: {alert.get('description', '')}",
                        details=alert,
                        severity=AlertSeverity.CRITICAL
                    )
            
            # Mark prescription medications with interactions
            for med in prescription.medications:
                for interaction in result.get('interactions', []):
                    if med.name.lower() in [interaction.get('drug1', '').lower(), interaction.get('drug2', '').lower()]:
                        med.has_interaction = True
                        med.interaction_details = interaction
                
                for allergy_alert in result.get('allergy_alerts', []):
                    if med.name.lower() == allergy_alert.get('drug', '').lower():
                        med.has_allergy_risk = True
                        med.allergy_details = allergy_alert
            
            return result
            
        except Exception as e:
            return {'risk_level': 'ERROR', 'error': str(e), 'interactions': [], 'allergy_alerts': []}
    
    def _recheck_safety(self, db: Session, patient: Patient):
        """Re-run safety analysis after allergy/condition update"""
        # Get active medications
        active_meds = db.query(PatientMedication).filter(
            PatientMedication.patient_id == patient.id,
            PatientMedication.is_active == True
        ).all()
        
        if not active_meds:
            return
        
        med_names = [m.name for m in active_meds]
        allergies = [a.name for a in patient.allergies]
        conditions = [c.name for c in patient.conditions]
        
        try:
            result = self.drug_interaction_service.comprehensive_safety_check(
                medications=med_names,
                patient_allergies=allergies,
                patient_conditions=conditions
            )
            
            # Create new alerts if any
            for alert in result.get('high_priority_alerts', []):
                severity = AlertSeverity.CRITICAL if alert.get('severity') in ['major', 'contraindicated'] else AlertSeverity.WARNING
                
                # Check if alert already exists
                existing = db.query(SafetyAlert).filter(
                    SafetyAlert.patient_id == patient.id,
                    SafetyAlert.description == alert.get('description', ''),
                    SafetyAlert.is_resolved == False
                ).first()
                
                if not existing:
                    safety_alert = SafetyAlert(
                        patient_id=patient.id,
                        alert_type=alert.get('type', 'interaction'),
                        severity=severity,
                        title=f"New {alert.get('type', 'Alert').upper()}",
                        description=alert.get('description', ''),
                        recommendation=alert.get('action_required', '')
                    )
                    db.add(safety_alert)
            
            db.commit()
            
        except Exception:
            pass
    
    # ==================== Timeline & History ====================
    
    def get_patient_timeline(
        self,
        db: Session,
        patient_id: int,
        event_type: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get patient's complete timeline"""
        q = db.query(TimelineEvent).filter(TimelineEvent.patient_id == patient_id)
        
        if event_type:
            q = q.filter(TimelineEvent.event_type == event_type)
        
        events = q.order_by(TimelineEvent.event_date.desc()).limit(limit).all()
        
        return [
            {
                'event_id': e.id,
                'event_type': e.event_type,
                'event_date': e.event_date.isoformat() if e.event_date else None,
                'description': e.description,
                'details': e.details,
                'severity': e.severity.value if e.severity else 'info'
            }
            for e in events
        ]
    
    def get_patient_summary(self, db: Session, patient_id: int) -> Dict[str, Any]:
        """Get comprehensive patient summary"""
        patient = self.get_patient(db, patient_id)
        if not patient:
            return {'error': 'Patient not found'}
        
        # Count statistics
        total_prescriptions = db.query(func.count(Prescription.id)).filter(
            Prescription.patient_id == patient_id
        ).scalar()
        
        active_meds = db.query(func.count(PatientMedication.id)).filter(
            PatientMedication.patient_id == patient_id,
            PatientMedication.is_active == True
        ).scalar()
        
        historical_meds = db.query(func.count(PatientMedication.id)).filter(
            PatientMedication.patient_id == patient_id,
            PatientMedication.is_active == False
        ).scalar()
        
        timeline_events = db.query(func.count(TimelineEvent.id)).filter(
            TimelineEvent.patient_id == patient_id
        ).scalar()
        
        active_alerts = db.query(func.count(SafetyAlert.id)).filter(
            SafetyAlert.patient_id == patient_id,
            SafetyAlert.is_resolved == False
        ).scalar()
        
        # Get current medications
        current_medications = db.query(PatientMedication).filter(
            PatientMedication.patient_id == patient_id,
            PatientMedication.is_active == True
        ).all()
        
        # Get unique doctors
        doctors = db.query(Prescription.doctor_name).filter(
            Prescription.patient_id == patient_id,
            Prescription.doctor_name != None
        ).distinct().all()
        
        return {
            'patient': {
                'id': patient.id,
                'patient_uid': patient.patient_uid,
                'name': patient.full_name,
                'age': patient.age,
                'gender': patient.gender,
                'phone': patient.phone,
                'allergies': [a.name for a in patient.allergies],
                'conditions': [c.name for c in patient.conditions]
            },
            'statistics': {
                'total_prescriptions': total_prescriptions,
                'active_medications': active_meds,
                'discontinued_medications': historical_meds,
                'timeline_events': timeline_events,
                'active_alerts': active_alerts
            },
            'current_medications': [
                {
                    'name': m.name,
                    'generic_name': m.generic_name,
                    'dosage': m.dosage,
                    'frequency': m.frequency,
                    'start_date': m.start_date.isoformat() if m.start_date else None,
                    'prescriber': m.prescriber
                }
                for m in current_medications
            ],
            'doctors': [d[0] for d in doctors if d[0]]
        }
    
    # ==================== Helpers ====================
    
    def _normalize_med_name(self, name: str) -> str:
        """Normalize medication name for comparison"""
        if not name:
            return ""
        normalized = self.drug_normalizer.normalize(name)
        return normalized.generic_name.lower()
    
    def _get_or_create_allergy(self, db: Session, name: str) -> Allergy:
        """Get or create an allergy record"""
        allergy = db.query(Allergy).filter(func.lower(Allergy.name) == name.lower()).first()
        if not allergy:
            allergy = Allergy(name=name, category='drug')
            db.add(allergy)
            db.flush()
        return allergy
    
    def _get_or_create_condition(self, db: Session, name: str) -> Condition:
        """Get or create a condition record"""
        condition = db.query(Condition).filter(func.lower(Condition.name) == name.lower()).first()
        if not condition:
            condition = Condition(name=name)
            db.add(condition)
            db.flush()
        return condition
    
    def _add_timeline_event(
        self,
        db: Session,
        patient_id: int,
        event_type: str,
        event_date: datetime,
        description: str,
        details: Dict = None,
        prescription_id: int = None,
        severity: AlertSeverity = AlertSeverity.INFO
    ):
        """Add a timeline event"""
        event = TimelineEvent(
            patient_id=patient_id,
            prescription_id=prescription_id,
            event_type=event_type,
            event_date=event_date,
            description=description,
            details=details or {},
            severity=severity
        )
        db.add(event)


# Singleton instance
_production_patient_service = None

def get_production_patient_service() -> ProductionPatientService:
    global _production_patient_service
    if _production_patient_service is None:
        _production_patient_service = ProductionPatientService()
    return _production_patient_service
