"""
Patient Prescription Service
Handles automated prescription processing, timeline building, and change tracking
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass, field
import uuid


@dataclass
class MedicationRecord:
    """A medication record with full details"""
    name: str
    generic_name: str
    dosage: str
    frequency: str
    duration: str
    instructions: str
    start_date: str
    end_date: Optional[str] = None
    prescriber: Optional[str] = None
    prescription_id: str = ""
    is_active: bool = True
    
    def to_dict(self):
        return {
            'name': self.name,
            'generic_name': self.generic_name,
            'dosage': self.dosage,
            'frequency': self.frequency,
            'duration': self.duration,
            'instructions': self.instructions,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'prescriber': self.prescriber,
            'prescription_id': self.prescription_id,
            'is_active': self.is_active
        }


@dataclass
class PrescriptionRecord:
    """A complete prescription record"""
    prescription_id: str
    prescription_date: str
    doctor_name: str
    doctor_qualification: str
    clinic_name: str
    diagnosis: List[str]
    medications: List[Dict]
    vitals: Dict[str, Any]
    advice: List[str]
    follow_up: str
    raw_ocr_text: str
    confidence: float
    file_name: str
    processed_at: str
    
    def to_dict(self):
        return {
            'prescription_id': self.prescription_id,
            'prescription_date': self.prescription_date,
            'doctor_name': self.doctor_name,
            'doctor_qualification': self.doctor_qualification,
            'clinic_name': self.clinic_name,
            'diagnosis': self.diagnosis,
            'medications': self.medications,
            'vitals': self.vitals,
            'advice': self.advice,
            'follow_up': self.follow_up,
            'confidence': self.confidence,
            'file_name': self.file_name,
            'processed_at': self.processed_at
        }


@dataclass
class TimelineEvent:
    """A timeline event"""
    event_id: str
    event_type: str  # prescription_added, medication_started, medication_stopped, medication_changed, diagnosis_added
    event_date: str
    description: str
    details: Dict[str, Any]
    prescription_id: Optional[str] = None
    severity: str = "info"  # info, warning, danger
    
    def to_dict(self):
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'event_date': self.event_date,
            'description': self.description,
            'details': self.details,
            'prescription_id': self.prescription_id,
            'severity': self.severity
        }


@dataclass 
class PatientProfile:
    """Complete patient profile with all history"""
    patient_id: str
    name: str
    age: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    allergies: List[str] = field(default_factory=list)
    chronic_conditions: List[str] = field(default_factory=list)
    current_medications: List[MedicationRecord] = field(default_factory=list)
    historical_medications: List[MedicationRecord] = field(default_factory=list)
    prescriptions: List[PrescriptionRecord] = field(default_factory=list)
    timeline: List[TimelineEvent] = field(default_factory=list)
    safety_alerts: List[Dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    
    def to_dict(self):
        return {
            'patient_id': self.patient_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'phone': self.phone,
            'address': self.address,
            'allergies': self.allergies,
            'chronic_conditions': self.chronic_conditions,
            'current_medications': [m.to_dict() for m in self.current_medications],
            'historical_medications': [m.to_dict() for m in self.historical_medications],
            'prescriptions': [p.to_dict() for p in self.prescriptions],
            'timeline': [t.to_dict() for t in self.timeline],
            'safety_alerts': self.safety_alerts,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'total_prescriptions': len(self.prescriptions),
            'active_medications_count': len(self.current_medications)
        }


class PatientPrescriptionService:
    """
    Service for managing patient prescriptions and building automated timelines
    Now integrated with database for persistent storage
    """
    
    def __init__(self):
        # In-memory patient store for quick access (also syncs with database)
        self.patients: Dict[str, PatientProfile] = {}
        
        # Import services
        from backend.services.drug_normalization_service import DrugNormalizationService
        from backend.services.drug_interaction_service import DrugInteractionService
        
        self.drug_normalizer = DrugNormalizationService()
        self.drug_interaction_service = DrugInteractionService()
        
        # Load existing patients from database on startup
        self._load_from_database()
    
    def _load_from_database(self):
        """Load patients from database into memory"""
        try:
            from backend.database.connection import get_db
            from backend.database.models import Patient, Prescription, PrescriptionMedication, TimelineEvent
            
            db = next(get_db())
            
            # Load all patients
            db_patients = db.query(Patient).all()
            
            for db_patient in db_patients:
                # Convert to in-memory format
                patient = PatientProfile(
                    patient_id=db_patient.patient_uid,
                    name=f"{db_patient.first_name} {db_patient.last_name}".strip(),
                    age=str(db_patient.age) if hasattr(db_patient, 'age') and db_patient.age else None,
                    gender=db_patient.gender,
                    phone=db_patient.phone,
                    address=db_patient.address,
                    allergies=[a.name for a in db_patient.allergies] if db_patient.allergies else [],
                    chronic_conditions=[c.name for c in db_patient.conditions] if db_patient.conditions else [],
                    created_at=db_patient.created_at.isoformat() if db_patient.created_at else "",
                    updated_at=db_patient.updated_at.isoformat() if db_patient.updated_at else ""
                )
                
                # Load prescriptions
                for db_presc in db_patient.prescriptions:
                    meds = []
                    for med in db_presc.medications:
                        meds.append({
                            'name': med.name,
                            'generic_name': med.generic_name,
                            'dosage': med.dosage,
                            'frequency': med.frequency,
                            'timing': med.timing,
                            'duration': med.duration,
                            'instructions': med.instructions
                        })
                        
                        # Add to current medications
                        med_record = MedicationRecord(
                            name=med.name,
                            generic_name=med.generic_name or med.name,
                            dosage=med.dosage or "",
                            frequency=med.frequency or "",
                            duration=med.duration or "",
                            instructions=med.instructions or "",
                            start_date=db_presc.prescription_date.isoformat() if db_presc.prescription_date else "",
                            prescriber=db_presc.doctor_name,
                            prescription_id=db_presc.prescription_uid,
                            is_active=True
                        )
                        patient.current_medications.append(med_record)
                    
                    presc_record = PrescriptionRecord(
                        prescription_id=db_presc.prescription_uid,
                        prescription_date=db_presc.prescription_date.isoformat() if db_presc.prescription_date else "",
                        doctor_name=db_presc.doctor_name or "",
                        doctor_qualification=db_presc.doctor_qualification or "",
                        clinic_name=db_presc.clinic_name or "",
                        diagnosis=db_presc.diagnosis if db_presc.diagnosis else [],
                        medications=meds,
                        vitals=db_presc.vitals if db_presc.vitals else {},
                        advice=db_presc.advice if db_presc.advice else [],
                        follow_up=str(db_presc.follow_up_date) if db_presc.follow_up_date else "",
                        raw_ocr_text=db_presc.raw_ocr_text or "",
                        confidence=db_presc.extraction_confidence or 0.0,
                        file_name=db_presc.original_filename or "",
                        processed_at=db_presc.created_at.isoformat() if db_presc.created_at else ""
                    )
                    patient.prescriptions.append(presc_record)
                
                # Load timeline events
                for db_event in db_patient.timeline_events:
                    event = TimelineEvent(
                        event_id=str(db_event.id),
                        event_type=db_event.event_type,
                        event_date=db_event.event_date.isoformat() if db_event.event_date else "",
                        description=db_event.description,
                        details=db_event.details if db_event.details else {},
                        severity=db_event.severity.value if db_event.severity else "info"
                    )
                    patient.timeline.append(event)
                
                self.patients[db_patient.patient_uid] = patient
            
            print(f"✓ Loaded {len(self.patients)} patients from database")
            
        except Exception as e:
            print(f"Warning: Could not load from database: {e}")
            # Continue with empty in-memory store
    
    def _save_patient_to_database(self, patient: PatientProfile):
        """Save patient profile to database"""
        try:
            from backend.database.connection import get_db
            from backend.database.models import Patient, Allergy, Condition
            
            db = next(get_db())
            
            # Check if patient exists
            db_patient = db.query(Patient).filter(Patient.patient_uid == patient.patient_id).first()
            
            # Parse name into first/last
            name_parts = patient.name.split() if patient.name else ["Unknown"]
            first_name = name_parts[0]
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            
            if not db_patient:
                # Create new patient
                db_patient = Patient(
                    patient_uid=patient.patient_id,
                    first_name=first_name,
                    last_name=last_name,
                    gender=patient.gender,
                    phone=patient.phone,
                    address=patient.address
                )
                db.add(db_patient)
            else:
                # Update existing
                db_patient.first_name = first_name
                db_patient.last_name = last_name
                if patient.gender:
                    db_patient.gender = patient.gender
                if patient.phone:
                    db_patient.phone = patient.phone
                if patient.address:
                    db_patient.address = patient.address
            
            # Handle allergies
            for allergy_name in patient.allergies:
                allergy = db.query(Allergy).filter(Allergy.name == allergy_name).first()
                if not allergy:
                    allergy = Allergy(name=allergy_name, category="drug")
                    db.add(allergy)
                if allergy not in db_patient.allergies:
                    db_patient.allergies.append(allergy)
            
            # Handle conditions
            for condition_name in patient.chronic_conditions:
                condition = db.query(Condition).filter(Condition.name == condition_name).first()
                if not condition:
                    condition = Condition(name=condition_name)
                    db.add(condition)
                if condition not in db_patient.conditions:
                    db_patient.conditions.append(condition)
            
            db.commit()
            return db_patient.id
            
        except Exception as e:
            print(f"Warning: Could not save patient to database: {e}")
            return None
    
    def _save_prescription_to_database(self, patient_id: str, prescription: PrescriptionRecord):
        """Save prescription to database"""
        try:
            from backend.database.connection import get_db
            from backend.database.models import Patient, Prescription, PrescriptionMedication, TimelineEvent as DBTimelineEvent, AlertSeverity
            from datetime import datetime
            
            db = next(get_db())
            
            # Get patient
            db_patient = db.query(Patient).filter(Patient.patient_uid == patient_id).first()
            if not db_patient:
                return None
            
            # Check if prescription already exists
            existing = db.query(Prescription).filter(Prescription.prescription_uid == prescription.prescription_id).first()
            if existing:
                return existing.id
            
            # Parse date
            presc_date = None
            try:
                from dateutil import parser
                presc_date = parser.parse(prescription.prescription_date, dayfirst=True) if prescription.prescription_date else datetime.utcnow()
            except:
                presc_date = datetime.utcnow()
            
            # Create prescription
            db_presc = Prescription(
                prescription_uid=prescription.prescription_id,
                patient_id=db_patient.id,
                prescription_date=presc_date,
                doctor_name=prescription.doctor_name,
                doctor_qualification=prescription.doctor_qualification,
                clinic_name=prescription.clinic_name,
                diagnosis=prescription.diagnosis,
                vitals=prescription.vitals,
                advice=prescription.advice,
                raw_ocr_text=prescription.raw_ocr_text,
                extraction_confidence=prescription.confidence,
                original_filename=prescription.file_name
            )
            db.add(db_presc)
            db.flush()  # Get the ID
            
            # Add medications
            for med in prescription.medications:
                db_med = PrescriptionMedication(
                    prescription_id=db_presc.id,
                    name=med.get('name', ''),
                    generic_name=med.get('generic_name', ''),
                    dosage=med.get('dosage', ''),
                    frequency=med.get('frequency', ''),
                    timing=med.get('timing', ''),
                    duration=med.get('duration', ''),
                    instructions=med.get('instructions', '')
                )
                db.add(db_med)
            
            # Add timeline event
            db_event = DBTimelineEvent(
                patient_id=db_patient.id,
                prescription_id=db_presc.id,
                event_type='prescription_added',
                event_date=presc_date,
                description=f"New prescription from Dr. {prescription.doctor_name}",
                details={
                    'doctor': prescription.doctor_name,
                    'clinic': prescription.clinic_name,
                    'medications_count': len(prescription.medications)
                },
                severity=AlertSeverity.INFO
            )
            db.add(db_event)
            
            db.commit()
            return db_presc.id
            
        except Exception as e:
            print(f"Warning: Could not save prescription to database: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_or_create_patient(self, patient_id: str, name: str = None, **kwargs) -> PatientProfile:
        """Get existing patient or create new one"""
        if patient_id not in self.patients:
            self.patients[patient_id] = PatientProfile(
                patient_id=patient_id,
                name=name or f"Patient {patient_id}",
                age=kwargs.get('age'),
                gender=kwargs.get('gender'),
                phone=kwargs.get('phone'),
                address=kwargs.get('address'),
                allergies=kwargs.get('allergies', []),
                chronic_conditions=kwargs.get('chronic_conditions', []),
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            # Save to database
            self._save_patient_to_database(self.patients[patient_id])
        else:
            # Update with any new info
            patient = self.patients[patient_id]
            updated = False
            if name and name != patient.name:
                patient.name = name
                updated = True
            if kwargs.get('age'):
                patient.age = kwargs['age']
                updated = True
            if kwargs.get('gender'):
                patient.gender = kwargs['gender']
                updated = True
            if kwargs.get('allergies'):
                for allergy in kwargs['allergies']:
                    if allergy not in patient.allergies:
                        patient.allergies.append(allergy)
                        updated = True
            patient.updated_at = datetime.utcnow().isoformat()
            
            # Update database if changes made
            if updated:
                self._save_patient_to_database(patient)
        
        return self.patients[patient_id]
    
    def add_prescription(self, patient_id: str, prescription_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new prescription to patient's record and automatically:
        1. Update patient info
        2. Normalize medications
        3. Track medication changes
        4. Build timeline events
        5. Run safety analysis
        """
        # Get or create patient
        patient = self.get_or_create_patient(
            patient_id=patient_id,
            name=prescription_data.get('patient_name'),
            age=prescription_data.get('patient_age'),
            gender=prescription_data.get('patient_gender'),
            phone=prescription_data.get('patient_phone'),
            address=prescription_data.get('patient_address'),
            allergies=prescription_data.get('allergies', [])
        )
        
        # Create prescription record
        prescription_id = str(uuid.uuid4())[:8]
        prescription_date = prescription_data.get('prescription_date') or datetime.utcnow().date().isoformat()
        
        prescription = PrescriptionRecord(
            prescription_id=prescription_id,
            prescription_date=prescription_date,
            doctor_name=prescription_data.get('doctor_name', 'Unknown'),
            doctor_qualification=prescription_data.get('doctor_qualification', ''),
            clinic_name=prescription_data.get('clinic_name', ''),
            diagnosis=prescription_data.get('diagnosis', []),
            medications=prescription_data.get('medications', []),
            vitals=prescription_data.get('vitals', {}),
            advice=prescription_data.get('advice', []),
            follow_up=prescription_data.get('follow_up', ''),
            raw_ocr_text=prescription_data.get('raw_text', ''),
            confidence=prescription_data.get('confidence', 0),
            file_name=prescription_data.get('filename', ''),
            processed_at=datetime.utcnow().isoformat()
        )
        
        # Add prescription to patient
        patient.prescriptions.append(prescription)
        
        # Save prescription to database
        self._save_prescription_to_database(patient_id, prescription)
        
        # Add timeline event for prescription
        self._add_timeline_event(
            patient=patient,
            event_type='prescription_added',
            event_date=prescription_date,
            description=f"New prescription from Dr. {prescription.doctor_name}",
            details={
                'doctor': prescription.doctor_name,
                'clinic': prescription.clinic_name,
                'medications_count': len(prescription.medications),
                'diagnosis': prescription.diagnosis
            },
            prescription_id=prescription_id,
            severity='info'
        )
        
        # Process medications and track changes
        changes = self._process_medications(patient, prescription)
        
        # Run safety analysis
        safety_result = self._run_safety_analysis(patient)
        
        # Update patient
        patient.updated_at = datetime.utcnow().isoformat()
        
        return {
            'success': True,
            'patient_id': patient_id,
            'prescription_id': prescription_id,
            'prescription_number': len(patient.prescriptions),
            'changes_detected': changes,
            'safety_analysis': safety_result,
            'current_medications': [m.to_dict() for m in patient.current_medications],
            'timeline_events_added': len([t for t in patient.timeline if t.prescription_id == prescription_id])
        }
    
    def _process_medications(self, patient: PatientProfile, prescription: PrescriptionRecord) -> Dict[str, Any]:
        """Process medications and detect changes from previous prescriptions"""
        changes = {
            'new_medications': [],
            'continued_medications': [],
            'dose_changes': [],
            'stopped_medications': [],
            'restarted_medications': []
        }
        
        prescription_date = prescription.prescription_date
        previous_meds = {self._normalize_med_name(m.name): m for m in patient.current_medications}
        current_med_names = set()
        
        for med_data in prescription.medications:
            med_name = med_data.get('name', '')
            if not med_name:
                continue
            
            # Normalize the medication name
            normalized = self.drug_normalizer.normalize(med_name)
            generic_name = normalized.generic_name
            current_med_names.add(self._normalize_med_name(med_name))
            
            # Create medication record
            med_record = MedicationRecord(
                name=med_name,
                generic_name=generic_name,
                dosage=med_data.get('dosage', ''),
                frequency=med_data.get('frequency', ''),
                duration=med_data.get('duration', ''),
                instructions=med_data.get('instructions', ''),
                start_date=prescription_date,
                prescriber=prescription.doctor_name,
                prescription_id=prescription.prescription_id
            )
            
            normalized_name = self._normalize_med_name(med_name)
            
            # Check if this is a new, continued, or changed medication
            if normalized_name in previous_meds:
                prev_med = previous_meds[normalized_name]
                
                # Check for dose change
                if prev_med.dosage != med_record.dosage:
                    changes['dose_changes'].append({
                        'medication': med_name,
                        'previous_dosage': prev_med.dosage,
                        'new_dosage': med_record.dosage
                    })
                    
                    self._add_timeline_event(
                        patient=patient,
                        event_type='medication_changed',
                        event_date=prescription_date,
                        description=f"Dose change: {med_name} {prev_med.dosage} → {med_record.dosage}",
                        details={
                            'medication': med_name,
                            'previous_dosage': prev_med.dosage,
                            'new_dosage': med_record.dosage,
                            'prescriber': prescription.doctor_name
                        },
                        prescription_id=prescription.prescription_id,
                        severity='warning'
                    )
                    
                    # Update the existing medication
                    prev_med.dosage = med_record.dosage
                    prev_med.frequency = med_record.frequency
                    prev_med.prescription_id = prescription.prescription_id
                else:
                    changes['continued_medications'].append(med_name)
            else:
                # Check if it was previously stopped (restarted)
                historical_names = {self._normalize_med_name(m.name) for m in patient.historical_medications}
                
                if normalized_name in historical_names:
                    changes['restarted_medications'].append(med_name)
                    self._add_timeline_event(
                        patient=patient,
                        event_type='medication_restarted',
                        event_date=prescription_date,
                        description=f"Medication restarted: {med_name}",
                        details={
                            'medication': med_name,
                            'dosage': med_record.dosage,
                            'prescriber': prescription.doctor_name
                        },
                        prescription_id=prescription.prescription_id,
                        severity='info'
                    )
                else:
                    changes['new_medications'].append(med_name)
                    self._add_timeline_event(
                        patient=patient,
                        event_type='medication_started',
                        event_date=prescription_date,
                        description=f"New medication: {med_name} {med_record.dosage}",
                        details={
                            'medication': med_name,
                            'generic_name': generic_name,
                            'dosage': med_record.dosage,
                            'frequency': med_record.frequency,
                            'prescriber': prescription.doctor_name
                        },
                        prescription_id=prescription.prescription_id,
                        severity='info'
                    )
                
                # Add to current medications
                patient.current_medications.append(med_record)
        
        # Check for stopped medications (in previous but not in current)
        for prev_name, prev_med in previous_meds.items():
            if prev_name not in current_med_names:
                changes['stopped_medications'].append(prev_med.name)
                
                self._add_timeline_event(
                    patient=patient,
                    event_type='medication_stopped',
                    event_date=prescription_date,
                    description=f"Medication discontinued: {prev_med.name}",
                    details={
                        'medication': prev_med.name,
                        'was_dosage': prev_med.dosage,
                        'duration_on_med': self._calculate_duration(prev_med.start_date, prescription_date)
                    },
                    prescription_id=prescription.prescription_id,
                    severity='warning'
                )
                
                # Move to historical
                prev_med.is_active = False
                prev_med.end_date = prescription_date
                patient.historical_medications.append(prev_med)
                patient.current_medications.remove(prev_med)
        
        # Add diagnosis events
        for diagnosis in prescription.diagnosis:
            if diagnosis:
                self._add_timeline_event(
                    patient=patient,
                    event_type='diagnosis_recorded',
                    event_date=prescription_date,
                    description=f"Diagnosis: {diagnosis}",
                    details={'diagnosis': diagnosis, 'doctor': prescription.doctor_name},
                    prescription_id=prescription.prescription_id,
                    severity='info'
                )
        
        return changes
    
    def _run_safety_analysis(self, patient: PatientProfile) -> Dict[str, Any]:
        """Run comprehensive safety analysis on current medications"""
        if not patient.current_medications:
            return {'risk_level': 'NONE', 'alerts': []}
        
        med_names = [m.name for m in patient.current_medications]
        
        try:
            result = self.drug_interaction_service.comprehensive_safety_check(
                medications=med_names,
                patient_allergies=patient.allergies,
                patient_conditions=patient.chronic_conditions
            )
            
            # Store safety alerts
            patient.safety_alerts = result.get('high_priority_alerts', [])
            
            # Add timeline events for critical alerts
            for alert in result.get('high_priority_alerts', []):
                if alert.get('severity') in ['major', 'contraindicated']:
                    self._add_timeline_event(
                        patient=patient,
                        event_type='safety_alert',
                        event_date=datetime.utcnow().date().isoformat(),
                        description=f"⚠️ {alert.get('type', 'Safety').upper()} ALERT: {alert.get('description', '')}",
                        details=alert,
                        severity='danger'
                    )
            
            return {
                'risk_level': result.get('overall_risk_level', 'UNKNOWN'),
                'interactions_count': len(result.get('interactions', [])),
                'allergy_alerts_count': len(result.get('allergy_alerts', [])),
                'contraindications_count': len(result.get('contraindications', [])),
                'high_priority_alerts': result.get('high_priority_alerts', [])
            }
        except Exception as e:
            return {'risk_level': 'ERROR', 'error': str(e)}
    
    def _add_timeline_event(self, patient: PatientProfile, event_type: str, event_date: str,
                           description: str, details: Dict, prescription_id: str = None,
                           severity: str = 'info'):
        """Add a timeline event"""
        event = TimelineEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            event_date=event_date,
            description=description,
            details=details,
            prescription_id=prescription_id,
            severity=severity
        )
        patient.timeline.append(event)
    
    def _normalize_med_name(self, name: str) -> str:
        """Normalize medication name for comparison"""
        if not name:
            return ""
        normalized = self.drug_normalizer.normalize(name)
        return normalized.generic_name.lower()
    
    def _calculate_duration(self, start_date: str, end_date: str) -> str:
        """Calculate duration between two dates"""
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            days = (end - start).days
            if days < 7:
                return f"{days} days"
            elif days < 30:
                return f"{days // 7} weeks"
            else:
                return f"{days // 30} months"
        except:
            return "Unknown"
    
    def get_patient(self, patient_id: str) -> Optional[PatientProfile]:
        """Get patient profile"""
        return self.patients.get(patient_id)
    
    def get_patient_timeline(self, patient_id: str) -> List[Dict]:
        """Get patient's complete timeline sorted by date"""
        patient = self.patients.get(patient_id)
        if not patient:
            return []
        
        # Sort timeline by date (newest first)
        sorted_timeline = sorted(
            patient.timeline,
            key=lambda x: x.event_date,
            reverse=True
        )
        return [t.to_dict() for t in sorted_timeline]
    
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient summary"""
        patient = self.patients.get(patient_id)
        if not patient:
            return {'error': 'Patient not found'}
        
        # Calculate statistics
        total_prescriptions = len(patient.prescriptions)
        total_meds_ever = len(patient.current_medications) + len(patient.historical_medications)
        
        # Get medication history by generic name
        med_history = {}
        for med in patient.current_medications + patient.historical_medications:
            generic = med.generic_name
            if generic not in med_history:
                med_history[generic] = []
            med_history[generic].append({
                'name': med.name,
                'dosage': med.dosage,
                'start_date': med.start_date,
                'end_date': med.end_date,
                'is_active': med.is_active
            })
        
        return {
            'patient': patient.to_dict(),
            'statistics': {
                'total_prescriptions': total_prescriptions,
                'total_medications_prescribed': total_meds_ever,
                'current_active_medications': len(patient.current_medications),
                'discontinued_medications': len(patient.historical_medications),
                'timeline_events': len(patient.timeline),
                'active_safety_alerts': len(patient.safety_alerts)
            },
            'medication_history_by_drug': med_history,
            'all_diagnoses': list(set(
                d for p in patient.prescriptions for d in p.diagnosis if d
            )),
            'all_doctors': list(set(
                p.doctor_name for p in patient.prescriptions if p.doctor_name
            ))
        }
    
    def get_all_patients(self) -> List[Dict]:
        """Get list of all patients"""
        return [
            {
                'patient_id': p.patient_id,
                'name': p.name,
                'prescriptions_count': len(p.prescriptions),
                'active_medications': len(p.current_medications),
                'last_visit': p.prescriptions[-1].prescription_date if p.prescriptions else None
            }
            for p in self.patients.values()
        ]


# Singleton instance
_patient_prescription_service = None

def get_patient_prescription_service() -> PatientPrescriptionService:
    global _patient_prescription_service
    if _patient_prescription_service is None:
        _patient_prescription_service = PatientPrescriptionService()
    return _patient_prescription_service
