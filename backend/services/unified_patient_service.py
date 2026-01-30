"""
Unified Patient & Prescription Service
Single source of truth for all patient data operations
Stores everything in the SQLite/PostgreSQL database
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.database.connection import db_manager
from backend.database.models import (
    Patient, Prescription, PrescriptionMedication, 
    PatientMedication, TimelineEvent, Allergy, Condition,
    AlertSeverity
)

logger = logging.getLogger(__name__)


class UnifiedPatientService:
    """
    Unified service for all patient and prescription database operations.
    This is the SINGLE source of truth - used by scanning, AI, and all other features.
    """
    
    def __init__(self):
        self._ensure_db()
    
    def _ensure_db(self):
        """Ensure database is initialized"""
        if not db_manager._initialized:
            db_manager.init_db()
    
    def _get_session(self) -> Session:
        """Get database session"""
        return db_manager.get_session()
    
    # ==================== PATIENT OPERATIONS ====================
    
    def get_or_create_patient(
        self,
        patient_uid: str,
        name: str = None,
        age: int = None,
        gender: str = None,
        phone: str = None,
        address: str = None,
        allergies: List[str] = None,
        conditions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get existing patient by UHID or create new one.
        This is the main entry point for patient management.
        """
        session = self._get_session()
        try:
            # Try to find existing patient by UHID
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            # Parse name into first/last
            first_name = name or f"Patient"
            last_name = patient_uid
            if name and ' ' in name:
                parts = name.strip().split(' ', 1)
                first_name = parts[0]
                last_name = parts[1] if len(parts) > 1 else patient_uid
            
            if not patient:
                # Create new patient
                patient = Patient(
                    patient_uid=patient_uid,
                    first_name=first_name,
                    last_name=last_name,
                    gender=gender,
                    phone=phone,
                    address=address
                )
                session.add(patient)
                session.flush()
                logger.info(f"Created new patient: {patient_uid} - {name}")
            else:
                # Update existing patient with new info
                if name and first_name != "Patient":
                    patient.first_name = first_name
                    patient.last_name = last_name
                if gender:
                    patient.gender = gender
                if phone:
                    patient.phone = phone
                if address:
                    patient.address = address
                patient.updated_at = datetime.utcnow()
            
            # Handle allergies
            if allergies:
                for allergy_name in allergies:
                    if not allergy_name.strip():
                        continue
                    allergy = session.query(Allergy).filter(
                        func.lower(Allergy.name) == allergy_name.lower().strip()
                    ).first()
                    if not allergy:
                        allergy = Allergy(name=allergy_name.strip(), category="drug")
                        session.add(allergy)
                        session.flush()
                    if allergy not in patient.allergies:
                        patient.allergies.append(allergy)
            
            # Handle conditions
            if conditions:
                for condition_name in conditions:
                    if not condition_name.strip():
                        continue
                    condition = session.query(Condition).filter(
                        func.lower(Condition.name) == condition_name.lower().strip()
                    ).first()
                    if not condition:
                        condition = Condition(name=condition_name.strip())
                        session.add(condition)
                        session.flush()
                    if condition not in patient.conditions:
                        patient.conditions.append(condition)
            
            session.commit()
            
            return self._patient_to_dict(patient)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error in get_or_create_patient: {e}")
            raise
        finally:
            session.close()
    
    def get_patient_by_uid(self, patient_uid: str) -> Optional[Dict[str, Any]]:
        """Get patient by UHID"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if patient:
                return self._patient_to_dict(patient)
            return None
        finally:
            session.close()
    
    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Get all patients with summary info"""
        session = self._get_session()
        try:
            patients = session.query(Patient).order_by(desc(Patient.updated_at)).all()
            
            result = []
            for patient in patients:
                prescription_count = len(patient.prescriptions)
                active_meds = session.query(PatientMedication).filter(
                    PatientMedication.patient_id == patient.id,
                    PatientMedication.is_active == True
                ).count()
                
                result.append({
                    'patient_id': patient.patient_uid,
                    'name': patient.full_name,
                    'age': patient.age,
                    'gender': patient.gender,
                    'prescriptions_count': prescription_count,
                    'active_medications': active_meds,
                    'last_visit': patient.prescriptions[0].prescription_date.isoformat() if patient.prescriptions else None
                })
            
            return result
        finally:
            session.close()
    
    # ==================== PRESCRIPTION OPERATIONS ====================
    
    def add_prescription(
        self,
        patient_uid: str,
        prescription_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a new prescription for a patient.
        This is the main entry point for storing scanned prescriptions.
        """
        import uuid
        session = self._get_session()
        
        try:
            # Get or create patient
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                # Create patient from prescription data
                name = prescription_data.get('patient_name', f'Patient {patient_uid}')
                first_name = name
                last_name = patient_uid
                if name and ' ' in name:
                    parts = name.strip().split(' ', 1)
                    first_name = parts[0]
                    last_name = parts[1] if len(parts) > 1 else patient_uid
                
                patient = Patient(
                    patient_uid=patient_uid,
                    first_name=first_name,
                    last_name=last_name,
                    gender=prescription_data.get('patient_gender'),
                    phone=prescription_data.get('patient_phone'),
                    address=prescription_data.get('patient_address')
                )
                session.add(patient)
                session.flush()
            
            # Parse prescription date - try multiple sources
            prescription_date = datetime.utcnow()
            date_parsed = False
            
            # Try prescription_date field first
            date_str = prescription_data.get('prescription_date')
            if date_str:
                try:
                    from dateutil import parser
                    prescription_date = parser.parse(date_str, dayfirst=True)
                    date_parsed = True
                    logger.info(f"Parsed prescription_date: {prescription_date}")
                except Exception as e:
                    logger.warning(f"Failed to parse prescription_date '{date_str}': {e}")
            
            # Fallback to scan_timestamp if prescription_date not parsed
            if not date_parsed:
                scan_ts = prescription_data.get('scan_timestamp')
                if scan_ts:
                    try:
                        from dateutil import parser
                        prescription_date = parser.parse(scan_ts)
                        logger.info(f"Using scan_timestamp as prescription date: {prescription_date}")
                    except Exception as e:
                        logger.warning(f"Failed to parse scan_timestamp '{scan_ts}': {e}")
            
            # Create prescription
            prescription_uid = str(uuid.uuid4())[:8].upper()
            prescription = Prescription(
                prescription_uid=prescription_uid,
                patient_id=patient.id,
                prescription_date=prescription_date,
                doctor_name=prescription_data.get('doctor_name'),
                doctor_registration_no=prescription_data.get('doctor_reg_no'),
                doctor_qualification=prescription_data.get('doctor_qualification'),
                clinic_name=prescription_data.get('clinic_name'),
                diagnosis=prescription_data.get('diagnosis', []),
                chief_complaint='; '.join(prescription_data.get('chief_complaints', [])),
                vitals=prescription_data.get('vitals', {}),
                advice=prescription_data.get('advice', []),
                original_filename=prescription_data.get('filename'),
                raw_ocr_text=prescription_data.get('raw_ocr_text', ''),
                ocr_confidence=prescription_data.get('ocr_confidence', 0),
                extraction_confidence=prescription_data.get('confidence', 0)
            )
            session.add(prescription)
            session.flush()
            
            # Add medications
            medications_added = []
            for med_data in prescription_data.get('medications', []):
                med_name = med_data.get('name', '')
                if not med_name:
                    continue
                
                # Add to prescription medications
                presc_med = PrescriptionMedication(
                    prescription_id=prescription.id,
                    name=med_name,
                    dosage=med_data.get('dosage', ''),
                    frequency=med_data.get('frequency', ''),
                    timing=med_data.get('timing', ''),
                    duration=med_data.get('duration', ''),
                    route=med_data.get('route', 'oral'),
                    instructions=med_data.get('instructions', '')
                )
                session.add(presc_med)
                
                # Add/update patient medication (for tracking active meds)
                existing_med = session.query(PatientMedication).filter(
                    PatientMedication.patient_id == patient.id,
                    func.lower(PatientMedication.name) == med_name.lower(),
                    PatientMedication.is_active == True
                ).first()
                
                if existing_med:
                    # Update existing medication
                    existing_med.dosage = med_data.get('dosage', existing_med.dosage)
                    existing_med.frequency = med_data.get('frequency', existing_med.frequency)
                    existing_med.prescriber = prescription_data.get('doctor_name')
                    existing_med.updated_at = datetime.utcnow()
                else:
                    # Create new patient medication
                    patient_med = PatientMedication(
                        patient_id=patient.id,
                        prescription_id=prescription.id,
                        name=med_name,
                        dosage=med_data.get('dosage', ''),
                        frequency=med_data.get('frequency', ''),
                        start_date=prescription_date,
                        is_active=True,
                        prescriber=prescription_data.get('doctor_name')
                    )
                    session.add(patient_med)
                
                medications_added.append(med_name)
            
            # Add timeline event
            timeline_event = TimelineEvent(
                patient_id=patient.id,
                prescription_id=prescription.id,
                event_type='prescription_added',
                event_date=prescription_date,
                description=f"New prescription from Dr. {prescription.doctor_name or 'Unknown'}",
                details={
                    'doctor': prescription.doctor_name,
                    'clinic': prescription.clinic_name,
                    'medications': medications_added,
                    'diagnosis': prescription.diagnosis
                },
                severity=AlertSeverity.INFO
            )
            session.add(timeline_event)
            
            # Add medication timeline events
            for med_name in medications_added:
                med_event = TimelineEvent(
                    patient_id=patient.id,
                    prescription_id=prescription.id,
                    event_type='medication_started',
                    event_date=prescription_date,
                    description=f"New medication: {med_name}",
                    details={'medication': med_name, 'prescriber': prescription.doctor_name},
                    severity=AlertSeverity.INFO
                )
                session.add(med_event)
            
            session.commit()
            
            logger.info(f"Added prescription {prescription_uid} for patient {patient_uid} with {len(medications_added)} medications")
            
            return {
                'success': True,
                'prescription_uid': prescription_uid,
                'prescription_number': len(patient.prescriptions),
                'patient_id': patient.id,
                'patient_uid': patient_uid,
                'medications_added': medications_added,
                'prescription_date': prescription_date.isoformat()
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding prescription: {e}")
            raise
        finally:
            session.close()
    
    def get_patient_prescriptions(self, patient_uid: str) -> List[Dict[str, Any]]:
        """Get all prescriptions for a patient"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return []
            
            result = []
            for presc in patient.prescriptions:
                meds = []
                for med in presc.medications:
                    meds.append({
                        'name': med.name,
                        'dosage': med.dosage,
                        'frequency': med.frequency,
                        'timing': med.timing,
                        'duration': med.duration,
                        'instructions': med.instructions
                    })
                
                result.append({
                    'prescription_uid': presc.prescription_uid,
                    'prescription_date': presc.prescription_date.isoformat() if presc.prescription_date else None,
                    'doctor_name': presc.doctor_name,
                    'clinic_name': presc.clinic_name,
                    'diagnosis': presc.diagnosis or [],
                    'medications': meds,
                    'vitals': presc.vitals or {},
                    'advice': presc.advice or []
                })
            
            return result
        finally:
            session.close()
    
    def get_patient_medications(self, patient_uid: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get patient's medications"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return []
            
            query = session.query(PatientMedication).filter(
                PatientMedication.patient_id == patient.id
            )
            
            if active_only:
                query = query.filter(PatientMedication.is_active == True)
            
            medications = query.order_by(desc(PatientMedication.start_date)).all()
            
            return [
                {
                    'name': med.name,
                    'generic_name': med.generic_name,
                    'dosage': med.dosage,
                    'frequency': med.frequency,
                    'start_date': med.start_date.isoformat() if med.start_date else None,
                    'end_date': med.end_date.isoformat() if med.end_date else None,
                    'is_active': med.is_active,
                    'prescriber': med.prescriber
                }
                for med in medications
            ]
        finally:
            session.close()
    
    def get_patient_allergies(self, patient_uid: str) -> List[str]:
        """Get patient's allergies"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return []
            
            return [allergy.name for allergy in patient.allergies]
        finally:
            session.close()
    
    def get_patient_conditions(self, patient_uid: str) -> List[str]:
        """Get patient's chronic conditions"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return []
            
            return [condition.name for condition in patient.conditions]
        finally:
            session.close()
    
    def add_allergy(self, patient_uid: str, allergy_name: str) -> Dict[str, Any]:
        """Add an allergy to patient's record"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            # Check if allergy already exists
            allergy = session.query(Allergy).filter(
                func.lower(Allergy.name) == allergy_name.lower().strip()
            ).first()
            
            if not allergy:
                allergy = Allergy(name=allergy_name.strip(), category="drug")
                session.add(allergy)
                session.flush()
            
            if allergy not in patient.allergies:
                patient.allergies.append(allergy)
                session.commit()
                return {'success': True, 'message': f'Allergy "{allergy_name}" added'}
            else:
                return {'success': True, 'message': f'Allergy "{allergy_name}" already exists'}
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding allergy: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def add_condition(self, patient_uid: str, condition_name: str) -> Dict[str, Any]:
        """Add a chronic condition to patient's record"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            # Check if condition already exists
            condition = session.query(Condition).filter(
                func.lower(Condition.name) == condition_name.lower().strip()
            ).first()
            
            if not condition:
                condition = Condition(name=condition_name.strip())
                session.add(condition)
                session.flush()
            
            if condition not in patient.conditions:
                patient.conditions.append(condition)
                session.commit()
                return {'success': True, 'message': f'Condition "{condition_name}" added'}
            else:
                return {'success': True, 'message': f'Condition "{condition_name}" already exists'}
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding condition: {e}")
            return {'error': str(e)}
        finally:
            session.close()

    def get_patient_timeline(self, patient_uid: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get patient's medical timeline"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return []
            
            events = session.query(TimelineEvent).filter(
                TimelineEvent.patient_id == patient.id
            ).order_by(desc(TimelineEvent.event_date)).limit(limit).all()
            
            return [
                {
                    'event_type': event.event_type,
                    'event_date': event.event_date.isoformat() if event.event_date else None,
                    'description': event.description,
                    'details': event.details or {},
                    'severity': event.severity.value if event.severity else 'info'
                }
                for event in events
            ]
        finally:
            session.close()
    
    def get_patient_summary(self, patient_uid: str) -> Dict[str, Any]:
        """Get comprehensive patient summary for AI and display"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            # Get active medications
            active_meds = session.query(PatientMedication).filter(
                PatientMedication.patient_id == patient.id,
                PatientMedication.is_active == True
            ).all()
            
            # Get all prescriptions
            prescriptions = patient.prescriptions
            
            # Get timeline
            timeline = session.query(TimelineEvent).filter(
                TimelineEvent.patient_id == patient.id
            ).order_by(desc(TimelineEvent.event_date)).limit(20).all()
            
            # Collect all diagnoses
            all_diagnoses = set()
            all_doctors = set()
            for presc in prescriptions:
                if presc.diagnosis:
                    for d in presc.diagnosis:
                        all_diagnoses.add(d)
                if presc.doctor_name:
                    all_doctors.add(presc.doctor_name)
            
            return {
                'patient': self._patient_to_dict(patient),
                'statistics': {
                    'total_prescriptions': len(prescriptions),
                    'active_medications': len(active_meds),
                    'total_allergies': len(patient.allergies),
                    'total_conditions': len(patient.conditions),
                    'timeline_events': len(timeline)
                },
                'current_medications': [
                    {
                        'name': med.name,
                        'dosage': med.dosage,
                        'frequency': med.frequency,
                        'prescriber': med.prescriber,
                        'start_date': med.start_date.isoformat() if med.start_date else None
                    }
                    for med in active_meds
                ],
                'allergies': [a.name for a in patient.allergies],
                'conditions': [c.name for c in patient.conditions],
                'all_diagnoses': list(all_diagnoses),
                'treating_doctors': list(all_doctors),
                'recent_timeline': [
                    {
                        'event_type': e.event_type,
                        'event_date': e.event_date.isoformat() if e.event_date else None,
                        'description': e.description
                    }
                    for e in timeline[:10]
                ]
            }
        finally:
            session.close()
    
    def _patient_to_dict(self, patient: Patient) -> Dict[str, Any]:
        """Convert patient model to dictionary"""
        return {
            'id': patient.id,
            'patient_uid': patient.patient_uid,
            'name': patient.full_name,
            'first_name': patient.first_name,
            'last_name': patient.last_name,
            'age': patient.age,
            'gender': patient.gender,
            'phone': patient.phone,
            'address': patient.address,
            'allergies': [a.name for a in patient.allergies],
            'conditions': [c.name for c in patient.conditions],
            'created_at': patient.created_at.isoformat() if patient.created_at else None,
            'updated_at': patient.updated_at.isoformat() if patient.updated_at else None
        }
    
    def remove_allergy(self, patient_uid: str, allergy_name: str) -> Dict[str, Any]:
        """Remove an allergy from patient's record"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            allergy = session.query(Allergy).filter(
                func.lower(Allergy.name) == allergy_name.lower().strip()
            ).first()
            
            if allergy and allergy in patient.allergies:
                patient.allergies.remove(allergy)
                session.commit()
                return {'success': True, 'message': f'Allergy "{allergy_name}" removed'}
            
            return {'success': True, 'message': f'Allergy "{allergy_name}" not found on patient'}
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing allergy: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def remove_condition(self, patient_uid: str, condition_name: str) -> Dict[str, Any]:
        """Remove a condition from patient's record"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            condition = session.query(Condition).filter(
                func.lower(Condition.name) == condition_name.lower().strip()
            ).first()
            
            if condition and condition in patient.conditions:
                patient.conditions.remove(condition)
                session.commit()
                return {'success': True, 'message': f'Condition "{condition_name}" removed'}
            
            return {'success': True, 'message': f'Condition "{condition_name}" not found on patient'}
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing condition: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def add_symptom(self, patient_uid: str, symptom_name: str, severity: str = None) -> Dict[str, Any]:
        """Add a symptom to patient's record as a timeline event"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            # Create a timeline event for the symptom
            event = TimelineEvent(
                patient_id=patient.id,
                event_type='symptom_reported',
                event_date=datetime.utcnow(),
                description=f"Symptom reported: {symptom_name}" + (f" (Severity: {severity})" if severity else ""),
                details={'symptom': symptom_name, 'severity': severity},
                severity=AlertSeverity.WARNING if severity and severity.lower() == 'severe' else AlertSeverity.INFO
            )
            session.add(event)
            session.commit()
            
            return {'success': True, 'message': f'Symptom "{symptom_name}" recorded'}
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding symptom: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def add_medication_manual(
        self,
        patient_uid: str,
        medication_name: str,
        dosage: str = None,
        frequency: str = None,
        prescriber: str = None,
        reason: str = None,
        start_date: str = None
    ) -> Dict[str, Any]:
        """Manually add a medication (not from prescription OCR)"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            # Parse start date or use now
            med_start_date = datetime.utcnow()
            if start_date:
                try:
                    from dateutil import parser
                    med_start_date = parser.parse(start_date, dayfirst=True)
                except:
                    pass
            
            # Check if medication already exists for this patient
            existing = session.query(PatientMedication).filter(
                PatientMedication.patient_id == patient.id,
                func.lower(PatientMedication.name) == medication_name.lower().strip(),
                PatientMedication.is_active == True
            ).first()
            
            if existing:
                return {'success': True, 'message': f'Medication "{medication_name}" already active for patient'}
            
            # Create patient medication
            patient_med = PatientMedication(
                patient_id=patient.id,
                name=medication_name.strip(),
                dosage=dosage,
                frequency=frequency,
                start_date=med_start_date,
                is_active=True,
                prescriber=prescriber,
                change_reason=reason or 'Manually added'
            )
            session.add(patient_med)
            
            # Create timeline event
            event = TimelineEvent(
                patient_id=patient.id,
                event_type='medication_started',
                event_date=med_start_date,
                description=f"Medication started: {medication_name}" + (f" by Dr. {prescriber}" if prescriber else " (manual entry)"),
                details={'medication': medication_name, 'dosage': dosage, 'frequency': frequency, 'prescriber': prescriber},
                severity=AlertSeverity.INFO
            )
            session.add(event)
            session.commit()
            
            return {
                'success': True, 
                'message': f'Medication "{medication_name}" added',
                'medication_id': patient_med.id
            }
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding medication: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def stop_medication(self, patient_uid: str, medication_id: int, reason: str = None) -> Dict[str, Any]:
        """Stop/discontinue a medication"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            medication = session.query(PatientMedication).filter(
                PatientMedication.id == medication_id,
                PatientMedication.patient_id == patient.id
            ).first()
            
            if not medication:
                return {'error': f'Medication not found'}
            
            if not medication.is_active:
                return {'success': True, 'message': 'Medication already stopped'}
            
            medication.is_active = False
            medication.end_date = datetime.utcnow()
            medication.change_reason = reason or 'Discontinued'
            
            # Create timeline event
            event = TimelineEvent(
                patient_id=patient.id,
                event_type='medication_stopped',
                event_date=datetime.utcnow(),
                description=f"Medication stopped: {medication.name}" + (f" - {reason}" if reason else ""),
                details={'medication': medication.name, 'reason': reason},
                severity=AlertSeverity.INFO
            )
            session.add(event)
            session.commit()
            
            return {'success': True, 'message': f'Medication "{medication.name}" stopped'}
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error stopping medication: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def update_patient(self, patient_uid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update patient basic information"""
        session = self._get_session()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_uid == patient_uid
            ).first()
            
            if not patient:
                return {'error': f'Patient {patient_uid} not found'}
            
            # Update fields if provided
            if data.get('first_name'):
                patient.first_name = data['first_name']
            if data.get('last_name'):
                patient.last_name = data['last_name']
            if data.get('phone'):
                patient.phone = data['phone']
            if data.get('email'):
                patient.email = data['email']
            if data.get('address'):
                patient.address = data['address']
            if data.get('gender'):
                patient.gender = data['gender']
            if data.get('blood_group'):
                patient.blood_group = data['blood_group']
            if data.get('date_of_birth'):
                try:
                    from dateutil import parser
                    patient.date_of_birth = parser.parse(data['date_of_birth'])
                except:
                    pass
            if data.get('weight_kg') is not None:
                patient.weight_kg = float(data['weight_kg'])
            if data.get('height_cm') is not None:
                patient.height_cm = float(data['height_cm'])
            if data.get('emergency_contact_name'):
                patient.emergency_contact_name = data['emergency_contact_name']
            if data.get('emergency_contact_phone'):
                patient.emergency_contact_phone = data['emergency_contact_phone']
            if data.get('notes'):
                patient.notes = data['notes']
            
            patient.updated_at = datetime.utcnow()
            session.commit()
            
            return {
                'success': True,
                'patient_uid': patient.patient_uid,
                'name': patient.full_name,
                'updated_at': patient.updated_at.isoformat()
            }
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating patient: {e}")
            return {'error': str(e)}
        finally:
            session.close()


# Singleton instance
_unified_patient_service: Optional[UnifiedPatientService] = None


def get_unified_patient_service() -> UnifiedPatientService:
    """Get or create the unified patient service singleton"""
    global _unified_patient_service
    if _unified_patient_service is None:
        _unified_patient_service = UnifiedPatientService()
    return _unified_patient_service


# Export singleton for easy import
unified_patient_service = get_unified_patient_service()
