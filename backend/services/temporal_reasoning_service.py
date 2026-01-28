"""
Temporal Reasoning Service - Timeline building and medication tracking
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MedicationPeriod:
    """Period when a medication was active"""
    medication_name: str
    generic_name: Optional[str]
    start_date: date
    end_date: Optional[date]
    dosage: Optional[str]
    frequency: Optional[str]
    source_document_id: Optional[int]
    is_ongoing: bool
    confidence: float


@dataclass
class TimelineEntry:
    """Single entry on the medical timeline"""
    event_type: str  # prescription, medication_start, medication_end, visit, diagnosis, etc.
    event_date: datetime
    event_end_date: Optional[datetime]
    title: str
    description: Optional[str]
    related_medications: List[str]
    related_data: Dict[str, Any]
    source_document_id: Optional[int]
    confidence: float


@dataclass
class MedicationChange:
    """Detected change in medication regimen"""
    medication_name: str
    change_type: str  # started, stopped, dose_changed, frequency_changed
    old_value: Optional[str]
    new_value: Optional[str]
    change_date: date
    previous_prescription_date: Optional[date]
    current_prescription_date: date
    confidence: float


@dataclass
class OverlapAnalysis:
    """Analysis of overlapping medications"""
    medication1: str
    medication2: str
    overlap_start: date
    overlap_end: Optional[date]
    duration_days: int
    is_significant: bool
    notes: str


@dataclass
class TemporalAnalysisResult:
    """Complete temporal analysis result"""
    timeline: List[TimelineEntry]
    medication_periods: List[MedicationPeriod]
    medication_changes: List[MedicationChange]
    overlapping_medications: List[OverlapAnalysis]
    current_medications: List[Dict]
    historical_medications: List[Dict]


class TemporalReasoningService:
    """
    Temporal medical reasoning service
    
    Features:
    - Chronological timeline building
    - Medication start/stop tracking
    - Overlap detection
    - Change detection across visits
    - Current vs historical comparison
    """
    
    def __init__(self):
        # Default duration assumptions when not specified
        self.default_durations = {
            'antibiotic': 7,
            'analgesic': 5,
            'antipyretic': 3,
            'antihistamine': 14,
            'ppi': 14,
            'default': 30  # Chronic medications
        }
    
    def build_timeline(self, prescriptions: List[Dict], 
                       visits: List[Dict] = None,
                       diagnoses: List[Dict] = None,
                       vitals: List[Dict] = None) -> TemporalAnalysisResult:
        """
        Build comprehensive medical timeline
        
        Args:
            prescriptions: List of prescription records with dates
            visits: Optional list of visit records
            diagnoses: Optional list of diagnosis records
            vitals: Optional list of vital sign records
            
        Returns:
            TemporalAnalysisResult with complete timeline
        """
        timeline = []
        medication_periods = []
        
        # Process prescriptions
        for rx in prescriptions:
            rx_date = self._parse_date(rx.get('date') or rx.get('prescription_date'))
            if not rx_date:
                continue
            
            # Add prescription event
            timeline.append(TimelineEntry(
                event_type='prescription',
                event_date=datetime.combine(rx_date, datetime.min.time()),
                event_end_date=None,
                title=f"Prescription from {rx.get('prescriber', 'Unknown')}",
                description=rx.get('diagnosis'),
                related_medications=[m.get('medication_name', '') for m in rx.get('medications', [])],
                related_data={'prescription_id': rx.get('id')},
                source_document_id=rx.get('document_id'),
                confidence=rx.get('confidence', 0.9)
            ))
            
            # Process each medication
            for med in rx.get('medications', []):
                period = self._create_medication_period(med, rx_date, rx.get('document_id'))
                medication_periods.append(period)
                
                # Add medication start event
                timeline.append(TimelineEntry(
                    event_type='medication_start',
                    event_date=datetime.combine(rx_date, datetime.min.time()),
                    event_end_date=datetime.combine(period.end_date, datetime.min.time()) if period.end_date else None,
                    title=f"Started {med.get('medication_name', 'Unknown')}",
                    description=f"{med.get('dosage', '')} {med.get('frequency', '')}".strip(),
                    related_medications=[med.get('medication_name', '')],
                    related_data=med,
                    source_document_id=rx.get('document_id'),
                    confidence=med.get('confidence', 0.8)
                ))
        
        # Process visits
        if visits:
            for visit in visits:
                visit_date = self._parse_date(visit.get('date'))
                if visit_date:
                    timeline.append(TimelineEntry(
                        event_type='visit',
                        event_date=datetime.combine(visit_date, datetime.min.time()),
                        event_end_date=None,
                        title=f"Visit: {visit.get('type', 'Consultation')}",
                        description=visit.get('notes'),
                        related_medications=[],
                        related_data=visit,
                        source_document_id=visit.get('document_id'),
                        confidence=0.95
                    ))
        
        # Process diagnoses
        if diagnoses:
            for dx in diagnoses:
                dx_date = self._parse_date(dx.get('date'))
                if dx_date:
                    timeline.append(TimelineEntry(
                        event_type='diagnosis',
                        event_date=datetime.combine(dx_date, datetime.min.time()),
                        event_end_date=None,
                        title=f"Diagnosis: {dx.get('name', 'Unknown')}",
                        description=dx.get('notes'),
                        related_medications=[],
                        related_data=dx,
                        source_document_id=dx.get('document_id'),
                        confidence=dx.get('confidence', 0.8)
                    ))
        
        # Process vitals
        if vitals:
            for vital in vitals:
                vital_date = self._parse_date(vital.get('date'))
                if vital_date:
                    timeline.append(TimelineEntry(
                        event_type='vital',
                        event_date=datetime.combine(vital_date, datetime.min.time()),
                        event_end_date=None,
                        title=f"{vital.get('type', 'Vital')}: {vital.get('value', '')}",
                        description=vital.get('interpretation'),
                        related_medications=[],
                        related_data=vital,
                        source_document_id=vital.get('document_id'),
                        confidence=0.95
                    ))
        
        # Sort timeline by date
        timeline.sort(key=lambda x: x.event_date)
        
        # Detect medication changes
        medication_changes = self._detect_medication_changes(medication_periods)
        
        # Find overlapping medications
        overlapping = self._find_overlapping_medications(medication_periods)
        
        # Categorize current vs historical
        current_meds, historical_meds = self._categorize_medications(medication_periods)
        
        return TemporalAnalysisResult(
            timeline=timeline,
            medication_periods=medication_periods,
            medication_changes=medication_changes,
            overlapping_medications=overlapping,
            current_medications=current_meds,
            historical_medications=historical_meds
        )
    
    def _parse_date(self, date_value) -> Optional[date]:
        """Parse various date formats"""
        if not date_value:
            return None
        
        if isinstance(date_value, date):
            return date_value
        
        if isinstance(date_value, datetime):
            return date_value.date()
        
        if isinstance(date_value, str):
            # Try common formats
            formats = [
                '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
                '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
                '%d %b %Y', '%d %B %Y', '%b %d, %Y'
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_value, fmt).date()
                except ValueError:
                    continue
        
        return None
    
    def _create_medication_period(self, med: Dict, start_date: date, 
                                  document_id: Optional[int]) -> MedicationPeriod:
        """Create medication period from prescription medication"""
        # Calculate end date
        duration_days = med.get('duration_days')
        if not duration_days:
            # Infer from drug class
            drug_class = med.get('drug_class', 'default')
            duration_days = self.default_durations.get(drug_class, self.default_durations['default'])
        
        end_date = start_date + timedelta(days=duration_days)
        
        # Check if ongoing (no end date specified and chronic medication)
        is_ongoing = False
        if not med.get('duration') and duration_days >= 30:
            is_ongoing = True
            end_date = None
        
        return MedicationPeriod(
            medication_name=med.get('medication_name', ''),
            generic_name=med.get('generic_name'),
            start_date=start_date,
            end_date=end_date,
            dosage=med.get('dosage'),
            frequency=med.get('frequency'),
            source_document_id=document_id,
            is_ongoing=is_ongoing,
            confidence=med.get('confidence', 0.8)
        )
    
    def _detect_medication_changes(self, periods: List[MedicationPeriod]) -> List[MedicationChange]:
        """Detect changes in medication regimens over time"""
        changes = []
        
        # Group by medication name (normalized)
        med_groups = defaultdict(list)
        for period in periods:
            key = (period.generic_name or period.medication_name).lower()
            med_groups[key].append(period)
        
        # Analyze each medication's history
        for med_name, med_periods in med_groups.items():
            # Sort by start date
            sorted_periods = sorted(med_periods, key=lambda x: x.start_date)
            
            for i in range(1, len(sorted_periods)):
                prev = sorted_periods[i-1]
                curr = sorted_periods[i]
                
                # Check for dosage change
                if prev.dosage != curr.dosage:
                    changes.append(MedicationChange(
                        medication_name=curr.medication_name,
                        change_type='dose_changed',
                        old_value=prev.dosage,
                        new_value=curr.dosage,
                        change_date=curr.start_date,
                        previous_prescription_date=prev.start_date,
                        current_prescription_date=curr.start_date,
                        confidence=min(prev.confidence, curr.confidence)
                    ))
                
                # Check for frequency change
                if prev.frequency != curr.frequency:
                    changes.append(MedicationChange(
                        medication_name=curr.medication_name,
                        change_type='frequency_changed',
                        old_value=prev.frequency,
                        new_value=curr.frequency,
                        change_date=curr.start_date,
                        previous_prescription_date=prev.start_date,
                        current_prescription_date=curr.start_date,
                        confidence=min(prev.confidence, curr.confidence)
                    ))
        
        # Detect newly started medications
        all_meds = set()
        periods_by_date = sorted(periods, key=lambda x: x.start_date)
        for period in periods_by_date:
            key = (period.generic_name or period.medication_name).lower()
            if key not in all_meds:
                changes.append(MedicationChange(
                    medication_name=period.medication_name,
                    change_type='started',
                    old_value=None,
                    new_value=f"{period.dosage} {period.frequency}".strip(),
                    change_date=period.start_date,
                    previous_prescription_date=None,
                    current_prescription_date=period.start_date,
                    confidence=period.confidence
                ))
                all_meds.add(key)
        
        return sorted(changes, key=lambda x: x.change_date)
    
    def _find_overlapping_medications(self, periods: List[MedicationPeriod]) -> List[OverlapAnalysis]:
        """Find medications that overlap in time"""
        overlaps = []
        
        for i, p1 in enumerate(periods):
            for p2 in periods[i+1:]:
                # Skip same medication
                if (p1.generic_name or p1.medication_name).lower() == \
                   (p2.generic_name or p2.medication_name).lower():
                    continue
                
                # Check for time overlap
                p1_end = p1.end_date or date.today()
                p2_end = p2.end_date or date.today()
                
                overlap_start = max(p1.start_date, p2.start_date)
                overlap_end = min(p1_end, p2_end)
                
                if overlap_start < overlap_end:
                    duration = (overlap_end - overlap_start).days
                    
                    # Determine if overlap is significant
                    is_significant = duration >= 7  # At least a week overlap
                    
                    overlaps.append(OverlapAnalysis(
                        medication1=p1.medication_name,
                        medication2=p2.medication_name,
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                        duration_days=duration,
                        is_significant=is_significant,
                        notes=f"Concurrent use for {duration} days"
                    ))
        
        return overlaps
    
    def _categorize_medications(self, periods: List[MedicationPeriod]) -> Tuple[List[Dict], List[Dict]]:
        """Categorize medications as current or historical"""
        today = date.today()
        current = []
        historical = []
        
        # Group by medication
        med_latest = {}
        for period in periods:
            key = (period.generic_name or period.medication_name).lower()
            if key not in med_latest or period.start_date > med_latest[key].start_date:
                med_latest[key] = period
        
        for key, period in med_latest.items():
            med_dict = {
                'medication_name': period.medication_name,
                'generic_name': period.generic_name,
                'dosage': period.dosage,
                'frequency': period.frequency,
                'start_date': str(period.start_date),
                'end_date': str(period.end_date) if period.end_date else None,
                'is_ongoing': period.is_ongoing
            }
            
            # Check if still active
            if period.is_ongoing:
                current.append(med_dict)
            elif period.end_date and period.end_date >= today:
                current.append(med_dict)
            else:
                historical.append(med_dict)
        
        return current, historical
    
    def compare_prescriptions(self, prescription1: Dict, prescription2: Dict) -> Dict:
        """
        Compare two prescriptions to identify differences
        
        Args:
            prescription1: Earlier prescription
            prescription2: Later prescription
            
        Returns:
            Dict with comparison results
        """
        meds1 = {(m.get('generic_name') or m.get('medication_name', '')).lower(): m 
                 for m in prescription1.get('medications', [])}
        meds2 = {(m.get('generic_name') or m.get('medication_name', '')).lower(): m 
                 for m in prescription2.get('medications', [])}
        
        # Find new, discontinued, and continued medications
        new_meds = []
        discontinued = []
        continued = []
        changed = []
        
        for med_key, med in meds2.items():
            if med_key not in meds1:
                new_meds.append(med)
            else:
                old_med = meds1[med_key]
                continued.append(med)
                
                # Check for changes
                if old_med.get('dosage') != med.get('dosage'):
                    changed.append({
                        'medication': med.get('medication_name'),
                        'change': 'dosage',
                        'from': old_med.get('dosage'),
                        'to': med.get('dosage')
                    })
                if old_med.get('frequency') != med.get('frequency'):
                    changed.append({
                        'medication': med.get('medication_name'),
                        'change': 'frequency',
                        'from': old_med.get('frequency'),
                        'to': med.get('frequency')
                    })
        
        for med_key, med in meds1.items():
            if med_key not in meds2:
                discontinued.append(med)
        
        return {
            'new_medications': new_meds,
            'discontinued_medications': discontinued,
            'continued_medications': continued,
            'changes': changed,
            'summary': {
                'new_count': len(new_meds),
                'discontinued_count': len(discontinued),
                'continued_count': len(continued),
                'change_count': len(changed)
            }
        }
    
    def to_dict(self, result: TemporalAnalysisResult) -> Dict:
        """Convert TemporalAnalysisResult to dictionary"""
        return {
            'timeline': [
                {
                    'event_type': e.event_type,
                    'event_date': e.event_date.isoformat(),
                    'event_end_date': e.event_end_date.isoformat() if e.event_end_date else None,
                    'title': e.title,
                    'description': e.description,
                    'related_medications': e.related_medications,
                    'confidence': e.confidence
                }
                for e in result.timeline
            ],
            'medication_periods': [
                {
                    'medication_name': p.medication_name,
                    'generic_name': p.generic_name,
                    'start_date': str(p.start_date),
                    'end_date': str(p.end_date) if p.end_date else None,
                    'dosage': p.dosage,
                    'frequency': p.frequency,
                    'is_ongoing': p.is_ongoing
                }
                for p in result.medication_periods
            ],
            'medication_changes': [
                {
                    'medication_name': c.medication_name,
                    'change_type': c.change_type,
                    'old_value': c.old_value,
                    'new_value': c.new_value,
                    'change_date': str(c.change_date)
                }
                for c in result.medication_changes
            ],
            'overlapping_medications': [
                {
                    'medication1': o.medication1,
                    'medication2': o.medication2,
                    'overlap_start': str(o.overlap_start),
                    'overlap_end': str(o.overlap_end),
                    'duration_days': o.duration_days,
                    'is_significant': o.is_significant
                }
                for o in result.overlapping_medications
            ],
            'current_medications': result.current_medications,
            'historical_medications': result.historical_medications
        }
