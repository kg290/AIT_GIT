"""
Production Database Configuration
Supports SQLite (dev) and PostgreSQL (production)
"""
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from backend.database.models import Base


class DatabaseManager:
    """Database connection manager with support for multiple backends"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
    
    def init_db(self, database_url: str = None):
        """Initialize database connection"""
        if self._initialized:
            return
        
        # Get database URL from environment or use SQLite default
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL",
                "sqlite:///./data/hospital_prescriptions.db"
            )
        
        # Handle PostgreSQL URL format from some cloud providers
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        # Create engine with appropriate settings
        if "sqlite" in database_url:
            # SQLite settings
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
            )
            
            # Enable foreign keys for SQLite
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            # PostgreSQL settings
            self.engine = create_engine(
                database_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        
        self._initialized = True
        print(f"✓ Database initialized: {database_url.split('@')[-1] if '@' in database_url else database_url}")
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self._initialized:
            self.init_db()
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self._initialized = False
    
    def init_database(self):
        """Initialize database with tables and initial data"""
        self.init_db()
        with self.session_scope() as db:
            create_initial_data(db)


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions"""
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


def init_database(database_url: str = None):
    """Initialize the database"""
    db_manager.init_db(database_url)


def create_initial_data(db: Session):
    """Create initial data for a new database"""
    from backend.database.models import User, UserRole, Allergy, Condition, SystemSetting
    from backend.services.auth_service import AuthService
    
    auth = AuthService()
    
    # Check if admin exists
    admin = db.query(User).filter(User.username == "admin").first()
    if not admin:
        # Create default admin user
        admin = User(
            username="admin",
            email="admin@hospital.com",
            password_hash=auth.hash_password("admin123"),  # Change in production!
            full_name="System Administrator",
            role=UserRole.ADMIN,
            employee_id="EMP001",
            is_active=True
        )
        db.add(admin)
        print("✓ Created default admin user (username: admin, password: admin123)")
    
    # Create demo doctor user
    doctor = db.query(User).filter(User.username == "dr.sharma").first()
    if not doctor:
        doctor = User(
            username="dr.sharma",
            email="dr.sharma@hospital.com",
            password_hash=auth.hash_password("doctor123"),
            full_name="Dr. Amit Sharma",
            role=UserRole.DOCTOR,
            employee_id="DOC001",
            department="General Medicine",
            is_active=True
        )
        db.add(doctor)
        print("✓ Created demo doctor (username: dr.sharma, password: doctor123)")
    
    # Create demo pharmacist user
    pharmacist = db.query(User).filter(User.username == "pharma.patel").first()
    if not pharmacist:
        pharmacist = User(
            username="pharma.patel",
            email="pharma.patel@hospital.com",
            password_hash=auth.hash_password("pharma123"),
            full_name="Priya Patel",
            role=UserRole.PHARMACIST,
            employee_id="PHA001",
            department="Pharmacy",
            is_active=True
        )
        db.add(pharmacist)
        print("✓ Created demo pharmacist (username: pharma.patel, password: pharma123)")
    
    # Create common allergies
    common_allergies = [
        ("Penicillin", "drug", "severe"),
        ("Sulfa", "drug", "moderate"),
        ("Aspirin", "drug", "moderate"),
        ("NSAIDs", "drug", "moderate"),
        ("Cephalosporins", "drug", "moderate"),
        ("Latex", "environmental", "moderate"),
        ("Iodine", "drug", "moderate"),
        ("Codeine", "drug", "moderate"),
    ]
    
    for name, category, severity in common_allergies:
        existing = db.query(Allergy).filter(Allergy.name == name).first()
        if not existing:
            db.add(Allergy(name=name, category=category, severity=severity))
    
    # Create common conditions
    common_conditions = [
        ("Diabetes Mellitus", "E11", "metabolic"),
        ("Hypertension", "I10", "cardiovascular"),
        ("Chronic Kidney Disease", "N18", "renal"),
        ("Heart Failure", "I50", "cardiovascular"),
        ("Asthma", "J45", "respiratory"),
        ("COPD", "J44", "respiratory"),
        ("Liver Disease", "K76", "hepatic"),
        ("Pregnancy", "Z33", "obstetric"),
    ]
    
    for name, icd_code, category in common_conditions:
        existing = db.query(Condition).filter(Condition.name == name).first()
        if not existing:
            db.add(Condition(name=name, icd_code=icd_code, category=category))
    
    # Create system settings
    default_settings = [
        ("hospital_name", "City General Hospital", "Hospital name displayed in reports"),
        ("hospital_address", "", "Hospital address for reports"),
        ("hospital_phone", "", "Hospital contact number"),
        ("hospital_logo", "", "Path to hospital logo"),
        ("ocr_confidence_threshold", "0.7", "Minimum OCR confidence for auto-approval"),
        ("require_pharmacist_verification", "true", "Require pharmacist to verify prescriptions"),
        ("enable_safety_alerts", "true", "Enable drug interaction and allergy alerts"),
        ("session_timeout_minutes", "30", "Session timeout in minutes"),
        ("max_login_attempts", "5", "Maximum failed login attempts before lockout"),
    ]
    
    for key, value, description in default_settings:
        existing = db.query(SystemSetting).filter(SystemSetting.key == key).first()
        if not existing:
            db.add(SystemSetting(key=key, value=value, description=description))
    
    db.commit()
    print("✓ Initial data created")
