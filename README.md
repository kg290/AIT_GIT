# 🏥 Medical AI Gateway 2.0 - Hospital Edition

> **Transforming Handwritten Prescriptions into Intelligent, Time-Aware Clinical Intelligence**

A production-ready medical document processing system that digitizes prescriptions, builds comprehensive patient timelines, and provides explainable AI-driven insights for better clinical decisions.

---

## 🎯 What Makes Us Different

| 🌟 Feature | Description |
|------------|-------------|
| **📊 Longitudinal Patient Timeline** | Complete visual history of every prescription, medication change, and treatment across time |
| **🧠 Explainable AI** | Every AI recommendation comes with clear reasoning - doctors know *why* not just *what* |
| **📋 Whole Patient Data** | 360° view of patient: medications, conditions, allergies, vitals, symptoms - all in one place |
| **⏱️ Temporal Intelligence** | Understands medication overlaps, gaps, dose changes, and treatment patterns over time |

---

## 🔄 Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MEDICAL AI GATEWAY - END TO END FLOW                  │
└─────────────────────────────────────────────────────────────────────────────────┘

  📄 PRESCRIPTION                👨‍💼 STAFF                   🤖 AI ENGINE                  👨‍⚕️ DOCTOR
  ─────────────                  ────────                   ──────────                    ────────
       │                             │                           │                            │
       │   Handwritten/Printed       │                           │                            │
       │   Prescription arrives      │                           │                            │
       ▼                             ▼                           │                            │
  ┌─────────┐                  ┌─────────────┐                   │                            │
  │  Scan/  │ ───────────────► │ Staff Portal│                   │                            │
  │ Upload  │                  │   Upload    │                   │                            │
  └─────────┘                  └──────┬──────┘                   │                            │
                                      │                          │                            │
                                      ▼                          ▼                            │
                               ┌──────────────┐           ┌─────────────┐                     │
                               │ Patient Reg/ │ ────────► │  OCR Engine │                     │
                               │ QR Generation│           │  (Vision AI)│                     │
                               └──────────────┘           └──────┬──────┘                     │
                                                                 │                            │
                                                                 ▼                            │
                                                          ┌─────────────┐                     │
                                                          │    Text     │                     │
                                                          │  Cleaning   │                     │
                                                          └──────┬──────┘                     │
                                                                 │                            │
                                                                 ▼                            │
                                                          ┌─────────────┐                     │
                                                          │   Entity    │                     │
                                                          │ Extraction  │                     │
                                                          │  (Gemini)   │                     │
                                                          └──────┬──────┘                     │
                                                                 │                            │
                                                                 ▼                            │
                                                          ┌─────────────┐                     │
                                                          │    Drug     │                     │
                                                          │Normalization│                     │
                                                          └──────┬──────┘                     │
                                                                 │                            │
                                                                 ▼                            │
                                                          ┌─────────────┐                     │
                                                          │   Safety    │                     │
                                                          │  Analysis   │                     │
                                                          └──────┬──────┘                     │
                                                                 │                            │
                                                                 ▼                            │
                                                          ┌─────────────┐                     │
                                                          │  Timeline   │                     │
                                                          │  Building   │                     │
                                                          └──────┬──────┘                     │
                                                                 │                            │
                                                                 ▼                            ▼
                                                          ┌─────────────┐           ┌─────────────┐
                                                          │ Structured  │ ────────► │   Doctor    │
                                                          │    Data     │           │  Dashboard  │
                                                          └─────────────┘           └─────────────┘
```

---

## 👥 Role-Based Workflow

### 👨‍💼 What Staff Does

| Step | Action | Outcome |
|------|--------|---------|
| 1️⃣ | **Register New Patient** | Enter demographics, allergies, existing conditions, emergency contact |
| 2️⃣ | **Scan/Upload Prescription** | Drag & drop or upload prescription images (supports multiple at once) |
| 3️⃣ | **Generate QR Code** | Unique patient QR code created automatically |
| 4️⃣ | **Add Future Prescriptions** | Scan QR or enter UID to add new prescriptions to existing patient |

> **Staff Portal Location:** `/staff`

---

### 🤖 What AI Does (Behind the Scenes)

| Stage | AI Action | Technology |
|-------|-----------|------------|
| **OCR** | Digitizes handwritten/printed prescriptions | Google Cloud Vision |
| **Text Cleaning** | Fixes OCR errors, expands abbreviations (qd→once daily) | Medical Dictionary |
| **Entity Extraction** | Extracts patient info, doctor info, medications, diagnosis, vitals | Google Gemini AI |
| **Drug Normalization** | Maps brand names to generics (Lipitor→Atorvastatin) | Drug Database |
| **Safety Analysis** | Checks drug interactions, allergies, contraindications | Safety Engine |
| **Timeline Building** | Creates temporal medication history with change detection | Temporal Reasoner |
| **Explainability** | Generates human-readable explanations for all AI decisions | Explainability Engine |

---

### 👨‍⚕️ What Doctors Get

| Benefit | Description |
|---------|-------------|
| **📊 Complete Timeline View** | Visual timeline showing every medication prescribed across all visits |
| **🔄 Medication Change Tracking** | See what was started, stopped, or dose-changed over time |
| **⚠️ Safety Alerts** | Instant warnings for drug interactions, allergies, contraindications |
| **💡 Explainable Insights** | Every AI recommendation includes clear reasoning |
| **🔍 Quick Patient Lookup** | Scan QR code or enter UID for instant access |
| **💬 AI Assistant** | Ask natural language questions about the patient |
| **📈 Longitudinal Analysis** | Understand treatment patterns over weeks, months, years |
| **🩺 Active Medications** | Current medication list with dosages at a glance |

> **Doctor Dashboard Location:** `/` (Home)

---

## ⭐ Core Features Explained

### 1. 📊 Longitudinal Patient Timeline

The heart of our system - a **complete temporal view** of patient's medical journey:

```
PATIENT TIMELINE EXAMPLE:
────────────────────────────────────────────────────────────────────────────
Jan 2024     │ Feb 2024      │ Mar 2024       │ Apr 2024       │ May 2024
────────────────────────────────────────────────────────────────────────────
Metformin    │ Metformin     │ Metformin      │ Metformin      │ Metformin
500mg BD     │ 500mg BD      │ ⬆️ 1000mg BD   │ 1000mg BD      │ 1000mg BD
             │               │                │                │
             │ +Lisinopril   │ Lisinopril     │ Lisinopril     │ Lisinopril
             │ 5mg OD        │ 5mg OD         │ ⬆️ 10mg OD     │ 10mg OD
             │               │                │                │
             │               │ +Atorvastatin  │ Atorvastatin   │ Atorvastatin
             │               │ 10mg OD        │ 10mg OD        │ 10mg OD
────────────────────────────────────────────────────────────────────────────
```

**What it tracks:**
- ✅ Medication starts and stops
- ✅ Dose changes over time
- ✅ Overlapping medications
- ✅ Treatment gaps
- ✅ Visit-by-visit changes

---

### 2. 🧠 Explainable AI

Every AI decision comes with **transparent reasoning**:

```
┌─────────────────────────────────────────────────────────────────────┐
│ ⚠️ DRUG INTERACTION ALERT                                          │
├─────────────────────────────────────────────────────────────────────┤
│ Warfarin + Aspirin detected                                         │
│                                                                     │
│ 🔍 EXPLANATION:                                                     │
│ • Both medications affect blood clotting                            │
│ • Combined use increases bleeding risk by 40%                       │
│ • Source: FDA Drug Interaction Database                             │
│ • Confidence: 95%                                                   │
│                                                                     │
│ 💡 RECOMMENDATION:                                                  │
│ Consider lower aspirin dose or alternative antiplatelet therapy     │
│                                                                     │
│ 📊 EVIDENCE:                                                        │
│ Based on patient's current medications and known interactions       │
└─────────────────────────────────────────────────────────────────────┘
```

**Why explainability matters:**
- Doctors understand the *reasoning* behind alerts
- Builds trust in AI recommendations
- Supports informed clinical decisions
- Reduces alert fatigue with context

---

### 3. 📋 Whole Patient Data (360° View)

Everything about the patient in one unified view:

| Data Category | What's Captured |
|---------------|-----------------|
| **Demographics** | Name, Age, Gender, Phone, Address, Emergency Contact |
| **Medical History** | Chronic conditions, past surgeries, hospitalizations |
| **Allergies** | Drug allergies, food allergies, environmental |
| **Current Medications** | Active drugs with dosage, frequency, duration |
| **Past Medications** | Complete prescription history with dates |
| **Vitals History** | BP, Pulse, Temperature, Weight, SpO2 over time |
| **Symptoms** | Current and historical symptom records |
| **Diagnosis** | ICD-10 compatible condition tracking |
| **Lab Results** | Integration-ready for lab data |

---

### 4. ⏱️ Temporal Intelligence

Our AI understands **time** in medical context:

| Capability | Description |
|------------|-------------|
| **Change Detection** | Automatically identifies when medications were started, stopped, or changed |
| **Overlap Analysis** | Detects when multiple prescriptions have overlapping medications |
| **Gap Identification** | Flags treatment gaps or missed refills |
| **Pattern Recognition** | Identifies recurring prescription patterns |
| **Visit Comparison** | Compares what changed between consecutive visits |

---

## 🔒 Safety & Compliance

| Feature | Description |
|---------|-------------|
| **Drug-Drug Interactions** | 30+ specific interaction pairs monitored |
| **Allergy Checking** | Cross-references every prescription with patient allergies |
| **Contraindications** | Medication vs condition conflict detection |
| **Duplicate Therapy** | Alerts for same therapeutic class |
| **HIPAA Compliance** | Complete audit logging of all actions |
| **Audit Trail** | Every action tracked for compliance reporting |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                           │
├────────────────────────────┬─────────────────────────────────────┤
│      Staff Portal          │         Doctor Dashboard            │
│    (Patient Registration)  │    (Patient Lookup & Analysis)      │
└────────────────────────────┴─────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                          API LAYER                               │
│    FastAPI with REST endpoints for all operations                │
└──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       AI SERVICES LAYER                          │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  OCR Service │ AI Extractor │ Drug Safety  │ Timeline Builder   │
│ (Cloud Vision)│  (Gemini)   │   Engine     │ (Temporal Reasoning)│
└──────────────┴──────────────┴──────────────┴────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                 │
│   Patient Database │ Prescription Store │ Audit Logs             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Benefits Summary

| For Staff | For AI | For Doctors |
|-----------|--------|-------------|
| ✅ Easy patient registration | ✅ Accurate OCR processing | ✅ Complete patient timeline |
| ✅ Simple prescription upload | ✅ Intelligent entity extraction | ✅ Explainable recommendations |
| ✅ QR code for quick lookup | ✅ Drug normalization | ✅ Safety alerts with context |
| ✅ Batch processing support | ✅ Automated safety checks | ✅ 360° patient view |
| ✅ Minimal training needed | ✅ Temporal analysis | ✅ AI-powered Q&A assistant |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud Vision API credentials

### Installation

```bash
# Clone the repository
cd "D:\Hackathon\OCR AIT"

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up Google Cloud credentials
# Place your service account JSON file as: kg-hackathon-e3f03b59d928.json
```

### Running the Application

```bash
# Start the server
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access the Application
| URL | Description |
|-----|-------------|
| http://localhost:8000 | Doctor Dashboard |
| http://localhost:8000/staff | Staff Portal |
| http://localhost:8000/api/docs | API Documentation (Swagger) |
| http://localhost:8000/api/redoc | API Documentation (ReDoc) |

---

## 📚 API Endpoints

### Staff API (`/api/staff/`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/create-patient` | POST | Register new patient with prescription |
| `/patient/{uid}` | GET | Get patient by UID |
| `/patient/{uid}/full-details` | GET | Complete patient data for doctors |
| `/patient/{uid}/ai-context` | GET | AI-optimized patient data |
| `/patient/{uid}/prescriptions` | GET | All patient prescriptions |
| `/patient/{uid}/timeline` | GET | Medication timeline |
| `/add-prescription` | POST | Add single prescription |
| `/add-prescriptions` | POST | Add multiple prescriptions |
| `/decode-qr` | POST | Decode QR code image |
| `/doctor/scan-qr` | POST | Doctor scans QR → full details |
| `/patients` | GET | List all patients |

### Documents API (`/api/documents/`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload and process a document |
| `/process-text` | POST | Process raw text |
| `/{id}` | GET | Get processed document |
| `/{id}/correct` | POST | Submit correction |

### Patients API (`/api/patients/`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | POST | Create patient |
| `/{id}` | GET | Get patient info |
| `/{id}/medications` | GET | Get medications |
| `/{id}/medications` | POST | Add medication |
| `/{id}/interactions` | GET | Check drug interactions |
| `/{id}/timeline` | GET | Get medical timeline |
| `/{id}/graph` | GET | Get knowledge graph |

### Query API (`/api/query/`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Ask medical questions |
| `/chat` | POST | Chat interface |
| `/suggestions` | GET | Get query suggestions |

### Analytics API (`/api/analytics/`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard` | GET | Dashboard statistics |
| `/medications/frequency` | GET | Medication distribution |
| `/interactions/summary` | GET | Interaction summary |

---

## 🏗️ Architecture

```
medical-ai-gateway/
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build
├── docker-compose.yml          # Multi-container orchestration
├── init-db.sql                 # Database schema
│
├── backend/
│   ├── config.py               # Application configuration
│   ├── database/               # Database connection & models
│   │   ├── connection.py
│   │   └── models.py
│   │
│   ├── models/                 # SQLAlchemy models
│   │   ├── patient.py
│   │   ├── document.py
│   │   ├── prescription.py
│   │   ├── medication.py
│   │   ├── knowledge_graph.py
│   │   ├── medical_entity.py
│   │   ├── audit.py
│   │   └── timeline.py
│   │
│   ├── services/               # Business logic
│   │   ├── ocr_service.py              # Google Vision OCR
│   │   ├── text_cleaning_service.py    # OCR text cleanup
│   │   ├── entity_extraction_service.py # Extract medical entities
│   │   ├── drug_normalization_service.py # Brand → Generic
│   │   ├── drug_interaction_service.py  # Safety checks
│   │   ├── prescription_structuring_service.py
│   │   ├── temporal_reasoning_service.py # Timeline analysis
│   │   ├── knowledge_graph_service.py   # Entity relationships
│   │   ├── patient_history_service.py   # Longitudinal tracking
│   │   ├── unified_patient_service.py   # Patient CRUD
│   │   ├── medical_ai_assistant.py      # AI queries
│   │   ├── gemini_service.py            # Gemini AI integration
│   │   ├── audit_service.py             # Action logging
│   │   ├── human_review_service.py      # Review queue
│   │   ├── uncertainty_service.py       # Confidence scoring
│   │   └── complete_processor.py        # Full pipeline
│   │
│   └── api/                    # API routes
│       ├── documents.py
│       ├── patients.py
│       ├── staff_api.py        # Staff portal endpoints
│       ├── query.py
│       └── analytics.py
│
├── frontend/
│   ├── index.html              # Landing page
│   ├── dashboard.html          # Doctor dashboard
│   └── staff.html              # Staff portal
│
└── data/                       # Data storage
    ├── uploads/                # Uploaded documents
    ├── processed/              # Processed results
    └── audit_logs/             # Audit logs
```

---

## 🔧 Configuration

Environment variables (or `.env` file):

```env
# Database
DATABASE_URL=sqlite:///./data/medical_gateway.db

# Google Cloud Vision OCR
GOOGLE_CREDENTIALS_PATH=kg-hackathon-e3f03b59d928.json

# Optional: Neo4j for Knowledge Graph
USE_NEO4J=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional: LLM Integration
USE_LLM=false
OPENAI_API_KEY=your-key-here
```

---

## 📋 Example Usage

### Staff Portal: Register New Patient

```python
import requests

# Create patient with prescription
files = {'file': open('prescription.pdf', 'rb')}
data = {
    'first_name': 'John',
    'last_name': 'Doe',
    'age': 45,
    'gender': 'Male',
    'phone': '9876543210',
    'allergies': 'Penicillin,Sulfa',
    'conditions': 'Diabetes,Hypertension'
}

response = requests.post(
    'http://localhost:8000/api/staff/create-patient',
    files=files,
    data=data
)

result = response.json()
print(f"Patient UID: {result['patient']['uid']}")
print(f"QR Code: Generated for quick lookup")
```

### Doctor: Lookup Patient by UID

```python
# Get complete patient details
response = requests.get(
    'http://localhost:8000/api/staff/patient/PT20260130-A1B2/full-details'
)

patient = response.json()
print(f"Name: {patient['patient']['name']}")
print(f"Allergies: {patient['patient']['allergies']}")
print(f"Active Medications: {patient['active_medications']}")
print(f"Prescription Count: {patient['summary']['total_prescriptions']}")
```

### Check Drug Interactions

```python
# Add medications and check interactions
requests.post(
    'http://localhost:8000/api/patients/P001/medications',
    params={'medication_name': 'Warfarin', 'dosage': '5mg'}
)

requests.post(
    'http://localhost:8000/api/patients/P001/medications',
    params={'medication_name': 'Aspirin', 'dosage': '325mg'}
)

# Check interactions
response = requests.get('http://localhost:8000/api/patients/P001/interactions')
interactions = response.json()
# Will detect Warfarin + Aspirin interaction (increased bleeding risk)
```

### AI Assistant Query

```python
# Get AI context for patient
response = requests.get(
    'http://localhost:8000/api/staff/patient/PT20260130-A1B2/ai-context'
)

context = response.json()['ai_context']
print(f"Summary: {context['summary_text']}")
# "Patient John Doe, 45 years old, Male. Known allergies: Penicillin, Sulfa. 
#  Chronic conditions: Diabetes, Hypertension. Currently taking: Metformin 500mg, 
#  Lisinopril 10mg. Total prescriptions on record: 5."
```

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t medai-gateway .
docker run -p 8000:8000 medai-gateway
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `sqlalchemy` | ORM |
| `google-cloud-vision` | OCR |
| `pyzbar` | QR code decoding |
| `opencv-python` | Image processing |
| `python-multipart` | File uploads |
| `qrcode` | QR code generation |

---

## 🔒 Security Features

- ✅ **JWT Authentication** - Secure token-based login
- ✅ **Role-Based Access Control** - Admin, Doctor, Pharmacist, Nurse, Receptionist
- ✅ **HIPAA-Compliant Audit Logging** - Complete action tracking
- ✅ **Password Hashing** - bcrypt with salt
- ✅ **Session Management** - Token expiration & refresh

---

## 💡 Why Choose Medical AI Gateway?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   📄 Messy Prescription  ➜  🤖 AI Processing  ➜  📊 Actionable Intelligence │
│                                                                             │
│   • Staff uploads in seconds                                                │
│   • AI extracts, normalizes, and analyzes automatically                     │
│   • Doctors get complete, explainable, time-aware patient insights          │
│                                                                             │
│   ✨ Result: Better clinical decisions, faster patient care                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 License

This project is for educational/hackathon purposes.

---

## 🙏 Acknowledgments

- Google Cloud Vision API for OCR
- Google Gemini AI for intelligent entity extraction
- FastAPI for the high-performance web framework

---

**Made with ❤️ for Healthcare | Transforming Prescriptions into Intelligence**
