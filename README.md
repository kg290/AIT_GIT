# Medical AI Gateway 2.0 - Hospital Edition

🏥 **Production-Ready Hospital Prescription Management System**

A comprehensive medical document processing system that turns messy handwritten prescriptions into structured, verified, time-aware clinical data with full HIPAA compliance.

---

## 🎯 System Overview

| Component | Description |
|-----------|-------------|
| **Staff Portal** | Patient registration, prescription scanning, QR code generation |
| **Doctor Dashboard** | Patient lookup via QR, prescription history, AI assistant |
| **OCR Engine** | Google Cloud Vision for prescription digitization |
| **AI Assistant** | Natural language queries about patients and medications |
| **Safety Engine** | Drug interactions, allergy checks, contraindications |

---

## 🏥 Portal Features

### 👨‍💼 Staff Portal (`/staff`)
| Feature | Description |
|---------|-------------|
| **New Patient Registration** | Full demographics, allergies, conditions, emergency contact |
| **Prescription Upload** | Multi-file upload with drag & drop |
| **QR Code Generation** | Unique patient QR for quick lookup |
| **Existing Patient Lookup** | Scan QR or enter UID to add prescriptions |
| **Batch Processing** | Upload multiple prescriptions at once |

### 🩺 Doctor Dashboard (`/`)
| Feature | Description |
|---------|-------------|
| **QR Code Scanner** | Upload patient QR image to view full history |
| **Patient Lookup** | Search by UID, view complete medical profile |
| **Prescription History** | All prescriptions with medications, dates, doctors |
| **Active Medications** | Current medication list with dosages |
| **AI Assistant** | Ask questions about patient in natural language |
| **Safety Analysis** | Drug interactions, allergy checks |
| **Medication Timeline** | Visual timeline of medication changes |
| **Knowledge Graph** | Entity relationships visualization |

---

## 🔬 Core Features

### 1. Prescription OCR Processing
- Multi-format support (PDF, PNG, JPG, TIFF, BMP, WEBP)
- Google Cloud Vision AI integration
- Bounding box extraction with confidence scores
- Handwriting detection
- Mixed content handling

### 2. Text Cleaning & Normalization
- OCR error correction
- Medical abbreviation expansion (qd → once daily, bid → twice daily)
- Spelling correction with medical dictionary
- Dosage and frequency normalization

### 3. AI-Powered Entity Extraction
| Entity Type | Examples |
|-------------|----------|
| **Patient Info** | Name, Age, Gender, Phone, Address |
| **Doctor Info** | Name, Qualification, Registration No., Clinic |
| **Medications** | Drug name, Dosage, Frequency, Duration, Route |
| **Diagnosis** | ICD-10 compatible conditions |
| **Vitals** | BP, Pulse, Temperature, Weight, SpO2 |

### 4. Drug Normalization
- Brand name to generic mapping (50+ medications)
- Drug class identification
- Duplicate detection (same drug, different names)
- Example: Lipitor → Atorvastatin, Glucophage → Metformin

### 5. Drug Safety Analysis
| Check Type | Description |
|------------|-------------|
| **Drug-Drug Interactions** | 30+ specific interaction pairs |
| **Class-Level Interactions** | NSAIDs + Anticoagulants, etc. |
| **Allergy Checking** | Cross-reference with patient allergies |
| **Contraindications** | Medication vs condition conflicts |
| **Duplicate Therapy** | Same therapeutic class alerts |

### 6. Temporal Reasoning
- Medication timeline building
- Change detection (started/stopped/dose changed)
- Overlap analysis between prescriptions
- Visit comparison

### 7. Patient History Management
- Full medication history with start/end dates
- Condition tracking (active, resolved, chronic)
- Symptom history
- Visit summaries with snapshots

### 8. Knowledge Graph
- Patient ↔ Medication relationships
- Medication ↔ Condition relationships
- Condition ↔ Symptom relationships
- Visual graph exploration

### 9. AI Medical Assistant
- Natural language question answering
- Patient context-aware responses
- Load patient data by UID
- Quick action buttons for common queries

### 10. Audit & Compliance
- HIPAA-compliant audit logging
- Complete action tracking
- Correction history
- Export for compliance reporting

### 11. QR Code System
| Feature | Description |
|---------|-------------|
| **Generation** | Auto-generated on patient creation |
| **Format** | Contains patient UID (e.g., PT20260130-A1B2) |
| **Scanning** | Upload image to decode (pyzbar + OpenCV) |
| **Use Case** | Quick patient lookup for doctors |

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

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

---

## 📄 License

This project is for educational/hackathon purposes.

---

## 🙏 Acknowledgments

- Google Cloud Vision API for OCR
- FastAPI for the web framework
- Gemini AI for natural language processing
- The medical NLP community for inspiration

---

## 📞 Support

For issues and feature requests, please open a GitHub issue.

---

**Made with ❤️ for Healthcare**
