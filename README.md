# Medical AI Gateway 2.0 - Hospital Edition

🏥 **Production-Ready Hospital Prescription Management System**

A comprehensive medical document processing system that turns messy handwritten prescriptions into structured, verified, time-aware clinical data with full HIPAA compliance.

## 🎯 Hospital-Ready Features

### Security & Compliance
- ✅ **JWT Authentication** - Secure token-based login
- ✅ **Role-Based Access Control** - Admin, Doctor, Pharmacist, Nurse, Receptionist
- ✅ **HIPAA-Compliant Audit Logging** - Complete action tracking
- ✅ **Password Hashing** - bcrypt with salt
- ✅ **Session Management** - Token expiration & refresh

### Core Features
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

### 3. Entity Extraction
- **Medications**: Comprehensive drug database with 100+ medications
- **Dosages**: Pattern matching for various dosage formats
- **Frequencies**: Recognition of medical frequency terms
- **Routes**: Oral, IV, topical, etc.
- **Diagnoses**: ICD-10 compatible
- **Symptoms**: Patient-reported and clinical
- **Vitals**: Blood pressure, heart rate, temperature, etc.
- **Dates**: Multiple date format recognition

### 4. Drug Normalization
- Brand name to generic mapping (50+ medications)
- Drug class identification
- Duplicate detection (same drug, different names)

### 5. Prescription Structuring
- Free-text to structured format conversion
- Header extraction (patient, prescriber, date)
- Medication block identification
- Missing field detection

### 6. Drug Safety Analysis
- **Drug-Drug Interactions**: 30+ specific interaction pairs
- **Class-Level Interactions**: NSAIDs + Anticoagulants, etc.
- **Allergy Checking**: Patient allergy cross-reference
- **Contraindication Detection**
- **Duplicate Therapy Alerts**
- Severity classification (Minor/Moderate/Major/Contraindicated)

### 7. Temporal Reasoning
- Timeline building from prescriptions
- Medication period calculation
- Change detection (started/stopped/dose changed)
- Overlap analysis

### 8. Knowledge Graph
- Patient ↔ Medication relationships
- Medication ↔ Condition relationships
- Condition ↔ Symptom relationships
- In-memory or Neo4j backend

### 9. Conversational Querying
- Natural language question answering
- Time-aware queries
- Evidence-based responses with source references
- Related query suggestions

### 10. Audit & Compliance
- Immutable audit logging
- Correction tracking
- Version history
- Export for compliance reporting

### 11. Frontend Dashboard
- Document upload with drag-and-drop
- Real-time processing visualization
- Interactive medication management
- Drug interaction alerts
- Patient timeline view
- Chat interface for queries
- Knowledge graph visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud Vision API credentials

### Installation

```bash
# Clone the repository
cd "D:\Hackathon\OCR AIT"

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
- **Frontend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## 📚 API Endpoints

### Documents
- `POST /api/documents/upload` - Upload and process a document
- `POST /api/documents/process-text` - Process raw text
- `GET /api/documents/{id}` - Get processed document
- `POST /api/documents/{id}/correct` - Submit correction

### Patients
- `POST /api/patients/` - Create patient
- `GET /api/patients/{id}` - Get patient info
- `GET /api/patients/{id}/medications` - Get medications
- `POST /api/patients/{id}/medications` - Add medication
- `GET /api/patients/{id}/interactions` - Check drug interactions
- `GET /api/patients/{id}/timeline` - Get medical timeline
- `GET /api/patients/{id}/graph` - Get knowledge graph

### Query
- `GET /api/query/?question=...` - Ask medical questions
- `POST /api/query/chat` - Chat interface
- `GET /api/query/suggestions` - Get query suggestions

### Analytics
- `GET /api/analytics/dashboard` - Dashboard statistics
- `GET /api/analytics/medications/frequency` - Medication distribution
- `GET /api/analytics/interactions/summary` - Interaction summary

## 🏗️ Architecture

```
medical-ai-gateway/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── backend/
│   ├── config.py          # Application configuration
│   ├── database.py        # Database connection
│   ├── models/            # SQLAlchemy models
│   │   ├── base.py
│   │   ├── patient.py
│   │   ├── document.py
│   │   ├── prescription.py
│   │   ├── medication.py
│   │   ├── interaction.py
│   │   ├── medical_entity.py
│   │   ├── audit.py
│   │   └── timeline.py
│   ├── services/          # Business logic
│   │   ├── ocr_service.py
│   │   ├── text_cleaning_service.py
│   │   ├── entity_extraction_service.py
│   │   ├── drug_normalization_service.py
│   │   ├── prescription_structuring_service.py
│   │   ├── drug_interaction_service.py
│   │   ├── temporal_reasoning_service.py
│   │   ├── knowledge_graph_service.py
│   │   ├── audit_service.py
│   │   ├── query_service.py
│   │   └── document_processor.py
│   └── api/               # API routes
│       ├── documents.py
│       ├── patients.py
│       ├── query.py
│       └── analytics.py
├── frontend/
│   └── index.html         # Single-page application
└── data/                  # Data storage
    ├── uploads/           # Uploaded documents
    ├── processed/         # Processed results
    └── audit/             # Audit logs
```

## 🔧 Configuration

Environment variables (or `.env` file):

```env
DATABASE_URL=sqlite:///./data/medical_gateway.db
GOOGLE_CREDENTIALS_PATH=kg-hackathon-e3f03b59d928.json
USE_NEO4J=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
USE_LLM=false
OPENAI_API_KEY=your-key-here
```

## 📋 Example Usage

### Upload a Prescription

```python
import requests

# Upload a document
with open('prescription.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/documents/upload',
        files={'file': f},
        params={'patient_id': 'P001'}
    )

result = response.json()
print(f"Extracted medications: {result['entities']['medications']}")
print(f"Safety score: {result['safety']['safety_score']}")
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

### Ask Questions

```python
response = requests.get(
    'http://localhost:8000/api/query/',
    params={
        'question': 'What medications is the patient taking?',
        'patient_id': 'P001'
    }
)
print(response.json()['answer'])
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is for educational/hackathon purposes.

## 🙏 Acknowledgments

- Google Cloud Vision API for OCR
- FastAPI for the web framework
- The medical NLP community for inspiration
