# 🩺 Healthcare AI Clinical Decision Support System

An advanced, industry-grade clinical decision support system (CDSS) built on **LangGraph**, **OpenAI**, and **PubMed**, designed for high-precision patient monitoring, risk stratification, and evidence-based treatment synthesis.

## 🚀 Overview
This system serves as a world-class clinical co-pilot, empowering healthcare providers with real-time, data-driven insights. It leverages a sophisticated agentic workflow to analyze patient biometrics, predict multi-disease risks (Oncology, Infectious, Chronic), and generate personalized clinical road maps backed by rigorous medical literature.

---

## 🛠️ High-Performance Architecture

The system is powered by a **Non-Linear StateGraph** architecture, optimizing for both speed and clinical rigor through conditional triage and parallel processing.

### 🔀 Intelligent Workflow Orchestration
- **Clinical Triage Router**: Dynamically stratifies patients. High-risk profiles trigger an intensive clinical research track, while low-risk profiles are triaged to a rapid "Wellness Optimization" path.
- **Parallel Treatment Tracks**: Executes Medication Prescriptions and Lifestyle Advice generation simultaneously, significantly reducing latency and mirroring specialized clinical workflows.
- **Drug Safety Guardrails**: An automated validation layer integrating **OpenFDA** and **RxClass** APIs to check for boxed warnings, drug-drug interactions, and patient-specific contraindications (e.g., Metformin in hypoglycemia).

### 📚 Advanced RAG (Retrieval-Augmented Generation)
- **Multi-Source Fetching**: Integrates **PubMed (E-Utilities)** for academic literature and **TavilySearch** for real-time clinical guidelines.
- **Corrective RAG (C-RAG)**: Implements a relevance-based filtering mechanism that automatically falls back to web retrieval if PubMed results are deemed ambiguous or insufficient.

---

## 🏗️ Core Components

### 1. Risk Stratification Node (`early_disease_detection`)
Leverages **GPT-4o-mini** with structured outputs to identify potential risks across:
- **Chronic**: Diabetes, Cardiovascular, Hypertension, Metabolic.
- **Oncology**: Hematological markers (WBC, Platelets) and constitutional symptoms.
- **Infectious**: Acute markers (Temp, SpO2, Resp Rate) for COVID-19 and viral/bacterial screening.

### 2. Clinical Evidence Engine (`fetch_medical_literature`)
A state-of-the-art literature synthesis pipeline:
- **Vector Store**: FAISS-based similarity search on top of PubMed abstracts.
- **Refinement**: Sentence-level decomposition and clinical summarization.

### 3. Safety-First Prescription (`drug_safety_guardrails`)
A rule-based and API-driven safety layer:
- **RxNorm Interaction**: Detects dangerous combinations (e.g., Anticoagulants + NSAIDs).
- **Contraindications**: Validates meds against patient allergies and biometric thresholds (BP/Sugar).

---

## 💻 Streamlit Interface
The application provides a premium, user-friendly interface for clinicians:
- **Interactive Forms**: Captures comprehensive biometrics, including acute clinical markers (Temp, WBC, SpO2).
- **🗺️ Clinical Road Map**: A high-level, paragraph-style narrative that synthesizes the entire strategy.
- **📊 Real-time Risk Panels**: Visual breakdown of disease risks and prioritized clinical flags.
- **⚠️ Urgent Alerts**: Tiered escalation alerts (LOW to CRITICAL) with clear recommended actions.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- API Keys: OpenAI, Tavily, OpenFDA

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Healthcare AI Clinical Support System"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure Secrets:
   Create `.streamlit/secrets.toml` or set environment variables:
   ```toml
   [secrets]
   OPENAI_API_KEY = "your_key"
   TAVILY_API_KEY = "your_key"
   OPENFDA_API_KEY = "your_key"
   ```

### Running the Application
```bash
streamlit run app.py
```

---

## 🔬 Scalability & Standards
- **Modular Design**: Every functional block is a LangGraph node, allowing for easy integration of new markers or APIs (e.g., Epic/FHIR).
- **Pydantic Validation**: Uses strict schema validation throughout the workflow to ensure clinical data integrity.
- **Medically Cautious**: Designed as a *support* system; it generates professional summaries while strictly avoiding diagnosis or unauthorized prescriptions.

---
*Developed as a State-of-the-art Clinical Decision Support Tool.*
