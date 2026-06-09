# 🏥 Intelligent Doctor Assistant (IDA)

An Agentic AI framework for disease diagnosis with explanations, built using LangGraph, Flask, and LLaMA-3.

## Architecture

```
User (Medical Practitioner)
        ↓
Intelligent Doctor Assistant (IDA) — LangGraph Orchestrator
        ├── Semantic Processing Agent (SPA)
        │     ├── Tool: KG Construction (CheXzero + N2K mapper)
        │     └── Tool: Prompt Construction
        ├── Diagnosis Agent (DA)
        │     ├── Tool: Quantisation & Fine-tuning (QLoRA/LLaMA-3)
        │     └── Tool: Prediction
        └── Explanation Agent (EA)
              └── Tool: Explanation Generation (RAG)
```

## Dataset
- **NIH Chest X-Ray Dataset** (14 diseases, 112,120 images)
- Download: https://www.kaggle.com/datasets/nih-chest-xrays/data

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NIH Chest X-Ray Data
```bash
# Set up Kaggle API credentials first
kaggle datasets download -d nih-chest-xrays/data -p data/raw/
unzip data/raw/data.zip -d data/raw/nih_xray/
```

### 3. Configure Environment
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your API keys and paths
```

### 4. Run Data Preprocessing
```bash
python utils/preprocess.py
```

### 5. Fine-tune LLaMA-3 (Diagnosis Agent)
```bash
python agents/diagnosis_agent.py --mode finetune
```

### 6. Launch Application
```bash
python app.py
```

Open: http://localhost:5000

## Supported Diseases (NIH Dataset)
- Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass
- Nodule, Pneumonia, Pneumothorax, Consolidation, Edema
- Emphysema, Fibrosis, Pleural Thickening, Hernia

## Performance (LLaMA-3, Token=512, Epoch=2)
| Metric    | CHF   | UTI   |
|-----------|-------|-------|
| AUROC     | 0.88  | 0.90  |
| AUPRC     | 0.91  | 0.83  |
| F1-Score  | 0.83  | 0.83  |
# IDA
# IDA
