"""agents/explanation_agent.py
Explanation Agent (EA) - RAG over medical guidelines using TF-IDF (offline).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import get_logger, load_config

logger = get_logger("EA")

# ─── Built-in Medical Guidelines ─────────────────────────────────────────────
BUILTIN_GUIDELINES = {
    "Congestive Heart Failure": """
ACC/AHA Guidelines for Heart Failure. CHF is the heart's inability to pump sufficient blood.
Diagnostic criteria: dyspnea, orthopnea, paroxysmal nocturnal dyspnea, peripheral edema, S3 gallop.
Chest X-ray: cardiomegaly (CTR > 0.5), pulmonary vascular congestion, Kerley B lines, pleural effusion.
BNP/NT-proBNP elevation is key biomarker. Echocardiography is gold standard.
Management: ACE inhibitors/ARBs, beta-blockers, diuretics (furosemide), aldosterone antagonists.
Ejection fraction < 40% = HFrEF; >= 50% = HFpEF. NYHA class guides severity.
Restrict fluid/sodium. Daily weight monitoring. Cardiac rehabilitation recommended.
""",
    "Pneumonia": """
ATS/IDSA Guidelines for Community-Acquired Pneumonia (CAP).
Diagnosis: new pulmonary infiltrate + fever, cough with sputum, pleuritic chest pain, or dyspnea.
Chest X-ray: lobar or segmental consolidation, air bronchograms, ground-glass opacity.
CURB-65 score guides admission. Score >= 2: consider hospitalization.
Empiric outpatient: amoxicillin or doxycycline for typical CAP.
Inpatient: beta-lactam + macrolide or respiratory fluoroquinolone.
Severe CAP (ICU): antipseudomonal beta-lactam + azithromycin or fluoroquinolone.
O2 saturation < 92% requires supplemental oxygen. Blood cultures for severe CAP.
""",
    "Cardiomegaly": """
Cardiomegaly: cardiothoracic ratio > 0.5 on PA chest X-ray.
Causes: dilated cardiomyopathy, hypertensive heart disease, valvular disease, pericardial effusion.
Echocardiography determines underlying etiology and LV function.
Associated findings: pulmonary vascular congestion, pleural effusion, Kerley B lines.
Management depends on cause. Regular cardiology follow-up required.
Evaluate for coronary artery disease, valve pathology, and systolic/diastolic dysfunction.
""",
    "Effusion": """
Pleural effusion: fluid in pleural space. Chest X-ray: blunted costophrenic angle (>200mL), meniscus sign.
Large effusion may cause mediastinal shift away from affected side.
Light's criteria distinguish transudate from exudate (LDH, protein ratios).
Transudates: CHF, cirrhosis, nephrotic syndrome. Exudates: infection, malignancy, PE.
Therapeutic thoracentesis for symptomatic relief. Diagnostic tap for undiagnosed effusions.
Send fluid for cell count, protein, LDH, glucose, culture, cytology.
""",
    "Atelectasis": """
Atelectasis: partial or complete lung collapse. Chest X-ray: linear/plate atelectasis, volume loss.
Tracheal/mediastinal deviation toward affected side. Elevated hemidiaphragm.
Causes: mucus plugging, compression, absorption atelectasis, contraction.
Management: incentive spirometry, chest physiotherapy, early ambulation.
Bronchoscopy for mucus plugging. Treat underlying cause (pain control post-surgery).
""",
    "Pneumothorax": """
Pneumothorax: air in pleural space. Chest X-ray: absent lung markings, visible pleural line.
Tension pneumothorax emergency: tracheal deviation away, hemodynamic compromise.
Tension PTX: immediate needle decompression then chest tube.
Primary spontaneous < 2cm: observation or aspiration. Secondary/large: chest tube.
High-flow O2 accelerates reabsorption. Avoid air travel for 6 weeks post-resolution.
""",
    "Edema": """
Pulmonary edema: fluid in alveoli. Cardiogenic (elevated PCWP) vs non-cardiogenic (ARDS).
Chest X-ray: bilateral perihilar infiltrates (bat-wing), Kerley B lines, pleural effusion.
Cardiogenic: diuresis (furosemide IV), preload/afterload reduction, treat underlying CHF.
ARDS: lung-protective ventilation 6 mL/kg IBW, PEEP titration, prone positioning if P/F < 150.
Monitor urine output and daily weight during diuresis.
""",
    "Consolidation": """
Pulmonary consolidation: airspace filling. Chest X-ray: homogeneous opacity, air bronchograms.
Most common cause: bacterial pneumonia (Strep pneumoniae).
Other causes: pulmonary hemorrhage, cardiogenic edema, malignancy (BAC/adenocarcinoma).
Assess with sputum Gram stain and culture, blood cultures, procalcitonin, CRP.
Antibiotic therapy guided by severity and likely organism.
""",
    "No Finding": """
Normal chest radiograph. No acute cardiopulmonary process.
Normal cardiac silhouette, clear lung fields bilaterally.
No pleural effusion, no pneumothorax, no consolidation, no vascular congestion.
Clinical correlation recommended if symptoms persist despite normal imaging.
CT chest for further evaluation if clinical suspicion remains high.
""",
    "Infiltration": """
Pulmonary infiltrate: nonspecific opacity on chest X-ray suggesting airspace or interstitial disease.
Differential: pneumonia, pulmonary edema, aspiration, hemorrhage, atelectasis.
Clinical context essential for interpretation. Correlate with symptoms, fever, WBC.
Serial radiographs may help differentiate. CT chest provides better characterization.
""",
}


class ExplanationGenerationTool:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._docs = []
        self._metas = []
        self._vectorizer = None
        self._tfidf_matrix = None
        self._initialized = False
        self._top_n = cfg.get("rag", {}).get("top_n_results", 3)

    def _chunk_text(self, text: str, size: int = 400) -> list[str]:
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        chunks, current = [], []
        for s in sentences:
            current.append(s)
            if sum(len(x) for x in current) > size:
                chunks.append('. '.join(current))
                current = []
        if current:
            chunks.append('. '.join(current))
        return chunks or [text]

    def _init_rag(self):
        if self._initialized:
            return
        logger.info("[EA] Building TF-IDF index over guidelines ...")
        from sklearn.feature_extraction.text import TfidfVectorizer

        for disease, text in BUILTIN_GUIDELINES.items():
            for i, chunk in enumerate(self._chunk_text(text)):
                self._docs.append(chunk)
                self._metas.append({"disease": disease})

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
        self._tfidf_matrix = self._vectorizer.fit_transform(self._docs)
        self._initialized = True
        logger.info(f"[EA] Index ready: {len(self._docs)} chunks")

    def retrieve(self, query: str, disease: str = None, top_n: int = None) -> list[str]:
        self._init_rag()
        from sklearn.metrics.pairwise import cosine_similarity

        top_n = top_n or self._top_n
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]

        indexed = [(i, scores[i]) for i in range(len(scores))
                   if disease is None or self._metas[i]["disease"] == disease]
        if not indexed:
            indexed = [(i, scores[i]) for i in range(len(scores))]

        indexed.sort(key=lambda x: x[1], reverse=True)

        selected, seen_tokens = [], set()
        for idx, _ in indexed[:top_n * 4]:
            doc = self._docs[idx]
            tokens = set(doc.lower().split())
            if len(tokens & seen_tokens) / max(len(tokens), 1) < 0.6:
                selected.append(doc)
                seen_tokens.update(tokens)
            if len(selected) >= top_n:
                break
        return selected

    def generate_explanation(self, patient_record: dict, diagnosis: str, query: str = None) -> str:
        query = query or f"diagnosis treatment {diagnosis}"
        chunks = self.retrieve(query, disease=diagnosis)
        if not chunks:
            chunks = self.retrieve(query)
        guidelines_context = "\n".join(chunks)
        return self._build_explanation(patient_record, diagnosis, guidelines_context)

    def _build_explanation(self, record: dict, diagnosis: str, context: str) -> str:
        symptoms = record.get("symptoms", [])
        labels = record.get("disease_labels", [])
        age = record.get("age", "?")
        gender = "male" if record.get("gender") == "M" else "female"
        hr = record.get("heart_rate", "?")
        o2 = record.get("o2_saturation", "?")
        temp = record.get("temperature", "?")
        bp_s = record.get("bp_systolic", "?")
        bp_d = record.get("bp_diastolic", "?")
        rr = record.get("respiratory_rate", "?")

        return f"""## Clinical Explanation: {diagnosis}

### Patient Summary
{age}-year-old {gender} presenting with {', '.join(symptoms) if symptoms else 'presenting complaints as documented'}.

### Key Clinical Findings Supporting This Diagnosis
- **Vital Signs:** HR {hr} bpm | BP {bp_s}/{bp_d} mmHg | Temp {temp}°C | O2 {o2}% | RR {rr}/min
- **Radiological Findings:** {', '.join(labels) if labels else 'as documented'}
- **Presenting Symptoms:** {', '.join(symptoms) if symptoms else 'as documented'}

### Evidence-Based Guidelines
{context}

### Clinical Reasoning
The combination of this patient's clinical presentation, vital signs, and radiological findings is consistent with **{diagnosis}**.
{"Low O2 saturation suggests significant cardiopulmonary compromise requiring immediate evaluation." if isinstance(o2, int) and o2 < 92 else ""}
{"Elevated heart rate may reflect compensatory mechanism or underlying pathology." if isinstance(hr, int) and hr > 100 else ""}
{"Elevated temperature raises concern for infectious etiology." if isinstance(temp, float) and temp > 38.0 else ""}

### Recommended Next Steps
1. Confirm diagnosis with appropriate laboratory and imaging studies
2. Initiate guideline-directed medical therapy per the evidence above
3. Arrange specialist consultation as appropriate
4. Monitor clinical response and adjust management accordingly
5. Patient/family education regarding diagnosis, prognosis, and management

*This explanation is AI-generated from evidence-based guidelines and must be reviewed by a qualified physician.*"""


class ExplanationAgent:
    def __init__(self, config: dict = None):
        self.cfg = config or load_config()
        self.tool = ExplanationGenerationTool(self.cfg)

    def explain(self, patient_record: dict, diagnosis_result: dict) -> dict:
        pid = patient_record.get("patient_id", "unknown")
        diagnosis = diagnosis_result.get("primary_diagnosis", "Unknown")
        confidence = diagnosis_result.get("confidence", 0.0)
        logger.info(f"[EA] Generating explanation for {pid} — {diagnosis}")
        explanation = self.tool.generate_explanation(patient_record, diagnosis)
        return {
            "patient_id": pid,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "explanation": explanation,
            "differential_diagnoses": diagnosis_result.get("differential_diagnoses", []),
        }
