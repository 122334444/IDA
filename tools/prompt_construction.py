"""tools/prompt_construction.py
Prompt Construction Tool — Part of Semantic Processing Agent (SPA)

Converts patient Knowledge Graph triples into structured LLM-compatible prompts.
Uses Prompt Engineering best practices for disease prediction.
"""

from typing import Optional
from utils.helpers import get_logger

logger = get_logger("prompt_construction")


# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert physician AI assistant specializing in disease diagnosis.
You will be provided with a patient's clinical data including:
- Demographics (age, gender)
- Vital signs (heart rate, blood pressure, temperature, oxygen saturation)
- Symptoms and chief complaints
- Radiological findings from chest X-ray analysis

Based on this clinical information, predict the most likely diagnosis and provide your confidence.
Respond ONLY with a valid JSON object in this exact format:
{
  "primary_diagnosis": "<disease name>",
  "confidence": <float between 0 and 1>,
  "differential_diagnoses": ["<disease1>", "<disease2>"],
  "clinical_reasoning": "<brief 1-2 sentence reasoning>"
}"""

PREDICTION_PROMPT_TEMPLATE = """## Patient Clinical Data

**Patient ID:** {patient_id}

### Demographics
- Age: {age} years
- Gender: {gender}

### Vital Signs
- Heart Rate: {heart_rate} bpm
- Blood Pressure: {bp_systolic}/{bp_diastolic} mmHg
- Temperature: {temperature}°C
- O2 Saturation: {o2_saturation}%
- Respiratory Rate: {respiratory_rate} breaths/min

### Chief Complaints & Symptoms
{symptoms_text}

### Radiological Findings (Chest X-ray)
{radiology_text}

### Clinical Narrative
{clinical_narrative}

### Knowledge Graph Triples
{kg_triples_text}

Based on the above patient data, provide your diagnosis prediction."""

EXPLANATION_PROMPT_TEMPLATE = """You are an expert physician explaining a diagnosis to a colleague.

**Patient ID:** {patient_id}

**Predicted Diagnosis:** {predicted_diagnosis}

**Patient Clinical Summary:**
- {age}-year-old {gender}
- Vitals: HR {heart_rate} bpm, BP {bp_systolic}/{bp_diastolic} mmHg, Temp {temperature}°C, O2 {o2_saturation}%
- Symptoms: {symptoms_text}
- Chest X-ray: {radiology_text}

**Relevant Medical Guidelines Context:**
{guidelines_context}

Please provide a clear, evidence-based explanation for this diagnosis referencing the medical guidelines above.
Include:
1. Why this diagnosis fits the clinical picture
2. Key supporting findings
3. What the medical guidelines say about this condition
4. Recommended next steps / management

Explain in physician-level language suitable for clinical documentation."""


class PromptConstructionTool:
    """
    Constructs LLM-compatible prompts from patient KG data.
    Stored in a prompt store keyed by patient_id for retrieval by DA.
    """

    def __init__(self):
        self._prompt_store: dict[str, dict] = {}

    def build_prediction_prompt(
        self,
        patient_record: dict,
        kg_triples: list[tuple],
    ) -> dict:
        """
        Generate a prediction prompt from patient record + KG triples.

        Returns:
            {
                patient_id: str,
                system: str,
                user: str,
                messages: [{"role": ..., "content": ...}]
            }
        """
        pid = patient_record["patient_id"]

        # Format symptoms
        symptoms = patient_record.get("symptoms", [])
        symptoms_text = "\n".join(f"- {s}" for s in symptoms) if symptoms else "- Not reported"

        # Extract radiological findings from triples
        rad_findings = [
            o for s, p, o in kg_triples
            if p == "has_radiological_finding"
        ]
        radiology_text = (
            "\n".join(f"- {f}" for f in rad_findings)
            if rad_findings else "- No radiological findings recorded"
        )

        # Format KG triples (first 20 to keep prompt compact)
        kg_text = "\n".join(
            f"({s}, {p}, {o})" for s, p, o in kg_triples[:20]
        )

        user_content = PREDICTION_PROMPT_TEMPLATE.format(
            patient_id=pid,
            age=patient_record.get("age", "unknown"),
            gender="Male" if patient_record.get("gender") == "M" else "Female",
            heart_rate=patient_record.get("heart_rate", "?"),
            bp_systolic=patient_record.get("bp_systolic", "?"),
            bp_diastolic=patient_record.get("bp_diastolic", "?"),
            temperature=patient_record.get("temperature", "?"),
            o2_saturation=patient_record.get("o2_saturation", "?"),
            respiratory_rate=patient_record.get("respiratory_rate", "?"),
            symptoms_text=symptoms_text,
            radiology_text=radiology_text,
            clinical_narrative=patient_record.get("clinical_narrative", ""),
            kg_triples_text=kg_text,
        )

        prompt = {
            "patient_id": pid,
            "system": SYSTEM_PROMPT,
            "user": user_content,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        # Store for retrieval by DA
        self._prompt_store[pid] = prompt
        logger.info(f"Prediction prompt built for patient {pid} ({len(user_content)} chars)")
        return prompt

    def build_explanation_prompt(
        self,
        patient_record: dict,
        predicted_diagnosis: str,
        guidelines_context: str,
    ) -> dict:
        """Build explanation prompt for EA."""
        pid = patient_record["patient_id"]
        symptoms = patient_record.get("symptoms", [])
        rad = patient_record.get("disease_labels", [])

        user_content = EXPLANATION_PROMPT_TEMPLATE.format(
            patient_id=pid,
            predicted_diagnosis=predicted_diagnosis,
            age=patient_record.get("age", "?"),
            gender="Male" if patient_record.get("gender") == "M" else "Female",
            heart_rate=patient_record.get("heart_rate", "?"),
            bp_systolic=patient_record.get("bp_systolic", "?"),
            bp_diastolic=patient_record.get("bp_diastolic", "?"),
            temperature=patient_record.get("temperature", "?"),
            o2_saturation=patient_record.get("o2_saturation", "?"),
            symptoms_text=", ".join(symptoms) if symptoms else "not specified",
            radiology_text=", ".join(rad) if rad else "not available",
            guidelines_context=guidelines_context,
        )

        prompt = {
            "patient_id": pid,
            "system": "You are an expert physician providing clinical explanations based on medical guidelines.",
            "user": user_content,
            "messages": [
                {"role": "system", "content": "You are an expert physician providing clinical explanations based on medical guidelines."},
                {"role": "user", "content": user_content},
            ],
        }
        return prompt

    def get_prompt(self, patient_id: str) -> Optional[dict]:
        """Retrieve stored prompt by patient ID (used by DA)."""
        return self._prompt_store.get(patient_id)

    def list_patients(self) -> list[str]:
        return list(self._prompt_store.keys())


# ─── Fine-tuning Dataset Builder ─────────────────────────────────────────────

def build_finetuning_dataset(
    records: list[dict],
    kg_map: dict[str, list[tuple]],
) -> list[dict]:
    """
    Build instruction-tuning dataset from patient records + KGs.
    Format: [{"text": "<s>[INST] ... [/INST] {answer}</s>"}]
    Compatible with LLaMA-3 chat template.
    """
    tool = PromptConstructionTool()
    dataset = []

    for record in records:
        pid = record["patient_id"]
        triples = kg_map.get(pid, [])
        prompt = tool.build_prediction_prompt(record, triples)

        # Ground truth answer
        disease = record.get("primary_diagnosis", "Unknown")
        answer_json = (
            f'{{"primary_diagnosis": "{disease}", '
            f'"confidence": 0.85, '
            f'"differential_diagnoses": [], '
            f'"clinical_reasoning": "Based on clinical presentation and radiological findings."}}'
        )

        # LLaMA-3 chat format
        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{prompt['system']}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt['user']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{answer_json}<|eot_id|>"
        )
        dataset.append({"text": text, "patient_id": pid, "label": disease})

    logger.info(f"Fine-tuning dataset built: {len(dataset)} examples")
    return dataset
