"""agents/diagnosis_agent.py
Diagnosis Agent (DA)
- Tool 1: Quantisation & Fine-tuning (QLoRA / LLaMA-3)
- Tool 2: Prediction
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import get_logger, load_config

logger = get_logger("DA")


class QuantisationFineTuningTool:
    """
    Q&F Tool: Quantises LLaMA-3 with 4-bit QLoRA and fine-tunes
    on patient prompts from SPA.
    """

    def __init__(self, cfg: dict):#
        self.cfg = cfg
        self.llm_cfg = cfg["llm"]
        self.model_output = cfg["paths"]["fine_tuned_model"]

    def run(self, dataset: list[dict]):
        """
        Fine-tune LLaMA-3 with QLoRA.
        Args:
            dataset: list of {"text": "<formatted training text>"}
        """
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TrainingArguments,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from trl import SFTTrainer
            from datasets import Dataset
        except ImportError as e:
            logger.error(f"Missing dependency for fine-tuning: {e}")
            logger.info("Install: pip install transformers peft trl bitsandbytes datasets")
            return

        base_model = self.llm_cfg["base_model"]
        q_cfg = self.llm_cfg["quantisation"]
        lora_cfg = self.llm_cfg["lora"]
        train_cfg = self.llm_cfg["training"]

        logger.info(f"[DA/Q&F] Loading base model: {base_model}")

        # 4-bit quantisation config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q_cfg["load_in_4bit"],
            bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, q_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # LoRA config
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )
        model = get_peft_model(model, peft_config)#Atach lora adopters 
        model.print_trainable_parameters()

        # Dataset -> python data into hugging face dataset
        hf_dataset = Dataset.from_list(dataset)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_output,
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            warmup_ratio=train_cfg["warmup_ratio"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            fp16=train_cfg["fp16"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=hf_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=train_cfg["max_token_length"],
            tokenizer=tokenizer,
            args=training_args,
        )

        logger.info("[DA/Q&F] Starting fine-tuning ...")
        trainer.train()
        trainer.save_model(self.model_output)
        tokenizer.save_pretrained(self.model_output)
        logger.info(f"[DA/Q&F] Fine-tuned model saved to {self.model_output}")


class PredictionTool:
    """
    Prediction Tool: Uses fine-tuned (or base) LLaMA-3 to predict diseases.
    Falls back to a rule-based heuristic if model not available.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_path = cfg["paths"]["fine_tuned_model"]
        self.infer_cfg = cfg["llm"]["inference"]
        self._model = None
        self._tokenizer = None
        self._use_fallback = False

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_path = self.model_path
            if not Path(model_path).exists():
                # Try HuggingFace base model (requires auth for LLaMA)
                model_path = self.cfg["llm"]["base_model"]
                logger.warning(f"Fine-tuned model not found. Using base: {model_path}")

            logger.info(f"[DA/Predict] Loading model from {model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            self._model.eval()
            logger.info("[DA/Predict] Model loaded")
        except Exception as e:
            logger.warning(f"[DA/Predict] Cannot load LLM ({e}). Using rule-based fallback.")
            self._use_fallback = True

    def predict(self, prompt: dict, patient_record: dict = None) -> dict:
        """
        Run prediction for a patient.
        Args:
            prompt: dict with 'messages' list from PromptConstructionTool
            patient_record: raw patient data (used for fallback)
        Returns:
            { primary_diagnosis, confidence, differential_diagnoses, clinical_reasoning }
        """
        self._load_model()

        if self._use_fallback:
            return self._rule_based_predict(patient_record or {})

        messages = prompt.get("messages", [])
        if not messages:
            return self._rule_based_predict(patient_record or {})

        try:
            import torch
            from transformers import pipeline

            pipe = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )
            # Format for LLaMA-3
            full_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            output = pipe(
                full_prompt,
                max_new_tokens=self.infer_cfg["max_new_tokens"],
                temperature=self.infer_cfg["temperature"],
                do_sample=self.infer_cfg["do_sample"],
                return_full_text=False,
            )
            raw_text = output[0]["generated_text"].strip()
            return self._parse_json_response(raw_text)
        except Exception as e:
            logger.error(f"[DA/Predict] Inference error: {e}")
            return self._rule_based_predict(patient_record or {})

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response."""
        try:
            # Find JSON block
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {
            "primary_diagnosis": "Unable to parse",
            "confidence": 0.0,
            "differential_diagnoses": [],
            "clinical_reasoning": text[:200],
        }

    def _rule_based_predict(self, record: dict) -> dict:
        """
        Clinical rule-based fallback when LLM not available.
        Uses vital signs + symptoms + X-ray labels.
        """
        labels = record.get("disease_labels", [])
        symptoms = [s.lower() for s in record.get("symptoms", [])]
        o2 = record.get("o2_saturation", 97)
        hr = record.get("heart_rate", 80)
        temp = record.get("temperature", 36.8)

        # CHF indicators
        chf_score = 0
        if "Cardiomegaly" in labels: chf_score += 3
        if "Edema" in labels: chf_score += 2
        if "Effusion" in labels: chf_score += 2
        if any("dyspnea" in s or "edema" in s or "orthopnea" in s for s in symptoms): chf_score += 2
        if o2 < 92: chf_score += 1
        if hr > 100: chf_score += 1

        # Pneumonia / infection indicators
        pneu_score = 0
        if "Pneumonia" in labels: pneu_score += 3
        if "Consolidation" in labels: pneu_score += 2
        if "Infiltration" in labels: pneu_score += 1
        if any("cough" in s or "fever" in s or "chills" in s for s in symptoms): pneu_score += 2
        if temp > 38.0: pneu_score += 2

        # Other
        other_map = {
            "Atelectasis": 1 if "Atelectasis" in labels else 0,
            "Pneumothorax": 3 if "Pneumothorax" in labels else 0,
            "Cardiomegaly": chf_score,
        }

        scores = {
            "Congestive Heart Failure": chf_score,
            "Pneumonia": pneu_score,
        }
        scores.update({k: v for k, v in other_map.items() if v > 0})

        if not scores or max(scores.values()) == 0:
            return {
                "primary_diagnosis": "No Finding",
                "confidence": 0.80,
                "differential_diagnoses": [],
                "clinical_reasoning": "No significant clinical indicators detected.",
            }

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        conf = round(scores[best] / total, 2) if total > 0 else 0.5
        differentials = [k for k in scores if k != best and scores[k] > 0]

        return {
            "primary_diagnosis": best,
            "confidence": conf,
            "differential_diagnoses": differentials,
            "clinical_reasoning": (
                f"Rule-based prediction based on X-ray findings {labels}, "
                f"symptoms, and vital signs (HR={hr}, O2={o2}%, Temp={temp}°C)."
            ),
        }


class DiagnosisAgent:
    """
    DA: Predicts diseases by fine-tuning and inference with LLaMA-3.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or load_config()
        self.qf_tool = QuantisationFineTuningTool(self.cfg)
        self.predict_tool = PredictionTool(self.cfg)
        self._new_patient_count = 0
        self._threshold = self.cfg.get("finetune_threshold", 50)

    def finetune(self, dataset: list[dict]):
        """Trigger fine-tuning with the given dataset."""
        logger.info(f"[DA] Starting fine-tuning with {len(dataset)} examples")
        self.qf_tool.run(dataset)

    def predict(self, patient_id: str, spa_agent) -> dict:
        """
        Predict disease for a patient using their stored prompt.
        Args:
            patient_id: patient identifier
            spa_agent: SemanticProcessingAgent instance (to retrieve prompt + record)
        """
        prompt = spa_agent.get_prompt(patient_id)
        record = spa_agent.get_patient(patient_id)

        if not prompt:
            logger.error(f"[DA] No prompt found for patient {patient_id}")
            return {"error": f"No prompt for patient {patient_id}"}

        logger.info(f"[DA] Running prediction for patient {patient_id}")
        result = self.predict_tool.predict(prompt, record)
        result["patient_id"] = patient_id

        # Track new patients for threshold-based retraining
        self._new_patient_count += 1
        if self._new_patient_count >= self._threshold:
            logger.info(f"[DA] Threshold reached ({self._threshold}). Retraining scheduled.")
            self._new_patient_count = 0

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["finetune", "predict"], default="predict")
    parser.add_argument("--patient_id", default=None)
    args = parser.parse_args()

    cfg = load_config()
    da = DiagnosisAgent(cfg)

    if args.mode == "finetune":
        import pandas as pd
        from agents.semantic_processing_agent import SemanticProcessingAgent

        spa = SemanticProcessingAgent(cfg)
        train_df = pd.read_parquet(cfg["paths"]["data_processed"] + "/train.parquet")
        records = train_df.to_dict("records")
        dataset = spa.build_finetuning_dataset(records[:100])  # start small
        da.finetune(dataset)
