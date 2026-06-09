"""agents/semantic_processing_agent.py
Semantic Processing Agent (SPA)
- Tool 1: KG Construction (CheXzero + N2K mapper)
- Tool 2: Prompt Construction
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.kg_construction import KGConstructionTool
from tools.prompt_construction import PromptConstructionTool, build_finetuning_dataset
from utils.helpers import get_logger, load_config

logger = get_logger("SPA")


class SemanticProcessingAgent:
    """
    SPA: Maps multimodal patient clinical details to a Knowledge Graph
    and generates prompts for LLM fine-tuning and inference.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or load_config()
        self.kg_tool = KGConstructionTool(config=self.cfg)
        self.prompt_tool = PromptConstructionTool()
        self._patient_store: dict[str, dict] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def process(self, patient_record: dict) -> dict:
        """
        Full SPA pipeline for a single patient.
        1. Build KG from multimodal data
        2. Build LLM-compatible prompt from KG
        Returns: { patient_id, triples, prompt, kg_json }
        """
        pid = patient_record["patient_id"]
        logger.info(f"[SPA] Processing patient {pid}")

        # Tool 1: KG Construction
        kg_result = self.kg_tool.build(patient_record)
        triples = kg_result["triples"]

        # Store patient
        self._patient_store[pid] = patient_record

        # Tool 2: Prompt Construction
        prompt = self.prompt_tool.build_prediction_prompt(patient_record, triples)

        # KG as JSON for visualization
        kg_json = self.kg_tool.get_graph_json(pid)

        logger.info(f"[SPA] Done — {len(triples)} triples, prompt ready")

        return {
            "patient_id": pid,
            "triples": triples,
            "prompt": prompt,
            "kg_json": kg_json,
        }

    def get_prompt(self, patient_id: str) -> dict:
        """DA calls this to retrieve the prompt."""
        return self.prompt_tool.get_prompt(patient_id)

    def get_patient(self, patient_id: str) -> dict:
        return self._patient_store.get(patient_id, {})

    def get_kg_json(self, patient_id: str) -> dict:
        return self.kg_tool.get_graph_json(patient_id)

    def build_finetuning_dataset(self, records: list[dict]) -> list[dict]:
        """Build fine-tuning dataset from list of patient records."""
        logger.info(f"[SPA] Building fine-tuning dataset from {len(records)} records")
        # First process all records to build KGs
        kg_map = {}
        for record in records:
            result = self.process(record)
            kg_map[record["patient_id"]] = result["triples"]
        return build_finetuning_dataset(records, kg_map)

    def augment_kg(self, patient_id: str, new_data: dict) -> dict:
        """Augment existing patient KG with new information."""
        return self.kg_tool.augment(patient_id, new_data)
