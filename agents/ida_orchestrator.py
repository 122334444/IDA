"""agents/ida_orchestrator.py
Intelligent Doctor Assistant (IDA) — LangGraph Orchestrator

Routes user queries to appropriate agents:
  - "generate knowledge graph" → SPA
  - "diagnosis prediction" → SPA → DA
  - "explanation with diagnosis" → SPA → DA → EA
"""

import sys
import json
from pathlib import Path
from typing import TypedDict, Annotated, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.semantic_processing_agent import SemanticProcessingAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.explanation_agent import ExplanationAgent
from utils.helpers import get_logger, load_config

logger = get_logger("IDA")


# ─── State Schema ─────────────────────────────────────────────────────────────

class IDAState(TypedDict):
    patient_id: str
    patient_record: dict
    action: str                      # "kg_only" | "predict" | "explain"
    kg_result: dict
    diagnosis_result: dict
    explanation_result: dict
    error: str
    done: bool


# ─── IDA Orchestrator ─────────────────────────────────────────────────────────

class IDAOrchestrator:
    """
    Intelligent Doctor Assistant — orchestrates all agents.
    Implements the routing logic from Fig. 1.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or load_config()
        logger.info("[IDA] Initialising agents ...")
        self.spa = SemanticProcessingAgent(self.cfg)
        self.da = DiagnosisAgent(self.cfg)
        self.ea = ExplanationAgent(self.cfg)
        logger.info("[IDA] All agents ready")

    # ── Node functions (LangGraph-style) ──────────────────────────────────────

    def node_spa(self, state: IDAState) -> IDAState:
        """SPA node: build KG + prompt."""
        try:
            result = self.spa.process(state["patient_record"])
            state["kg_result"] = {
                "triples": result["triples"],
                "kg_json": result["kg_json"],
                "triple_count": len(result["triples"]),
            }
            logger.info(f"[IDA→SPA] KG built: {len(result['triples'])} triples")
        except Exception as e:
            logger.error(f"[IDA→SPA] Error: {e}")
            state["error"] = str(e)
        return state

    def node_da(self, state: IDAState) -> IDAState:
        """DA node: predict diagnosis."""
        try:
            result = self.da.predict(state["patient_id"], self.spa)
            state["diagnosis_result"] = result
            logger.info(f"[IDA→DA] Prediction: {result.get('primary_diagnosis')}")
        except Exception as e:
            logger.error(f"[IDA→DA] Error: {e}")
            state["error"] = str(e)
        return state

    def node_ea(self, state: IDAState) -> IDAState:
        """EA node: generate explanation."""
        try:
            result = self.ea.explain(
                state["patient_record"],
                state["diagnosis_result"],
            )
            state["explanation_result"] = result
            logger.info(f"[IDA→EA] Explanation generated")
        except Exception as e:
            logger.error(f"[IDA→EA] Error: {e}")
            state["error"] = str(e)
        return state

    # ── Router ────────────────────────────────────────────────────────────────

    def _route(self, action: str) -> list[str]:
        """Determine which agents to call based on action."""
        routes = {
            "kg_only": ["spa"],
            "predict": ["spa", "da"],
            "explain": ["spa", "da", "ea"],
        }
        return routes.get(action, ["spa"])

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, patient_record: dict, action: str = "explain") -> dict:
        """
        Run the full IDA pipeline.

        Args:
            patient_record: dict with patient clinical data
            action: "kg_only" | "predict" | "explain"

        Returns:
            Full state dict with all results
        """
        pid = patient_record.get("patient_id", "unknown")
        logger.info(f"[IDA] Processing patient={pid}, action={action}")

        state: IDAState = {
            "patient_id": pid,
            "patient_record": patient_record,
            "action": action,
            "kg_result": {},
            "diagnosis_result": {},
            "explanation_result": {},
            "error": "",
            "done": False,
        }

        nodes = self._route(action)

        for node in nodes:
            if state.get("error"):
                break
            if node == "spa":
                state = self.node_spa(state)
            elif node == "da":
                state = self.node_da(state)
            elif node == "ea":
                state = self.node_ea(state)

        state["done"] = True

        # Build clean response
        response = self._build_response(state)
        return response

    def _build_response(self, state: IDAState) -> dict:
        """Build final structured response."""
        action = state["action"]
        pid = state["patient_id"]

        response = {
            "patient_id": pid,
            "action": action,
            "status": "error" if state["error"] else "success",
        }

        if state["error"]:
            response["error"] = state["error"]
            return response

        if action in ("kg_only", "predict", "explain"):
            response["knowledge_graph"] = state["kg_result"]

        if action in ("predict", "explain"):
            diag = state["diagnosis_result"]
            response["diagnosis"] = {
                "primary": diag.get("primary_diagnosis", "Unknown"),
                "confidence": diag.get("confidence", 0.0),
                "differentials": diag.get("differential_diagnoses", []),
                "reasoning": diag.get("clinical_reasoning", ""),
            }

        if action == "explain":
            exp = state["explanation_result"]
            response["explanation"] = exp.get("explanation", "")

        return response
