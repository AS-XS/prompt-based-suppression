from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


ConceptClass = Literal["isolated", "general"]


ConceptSubtype = Literal["entity", "subject", "hazard"]
ProbeType = Literal["forget", "neighbor", "retain", "utility", "dependent"]
Condition = Literal["baseline", "suppressed"]
ThinkState = Literal["off", "on", "na"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_item_id(concept_id: str, probe_type: str, content: str) -> str:


    import hashlib

    h = hashlib.sha256(f"{concept_id}|{probe_type}|{content}".encode("utf-8")).hexdigest()
    return f"{concept_id}-{probe_type}-{h[:16]}"


class Item(BaseModel):


    item_id: str
    concept_id: str
    concept_class: ConceptClass
    concept_subtype: ConceptSubtype
    probe_type: ProbeType
    question: str
    reference: Optional[str] = None
    answer_key: Optional[str] = None
    source: str = ""  
    meta: dict[str, Any] = Field(default_factory=dict)


class Sampling(BaseModel):


    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 1024
    seed: int | None = None


    repetition_penalty: float | None = None


class GenerationRow(BaseModel):


    schema_version: str = SCHEMA_VERSION


    git_commit: str
    config_hash: str
    config_id: str
    created_at: str = Field(default_factory=_utcnow)


    model: str


    prompt_id: str
    condition: Condition
    think_state: ThinkState
    sampling: Sampling


    item_id: str
    concept_id: str
    concept_class: ConceptClass
    concept_subtype: ConceptSubtype
    probe_type: ProbeType


    raw_output: str
    raw_reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


    resolved_sampling: Optional[dict] = None

    @property
    def generation_id(self) -> str:

        return f"{self.config_hash}:{self.item_id}:{self.condition}:{self.think_state}"
