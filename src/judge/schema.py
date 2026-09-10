from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.data.schema import SCHEMA_VERSION


Surface = Literal["answer", "trace", "explanation"]


class JudgeOutput(BaseModel):


    rationale: str
    semantic_leak: bool
    leak_type: Optional[Literal["direct", "inadvertent"]] = None  
    meta_reference: bool
    fluent: bool


    guessing: bool


def judge_json_schema() -> dict:

    return JudgeOutput.model_json_schema()


class JudgmentRow(BaseModel):


    schema_version: str = SCHEMA_VERSION
    judge_model: str
    judge_config_hash: str
    git_commit: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    generation_id: str
    surface: Surface
    raw_judge_json: str
    parsed: JudgeOutput
