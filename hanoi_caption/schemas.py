"""Pydantic schemas exchanged across pipeline modules."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class KBNode(BaseModel):
    id: str
    kb_id: str | None = None
    name_en: str
    name_vi: str
    type: Literal["object", "category", "action"]
    parent_id: str | None = None
    description_en: str
    description_vi: str
    visual_cues_en: str
    visual_cues_vi: str
    tags: list[str] = Field(default_factory=list)


class VideoSegment(BaseModel):
    start_s: float
    end_s: float
    kb_id: str
    node_id: str
    name_en: str
    confidence: float
    caption: str
    debug: dict[str, Any] = Field(default_factory=dict)
