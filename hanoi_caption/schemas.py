"""Pydantic schemas exchanged across pipeline modules."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class KBNode(BaseModel):
    id: str
    name_en: str
    name_vi: str
    type: Literal["object", "category"]
    parent_id: str | None = None
    description_en: str
    description_vi: str
    visual_cues_en: str
    visual_cues_vi: str
    tags: list[str] = Field(default_factory=list)


class MatchCandidate(BaseModel):
    node_id: str
    score: float


class MatchResult(BaseModel):
    node_id: str | None
    confidence: float
    top_k: list[MatchCandidate]


class Region(BaseModel):
    box: tuple[float, float, float, float]  # xyxy in pixel coords
    mask_png_b64: str                        # PNG-encoded binary mask
    query: str                               # detection query that produced this region
    score: float                             # detector score


class RegionDescription(BaseModel):
    query: str
    text: str


class CaptionResult(BaseModel):
    caption: str | None
    refusal: str | None
    debug: dict[str, Any] = Field(default_factory=dict)
