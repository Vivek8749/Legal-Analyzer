"""Structured-output schemas that Claude fills in (sent as ``output_format``).

The API constrains decoding to these JSON schemas, so responses always parse. Numeric and
length constraints are not supported by structured outputs, so validation beyond types
(for example, that chunk IDs exist) happens in the analyzer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .taxonomy import CATEGORY_IDS

CategoryId = Literal[CATEGORY_IDS]  # type: ignore[valid-type]

IssueType = Literal[
    "one_sided_term",
    "loophole",
    "waived_right",
    "ambiguity",
    "hidden_cost_or_obligation",
    "missing_protection",
]
Favors = Literal["drafter", "reviewed_party", "balanced", "unclear"]
Confidence = Literal["low", "medium", "high"]


class LLMFinding(BaseModel):
    category_id: CategoryId
    title: str = Field(description="Short headline in plain English, e.g. 'They can cancel your account without warning'.")
    chunk_ids: list[int] = Field(description="Numeric IDs of the excerpts this finding relies on (C12 -> 12), quote's excerpt first.")
    quote: str = Field(description="Verbatim passage copied from one excerpt; the shortest span that shows the problem.")
    issue_type: IssueType
    severity: Literal["low", "medium", "high", "critical"]
    favors: Favors = Field(description="Which party the term benefits.")
    plain_english: str = Field(description="What the clause means, in plain English.")
    why_it_matters: str = Field(description="Why it is one-sided, risky, or exploitable for the reviewed party.")
    loophole_scenario: str = Field(description="Concrete example of how it could be used against the reviewed party, or ''.")
    what_to_do: str = Field(description="Practical step: change to request, question to ask, or record to keep.")
    confidence: Confidence


class LLMTerm(BaseModel):
    term: str
    definition: str = Field(description="One plain-English sentence, as the term is used in this document.")
    chunk_ids: list[int]


class LLMGap(BaseModel):
    category_id: CategoryId
    observation: str = Field(description="Phrase as 'No clause found that ...'.")


class PassOutput(BaseModel):
    findings: list[LLMFinding]
    key_terms: list[LLMTerm]
    gaps: list[LLMGap]


class LLMParty(BaseModel):
    name: str
    role: str


class SynthesisOutput(BaseModel):
    document_type: str
    parties: list[LLMParty]
    favoured_party: str = Field(description="The party the document favours overall, or 'Balanced'.")
    governing_law: str = Field(description="Governing law if stated in the excerpts, else ''.")
    what_this_document_does: str
    headline: str
    overall_assessment: str
    top_concern_ids: list[str] = Field(description="Up to 5 finding IDs (e.g. 'F3'), most serious first.")
    recommendations: list[str]
    questions_to_ask: list[str]
