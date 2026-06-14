from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RagChunk:
    document_id: str
    chunk_id: str
    source: str
    title: str
    text: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    result_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EscalationDecision:
    required: bool
    trigger: Optional[str] = None
    priority: str = "normal"
    reason: Optional[str] = None
    ticket_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    answer: str
    outcome_type: str
    intent: str
    safety_status: str
    sources: List[RagChunk] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    escalation: EscalationDecision = field(default_factory=lambda: EscalationDecision(False))
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "outcome_type": self.outcome_type,
            "intent": self.intent,
            "safety_status": self.safety_status,
            "sources": [chunk.to_dict() for chunk in self.sources],
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "escalation": self.escalation.to_dict(),
            "trace_id": self.trace_id,
        }
