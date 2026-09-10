from pydantic import BaseModel, Field
from typing import List, Optional

class SupportResponseSchema(BaseModel):
    """
    Structured Pydantic schema for grounded customer support reply generation.
    """
    reply: str = Field(..., description="The generated grounded customer support response.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model self-reported confidence score between 0.0 and 1.0.")
    evidence_ids: List[str] = Field(default_factory=list, description="List of historical case_ids used as grounding evidence.")
    should_escalate: bool = Field(default=False, description="Whether the response requires human agent escalation.")
    escalation_reason: Optional[str] = Field(default=None, description="Explicit reason code or description if escalated.")
