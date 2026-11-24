"""
Agent-to-Agent (A2A) Communication Protocol Module.
Fulfills Requirement 6:
- Standardized Agent-to-Agent communication protocol for heterogeneous multi-agent systems.
- Enables seamless interoperability between different frameworks (e.g. CrewAI and Google ADK).
- Supports FIPA-compliant performatives (REQUEST, INFORM, PROPOSE, CONFIRM, FAILURE).
- Provides message validation, envelope serialization, and full audit logging.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from pydantic import BaseModel, Field, ValidationError

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
A2A_AUDIT_LOG = os.path.join(LOG_DIR, "a2a_events.jsonl")


class A2APerformative(str, Enum):
    REQUEST = "REQUEST"
    INFORM = "INFORM"
    PROPOSE = "PROPOSE"
    CONFIRM = "CONFIRM"
    REJECT = "REJECT"
    FAILURE = "FAILURE"


class A2AMessage(BaseModel):
    """Standardized A2A Protocol Envelope."""
    protocol_version: str = Field(default="A2A/1.0", description="A2A Protocol specification version")
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:10]}")
    conversation_id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sender_framework: str = Field(..., description="Framework originating the message (e.g., 'CrewAI', 'Google-ADK')")
    sender_agent: str = Field(..., description="Logical agent identity in sender framework")
    recipient_framework: str = Field(..., description="Framework receiving the message (e.g., 'Google-ADK', 'CrewAI')")
    recipient_agent: str = Field(..., description="Target agent identity in recipient framework")
    performative: A2APerformative = Field(..., description="FIPA-style performative communicative action")
    action: str = Field(..., description="Target skill/capability requested or reported")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured message content")
    reply_to: Optional[str] = Field(default=None, description="Reference to preceding message_id in conversation")
    run_id: Optional[str] = Field(default=None, description="Linked execution run_id for end-to-end tracing")
    architecture: Optional[str] = Field(default=None, description="Active architecture pattern (sequential, parallel, hierarchical)")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class A2ABus:
    """In-memory A2A Protocol Router and Message Bus for heterogeneous agents."""

    def __init__(self):
        self._handlers: Dict[str, Callable[[A2AMessage], A2AMessage]] = {}
        self._message_history: List[A2AMessage] = []

    def register_endpoint(self, agent_id: str, handler: Callable[[A2AMessage], A2AMessage]):
        """Register an agent's A2A message handler."""
        self._handlers[agent_id] = handler

    def unregister_endpoint(self, agent_id: str):
        self._handlers.pop(agent_id, None)

    def dispatch(self, message: A2AMessage) -> A2AMessage:
        """Route message to recipient endpoint across frameworks and record audit trail."""
        self._log_a2a_event(message, direction="OUTBOUND")
        self._message_history.append(message)

        recipient_key = f"{message.recipient_agent}"
        handler = self._handlers.get(recipient_key)
        
        if not handler:
            # Fallback check by agent name only
            for k, h in self._handlers.items():
                if k.lower() in recipient_key.lower() or recipient_key.lower() in k.lower():
                    handler = h
                    break

        if not handler:
            err_resp = A2AMessage(
                conversation_id=message.conversation_id,
                sender_framework="A2A-Bus",
                sender_agent="Router",
                recipient_framework=message.sender_framework,
                recipient_agent=message.sender_agent,
                performative=A2APerformative.FAILURE,
                action=message.action,
                payload={"error": f"Recipient '{message.recipient_agent}' not registered on A2A bus."},
                reply_to=message.message_id,
            )
            self._log_a2a_event(err_resp, direction="ERROR")
            return err_resp

        try:
            response = handler(message)
            self._log_a2a_event(response, direction="INBOUND")
            self._message_history.append(response)
            return response
        except Exception as ex:
            err_resp = A2AMessage(
                conversation_id=message.conversation_id,
                sender_framework=message.recipient_framework,
                sender_agent=message.recipient_agent,
                recipient_framework=message.sender_framework,
                recipient_agent=message.sender_agent,
                performative=A2APerformative.FAILURE,
                action=message.action,
                payload={"error": f"Execution failed: {str(ex)}"},
                reply_to=message.message_id,
            )
            self._log_a2a_event(err_resp, direction="ERROR")
            return err_resp

    def get_history(self) -> List[A2AMessage]:
        return list(self._message_history)

    def _log_a2a_event(self, msg: A2AMessage, direction: str):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(A2A_AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "run_id": getattr(msg, "run_id", None) or msg.payload.get("run_id"),
                    "architecture": getattr(msg, "architecture", None) or msg.payload.get("architecture"),
                    "direction": direction,
                    "timestamp": msg.timestamp,
                    "message_id": msg.message_id,
                    "conversation_id": msg.conversation_id,
                    "sender": f"{msg.sender_framework}:{msg.sender_agent}",
                    "recipient": f"{msg.recipient_framework}:{msg.recipient_agent}",
                    "performative": msg.performative.value,
                    "action": msg.action,
                    "payload_keys": list(msg.payload.keys()),
                }) + "\n")
        except Exception:
            pass


# Singleton A2A Protocol Router
_GLOBAL_A2A_BUS: Optional[A2ABus] = None


def get_a2a_bus() -> A2ABus:
    global _GLOBAL_A2A_BUS
    if _GLOBAL_A2A_BUS is None:
        _GLOBAL_A2A_BUS = A2ABus()
    return _GLOBAL_A2A_BUS


def reset_a2a_bus() -> A2ABus:
    global _GLOBAL_A2A_BUS
    _GLOBAL_A2A_BUS = A2ABus()
    return _GLOBAL_A2A_BUS

