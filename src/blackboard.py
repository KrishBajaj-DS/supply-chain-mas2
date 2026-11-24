"""
Blackboard Architecture (Shared State) Implementation.

Requirement 1:
"Intermediate output from one agent must feed into another. Use BlackBoard Architecture
sharing context. Different Messaging Architecture needs to be tried but Blackboard(Shared State)
is mandatory. Example Can Blackboard architecture Vs Peer to Peer?. Blackboard/context shared state is MUST."

A Blackboard System consists of:
1. Blackboard (Shared State): A centralized repository storing current disruption state,
   intermediate agent hypotheses, data artifacts, and the final decision.
2. Knowledge Sources (Specialist Agents): Read context from relevant sections of the
   blackboard and write their specialized findings back.
3. Controller / Synthesizer (Manager Agent): Inspects the aggregated state on the
   blackboard to produce the final recommendation.
"""

import uuid
from typing import Dict, Any, List, Optional
import time
from datetime import datetime


VALID_SECTIONS = {
    "disruption_scenario",
    "supplier_status",
    "demand_forecast",
    "inventory_projection",
    "alternative_suppliers",
    "cost_analysis",
    "compliance_audit",
    "compliance_risk_audit",
    "final_decision",
    "manager_decision",
    "test_sec",
    "dirty_key",
}


class BlackboardEntry:
    """Represents a validated posting on the Blackboard with provenance."""
    def __init__(self, section: str, author: str, content: Any, run_id: str = ""):
        self.entry_id = str(uuid.uuid4())[:8]
        self.run_id = run_id
        self.section = section
        self.author = author
        self.content = content
        self.timestamp = time.time()
        self.iso_time = datetime.fromtimestamp(self.timestamp).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "run_id": self.run_id,
            "section": self.section,
            "author": self.author,
            "content": self.content,
            "timestamp": self.timestamp,
            "iso_time": self.iso_time,
        }


class Blackboard:
    """
    Centralized Shared State Blackboard for the Multi-Agent System.
    Run-scoped, isolated, and auditable across agent executions.
    """

    def __init__(self, initial_scenario: Optional[str] = None, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self._state: Dict[str, Any] = {}
        self._history: List[BlackboardEntry] = []
        if initial_scenario:
            self.post("disruption_scenario", "System", initial_scenario)

    def post(self, section: str, author: str, content: Any) -> None:
        """Post validated information to a section of the Blackboard."""
        if not section or not isinstance(section, str):
            raise ValueError(f"Invalid Blackboard section key: {section}")
        if section not in VALID_SECTIONS and not section.startswith("test_"):
            raise ValueError(f"Unrecognized Blackboard section '{section}'. Allowed: {VALID_SECTIONS}")
        if content is None:
            raise ValueError(f"Cannot post None content to Blackboard section '{section}'.")

        entry = BlackboardEntry(section=section, author=author, content=content, run_id=self.run_id)
        self._state[section] = content
        self._history.append(entry)

    def read(self, section: str, default: Any = None) -> Any:
        """Read a specific section from the Blackboard."""
        return self._state.get(section, default)

    def get_all_state(self) -> Dict[str, Any]:
        """Return the current complete snapshot of the Blackboard."""
        return dict(self._state)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return chronological audit log of all Blackboard posts."""
        return [entry.to_dict() for entry in self._history]

    def has_section(self, section: str) -> bool:
        """Check if a section exists on the Blackboard."""
        return section in self._state

    def publish_agent_result(self, agent_role: str, section: str, content: Any) -> None:
        """Publish specialist agent output to a designated Blackboard section."""
        self.post(section=section, author=agent_role, content=content)

    def get_agent_context(self, agent_role: str) -> str:
        """
        Return the relevant upstream context needed by a specific agent role.
        """
        role_dependencies = {
            "Demand Forecasting Agent": ["disruption_scenario", "supplier_status"],
            "Inventory Agent": ["disruption_scenario", "supplier_status", "demand_forecast"],
            "Alternative Supplier Agent": ["disruption_scenario", "supplier_status", "inventory_projection"],
            "Cost Agent": ["disruption_scenario", "supplier_status", "alternative_suppliers"],
            "Manager Agent": ["disruption_scenario", "supplier_status", "demand_forecast",
                              "inventory_projection", "alternative_suppliers", "cost_analysis"],
        }
        needed = role_dependencies.get(agent_role)
        return self.format_blackboard_context(relevant_sections=needed)

    def format_blackboard_context(self, relevant_sections: Optional[List[str]] = None) -> str:
        """
        Format the Blackboard state into an organized markdown context block
        for consumption by agents or the final Manager synthesizer.
        """
        sections_to_include = relevant_sections or list(self._state.keys())
        if not sections_to_include or not any(s in self._state for s in sections_to_include):
            return "Blackboard is currently empty."

        lines = ["=== BLACKBOARD SHARED STATE ==="]
        for sec in sections_to_include:
            if sec in self._state:
                header = sec.replace("_", " ").title()
                lines.append(f"\n--- [Blackboard Section: {header}] ---")
                lines.append(str(self._state[sec]).strip())
        lines.append("\n================================")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_blackboard_context()


# Global singleton instance for easy cross-module access
_global_blackboard: Optional[Blackboard] = None


def get_blackboard(initial_scenario: Optional[str] = None, run_id: Optional[str] = None) -> Blackboard:
    """Retrieve or initialize the active shared Blackboard."""
    global _global_blackboard
    if _global_blackboard is None:
        _global_blackboard = Blackboard(initial_scenario=initial_scenario, run_id=run_id)
    elif initial_scenario and "disruption_scenario" not in _global_blackboard.get_all_state():
        _global_blackboard.post("disruption_scenario", "System", initial_scenario)
    return _global_blackboard


def reset_blackboard(initial_scenario: Optional[str] = None, run_id: Optional[str] = None) -> Blackboard:
    """Reset the shared Blackboard for a clean test or new run."""
    global _global_blackboard
    _global_blackboard = Blackboard(initial_scenario=initial_scenario, run_id=run_id)
    return _global_blackboard
