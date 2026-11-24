"""
CrewAI to Google ADK Bridge using the A2A (Agent-to-Agent) Protocol.
Fulfills Requirements 6 and 7:
- CrewAI agents invoke Google ADK agents via standardized A2A protocol envelopes.
- Demonstrates true multi-framework interoperability between CrewAI and Google ADK.
- Automatically shares results to the shared Blackboard state.
"""

from typing import Dict, Any
try:
    from crewai.tools import tool
except ImportError:
    def tool(name=None):
        def decorator(f):
            f.name = name or getattr(f, "__name__", "tool")
            f.func = f
            return f
        return decorator

from src.a2a_protocol import A2AMessage, A2APerformative, get_a2a_bus
from src.adk_agent import get_adk_agent
from src.blackboard import get_blackboard


@tool("query_adk_agent_via_a2a")
def query_adk_agent_via_a2a(supplier_id: str = "SUP-201", product: str = "Widget-A", delay_days: int = 10, shortfall_units: int = 0) -> str:
    """Send an A2A protocol request from CrewAI to the Google ADK Compliance & Risk Agent.
    Audits vendor compliance, calculates supply chain vulnerability index, and returns recommendation."""
    # Ensure ADK agent is initialized and registered on A2A bus
    adk_agent = get_adk_agent()
    bus = get_a2a_bus()

    # 1. Format A2A protocol request envelope
    req_message = A2AMessage(
        sender_framework="CrewAI",
        sender_agent="CrewAI:SupplierMonitoringAgent",
        recipient_framework="Google-ADK",
        recipient_agent=adk_agent.agent_id,
        performative=A2APerformative.REQUEST,
        action="AUDIT_SUPPLIER_RISK",
        payload={
            "supplier_id": supplier_id,
            "product": product,
            "delay_days": delay_days,
            "shortfall_units": shortfall_units,
        },
    )

    # 2. Dispatch via A2A protocol router
    response_msg = bus.dispatch(req_message)

    # 3. Post audit result to Blackboard shared state
    bb = get_blackboard()
    adk_payload = response_msg.payload
    bb.post(
        section="compliance_risk_audit",
        author="GoogleADK:ComplianceRiskAgent",
        content=f"Risk Score: {adk_payload.get('risk_score_index')}/100 ({adk_payload.get('risk_tier')}) | Recommendation: {adk_payload.get('recommendation')}"
    )

    return (
        f"[A2A INTEROPERABILITY PROTOCOL RESPONSE]\n"
        f"Protocol: {response_msg.protocol_version} | ConvID: {response_msg.conversation_id}\n"
        f"Sender: {response_msg.sender_framework} ({response_msg.sender_agent})\n"
        f"Recipient: {response_msg.recipient_framework} ({response_msg.recipient_agent})\n"
        f"Performative: {response_msg.performative.value}\n"
        f"Result: Risk Tier={adk_payload.get('risk_tier')}, Score={adk_payload.get('risk_score_index')}/100\n"
        f"Recommendation: {adk_payload.get('recommendation')}"
    )


def execute_a2a_interoperability_demonstration(scenario_text: str = "SUP-305 delay") -> Dict[str, Any]:
    """Execute a complete A2A protocol exchange between CrewAI and Google ADK."""
    adk_agent = get_adk_agent()
    bus = get_a2a_bus()

    # Step 1: CrewAI sends request to ADK
    msg1 = A2AMessage(
        sender_framework="CrewAI",
        sender_agent="CrewAI:ManagerAgent",
        recipient_framework="Google-ADK",
        recipient_agent=adk_agent.agent_id,
        performative=A2APerformative.REQUEST,
        action="AUDIT_SUPPLIER_RISK",
        payload={"supplier_id": "SUP-305", "product": "Sensor-B", "delay_days": 14, "shortfall_units": 350},
    )
    resp1 = bus.dispatch(msg1)

    # Step 2: CrewAI sends follow-up query for alternative vendor evaluation
    msg2 = A2AMessage(
        conversation_id=msg1.conversation_id,
        sender_framework="CrewAI",
        sender_agent="CrewAI:AlternativeSupplierAgent",
        recipient_framework="Google-ADK",
        recipient_agent=adk_agent.agent_id,
        performative=A2APerformative.REQUEST,
        action="EVALUATE_ALTERNATIVE_VENDORS",
        payload={"candidate_ids": ["SUP-101", "SUP-104", "SUP-105"], "urgency": "critical"},
        reply_to=resp1.message_id,
    )
    resp2 = bus.dispatch(msg2)

    return {
        "status": "SUCCESS",
        "protocol": "A2A/1.0",
        "dialogue_steps": [
            {"request": msg1.model_dump(), "response": resp1.model_dump()},
            {"request": msg2.model_dump(), "response": resp2.model_dump()},
        ],
        "audit_event_count": len(bus.get_history()),
    }


def extract_scenario_metadata(scenario_text: str) -> dict:
    import re
    sup = "SUP-201"
    prod = "Widget-A"
    delay = 10
    if "SUP-305" in scenario_text or "Sensor-B" in scenario_text:
        sup = "SUP-305"
        prod = "Sensor-B"
        delay = 14
    elif "SUP-410" in scenario_text or "Cable-Harness" in scenario_text:
        sup = "SUP-410"
        prod = "Cable-Harness"
        delay = 7
    m = re.search(r"(\d+)[- ]day delay", scenario_text, re.IGNORECASE)
    if m:
        delay = int(m.group(1))
    return {"supplier_id": sup, "product": prod, "delay_days": delay}


def dispatch_a2a_compliance_audit(
    supplier_id: str,
    product: str,
    delay_days: int,
    shortfall_units: int = 0,
    run_id: str = None,
    architecture: str = None,
) -> Dict[str, Any]:
    """Execute A2A request from CrewAI to Google ADK Compliance agent and publish to Blackboard."""
    import json
    adk_agent = get_adk_agent()
    bus = get_a2a_bus()
    req = A2AMessage(
        sender_framework="CrewAI",
        sender_agent="CrewAI:ManagerAgent",
        recipient_framework="Google-ADK",
        recipient_agent=adk_agent.agent_id,
        performative=A2APerformative.REQUEST,
        action="AUDIT_SUPPLIER_RISK",
        payload={
            "supplier_id": supplier_id,
            "product": product,
            "delay_days": delay_days,
            "shortfall_units": shortfall_units,
        },
        run_id=run_id,
        architecture=architecture,
    )
    resp = bus.dispatch(req)
    audit_data = resp.payload

    bb = get_blackboard()
    formatted_audit_content = (
        f"Risk Index: {audit_data.get('risk_score_index')}/100 ({audit_data.get('risk_tier')}) | "
        f"Recommendation: {audit_data.get('recommendation')} | "
        f"Payload: {json.dumps(audit_data)}"
    )
    bb.publish_agent_result(
        agent_role="GoogleADK:ComplianceRiskAgent",
        section="compliance_audit",
        content=formatted_audit_content
    )
    return audit_data


def perform_a2a_compliance_step(architecture: str, scenario_text: str, run_id: str = None) -> Dict[str, Any]:
    """Convenience helper to extract scenario metadata, dispatch A2A audit, and write to blackboard."""
    meta = extract_scenario_metadata(scenario_text)
    return dispatch_a2a_compliance_audit(
        supplier_id=meta["supplier_id"],
        product=meta["product"],
        delay_days=meta["delay_days"],
        shortfall_units=0,
        run_id=run_id,
        architecture=architecture,
    )

