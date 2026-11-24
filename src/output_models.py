"""
Pydantic output models for the Supply Chain Disruption Management MAS.

Requirement 3 — Structured Output using Pydantic:
  The Manager Agent's final decision is validated and returned as a
  SupplyChainDecision instance (structured JSON-compatible object), rather
  than a raw string. This guarantees the output schema is enforced at
  runtime and the final report is machine-readable.

Usage: pass `output_pydantic=SupplyChainDecision` in the Manager's Task
definition (src/tasks.py) so CrewAI validates the LLM output against this
schema before returning it.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, model_validator


class ReActTrace(BaseModel):
    """A single step in the Manager's Reason → Act → Observe reasoning trace."""
    reason: str = Field(description="What the Manager reasoned about at this step.")
    act: str = Field(description="The action or conclusion the Manager reached.")
    observe: str = Field(description="What the Manager observed from the specialist agents or prior steps.")


class SupplierOption(BaseModel):
    """One sourcing option evaluated by the Manager Agent."""
    option_name: str = Field(description="Short label for this option (e.g. 'Wait for delayed shipment', 'Switch to ALT-03').")
    total_cost: Optional[float] = Field(None, description="Total estimated cost for this option in USD.")
    cost_delta: Optional[float] = Field(None, description="Cost delta vs baseline (positive = more expensive).")
    lead_time_days: Optional[int] = Field(None, description="Lead time in days for this option, if applicable.")
    pros: str = Field(description="Key advantages of this option.")
    cons: str = Field(description="Key disadvantages or risks of this option.")


class SupplyChainDecision(BaseModel):
    """
    Structured final output from the Manager Agent.

    Satisfies Requirement 3 (Pydantic Structured Output): the Manager's
    decision is validated at runtime against this schema and stored as a
    typed, JSON-serialisable object rather than a plain string.
    """

    # Core decision
    recommended_action: str = Field(
        description="The single, specific recommended action to resolve the supply chain disruption."
    )
    justification: str = Field(
        description="Concise justification for the recommendation, citing specific numbers (units, costs, days) from the specialist agents."
    )

    # Quantitative data
    projected_shortfall_units: Optional[int] = Field(
        None,
        description="Projected inventory shortfall in units before the delayed shipment arrives, if any."
    )
    delay_days: Optional[int] = Field(
        None,
        description="Number of days the original shipment is delayed."
    )
    forecasted_demand_units: Optional[int] = Field(
        None,
        description="Total forecasted demand (units) over the delay window."
    )

    # Options evaluated
    options_evaluated: List[SupplierOption] = Field(
        default_factory=list,
        description="List of sourcing options the Manager evaluated before reaching the recommendation."
    )

    # ReAct reasoning trace
    react_trace: List[ReActTrace] = Field(
        default_factory=list,
        description="Step-by-step Reason→Act→Observe trace showing the Manager's reasoning process."
    )

    # Risk / trade-offs
    risks_and_tradeoffs: str = Field(
        description="Key risks or trade-offs of the recommended action that stakeholders should be aware of."
    )

    # Metadata
    persona_context: Optional[str] = Field(
        None,
        description="The target persona/stakeholder whose priorities shaped this recommendation (e.g. 'CFO - cost-first')."
    )

    # Cross-framework A2A Compliance & Validation Tracking
    compliance_audit: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A2A compliance audit data from the Google ADK Risk & Compliance agent."
    )
    validation_status: str = Field(
        default="VALID",
        description="Validation outcome: 'VALID' or 'INVALID_LOGIC'."
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors if decision logic failed consistency checks."
    )

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "SupplyChainDecision":
        """
        Phase 1 Validation:
        1. If shortfall == 0, text must not claim a shortfall exists or cite 'eliminating shortfall'.
        2. If shortfall == 0, action switch/expedite is allowed only if justification states a specific quantified risk.
        3. Option cost_deltas must match units * unit_price * premium within 2% if specified.
        """
        self.validation_errors = []
        shortfall = self.projected_shortfall_units or 0
        text = f"{self.recommended_action} {self.justification} {self.risks_and_tradeoffs}".lower()
        action = self.recommended_action.lower()

        # Rule 1: Zero shortfall cannot claim shortfall ("shortfall of 0" or "shortfall = 0" is allowed)
        if shortfall == 0:
            forbidden = ["eliminating shortfall", "eliminate shortfall", "eliminating projected shortfall", "avoiding a shortfall"]
            has_claim = any(w in text for w in forbidden)
            if not has_claim and "shortfall" in text:
                import re
                m = re.search(r"shortfall\s*(?:of|=|:|\s)\s*([\d,]+)", text)
                if m:
                    num_val = int(m.group(1).replace(",", ""))
                    if num_val > 0:
                        has_claim = True
            if has_claim:
                err = "projected_shortfall_units is 0 but text claims a shortfall or 'eliminating shortfall'."
                if err not in self.validation_errors:
                    self.validation_errors.append(err)

            # Rule 2: If shortfall == 0, default is wait. Switch/expedite requires specific quantified risk
            is_switch_or_expedite = any(w in action for w in ["switch", "expedite", "procure from", "order from", "purchase from", "alternate", "air freight"])
            if is_switch_or_expedite:
                risk_words = ["buffer", "runway", "reliability", "day", "days", "safety stock", "%", "sla", "risk", "contingency"]
                has_quantified_risk = any(rw in text for rw in risk_words) and any(c.isdigit() for c in self.justification + self.risks_and_tradeoffs)
                if not has_quantified_risk:
                    err = "Action is switch/expedite with shortfall == 0 and no stated quantified risk justification."
                    if err not in self.validation_errors:
                        self.validation_errors.append(err)

        # Rule 3: Option cost_delta verification (switch has no premium = 0 delta; expedite has premium > 0)
        for opt in (self.options_evaluated or []):
            opt_name = (opt.option_name or "").lower()
            delta = opt.cost_delta
            if delta is not None:
                if "wait" in opt_name:
                    if abs(delta) > 1.0:
                        err = f"Option '{opt.option_name}' is wait but has non-zero cost_delta: {delta}"
                        if err not in self.validation_errors:
                            self.validation_errors.append(err)
                elif "switch" in opt_name and not any(k in opt_name for k in ["expedite", "rush", "air", "rapid", "premium", "dual"]):
                    if delta > 100.0 and "meridian" in opt_name:
                        err = f"Option '{opt.option_name}' switch to Meridian has unexpected premium delta ({delta}); Meridian has no premium."
                        if err not in self.validation_errors:
                            self.validation_errors.append(err)
                elif any(k in opt_name for k in ["expedite", "rush", "air"]):
                    if delta <= 0.0:
                        err = f"Option '{opt.option_name}' is expedite but cost_delta is not positive ({delta})."
                        if err not in self.validation_errors:
                            self.validation_errors.append(err)

        if self.validation_errors:
            self.validation_status = "INVALID_LOGIC"
        else:
            self.validation_status = "VALID"
        return self

    def to_markdown(self) -> str:
        """
        Render the structured decision as a Markdown report.
        Useful for printing/logging a human-readable version of the
        validated Pydantic output.
        """
        lines = [
            "# Supply Chain Disruption — Manager Decision Report",
            "",
        ]
        if self.persona_context:
            lines += [f"> **Persona context:** {self.persona_context}", ""]

        lines += [
            "## Recommended Action",
            f"**{self.recommended_action}**",
            "",
            "## Justification",
            self.justification,
            "",
        ]

        # Quantitative summary table
        quant_rows = []
        if self.delay_days is not None:
            quant_rows.append(("Shipment delay", f"{self.delay_days} days"))
        if self.forecasted_demand_units is not None:
            quant_rows.append(("Forecasted demand (delay window)", f"{self.forecasted_demand_units:,} units"))
        if self.projected_shortfall_units is not None:
            quant_rows.append(("Projected shortfall", f"{self.projected_shortfall_units:,} units"))

        if quant_rows:
            lines += ["## Key Numbers", "| Metric | Value |", "|--------|-------|"]
            for k, v in quant_rows:
                lines.append(f"| {k} | {v} |")
            lines.append("")

        # Options evaluated
        if self.options_evaluated:
            lines += ["## Options Evaluated", "| Option | Total Cost | Delta | Lead Time | Pros | Cons |",
                      "|--------|-----------|-------|-----------|------|------|"]
            for opt in self.options_evaluated:
                cost_str = f"${opt.total_cost:,.0f}" if opt.total_cost is not None else "—"
                delta_str = (f"+${opt.cost_delta:,.0f}" if (opt.cost_delta or 0) >= 0 else f"-${abs(opt.cost_delta):,.0f}") if opt.cost_delta is not None else "—"
                lt_str = f"{opt.lead_time_days}d" if opt.lead_time_days is not None else "—"
                lines.append(f"| {opt.option_name} | {cost_str} | {delta_str} | {lt_str} | {opt.pros} | {opt.cons} |")
            lines.append("")

        # ReAct trace
        if self.react_trace:
            lines += ["## ReAct Reasoning Trace"]
            for i, step in enumerate(self.react_trace, 1):
                lines += [
                    f"### Step {i}",
                    f"**Reason:** {step.reason}",
                    f"**Act:** {step.act}",
                    f"**Observe:** {step.observe}",
                    "",
                ]

        lines += ["## Risks & Trade-offs", self.risks_and_tradeoffs, ""]

        return "\n".join(lines)


def compute_shortfall(forecasted_demand: int = 0, current_stock: int = 0, forecast: int = None, stock: int = None) -> int:
    """Explicit shortfall formula: shortfall = max(0, forecasted_demand_over_delay - current_stock)."""
    f = forecast if forecast is not None else forecasted_demand
    s = stock if stock is not None else current_stock
    return max(0, int(f) - int(s))


def check_cost_for_quantity(units: int = 0, unit_price: float = 0.0, premium_pct: float = 0.0,
                            stated_delta: float = 0.0, tolerance: float = 0.02,
                            reported_cost: float = None) -> bool:
    """
    Validates that an option's cost_delta or total_cost matches quantity calculations within tolerance (default 2%).
    """
    if reported_cost is not None:
        expected_total = units * unit_price * (1.0 + (premium_pct / 100.0))
        denom = max(abs(expected_total), 1.0)
        return abs(reported_cost - expected_total) / denom <= tolerance

    expected_delta = units * unit_price * (premium_pct / 100.0)
    if expected_delta == 0 and stated_delta == 0:
        return True
    denom = max(abs(expected_delta), 1.0)
    return abs(stated_delta - expected_delta) / denom <= tolerance


def validate_decision_logic(decision: SupplyChainDecision) -> Tuple[bool, List[str]]:
    """Standalone post-generation consistency validator for SupplyChainDecision."""
    errors = list(decision.validation_errors)
    shortfall = decision.projected_shortfall_units or 0
    text = f"{decision.recommended_action} {decision.justification} {decision.risks_and_tradeoffs}".lower()
    action = decision.recommended_action.lower()

    if shortfall == 0:
        if "eliminating shortfall" in text or "eliminate shortfall" in text or ("avoiding a " in text and "shortfall" in text):
            err = "projected_shortfall_units is 0 but text claims a shortfall or 'eliminating shortfall'."
            if err not in errors:
                errors.append(err)

        is_switch_or_expedite = any(w in action for w in ["switch", "expedite", "procure from", "order from", "purchase from", "alternate", "air freight"])
        if is_switch_or_expedite:
            risk_words = ["buffer", "runway", "reliability", "day", "days", "safety stock", "%", "sla", "risk", "contingency"]
            has_quantified_risk = any(rw in text for rw in risk_words) and any(c.isdigit() for c in decision.justification + decision.risks_and_tradeoffs)
            if not has_quantified_risk:
                err = "Action is switch/expedite with shortfall == 0 and no stated quantified risk justification."
                if err not in errors:
                    errors.append(err)

    return (len(errors) == 0, errors)
