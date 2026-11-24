"""
Mock supplier knowledge base used by the Alternative Supplier Agent's RAG tool.
Each entry simulates a supplier profile document you'd normally pull from a
vendor management system or procurement database.
"""

SUPPLIER_DOCS = [
    {
        "id": "SUP-101",
        "name": "Meridian Components Ltd.",
        "text": (
            "Meridian Components Ltd. is a Tier-1 electronics parts supplier based in "
            "Chennai, India. Average lead time is 7 days for standard orders. "
            "Historical on-time delivery rate: 96%. Capacity: up to 8,000 units/week. "
            "Specializes in connectors, PCBs, and small mechanical assemblies. "
            "Past incidents: one 3-day delay in 2023 due to customs hold, resolved quickly. "
            "Preferred payment terms: Net 30. Certified ISO 9001."
        ),
    },
    {
        "id": "SUP-102",
        "name": "Northline Industrial Supply",
        "text": (
            "Northline Industrial Supply operates out of Pune and specializes in bulk "
            "hardware and fasteners. Average lead time is 10 days. Historical on-time "
            "delivery rate: 89%. Capacity: up to 15,000 units/week, making them suitable "
            "for large shortfalls. Slightly higher unit cost (approx 8% above market "
            "average) but strong volume flexibility. No major disruption incidents in "
            "past 2 years."
        ),
    },
    {
        "id": "SUP-103",
        "name": "Vertex Global Traders",
        "text": (
            "Vertex Global Traders is an import/export intermediary with suppliers "
            "across Southeast Asia. Average lead time is 14-18 days due to customs "
            "processing. Historical on-time delivery rate: 82%. Capacity: highly "
            "variable, up to 20,000 units/week when available. Lowest unit cost option "
            "(approx 12% below market average) but higher risk of further delay. Best "
            "used when time is not critical and cost savings matter most."
        ),
    },
    {
        "id": "SUP-104",
        "name": "Rapid Source Logistics",
        "text": (
            "Rapid Source Logistics is an expedite-focused domestic supplier based in "
            "Gurugram. Average lead time is 3 days for emergency orders. Historical "
            "on-time delivery rate: 98%. Capacity: limited, up to 3,000 units/week. "
            "Premium pricing, approx 25% above market average, but ideal for urgent "
            "short-term gaps while a longer-term supplier is arranged. Certified for "
            "rush fulfillment SLAs."
        ),
    },
    {
        "id": "SUP-105",
        "name": "Anchor Bay Manufacturing",
        "text": (
            "Anchor Bay Manufacturing is a mid-size regional manufacturer near Mumbai. "
            "Average lead time is 9 days. Historical on-time delivery rate: 91%. "
            "Capacity: up to 6,000 units/week. Unit cost roughly at market average. "
            "Known for strong quality control and low defect rates (under 0.5%). "
            "Good balance of reliability and cost, commonly used as a secondary "
            "supplier in diversified sourcing strategies."
        ),
    },
]


def get_supplier_docs():
    """Return the mock supplier knowledge base as a list of dicts."""
    return SUPPLIER_DOCS
