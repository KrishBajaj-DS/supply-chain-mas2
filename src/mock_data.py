"""
Mock operational data used by the tools. Simulates what would normally come
from an ERP/inventory system and shipment tracking feed.
Supports multiple disruption scenarios across different suppliers and products.
"""

# Current shipment/supplier status feed
SHIPMENTS = {
    "SUP-201": {
        "supplier_name": "Supplier X",
        "product": "Widget-A",
        "ordered_units": 5000,
        "delay_days": 10,
        "reason": "Port congestion",
        "status": "DELAYED",
    },
    "SUP-305": {
        "supplier_name": "Apex Dynamics",
        "product": "Sensor-B",
        "ordered_units": 3000,
        "delay_days": 14,
        "reason": "Factory fire and machinery breakdown",
        "status": "DELAYED",
    },
    "SUP-410": {
        "supplier_name": "Pacific Logistics",
        "product": "Cable-Harness",
        "ordered_units": 6000,
        "delay_days": 7,
        "reason": "Customs regulatory audit",
        "status": "DELAYED",
    },
}

# Current on-hand inventory per product
INVENTORY = {
    "Widget-A": {
        "current_stock": 3200,
        "reorder_point": 2000,
        "daily_avg_usage": 220,
    },
    "Sensor-B": {
        "current_stock": 1400,
        "reorder_point": 1000,
        "daily_avg_usage": 125,
    },
    "Cable-Harness": {
        "current_stock": 5500,
        "reorder_point": 2500,
        "daily_avg_usage": 550,
    },
}

# Historical daily sales/usage (last 14 days) for forecasting
SALES_HISTORY = {
    "Widget-A": [210, 205, 230, 240, 195, 220, 235, 250, 245, 260, 255, 240, 265, 270],
    "Sensor-B": [112, 115, 118, 122, 120, 124, 128, 125, 130, 134, 132, 136, 138, 140],
    "Cable-Harness": [530, 540, 535, 545, 550, 555, 548, 560, 558, 565, 570, 562, 575, 580],
}

# Baseline market cost per unit, used by Cost Agent as a reference point
MARKET_UNIT_COST = {
    "Widget-A": 42.0,       # baseline/market rate for mechanical widget
    "Sensor-B": 85.0,       # baseline rate for electronic sensor unit
    "Cable-Harness": 18.5,  # baseline rate for wiring harness assembly
}
