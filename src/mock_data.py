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

# Product catalog mapping for UI dropdowns and product-to-supplier defaults
PRODUCT_CATALOG = {
    "Widget-A": {
        "supplier_id": "SUP-201",
        "supplier_name": "Supplier X",
        "ordered_units": 5000,
        "delay_days": 10,
        "reason": "Port congestion",
        "current_stock": 3200,
        "daily_avg_usage": 220,
        "unit_cost": 42.0,
    },
    "Sensor-B": {
        "supplier_id": "SUP-305",
        "supplier_name": "Apex Dynamics",
        "ordered_units": 3000,
        "delay_days": 14,
        "reason": "Factory fire and machinery breakdown",
        "current_stock": 1400,
        "daily_avg_usage": 125,
        "unit_cost": 85.0,
    },
    "Cable-Harness": {
        "supplier_id": "SUP-410",
        "supplier_name": "Pacific Logistics",
        "ordered_units": 6000,
        "delay_days": 7,
        "reason": "Customs regulatory audit",
        "current_stock": 5500,
        "daily_avg_usage": 550,
        "unit_cost": 18.5,
    },
}

import json
import os
from typing import Any, Dict, List, Optional

_RUNTIME_OVERRIDES: Optional[Dict[str, Any]] = None


def set_runtime_overrides(overrides: Optional[Dict[str, Any]]) -> None:
    """Set in-process custom scenario overrides (used by UI / single-run testing)."""
    global _RUNTIME_OVERRIDES
    _RUNTIME_OVERRIDES = dict(overrides) if overrides else None


def clear_runtime_overrides() -> None:
    """Clear in-process custom scenario overrides."""
    global _RUNTIME_OVERRIDES
    _RUNTIME_OVERRIDES = None


def _get_overrides() -> Dict[str, Any]:
    if _RUNTIME_OVERRIDES:
        return _RUNTIME_OVERRIDES
    raw = os.environ.get("MAS_CUSTOM_OVERRIDES")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def get_shipment_data(supplier_id: str) -> Optional[Dict[str, Any]]:
    """Return shipment record for supplier_id, applying any runtime custom overrides."""
    base = SHIPMENTS.get(supplier_id)
    if not base:
        return None
    data = dict(base)
    ov = _get_overrides()
    if ov and (ov.get("supplier_id") == supplier_id or ov.get("product") == data.get("product")):
        for k in ("ordered_units", "delay_days", "reason", "supplier_name", "status"):
            if ov.get(k) is not None:
                data[k] = ov[k]
    return data


def get_inventory_data(product: str) -> Optional[Dict[str, Any]]:
    """Return inventory record for product, applying any runtime custom overrides."""
    base = INVENTORY.get(product)
    if not base:
        return None
    data = dict(base)
    ov = _get_overrides()
    if ov and ov.get("product") == product:
        for k in ("current_stock", "daily_avg_usage", "reorder_point"):
            if ov.get(k) is not None:
                data[k] = ov[k]
    return data


def get_sales_history(product: str) -> Optional[List[float]]:
    """Return 14-day sales history for product, scaled if daily_avg_usage is overridden."""
    base = SALES_HISTORY.get(product)
    if not base:
        return None
    ov = _get_overrides()
    if ov and ov.get("product") == product and ov.get("daily_avg_usage") is not None:
        base_avg = INVENTORY.get(product, {}).get("daily_avg_usage") or (sum(base) / len(base))
        if base_avg > 0:
            scale = float(ov["daily_avg_usage"]) / float(base_avg)
            return [round(v * scale, 2) for v in base]
    return list(base)


def get_market_unit_cost(product: str) -> Optional[float]:
    """Return baseline market unit cost for product, applying any runtime custom override."""
    base = MARKET_UNIT_COST.get(product)
    if base is None:
        return None
    ov = _get_overrides()
    if ov and ov.get("product") == product and ov.get("unit_cost") is not None:
        return float(ov["unit_cost"])
    return float(base)
