import pytest
from stage3_ai.rl_router import route

def test_route_low_urgency():
    # Routine ops should route strictly to cloud
    assert route(0.1) == "cloud"
    assert route(0.3) == "cloud"

def test_route_medium_urgency():
    # Mild anomalies route to both for deeper cloud context but safe edge baseline
    assert route(0.5) == "both"
    assert route(0.7) == "both"

def test_route_critical_urgency():
    # Severe anomalies route ONLY to edge for < 50ms action
    assert route(0.85) == "edge"
    assert route(1.0) == "edge"
