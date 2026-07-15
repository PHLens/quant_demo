"""The legacy reconciliation route is intentionally absent from R0."""
from __future__ import annotations

from web.app import create_app


def test_live_reconcile_endpoint_is_not_registered():
    app = create_app()
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert '/api/live/reconcile' not in rules
