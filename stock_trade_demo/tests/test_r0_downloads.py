"""Exact 17-slot Legacy Artifact inventory and safe download tests."""
from __future__ import annotations

import hashlib

import pytest

from web.app import create_app
from web.v01 import legacy


@pytest.fixture()
def client(tmp_path):
    app = create_app()
    app.config.update(TESTING=True, R0_GENERATION_ROOT=tmp_path / 'generations', R0_ACTION_MARKER_PATH=tmp_path / 'marker')
    return app.test_client()


def test_registry_is_exact_ordered_17_slot_closed_schema(client):
    response = client.get('/api/r0/legacy-artifacts')
    assert response.status_code == 200
    body = response.get_json()
    assert body['bounded'] is True and body['max_items'] == 17 and body['total'] == 17
    assert [item['legacy_artifact_id'] for item in body['items']] == [slot.artifact_id for slot in legacy.SLOTS]
    expected = {
        'legacy_artifact_id', 'label', 'kind', 'format', 'file_name', 'available',
        'size_bytes', 'provenance', 'evidence_status', 'viewer_mode', 'download_state',
        'warning_code', 'download_url', 'summary_url', 'resource_capabilities',
    }
    assert all(set(item) == expected for item in body['items'])
    assert all(not {'source_id', 'strategy_id', 'variant_id', 'snapshot_id', 'view_id'} & set(item) for item in body['items'])


def test_detail_missing_is_200_and_unknown_slot_404(client, monkeypatch, tmp_path):
    slot = legacy.get_slot('profile-csi1000')
    monkeypatch.setattr(legacy, 'REPO_ROOT', tmp_path)
    response = client.get(f'/api/r0/legacy-artifacts/{slot.artifact_id}')
    assert response.status_code == 200
    assert response.get_json()['download_state'] == 'missing'
    assert client.get('/api/r0/legacy-artifacts/not-listed').get_json()['error'] == 'legacy_artifact_not_found'


def test_safe_download_returns_original_bytes_and_metadata_only_never_opens(client, monkeypatch, tmp_path):
    monkeypatch.setattr(legacy, 'REPO_ROOT', tmp_path)
    profile = tmp_path / 'strategy/best_profile_csi1000_timing.json'
    profile.parent.mkdir(parents=True)
    raw = b'{"profile":"safe"}\n'
    profile.write_bytes(raw)
    listing = client.get('/api/r0/legacy-artifacts').get_json()
    item = next(item for item in listing['items'] if item['legacy_artifact_id'] == 'profile-csi1000')
    assert item['download_state'] == 'available'
    response = client.get('/api/r0/downloads/profile-csi1000')
    assert response.status_code == 200
    assert response.data == raw
    assert hashlib.sha256(response.data).hexdigest() == hashlib.sha256(raw).hexdigest()
    assert client.get('/api/r0/downloads/holdout-csi1000').status_code == 403


def test_symlink_sensitive_and_arbitrary_paths_are_rejected(client, monkeypatch, tmp_path):
    monkeypatch.setattr(legacy, 'REPO_ROOT', tmp_path)
    strategy = tmp_path / 'strategy'; strategy.mkdir()
    outside = tmp_path / 'outside.json'; outside.write_text('{"ok":true}')
    (strategy / 'best_profile_csi1000_timing.json').symlink_to(outside)
    assert client.get('/api/r0/downloads/profile-csi1000').get_json()['error'] == 'artifact_download_rejected'
    (strategy / 'best_profile_csi1000_timing.json').unlink()
    (strategy / 'best_profile_csi1000_timing.json').write_text('{"api_key":"secret"}')
    assert client.get('/api/r0/downloads/profile-csi1000').status_code == 403
    assert client.get('/api/r0/downloads/etc-passwd').status_code == 404
