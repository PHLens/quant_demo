"""R0 public download allowlist and content-safety checks."""
from __future__ import annotations

from pathlib import Path

import pytest

from web.app import create_app
from web.blueprints import r0_viewer_api as viewer


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_known_json_and_csv_downloads_are_attachments(client):
    json_response = client.get('/api/r0/artifacts/profile-csi1000/download')
    assert json_response.status_code == 200
    assert json_response.mimetype == 'application/json'
    assert 'attachment' in json_response.headers['Content-Disposition']

    csv_response = client.get('/api/r0/artifacts/sensitivity-csi1000/download')
    assert csv_response.status_code == 200
    assert csv_response.mimetype == 'text/csv'
    assert 'attachment' in csv_response.headers['Content-Disposition']


def test_non_allowlisted_or_metadata_only_download_is_denied(client):
    assert client.get('/api/r0/artifacts/not-listed/download').status_code == 404
    assert client.get('/api/r0/artifacts/holdout-csi1000/download').status_code == 403
    assert client.get('/api/r0/artifacts/cache-web/download').status_code == 403


def _artifact(relative_path: str, downloadable: bool = True) -> viewer.LegacyArtifact:
    return viewer.LegacyArtifact('fixture', 'Fixture', 'fixture', relative_path, downloadable)


def test_validation_rejects_sensitive_json_key(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, '_REPO_ROOT', tmp_path)
    (tmp_path / 'blocked.json').write_text('{"api_key": "redacted"}', encoding='utf-8')
    with pytest.raises(ValueError, match='sensitive'):
        viewer._validated_download(_artifact('blocked.json'))


def test_validation_rejects_sensitive_csv_header(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, '_REPO_ROOT', tmp_path)
    (tmp_path / 'blocked.csv').write_text('date,access_token\n2026-01-01,redacted\n', encoding='utf-8')
    with pytest.raises(ValueError, match='sensitive'):
        viewer._validated_download(_artifact('blocked.csv'))


def test_validation_rejects_private_absolute_path_in_value(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, '_REPO_ROOT', tmp_path)
    (tmp_path / 'blocked.json').write_text('{"source": "/home/example/private.csv"}', encoding='utf-8')
    with pytest.raises(ValueError, match='sensitive'):
        viewer._validated_download(_artifact('blocked.json'))


def test_validation_rejects_path_traversal(tmp_path, monkeypatch):
    root = tmp_path / 'public'
    root.mkdir()
    (tmp_path / 'outside.json').write_text('{"ok": true}', encoding='utf-8')
    monkeypatch.setattr(viewer, '_REPO_ROOT', root)
    with pytest.raises(ValueError, match='escapes'):
        viewer._validated_download(_artifact('../outside.json'))


def test_validation_rejects_symlink(tmp_path, monkeypatch):
    target = tmp_path / 'target.json'
    target.write_text('{"ok": true}', encoding='utf-8')
    link = tmp_path / 'link.json'
    link.symlink_to(target)
    monkeypatch.setattr(viewer, '_REPO_ROOT', tmp_path)
    with pytest.raises(ValueError, match='symlink'):
        viewer._validated_download(_artifact('link.json'))


def test_validation_rejects_unsupported_type_and_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, '_REPO_ROOT', tmp_path)
    (tmp_path / 'note.md').write_text('plain text', encoding='utf-8')
    with pytest.raises(ValueError, match='type'):
        viewer._validated_download(_artifact('note.md'))

    large = tmp_path / 'large.json'
    large.write_text('{"value":"1234567890"}', encoding='utf-8')
    monkeypatch.setattr(viewer, '_MAX_DOWNLOAD_BYTES', 8)
    with pytest.raises(ValueError, match='size'):
        viewer._validated_download(_artifact('large.json'))


def test_validation_accepts_small_public_json(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, '_REPO_ROOT', tmp_path)
    path = tmp_path / 'safe.json'
    path.write_text('{"data_as_of": "2026-01-01", "value": 1}', encoding='utf-8')
    resolved, mime = viewer._validated_download(_artifact('safe.json'))
    assert resolved == path
    assert mime == 'application/json'
