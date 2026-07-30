"""Learn/Lab-1 API: strict POST commands and read-only GET projections."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from lab.service import LabError

bp = Blueprint('lab_api', __name__)


def _service():
    return current_app.extensions['lab_service']


def _json_body():
    value = request.get_json(silent=True)
    if value is None and request.data:
        raise LabError('invalid_json', 'request body must be valid JSON')
    return value


@bp.errorhandler(LabError)
def handle_lab_error(error: LabError):
    payload = {
        'error': error.code,
        'message': error.message,
    }
    if error.details is not None:
        payload['details'] = error.details
    return jsonify(payload), error.status


@bp.get('/api/learn/topics')
def api_learn_topics():
    return jsonify(_service().learn_topics())


@bp.get('/api/lab/templates')
def api_lab_templates():
    return jsonify({'templates': [_service().template()]})


@bp.post('/api/lab/experiments')
def api_create_experiment():
    return jsonify(_service().create_experiment(_json_body())), 201


@bp.get('/api/lab/experiments/<experiment_id>')
def api_get_experiment(experiment_id):
    return jsonify(_service().get_experiment(experiment_id))


@bp.post('/api/lab/experiments/<experiment_id>/variants')
def api_create_variant(experiment_id):
    variant, created = _service().create_variant(experiment_id, _json_body())
    return jsonify(variant), 201 if created else 200


@bp.post('/api/lab/variants/<variant_id>/runs')
def api_submit_run(variant_id):
    run, status = _service().submit_run(variant_id, _json_body())
    return jsonify(run), status


@bp.get('/api/lab/runs/<run_id>')
def api_get_run(run_id):
    return jsonify(_service().get_run(run_id))


@bp.get('/api/lab/results/<result_id>')
def api_get_result(result_id):
    return jsonify(_service().get_result(result_id))


@bp.post('/api/lab/comparisons')
def api_create_comparison():
    return jsonify(_service().compare(_json_body()))
