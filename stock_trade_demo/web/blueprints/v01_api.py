"""Canonical quant_demo v0.1 HTTP contract."""
from __future__ import annotations

from io import BytesIO
from typing import Any
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request, send_file

from web.v01 import actions, legacy, manual_ledger, projections, reconciliation
from web.v01.catalog import (
    MANUAL_CAPABILITIES, RESOURCE_ORDER, SOURCE_SPECS, STRATEGY_SPECS,
    find_variant, get_source, get_strategy, lookup_variant, recovery_plan_id,
    variants_for,
)
from web.v01.contracts import (
    ApiError, bounded_envelope, ensure_allowed_args, paginate, parse_limit,
    require_no_args,
)
from web.v01.snapshot_store import (
    canonical_view, current_snapshot, filter_frame,
    inspect_target, published_signal, remember_view, source_date_bounds,
    snapshot_date_bounds, validate_view,
)


bp = Blueprint('v01_api', __name__)


@bp.before_request
def _startup_gate():
    if request.endpoint == 'v01_api.api_health':
        return None
    actions.runtime().ensure_ready()
    return None


def _recovery(spec, variant, target_state) -> dict[str, Any]:
    supported = bool(spec.recovery_supported)
    recoverable = supported and target_state.blocker_code is None and (
        target_state.degradation is not None
        or target_state.cache_state in {'missing', 'corrupt', 'artifact_stale'}
    )
    if not supported:
        return {
            'recovery_supported': False, 'recoverable_now': False, 'blocker_code': None,
            'recovery_plan_id': None, 'recovery_scopes': None, 'recovery_steps': None,
            'eta_seconds': None,
        }
    plan_id = recovery_plan_id(spec, variant)
    steps = [
        {'step_id': f'pull-{scope}', 'label': f'Needed {scope} data', 'scope': scope}
        for scope in spec.recovery_scopes
    ] + [{'step_id': 'build-target', 'label': 'Build and validate exact catalog target', 'scope': 'build'}]
    return {
        'recovery_supported': True, 'recoverable_now': recoverable,
        'blocker_code': target_state.blocker_code, 'recovery_plan_id': plan_id,
        'recovery_scopes': list(spec.recovery_scopes), 'recovery_steps': steps,
        'eta_seconds': spec.eta_seconds,
    }


def _state_reason(target_state) -> dict[str, Any] | None:
    if target_state.readable:
        if target_state.degradation:
            return {'code': 'degraded', 'detail': target_state.degradation['code']}
        if target_state.freshness_state == 'stale':
            return {'code': 'data_stale', 'detail': target_state.freshness_reason['code']}
        return None
    detail = target_state.blocker_code if target_state.cache_state == 'missing' else None
    return {'code': target_state.cache_state, 'detail': detail}


def _variant_payload(spec, variant) -> dict[str, Any]:
    target_state, snapshot = inspect_target(spec, variant)
    overlay = actions.runtime().overlay(spec.source_id, spec.strategy_id, variant.variant_id)
    recovery = _recovery(spec, variant, target_state)
    return {
        'source_id': spec.source_id, 'strategy_id': spec.strategy_id,
        'variant_id': variant.variant_id, 'canonical_params': variant.canonical_params,
        'param_schema': variant.param_schema, 'default': variant.default,
        **target_state.as_fields(), 'recovery_supported': recovery['recovery_supported'],
        'recoverable_now': recovery['recoverable_now'], 'blocker_code': recovery['blocker_code'],
        'recovery_scopes': recovery['recovery_scopes'], 'recovery_plan_id': recovery['recovery_plan_id'],
        'recovery_steps': recovery['recovery_steps'], 'eta_seconds': recovery['eta_seconds'],
        'generated_at': snapshot.generated_at if snapshot else None,
        'operation_id': overlay[0] if overlay else None,
        'status_url': f'/api/r0/actions/{overlay[0]}' if overlay else None,
        'last_operation_id': None, 'last_status_url': None,
    }


def _latest_snapshot(spec, variant) -> dict[str, Any]:
    target_state, snapshot = inspect_target(spec, variant)
    overlay = actions.runtime().overlay(spec.source_id, spec.strategy_id, variant.variant_id)
    signal = published_signal(snapshot) if snapshot else None
    applicable = 'published_current_signal' in spec.capabilities
    recovery = _recovery(spec, variant, target_state) if not target_state.readable or target_state.degradation else None
    error = None
    if not target_state.readable:
        code = {'recovering': 'cache_recovering', 'updating': 'snapshot_updating'}.get(target_state.cache_state, 'cache_miss')
        error = {'code': code, 'blocker_code': target_state.blocker_code}
    return {
        'variant_id': variant.variant_id, 'snapshot_id': snapshot.snapshot_id if snapshot else None,
        **target_state.as_fields(),
        'data_as_of': snapshot_date_bounds(snapshot)[1] if snapshot else None,
        'cache_generated_at': snapshot.generated_at if snapshot else None,
        'current_signal': signal, 'signal_state': 'available' if signal else ('not_applicable' if not applicable and target_state.readable else 'unavailable'),
        'state_reason': _state_reason(target_state),
        'snapshot_url': f'/api/r0/sources/{spec.source_id}/strategies/{spec.strategy_id}/snapshots?variant_id={variant.variant_id}' if snapshot else None,
        'operation_id': overlay[0] if overlay else None,
        'status_url': f'/api/r0/actions/{overlay[0]}' if overlay else None,
        'recovery': recovery, 'error': error,
    }


@bp.get('/api/r0/sources')
def api_sources():
    require_no_args(request.args)
    items = []
    for source in SOURCE_SPECS:
        low, high = source_date_bounds(source.source_id)
        items.append({
            'source_id': source.source_id, 'display_name': source.display_name,
            'related_only': source.related_only, 'data_min_date': low, 'data_max_date': high,
            'resource_capabilities': [item for item in RESOURCE_ORDER if item in source.capabilities],
        })
    return jsonify(bounded_envelope('sources', items, 16))


@bp.get('/api/r0/sources/<source_id>/strategies')
def api_strategies(source_id: str):
    ensure_allowed_args(request.args, {'include'})
    source = get_source(source_id)
    include = request.args.get('include')
    if include not in (None, 'latest_snapshot'):
        raise ApiError(400, 'invalid_params', 'include only accepts latest_snapshot.')
    items = []
    for spec in STRATEGY_SPECS:
        if spec.source_id != source_id:
            continue
        variants = variants_for(spec)
        default = next(item for item in variants if item.default)
        row = {
            'source_id': source_id, 'strategy_id': spec.strategy_id,
            'display_name': spec.display_name, 'default_variant_id': default.variant_id,
            'resource_capabilities': [item for item in RESOURCE_ORDER if item in spec.capabilities],
        }
        if include == 'latest_snapshot':
            row['latest_snapshot'] = _latest_snapshot(spec, default)
        items.append(row)
    return jsonify(bounded_envelope('strategies', items, 64, source_id=source.source_id))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/variants')
def api_variants(source_id: str, strategy_id: str):
    require_no_args(request.args)
    spec = get_strategy(source_id, strategy_id)
    items = [_variant_payload(spec, variant) for variant in variants_for(spec)]
    items.sort(key=lambda item: (not item['default'], json_key(item['canonical_params']), item['variant_id']))
    return jsonify(bounded_envelope('variants', items, 256, source_id=source_id, strategy_id=strategy_id))


def json_key(value: Any) -> str:
    from web.v01.contracts import canonical_json
    return canonical_json(value)


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/variant-lookup')
def api_variant_lookup(source_id: str, strategy_id: str):
    variant = lookup_variant(source_id, strategy_id, request.args.to_dict(flat=True))
    spec = get_strategy(source_id, strategy_id)
    return jsonify(_variant_payload(spec, variant))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots')
def api_snapshot(source_id: str, strategy_id: str):
    ensure_allowed_args(request.args, {'variant_id', 'start', 'end', 'benchmark', 'resolution'})
    variant_id = request.args.get('variant_id')
    if not variant_id:
        raise ApiError(400, 'invalid_params', 'variant_id is required.')
    variant = find_variant(source_id, strategy_id, variant_id)
    snapshot = current_snapshot(source_id, strategy_id, variant)
    view, view_id = canonical_view(snapshot, request.args.to_dict(flat=True))
    remember_view(snapshot, view_id, view)
    return jsonify(projections.summary(snapshot, view, view_id))


def _snapshot_for_path(source_id: str, strategy_id: str, snapshot_id: str):
    spec = get_strategy(source_id, strategy_id)
    saw_snapshot = False
    for variant in variants_for(spec):
        target_state, snapshot = inspect_target(spec, variant)
        if snapshot is not None and snapshot.snapshot_id == snapshot_id:
            return snapshot
        saw_snapshot = saw_snapshot or snapshot is not None
    # The strategy exists and its current target no longer matches the path.
    if saw_snapshot:
        raise ApiError(409, 'snapshot_changed', 'The target fingerprint changed; reopen Summary.')
    raise ApiError(404, 'snapshot_not_found', 'Canonical snapshot does not exist for this strategy.')


def _require_capability(snapshot, capability: str) -> None:
    if capability not in snapshot.spec.capabilities:
        raise ApiError(404, 'resource_not_supported', 'The source does not publish this resource.', resource=capability)


def _resource_context(source_id: str, strategy_id: str, snapshot_id: str, allowed: set[str]):
    ensure_allowed_args(request.args, allowed)
    snapshot = _snapshot_for_path(source_id, strategy_id, snapshot_id)
    view = validate_view(snapshot, request.args.get('view_id'))
    frame = filter_frame(snapshot, view)
    return snapshot, view, frame


def _paged_response(snapshot, view_id: str, resource: str, items: list[dict[str, Any]], *, kind: str | None, binding_extra: dict[str, Any], sort_key, strip_keys=(), extra=None):
    limit = parse_limit(request.args.get('limit'))
    page, total, next_cursor = paginate(
        items, limit=limit, cursor=request.args.get('cursor'),
        binding={'snapshot_id': snapshot.snapshot_id, 'view_id': view_id, 'resource': resource, **binding_extra},
        sort_key=sort_key,
    )
    clean = [{key: value for key, value in item.items() if key not in strip_keys} for item in page]
    payload = {'snapshot_id': snapshot.snapshot_id, 'view_id': view_id, 'resource': resource}
    if kind is not None:
        payload['kind'] = kind
    if extra:
        payload.update(extra)
    payload.update({'items': clean, 'total': total, 'next_cursor': next_cursor})
    return payload


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/series')
def api_series(source_id: str, strategy_id: str, snapshot_id: str):
    allowed = {'view_id', 'kind', 'window', 'resolution', 'benchmark_id', 'series_id', 'cursor', 'limit'}
    snapshot, view, frame = _resource_context(source_id, strategy_id, snapshot_id, allowed)
    _require_capability(snapshot, 'series')
    kind = request.args.get('kind') or 'equity'
    window = request.args.get('window') or 'full'
    resolution = request.args.get('resolution') or view['resolution']
    items = projections.series(
        snapshot, frame, kind=kind, window=window, resolution=resolution,
        benchmark_id=request.args.get('benchmark_id'),
        series_id=request.args.get('series_id'), view=view,
    )
    return jsonify(_paged_response(snapshot, request.args['view_id'], 'series', items, kind=kind,
        binding_extra={'window': window, 'resolution': resolution, 'benchmark_id': request.args.get('benchmark_id'), 'series_id': request.args.get('series_id'), 'direction': 'ASC'},
        sort_key=lambda item: (item.get('date') or item.get('year') or '', item['artifact_row_ordinal']), strip_keys={'artifact_row_ordinal'}))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/signals')
def api_signals(source_id: str, strategy_id: str, snapshot_id: str):
    snapshot, _, frame = _resource_context(source_id, strategy_id, snapshot_id, {'view_id', 'event', 'cursor', 'limit'})
    _require_capability(snapshot, 'signals')
    event = request.args.get('event') or 'all'
    items = projections.signals(frame, event)
    return jsonify(_paged_response(snapshot, request.args['view_id'], 'signals', items, kind=None,
        binding_extra={'event': event, 'direction': 'ASC'}, sort_key=lambda item: (item['date'] or '', item['artifact_row_ordinal']), strip_keys={'artifact_row_ordinal'}))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/position')
def api_position(source_id: str, strategy_id: str, snapshot_id: str):
    snapshot, _, frame = _resource_context(source_id, strategy_id, snapshot_id, {'view_id'})
    _require_capability(snapshot, 'position')
    return jsonify({'snapshot_id': snapshot.snapshot_id, 'view_id': request.args['view_id'], 'resource': 'position', 'position': projections._position(frame)})


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/interval-windows')
def api_windows(source_id: str, strategy_id: str, snapshot_id: str):
    snapshot, _, frame = _resource_context(source_id, strategy_id, snapshot_id, {'view_id'})
    _require_capability(snapshot, 'interval_windows')
    items = projections.interval_windows(snapshot, frame, request.args['view_id'])
    return jsonify(bounded_envelope('interval_windows', items, 64, snapshot_id=snapshot.snapshot_id, view_id=request.args['view_id']))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/holding-periods')
def api_holding_periods(source_id: str, strategy_id: str, snapshot_id: str):
    snapshot, _, frame = _resource_context(source_id, strategy_id, snapshot_id, {'view_id', 'window', 'cursor', 'limit'})
    _require_capability(snapshot, 'holding_periods')
    window = request.args.get('window') or 'full'
    context, items, _ = projections.holding_periods(snapshot, frame, window)
    base = f'/api/r0/sources/{source_id}/strategies/{strategy_id}/snapshots/{snapshot_id}/holdings'
    for item in items:
        item['stocks_url'] = f'{base}?view_id={request.args["view_id"]}&period_id={item["period_id"]}'
    return jsonify(_paged_response(snapshot, request.args['view_id'], 'holding-periods', items, kind=None,
        binding_extra={'window': window, 'direction': 'DESC'}, sort_key=lambda item: (item['date'] or '', item['artifact_period_ordinal']),
        strip_keys={'artifact_period_ordinal'}, extra={'context': context}))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/holdings')
def api_holdings(source_id: str, strategy_id: str, snapshot_id: str):
    snapshot, _, frame = _resource_context(source_id, strategy_id, snapshot_id, {'view_id', 'period_id', 'cursor', 'limit'})
    _require_capability(snapshot, 'holdings')
    period_id = request.args.get('period_id')
    if not period_id:
        raise ApiError(400, 'invalid_params', 'period_id is required.')
    _, _, mapping = projections.holding_periods(snapshot, frame, 'full')
    if period_id not in mapping:
        raise ApiError(404, 'holding_period_not_found', 'Holding period does not exist in this view.')
    items = mapping[period_id]
    return jsonify(_paged_response(snapshot, request.args['view_id'], 'holdings', items, kind=None,
        binding_extra={'period_id': period_id, 'direction': 'ASC'}, sort_key=lambda item: (item['artifact_stock_ordinal'],)))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/trades')
def api_trades(source_id: str, strategy_id: str, snapshot_id: str):
    snapshot, _, frame = _resource_context(source_id, strategy_id, snapshot_id, {'view_id', 'projection', 'cursor', 'limit'})
    _require_capability(snapshot, 'trades')
    projection = request.args.get('projection') or 'detail'
    items, trade_summary = projections.trades(frame, projection)
    return jsonify(_paged_response(snapshot, request.args['view_id'], 'trades', items, kind=None,
        binding_extra={'projection': projection, 'direction': 'ASC'}, sort_key=lambda item: (item['date'] or '', item['artifact_row_ordinal']),
        strip_keys={'artifact_row_ordinal'}, extra={'trade_summary': trade_summary}))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/fees')
def api_fees(source_id: str, strategy_id: str, snapshot_id: str):
    snapshot, _, frame = _resource_context(source_id, strategy_id, snapshot_id, {'view_id'})
    _require_capability(snapshot, 'fees')
    return jsonify({'snapshot_id': snapshot.snapshot_id, 'view_id': request.args['view_id'], 'resource': 'fees', 'fees': projections.fees(snapshot, frame)})


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/factors')
def api_factors(source_id: str, strategy_id: str, snapshot_id: str):
    if 'factor_id' in request.args:
        raise ApiError(400, 'invalid_params', 'factor_id is not a v0.1 factors query.')
    snapshot = _snapshot_for_path(source_id, strategy_id, snapshot_id)
    _require_capability(snapshot, 'factors')
    allowed = {'view_id', 'kind', 'weeks', 'cursor', 'limit'}
    ensure_allowed_args(request.args, allowed)
    view = validate_view(snapshot, request.args.get('view_id'))
    kind = request.args.get('kind')
    if kind == 'metadata':
        if 'cursor' in request.args or 'limit' in request.args or 'weeks' in request.args:
            raise ApiError(400, 'invalid_params', 'metadata is a singleton and rejects pagination.')
        return jsonify({'snapshot_id': snapshot.snapshot_id, 'view_id': request.args['view_id'], 'resource': 'factors', 'kind': 'metadata', 'metadata': projections.factor_metadata(snapshot)})
    extra = {}
    if kind == 'overview':
        if 'weeks' in request.args:
            raise ApiError(400, 'invalid_params', 'weeks only applies to sector_heat.')
        extra, items = projections.factor_overview(snapshot)
    elif kind == 'sector_heat':
        try:
            weeks = int(request.args.get('weeks', 8))
        except ValueError as exc:
            raise ApiError(400, 'invalid_params', 'weeks must be an integer.') from exc
        if weeks < 1 or weeks > 52:
            raise ApiError(400, 'invalid_params', 'weeks must be within 1..52.')
        extra, items = projections.sector_heat(snapshot, weeks)
    elif kind == 'single_factor':
        if 'weeks' in request.args:
            raise ApiError(400, 'invalid_params', 'weeks only applies to sector_heat.')
        extra, items = projections.single_factor(snapshot)
        for item in items:
            item['series_url'] = f'/api/r0/sources/{source_id}/strategies/{strategy_id}/snapshots/{snapshot_id}/series?view_id={request.args["view_id"]}&kind=factor_nav&window=full&resolution=day&series_id={item["series_id"]}'
    else:
        raise ApiError(400, 'invalid_params', 'kind must be metadata, overview, sector_heat or single_factor.')
    return jsonify(_paged_response(snapshot, request.args['view_id'], 'factors', items, kind=kind,
        binding_extra={'kind': kind, 'weeks': request.args.get('weeks'), 'direction': 'ASC'},
        sort_key=lambda item: (item.get('artifact_factor_ordinal', item.get('artifact_cell_ordinal', 0)),), extra=extra))


@bp.get('/api/r0/sources/<source_id>/strategies/<strategy_id>/snapshots/<snapshot_id>/configuration')
def api_configuration(source_id: str, strategy_id: str, snapshot_id: str):
    require_no_args(request.args)
    snapshot = _snapshot_for_path(source_id, strategy_id, snapshot_id)
    _require_capability(snapshot, 'configuration')
    return jsonify({'snapshot_id': snapshot.snapshot_id, 'resource': 'configuration', **projections.configuration(snapshot)})


@bp.get('/api/r0/legacy-artifacts')
def api_legacy_artifacts():
    require_no_args(request.args)
    return jsonify(bounded_envelope('legacy_artifacts', [legacy.item(slot) for slot in legacy.SLOTS], 17))


@bp.get('/api/r0/legacy-artifacts/<artifact_id>')
def api_legacy_artifact(artifact_id: str):
    require_no_args(request.args)
    return jsonify(legacy.item(legacy.get_slot(artifact_id)))


@bp.get('/api/r0/downloads/<resource_id>')
def api_download(resource_id: str):
    require_no_args(request.args)
    slot = legacy.get_slot(resource_id)
    path, mime, raw = legacy.validated_download(slot)
    # Serve the exact bytes that passed validation; reopening the pathname here
    # would reintroduce a validation-to-send replacement race.
    return send_file(BytesIO(raw), mimetype=mime, as_attachment=True, download_name=path.name, conditional=False, max_age=0)


@bp.get('/api/r0/data-status')
def api_data_status():
    require_no_args(request.args)
    check = actions.data_check(None)
    rows = []
    for source in SOURCE_SPECS:
        low, high = source_date_bounds(source.source_id)
        rows.append({
            'source_id': source.source_id, 'display_name': source.display_name,
            'related_only': source.related_only, 'data_min_date': low,
            'data_max_date': high,
            'resource_capabilities': [
                item for item in RESOURCE_ORDER if item in source.capabilities
            ],
        })
    return jsonify({'checked_at': check['checked_at'], 'sources': rows, 'scope_status': check['scopes'], 'public_unsafe_warning': actions.PUBLIC_UNSAFE_WARNING, 'restart': actions.restart_preflight()})


@bp.get('/api/r0/data-check')
def api_data_check():
    ensure_allowed_args(request.args, {'scope'})
    return jsonify(actions.data_check(request.args.get('scope')))


@bp.get('/api/r0/data-update-plan')
def api_data_update_plan():
    ensure_allowed_args(request.args, {'scopes', 'force'})
    return jsonify(actions.preview_update_plan(request.args.get('scopes'), request.args.get('force')))


@bp.post('/api/r0/actions/data-update')
def api_data_update():
    operation = actions.start_update(request.get_json(silent=True))
    body = operation.payload()
    return jsonify({key: body[key] for key in ('operation_id', 'status_url', 'plan_id', 'plan_hash', 'resolved_plan')}), 202


@bp.post('/api/r0/actions/cache-recover')
def api_cache_recover():
    operation, ready = actions.start_recovery(request.get_json(silent=True))
    if ready is not None:
        return jsonify(ready), 200
    body = operation.payload()
    return jsonify({key: body[key] for key in ('operation_id', 'status_url', 'plan_id', 'plan_hash')}), 202


@bp.get('/api/r0/actions/<operation_id>')
def api_action_status(operation_id: str):
    require_no_args(request.args)
    return jsonify(actions.runtime().operation(operation_id).payload())


@bp.post('/api/r0/actions/<operation_id>/retry')
def api_action_retry(operation_id: str):
    body = request.get_json(silent=True)
    if request.data and body is None:
        raise ApiError(400, 'invalid_params', 'Retry body must be empty.')
    operation = actions.retry(operation_id, body)
    payload = operation.payload()
    return jsonify({key: payload[key] for key in ('operation_id', 'status_url', 'plan_id', 'plan_hash', 'resolved_plan', 'retry_of')}), 202


@bp.post('/api/r0/actions/restart')
def api_restart():
    if request.data and request.get_json(silent=True) not in (None, {}):
        raise ApiError(400, 'invalid_params', 'Restart body must be empty.')
    actions.request_restart()


@bp.get('/api/r0/health')
def api_health():
    require_no_args(request.args)
    value = actions.runtime()
    return jsonify({'boot_id': value.boot_id, 'started_at': value.started_at, 'ready': value.ready, 'error': value.startup_error})


@bp.get('/api/r0/manual-records/capabilities')
def api_manual_capabilities():
    require_no_args(request.args)
    items = [
        {
            'source_id': source_id, 'strategy_id': strategy_id, 'display_name': display_name,
            'currency': currency, 'lot_size': lot_size, 'initial_capital': initial_capital,
            'signal_reference_link': f'/api/r0/manual-records/signal-reference?strategy={strategy_id}',
            'create_supported': True, 'reconcile_supported': True,
        }
        for source_id, strategy_id, display_name, currency, lot_size, initial_capital in MANUAL_CAPABILITIES
    ]
    return jsonify(bounded_envelope('manual_record_capabilities', items, 64))


@bp.get('/api/r0/manual-records')
def api_manual_records():
    ensure_allowed_args(request.args, {'strategy', 'cursor', 'limit'})
    strategy, _ = manual_ledger.require_strategy(request.args.get('strategy'))
    return jsonify(manual_ledger.list_records(strategy, cursor=request.args.get('cursor'), limit_raw=request.args.get('limit')))


@bp.get('/api/r0/manual-records/signal-reference')
def api_manual_signal_reference():
    ensure_allowed_args(request.args, {'strategy', 'snapshot_id'})
    strategy, _ = manual_ledger.require_strategy(request.args.get('strategy'))
    return jsonify(reconciliation.signal_reference(strategy, request.args.get('snapshot_id')))


@bp.get('/api/r0/manual-records/reconciliation')
def api_manual_reconciliation():
    ensure_allowed_args(request.args, {'strategy', 'snapshot_id', 'reconciliation_id', 'resource', 'row_id', 'cursor', 'limit'})
    # reconciliation.response preserves the required validation order.
    return jsonify(reconciliation.response(request.args.to_dict(flat=True)))


@bp.post('/api/r0/manual-records')
def api_manual_create():
    request_id = str(uuid4())
    with actions.runtime().manual_mutation(request_id):
        record, status = manual_ledger.create(request.get_json(silent=True), request.headers.get('Idempotency-Key'))
    return jsonify({'record': record} if status == 201 else {'record': {key: value for key, value in record.items() if key != 'replayed'}, 'replayed': True}), status


@bp.delete('/api/r0/manual-records/<record_id>')
def api_manual_delete(record_id: str):
    request_id = str(uuid4())
    with actions.runtime().manual_mutation(request_id):
        manual_ledger.delete(record_id)
    return '', 204
