"""Fixed v0.1 source, strategy, variant, recovery and Manual catalogs."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
from typing import Any

from web import state
from web.v01.contracts import ApiError, canonical_json, decimal_string, digest_id


RESOURCE_ORDER = (
    'summary', 'published_current_signal', 'series', 'signals', 'position',
    'interval_windows', 'holding_periods', 'holdings', 'trades', 'fees',
    'factors', 'configuration',
)
SCOPE_ORDER = ('index', 'aux', 'stock', 'factor')


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    display_name: str
    related_only: bool
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class StrategySpec:
    source_id: str
    strategy_id: str
    display_name: str
    defaults: dict[str, Any]
    capabilities: tuple[str, ...]
    recovery_supported: bool
    recovery_scopes: tuple[str, ...]
    eta_seconds: int | None


@dataclass(frozen=True)
class VariantSpec:
    source_id: str
    strategy_id: str
    variant_id: str
    canonical_params: dict[str, Any]
    param_schema: dict[str, Any]
    default: bool


SOURCE_SPECS = (
    SourceSpec('selection', 'A-share Selection', False, (
        'summary', 'series', 'interval_windows', 'holding_periods', 'holdings', 'fees', 'factors', 'configuration',
    )),
    SourceSpec('a_share_timing', 'A-share Timing', False, (
        'summary', 'published_current_signal', 'series', 'signals', 'position', 'interval_windows', 'trades', 'fees', 'configuration',
    )),
    SourceSpec('us_timing', 'US Timing', False, (
        'summary', 'published_current_signal', 'series', 'signals', 'position', 'interval_windows', 'trades', 'fees', 'configuration',
    )),
    SourceSpec('hk_timing', 'Hong Kong Timing', False, (
        'summary', 'published_current_signal', 'series', 'signals', 'position', 'interval_windows', 'trades', 'fees', 'configuration',
    )),
    SourceSpec('commodity', 'Commodity Timing', False, (
        'summary', 'published_current_signal', 'series', 'signals', 'position', 'interval_windows', 'trades', 'fees', 'configuration',
    )),
    SourceSpec('selection_factor', 'Selection Factor Artifacts', True, ('summary', 'series', 'factors', 'configuration')),
    SourceSpec('decision_context', 'Decision Context', True, ('summary', 'configuration')),
)
SOURCES = {item.source_id: item for item in SOURCE_SPECS}


def _class_display(registry: dict[str, Any], strategy_id: str) -> str:
    cls = registry.get(strategy_id)
    return getattr(cls, 'display_name', None) or strategy_id


def _selection_specs() -> list[StrategySpec]:
    order = ('original_ensemble', 'original', 'chan_enhanced', 'chan_only', 'method_a', 'quality_value', 'sector_heat')
    items = []
    for strategy_id in order:
        if strategy_id not in state.STRATEGY_MAP:
            continue
        defaults = dict(state._CACHE_DEFAULTS.get(strategy_id, {}))
        items.append(StrategySpec(
            'selection', strategy_id, _class_display(state.STRATEGY_MAP, strategy_id), defaults,
            SOURCES['selection'].capabilities, True, ('stock', 'index'), 180,
        ))
    return items


def _timing_specs(source_id: str, registry: dict[str, Any], defaults_map: dict[str, dict[str, Any]], order: tuple[str, ...], scopes: tuple[str, ...]) -> list[StrategySpec]:
    return [
        StrategySpec(
            source_id, strategy_id, _class_display(registry, strategy_id),
            dict(defaults_map.get(strategy_id, {})), SOURCES[source_id].capabilities,
            True, scopes, 150,
        )
        for strategy_id in order if strategy_id in registry
    ]


STRATEGY_SPECS = tuple(
    _selection_specs()
    + _timing_specs('a_share_timing', state.TIMING_STRATEGY_MAP, state._TIMING_CACHE_DEFAULTS,
                    ('csi1000_timing', 'star50_timing', 'chinext_timing'), ('index',))
    + _timing_specs('us_timing', state.US_TIMING_STRATEGY_MAP, state._US_TIMING_CACHE_DEFAULTS,
                    ('macro_v32_timing', 'sp500_timing'), ('index', 'aux'))
    + _timing_specs('hk_timing', state.HK_STRATEGY_MAP, {}, ('hsi_timing', 'hstech_timing'), ('index',))
    + _timing_specs('commodity', state.COMMODITY_STRATEGY_MAP, {}, ('gold_timing',), ('index',))
    + [
        StrategySpec('selection_factor', 'sector_heat', 'Sector Heat', {}, SOURCES['selection_factor'].capabilities, True, ('stock', 'factor'), 120),
        StrategySpec('selection_factor', 'single_factor', 'Single Factor', {'top_k': 5}, SOURCES['selection_factor'].capabilities, True, ('stock',), 180),
        StrategySpec('decision_context', 'risk_signals', 'Risk Signals', {}, SOURCES['decision_context'].capabilities, False, (), None),
    ]
)
STRATEGIES = {(item.source_id, item.strategy_id): item for item in STRATEGY_SPECS}
_DECLARED_METADATA: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}


def get_source(source_id: str) -> SourceSpec:
    item = SOURCES.get(source_id)
    if item is None:
        raise ApiError(404, 'source_not_found', 'Unknown source.', source_id=source_id)
    return item


def get_strategy(source_id: str, strategy_id: str) -> StrategySpec:
    get_source(source_id)
    item = STRATEGIES.get((source_id, strategy_id))
    if item is None:
        raise ApiError(404, 'strategy_not_found', 'Unknown strategy for this source.', source_id=source_id, strategy_id=strategy_id)
    return item


def _metadata_schema(source_id: str, strategy_id: str, defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return declaration metadata cached at module/app startup, never on GET."""
    cache_key = (source_id, strategy_id)
    if cache_key in _DECLARED_METADATA:
        return {key: value for key, value in _DECLARED_METADATA[cache_key].items() if key in defaults}
    registry = {
        'selection': state.STRATEGY_MAP,
        'a_share_timing': state.TIMING_STRATEGY_MAP,
        'us_timing': state.US_TIMING_STRATEGY_MAP,
        'hk_timing': state.HK_STRATEGY_MAP,
        'commodity': state.COMMODITY_STRATEGY_MAP,
    }.get(source_id, {})
    definitions: dict[str, dict[str, Any]] = {}
    cls = registry.get(strategy_id)
    if cls is not None:
        try:
            instance = cls()
            rows = instance.get_parameter_definitions()
            if source_id != 'selection' and hasattr(instance, 'get_shared_parameter_definitions'):
                rows = list(rows) + list(instance.get_shared_parameter_definitions())
            for row in rows or []:
                if isinstance(row, dict) and row.get('key') in defaults:
                    definitions[row['key']] = row
        except Exception:
            definitions = {}
    _DECLARED_METADATA[cache_key] = definitions
    return {key: value for key, value in definitions.items() if key in defaults}


def _field_schema(source_id: str, strategy_id: str, defaults: dict[str, Any]) -> dict[str, Any]:
    declared = _metadata_schema(source_id, strategy_id, defaults)
    fields = {}
    for key in sorted(defaults):
        default = defaults[key]
        row = declared.get(key, {})
        if isinstance(default, bool):
            field_type = 'boolean'
            step = None
        elif isinstance(default, int):
            field_type = 'integer'
            step = int(row.get('step') or 1)
        elif isinstance(default, float):
            field_type = 'decimal'
            raw_step = row.get('step')
            if raw_step is None:
                exponent = max(1, -Decimal(str(default)).as_tuple().exponent)
                raw_step = Decimal(1).scaleb(-exponent)
            step = str(raw_step)
        elif row.get('enum_values') or row.get('options'):
            field_type = 'enum'
            step = None
        else:
            field_type = 'string'
            step = None
        enum_values = row.get('enum_values') or row.get('options')
        if enum_values and isinstance(enum_values, list) and enum_values and isinstance(enum_values[0], dict):
            enum_values = [item.get('value') for item in enum_values]
        normalized_row = dict(row)
        normalized_row['enum_values'] = enum_values
        normalized_row.pop('options', None)
        fields[key] = {
            'type': field_type,
            'required': False,
            'default': _canonical_param(default, field_type, step, normalized_row),
            'minimum': row.get('min'),
            'maximum': row.get('max'),
            'step': step,
            'enum_values': enum_values,
        }
    return {'version': 'v0.1', 'fields': fields}


def _canonical_param(value: Any, field_type: str, step: Any, row: dict[str, Any]) -> Any:
    if field_type == 'boolean':
        if isinstance(value, bool):
            return value
        raw = str(value).lower()
        if raw not in {'true', 'false'}:
            raise ApiError(400, 'invalid_params', 'Boolean parameters require true or false.')
        return raw == 'true'
    if field_type == 'integer':
        try:
            if isinstance(value, bool) or str(int(value)) != str(value).strip():
                raise ValueError
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, 'invalid_params', 'Invalid integer parameter.') from exc
    elif field_type == 'decimal':
        result = decimal_string(value, step)
    else:
        result = str(value)
        enum_values = row.get('enum_values') or row.get('options')
        if enum_values and result not in enum_values:
            raise ApiError(400, 'invalid_params', 'Invalid enum parameter.')
    if field_type in {'integer', 'decimal'}:
        number = Decimal(str(result))
        minimum = row.get('min')
        maximum = row.get('max')
        if minimum is not None and number < Decimal(str(minimum)):
            raise ApiError(400, 'invalid_params', 'Parameter is below its minimum.')
        if maximum is not None and number > Decimal(str(maximum)):
            raise ApiError(400, 'invalid_params', 'Parameter exceeds its maximum.')
    return result


def canonicalize_params(spec: StrategySpec, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = dict(spec.defaults)
    if spec.source_id == 'selection_factor' and spec.strategy_id == 'single_factor':
        defaults = {'top_k': 5}
    schema = _field_schema(spec.source_id, spec.strategy_id, defaults)
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise ApiError(400, 'invalid_params', 'Unknown strategy parameters.', invalid_params=unknown)
    canonical = {}
    for key in sorted(defaults):
        field = schema['fields'][key]
        canonical[key] = _canonical_param(raw.get(key, defaults[key]), field['type'], field['step'], {
            'min': field['minimum'], 'max': field['maximum'], 'enum_values': field['enum_values'],
        })
    return canonical, schema


def _variant(source_id: str, strategy_id: str, params: dict[str, Any], schema: dict[str, Any], *, default: bool) -> VariantSpec:
    variant_id = digest_id('v', {
        'source_id': source_id,
        'strategy_id': strategy_id,
        'param_schema_version': schema['version'],
        'canonical_params': params,
    })
    return VariantSpec(source_id, strategy_id, variant_id, params, schema, default)


def variants_for(spec: StrategySpec) -> tuple[VariantSpec, ...]:
    if spec.source_id == 'selection_factor' and spec.strategy_id == 'single_factor':
        variants = []
        for top_k in (3, 5, 8, 10):
            canonical, schema = canonicalize_params(spec, {'top_k': top_k})
            variants.append(_variant(spec.source_id, spec.strategy_id, canonical, schema, default=top_k == 5))
        return tuple(variants)
    canonical, schema = canonicalize_params(spec, {})
    return (_variant(spec.source_id, spec.strategy_id, canonical, schema, default=True),)


def find_variant(source_id: str, strategy_id: str, variant_id: str) -> VariantSpec:
    spec = get_strategy(source_id, strategy_id)
    for item in variants_for(spec):
        if item.variant_id == variant_id:
            return item
    raise ApiError(404, 'variant_not_cataloged', 'Variant is not in the fixed catalog.', variant_id=variant_id, recoverable=False)


def recovery_plan_id(spec: StrategySpec, variant: VariantSpec) -> str:
    """Stable fixed-recipe identity, independent of one admission's pre-state."""
    recipe = {
        'plan_version': 'v0.1',
        'target': {
            'source_id': spec.source_id, 'strategy_id': spec.strategy_id,
            'variant_id': variant.variant_id,
        },
        'recovery_scopes': list(spec.recovery_scopes),
        'builder': 'build-target',
    }
    digest = hashlib.sha256(canonical_json(recipe).encode('utf-8')).hexdigest()
    return f'recovery-v0.1-{digest[:16]}'


def lookup_variant(source_id: str, strategy_id: str, raw: dict[str, Any]) -> VariantSpec:
    spec = get_strategy(source_id, strategy_id)
    canonical, _ = canonicalize_params(spec, raw)
    key = canonical_json(canonical)
    for item in variants_for(spec):
        if canonical_json(item.canonical_params) == key:
            return item
    raise ApiError(404, 'variant_not_cataloged', 'Parameter combination is not in the fixed catalog.', recoverable=False)


MANUAL_CAPABILITIES = (
    ('a_share_timing', 'csi1000_timing', '中证1000', 'CNY', 100, 50000.0),
    ('a_share_timing', 'star50_timing', '科创50', 'CNY', 100, 50000.0),
    ('a_share_timing', 'chinext_timing', '创业板', 'CNY', 100, 50000.0),
    ('us_timing', 'macro_v32_timing', 'Macro v3.2', 'USD', 1, 50000.0),
    ('us_timing', 'sp500_timing', 'S&P 500', 'USD', 1, 50000.0),
)


def manual_capability(strategy_id: str):
    for item in MANUAL_CAPABILITIES:
        if item[1] == strategy_id:
            return item
    return None


# Freeze declaration-only parameter metadata during process initialization so
# every catalog GET is a pure lookup and never constructs a strategy object.
for _spec in STRATEGY_SPECS:
    _metadata_schema(_spec.source_id, _spec.strategy_id, _spec.defaults)
