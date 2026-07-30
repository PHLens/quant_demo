"""页面 blueprint：渲染所有静态模板。"""
from __future__ import annotations

from flask import Blueprint, render_template

bp = Blueprint('pages', __name__)


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/timing')
def timing_page():
    return render_template('timing.html')


@bp.route('/us_timing')
def us_timing_page():
    return render_template('us_timing.html')


@bp.route('/live')
def page_live():
    return render_template('live.html')


@bp.route('/commodity')
def commodity_page():
    return render_template('commodity.html')


@bp.route('/hk_timing')
def hk_timing_page():
    return render_template('hk_timing.html')


@bp.route('/learn')
def learn_page():
    return render_template('learn.html', page_key='learn')


@bp.route('/lab')
def lab_page():
    return render_template('lab.html', page_key='lab')


@bp.route('/compare')
def compare_page():
    return render_template('compare.html', page_key='compare')
