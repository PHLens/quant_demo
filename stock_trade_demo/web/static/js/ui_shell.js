(function () {
  'use strict';

  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
  let activeDrawer = null;
  let returnFocus = null;

  function setExpanded(selector, value) {
    document.querySelectorAll(selector).forEach((button) => {
      button.setAttribute('aria-expanded', String(value));
    });
  }

  function openDrawer(layer, drawer, trigger, expandedSelector) {
    if (!layer || !drawer) return;
    if (activeDrawer) closeActiveDrawer();
    returnFocus = trigger || document.activeElement;
    layer.hidden = false;
    document.body.classList.add('qr-drawer-open');
    setExpanded(expandedSelector, true);
    activeDrawer = { layer, drawer, expandedSelector };
    const firstTarget = drawer.querySelector('[data-guide-close], [data-mobile-nav-close], ' + FOCUSABLE);
    (firstTarget || drawer).focus();
  }

  function closeActiveDrawer() {
    if (!activeDrawer) return;
    activeDrawer.layer.hidden = true;
    document.body.classList.remove('qr-drawer-open');
    setExpanded(activeDrawer.expandedSelector, false);
    activeDrawer = null;
    if (returnFocus && typeof returnFocus.focus === 'function') returnFocus.focus();
    returnFocus = null;
  }

  function keepFocusInside(event) {
    if (!activeDrawer || event.key !== 'Tab') return;
    const targets = Array.from(activeDrawer.drawer.querySelectorAll(FOCUSABLE)).filter((node) => !node.hidden);
    if (!targets.length) {
      event.preventDefault();
      activeDrawer.drawer.focus();
      return;
    }
    const first = targets[0];
    const last = targets[targets.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function init() {
    const guideLayer = document.querySelector('[data-guide-layer]');
    const guideDrawer = document.getElementById('new-user-guide');
    document.querySelectorAll('[data-guide-open]').forEach((button) => {
      button.addEventListener('click', () => openDrawer(guideLayer, guideDrawer, button, '[data-guide-open]'));
    });
    document.querySelectorAll('[data-guide-close]').forEach((button) => button.addEventListener('click', closeActiveDrawer));

    const navLayer = document.querySelector('[data-mobile-nav-layer]');
    const navDrawer = document.getElementById('mobile-nav-drawer');
    document.querySelectorAll('[data-mobile-nav-open]').forEach((button) => {
      button.addEventListener('click', () => openDrawer(navLayer, navDrawer, button, '[data-mobile-nav-open]'));
    });
    document.querySelectorAll('[data-mobile-nav-close]').forEach((button) => button.addEventListener('click', closeActiveDrawer));

    document.addEventListener('keydown', (event) => {
      if (!activeDrawer) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        closeActiveDrawer();
        return;
      }
      keepFocusInside(event);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
