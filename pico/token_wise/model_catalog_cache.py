"""Compatibility alias for the historical TokenWise catalog module."""

from __future__ import annotations

import sys

from pico.call_efficiency import model_catalog_cache as _model_catalog_cache

sys.modules[__name__] = _model_catalog_cache
