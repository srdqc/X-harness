"""Compatibility alias for historical TokenWise pricing imports."""

from __future__ import annotations

import sys

from pico.call_efficiency import pricing as _pricing

sys.modules[__name__] = _pricing
