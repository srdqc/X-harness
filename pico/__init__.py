"""
Pico is a compact, composable, benchmark-gated Agent Harness with an opt-in
evolution path.

Three feature pillars:
    1. Context Management   — context_engine/          (Curator engine)
    2. Call Efficiency      — call_efficiency/
    3. Skill Self-Evolution — memory_engine/skill_forge/

The public and internal Python namespace is ``pico``. See ``NOTICES.md`` and
``LICENSES/`` for license and third-party attribution details.
"""

import logging as _logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from pico.product import PRODUCT_LOGO


class _LiteLLMBotocorePreloadFilter(_logging.Filter):
    """Drop LiteLLM's bedrock/sagemaker `botocore`-missing pre-load warnings.

    LiteLLM emits these at import-time when ``botocore`` is not installed
    (``boto3`` is an optional extra in Pico; not in the default deps).
    The warnings are noise for users who don't target AWS Bedrock / SageMaker.
    If those providers are actually used, the downstream call will surface
    a clearer error than this pre-load chatter.
    """

    _patterns = (
        "could not pre-load bedrock-runtime response stream shape",
        "could not pre-load sagemaker-runtime response stream shape",
    )

    def filter(self, record: _logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(p in msg for p in self._patterns)


_logging.getLogger("LiteLLM").addFilter(_LiteLLMBotocorePreloadFilter())

try:
    __version__ = _pkg_version("pico-harness")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
__logo__ = PRODUCT_LOGO
