"""Multi-source plugin discovery.

A discovery pass scans every source the host knows about (bundled
sub-tree, user-level dir, project-level dir, pip entry points),
deduplicates by plugin id, and returns a stable list of
:class:`DiscoveredPlugin` records. Discovery only reads manifests — no
plugin Python code is imported here. The :class:`Source` enum doubles
as the conflict-resolution priority ordering (higher wins).

Priority order (`bundled > user > project > entry_points`) follows the
design's "builtin shadow rule": a bundled plugin can never be shadowed
by a same-named local or pip-installed one. Among the non-bundled
sources, user-level wins so a developer can substitute a locally edited
copy for a pip-installed version while iterating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from importlib import metadata
from pathlib import Path, PurePosixPath

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from pico import __version__ as _installed_pico_version
from pico.plugin.manifest import PluginManifest

logger = logging.getLogger(__name__)


_MANIFEST_FILENAME = "pico-plugin.toml"


class PluginCompatibilityError(RuntimeError):
    """A plugin manifest excludes the installed Pico version."""


class PluginIdentityError(RuntimeError):
    """Installed distribution metadata disagrees with its plugin manifest."""


class Source(IntEnum):
    """Where a manifest came from. Numeric value is conflict priority —
    higher wins. Lower values lose silently and are logged."""

    ENTRY_POINTS = 1
    PROJECT = 2
    USER = 3
    BUNDLED = 4


@dataclass(frozen=True)
class DiscoveredPlugin:
    """A manifest read from a specific source, awaiting activation."""

    manifest: PluginManifest
    source: Source
    location: Path | None
    """Path to the manifest file. ``None`` for entry-points-discovered
    plugins where the manifest lives inside a wheel's package data."""


class PluginDiscovery:
    """Scans every configured source and returns deduplicated plugins.

    Constructor params default to "off"; callers pass concrete paths
    or the entry-point group name to opt each source in. This keeps
    tests hermetic — they construct a discovery instance pointing at
    tmp dirs and don't accidentally pick up real plugins on the
    developer's machine.
    """

    def __init__(
        self,
        *,
        bundled_dir: Path | None = None,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
        entry_points_group: str | None = None,
        pico_version: str = _installed_pico_version,
    ) -> None:
        self._bundled_dir = bundled_dir
        self._user_dir = user_dir
        self._project_dir = project_dir
        self._entry_points_group = entry_points_group
        self._pico_version = pico_version

    def discover(self) -> list[DiscoveredPlugin]:
        """Run all enabled sources and resolve conflicts.

        The returned list is stable-ordered by plugin id so callers can
        log / display it deterministically.
        """
        all_found: list[DiscoveredPlugin] = []
        if self._bundled_dir is not None:
            all_found.extend(
                self._scan_dir(self._bundled_dir, Source.BUNDLED),
            )
        if self._user_dir is not None:
            all_found.extend(self._scan_dir(self._user_dir, Source.USER))
        if self._project_dir is not None:
            all_found.extend(
                self._scan_dir(self._project_dir, Source.PROJECT),
            )
        if self._entry_points_group is not None:
            all_found.extend(self._scan_entry_points(self._entry_points_group))

        return self._resolve_conflicts(all_found)

    # ── 基于文件的来源 ───────────────────────────────────────────

    def _scan_dir(
        self,
        root: Path,
        source: Source,
    ) -> list[DiscoveredPlugin]:
        """Look for ``<root>/<plugin_id>/pico-plugin.toml``.

        Subdir name is informational only — the canonical plugin id is
        the one inside the manifest. A mismatch is logged but the
        manifest still loads.
        """
        out: list[DiscoveredPlugin] = []
        if not root.is_dir():
            return out
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            manifest_path = sub / _MANIFEST_FILENAME
            if not manifest_path.is_file():
                continue
            try:
                mf = PluginManifest.from_toml_path(manifest_path)
            except Exception as e:
                logger.warning(
                    "plugin manifest %s failed to parse (%s); skipping",
                    manifest_path,
                    e,
                )
                continue
            self._validate_compatibility(mf)
            if mf.id != sub.name:
                logger.info(
                    "plugin %s lives in directory %s — id and dir name differ",
                    mf.id,
                    sub.name,
                )
            out.append(
                DiscoveredPlugin(
                    manifest=mf,
                    source=source,
                    location=manifest_path,
                ),
            )
        return out

    # ── 入口点来源 ───────────────────────────────────────────────

    def _scan_entry_points(self, group: str) -> list[DiscoveredPlugin]:
        """Resolve every entry point in ``group`` and read the manifest
        shipped inside that entry point's package.

        Entry-point value names the package that owns ``pico-plugin.toml``.
        The manifest is located through the owning distribution's installed
        file inventory, so discovery does not import the plugin package.
        """
        out: list[DiscoveredPlugin] = []
        try:
            eps = metadata.entry_points(group=group)
        except Exception as e:
            logger.warning("entry_points discovery failed (%s); skipping", e)
            return out

        for ep in eps:
            package_name = ep.value.split(":", 1)[0]
            try:
                manifest_path = self._entry_point_manifest_path(ep, package_name)
                if manifest_path is None:
                    logger.warning(
                        "entry-point %s points at package %s but no %s found; skipping",
                        ep.name,
                        package_name,
                        _MANIFEST_FILENAME,
                    )
                    continue
                mf = PluginManifest.from_toml_path(manifest_path)
            except Exception as e:
                if isinstance(e, (PluginCompatibilityError, PluginIdentityError)):
                    raise
                logger.warning(
                    "failed to load manifest for entry-point %s (%s); skipping",
                    ep.name,
                    e,
                )
                continue
            self._validate_entry_point_identity(ep, mf)
            self._validate_compatibility(mf)
            out.append(
                DiscoveredPlugin(
                    manifest=mf,
                    source=Source.ENTRY_POINTS,
                    location=None,
                ),
            )
        return out

    @staticmethod
    def _entry_point_manifest_path(ep, package_name: str) -> Path | None:
        distribution = getattr(ep, "dist", None)
        distribution_files = getattr(distribution, "files", None)
        if distribution is None or distribution_files is None:
            raise PluginIdentityError(
                f"entry-point {ep.name!r} has no owning distribution file inventory; reinstall the plugin"
            )
        expected = PurePosixPath(*package_name.split("."), _MANIFEST_FILENAME)
        for item in distribution_files:
            if PurePosixPath(str(item)) == expected:
                path = Path(distribution.locate_file(item))
                return path if path.is_file() else None
        return None

    @staticmethod
    def _validate_entry_point_identity(ep, manifest: PluginManifest) -> None:
        distribution = getattr(ep, "dist", None)
        if distribution is None:
            raise PluginIdentityError(
                f"entry-point {ep.name!r} has no owning distribution metadata; reinstall the plugin"
            )
        distribution_name = getattr(distribution, "name", None)
        distribution_version = getattr(distribution, "version", None)
        if not distribution_name or canonicalize_name(distribution_name) != canonicalize_name(manifest.id):
            raise PluginIdentityError(
                f"entry-point {ep.name!r} distribution {distribution_name!r} does not match "
                f"manifest id {manifest.id!r}; reinstall the plugin from its official distribution"
            )
        try:
            version_matches = Version(str(distribution_version)) == Version(manifest.version)
        except InvalidVersion as exc:
            raise PluginIdentityError(
                f"entry-point {ep.name!r} has invalid distribution or manifest version metadata"
            ) from exc
        if not version_matches:
            raise PluginIdentityError(
                f"entry-point {ep.name!r} distribution version {distribution_version} does not match "
                f"manifest version {manifest.version}; reinstall the plugin"
            )

    def _validate_compatibility(self, manifest: PluginManifest) -> None:
        if not manifest.pico:
            return
        try:
            compatible = Version(self._pico_version) in SpecifierSet(manifest.pico)
        except (InvalidSpecifier, InvalidVersion) as exc:
            raise PluginCompatibilityError(
                f"plugin {manifest.id!r} has invalid Pico compatibility metadata; reinstall a compatible plugin"
            ) from exc
        if not compatible:
            raise PluginCompatibilityError(
                f"plugin {manifest.id!r} {manifest.version} requires Pico {manifest.pico}, but installed Pico is "
                f"{self._pico_version}; install a compatible plugin version or set memory.backend to null"
            )

    # ── 冲突解决 ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_conflicts(
        found: list[DiscoveredPlugin],
    ) -> list[DiscoveredPlugin]:
        """Group by plugin id, keep the highest-priority source.

        Lower-priority duplicates are logged once each so a misconfigured
        setup is debuggable without silently dropping plugins.
        """
        by_id: dict[str, DiscoveredPlugin] = {}
        for d in found:
            current = by_id.get(d.manifest.id)
            if current is None or d.source > current.source:
                if current is not None:
                    logger.info(
                        "plugin %s: %s shadows %s",
                        d.manifest.id,
                        d.source.name,
                        current.source.name,
                    )
                by_id[d.manifest.id] = d
            elif d.source < current.source:
                logger.info(
                    "plugin %s: %s shadowed by %s",
                    d.manifest.id,
                    d.source.name,
                    current.source.name,
                )
        # 按 id 稳定排序，确保调用方展示顺序确定。
        return sorted(by_id.values(), key=lambda p: p.manifest.id)


__all__ = [
    "DiscoveredPlugin",
    "PluginCompatibilityError",
    "PluginDiscovery",
    "PluginIdentityError",
    "Source",
]
