# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""The AIE device catalog (aie_devices.json): hardware facts by part, shared by generation."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict

from .ir.context import DeviceSpec

_DEVICE_CATALOG: Dict[str, Any] | None = None


def load_device_catalog() -> Dict[str, Any]:
    """Return the cached device catalog loaded from aie_devices.json."""
    global _DEVICE_CATALOG
    if _DEVICE_CATALOG is None:
        _DEVICE_CATALOG = json.loads(Path(__file__).with_name('aie_devices.json').read_text())
    return _DEVICE_CATALOG


_RELEASE_SUFFIX = re.compile(r'_\d{6}_\d+$')


def lookup_device(part_name: str) -> Dict[str, Any]:
    """The facts of a Vitis platform or device part and its AIECompilerTarget; {} for a name the catalog lacks."""
    catalog = load_device_catalog()
    name = str(part_name).lower()
    for key in (name, _RELEASE_SUFFIX.sub('', name)):
        if key in catalog['platforms']:
            platform = dict(catalog['platforms'][key])
            return _facts(catalog, platform.pop('Part'), platform, 'platform')
        if key in catalog['parts']:
            return _facts(catalog, key, {}, 'part')
    return {}


def _facts(catalog: Dict[str, Any], part: str, platform: Dict[str, Any], target: str) -> Dict[str, Any]:
    facts = {'AIECompilerTarget': target, 'Part': part}
    entry = catalog['parts'][part]
    for layer in (catalog['generations'][entry['Generation']], entry, platform):
        for key, value in layer.items():
            if key in facts:
                raise ValueError(f'aie_devices.json gives {key!r} for part {part!r} more than once.')
            facts[key] = value
    return facts


PART_HELP = (
    'Pass a known Vitis platform name or raw AIE device part. Platform release suffixes must match '
    'the installed Vitis version. Known targets: {boards}.'
)


def known_boards() -> str:
    catalog = load_device_catalog()
    return ', '.join(sorted([*catalog['platforms'], *catalog['parts']]))


def installed_platforms() -> list[str]:
    """Platform names in the local Vitis install, or [] when Vitis is not reachable.

    Mirrors the generated Makefile, which takes VITIS_HOME as dirname(dirname(which vitis)).
    """

    vitis = shutil.which('vitis')
    root = Path(vitis).parent.parent if vitis else Path(os.environ.get('XILINX_VITIS', ''))
    base = root / 'base_platforms'
    return sorted(entry.name for entry in base.iterdir() if entry.is_dir()) if base.is_dir() else []


def resolve_device(part_name: Any, aie_cfg: Dict[str, Any]) -> tuple[DeviceSpec, Dict[str, Any]]:
    if 'BatchSize' in aie_cfg:
        raise ValueError(
            "AIEConfig.BatchSize is not read: the batch is the leading axis of the model's input (with hls4ml, "
            'convert_from_keras_model(batch_size=...)).'
        )
    entry = lookup_device(part_name)
    if not entry and 'Columns' not in aie_cfg:
        raise ValueError(f'Unknown part "{part_name}". {PART_HELP.format(boards=known_boards())}')
    merged = dict(entry)
    merged.update(aie_cfg)

    installed = installed_platforms()
    if merged.get('AIECompilerTarget') == 'platform' and installed and str(part_name) not in installed:
        # A platform named without its release suffix is the one release of it this Vitis installs.
        base = str(part_name).lower()
        releases = (
            []
            if _RELEASE_SUFFIX.search(base)
            else [name for name in installed if _RELEASE_SUFFIX.sub('', name.lower()) == base]
        )
        if len(releases) != 1:
            raise ValueError(
                f'Platform "{part_name}" is not in this Vitis install'
                + (f' and matches several releases ({", ".join(releases)}); name one.' if releases else '.')
                + f' Installed: {", ".join(installed)}.'
            )
        part_name = releases[0]

    return DeviceSpec.from_config(str(part_name), merged), merged
