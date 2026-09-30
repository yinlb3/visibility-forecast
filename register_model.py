#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Register a trained PDFM model into the operational model registry.

Copies the .dat model file into the registry directory and upserts its
metadata entry (model_id, training_end, created_at, station_ids) into
model_registry.json, so hand-writing 1000+ station ids is avoided and
only a final human review is needed.

Usage:
    python register_model.py <dat_path> <training_end> [model_id] [registry_dir]

Founded in 2026-07-31
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import json
import os
import pathlib
import shutil
import sys
import typing

import arrow

from src import logger, model_registry, near_map, postprocess, utils


def _parse_args(argv: typing.List[str]) -> typing.Tuple:
    """Parse and validate register_model command-line arguments.

    Args:
        argv: sys.argv list.

    Returns:
        Tuple of (dat_path, training_end, model_id, registry_dir);
        optional items are None when omitted.

    Raises:
        SystemExit: If argument count is not between 2 and 4.
    """
    args = tuple(argv[1:])
    if not 2 <= len(args) <= 4:
        print(
            'Usage: python register_model.py <dat_path> <training_end> '
            '[model_id] [registry_dir]'
        )
        raise SystemExit(1)
    return args + (None,) * (4 - len(args))


def _inspect_model(dat_path: pathlib.Path) -> int:
    """Load a .dat file and return the number of PDFM models inside.

    Args:
        dat_path: Path to the joblib-serialized model file.

    Returns:
        int: 1 for a single PDFM, list length for per-station models.

    Raises:
        TypeError: If the file contains neither PDFM nor list[PDFM].
    """
    model = postprocess.load_pdfm_file(dat_path)
    if isinstance(model, postprocess.PDFM):
        return 1
    if isinstance(model, list) and all(
        isinstance(m, postprocess.PDFM) for m in model
    ):
        return len(model)
    raise TypeError(f'Expected PDFM or list[PDFM], got {type(model)}')


def _resolve_registry_dir(
    dir_arg: typing.Optional[str],
    cfg: typing.Dict
) -> pathlib.Path:
    """Resolve registry directory from argument or configuration.

    Args:
        dir_arg: Optional explicit directory from the command line.
        cfg: Merged configuration dictionary.

    Returns:
        pathlib.Path: Target registry directory.

    Raises:
        ValueError: If the resolved directory is still a placeholder.
    """
    dir_str = dir_arg or cfg['operational']['paths']['model_registry_dir']
    if '<' in dir_str:
        raise ValueError(
            f'model_registry_dir is a placeholder: {dir_str}. '
            f'Set it in config.local.*.yaml or pass registry_dir.'
        )
    return pathlib.Path(dir_str)


def _upsert_entry(json_path: pathlib.Path, entry: typing.Dict) -> None:
    """Insert or replace a model entry in model_registry.json.

    Args:
        json_path: Path to model_registry.json.
        entry: Model metadata entry to write.
    """
    data = {'models': []}
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    models = data.get('models', [])
    if isinstance(models, dict):
        models = list(models.values())
    replaced = any(
        m.get('model_id') == entry['model_id'] for m in models
    )
    models = [
        m for m in models if m.get('model_id') != entry['model_id']
    ]
    models.append(entry)
    data['models'] = models
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    action = 'Updated' if replaced else 'Added'
    logger.info(f'{action} registry entry in {json_path}')


def main(args: typing.Optional[typing.Tuple] = None) -> None:
    """
    Main entry: register one trained model into the registry.

    Args:
        args: Optional parsed arguments; defaults to command-line args.
    """
    # 1. Parse arguments and load configuration
    if args is None:
        args = _parse_args(sys.argv)
    dat_arg, tend_arg, model_id, dir_arg = args
    cfg = utils.load_config(operational=True)
    op_cfg = cfg['operational']

    # 2. Setup logger and validate the model file
    logger.setup_logger(
        log_dir=op_cfg['paths']['log_dir'],
        log_level=op_cfg['output']['log_level'],
        run_id=arrow.now().format('YYYYMMDDHHmmss')
    )
    dat_path = pathlib.Path(dat_arg)
    if not dat_path.exists():
        raise FileNotFoundError(f'Model file not found: {dat_path}')
    n_models = _inspect_model(dat_path)
    logger.info(f'Model file {dat_path}: {n_models} PDFM model(s)')

    # 3. Build station_ids and check the model/station count
    sta = near_map.load_stations(cfg)
    station_ids = [int(sid) for sid in sta['id'].values]
    if n_models > 1 and n_models != len(station_ids):
        raise ValueError(
            f'Model count {n_models} != station count '
            f'{len(station_ids)}. Check that region/station_csv '
            f'matches the training station set.'
        )

    # 4. Copy the model file into the registry directory
    reg_dir = _resolve_registry_dir(dir_arg, cfg)
    os.makedirs(str(reg_dir), exist_ok=True)
    dst_path = reg_dir / dat_path.name
    if dat_path.resolve() != dst_path.resolve():
        shutil.copy2(str(dat_path), str(dst_path))
    logger.info(f'Model file deployed to {dst_path}')

    # 5. Upsert the registry entry
    tend = arrow.get(tend_arg, ['YYYYMMDD', 'YYYYMMDDHH'])
    if model_id is None:
        model_id = f'{dat_path.stem}_tend{tend.format("YYYYMMDD")}'
    entry = {
        'model_id': model_id,
        'model_type': 'PDFM',
        'path': dat_path.name,
        'training_end': tend.isoformat(),
        'created_at': arrow.now().isoformat(),
        'station_ids': station_ids,
    }
    json_path = reg_dir / 'model_registry.json'
    _upsert_entry(json_path, entry)

    # 6. Print review summary for human check
    reg = model_registry.ModelRegistry(
        registry_dir=str(reg_dir),
        default_model_id=op_cfg['model']['default_model_id']
    )
    logger.info(
        f'Registered {model_id}: {n_models} model(s), '
        f'{len(station_ids)} stations '
        f'({station_ids[:3]} ... {station_ids[-3:]}), '
        f'training_end {tend.format("YYYY-MM-DD")}'
    )
    logger.info(f'Registry now holds: {reg.list_models()}')


if __name__ == '__main__':
    print('Program register_model.py started')
    total_start = arrow.now()

    parsed = _parse_args(sys.argv)
    main(args=parsed)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        f'Program register_model.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
