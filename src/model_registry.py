# -*- coding: utf-8 -*-
"""Model registry for operational visibility inference.

Discovers, selects, and loads fitted models from a registry directory.
Model metadata is stored in model_registry.json; model files are loaded
via type-specific loaders.

Founded in 2026-07-16
Modified in 2026-09-30
@author: yinlb, space-bunny
"""

import copy
import json
import pathlib
import typing

import arrow

from src import postprocess


class ModelRegistry:
    """Registry that manages available inference models.

    Attributes:
        _registry_dir: Directory containing model_registry.json and files.
        _default_model_id: Fallback model id when none is specified.
        _models: Dict mapping model_id to metadata dict.
        _loaders: Dict mapping model_type to loader callable.
    """

    def __init__(
        self,
        registry_dir: str,
        default_model_id: str = 'latest'
    ) -> None:
        """
        Initialize registry with directory and default model id.

        Args:
            registry_dir: Directory containing model_registry.json.
            default_model_id: Model id to use when none is provided.
        """
        self._registry_dir = pathlib.Path(registry_dir)
        self._default_model_id = default_model_id
        self._models: typing.Dict[str, typing.Dict] = {}
        self._loaders: typing.Dict[str, typing.Callable] = {
            # Both keys map to the same loader: registry files written
            # before the PDFM rename still carry model_type 'PDF'.
            'PDFM': self._load_pdfm,
            'PDF': self._load_pdfm,
        }
        self.discover()

    def discover(self) -> None:
        """Load model metadata from model_registry.json if it exists."""
        self._models = {}
        registry_path = self._registry_path()
        if not registry_path.exists():
            return
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Both shapes are accepted: a mapping keyed by model_id, and the
        # older list-of-entries form. The list form is converted in place.
        models = data.get('models', {})
        if isinstance(models, list):
            self._models = {entry['model_id']: entry for entry in models}
        else:
            self._models = models

    def list_models(self) -> typing.List[str]:
        """Return list of registered model ids."""
        return list(self._models.keys())

    def get_metadata(self, model_id: str) -> typing.Dict:
        """
        Return metadata for a registered model.

        Args:
            model_id: Registered model identifier.

        Returns:
            Metadata dictionary.

        Raises:
            KeyError: If model_id is not registered.
        """
        if model_id not in self._models:
            raise KeyError(
                f'Model \'{model_id}\' not found in registry. '
                f'Available: {self.list_models()}'
            )
        # Return a copy so callers cannot mutate the registry state.
        return self._models[model_id].copy()

    def select_model_id(
        self,
        model_id: typing.Optional[str] = None,
        init_time: typing.Optional[str] = None
    ) -> str:
        """
        Resolve model id based on explicit id, latest, or init time.

        Args:
            model_id: Explicit model id, or 'latest'.
            init_time: Optional YYYYMMDDHH string for time-based selection.

        Returns:
            Resolved model id string.
        """
        return self._resolve_model_id(model_id, init_time)

    def load_model(
        self,
        model_id: typing.Optional[str] = None,
        init_time: typing.Optional[str] = None,
    ) -> postprocess.PDFM:
        """
        Load and return a model instance.

        Args:
            model_id: Explicit model id, or 'latest'.
            init_time: Optional YYYYMMDDHH string for backfill selection.

        Returns:
            Loaded model instance.

        Raises:
            KeyError: If resolved model_id is not registered.
            ValueError: If model_type has no registered loader.
        """
        resolved_id = self._resolve_model_id(model_id, init_time)
        metadata = self.get_metadata(resolved_id)
        model_type = metadata.get('model_type')
        if model_type not in self._loaders:
            raise ValueError(
                f'No loader registered for model type \'{model_type}\''
            )
        rel_path = metadata.get('path', '')
        # Paths in the registry are relative so the registry stays portable
        # across machines and deployment directories.
        full_path = self._registry_dir / rel_path
        return self._loaders[model_type](full_path)

    def register_loader(
        self,
        model_type: str,
        loader: typing.Callable[[pathlib.Path], postprocess.PDFM],
    ) -> None:
        """
        Register a custom loader for a model type.

        Args:
            model_type: Model type identifier.
            loader: Callable that receives a pathlib.Path and returns a model.
        """
        self._loaders[model_type] = loader

    def copy(self) -> 'ModelRegistry':
        """Return a copy of the registry with the same configuration.

        Returns:
            ModelRegistry: A new instance with copied model metadata.
        """
        new = ModelRegistry(
            registry_dir=str(self._registry_dir),
            default_model_id=self._default_model_id
        )
        new._models = copy.deepcopy(self._models)
        return new

    def _registry_path(self) -> pathlib.Path:
        """Return path to model_registry.json."""
        return self._registry_dir / 'model_registry.json'

    def _resolve_model_id(
        self,
        model_id: typing.Optional[str],
        init_time: typing.Optional[str]
    ) -> str:
        """Resolve model id from explicit, latest, or time-based selection.

        Args:
            model_id: Explicit model id, or None to use the default.
            init_time: Optional YYYYMMDDHH string for time-based selection.

        Returns:
            Resolved model id string.

        Raises:
            RuntimeError: If no models are registered, or none is valid
                for the given init_time.
        """
        target = model_id if model_id is not None else self._default_model_id
        # An explicit id is taken as-is; only 'latest' needs resolution.
        if target != 'latest':
            return target

        candidates = list(self._models.values())
        if not candidates:
            raise RuntimeError('No models registered in registry.')

        if init_time is not None:
            # Backfill safety: only models whose training ended before the
            # init time are valid, otherwise the run leaks future information.
            dt = arrow.get(init_time, 'YYYYMMDDHH')
            valid = [
                model_meta for model_meta in candidates
                if 'training_end' in model_meta
                and arrow.get(model_meta['training_end']) <= dt
            ]
            if not valid:
                raise RuntimeError(
                    f'No model available for init time {init_time}'
                )
            candidates = valid

        latest = max(
            candidates,
            key=lambda model_meta: arrow.get(
                model_meta.get('created_at', '1970-01-01')
            )
        )
        return latest['model_id']

    def _load_pdfm(
        self, path: pathlib.Path
    ) -> typing.Union[
        postprocess.PDFM, typing.List[postprocess.PDFM]
    ]:
        """Load a PDFM model or a list of station-specific PDFM models.

        Args:
            path: Path to the joblib-serialized model file.

        Returns:
            A PDFM instance, or a list of PDFM instances.

        Raises:
            TypeError: If the file contains neither PDFM nor list[PDFM].
        """
        model = postprocess.load_pdfm_file(path)
        if isinstance(model, postprocess.PDFM):
            return model
        # Per-station models are stored as a plain list in one file.
        if isinstance(model, list) and all(
            isinstance(item, postprocess.PDFM) for item in model
        ):
            return model
        raise TypeError(
            f'Expected PDFM or list[PDFM], got {type(model)}'
        )
