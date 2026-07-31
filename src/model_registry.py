# -*- coding: utf-8 -*-
"""Model registry for operational visibility inference.

Discovers, selects, and loads fitted models from a registry directory.
Model metadata is stored in model_registry.json; model files are loaded
via type-specific loaders.

Founded in 2026-07-16
Modified in 2026-07-31
@author: yinlb
"""

import copy
import json
import pathlib
import typing

import arrow

from src import pdf_model


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
            'PDF': self._load_pdf,
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
        init_time: typing.Optional[str] = None
    ) -> typing.Any:
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
        full_path = self._registry_dir / rel_path
        return self._loaders[model_type](full_path)

    def register_loader(
        self,
        model_type: str,
        loader: typing.Callable[[pathlib.Path], typing.Any]
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
        if target != 'latest':
            return target

        candidates = list(self._models.values())
        if not candidates:
            raise RuntimeError('No models registered in registry.')

        if init_time is not None:
            dt = arrow.get(init_time, 'YYYYMMDDHH')
            valid = [
                m for m in candidates
                if 'training_end' in m
                and arrow.get(m['training_end']) <= dt
            ]
            if not valid:
                raise RuntimeError(
                    f'No model available for init time {init_time}'
                )
            candidates = valid

        latest = max(
            candidates,
            key=lambda m: arrow.get(m.get('created_at', '1970-01-01'))
        )
        return latest['model_id']

    def _load_pdf(
        self, path: pathlib.Path
    ) -> typing.Union[pdf_model.PDF, typing.List[pdf_model.PDF]]:
        """Load a PDF model or a list of station-specific PDF models.

        Args:
            path: Path to the joblib-serialized model file.

        Returns:
            A PDF instance, or a list of PDF instances.

        Raises:
            TypeError: If the file contains neither PDF nor list[PDF].
        """
        model = pdf_model.load_pdf_file(path)
        if isinstance(model, pdf_model.PDF):
            return model
        if isinstance(model, list) and all(
            isinstance(m, pdf_model.PDF) for m in model
        ):
            return model
        raise TypeError(
            f'Expected PDF or list[PDF], got {type(model)}'
        )
