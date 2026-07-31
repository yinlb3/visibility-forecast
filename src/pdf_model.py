#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF matching model for visibility forecast correction.

Founded in 2024-03-24
Modified in 2026-07-31
@author: yinlb
"""
import pathlib
import sys
import typing

import joblib
import numpy as np


class PDF:
    """PDF matching model for visibility forecast correction."""

    def __init__(self):
        """Initialize empty matching curve."""
        self.c = None

    @classmethod
    def load(cls, path: typing.Union[str, pathlib.Path]) -> 'PDF':
        """Load a fitted PDF model from a joblib-serialized file.

        Args:
            path: Path to the .dat file saved by joblib.dump().

        Returns:
            PDF instance with fitted matching curve.

        Raises:
            TypeError: If the loaded object is not a PDF instance.
        """
        model = load_pdf_file(path)
        if not isinstance(model, cls):
            raise TypeError(
                f'Expected {cls.__name__} instance, got {type(model).__name__}'
            )
        return model

    def fit(self, ob: np.ndarray, pr: np.ndarray):
        """Fit PDF matching curve from obs and forecast samples.

        Args:
            ob: Observation array.
            pr: Forecast array.
        """
        ob = ob[~np.isnan(ob)][::10000]
        pr = pr[~np.isnan(pr)][::10000]
        a = np.unique(ob)
        a = np.sort(a)
        self.c = np.zeros((len(a), 2), dtype=np.float32)
        self.c[:, 0] = a
        pr = np.sort(pr)
        for i, a0 in enumerate(a):
            p0 = np.mean(ob <= a0)
            j = round(p0 * (len(pr) - 1))
            self.c[i, 1] = pr[j]

    def predict(self, pr0: np.ndarray) -> typing.Optional[np.ndarray]:
        """Apply PDF matching correction to forecast array using np.interp.

        Args:
            pr0: Raw forecast array.

        Returns:
            Corrected forecast array, or None if model is not fitted.
        """
        if self.c is None:
            return None
        shape = pr0.shape
        pr0 = pr0.flatten().astype(np.float32)
        out = np.zeros_like(pr0) + np.nan
        valid = ~np.isnan(pr0)
        out[valid] = np.interp(pr0[valid], self.c[:, 1], self.c[:, 0])
        return out.reshape(shape)


def load_pdf_file(
    path: typing.Union[str, pathlib.Path]
) -> typing.Union[PDF, typing.List[PDF]]:
    """Load a joblib file with __main__.PDF pickle compatibility.

    Model files saved before the PDF class moved into this module
    reference __main__.PDF in their pickles; alias it so they load.

    Args:
        path: Path to the joblib-serialized file.

    Returns:
        The unpickled PDF or list[PDF] object.
    """
    main_mod = sys.modules.get('__main__')
    if main_mod is not None and not hasattr(main_mod, 'PDF'):
        main_mod.PDF = PDF
    return joblib.load(str(path))
