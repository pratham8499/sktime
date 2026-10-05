"""Padding transformer, pad unequal length panel to max length or fixed length."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__all__ = ["PaddingTransformer"]
__author__ = ["abostrom"]


class PaddingTransformer(BaseTransformer):
    """Padding panel of unequal length time series to equal, fixed length.

    Pads the input dataset to either a optional fixed length
    (longer than the longest series).
    Or finds the max length series across all series and dimensions and
    pads to that with zeroes.

    Parameters
    ----------
    pad_length : int, optional (default=None) length to pad the series too.
        if None, will find the longest sequence and use instead.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.transformations.padder import PaddingTransformer
    >>>
    >>> # Create a sample nested DataFrame with unequal length time series
    >>> data = {
    ...     'feature1': [
    ...         pd.Series([1, 2, 3]), pd.Series([4, 5]), pd.Series([6, 7, 8, 9])
    ...     ],
    ...     'feature2': [
    ...         pd.Series([10, 11]), pd.Series([12, 13, 14]), pd.Series([15])
    ...     ]
    ... }
    >>> X = pd.DataFrame(data)
    >>>
    >>> # Initialize the PaddingTransformer
    >>> padder = PaddingTransformer()
    >>>
    >>> # Fit the transformer to the data
    >>> padder.fit(X)
    PaddingTransformer()
    >>>
    >>> # Transform the data
    >>> Xt = padder.transform(X)
    >>>
    >>> # Display the transformed data
    >>> # print(Xt)
    """

    _tags = {
        "authors": ["abostrom"],
        "maintainers": ["abostrom"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "pd-multiindex",  # which mtypes do _fit/_predict support?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,
        "capability:unequal_length:removes": True,
        "capability:categorical_in_X": False,
        # is transform result always guaranteed to be equal length (and series)?
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, pad_length=None, fill_value=0):
        self.pad_length = pad_length
        self.fill_value = fill_value
        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame with pd.MultiIndex
            Panel data in pd-multiindex format. MultiIndex has two levels:
            first level is instance index, second level is time index.
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        if self.pad_length is None:
            lengths = X.groupby(level=0).size()
            self.pad_length_ = lengths.max()
        else:
            self.pad_length_ = self.pad_length

        return self

    def _create_pad(self, series):
        out = np.full(self.pad_length_, self.fill_value, float)
        out[: len(series)] = series.iloc[: len(series)]
        return out

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame with pd.MultiIndex
            Panel data in pd-multiindex format. MultiIndex has two levels:
            first level is instance index, second level is time index.
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.DataFrame with pd.MultiIndex, same format as X
            transformed version of X
        """
        lengths = X.groupby(level=0).size()
        max_length = lengths.max()

        if max_length > self.pad_length_:
            raise ValueError(
                "Error: max_length of series \
                    is greater than the one found when fit or set."
            )

        instances = X.index.get_level_values(0).unique()

        result_frames = []
        for inst_id in instances:
            inst_data = X.loc[inst_id]
            transformed = {}
            for col in X.columns:
                series = inst_data[col].dropna()
                transformed[col] = self._create_pad(series)

            inst_df = pd.DataFrame(transformed)
            inst_df.index = pd.MultiIndex.from_arrays(
                [[inst_id] * self.pad_length_, range(self.pad_length_)],
                names=X.index.names,
            )
            result_frames.append(inst_df)

        Xt = pd.concat(result_frames)
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [
            {"pad_length": None, "fill_value": 0},
            {"pad_length": 120, "fill_value": -1.0},
        ]
        return params
