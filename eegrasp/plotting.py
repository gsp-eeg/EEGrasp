"""Define default values and functions used for plotting function in eegrasp main module."""

import dataclasses


@dataclasses.dataclass
class PlottingDefaults:
    """Class containing default values and functions."""
    DEFAULT_CMAP: str = 'Spectral'
    DEFAULT_VERTEX_COLOR: str = 'teal'
    DEFAULT_SPHERE: str = 'eeglab'
    DEFAULT_ALPHAN: float = 0.5
    DEFAULT_VERTEX_SIZE: float = 10.
    DEFAULT_POINTSIZE: float = 0.5
    DEFAULT_LINEWIDTH: float = 1.
    DEFAULT_EDGE_WIDTH: float = 2.
    DEFAULT_EDGE_COLOR: str = 'black'
    DEFAULT_ALPHAV: float = 1.

    def load_defaults(self, kwargs):
        """Return dictionary with added default values for plotting functions if parameters have not
        been set.

        Parameters
        ----------
        kwargs : dict.
            Dictionary containing the variables to be updated.

        Returns
        -------
        kwargs : dict.
            Dictionary with default values added.
        """
        for key, value in self.__dict__.items():
            if key not in kwargs.keys():
                true_key = key.lower().split('default_')[-1]
                kwargs[true_key] = value
        return kwargs


def _update_locals(kwargs, local_vars):
    """Update local variables with the kwargs dict.
    Parameters
    ----------
    kwargs : dict.
        Dictionary containing the variables to be updated.
    local_vars : list.
        List of local variables to be updated.
    """
    new_kwargs = kwargs.copy()
    for key in kwargs.keys():
        if key in local_vars:
            local_vars[key] = kwargs[key]
            del new_kwargs[key]
    return new_kwargs


def _separate_kwargs(kwargs, names):
    """Separate kwargs into two dictionaries based on names on vars."""
    var1 = {}
    var2 = {}
    for key in kwargs.keys():
        if key in names:
            var1[key] = kwargs[key]
        else:
            var2[key] = kwargs[key]
    return var1, var2


if __name__ == '__main__':
    defaults = PlottingDefaults()
    print(defaults.__dict__)
