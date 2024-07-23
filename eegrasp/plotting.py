"""Define default values and functions used for plotting function in eegrasp main module."""

DEFAULT_CMAP = 'Spectral'
DEFAULT_VERTEX_COLOR = 'teal'
DEFAULT_SPHERE = 'eeglab'
DEFAULT_ALPHAN = 0.5
DEFAULT_VERTEX_SIZE = 10
DEFAULT_POINTSIZE = 0.5
DEFAULT_LINEWIDTH = 1


def _update_locals(kwargs, local_vars):
    """Update local variables with the kwargs."""
    new_kwargs = kwargs.copy()
    for key in kwargs.keys():
        if key in local_vars:
            local_vars[key] = kwargs[key]
            del new_kwargs[key]
    return new_kwargs
