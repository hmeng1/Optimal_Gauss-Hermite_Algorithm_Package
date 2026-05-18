"""Shared utilities for loading and saving pickle-based caches."""
import pickle
import warnings


def load_pickle_cache(cache_file, default=None):
    """Load a pickle cache file, returning *default* on any failure.

    Parameters
    ----------
    cache_file : str
        Path to the pickle file.
    default : object
        Value to return if the file is missing or corrupted.

    Returns
    -------
    object
        The unpickled data, or *default*.
    """
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default
    except (OSError, pickle.UnpicklingError, EOFError) as e:
        warnings.warn(
            f"Cache file {cache_file} corrupted ({e}), ignoring.",
            RuntimeWarning,
        )
        return default


def save_pickle_cache(cache_file, data):
    """Save *data* to a pickle file, warning on failure.

    Parameters
    ----------
    cache_file : str
        Path to the pickle file.
    data : object
        Pickleable data to save.
    """
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=4)
    except (OSError, pickle.PickleError) as e:
        warnings.warn(
            f"Failed to save cache to {cache_file}: {e}",
            RuntimeWarning,
        )
