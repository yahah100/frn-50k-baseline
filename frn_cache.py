"""On-disk cache for the FreshRetailNet-50K splits.

`datasets.load_dataset("Dingdong-Inc/FreshRetailNet-50K")` already caches the *download*
under the HuggingFace datasets cache. The slow part of every run is the next step: each
script calls `.to_pandas()`, which re-materialises ~2 GB of Arrow into a pandas DataFrame
on every invocation.

`load_frn(split)` caches that pandas result to a local parquet file the first time and reads
it back directly afterwards, so repeated runs skip the Arrow->pandas conversion.

The cache lives next to this file (``.frn_cache/`` at the repo root) so it is found
regardless of which subdirectory a script is launched from. Override the location with the
``FRN_CACHE_DIR`` environment variable, or pass ``refresh=True`` to force a rebuild.
"""

import os

import pandas as pd
from datasets import load_dataset

_DATASET = "Dingdong-Inc/FreshRetailNet-50K"
_CACHE_DIR = os.environ.get(
    "FRN_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".frn_cache"),
)


def load_frn(split="train", refresh=False):
    """Return a FreshRetailNet-50K split as a pandas DataFrame, cached to parquet.

    Parameters
    ----------
    split : str
        Dataset split to load (``"train"`` or ``"eval"``).
    refresh : bool
        If True, ignore any existing cache file and rebuild it from HuggingFace.
    """
    cache_path = os.path.join(_CACHE_DIR, f"{split}.parquet")
    if not refresh and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    df = load_dataset(_DATASET)[split].to_pandas()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    return df
