import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from sklearn.cluster import DBSCAN
import umap
from sklearn.model_selection import train_test_split
from matcha.datamodules.classic.rdkit_engine import Engine


def random_split(
    df: pd.DataFrame, split_size: float = 0.2, random_flag: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a random train/test split on a DataFrame.

    :param pd.DataFrame df: The input DataFrame to split.
    :param float split_size: Fraction of data to allocate to the test set.
    :param int random_flag: Random seed for reproducibility.
    :returns: A tuple of (train_df, test_df).
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    train_df, test_df = train_test_split(
        df, test_size=split_size, random_state=random_flag
    )

    return train_df, test_df


def time_split(
    df: pd.DataFrame,
    column_name: str | None,
    split_value: float | str | None = None,
    split_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a time-based or ordered split on a DataFrame.

    Sorts the DataFrame by the specified column and splits it into train and
    test sets based on either a split value threshold or a fractional size.

    :param pd.DataFrame df: The input DataFrame to split.
    :param column_name: Name of the column to sort by. If ``None``, the
        DataFrame is used in its current order.
    :type column_name: str | None
    :param split_value: Threshold value; rows with values >= this go to test.
    :type split_value: float | str | None
    :param split_size: Fraction of data to allocate to the test set.
    :type split_size: float | None
    :returns: A tuple of (train_df, test_df).
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    :raises ValueError: If the column is not found, column type is unsupported,
        split_size is not between 0 and 1, or neither split_value nor
        split_size is provided.
    """

    if column_name is not None:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        # Check for data type in the column
        if (
            pd.api.types.is_numeric_dtype(df[column_name])
            or df[column_name].dtype == "object"
        ):
            df = df.sort_values(by=column_name).reset_index(drop=True)
        elif pd.api.types.is_datetime64_any_dtype(df[column_name]):
            df = df.sort_values(by=column_name).reset_index(drop=True)
        else:
            raise ValueError("Column type is not supported for splitting.")
    else:
        pass

    if split_size is not None:
        if not (0 < split_size < 1):
            raise ValueError("Percentage must be between 0 and 1.")
        split_index = int(len(df) * (1 - split_size))
    elif split_value is not None:
        print(
            f"Split will be done based on the value {split_value}. The entries with the value {split_value} will be part of test_df"
        )
        if pd.api.types.is_datetime64_any_dtype(df[column_name]):
            split_index = df[df[column_name] >= split_value].index[0]
        else:
            split_index = df[df[column_name] >= split_value].index[0]
    else:
        raise ValueError("Either split_value or percentage must be provided.")

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    return train_df, test_df


def cluster_split(
    df,
    split_size=0.2,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=np.random.randint(low=0, high=100000),
    feature_set="ecfp",
    metric="jaccard",
    n_jobs=1,
):
    """Perform a cluster-based train/test split to minimize data leakage.

    Computes molecular fingerprints, reduces dimensionality with UMAP, clusters
    with DBSCAN, and assigns entire clusters to either train or test to prevent
    similar molecules from appearing in both sets.

    :param pd.DataFrame df: Input DataFrame containing a ``SMILES`` column.
    :param float split_size: Fraction of data to allocate to the test set.
    :param int n_neighbors: Number of neighbors for UMAP.
    :param float min_dist: Minimum distance parameter for UMAP.
    :param int n_components: Number of UMAP embedding dimensions.
    :param int random_state: Random seed for UMAP.
    :param str feature_set: Fingerprint type to compute (e.g., ``"ecfp"``).
    :param str metric: Distance metric for UMAP.
    :param int n_jobs: Number of parallel jobs for fingerprint computation.
    :returns: A tuple of (train_df, test_df).
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    df = df.copy()

    # Add molecules and compute fingerprints
    if "ROMol" not in df.columns:
        PandasTools.AddMoleculeColumnToFrame(df, "SMILES", "ROMol")

    engine = Engine(n_jobs=n_jobs)
    X = engine.get_features(df.ROMol.tolist(), [feature_set])

    # Run UMAP dimensionality reduction
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric=metric,
    )
    X_umap = umap_model.fit_transform(X)

    # Cluster using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df["Cluster"] = dbscan.fit_predict(X_umap)

    # Now split the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    cluster_counts = df["Cluster"].value_counts()
    clusters = cluster_counts.index.tolist()
    total_rows = len(df)
    test_rows_count = int(total_rows * (1 - split_size))
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    shuffled_clusters = np.random.permutation(clusters)
    # Selcet clusters for training and test set fulfilling to be as close to the desired split as possible while preventing data leakage
    for cluster in shuffled_clusters:
        cluster_size = cluster_counts[cluster]
        # Check if adding this cluster would exceed the training row limit
        if len(test_df) + cluster_size <= test_rows_count:
            test_df = pd.concat([test_df, df[df["Cluster"] == cluster]])
        else:
            # If it exceeds, add it to the train set instead
            train_df = pd.concat([train_df, df[df["Cluster"] == cluster]])

    return train_df, test_df
