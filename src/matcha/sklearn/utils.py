"""Legacy utility functions for parameter parsing and data splitting."""

from sklearn.model_selection import train_test_split


def parse_params(dictionary: dict):
    """Flatten a structured parameter dictionary into a single dict.

    Merges the ``Training``, ``DataModule``, and ``Model`` keys into
    one flat dictionary suitable for passing to a model constructor.

    :param dict dictionary: structured config with top-level section keys
    :returns: flattened parameter dictionary
    :rtype: dict
    """
    args = {}
    args.update(dictionary["Training"])
    args.update(dictionary["DataModule"])
    args.update(dictionary["Model"])
    # args.pop("is_classification")
    # args.pop("additional_mol_features_dim")
    return args


def random_split(x, y, bound, seed=42):
    """Split data into 90/10 train/validation sets with optional bound masks.

    :param list x: input features (list of molecules or SMILES)
    :param np.ndarray y: target labels array
    :param list | None bound: bound mask(s) or None if not applicable
    :param int seed: random seed for reproducibility, defaults to 42
    :returns: tuple of (x_train, y_train, bound_train, x_val, y_val, bound_val)
    :rtype: tuple
    """
    idx = list(range(len(x)))
    idx_train, idx_val = train_test_split(idx, test_size=0.1, random_state=seed)
    x_train = [x[i] for i in idx_train]
    x_val = [x[i] for i in idx_val]
    y_train = y[idx_train]
    y_val = y[idx_val]
    if isinstance(bound, list):
        if isinstance(bound[0], str):
            bound_train = [bound[i] for i in idx_train]
            bound_val = [bound[i] for i in idx_val]
        else:
            bound_train = []
            bound_val = []
            for b in bound:
                bound_train.append([b[i] for i in idx_train])
                bound_val.append([b[i] for i in idx_val])
    else:
        bound_train = None
        bound_val = None

    return x_train, y_train, bound_train, x_val, y_val, bound_val
