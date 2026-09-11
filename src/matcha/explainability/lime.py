from math import ceil

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from matcha.datamodules.classic.rdkit_engine import Engine
import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN
from sklearn.metrics import r2_score
import pandas as pd
import collections as cl

_default = [
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "MolLogP",
    "MolMR",
    "TPSA",
    "SPS",
    "MolWt",
    "NumValenceElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "FpDensityMorgan2",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_S",
    "fr_HOCCN",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_SH",
]
_fp_default = {"nBits": 1024, "radius": 3, "useFeatures": False}


class LIME:
    """Local Interpretable Model-agnostic Explanations for molecular predictions.

    Fits bootstrapped Ridge regression models on molecular descriptors or
    ECFP fingerprints to identify which features most influence a prediction.
    Coefficients are optionally scaled to sum to 1 for interpretability.
    """

    def __init__(
        self,
        descriptor_set: list[str] | None = None,
        fingerprint_params: dict | None = None,
        scale_coeff: bool = True,
        use_fingerprints: bool = False,
    ):
        """Initialize LIME explainer.

        :param list[str] | None descriptor_set: RDKit descriptor names to use as features.
            Defaults to a curated set of 42 physicochemical descriptors.
        :param dict | None fingerprint_params: Parameters for Morgan fingerprint generation
            (keys: ``nBits``, ``radius``, ``useFeatures``). Defaults to 1024-bit, radius 3.
        :param bool scale_coeff: Whether to normalize coefficients so absolute values
            sum to 1. Defaults to True.
        :param bool use_fingerprints: If True, uses ECFP fingerprints instead of
            RDKit descriptors. Defaults to False.
        """
        self._descriptor_set = (
            descriptor_set if descriptor_set is not None else _default
        )
        self._fingerprint_params_set = (
            fingerprint_params if fingerprint_params is not None else _fp_default
        )
        self._model_box = []
        self._coeff_box = None
        self._r2_box = []
        self._scale_coeff = scale_coeff
        self._use_fingerprints = use_fingerprints

    @property
    def descriptor_set(self) -> str:
        """The list of RDKit descriptor names used as features."""
        return self._descriptor_set

    @property
    def scale_coeff(self) -> bool:
        """Whether coefficients are scaled to sum to 1."""
        return self._scale_coeff

    @property
    def r2_box(self) -> list[float]:
        """R-squared values from each bootstrap iteration."""
        return self._r2_box

    def _get_features(
        self, mols: list[Mol], descriptor_list: list[str] | None
    ) -> np.ndarray:
        """Compute RDKit descriptor features for a list of molecules.

        :param list[Mol] mols: RDKit molecule objects.
        :param list[str] | None descriptor_list: Descriptor names to compute.

        :returns: Feature matrix of shape ``(n_molecules, n_descriptors)``.
        """
        engine = Engine(n_jobs=1)
        feats = engine.get_arbitrary_rdkit_descriptors(mols, descriptor_list)
        return feats

    def _get_ecfps(self, mols: list[Mol], fp_params: dict | None) -> np.ndarray:
        """Compute ECFP fingerprint features for a list of molecules.

        :param list[Mol] mols: RDKit molecule objects.
        :param dict | None fp_params: Morgan fingerprint parameters.

        :returns: Fingerprint bit matrix of shape ``(n_molecules, nBits)``.
        """
        engine = Engine(n_jobs=1)
        engine._defaults["ecfp"] = (
            fp_params if fp_params is not None else self._fingerprint_params_set
        )
        feats = engine.get_ECFP(mols)
        return feats

    def _fit(self, feats: np.ndarray, y: np.ndarray, bootstrap_num: int):
        """Fit bootstrapped Ridge regression models on feature subsets.

        Uses K-Fold splitting to subsample both samples and features, fitting
        a separate Ridge model on each fold and collecting coefficients.

        :param np.ndarray feats: Feature matrix, shape ``(n_samples, n_features)``.
        :param np.ndarray y: Target values, shape ``(n_samples,)``.
        :param int bootstrap_num: Number of bootstrap iterations (K-Fold splits).

        :returns: Coefficient matrix of shape ``(bootstrap_num, n_features)``.
        """
        n_rows, feature_count = feats.shape
        sample_splits = max(bootstrap_num, ceil(n_rows / (n_rows - 2)))
        sample_idx = [
            train_rows for train_rows, _ in KFold(n_splits=sample_splits).split(feats)
        ][:bootstrap_num]
        if bootstrap_num == 1:
            feat_idx = [np.arange(feature_count)]
        else:
            feat_idx = [
                train_columns
                for train_columns, _ in KFold(n_splits=bootstrap_num).split(
                    np.arange(feature_count)
                )
            ]

        self._model_box = []
        self._r2_box = []
        self._coeff_box = np.full((bootstrap_num, feature_count), np.nan)
        for fit_index, (sample_rows, feature_columns) in enumerate(
            zip(sample_idx, feat_idx)
        ):
            fit_features = feats[np.ix_(sample_rows, feature_columns)]
            fit_features = StandardScaler().fit_transform(fit_features)
            fit_targets = y[sample_rows]
            model = Ridge()
            model.fit(fit_features, fit_targets)
            predictions = model.predict(fit_features)
            self._r2_box.append(r2_score(fit_targets, predictions))
            self._model_box.append(model)
            self._coeff_box[fit_index, feature_columns] = model.coef_

        return self._coeff_box

    def _get_ECFP_envs(
        self, mol: Mol, radius: int = 3, nBits: int = 1024, useFeatures: bool = False
    ) -> dict[int, set[int]]:
        """Compute atomic environments for bits of Extended Connectivity Fingerprints (ECFP).

        Reference: https://pubs.acs.org/doi/10.1021/ci100050t

        :param Mol mol: RDKit molecule to compute environments for.
        :param int radius: Radius for Morgan fingerprint. Defaults to 3.
        :param int nBits: Number of bits in fingerprint. Defaults to 1024.
        :param bool useFeatures: Whether to use feature-based fingerprints.
            Defaults to False.

        :returns: Dictionary mapping bit IDs to sets of atom indices involved
            in that fingerprint bit's environment.
        """
        bitinfo = dict()
        envs = cl.defaultdict(set)
        # generate the ECFP and store bit information
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=nBits, useFeatures=useFeatures, bitInfo=bitinfo
        )
        # iterate over collected information
        for bitid, examples in bitinfo.items():
            for aid, rad in examples:
                envs[bitid].add(aid)
                path = FindAtomEnvironmentOfRadiusN(mol, rad, aid)
                for bid in path:
                    envs[bitid].add(mol.GetBondWithIdx(bid).GetBeginAtomIdx())
                    envs[bitid].add(mol.GetBondWithIdx(bid).GetEndAtomIdx())
        return envs

    def get_envs_and_weights(self, mol: Mol, out: pd.DataFrame):
        """Extracts atomic environments and weights for a molecule given a lime analysis result

        :param Mol mol: rdkit molecule
        :param pd.DataFrame out: lime analysis result
        """
        params = self._fingerprint_params_set
        envs = self._get_ECFP_envs(
            mol,
            radius=params.get("radius", 3),
            nBits=params.get("nBits", 1024),
            useFeatures=params.get("useFeatures", False),
        )
        _out = out.drop(index=out.index[-1], axis=0, inplace=False).reset_index(
            drop=True
        )
        _out["PID"] = _out.Descriptor.str.split("_").str[-1].astype(int)
        weights = {
            int(pid): weight for pid, weight in _out[["PID", "Coefficient"]].values
        }
        return envs, weights

    def explain(
        self, X: list[Mol], Y: np.ndarray, bootstrap_num: int = 25
    ) -> pd.DataFrame:
        """Perform LIME analysis on molecules.

        Fits bootstrapped Ridge regression models to explain how molecular
        descriptors (or fingerprint bits) relate to the target values. Returns
        a DataFrame of coefficients sorted by importance.

        :param list[Mol] X: RDKit molecule objects to explain.
        :param np.ndarray Y: Target values (predictions or any endpoint),
            shape ``(n_molecules,)``.
        :param int bootstrap_num: Number of bootstrap iterations. Defaults to 25.

        :returns: DataFrame with columns ``Descriptor``, ``Coefficient``, and
            ``Standard deviation``, sorted by coefficient magnitude. The last
            row contains the local fit R-squared summary.
        """
        targets = np.asarray(Y)
        if targets.ndim != 1:
            raise ValueError("LIME targets must be one-dimensional.")
        if len(X) != len(targets):
            raise ValueError(
                "LIME molecule and target counts must match: "
                f"received {len(X)} molecules and {len(targets)} targets."
            )
        if len(X) < 3:
            raise ValueError(f"LIME requires at least 3 molecules; received {len(X)}.")
        if bootstrap_num < 1:
            raise ValueError("bootstrap_num must be at least 1.")

        if self._use_fingerprints:
            feats = self._get_ecfps(X, self._fingerprint_params_set)
            columns = [f"F_{i}" for i in range(feats.shape[1])]
        else:
            feats = self._get_features(X, self.descriptor_set)
            columns = self._descriptor_set

        feature_count = feats.shape[1]
        if bootstrap_num > feature_count:
            raise ValueError(
                f"bootstrap_num ({bootstrap_num}) cannot exceed available feature "
                f"count ({feature_count})."
            )

        requested_fits = min(bootstrap_num, len(X))
        coeff_box = self._fit(feats, targets, requested_fits)
        coeff_median = np.zeros(feature_count)
        coeff_std = np.zeros(feature_count)
        for column_index in range(feature_count):
            selected_coefficients = coeff_box[:, column_index]
            selected_coefficients = selected_coefficients[
                ~np.isnan(selected_coefficients)
            ]
            if selected_coefficients.size:
                coeff_median[column_index] = np.median(selected_coefficients)
                coeff_std[column_index] = np.std(selected_coefficients)

        if self.scale_coeff:
            coefficient_norm = np.sum(np.abs(coeff_median))
            if coefficient_norm != 0:
                coeff_median = coeff_median / coefficient_norm
                coeff_std = coeff_std / coefficient_norm

        df_out = pd.DataFrame(
            {
                "Descriptor": columns,
                "Coefficient": coeff_median,
                "Standard deviation": coeff_std,
            }
        )

        df_out = df_out.sort_values("Coefficient", ascending=False)
        row = ["Local fit R2", np.median(self.r2_box), np.std(self.r2_box)]
        df_out.loc[len(df_out)] = row

        return df_out
