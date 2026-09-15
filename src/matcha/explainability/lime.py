from sklearn.linear_model import Ridge
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
_fp_default = {"nBits": 8192, "radius": 3, "useFeatures": False}


class LIME:
    """Local Interpretable Model-agnostic Explanations for molecular predictions.

    Fits bootstrapped Ridge regression models on molecular descriptors or
    ECFP fingerprints to identify which features most influence a prediction.
    """

    def __init__(
        self,
        descriptor_set: list[str] | None = None,
        fingerprint_params: dict | None = None,
        use_fingerprints: bool = False,
        random_seed: int = 0,
    ):
        """Initialize LIME explainer.

        :param list[str] | None descriptor_set: RDKit descriptor names to use as features.
            Defaults to a curated set of 42 physicochemical descriptors.
        :param dict | None fingerprint_params: Parameters for Morgan fingerprint generation
            (keys: ``nBits``, ``radius``, ``useFeatures``). Defaults to 8192-bit, radius 3.
        :param bool use_fingerprints: If True, uses ECFP fingerprints instead of
            RDKit descriptors. Defaults to False.
        :param int random_seed: Seed for bootstrap sampling. Defaults to 0.
        """
        self._descriptor_set = (
            descriptor_set if descriptor_set is not None else _default
        )
        self._fingerprint_params_set = _fp_default.copy()
        if fingerprint_params is not None:
            self._fingerprint_params_set.update(fingerprint_params)
        self._model_box = []
        self._coeff_box = None
        self._r2_box = []
        self._use_fingerprints = use_fingerprints
        self._random_seed = random_seed

    @property
    def descriptor_set(self) -> str:
        """The list of RDKit descriptor names used as features."""
        return self._descriptor_set

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

    def _get_ecfps(self, mols: list[Mol]) -> np.ndarray:
        """Compute ECFP fingerprint features for a list of molecules.

        :param list[Mol] mols: RDKit molecule objects.

        :returns: Fingerprint bit matrix of shape ``(n_molecules, nBits)``.
        """
        engine = Engine(n_jobs=1)
        engine._defaults["ecfp"] = self._fingerprint_params_set
        feats = engine.get_ECFP(mols)
        return feats

    def _fit(self, feats: np.ndarray, y: np.ndarray, bootstrap_num: int):
        """Fit Ridge models on full-size row-bootstrap samples.

        Each fit samples rows with replacement and retains every feature column.
        Descriptor features are standardized independently per fit, while ECFP
        features are fitted as raw binary values.

        :param np.ndarray feats: Feature matrix, shape ``(n_samples, n_features)``.
        :param np.ndarray y: Target values, shape ``(n_samples,)``.
        :param int bootstrap_num: Exact number of bootstrap fits.

        :returns: Coefficient matrix of shape ``(bootstrap_num, n_features)``.
        """
        n_rows, feature_count = feats.shape
        rng = np.random.default_rng(self._random_seed % (2**64))

        self._model_box = []
        self._r2_box = []
        self._coeff_box = np.empty((bootstrap_num, feature_count))
        for fit_index in range(bootstrap_num):
            sample_rows = rng.choice(n_rows, size=n_rows, replace=True)
            fit_features = feats[sample_rows]
            if not self._use_fingerprints:
                fit_features = StandardScaler().fit_transform(fit_features)
            fit_targets = y[sample_rows]
            model = Ridge()
            model.fit(fit_features, fit_targets)
            predictions = model.predict(fit_features)
            self._r2_box.append(r2_score(fit_targets, predictions))
            self._model_box.append(model)
            self._coeff_box[fit_index] = model.coef_

        return self._coeff_box

    def _get_ECFP_envs(
        self, mol: Mol, radius: int = 3, nBits: int = 8192, useFeatures: bool = False
    ) -> dict[int, set[int]]:
        """Compute atomic environments for bits of Extended Connectivity Fingerprints (ECFP).

        Reference: https://pubs.acs.org/doi/10.1021/ci100050t

        :param Mol mol: RDKit molecule to compute environments for.
        :param int radius: Radius for Morgan fingerprint. Defaults to 3.
        :param int nBits: Number of bits in fingerprint. Defaults to 8192.
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
            radius=params["radius"],
            nBits=params["nBits"],
            useFeatures=params["useFeatures"],
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
        :param int bootstrap_num: Exact number of full-size row-bootstrap fits.
            Defaults to 25.

        :returns: DataFrame with columns ``Descriptor``, raw ``Coefficient``, and
            raw ``Standard deviation``, sorted by coefficient magnitude. The last
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
            feats = self._get_ecfps(X)
            columns = [f"F_{i}" for i in range(feats.shape[1])]
        else:
            feats = self._get_features(X, self.descriptor_set)
            columns = self._descriptor_set

        coeff_box = self._fit(feats, targets, bootstrap_num)
        coeff_median = np.median(coeff_box, axis=0)
        coeff_std = np.std(coeff_box, axis=0)

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
