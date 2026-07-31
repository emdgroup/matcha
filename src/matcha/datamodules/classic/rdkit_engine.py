"""RDKit-based molecular descriptor and fingerprint calculation engine."""

import numpy as np
import joblib.externals.loky
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.EState import Fingerprinter
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import Mol
from skfp.fingerprints import (
    MAPFingerprint,
    MHFPFingerprint,
    RDKitFingerprint,
    PubChemFingerprint,
    MordredFingerprint,
)

from matcha.utils.wrapper import Wrapper, parallelize
from matcha.utils.warnings import silence_nuisance_warnings

joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(5e10)

silence_nuisance_warnings()

# due to how joblib works, some rdkit functions require an additional wrapper to
# be properly parallelized
_wrapped_GetErGFingerprint = Wrapper("GetErGFingerprint", "rdkit.Chem.rdReducedGraphs")
_wrapped_GetAvalonCountFP = Wrapper("GetAvalonCountFP", "rdkit.Avalon.pyAvalonTools")


class Engine:
    """RDKIT descriptor and fingerprint calculation class. Each molecular
    representation is computed according to the parameters set in self.defaults,
    so that the featurization performed by a given Engine object is reproducible
    across datasets.
    The class can be instantiated directly, but in general it should be used
    within the TabularFeaturizer class.

    :param int n_jobs: number of cores to use when featurizing compounds, defaults
        to 32
    """

    def __init__(self, n_jobs: int = 32):
        self._n_jobs = n_jobs

        # define set of all RDKIT descriptors
        self._all_descs = [x[0] for x in Descriptors._descList]

        # map featurization name to class method
        self._mapping = {
            "ecfp": self.get_ECFP,
            "ecfp_count": self.get_ECFP_count,
            "erg": self.get_ERG,
            "avalon": self.get_Avalon,
            "estate": self.get_ESTATE,
            "rdkit_all_descriptors": self.get_rdkit_all_descriptors,
            "map4": self.get_MAP4,
            "mhfp": self.get_MHFP,
            "rdkit_fp": self.get_rdkit_fp,
            "pubchem_fp": self.get_pubchem_fp,
            "mordred": self.get_mordred,
        }

        # set featurization default parameters
        self._defaults = {
            "ecfp": {"nBits": 1024, "radius": 3, "useFeatures": False},
            "ecfp_count": {"nBits": 2048, "radius": 3, "useFeatures": False},
            "erg": {"fuzzIncrement": 0.3},
            "avalon": {"nBits": 4096},
            "estate": {"use_bits": True},
            "rdkit_all_descriptors": {"posinf": 10000, "neginf": -10000},
            "map4": {"fp_size": 2048, "radius": 2},
            "mhfp": {"fp_size": 2048, "radius": 3},
            "rdkit_fp": {"fp_size": 2048},
            "mordred": {"use_3D": False, "posinf": 10000, "neginf": -10000},
        }

        # store dimensionality for each feature option
        self._dims = {
            "ecfp": self._defaults["ecfp"]["nBits"],
            "ecfp_count": self._defaults["ecfp_count"]["nBits"],
            "avalon": self._defaults["avalon"]["nBits"],
            "rdkit_all_descriptors": len(self._all_descs),
            "erg": 315,
            "estate": 79,
            "map4": self._defaults["map4"]["fp_size"],
            "mhfp": self._defaults["mhfp"]["fp_size"],
            "pubchem_fp": 881,
            "rdkit_fp": self._defaults["rdkit_fp"]["fp_size"],
            "mordred": 1613,
        }

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, i):
        if isinstance(i, int):
            if i >= 0:
                self._n_jobs = i
            else:
                raise ValueError("Number of jobs must be a positive integer")
        else:
            raise ValueError("Number of jobs must be a positive integer")

    @property
    def pretrained_model_dir(self) -> str:
        return self._pretrained_model_dir

    @property
    def rdkit_common_descriptors(self) -> list[str]:
        return self._common_descs

    @property
    def rdkit_custom_descriptors(self) -> list[str]:
        return self._custom_descs

    @property
    def rdkit_all_descriptors(self) -> list[str]:
        return self._all_descs

    @property
    def defaults(self) -> dict:
        return self._defaults

    def set_defaults(self, key: str, params: dict):
        """Set default parameters for a specific feature type.

        :param key: The feature type (e.g., 'ecfp', 'erg', etc.)
        :param params: Dictionary of parameters to set for that feature type
        """
        if key.lower() in self._defaults:
            self._defaults[key.lower()] = params
        else:
            raise ValueError(
                f"Fingerprint/descriptor type not supported. Valid options are {self._defaults.keys()}"
            )

    @property
    def feature_dimensionality(self, key: str) -> int:
        return self._dims[key]

    def get_ECFP(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes binary Extended Connectivity Fingerprints (ECFP) for a list
        of molecules. Calculation will be performed according to the params in
        self.defaults.
        Reference: https://pubs.acs.org/doi/10.1021/ci100050t

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_ECFP(mols)

        :param list[Mol] mols: list of rdkit molecules to compute ECFPs for

        :return np.ndarray
        """

        def batch_wrapper(batch):
            params = self.defaults["ecfp"]
            return [
                AllChem.GetMorganFingerprintAsBitVect(mol, **params) for mol in batch
            ]

        fps = parallelize(
            batch_wrapper, mols, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
        )

        array = np.empty((len(mols), len(fps[0])), dtype=np.float32)
        for i in range(len(array)):
            DataStructs.ConvertToNumpyArray(fps[i], array[i])
        return array

    def get_ECFP_count(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes count-based Extended Connectivity Fingerprints (ECFP) for a
        list of molecules. Calculation will be performed according to the params in
        self.defaults.
        Reference: https://pubs.acs.org/doi/10.1021/ci100050t

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_ECFP_count(mols)

        :param list[Mol] mols: list of rdkit molecules to compute ECFPs for

        :return np.ndarray
        """

        def batch_wrapper(batch):
            params = self.defaults["ecfp_count"]
            return [AllChem.GetHashedMorganFingerprint(mol, **params) for mol in batch]

        fps = parallelize(
            batch_wrapper, mols, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
        )

        array = np.empty(
            (len(mols), self.defaults["ecfp_count"]["nBits"]), dtype=np.float32
        )
        for i in range(len(array)):
            DataStructs.ConvertToNumpyArray(fps[i], array[i])
        return array

    def get_ERG(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes Extended Reduced Graph (ERG) 2D pharmacophore fingerprints
        for a list of molecules. Calculation will be performed according to the params in
        self.defaults.
        Reference: https://pubmed.ncbi.nlm.nih.gov/16426057/

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_ERG(mols)

        :param list[Mol] mols: list of rdkit molecules to compute ERG fingerprints for

        :return np.ndarray
        """

        def batch_wrapper(batch):
            params = self.defaults["erg"]
            return [_wrapped_GetErGFingerprint(mol, **params) for mol in batch]

        fps = parallelize(
            batch_wrapper, mols, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
        )
        return np.array(fps)

    def get_Avalon(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes Avalon path-based fingerprints for a list of molecules. Calculation
        will be performed according to the params in self.defaults.
        Reference: https://pubs.acs.org/doi/10.1021/ci050413p

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_Avalon(mols)

        :param list[Mol] mols: list of N rdkit molecules to compute Avalon fingerprints for

        :return np.ndarray
        """

        def batch_wrapper(batch):
            params = self.defaults["avalon"]
            return [_wrapped_GetAvalonCountFP(mol, **params) for mol in batch]

        fps = parallelize(
            batch_wrapper, mols, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
        )

        array = np.empty(
            (len(mols), self.defaults["avalon"]["nBits"]), dtype=np.float32
        )
        for i in range(len(array)):
            DataStructs.ConvertToNumpyArray(fps[i], array[i])
        return array

    def get_ESTATE(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes Electrotopological state (ESTATE) fingerprints for a list of molecules.
        Calculation will be performed according to the params in self.defaults.
        Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6147309/

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_ESTATE(mols)

        :param list[Mol] mols: list of N rdkit molecules to compute ESTATE fingerprints for

        :return np.ndarray
        """

        def batch_wrapper(batch):
            return [Fingerprinter.FingerprintMol(mol) for mol in batch]

        fps = parallelize(
            batch_wrapper, mols, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
        )

        if self.defaults["estate"]["use_bits"]:
            fps = [x[0] for x in fps]
        else:
            fps = [x[1] for x in fps]
        return np.array(fps)

    def get_rdkit_all_descriptors(
        self, mols: list[Mol], n_jobs: int | None = None
    ) -> np.ndarray:
        """Computes all RDKIT descriptors for a list of molecules. The list
        of descriptors is stored in self._all_descs. Calculation will be performed
        according to the params in self.defaults.

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_rdkit_all_descriptors(mols)

        :param List mols: list of N rdkit molecules to compute descriptors for

        :return np.ndarray
        """

        return self.get_arbitrary_rdkit_descriptors(
            mols, self._all_descs, n_jobs=n_jobs
        )

    def get_arbitrary_rdkit_descriptors(
        self, mols: list[Mol], desc_list: list[str], n_jobs: int | None = None
    ) -> np.ndarray:
        """Computes a set or arbitrary RDKIT descriptors for a list of molecules.

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_arbitrary_rdkit_descriptors(mols, desc_list)

        :param list[Mol] mols: list of N rdkit molecules to compute descriptors for
        :param list[str] desc_list: descriptors to compute
        :return np.ndarray
        """
        # instantiate calculator with desired descriptors
        calc = MolecularDescriptorCalculator(desc_list)

        def batch_wrapper(batch):
            return [np.array(calc.CalcDescriptors(mol)) for mol in batch]

        out = parallelize(
            batch_wrapper, mols, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
        )
        out = np.array(out)

        desc_min = self.defaults["rdkit_all_descriptors"]["neginf"]
        desc_max = self.defaults["rdkit_all_descriptors"]["posinf"]
        out[out > desc_max] = desc_max
        out[out < desc_min] = desc_min
        return np.nan_to_num(out, posinf=desc_max, neginf=desc_min)

    def get_MAP4(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes MAP4 fingerprints for a list of molecules.
        Calculation will be performed according to the params in self.defaults.
        Reference: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00445-4

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_MAP4(mols)

        :param list[Mol] mols: list of N rdkit molecules to compute fingerprints for

        :return np.ndarray
        """
        fp = MAPFingerprint(
            fp_size=self.defaults["map4"]["fp_size"],
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
        )
        return fp.transform(mols)

    def get_MHFP(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes MHFP fingerprints for a list of molecules.
        Calculation will be performed according to the params in self.defaults.
        Reference: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_MHFP(mols)

        :param list[Mol] mols: list of N rdkit molecules to compute fingerprints for

        :return np.ndarray
        """
        fp = MHFPFingerprint(
            fp_size=self.defaults["mhfp"]["fp_size"],
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
        )
        return fp.transform(mols)

    def get_rdkit_fp(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes RDKIT fingerprints for a list of molecules.
        Calculation will be performed according to the params in self.defaults.
        Reference: https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_rdkit_fp(mols)

        :param list[Mol] mols: list of N rdkit molecules to compute fingerprints for

        :return np.ndarray
        """
        fp = RDKitFingerprint(
            fp_size=self.defaults["rdkit_fp"]["fp_size"],
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
        )
        return fp.transform(mols)

    def get_pubchem_fp(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes Electrotopological state (ESTATE) fingerprints for a list of molecules.
        Calculation will be performed according to the params in self.defaults.
        Reference: https://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_pubchem_fp(mols)

        :param list[Mol] mols: list of N rdkit molecules to compute fingerprints for

        :return np.ndarray
        """
        fp = PubChemFingerprint(n_jobs=n_jobs if n_jobs is not None else self.n_jobs)
        return fp.transform(mols)

    def get_mordred(self, mols: list[Mol], n_jobs: int | None = None) -> np.ndarray:
        """Computes Mordred descriptors for a list of molecules.
        Calculation will be performed according to the params in self.defaults.
        Reference: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0258-y

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_mordred(mols)

        :param list[Mol] mols: list of N rdkit molecules to compute descriptors for

        :return np.ndarray
        """
        fp = MordredFingerprint(
            use_3D=self.defaults["mordred"]["use_3D"],
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
        )
        descs = fp.transform(mols)
        desc_min = self.defaults["mordred"]["neginf"]
        desc_max = self.defaults["mordred"]["posinf"]
        descs[descs > desc_max] = desc_max
        descs[descs < desc_min] = desc_min
        return np.nan_to_num(descs, posinf=desc_max, neginf=desc_min)

    def get_features(
        self, mols: list[Mol], feature_names: list[str], n_jobs: int | None = None
    ) -> np.ndarray:
        """Utility method to compute several feature sets for a list of molecules
        and concatenate them along axis 1 into a single array. Each feature set
        will be computed according to the params in self.defaults.

        Example usage:

        .. code-block:: python
            engine = Engine()
            x = engine.get_features(mols, ['ecpf', 'rdkit_all_descs'])

        :param list[Mol] mols: list of N rdkit molecules to compute descriptors for

        :param list[str] feature_names: list of strings defining which feature set
            to compute

        :return np.ndarray: array (N,X), axis 1 shape depends on chosen feature sets
        """
        feature_names = [x.lower() for x in feature_names]
        output = []
        for feature_name in feature_names:
            output.append(self._mapping[feature_name](mols, n_jobs=n_jobs))

        return np.concatenate(output, axis=1)

    def calculate_feature_dim(self, feature_list):
        """Utility function to dynamically calculate the dimensionality of the
        output from the datamodule without needing to first run the engine on
        some molecules

        :return int input_dim: dimensionality of the concatenated feature vector
            obtained from the list of features stored in self.params.feature_list
        """
        input_dim = 0
        for feat in feature_list:
            feat_lower = feat.lower()
            input_dim += self._dims[feat_lower]
        return input_dim
