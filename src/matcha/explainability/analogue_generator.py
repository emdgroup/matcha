from rdkit import Chem
from rdkit.Chem import CombineMols
from itertools import combinations
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import time
from rdkit.Chem.BRICS import BRICSDecompose


class AnalogueGenerator:
    """Generator of structural analogues for molecular explainability.

    Provides class methods that produce analogues of a query molecule using
    two complementary strategies:

    - **Positional analogue scanning:** Substitutes atoms at anchor positions.
    - **Nitrogen walk:** Replaces aromatic CH with nitrogen.

    All generated analogues are validated via RDKit sanitization and deduplicated.
    """

    @classmethod
    def generate_analogues(
        cls,
        mol: Mol,
        positional_analogue_scanning_params: dict | None = {
            "substituents": [
                "F",
                "I",
                "Br",
                "Cl",
                "O",
                "C",
                "[*]C(F)(F)F",
                "[*]C#N",
                "[*]OC",
                "[*]C1CC1",
                "[*]C(=O)N",
                "[*]S(=O)(=O)C",
            ],
            "anchors": ["[cH]", "C"],
            "num_sub": 1,
        },
        nitrogen_walk_params: dict | None = {"num_sub": 1},
    ) -> list[Mol]:
        """Generate structural analogues of a molecule using scaffold-based and pairwise strategies.

        Runs positional analogue scanning (PAS) and nitrogen walk on both the
        input molecule and its Murcko scaffold, then applies a second round of
        PAS and nitrogen walk to each first-pass PAS result. All results are
        pooled, validated, deduplicated, and the input molecule is excluded.

        :param Mol mol: Query RDKit molecule.
        :param dict | None positional_analogue_scanning_params: Parameters for
            positional analogue scanning. Set to None to skip. Keys:
            ``substituents``, ``anchors``, ``num_sub``.
        :param dict | None nitrogen_walk_params: Parameters for nitrogen walking.
            Set to None to skip. Keys: ``num_sub``.

        :returns: List of unique, sanitized RDKit molecule objects (excluding
            the input molecule).
        """
        scaffold = GetScaffoldForMol(mol)

        pas_mol = (
            cls.positional_analogue_scanning(
                mol_in=mol, **positional_analogue_scanning_params
            )
            if positional_analogue_scanning_params is not None
            else []
        )
        nw_mol = (
            cls.nitrogen_walk(mol_in=mol, **nitrogen_walk_params)
            if nitrogen_walk_params is not None
            else []
        )
        pas_scaffold = (
            cls.positional_analogue_scanning(
                mol_in=scaffold, **positional_analogue_scanning_params
            )
            if positional_analogue_scanning_params is not None
            else []
        )
        nw_scaffold = (
            cls.nitrogen_walk(mol_in=scaffold, **nitrogen_walk_params)
            if nitrogen_walk_params is not None
            else []
        )

        pas_pas = []
        pas_nw = []
        for analogue in pas_mol + pas_scaffold:
            if positional_analogue_scanning_params is not None:
                pas_pas += cls.positional_analogue_scanning(
                    mol_in=analogue, **positional_analogue_scanning_params
                )
            if nitrogen_walk_params is not None:
                pas_nw += cls.nitrogen_walk(mol_in=analogue, **nitrogen_walk_params)

        all_analogues = pas_mol + nw_mol + pas_scaffold + nw_scaffold + pas_pas + pas_nw

        input_smi = Chem.MolToSmiles(mol)
        validated_smi = set()
        for analogue in all_analogues:
            try:
                smi = Chem.MolToSmiles(analogue)
                validated_smi.add(smi)
            except Exception:
                pass

        validated_smi.discard(input_smi)
        validated = [Chem.MolFromSmiles(smi) for smi in validated_smi]
        return list(filter(None, validated))

    @classmethod
    def positional_analogue_scanning(
        cls,
        mol_in: Mol,
        substituents: list = [
            "F",
            "I",
            "Br",
            "Cl",
            "O",
            "C",
            "[*]C(F)(F)F",
            "[*]C#N",
            "[*]OC",
            "[*]C1CC1",
            "[*]C(=O)N",
            "[*]S(=O)(=O)C",
        ],
        anchors: list = ["[cH]", "C"],
        num_sub: int = 1,
        timeout: int = 60,
    ) -> list[Mol]:
        """Generate analogues via Positional Analogue Scanning.

        Adds substituents at anchor positions in the molecule. Each entry in
        ``substituents`` is either an element symbol (e.g. ``"F"``) or a
        fragment SMILES with a ``[*]`` dummy attachment point
        (e.g. ``"[*]C(F)(F)F"``). Based on the algorithm described in
        https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b02092.

        Implementation adapted from:
        https://practicalcheminformatics.blogspot.com/2020/04/positional-analogue-scanning.html

        :param Mol mol_in: Input RDKit molecule.
        :param list substituents: Element symbols or fragment SMILES (with
            ``[*]`` attachment point) to try as substituents.
        :param list anchors: SMARTS patterns identifying attachment positions.
        :param int num_sub: Number of simultaneous substitutions. Defaults to 1.
        :param int timeout: Maximum runtime in seconds. Defaults to 60.

        :returns: List of unique, deduplicated analogue molecules.
        """
        pt = Chem.GetPeriodicTable()
        out_mol_list = []
        start = time.time()
        for substituent in substituents:
            frag = Chem.MolFromSmiles(substituent)
            is_fragment = frag is not None and any(
                a.GetAtomicNum() == 0 for a in frag.GetAtoms()
            )
            if not is_fragment:
                atomic_num = pt.GetAtomicNumber(substituent)
                if not (
                    atomic_num <= 8
                    or any(
                        atm.GetAtomicNum() == atomic_num for atm in mol_in.GetAtoms()
                    )
                ):
                    continue
            for anchor in anchors:
                query = Chem.MolFromSmarts(anchor)
                match_atms = [x[0] for x in mol_in.GetSubstructMatches(query)]
                n_combos = combinations(match_atms, num_sub)
                for combo in n_combos:
                    new_mol = Chem.RWMol(mol_in)
                    success = True
                    for idx in combo:
                        result = cls._attach_substituent(new_mol, idx, substituent)
                        if result is None:
                            success = False
                            break
                        new_mol = result
                    if success:
                        out_mol_list.append(new_mol)
            if time.time() - start > timeout:
                break

        return cls._remove_duplicate(mol_in, out_mol_list)

    @classmethod
    def nitrogen_walk(
        cls, mol_in: Mol, num_sub: int = 1, timeout: int = 60
    ) -> list[Mol]:
        """Generate analogues by replacing aromatic CH atoms with nitrogen.

        Systematically walks aromatic carbon-hydrogen positions, replacing them
        with nitrogen to produce aza-analogues.

        Implementation adapted from:
        https://practicalcheminformatics.blogspot.com/2020/04/positional-analogue-scanning.html

        :param Mol mol_in: Input RDKit molecule.
        :param int num_sub: Number of simultaneous CH-to-N replacements. Defaults to 1.
        :param int timeout: Maximum runtime in seconds. Defaults to 60.

        :returns: List of unique, deduplicated analogue molecules.
        """

        out_mol_list = []
        aromatic_cH = Chem.MolFromSmarts("[cH]")
        match_atms = [x[0] for x in mol_in.GetSubstructMatches(aromatic_cH)]
        n_combos = combinations(match_atms, num_sub)
        start = time.time()
        for combo in n_combos:
            new_mol = Chem.RWMol(mol_in)
            for idx in combo:
                atm = new_mol.GetAtomWithIdx(idx)
                atm.SetAtomicNum(7)
            try:
                Chem.SanitizeMol(new_mol)
                out_mol_list.append(new_mol)
            except Exception:
                pass
            if time.time() - start > timeout:
                break
        return cls._remove_duplicate(mol_in, out_mol_list)

    @classmethod
    def decompose(
        cls,
        mol_in: Mol,
    ):
        """Decompose a molecule into BRICS fragments.

        Performs a single-pass BRICS decomposition and returns cleaned fragments
        as RDKit molecules (with attachment-point labels removed).

        :param Mol mol_in: Input RDKit molecule.

        :returns: List of fragment molecules.
        """
        out = list(BRICSDecompose(mol_in, singlePass=True))
        filtered_list = [s for s in out if "[" in s and "]" in s]
        cleaned_list = [
            s.replace(s[s.index("[") : s.index("]") + 1], "") for s in filtered_list
        ]
        brics = [Chem.MolFromSmiles(x) for x in cleaned_list]
        return list(filter(None, brics))

    @classmethod
    def _attach_substituent(
        cls, mol: Mol, anchor_idx: int, substituent: str
    ) -> Mol | None:
        """Attach a substituent at a given anchor atom index.

        Detects whether ``substituent`` is a fragment SMILES containing a
        ``[*]`` dummy attachment point (e.g. ``"[*]C(F)(F)F"``) or a bare
        element symbol (e.g. ``"F"``), and applies the appropriate attachment.

        :param Mol mol: Input molecule.
        :param int anchor_idx: Index of the anchor atom to attach at.
        :param str substituent: Element symbol or fragment SMILES with ``[*]``.

        :returns: Sanitized RWMol with substituent attached, or ``None`` on failure.
        """
        try:
            frag = Chem.MolFromSmiles(substituent)
            is_fragment = frag is not None and any(
                a.GetAtomicNum() == 0 for a in frag.GetAtoms()
            )
            if is_fragment:
                sidechain = cls._prep_sidechain(substituent)
                new_mol = Chem.RWMol(CombineMols(mol, sidechain))
                attach_atm = -1
                for atm in new_mol.GetAtoms():
                    if atm.GetAtomMapNum() == 1:
                        attach_atm = atm.GetIdx()
                        break
                if attach_atm == -1:
                    return None
                new_mol.AddBond(
                    anchor_idx, attach_atm, order=Chem.rdchem.BondType.SINGLE
                )
                for atm in new_mol.GetAtoms():
                    atm.SetAtomMapNum(0)
            else:
                pt = Chem.GetPeriodicTable()
                atomic_num = pt.GetAtomicNumber(substituent)
                new_mol = Chem.RWMol(mol)
                new_idx = new_mol.AddAtom(Chem.Atom(atomic_num))
                new_mol.AddBond(anchor_idx, new_idx, order=Chem.rdchem.BondType.SINGLE)
            Chem.SanitizeMol(new_mol)
            return new_mol
        except Exception:
            return None

    @classmethod
    def _remove_duplicate(cls, target: Mol, analogues: list[Mol]) -> list[Mol]:
        smi_target = Chem.MolToSmiles(target)
        analogues = [Chem.MolToSmiles(x) for x in analogues]
        unique_smiles = set(analogues)
        unique_smiles.discard(smi_target)
        return [Chem.MolFromSmiles(smi) for smi in unique_smiles]

    @classmethod
    def _prep_sidechain(cls, smi):
        """Prepare a fragment SMILES as an RDKit sidechain ready for bonding.

        Removes the ``[*]`` dummy attachment atom and marks its neighbor with
        atom-map number 1 so it can be found by :meth:`_attach_substituent`.
        """
        mol = Chem.MolFromSmiles(smi)
        rw_mol = Chem.RWMol(mol)
        remove_idx = -1
        for atm in rw_mol.GetAtoms():
            if atm.GetAtomicNum() == 0:
                remove_idx = atm.GetIdx()
                for nbr in atm.GetNeighbors():
                    nbr.SetAtomMapNum(1)
        rw_mol.RemoveAtom(remove_idx)
        Chem.SanitizeMol(rw_mol)
        return rw_mol
