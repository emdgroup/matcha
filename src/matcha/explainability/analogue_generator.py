import math
import random
import time

from rdkit import Chem
from rdkit.Chem import CombineMols
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit.Chem.rdchem import Mol, MolSanitizeException
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


_DEFAULT_SUBSTITUENTS = [
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
]


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
            "substituents": _DEFAULT_SUBSTITUENTS,
            "anchors": ["[cH]", "C"],
        },
        nitrogen_walk_params: dict | None = {},
        reverse_positional_analogue_scanning: bool = True,
        generation_timeout: float = 60.0,
        num_sample: int = 100,
        random_seed: int = 0,
    ) -> list[Mol]:
        """Generate structural analogues of a molecule using one time budget.

        Runs five systematic first-pass sources -- query PAS, query reverse
        PAS, query nitrogen walk, scaffold PAS, scaffold nitrogen walk -- then
        performs bounded deterministic multi-step sampling from two
        independent parent pools. The query pool combines every query-derived
        first-pass source; the scaffold pool combines every scaffold-derived
        first-pass source. Each branch selects one enabled strategy per
        attempt uniformly among PAS, reverse PAS, and nitrogen walk, targeting
        ``num_sample`` valid, branch-local canonical-unique candidates and
        stopping after ``2 * num_sample`` attempts. The cooperative deadline is
        checked around RDKit operations but cannot preempt an RDKit primitive
        that is already running.

        :param Mol mol: Query RDKit molecule.
        :param dict | None positional_analogue_scanning_params: Parameters for
            positional analogue scanning. Set to None to skip. Keys:
            ``substituents``, ``anchors``.
        :param dict | None nitrogen_walk_params: Parameters for nitrogen walking.
            Set to None to skip.
        :param bool reverse_positional_analogue_scanning: Whether to remove
            peripheral groups from the forward PAS vocabulary. Defaults to True.
        :param float generation_timeout: Maximum aggregate runtime in seconds.
            Defaults to 60. Zero causes immediate expiry.
        :param int num_sample: Per-branch multi-step sampling quota. Each of
            the query and scaffold branches contributes up to ``num_sample``
            unique valid candidates. Defaults to 100. Zero disables second-step
            sampling; first-pass generation still runs.
        :param int random_seed: Seed for the deterministic sampling RNG.
            Defaults to 0. Identical inputs, configuration, and seed produce
            identical sampled candidates in identical order.

        :returns: List of unique, sanitized RDKit molecule objects (excluding
            the input molecule). Sampled candidates follow every first-pass
            source in the returned order.
        """
        generation_timeout = cls._validate_timeout(
            generation_timeout, "generation_timeout"
        )
        num_sample = cls._validate_non_negative_int(num_sample, "num_sample")
        random_seed = cls._validate_int(random_seed, "random_seed")
        start = time.monotonic()
        overall_deadline = start + generation_timeout

        pas_mol = []
        if positional_analogue_scanning_params is not None:
            stage = "query forward PAS"
            cls._check_deadline(overall_deadline, generation_timeout, stage)
            deadline, budget = cls._aggregate_strategy_deadline(
                start,
                overall_deadline,
                generation_timeout,
                positional_analogue_scanning_params.get("timeout", 60),
            )
            pas_mol = cls.positional_analogue_scanning(
                mol_in=mol,
                **positional_analogue_scanning_params,
                _deadline=deadline,
                _timeout_budget=budget,
                _stage=stage,
            )

        reverse_substituents = (
            positional_analogue_scanning_params.get(
                "substituents", _DEFAULT_SUBSTITUENTS
            )
            if positional_analogue_scanning_params is not None
            else None
        )
        reverse_mol = []
        if reverse_positional_analogue_scanning:
            stage = "query reverse PAS"
            cls._check_deadline(overall_deadline, generation_timeout, stage)
            reverse_timeout = (
                positional_analogue_scanning_params.get("timeout", 60)
                if positional_analogue_scanning_params is not None
                else 60
            )
            deadline, budget = cls._aggregate_strategy_deadline(
                start, overall_deadline, generation_timeout, reverse_timeout
            )
            reverse_mol = cls.reverse_positional_analogue_scanning(
                mol_in=mol,
                substituents=reverse_substituents,
                _deadline=deadline,
                _timeout_budget=budget,
                _stage=stage,
            )

        nw_mol = []
        if nitrogen_walk_params is not None:
            stage = "query nitrogen walk"
            cls._check_deadline(overall_deadline, generation_timeout, stage)
            deadline, budget = cls._aggregate_strategy_deadline(
                start,
                overall_deadline,
                generation_timeout,
                nitrogen_walk_params.get("timeout", 60),
            )
            nw_mol = cls.nitrogen_walk(
                mol_in=mol,
                **nitrogen_walk_params,
                _deadline=deadline,
                _timeout_budget=budget,
                _stage=stage,
            )

        cls._check_deadline(overall_deadline, generation_timeout, "scaffold generation")
        scaffold = GetScaffoldForMol(mol)
        cls._check_deadline(overall_deadline, generation_timeout, "scaffold generation")

        pas_scaffold = []
        if positional_analogue_scanning_params is not None:
            stage = "scaffold forward PAS"
            deadline, budget = cls._aggregate_strategy_deadline(
                start,
                overall_deadline,
                generation_timeout,
                positional_analogue_scanning_params.get("timeout", 60),
            )
            pas_scaffold = cls.positional_analogue_scanning(
                mol_in=scaffold,
                **positional_analogue_scanning_params,
                _deadline=deadline,
                _timeout_budget=budget,
                _stage=stage,
            )

        nw_scaffold = []
        if nitrogen_walk_params is not None:
            stage = "scaffold nitrogen walk"
            deadline, budget = cls._aggregate_strategy_deadline(
                start,
                overall_deadline,
                generation_timeout,
                nitrogen_walk_params.get("timeout", 60),
            )
            nw_scaffold = cls.nitrogen_walk(
                mol_in=scaffold,
                **nitrogen_walk_params,
                _deadline=deadline,
                _timeout_budget=budget,
                _stage=stage,
            )

        substituents = (
            positional_analogue_scanning_params.get(
                "substituents", _DEFAULT_SUBSTITUENTS
            )
            if positional_analogue_scanning_params is not None
            else []
        )
        enabled_strategies: list[str] = []
        if positional_analogue_scanning_params is not None:
            enabled_strategies.append("pas")
        if reverse_positional_analogue_scanning and substituents:
            enabled_strategies.append("reverse_pas")
        if nitrogen_walk_params is not None:
            enabled_strategies.append("nitrogen_walk")

        sampler_rng = random.Random(random_seed)
        query_seed = sampler_rng.getrandbits(64)
        scaffold_seed = sampler_rng.getrandbits(64)

        query_pool = pas_mol + reverse_mol + nw_mol
        scaffold_pool = pas_scaffold + nw_scaffold

        query_sampled = cls._sample_branch(
            mol_in=mol,
            parent_pool=query_pool,
            enabled_strategies=enabled_strategies,
            pas_params=positional_analogue_scanning_params,
            num_sample=num_sample,
            seed=query_seed,
            overall_deadline=overall_deadline,
            generation_timeout=generation_timeout,
            stage="query sampled multi-step",
        )
        scaffold_sampled = cls._sample_branch(
            mol_in=mol,
            parent_pool=scaffold_pool,
            enabled_strategies=enabled_strategies,
            pas_params=positional_analogue_scanning_params,
            num_sample=num_sample,
            seed=scaffold_seed,
            overall_deadline=overall_deadline,
            generation_timeout=generation_timeout,
            stage="scaffold sampled multi-step",
        )

        all_analogues = (
            pas_mol
            + reverse_mol
            + nw_mol
            + pas_scaffold
            + nw_scaffold
            + query_sampled
            + scaffold_sampled
        )
        return cls._remove_duplicate(
            mol,
            all_analogues,
            deadline=overall_deadline,
            timeout_budget=generation_timeout,
            stage="finalization",
        )

    @classmethod
    def positional_analogue_scanning(
        cls,
        mol_in: Mol,
        substituents: list = _DEFAULT_SUBSTITUENTS,
        anchors: list = ["[cH]", "C"],
        timeout: float = 60,
        *,
        _deadline: float | None = None,
        _timeout_budget: float | None = None,
        _stage: str = "direct forward PAS",
    ) -> list[Mol]:
        """Generate analogues via Positional Analogue Scanning.

        Adds one substituent at an eligible anchor position in the molecule.
        Each entry in ``substituents`` is either an element symbol
        (e.g. ``"F"``) or a fragment SMILES with a ``[*]`` dummy attachment
        point (e.g. ``"[*]C(F)(F)F"``). Based on the algorithm described in
        https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b02092.

        Implementation adapted from:
        https://practicalcheminformatics.blogspot.com/2020/04/positional-analogue-scanning.html

        :param Mol mol_in: Input RDKit molecule.
        :param list substituents: Element symbols or fragment SMILES (with
            ``[*]`` attachment point) to try as substituents.
        :param list anchors: SMARTS patterns identifying attachment positions.
        :param float timeout: Maximum runtime in seconds. Defaults to 60.

        :returns: List of unique, deduplicated analogue molecules.
        """
        deadline, timeout_budget = cls._operation_deadline(
            timeout, _deadline, _timeout_budget
        )
        cls._check_deadline(deadline, timeout_budget, _stage)

        for substituent in substituents:
            cls._check_deadline(deadline, timeout_budget, _stage)
            cls._validate_substituent(substituent)
            cls._check_deadline(deadline, timeout_budget, _stage)

        queries = []
        for anchor in anchors:
            cls._check_deadline(deadline, timeout_budget, _stage)
            if not isinstance(anchor, str) or not anchor:
                raise ValueError(f"Invalid anchor SMARTS: {anchor!r}")
            query = Chem.MolFromSmarts(anchor)
            cls._check_deadline(deadline, timeout_budget, _stage)
            if query is None or query.GetNumAtoms() == 0:
                raise ValueError(f"Invalid anchor SMARTS: {anchor!r}")
            queries.append(query)

        out_mol_list = []
        for substituent in substituents:
            cls._check_deadline(deadline, timeout_budget, _stage)
            for query in queries:
                cls._check_deadline(deadline, timeout_budget, _stage)
                match_atms = [x[0] for x in mol_in.GetSubstructMatches(query)]
                cls._check_deadline(deadline, timeout_budget, _stage)
                for idx in match_atms:
                    cls._check_deadline(deadline, timeout_budget, _stage)
                    result = cls._attach_substituent(
                        Chem.RWMol(mol_in),
                        idx,
                        substituent,
                        deadline=deadline,
                        timeout_budget=timeout_budget,
                        stage=_stage,
                    )
                    if result is not None:
                        out_mol_list.append(result)

        return cls._remove_duplicate(
            mol_in,
            out_mol_list,
            deadline=deadline,
            timeout_budget=timeout_budget,
            stage=_stage,
        )

    @classmethod
    def reverse_positional_analogue_scanning(
        cls,
        mol_in: Mol,
        substituents: list | None,
        timeout: float = 60,
        *,
        _deadline: float | None = None,
        _timeout_budget: float | None = None,
        _stage: str = "direct reverse PAS",
    ) -> list[Mol]:
        """Remove one peripheral group from the forward PAS vocabulary.

        :param Mol mol_in: Input RDKit molecule.
        :param list | None substituents: Forward PAS substituent vocabulary.
        :param float timeout: Maximum runtime in seconds. Defaults to 60.

        :returns: List of unique, deduplicated parent molecules.
        """
        if not substituents:
            return []

        deadline, timeout_budget = cls._operation_deadline(
            timeout, _deadline, _timeout_budget
        )
        cls._check_deadline(deadline, timeout_budget, _stage)
        out_mol_list = []
        for substituent in substituents:
            cls._check_deadline(deadline, timeout_budget, _stage)
            fragment, dummy_idx = cls._validate_substituent(substituent)
            cls._check_deadline(deadline, timeout_budget, _stage)
            if dummy_idx is None:
                atomic_num = fragment.GetAtomWithIdx(0).GetAtomicNum()
                for atom in mol_in.GetAtoms():
                    cls._check_deadline(deadline, timeout_budget, _stage)
                    if atom.GetAtomicNum() != atomic_num or atom.GetDegree() != 1:
                        continue
                    parent = cls._remove_atoms(
                        mol_in,
                        [atom.GetIdx()],
                        deadline=deadline,
                        timeout_budget=timeout_budget,
                        stage=_stage,
                    )
                    if parent is not None:
                        out_mol_list.append(parent)
            else:
                dummy = fragment.GetAtomWithIdx(dummy_idx)
                attachment_idx = dummy.GetNeighbors()[0].GetIdx()
                query = Chem.RWMol(fragment)
                query.RemoveAtom(dummy_idx)
                if dummy_idx < attachment_idx:
                    attachment_idx -= 1
                cls._check_deadline(deadline, timeout_budget, _stage)
                Chem.SanitizeMol(query)
                cls._check_deadline(deadline, timeout_budget, _stage)

                cls._check_deadline(deadline, timeout_budget, _stage)
                matches = mol_in.GetSubstructMatches(query)
                cls._check_deadline(deadline, timeout_budget, _stage)
                for match in matches:
                    cls._check_deadline(deadline, timeout_budget, _stage)
                    matched = set(match)
                    boundary = [
                        (atom_idx, neighbor.GetIdx())
                        for atom_idx in match
                        for neighbor in mol_in.GetAtomWithIdx(atom_idx).GetNeighbors()
                        if neighbor.GetIdx() not in matched
                    ]
                    if len(boundary) != 1 or boundary[0][0] != match[attachment_idx]:
                        continue

                    remove_indices = list(match)
                    if len(remove_indices) > 1:
                        remove_indices.remove(match[attachment_idx])
                    parent = cls._remove_atoms(
                        mol_in,
                        remove_indices,
                        deadline=deadline,
                        timeout_budget=timeout_budget,
                        stage=_stage,
                    )
                    if parent is not None and parent.GetNumAtoms() > 1:
                        out_mol_list.append(parent)

        return cls._remove_duplicate(
            mol_in,
            out_mol_list,
            deadline=deadline,
            timeout_budget=timeout_budget,
            stage=_stage,
        )

    @classmethod
    def nitrogen_walk(
        cls,
        mol_in: Mol,
        timeout: float = 60,
        *,
        _deadline: float | None = None,
        _timeout_budget: float | None = None,
        _stage: str = "direct nitrogen walk",
    ) -> list[Mol]:
        """Generate analogues by replacing aromatic CH atoms with nitrogen.

        Systematically walks aromatic carbon-hydrogen positions, replacing
        each one with nitrogen to produce aza-analogues.

        Implementation adapted from:
        https://practicalcheminformatics.blogspot.com/2020/04/positional-analogue-scanning.html

        :param Mol mol_in: Input RDKit molecule.
        :param float timeout: Maximum runtime in seconds. Defaults to 60.

        :returns: List of unique, deduplicated analogue molecules.
        """

        deadline, timeout_budget = cls._operation_deadline(
            timeout, _deadline, _timeout_budget
        )
        cls._check_deadline(deadline, timeout_budget, _stage)
        out_mol_list = []
        aromatic_cH = Chem.MolFromSmarts("[cH]")
        cls._check_deadline(deadline, timeout_budget, _stage)
        match_atms = [x[0] for x in mol_in.GetSubstructMatches(aromatic_cH)]
        cls._check_deadline(deadline, timeout_budget, _stage)
        for idx in match_atms:
            cls._check_deadline(deadline, timeout_budget, _stage)
            new_mol = Chem.RWMol(mol_in)
            cls._check_deadline(deadline, timeout_budget, _stage)
            new_mol.GetAtomWithIdx(idx).SetAtomicNum(7)
            cls._check_deadline(deadline, timeout_budget, _stage)
            try:
                Chem.SanitizeMol(new_mol)
            except MolSanitizeException:
                pass
            else:
                out_mol_list.append(new_mol)
            cls._check_deadline(deadline, timeout_budget, _stage)
        return cls._remove_duplicate(
            mol_in,
            out_mol_list,
            deadline=deadline,
            timeout_budget=timeout_budget,
            stage=_stage,
        )

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
        cls,
        mol: Mol,
        anchor_idx: int,
        substituent: str,
        *,
        deadline: float | None = None,
        timeout_budget: float | None = None,
        stage: str = "direct attachment",
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
        if deadline is not None and timeout_budget is not None:
            cls._check_deadline(deadline, timeout_budget, stage)
        fragment, dummy_idx = cls._validate_substituent(substituent)
        if deadline is not None and timeout_budget is not None:
            cls._check_deadline(deadline, timeout_budget, stage)
        if dummy_idx is not None:
            sidechain = cls._prep_sidechain(substituent)
            if deadline is not None and timeout_budget is not None:
                cls._check_deadline(deadline, timeout_budget, stage)
            new_mol = Chem.RWMol(CombineMols(mol, sidechain))
            if deadline is not None and timeout_budget is not None:
                cls._check_deadline(deadline, timeout_budget, stage)
            attach_atm = next(
                atom.GetIdx()
                for atom in new_mol.GetAtoms()
                if atom.GetAtomMapNum() == 1
            )
            new_mol.AddBond(anchor_idx, attach_atm, order=Chem.rdchem.BondType.SINGLE)
            for atom in new_mol.GetAtoms():
                atom.SetAtomMapNum(0)
        else:
            new_mol = Chem.RWMol(mol)
            new_idx = new_mol.AddAtom(Chem.Atom(fragment.GetAtomWithIdx(0)))
            new_mol.AddBond(anchor_idx, new_idx, order=Chem.rdchem.BondType.SINGLE)

        if deadline is not None and timeout_budget is not None:
            cls._check_deadline(deadline, timeout_budget, stage)
        result = new_mol
        try:
            Chem.SanitizeMol(new_mol)
        except MolSanitizeException:
            result = None
        if deadline is not None and timeout_budget is not None:
            cls._check_deadline(deadline, timeout_budget, stage)
        return result

    @classmethod
    def _remove_atoms(
        cls,
        mol: Mol,
        atom_indices: list[int],
        *,
        deadline: float | None = None,
        timeout_budget: float | None = None,
        stage: str = "direct reverse PAS",
    ) -> Mol | None:
        if deadline is not None and timeout_budget is not None:
            cls._check_deadline(deadline, timeout_budget, stage)
        parent = Chem.RWMol(mol)
        for atom_idx in sorted(atom_indices, reverse=True):
            cls._check_optional_deadline(deadline, timeout_budget, stage)
            parent.RemoveAtom(atom_idx)
        if parent.GetNumAtoms() == 0:
            return None
        cls._check_optional_deadline(deadline, timeout_budget, stage)
        result = parent
        try:
            Chem.SanitizeMol(parent)
        except MolSanitizeException:
            result = None
        cls._check_optional_deadline(deadline, timeout_budget, stage)
        if result is None:
            return None
        if len(Chem.GetMolFrags(parent)) != 1:
            return None
        cls._check_optional_deadline(deadline, timeout_budget, stage)
        return parent

    @classmethod
    def _remove_duplicate(
        cls,
        target: Mol,
        analogues: list[Mol],
        *,
        deadline: float | None = None,
        timeout_budget: float | None = None,
        stage: str = "finalization",
    ) -> list[Mol]:
        cls._check_optional_deadline(deadline, timeout_budget, stage)
        seen = {Chem.MolToSmiles(target)}
        cls._check_optional_deadline(deadline, timeout_budget, stage)
        unique = []
        for analogue in analogues:
            cls._check_optional_deadline(deadline, timeout_budget, stage)
            if any(atom.GetAtomicNum() == 0 for atom in analogue.GetAtoms()):
                raise RuntimeError("Generated analogue contains a dummy atom")
            smiles = Chem.MolToSmiles(analogue)
            cls._check_optional_deadline(deadline, timeout_budget, stage)
            if smiles in seen:
                continue
            seen.add(smiles)
            unique.append(analogue)
        return unique

    @classmethod
    def _sample_branch(
        cls,
        mol_in: Mol,
        parent_pool: list[Mol],
        enabled_strategies: list[str],
        pas_params: dict | None,
        num_sample: int,
        seed: int,
        *,
        overall_deadline: float,
        generation_timeout: float,
        stage: str,
    ) -> list[Mol]:
        if num_sample == 0 or not parent_pool or not enabled_strategies:
            return []

        branch_rng = random.Random(seed)
        query_smiles = Chem.MolToSmiles(mol_in)
        seen: set[str] = {query_smiles}
        accepted: list[Mol] = []
        attempt_limit = 2 * num_sample
        for _ in range(attempt_limit):
            cls._check_deadline(overall_deadline, generation_timeout, stage)
            parent = branch_rng.choice(parent_pool)
            strategy = (
                enabled_strategies[0]
                if len(enabled_strategies) == 1
                else branch_rng.choice(enabled_strategies)
            )
            if strategy == "pas":
                candidate = cls._attempt_pas(
                    parent,
                    pas_params,
                    branch_rng,
                    deadline=overall_deadline,
                    timeout_budget=generation_timeout,
                    stage=stage,
                )
            elif strategy == "reverse_pas":
                candidate = cls._attempt_reverse_pas(
                    parent,
                    pas_params,
                    branch_rng,
                    deadline=overall_deadline,
                    timeout_budget=generation_timeout,
                    stage=stage,
                )
            else:
                candidate = cls._attempt_nitrogen_walk(
                    parent,
                    branch_rng,
                    deadline=overall_deadline,
                    timeout_budget=generation_timeout,
                    stage=stage,
                )
            if candidate is None:
                continue
            smiles = Chem.MolToSmiles(candidate)
            if smiles in seen:
                continue
            seen.add(smiles)
            accepted.append(candidate)
            if len(accepted) >= num_sample:
                break
        return accepted

    @classmethod
    def _attempt_pas(
        cls,
        parent: Mol,
        pas_params: dict,
        rng: random.Random,
        *,
        deadline: float,
        timeout_budget: float,
        stage: str,
    ) -> Mol | None:
        substituents = pas_params.get("substituents", _DEFAULT_SUBSTITUENTS)
        anchors = pas_params.get("anchors", ["[cH]", "C"])
        if not substituents or not anchors:
            return None
        substituent = rng.choice(substituents)
        anchor = rng.choice(anchors)
        query = Chem.MolFromSmarts(anchor)
        if query is None or query.GetNumAtoms() == 0:
            raise ValueError(f"Invalid anchor SMARTS: {anchor!r}")
        cls._check_deadline(deadline, timeout_budget, stage)
        match_atms = [x[0] for x in parent.GetSubstructMatches(query)]
        if not match_atms:
            return None
        idx = rng.choice(match_atms)
        return cls._attach_substituent(
            Chem.RWMol(parent),
            idx,
            substituent,
            deadline=deadline,
            timeout_budget=timeout_budget,
            stage=stage,
        )

    @classmethod
    def _attempt_reverse_pas(
        cls,
        parent: Mol,
        pas_params: dict,
        rng: random.Random,
        *,
        deadline: float,
        timeout_budget: float,
        stage: str,
    ) -> Mol | None:
        substituents = pas_params.get("substituents", _DEFAULT_SUBSTITUENTS)
        if not substituents:
            return None
        substituent = rng.choice(substituents)
        fragment, dummy_idx = cls._validate_substituent(substituent)
        cls._check_deadline(deadline, timeout_budget, stage)
        if dummy_idx is None:
            atomic_num = fragment.GetAtomWithIdx(0).GetAtomicNum()
            eligible = [
                atom.GetIdx()
                for atom in parent.GetAtoms()
                if atom.GetAtomicNum() == atomic_num and atom.GetDegree() == 1
            ]
            if not eligible:
                return None
            atom_idx = rng.choice(eligible)
            return cls._remove_atoms(
                parent,
                [atom_idx],
                deadline=deadline,
                timeout_budget=timeout_budget,
                stage=stage,
            )

        dummy = fragment.GetAtomWithIdx(dummy_idx)
        attachment_idx = dummy.GetNeighbors()[0].GetIdx()
        query = Chem.RWMol(fragment)
        query.RemoveAtom(dummy_idx)
        if dummy_idx < attachment_idx:
            attachment_idx -= 1
        Chem.SanitizeMol(query)
        cls._check_deadline(deadline, timeout_budget, stage)

        eligible_matches = []
        for match in parent.GetSubstructMatches(query):
            matched = set(match)
            boundary = [
                (atom_idx, neighbor.GetIdx())
                for atom_idx in match
                for neighbor in parent.GetAtomWithIdx(atom_idx).GetNeighbors()
                if neighbor.GetIdx() not in matched
            ]
            if len(boundary) == 1 and boundary[0][0] == match[attachment_idx]:
                eligible_matches.append(match)
        if not eligible_matches:
            return None
        match = rng.choice(eligible_matches)
        remove_indices = list(match)
        if len(remove_indices) > 1:
            remove_indices.remove(match[attachment_idx])
        result = cls._remove_atoms(
            parent,
            remove_indices,
            deadline=deadline,
            timeout_budget=timeout_budget,
            stage=stage,
        )
        if result is None or result.GetNumAtoms() <= 1:
            return None
        return result

    @classmethod
    def _attempt_nitrogen_walk(
        cls,
        parent: Mol,
        rng: random.Random,
        *,
        deadline: float,
        timeout_budget: float,
        stage: str,
    ) -> Mol | None:
        aromatic_cH = Chem.MolFromSmarts("[cH]")
        cls._check_deadline(deadline, timeout_budget, stage)
        match_atms = [x[0] for x in parent.GetSubstructMatches(aromatic_cH)]
        if not match_atms:
            return None
        idx = rng.choice(match_atms)
        new_mol = Chem.RWMol(parent)
        new_mol.GetAtomWithIdx(idx).SetAtomicNum(7)
        cls._check_deadline(deadline, timeout_budget, stage)
        try:
            Chem.SanitizeMol(new_mol)
        except MolSanitizeException:
            return None
        return new_mol

    @classmethod
    def _validate_non_negative_int(cls, value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _validate_int(cls, value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    @classmethod
    def _validate_timeout(cls, timeout: float, name: str) -> float:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError(f"{name} must be a finite, non-negative number")
        return float(timeout)

    @classmethod
    def _operation_deadline(
        cls,
        timeout: float,
        deadline: float | None,
        timeout_budget: float | None,
    ) -> tuple[float, float]:
        timeout = cls._validate_timeout(timeout, "timeout")
        if deadline is None:
            return time.monotonic() + timeout, timeout
        if timeout_budget is None:
            raise ValueError("timeout_budget is required with an absolute deadline")
        return deadline, timeout_budget

    @classmethod
    def _aggregate_strategy_deadline(
        cls,
        start: float,
        overall_deadline: float,
        generation_timeout: float,
        strategy_timeout: float,
    ) -> tuple[float, float]:
        strategy_timeout = cls._validate_timeout(strategy_timeout, "timeout")
        strategy_deadline = start + strategy_timeout
        if strategy_deadline < overall_deadline:
            return strategy_deadline, strategy_timeout
        return overall_deadline, generation_timeout

    @classmethod
    def _check_optional_deadline(
        cls,
        deadline: float | None,
        timeout_budget: float | None,
        stage: str,
    ) -> None:
        if deadline is not None and timeout_budget is not None:
            cls._check_deadline(deadline, timeout_budget, stage)

    @classmethod
    def _check_deadline(
        cls, deadline: float, timeout_budget: float, stage: str
    ) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Generation exceeded the {timeout_budget:g}-second budget during "
                f"{stage}; no partial results are returned. To continue, increase "
                "generation_timeout, disable a strategy, or narrow the vocabulary."
            )

    @classmethod
    def _validate_substituent(cls, substituent: str) -> tuple[Mol, int | None]:
        if not isinstance(substituent, str) or not substituent:
            raise ValueError(f"Invalid substituent: {substituent!r}")

        fragment = Chem.MolFromSmiles(substituent)
        if fragment is None or fragment.GetNumAtoms() == 0:
            raise ValueError(f"Invalid substituent: {substituent!r}")

        dummy_atoms = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
        if not dummy_atoms:
            if fragment.GetNumAtoms() != 1:
                raise ValueError(
                    f"Substituent {substituent!r} must contain exactly one atom "
                    "or one dummy attachment atom"
                )
            return fragment, None

        if len(dummy_atoms) != 1 or dummy_atoms[0].GetDegree() != 1:
            raise ValueError(
                f"Substituent {substituent!r} must contain exactly one atom "
                "or one dummy attachment atom"
            )
        return fragment, dummy_atoms[0].GetIdx()

    @classmethod
    def _prep_sidechain(cls, smi: str) -> Mol:
        """Prepare a fragment SMILES as an RDKit sidechain ready for bonding.

        Removes the ``[*]`` dummy attachment atom and marks its neighbor with
        atom-map number 1 so it can be found by :meth:`_attach_substituent`.
        """
        mol, dummy_idx = cls._validate_substituent(smi)
        if dummy_idx is None:
            raise ValueError(f"Substituent {smi!r} has no dummy attachment atom")

        rw_mol = Chem.RWMol(mol)
        dummy_atom = rw_mol.GetAtomWithIdx(dummy_idx)
        dummy_atom.GetNeighbors()[0].SetAtomMapNum(1)
        rw_mol.RemoveAtom(dummy_idx)
        Chem.SanitizeMol(rw_mol)
        return rw_mol
