"""Tests for matcha.explainability.analogue_generator.AnalogueGenerator."""

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from matcha.explainability.analogue_generator import AnalogueGenerator


# ===================================================================
# AnalogueGenerator – positional analogue scanning
# ===================================================================


class TestPositionalAnalogueScanning:
    """Tests for AnalogueGenerator.positional_analogue_scanning."""

    def test_returns_list(self, single_mol):
        result = AnalogueGenerator.positional_analogue_scanning(single_mol)
        assert isinstance(result, list)

    def test_returns_mol_objects(self, single_mol):
        result = AnalogueGenerator.positional_analogue_scanning(single_mol)
        assert all(isinstance(m, Mol) for m in result)

    def test_generates_analogues(self, single_mol):
        result = AnalogueGenerator.positional_analogue_scanning(single_mol)
        assert len(result) > 0

    def test_no_duplicate_of_input(self, single_mol):
        input_smi = Chem.MolToSmiles(single_mol)
        result = AnalogueGenerator.positional_analogue_scanning(single_mol)
        result_smiles = [Chem.MolToSmiles(m) for m in result]
        assert input_smi not in result_smiles

    def test_custom_substituents(self, single_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            single_mol, substituents=["F"], anchors=["[cH]"], num_sub=1
        )
        assert isinstance(result, list)
        # Should get some F-substituted analogues
        for m in result:
            smi = Chem.MolToSmiles(m)
            assert "F" in smi or len(result) == 0

    def test_custom_anchors(self, single_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            single_mol, substituents=["Cl"], anchors=["C"], num_sub=1
        )
        assert isinstance(result, list)

    def test_num_sub_parameter(self, benzene_mol):
        result_1 = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol, substituents=["F"], anchors=["[cH]"], num_sub=1
        )
        result_2 = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol, substituents=["F"], anchors=["[cH]"], num_sub=2
        )
        # More combinations with num_sub=2
        assert len(result_2) >= len(result_1)

    def test_fragment_substituent_generates_analogues(self, benzene_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol, substituents=["[*]C(F)(F)F"], anchors=["[cH]"], num_sub=1
        )
        assert len(result) > 0
        for m in result:
            smi = Chem.MolToSmiles(m)
            assert "F" in smi

    def test_mixed_substituents(self, benzene_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol, substituents=["F", "[*]C(F)(F)F"], anchors=["[cH]"], num_sub=1
        )
        assert len(result) > 0

    def test_timeout_respected(self, single_mol):
        """With a very short timeout, should still return a list (possibly truncated)."""
        result = AnalogueGenerator.positional_analogue_scanning(
            single_mol,
            timeout=0,  # immediate timeout
        )
        assert isinstance(result, list)

    def test_all_results_sanitizable(self, single_mol):
        result = AnalogueGenerator.positional_analogue_scanning(single_mol)
        for m in result:
            smi = Chem.MolToSmiles(m)
            round_trip = Chem.MolFromSmiles(smi)
            assert round_trip is not None


# ===================================================================
# AnalogueGenerator – _attach_substituent
# ===================================================================


class TestAttachSubstituent:
    """Tests for AnalogueGenerator._attach_substituent."""

    def test_element_path(self, benzene_mol):
        result = AnalogueGenerator._attach_substituent(benzene_mol, 0, "F")
        assert result is not None
        smi = Chem.MolToSmiles(result)
        assert "F" in smi

    def test_fragment_path(self, benzene_mol):
        result = AnalogueGenerator._attach_substituent(benzene_mol, 0, "[*]C(F)(F)F")
        assert result is not None
        smi = Chem.MolToSmiles(result)
        assert "F" in smi

    def test_invalid_smiles_returns_none(self, benzene_mol):
        result = AnalogueGenerator._attach_substituent(
            benzene_mol, 0, "NOT_VALID_SMILES"
        )
        assert result is None

    def test_out_of_bounds_anchor_returns_none(self, benzene_mol):
        result = AnalogueGenerator._attach_substituent(benzene_mol, 9999, "F")
        assert result is None


# ===================================================================
# AnalogueGenerator – nitrogen walk
# ===================================================================


class TestNitrogenWalk:
    """Tests for AnalogueGenerator.nitrogen_walk."""

    def test_returns_list(self, benzene_mol):
        result = AnalogueGenerator.nitrogen_walk(benzene_mol)
        assert isinstance(result, list)

    def test_returns_mol_objects(self, benzene_mol):
        result = AnalogueGenerator.nitrogen_walk(benzene_mol)
        assert all(isinstance(m, Mol) for m in result)

    def test_generates_analogues_for_aromatic(self, benzene_mol):
        result = AnalogueGenerator.nitrogen_walk(benzene_mol)
        assert len(result) > 0

    def test_no_duplicate_of_input(self, benzene_mol):
        input_smi = Chem.MolToSmiles(benzene_mol)
        result = AnalogueGenerator.nitrogen_walk(benzene_mol)
        result_smiles = [Chem.MolToSmiles(m) for m in result]
        assert input_smi not in result_smiles

    def test_contains_nitrogen(self, benzene_mol):
        result = AnalogueGenerator.nitrogen_walk(benzene_mol)
        for m in result:
            atom_nums = [a.GetAtomicNum() for a in m.GetAtoms()]
            assert 7 in atom_nums  # nitrogen

    def test_num_sub_1(self, benzene_mol):
        result = AnalogueGenerator.nitrogen_walk(benzene_mol, num_sub=1)
        for m in result:
            n_count = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7)
            assert n_count >= 1

    def test_num_sub_2(self, benzene_mol):
        result = AnalogueGenerator.nitrogen_walk(benzene_mol, num_sub=2)
        for m in result:
            n_count = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 7)
            assert n_count >= 2

    def test_no_analogues_without_aromatic_cH(self):
        """Molecule without aromatic cH should produce no nitrogen walk analogues."""
        aliphatic = Chem.MolFromSmiles("CC(=O)O")
        result = AnalogueGenerator.nitrogen_walk(aliphatic)
        assert result == []

    def test_timeout_respected(self, benzene_mol):
        result = AnalogueGenerator.nitrogen_walk(benzene_mol, timeout=0)
        assert isinstance(result, list)


# ===================================================================
# AnalogueGenerator – decompose
# ===================================================================


class TestDecompose:
    """Tests for AnalogueGenerator.decompose."""

    def test_returns_list(self, single_mol):
        result = AnalogueGenerator.decompose(single_mol)
        assert isinstance(result, list)

    def test_returns_mol_objects(self, single_mol):
        result = AnalogueGenerator.decompose(single_mol)
        assert all(isinstance(m, Mol) for m in result)

    def test_decompose_produces_fragments(self, small_mol_list):
        # Use a real drug-like molecule from testing_data.csv — decomposes
        # into BRICS fragments
        result = AnalogueGenerator.decompose(small_mol_list[0])
        assert len(result) > 0

    def test_fragments_are_valid_smiles(self, small_mol_list):
        result = AnalogueGenerator.decompose(small_mol_list[0])
        for m in result:
            smi = Chem.MolToSmiles(m)
            assert Chem.MolFromSmiles(smi) is not None

    def test_simple_molecule_decompose(self):
        """A simple molecule should decompose."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        result = AnalogueGenerator.decompose(mol)
        assert isinstance(result, list)


# ===================================================================
# AnalogueGenerator – generate_analogues (combined)
# ===================================================================


class TestGenerateAnalogues:
    """Tests for the combined AnalogueGenerator.generate_analogues method."""

    def test_returns_list(self, single_mol):
        result = AnalogueGenerator.generate_analogues(single_mol)
        assert isinstance(result, list)

    def test_returns_mol_objects(self, single_mol):
        result = AnalogueGenerator.generate_analogues(single_mol)
        assert all(isinstance(m, Mol) for m in result)

    def test_all_unique(self, single_mol):
        result = AnalogueGenerator.generate_analogues(single_mol)
        smiles = [Chem.MolToSmiles(m) for m in result]
        assert len(smiles) == len(set(smiles))

    def test_input_not_in_output(self, single_mol):
        input_smi = Chem.MolToSmiles(single_mol)
        result = AnalogueGenerator.generate_analogues(single_mol)
        result_smiles = [Chem.MolToSmiles(m) for m in result]
        assert input_smi not in result_smiles

    def test_disable_positional(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params=None,
            nitrogen_walk_params={"num_sub": 1},
        )
        assert isinstance(result, list)

    def test_disable_all_returns_empty(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params=None,
            nitrogen_walk_params=None,
        )
        assert result == []

    def test_all_results_valid_molecules(self, single_mol):
        result = AnalogueGenerator.generate_analogues(single_mol)
        for m in result:
            smi = Chem.MolToSmiles(m)
            assert smi is not None
            assert Chem.MolFromSmiles(smi) is not None


# ===================================================================
# AnalogueGenerator – generate_analogues (scaffold + pairwise)
# ===================================================================


class TestGenerateAnaloguesNewBehavior:
    """Tests for scaffold-based and pairwise logic in generate_analogues."""

    def test_scaffold_analogues_included(self, single_mol):
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

        # Use "C" (atomic num 6 <= 8) so the element filter does not skip it
        pos_params = {"substituents": ["C"], "anchors": ["[cH]"], "num_sub": 1}
        scaffold = GetScaffoldForMol(single_mol)
        scaffold_pas = AnalogueGenerator.positional_analogue_scanning(
            scaffold, **pos_params
        )
        scaffold_smi = {Chem.MolToSmiles(m) for m in scaffold_pas}
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params=pos_params,
            nitrogen_walk_params=None,
        )
        result_smi = {Chem.MolToSmiles(m) for m in result}
        assert len(scaffold_smi & result_smi) > 0

    def test_pairwise_pas_pas_generates_analogues(self, single_mol):
        # Use "C" (atomic num 6 <= 8) so the element filter does not skip it
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params={
                "substituents": ["C"],
                "anchors": ["[cH]"],
                "num_sub": 1,
            },
            nitrogen_walk_params=None,
        )
        assert len(result) > 0

    def test_pairwise_pas_nw_generates_analogues(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params={
                "substituents": ["F"],
                "anchors": ["[cH]"],
                "num_sub": 1,
            },
            nitrogen_walk_params={"num_sub": 1},
        )
        assert len(result) > 0

    def test_all_unique_across_all_sources(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params={
                "substituents": ["F"],
                "anchors": ["[cH]"],
                "num_sub": 1,
            },
            nitrogen_walk_params={"num_sub": 1},
        )
        smiles = [Chem.MolToSmiles(m) for m in result]
        assert len(smiles) == len(set(smiles))

    def test_input_excluded(self, single_mol):
        input_smi = Chem.MolToSmiles(single_mol)
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params={
                "substituents": ["F"],
                "anchors": ["[cH]"],
                "num_sub": 1,
            },
            nitrogen_walk_params={"num_sub": 1},
        )
        result_smi = [Chem.MolToSmiles(m) for m in result]
        assert input_smi not in result_smi


# ===================================================================
# AnalogueGenerator – private helpers
# ===================================================================


class TestAnalogueGeneratorHelpers:
    """Tests for private utility methods."""

    def test_remove_duplicate_excludes_target(self, single_mol):
        dup_list = [Chem.RWMol(single_mol)]  # copy of input
        result = AnalogueGenerator._remove_duplicate(single_mol, dup_list)
        assert result == []

    def test_remove_duplicate_keeps_unique(self, single_mol):
        other = Chem.MolFromSmiles("CCO")
        result = AnalogueGenerator._remove_duplicate(single_mol, [other])
        assert len(result) == 1

    def test_remove_duplicate_deduplicates(self, single_mol):
        m1 = Chem.MolFromSmiles("CCO")
        m2 = Chem.MolFromSmiles("CCO")
        result = AnalogueGenerator._remove_duplicate(single_mol, [m1, m2])
        assert len(result) == 1
