"""Tests for matcha.explainability.analogue_generator.AnalogueGenerator."""

import os
import random
import subprocess
import sys

import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, MolSanitizeException

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

    @pytest.mark.parametrize(
        ("substituent", "expected_smiles"),
        [
            ("F", "Fc1ccccc1"),
            ("Cl", "Clc1ccccc1"),
            ("Br", "Brc1ccccc1"),
            ("I", "Ic1ccccc1"),
        ],
    )
    def test_custom_substituents(self, benzene_mol, substituent, expected_smiles):
        result = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol,
            substituents=[substituent],
            anchors=["[cH]"],
        )
        assert [Chem.MolToSmiles(mol) for mol in result] == [expected_smiles]

    @pytest.mark.parametrize("substituent", ["Xx", "", "C(C"])
    def test_rejects_invalid_substituent(self, benzene_mol, substituent):
        with pytest.raises(ValueError, match="Invalid substituent"):
            AnalogueGenerator.positional_analogue_scanning(
                benzene_mol,
                substituents=[substituent],
                anchors=["[cH]"],
            )

    def test_rejects_multi_atom_substituent_without_dummy(self, benzene_mol):
        with pytest.raises(ValueError, match="exactly one atom or one dummy"):
            AnalogueGenerator.positional_analogue_scanning(
                benzene_mol,
                substituents=["CC"],
                anchors=["[cH]"],
            )

    def test_rejects_substituent_with_multiple_dummies(self, benzene_mol):
        with pytest.raises(ValueError, match="exactly one atom or one dummy"):
            AnalogueGenerator.positional_analogue_scanning(
                benzene_mol,
                substituents=["[*]C[*]"],
                anchors=["[cH]"],
            )

    @pytest.mark.parametrize("num_sub", [0, -1, 1.5, True])
    def test_rejects_num_sub_kwarg(self, benzene_mol, num_sub):
        with pytest.raises(TypeError):
            AnalogueGenerator.positional_analogue_scanning(
                benzene_mol,
                substituents=["F"],
                anchors=["[cH]"],
                num_sub=num_sub,
            )

    def test_rejects_invalid_anchor(self, benzene_mol):
        with pytest.raises(ValueError, match="Invalid anchor SMARTS"):
            AnalogueGenerator.positional_analogue_scanning(
                benzene_mol,
                substituents=["F"],
                anchors=["[cH]", "["],
            )

    def test_outputs_contain_no_dummy_atoms(self, benzene_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol,
            substituents=["[*]C(F)(F)F"],
            anchors=["[cH]"],
        )
        assert result
        assert all(
            atom.GetAtomicNum() != 0
            for analogue in result
            for atom in analogue.GetAtoms()
        )

    def test_sanitization_failure_skips_only_invalid_candidate(
        self, benzene_mol, monkeypatch
    ):
        sanitize = Chem.SanitizeMol
        call_count = 0

        def fail_once(mol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MolSanitizeException("invalid candidate")
            return sanitize(mol)

        monkeypatch.setattr(Chem, "SanitizeMol", fail_once)
        result = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol,
            substituents=["F"],
            anchors=["[cH]"],
        )

        assert [Chem.MolToSmiles(mol) for mol in result] == ["Fc1ccccc1"]

    def test_unexpected_attachment_failure_propagates(self, benzene_mol, monkeypatch):
        def fail(_mol):
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(Chem, "SanitizeMol", fail)
        with pytest.raises(RuntimeError, match="unexpected failure"):
            AnalogueGenerator.positional_analogue_scanning(
                benzene_mol,
                substituents=["F"],
                anchors=["[cH]"],
            )

    def test_custom_anchors(self, single_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            single_mol, substituents=["Cl"], anchors=["C"]
        )
        assert isinstance(result, list)

    def test_generates_one_output_per_eligible_site(self, benzene_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol, substituents=["F"], anchors=["[cH]"]
        )
        assert [Chem.MolToSmiles(mol) for mol in result] == ["Fc1ccccc1"]

    def test_fragment_substituent_generates_analogues(self, benzene_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol, substituents=["[*]C(F)(F)F"], anchors=["[cH]"]
        )
        assert len(result) > 0
        for m in result:
            smi = Chem.MolToSmiles(m)
            assert "F" in smi

    def test_mixed_substituents(self, benzene_mol):
        result = AnalogueGenerator.positional_analogue_scanning(
            benzene_mol, substituents=["F", "[*]C(F)(F)F"], anchors=["[cH]"]
        )
        assert len(result) > 0

    def test_timeout_raises_during_enumeration(self, single_mol, monkeypatch):
        now = 0.0

        def expire_during_attachment(cls, mol, anchor_idx, substituent, **kwargs):
            nonlocal now
            now = 2.0
            return Chem.RWMol(mol)

        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: now
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attach_substituent",
            classmethod(expire_during_attachment),
        )

        with pytest.raises(TimeoutError, match="no partial results"):
            AnalogueGenerator.positional_analogue_scanning(single_mol, timeout=1)

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

    def test_invalid_smiles_raises_value_error(self, benzene_mol):
        with pytest.raises(ValueError, match="Invalid substituent"):
            AnalogueGenerator._attach_substituent(benzene_mol, 0, "NOT_VALID_SMILES")

    def test_out_of_bounds_anchor_propagates(self, benzene_mol):
        with pytest.raises(RuntimeError):
            AnalogueGenerator._attach_substituent(benzene_mol, 9999, "F")

    def test_deadline_checked_before_sanitization(self, benzene_mol, monkeypatch):
        sanitize_calls = 0

        def record_sanitize(mol):
            nonlocal sanitize_calls
            sanitize_calls += 1
            return mol

        monkeypatch.setattr(Chem, "SanitizeMol", record_sanitize)
        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: 2.0
        )

        with pytest.raises(TimeoutError, match="attachment"):
            AnalogueGenerator._attach_substituent(
                benzene_mol,
                0,
                "F",
                deadline=1.0,
                timeout_budget=1.0,
                stage="attachment",
            )

        assert sanitize_calls == 0

    def test_deadline_checked_after_sanitization(self, benzene_mol, monkeypatch):
        now = 0.0
        sanitize = Chem.SanitizeMol

        def advance_clock(mol):
            nonlocal now
            result = sanitize(mol)
            now = 2.0
            return result

        monkeypatch.setattr(Chem, "SanitizeMol", advance_clock)
        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: now
        )

        with pytest.raises(TimeoutError, match="attachment"):
            AnalogueGenerator._attach_substituent(
                benzene_mol,
                0,
                "F",
                deadline=1.0,
                timeout_budget=1.0,
                stage="attachment",
            )


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

    def test_generates_one_output_per_eligible_site(self, benzene_mol):
        # Benzene has six symmetric aromatic CH sites; after dedup exactly one
        # unique pyridine is produced.
        result = AnalogueGenerator.nitrogen_walk(benzene_mol)
        assert [Chem.MolToSmiles(m) for m in result] == ["c1ccncc1"]

    def test_rejects_num_sub_kwarg(self, benzene_mol):
        with pytest.raises(TypeError):
            AnalogueGenerator.nitrogen_walk(benzene_mol, num_sub=1)

    def test_no_analogues_without_aromatic_cH(self):
        """Molecule without aromatic cH should produce no nitrogen walk analogues."""
        aliphatic = Chem.MolFromSmiles("CC(=O)O")
        result = AnalogueGenerator.nitrogen_walk(aliphatic)
        assert result == []

    def test_sanitization_failure_skips_only_invalid_candidate(
        self, benzene_mol, monkeypatch
    ):
        sanitize = Chem.SanitizeMol
        call_count = 0

        def fail_once(mol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MolSanitizeException("invalid candidate")
            return sanitize(mol)

        monkeypatch.setattr(Chem, "SanitizeMol", fail_once)
        result = AnalogueGenerator.nitrogen_walk(benzene_mol)

        assert [Chem.MolToSmiles(mol) for mol in result] == ["c1ccncc1"]

    def test_unexpected_sanitization_failure_propagates(self, benzene_mol, monkeypatch):
        def fail(_mol):
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(Chem, "SanitizeMol", fail)
        with pytest.raises(RuntimeError, match="unexpected failure"):
            AnalogueGenerator.nitrogen_walk(benzene_mol)

    def test_timeout_raises_during_enumeration(self, benzene_mol, monkeypatch):
        now = 0.0
        sanitize = Chem.SanitizeMol

        def expire_during_sanitization(mol):
            nonlocal now
            result = sanitize(mol)
            now = 2.0
            return result

        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: now
        )
        monkeypatch.setattr(Chem, "SanitizeMol", expire_during_sanitization)

        with pytest.raises(TimeoutError, match="no partial results"):
            AnalogueGenerator.nitrogen_walk(benzene_mol, timeout=1)


# ===================================================================
# AnalogueGenerator – reverse positional analogue scanning
# ===================================================================


class TestReversePositionalAnalogueScanning:
    """Tests for reverse positional analogue scanning."""

    @pytest.mark.parametrize(
        ("smiles", "expected_smiles"),
        [("Cc1ccccc1", "c1ccccc1"), ("CCc1ccccc1", "Cc1ccccc1")],
    )
    def test_removes_one_terminal_bare_atom(self, smiles, expected_smiles):
        result = AnalogueGenerator.reverse_positional_analogue_scanning(
            Chem.MolFromSmiles(smiles), substituents=["C"]
        )

        assert [Chem.MolToSmiles(mol) for mol in result] == [expected_smiles]

    def test_does_not_remove_ring_atom(self):
        result = AnalogueGenerator.reverse_positional_analogue_scanning(
            Chem.MolFromSmiles("C1CCCCC1"), substituents=["C"]
        )

        assert result == []

    def test_does_not_remove_internal_chain_atom(self):
        result = AnalogueGenerator.reverse_positional_analogue_scanning(
            Chem.MolFromSmiles("CCC"), substituents=["C"]
        )

        assert [Chem.MolToSmiles(mol) for mol in result] == ["CC"]

    def test_removes_only_one_matching_group_per_candidate(self):
        result = AnalogueGenerator.reverse_positional_analogue_scanning(
            Chem.MolFromSmiles("Cc1ccc(C)cc1"), substituents=["C"]
        )

        assert [Chem.MolToSmiles(mol) for mol in result] == ["Cc1ccccc1"]
        assert "c1ccccc1" not in [Chem.MolToSmiles(mol) for mol in result]

    @pytest.mark.parametrize(
        ("smiles", "substituent", "expected_smiles"),
        [
            ("Oc1ccccc1", "[*]O", "c1ccccc1"),
            ("COc1ccccc1", "[*]OC", "Oc1ccccc1"),
        ],
    )
    def test_removes_one_peripheral_fragment(
        self, smiles, substituent, expected_smiles
    ):
        result = AnalogueGenerator.reverse_positional_analogue_scanning(
            Chem.MolFromSmiles(smiles), substituents=[substituent]
        )

        assert [Chem.MolToSmiles(mol) for mol in result] == [expected_smiles]

    @pytest.mark.parametrize(
        ("smiles", "substituent"), [("CCOCC", "[*]OC"), ("CO", "[*]O")]
    )
    def test_rejects_non_peripheral_or_empty_parent(self, smiles, substituent):
        result = AnalogueGenerator.reverse_positional_analogue_scanning(
            Chem.MolFromSmiles(smiles), substituents=[substituent]
        )

        assert result == []

    @pytest.mark.parametrize("substituents", [None, []])
    def test_empty_vocabulary_is_a_silent_noop(self, benzene_mol, substituents):
        result = AnalogueGenerator.reverse_positional_analogue_scanning(
            benzene_mol, substituents=substituents
        )

        assert result == []

    def test_timeout_raises_during_enumeration(self, benzene_mol, monkeypatch):
        now = 0.0

        def expire_during_removal(cls, mol, atom_indices, **kwargs):
            nonlocal now
            now = 2.0
            return Chem.RWMol(mol)

        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: now
        )
        monkeypatch.setattr(
            AnalogueGenerator, "_remove_atoms", classmethod(expire_during_removal)
        )

        with pytest.raises(TimeoutError, match="no partial results"):
            AnalogueGenerator.reverse_positional_analogue_scanning(
                Chem.MolFromSmiles("Cc1ccccc1"),
                substituents=["C"],
                timeout=1,
            )


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
            nitrogen_walk_params={},
        )
        assert isinstance(result, list)

    def test_disable_all_returns_empty(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params=None,
            nitrogen_walk_params=None,
        )
        assert result == []

    def test_reverse_can_be_disabled(self):
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        params = {"substituents": ["C"], "anchors": []}

        enabled = AnalogueGenerator.generate_analogues(
            mol,
            positional_analogue_scanning_params=params,
            nitrogen_walk_params=None,
        )
        disabled = AnalogueGenerator.generate_analogues(
            mol,
            positional_analogue_scanning_params=params,
            nitrogen_walk_params=None,
            reverse_positional_analogue_scanning=False,
        )

        assert "c1ccccc1" in [Chem.MolToSmiles(analogue) for analogue in enabled]
        assert "c1ccccc1" not in [Chem.MolToSmiles(analogue) for analogue in disabled]

    def test_reverse_noops_when_forward_vocabulary_is_unavailable(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params=None,
            nitrogen_walk_params=None,
            reverse_positional_analogue_scanning=True,
        )

        assert result == []

    def test_reverse_runs_once_on_the_original_query(self, benzene_mol, monkeypatch):
        reverse_inputs = []

        monkeypatch.setattr(
            AnalogueGenerator,
            "positional_analogue_scanning",
            classmethod(lambda cls, mol_in, **kwargs: []),
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "nitrogen_walk",
            classmethod(lambda cls, mol_in, **kwargs: []),
        )

        def record_reverse(cls, mol_in, substituents, **kwargs):
            reverse_inputs.append(mol_in)
            return []

        monkeypatch.setattr(
            AnalogueGenerator,
            "reverse_positional_analogue_scanning",
            classmethod(record_reverse),
        )

        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params={
                "substituents": ["C"],
                "anchors": ["[cH]"],
            },
        )

        assert reverse_inputs == [benzene_mol]
        assert reverse_inputs[0] is benzene_mol

    def test_all_results_valid_molecules(self, single_mol):
        result = AnalogueGenerator.generate_analogues(single_mol)
        for m in result:
            smi = Chem.MolToSmiles(m)
            assert smi is not None
            assert Chem.MolFromSmiles(smi) is not None

    def test_generation_timeout_raises_actionable_error(self, benzene_mol):
        with pytest.raises(TimeoutError) as exc_info:
            AnalogueGenerator.generate_analogues(
                benzene_mol,
                positional_analogue_scanning_params={
                    "substituents": ["F"],
                    "anchors": ["[cH]"],
                },
                nitrogen_walk_params=None,
                generation_timeout=0,
            )

        message = str(exc_info.value)
        assert "0" in message
        assert "query forward PAS" in message
        assert "no partial results" in message
        assert "increase generation_timeout" in message
        assert "disable a strategy" in message
        assert "narrow the vocabulary" in message

    def test_aggregate_propagates_one_absolute_deadline(self, benzene_mol, monkeypatch):
        calls = []

        def record_pas(cls, mol_in, **kwargs):
            calls.append((kwargs["_stage"], kwargs["_deadline"]))
            return []

        def record_reverse(cls, mol_in, substituents, **kwargs):
            calls.append((kwargs["_stage"], kwargs["_deadline"]))
            return []

        def record_nitrogen(cls, mol_in, **kwargs):
            calls.append((kwargs["_stage"], kwargs["_deadline"]))
            return []

        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: 10.0
        )
        monkeypatch.setattr(
            AnalogueGenerator, "positional_analogue_scanning", classmethod(record_pas)
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "reverse_positional_analogue_scanning",
            classmethod(record_reverse),
        )
        monkeypatch.setattr(
            AnalogueGenerator, "nitrogen_walk", classmethod(record_nitrogen)
        )

        AnalogueGenerator.generate_analogues(benzene_mol, generation_timeout=2.5)

        assert calls == [
            ("query forward PAS", 12.5),
            ("query reverse PAS", 12.5),
            ("query nitrogen walk", 12.5),
            ("scaffold forward PAS", 12.5),
            ("scaffold nitrogen walk", 12.5),
        ]

    @pytest.mark.parametrize(
        ("reverse_enabled", "nitrogen_params", "expected_stage"),
        [
            (True, None, "query reverse PAS"),
            (False, {"timeout": 1}, "query nitrogen walk"),
        ],
    )
    def test_strategy_timeout_is_measured_from_aggregate_start(
        self,
        benzene_mol,
        monkeypatch,
        reverse_enabled,
        nitrogen_params,
        expected_stage,
    ):
        now = 0.0

        def finish_forward(cls, mol_in, **kwargs):
            nonlocal now
            now = 2.0
            return []

        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: now
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "positional_analogue_scanning",
            classmethod(finish_forward),
        )

        with pytest.raises(TimeoutError, match=expected_stage):
            AnalogueGenerator.generate_analogues(
                benzene_mol,
                positional_analogue_scanning_params={
                    "substituents": ["C"],
                    "anchors": ["[cH]"],
                    "timeout": 1,
                },
                nitrogen_walk_params=nitrogen_params,
                reverse_positional_analogue_scanning=reverse_enabled,
                generation_timeout=10,
            )

    def test_aggregate_expires_before_sampling(self, benzene_mol, monkeypatch):
        now = 0.0
        analogue = Chem.MolFromSmiles("Fc1ccccc1")

        def generate_forward(cls, mol_in, **kwargs):
            nonlocal now
            if kwargs["_stage"] == "query forward PAS":
                return [analogue]
            if kwargs["_stage"] == "scaffold forward PAS":
                now = 3.0
                return []
            pytest.fail("unexpected PAS stage after deadline expiry")

        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: now
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "positional_analogue_scanning",
            classmethod(generate_forward),
        )

        with pytest.raises(TimeoutError, match="query sampled multi-step"):
            AnalogueGenerator.generate_analogues(
                benzene_mol,
                positional_analogue_scanning_params={
                    "substituents": ["F"],
                    "anchors": ["[cH]"],
                },
                nitrogen_walk_params=None,
                reverse_positional_analogue_scanning=False,
                generation_timeout=2,
            )


# ===================================================================
# AnalogueGenerator – generate_analogues (scaffold + pairwise)
# ===================================================================


class TestGenerateAnaloguesNewBehavior:
    """Tests for scaffold-based and pairwise logic in generate_analogues."""

    def test_scaffold_analogues_included(self, single_mol):
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

        pos_params = {"substituents": ["C"], "anchors": ["[cH]"]}
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

    def test_sampled_pas_generates_analogues(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params={
                "substituents": ["C"],
                "anchors": ["[cH]"],
            },
            nitrogen_walk_params=None,
        )
        assert len(result) > 0

    def test_sampled_pas_nw_generates_analogues(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params={
                "substituents": ["F"],
                "anchors": ["[cH]"],
            },
            nitrogen_walk_params={},
        )
        assert len(result) > 0

    def test_all_unique_across_all_sources(self, single_mol):
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params={
                "substituents": ["F"],
                "anchors": ["[cH]"],
            },
            nitrogen_walk_params={},
        )
        smiles = [Chem.MolToSmiles(m) for m in result]
        assert len(smiles) == len(set(smiles))

    def test_preserves_source_and_first_discovery_order(self, benzene_mol):
        result = AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params={
                "substituents": ["C", "O"],
                "anchors": ["[cH]"],
            },
            nitrogen_walk_params={},
        )

        assert [Chem.MolToSmiles(mol) for mol in result[:3]] == [
            "Cc1ccccc1",
            "Oc1ccccc1",
            "c1ccncc1",
        ]

    def test_order_is_stable_with_random_hash_seed(self):
        expected = ["Fc1ccccc1", "Clc1ccccc1"]
        script = """
from rdkit import Chem
from matcha.explainability.analogue_generator import AnalogueGenerator
mol = Chem.MolFromSmiles("c1ccccc1")
result = AnalogueGenerator.positional_analogue_scanning(
    mol,
    substituents=["F", "Cl"],
    anchors=["[cH]"],
)
print("\\n".join(Chem.MolToSmiles(analogue) for analogue in result))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "random"},
        )

        assert completed.stdout.strip().splitlines() == expected

    def test_input_excluded(self, single_mol):
        input_smi = Chem.MolToSmiles(single_mol)
        result = AnalogueGenerator.generate_analogues(
            single_mol,
            positional_analogue_scanning_params={
                "substituents": ["F"],
                "anchors": ["[cH]"],
            },
            nitrogen_walk_params={},
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

    def test_remove_duplicate_keeps_original_molecule(self, single_mol):
        other = Chem.MolFromSmiles("CCO")
        result = AnalogueGenerator._remove_duplicate(single_mol, [other])
        assert result == [other]
        assert result[0] is other

    def test_remove_duplicate_preserves_first_occurrence_order(self, single_mol):
        first = Chem.MolFromSmiles("CCO")
        duplicate = Chem.MolFromSmiles("CCO")
        second = Chem.MolFromSmiles("CCN")
        result = AnalogueGenerator._remove_duplicate(
            single_mol, [first, duplicate, second]
        )

        assert result == [first, second]
        assert result[0] is first
        assert result[1] is second

    def test_remove_duplicate_rejects_wildcard_output(self, single_mol):
        wildcard = Chem.MolFromSmiles("[*]C")
        with pytest.raises(RuntimeError, match="dummy atom"):
            AnalogueGenerator._remove_duplicate(single_mol, [wildcard])


# ===================================================================
# AnalogueGenerator – bounded deterministic sampling
# ===================================================================


class TestBoundedSampling:
    """Tests for the bounded multi-step sampler in generate_analogues."""

    _PAS_PARAMS = {"substituents": ["F"], "anchors": ["[cH]"]}

    def test_signature_defaults(self):
        import inspect

        sig = inspect.signature(AnalogueGenerator.generate_analogues)
        params = list(sig.parameters.values())

        assert params[-2].name == "num_sample"
        assert params[-2].default == 100
        assert params[-1].name == "random_seed"
        assert params[-1].default == 0
        assert "num_sub" not in sig.parameters

    @pytest.mark.parametrize("value", [True, False, 1.0, 1.5, "100", None])
    def test_num_sample_rejects_non_int(self, benzene_mol, value):
        with pytest.raises(ValueError, match="num_sample"):
            AnalogueGenerator.generate_analogues(benzene_mol, num_sample=value)

    def test_num_sample_rejects_negative(self, benzene_mol):
        with pytest.raises(ValueError, match="non-negative"):
            AnalogueGenerator.generate_analogues(benzene_mol, num_sample=-1)

    @pytest.mark.parametrize("value", [True, False, 1.0, "0", None])
    def test_random_seed_rejects_non_int(self, benzene_mol, value):
        with pytest.raises(ValueError, match="random_seed"):
            AnalogueGenerator.generate_analogues(benzene_mol, random_seed=value)

    def test_num_sample_zero_disables_sampling(self, benzene_mol, monkeypatch):
        calls = []

        original = AnalogueGenerator._sample_branch

        def spy(cls, *args, **kwargs):
            calls.append(kwargs.get("stage"))
            return original.__func__(cls, *args, **kwargs)

        monkeypatch.setattr(AnalogueGenerator, "_sample_branch", classmethod(spy))

        result = AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params=None,
            reverse_positional_analogue_scanning=False,
            num_sample=0,
        )

        # First-pass output still present.
        assert [Chem.MolToSmiles(m) for m in result] == ["Fc1ccccc1"]
        # Sampler was invoked but returned empty for both branches.
        assert calls == ["query sampled multi-step", "scaffold sampled multi-step"]

    def test_num_sample_zero_still_validates_pas_params(self, benzene_mol):
        with pytest.raises(ValueError, match="Invalid substituent"):
            AnalogueGenerator.generate_analogues(
                benzene_mol,
                positional_analogue_scanning_params={
                    "substituents": ["NOT_VALID"],
                    "anchors": ["[cH]"],
                },
                num_sample=0,
            )

    def test_attempt_cap_is_two_times_num_sample(self, benzene_mol, monkeypatch):
        # A parent pool with an always-failing sampler proves the exact cap.
        attempts = 0

        def always_fail(cls, parent, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            return None

        monkeypatch.setattr(AnalogueGenerator, "_attempt_pas", classmethod(always_fail))
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_reverse_pas",
            classmethod(always_fail),
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_nitrogen_walk",
            classmethod(always_fail),
        )

        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params={},
            num_sample=7,
        )

        # Two branches × 2 * num_sample attempts each.
        assert attempts == 2 * (2 * 7)

    def test_partial_return_on_cap_exhaustion(self, benzene_mol, monkeypatch):
        # Sampler always fails; branches contribute nothing but first-pass
        # results are preserved.
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_pas",
            classmethod(lambda cls, parent, *args, **kwargs: None),
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_reverse_pas",
            classmethod(lambda cls, parent, *args, **kwargs: None),
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_nitrogen_walk",
            classmethod(lambda cls, parent, *args, **kwargs: None),
        )

        result = AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params=None,
            reverse_positional_analogue_scanning=False,
            num_sample=100,
        )

        assert [Chem.MolToSmiles(m) for m in result] == ["Fc1ccccc1"]

    def test_early_quota_completion(self, benzene_mol, monkeypatch):
        # Every attempt succeeds — sampler stops after num_sample accepted.
        counter = 0

        def unique_pas(cls, parent, *args, **kwargs):
            nonlocal counter
            counter += 1
            # A fresh unique canonical SMILES per call.
            return Chem.MolFromSmiles(f"C{'C' * counter}O")

        monkeypatch.setattr(AnalogueGenerator, "_attempt_pas", classmethod(unique_pas))
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_nitrogen_walk",
            classmethod(lambda cls, parent, *args, **kwargs: None),
        )

        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params=None,
            reverse_positional_analogue_scanning=False,
            num_sample=3,
        )

        # First pass = 1 (Fc1ccccc1) + 3 sampled (query) + 3 sampled (scaffold).
        # Scaffold of benzene = benzene, so query PAS "Fc1ccccc1" also appears
        # in the scaffold PAS pass; final dedup removes it. Sampled candidates
        # come from a fresh series so no collisions.
        # Exact attempt count: 3 accepted per branch × 2 branches = 6 calls.
        assert counter == 6

    def test_deterministic_repeated_calls(self, single_mol):
        first = AnalogueGenerator.generate_analogues(
            single_mol, num_sample=25, random_seed=42
        )
        second = AnalogueGenerator.generate_analogues(
            single_mol, num_sample=25, random_seed=42
        )
        assert [Chem.MolToSmiles(m) for m in first] == [
            Chem.MolToSmiles(m) for m in second
        ]

    def test_output_is_ambient_random_state_independent(self, single_mol):
        random.seed(0)
        first = AnalogueGenerator.generate_analogues(
            single_mol, num_sample=25, random_seed=42
        )
        random.seed(999)
        for _ in range(50):
            random.random()
        second = AnalogueGenerator.generate_analogues(
            single_mol, num_sample=25, random_seed=42
        )
        assert [Chem.MolToSmiles(m) for m in first] == [
            Chem.MolToSmiles(m) for m in second
        ]

    def test_branch_rng_isolation(self, benzene_mol, monkeypatch):
        # Perturbing query-branch attempt count must not change scaffold-branch
        # samples: the two branch RNGs are seeded independently before either
        # runs.
        scaffold_captured = []

        def record_scaffold(cls, parent, pas_params, rng, **kwargs):
            # Only capture scaffold-branch invocations via stage marker.
            if kwargs.get("stage") == "scaffold sampled multi-step":
                scaffold_captured.append(rng.random())
            return None

        monkeypatch.setattr(
            AnalogueGenerator, "_attempt_pas", classmethod(record_scaffold)
        )

        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params=None,
            reverse_positional_analogue_scanning=False,
            num_sample=5,
            random_seed=7,
        )
        first_scaffold = list(scaffold_captured)

        scaffold_captured.clear()
        # Change query-branch behavior by making it succeed with a fresh
        # candidate every attempt: the query RNG consumes different amounts
        # of state, but the scaffold branch is seeded independently.
        counter = 0

        def perturb_query(cls, parent, pas_params, rng, **kwargs):
            nonlocal counter
            if kwargs.get("stage") == "query sampled multi-step":
                counter += 1
                for _ in range(counter):
                    rng.random()
                return None
            scaffold_captured.append(rng.random())
            return None

        monkeypatch.setattr(
            AnalogueGenerator, "_attempt_pas", classmethod(perturb_query)
        )
        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params=None,
            reverse_positional_analogue_scanning=False,
            num_sample=5,
            random_seed=7,
        )
        assert scaffold_captured == first_scaffold

    def test_final_source_ordering_places_sampled_after_first_pass(self, benzene_mol):
        # First-pass sources come first; sampled candidates trail them.
        result = AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params={
                "substituents": ["F"],
                "anchors": ["[cH]"],
            },
            nitrogen_walk_params={},
            reverse_positional_analogue_scanning=False,
            num_sample=10,
            random_seed=0,
        )
        smiles = [Chem.MolToSmiles(m) for m in result]
        # First-pass anchors deterministic: query PAS -> Fc1ccccc1, query NW
        # -> c1ccncc1, then scaffold PAS/NW dedupe against query PAS/NW.
        assert smiles[0] == "Fc1ccccc1"
        assert smiles[1] == "c1ccncc1"

    def test_query_pool_includes_all_query_first_pass_sources(
        self, benzene_mol, monkeypatch
    ):
        # Verify the query pool passed to the collector is the concatenation of
        # query PAS + query reverse PAS + query nitrogen walk outputs.
        marker_a = Chem.MolFromSmiles("Fc1ccccc1")
        marker_b = Chem.MolFromSmiles("Clc1ccccc1")
        marker_c = Chem.MolFromSmiles("Brc1ccccc1")
        marker_d = Chem.MolFromSmiles("c1ccncc1")

        def fake_pas(cls, mol_in, **kwargs):
            if kwargs["_stage"] == "query forward PAS":
                return [marker_a]
            return [marker_b]  # scaffold PAS

        def fake_reverse(cls, mol_in, substituents, **kwargs):
            return [marker_c]

        def fake_nw(cls, mol_in, **kwargs):
            if kwargs["_stage"] == "query nitrogen walk":
                return [marker_d]
            return []  # scaffold NW

        captured = {}

        def spy(cls, *, mol_in, parent_pool, stage, **kwargs):
            captured[stage] = list(parent_pool)
            return []

        monkeypatch.setattr(
            AnalogueGenerator, "positional_analogue_scanning", classmethod(fake_pas)
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "reverse_positional_analogue_scanning",
            classmethod(fake_reverse),
        )
        monkeypatch.setattr(AnalogueGenerator, "nitrogen_walk", classmethod(fake_nw))
        monkeypatch.setattr(AnalogueGenerator, "_sample_branch", classmethod(spy))

        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params={},
        )

        assert captured["query sampled multi-step"] == [
            marker_a,
            marker_c,
            marker_d,
        ]
        assert captured["scaffold sampled multi-step"] == [marker_b]

    def test_all_three_strategy_routes_can_expand(self, benzene_mol, monkeypatch):
        # Force strategy selection order deterministically and verify all
        # three private helpers are dispatched.
        called = []

        def track(name):
            def helper(cls, parent, *args, **kwargs):
                called.append(name)
                return None

            return classmethod(helper)

        monkeypatch.setattr(AnalogueGenerator, "_attempt_pas", track("pas"))
        monkeypatch.setattr(
            AnalogueGenerator, "_attempt_reverse_pas", track("reverse_pas")
        )
        monkeypatch.setattr(
            AnalogueGenerator, "_attempt_nitrogen_walk", track("nitrogen_walk")
        )

        # Give enough attempts that all three strategies are almost certain to
        # be picked at least once under uniform selection.
        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params={},
            num_sample=100,
            random_seed=0,
        )

        assert "pas" in called
        assert "reverse_pas" in called
        assert "nitrogen_walk" in called

    def test_second_pass_public_apis_are_not_recursed(self, benzene_mol, monkeypatch):
        # The sampler must not call the exhaustive public PAS / nitrogen walk /
        # reverse-PAS methods on first-pass outputs.
        original_pas = AnalogueGenerator.positional_analogue_scanning
        original_nw = AnalogueGenerator.nitrogen_walk
        original_reverse = AnalogueGenerator.reverse_positional_analogue_scanning
        pas_inputs = []
        nw_inputs = []
        reverse_inputs = []

        def record_pas(cls, mol_in, **kwargs):
            pas_inputs.append(kwargs["_stage"])
            return original_pas.__func__(cls, mol_in, **kwargs)

        def record_nw(cls, mol_in, **kwargs):
            nw_inputs.append(kwargs["_stage"])
            return original_nw.__func__(cls, mol_in, **kwargs)

        def record_reverse(cls, mol_in, substituents, **kwargs):
            reverse_inputs.append(kwargs["_stage"])
            return original_reverse.__func__(cls, mol_in, substituents, **kwargs)

        monkeypatch.setattr(
            AnalogueGenerator,
            "positional_analogue_scanning",
            classmethod(record_pas),
        )
        monkeypatch.setattr(AnalogueGenerator, "nitrogen_walk", classmethod(record_nw))
        monkeypatch.setattr(
            AnalogueGenerator,
            "reverse_positional_analogue_scanning",
            classmethod(record_reverse),
        )

        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params={},
            num_sample=50,
            random_seed=0,
        )

        assert pas_inputs == ["query forward PAS", "scaffold forward PAS"]
        assert nw_inputs == ["query nitrogen walk", "scaffold nitrogen walk"]
        assert reverse_inputs == ["query reverse PAS"]

    def test_one_mutation_per_attempt(self, benzene_mol, monkeypatch):
        # Each attempt invokes exactly one of the three strategy helpers, once.
        per_call = []

        def make_counter(name):
            def helper(cls, parent, *args, **kwargs):
                per_call.append(name)
                return None

            return classmethod(helper)

        monkeypatch.setattr(AnalogueGenerator, "_attempt_pas", make_counter("pas"))
        monkeypatch.setattr(
            AnalogueGenerator, "_attempt_reverse_pas", make_counter("reverse_pas")
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_nitrogen_walk",
            make_counter("nitrogen_walk"),
        )

        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params={},
            num_sample=4,
            random_seed=0,
        )

        # 2 branches × 2*num_sample attempts = 16 total, each attempt = 1 call.
        assert len(per_call) == 16

    def test_branch_local_deduplication(self, benzene_mol, monkeypatch):
        # A helper that always returns the same molecule fills at most one slot
        # per branch.
        constant = Chem.MolFromSmiles("Nc1ccccc1")

        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_pas",
            classmethod(lambda cls, parent, *args, **kwargs: constant),
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_nitrogen_walk",
            classmethod(lambda cls, parent, *args, **kwargs: None),
        )

        result = AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params={},
            reverse_positional_analogue_scanning=False,
            num_sample=10,
            random_seed=0,
        )

        smiles = [Chem.MolToSmiles(m) for m in result]
        # Fc1ccccc1 from first-pass PAS, c1ccncc1 from first-pass NW, and
        # exactly one 'Nc1ccccc1' from sampling despite many attempts.
        assert smiles.count("Nc1ccccc1") == 1

    def test_query_equivalent_candidate_is_skipped(self, benzene_mol, monkeypatch):
        # A helper that returns the input query never counts toward the branch
        # quota. Restrict to a single enabled strategy so the attempt count is
        # exact.
        call_count = 0

        def return_query(cls, parent, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return Chem.MolFromSmiles("c1ccccc1")

        monkeypatch.setattr(
            AnalogueGenerator, "_attempt_pas", classmethod(return_query)
        )

        result = AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params=None,
            reverse_positional_analogue_scanning=False,
            num_sample=3,
            random_seed=0,
        )

        smiles = [Chem.MolToSmiles(m) for m in result]
        assert "c1ccccc1" not in smiles
        # Cap exhausted per branch: 2 * num_sample attempts × 2 branches.
        assert call_count == 2 * 3 * 2

    def test_timeout_during_sampling_propagates(self, benzene_mol, monkeypatch):
        now = 0.0

        def slow_attempt(cls, parent, *args, **kwargs):
            nonlocal now
            now = 100.0
            return None

        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: now
        )
        monkeypatch.setattr(
            AnalogueGenerator, "_attempt_pas", classmethod(slow_attempt)
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_nitrogen_walk",
            classmethod(slow_attempt),
        )

        with pytest.raises(TimeoutError, match="query sampled multi-step"):
            AnalogueGenerator.generate_analogues(
                benzene_mol,
                positional_analogue_scanning_params=self._PAS_PARAMS,
                nitrogen_walk_params={},
                reverse_positional_analogue_scanning=False,
                num_sample=5,
                generation_timeout=1,
            )

    def test_sampler_unexpected_exception_propagates(self, benzene_mol, monkeypatch):
        def blow_up(cls, parent, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(AnalogueGenerator, "_attempt_pas", classmethod(blow_up))
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_nitrogen_walk",
            classmethod(blow_up),
        )

        with pytest.raises(RuntimeError, match="boom"):
            AnalogueGenerator.generate_analogues(
                benzene_mol,
                positional_analogue_scanning_params=self._PAS_PARAMS,
                nitrogen_walk_params={},
                reverse_positional_analogue_scanning=False,
                num_sample=5,
            )

    def test_reverse_pas_participates_as_sampled_expansion(
        self, benzene_mol, monkeypatch
    ):
        # Force strategy selection to "reverse_pas" and verify the private
        # helper is dispatched (proving reverse PAS is a valid second-step
        # strategy).
        reverse_calls = []

        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_pas",
            classmethod(lambda cls, *a, **kw: None),
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_nitrogen_walk",
            classmethod(lambda cls, *a, **kw: None),
        )

        def record_reverse(cls, parent, *args, **kwargs):
            reverse_calls.append(parent)
            return None

        monkeypatch.setattr(
            AnalogueGenerator,
            "_attempt_reverse_pas",
            classmethod(record_reverse),
        )

        AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params=self._PAS_PARAMS,
            nitrogen_walk_params={},
            num_sample=50,
            random_seed=0,
        )

        assert len(reverse_calls) > 0
