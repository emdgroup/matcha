"""Tests for matcha.explainability.analogue_generator.AnalogueGenerator."""

import os
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
            num_sub=1,
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
    def test_rejects_invalid_num_sub(self, benzene_mol, num_sub):
        with pytest.raises(ValueError, match="num_sub must be a positive integer"):
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

    def test_reverse_can_be_disabled(self):
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        params = {"substituents": ["C"], "anchors": [], "num_sub": 1}

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
                "num_sub": 1,
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
                    "num_sub": 1,
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

    def test_aggregate_propagates_one_absolute_deadline(
        self, benzene_mol, monkeypatch
    ):
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
            (False, {"num_sub": 1, "timeout": 1}, "query nitrogen walk"),
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
                    "num_sub": 1,
                    "timeout": 1,
                },
                nitrogen_walk_params=nitrogen_params,
                reverse_positional_analogue_scanning=reverse_enabled,
                generation_timeout=10,
            )

    def test_aggregate_expires_before_second_pass(self, benzene_mol, monkeypatch):
        now = 0.0
        analogue = Chem.MolFromSmiles("Fc1ccccc1")

        def generate_forward(cls, mol_in, **kwargs):
            nonlocal now
            if kwargs["_stage"] == "query forward PAS":
                return [analogue]
            if kwargs["_stage"] == "scaffold forward PAS":
                now = 3.0
                return []
            pytest.fail("second-pass generation started after deadline expiry")

        monkeypatch.setattr(
            "matcha.explainability.analogue_generator.time.monotonic", lambda: now
        )
        monkeypatch.setattr(
            AnalogueGenerator,
            "positional_analogue_scanning",
            classmethod(generate_forward),
        )

        with pytest.raises(TimeoutError, match="second-pass generation"):
            AnalogueGenerator.generate_analogues(
                benzene_mol,
                positional_analogue_scanning_params={
                    "substituents": ["F"],
                    "anchors": ["[cH]"],
                    "num_sub": 1,
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

    def test_preserves_source_and_first_discovery_order(self, benzene_mol):
        result = AnalogueGenerator.generate_analogues(
            benzene_mol,
            positional_analogue_scanning_params={
                "substituents": ["C", "O"],
                "anchors": ["[cH]"],
                "num_sub": 1,
            },
            nitrogen_walk_params={"num_sub": 1},
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
