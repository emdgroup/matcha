from rdkit.Chem.rdchem import Mol

from matcha.explainability import MatchaExplainer, MatchaExplanation
from matcha.utils.logging import get_default_logger


class ExplainabilityManager:
    """Manages LIME-based explainability."""

    def __init__(self):
        self._explainer: MatchaExplainer | None = None
        self.logger = get_default_logger("XAI")

    @property
    def explainer(self):
        """The explainer object, if one has been created."""
        return self._explainer

    def create_explainer(self, params: dict) -> None:
        """Factory method to create an explainer instance.

        :param dict params: parameters for the MatchaExplainer
        """
        self._explainer = MatchaExplainer(**params)

    def explain(
        self,
        model_instance,
        input_mol: Mol,
        task_idx: int = 0,
        lime_bootstrap_num: int = 25,
        lime_descriptor_set: list[str] | None = None,
        use_std: bool = False,
    ) -> MatchaExplanation:
        """Generate LIME explanations for a molecule prediction.

        :param model_instance: the sklearn model instance
        :param Mol input_mol: RDKit molecule to explain
        :param int task_idx: index of the task to explain
        :param int lime_bootstrap_num: number of bootstrap iterations for LIME
        :param list[str] | None lime_descriptor_set: RDKit descriptors for LIME
        :param bool use_std: use uncertainty estimates for predictions
        :return MatchaExplanation: LIME results with plotting methods
        """
        self.logger.info("XAI: beginning explanation")

        if self._explainer is None:
            explainer = MatchaExplainer(
                positional_analogue_scanning_params={},
                nitrogen_walk_params={},
                lime_descriptor_set=lime_descriptor_set,
            )
        else:
            explainer = self._explainer

        analogues = explainer.generate_analogues(input_mol)
        targets = [input_mol] + analogues

        return self._get_explanations(
            model_instance=model_instance,
            explainer=explainer,
            mols=targets,
            task_idx=task_idx,
            use_std=use_std,
            bootstrap_num=lime_bootstrap_num,
        )

    def _get_explanations(
        self,
        model_instance,
        explainer: MatchaExplainer,
        mols: list[Mol],
        task_idx: int = 0,
        use_std: bool = False,
        bootstrap_num: int = 25,
    ) -> MatchaExplanation:
        """Internal method to run LIME explanation on molecules.

        :param model_instance: the sklearn model instance
        :param MatchaExplainer explainer: explainer instance to use
        :param list[Mol] mols: list of molecules to explain
        :param int task_idx: index of the task to explain
        :param bool use_std: use uncertainty estimates
        :param int bootstrap_num: number of bootstrap iterations for LIME
        :return MatchaExplanation: LIME results with plotting methods
        """
        self.logger.info(f"XAI: found {len(mols) - 1} analogues")

        if use_std:
            preds = model_instance.compute_uncertainty(mols, num_iterations=10)[
                :, task_idx
            ]
        else:
            preds = model_instance._default_predict(mols, accelerator="cpu")[
                :, task_idx
            ]

        explanation = explainer.explain(
            mols=mols,
            predictions=preds,
            bootstrap_num=bootstrap_num,
        )

        self.logger.info("XAI: explanation generated")
        return explanation
