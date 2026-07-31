from matcha.explainability.lime import LIME
from matcha.explainability.analogue_generator import AnalogueGenerator
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import MolToSmiles
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D as Draw
from PIL import Image
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import collections as cl
import sklearn.preprocessing as skp
from matcha.utils.schemas import ExplainerInputModel
from matcha.utils.logging import get_default_logger

logger = get_default_logger(__name__)


class MatchaExplanation:
    """Container for LIME explanation results with visualization methods.

    Holds the descriptor-level coefficients, ECFP atomic environments and
    weights, the query molecule, and generated analogues. Provides plotting
    utilities for coefficient bar charts and molecular heatmaps.
    """

    def __init__(
        self,
        df_desc: pd.DataFrame,
        envs: dict[int, list[int]],
        weights: dict[int, float],
        mol: Mol,
        analogues: list[str],
    ):
        """Initialize a MatchaExplanation.

        :param pd.DataFrame df_desc: LIME results from descriptor-based analysis,
            with columns ``Descriptor``, ``Coefficient``, ``Standard deviation``.
        :param dict[int, list[int]] envs: Mapping of fingerprint bit IDs to sets
            of atom indices in their environment.
        :param dict[int, float] weights: LIME coefficients per fingerprint bit ID.
        :param Mol mol: The query RDKit molecule.
        :param list[str] analogues: SMILES strings of the molecules used in the
            LIME analysis.
        """
        self.df_desc = df_desc
        self.envs = envs
        self.weights = weights
        self.analogues = analogues
        self._mol = mol

    def plot_coefficients(
        self, remove_noise: bool = True, keep_k: int = 10
    ) -> go.Figure:
        """Plot LIME coefficients as a horizontal bar chart.

        :param bool remove_noise: Whether to filter out unreliable coefficients
            (those where standard deviation exceeds the absolute coefficient or
            absolute coefficient is <= 0.1). Defaults to True.
        :param int keep_k: Maximum number of descriptors to display. Defaults to 10.

        :returns: Plotly figure with the LIME coefficient bar chart.

        :raises ValueError: If no reliable coefficients remain after noise removal.
        """
        df_plot = self.df_desc.copy()

        last_row = df_plot.iloc[-1]
        value = np.round(last_row["Coefficient"], 2)
        std = np.round(last_row["Standard deviation"], 2)

        df_plot = df_plot.head(-1)
        df_plot["Reliability"] = (
            df_plot["Coefficient"].abs() - df_plot["Standard deviation"]
        )
        if remove_noise:
            df_plot = df_plot[df_plot["Reliability"] > 0]
            df_plot = df_plot[df_plot["Coefficient"].abs() > 0.1]

        if len(df_plot) == 0:
            raise ValueError(
                "No reliable coefficients remain after noise removal. "
                "All descriptors have a standard deviation exceeding their "
                "absolute coefficient or an absolute coefficient ≤ 0.1. "
                "Try setting remove_noise=False or providing more data."
            )

        if len(df_plot) > keep_k:
            df_plot["Abs"] = df_plot["Coefficient"].abs()
            df_plot = df_plot.nlargest(keep_k, "Abs")
            df_plot.drop("Abs", axis=1)

        df_plot = df_plot.sort_values("Coefficient", ascending=True)

        descriptors = df_plot["Descriptor"].tolist()
        coefficients = df_plot["Coefficient"].tolist()
        std_devs = df_plot["Standard deviation"].tolist()

        # Interpolate bar color between two endpoints based on R²:
        #   R² <= 0.7  -> #2dbecd (teal)
        #   R² >= 1.0  -> #149b5f (green)
        t = float(np.clip((value - 0.7) / 0.3, 0, 1))
        low_rgb = pc.hex_to_rgb("#2dbecd")
        high_rgb = pc.hex_to_rgb("#149b5f")
        r = int(low_rgb[0] + t * (high_rgb[0] - low_rgb[0]))
        g = int(low_rgb[1] + t * (high_rgb[1] - low_rgb[1]))
        b = int(low_rgb[2] + t * (high_rgb[2] - low_rgb[2]))
        bar_color = f"rgb({r}, {g}, {b})"

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=descriptors,
                x=coefficients,
                error_x=dict(type="data", array=std_devs),
                orientation="h",
                marker_color=bar_color,
            )
        )

        fig.update_layout(
            title=dict(
                text=f"LIME analysis (R² = {value} ± {std})",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Importance score",
            xaxis=dict(
                range=[
                    min(min(c - s for c, s in zip(coefficients, std_devs)), 0),
                    max(c + s for c, s in zip(coefficients, std_devs)),
                ],
                showgrid=True,
            ),
            template="plotly_white",
            height=max(400, len(descriptors) * 40),
            width=800,
        )

        return fig

    def plot_heatmap(self, colormap: str = "RdYlGn") -> go.Figure:
        """Generate a heatmap of the molecule with atoms colored by LIME weights.

        Atom colors are determined by summing ECFP bit weights for all bits
        that include the atom, then scaling to [-1, +1].

        :param str colormap: Matplotlib colormap name. Defaults to ``"RdYlGn"``.

        :returns: Plotly figure containing the molecular similarity map image.
        """
        highlights = cl.defaultdict(float)
        # collect atomic weights by summing up weights originating from LIME analysis
        # an atom can be present in multiple bits, thus the summation
        for bitid, atoms in self.envs.items():
            for aid in atoms:
                highlights[aid] += self.weights[bitid]
        # scale the results between -1 and +1
        highlights = skp.minmax_scale(
            [weight for _, weight in sorted(highlights.items(), key=lambda x: x[0])],
            feature_range=(-1, 1),
            axis=0,
        ).tolist()

        # make the drawing using Cairo (PNG)
        d2d = Draw.MolDraw2DCairo(550, 350)
        dopts = d2d.drawOptions()
        dopts.useBWAtomPalette()
        dopts.addAtomIndices = True
        SimilarityMaps.GetSimilarityMapFromWeights(
            self._mol,
            highlights,
            draw2d=d2d,
            alpha=0.5,
        )
        d2d.FinishDrawing()

        # Convert PNG bytes to a Plotly figure
        png_data = d2d.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))

        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=img.height,
                sizex=img.width,
                sizey=img.height,
                sizing="stretch",
                layer="below",
            )
        )
        fig.update_xaxes(visible=False, range=[0, img.width])
        fig.update_yaxes(visible=False, range=[0, img.height], scaleanchor="x")
        fig.update_layout(
            width=550,
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig


_pos_params = {
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
}

_nitrogen_walk_params = {"num_sub": 1}


class MatchaExplainer:
    """High-level interface for molecular explainability.

    Combines LIME-based feature importance analysis with structural analogue
    generation. Produces :class:`MatchaExplanation` objects that contain both
    descriptor coefficients and atomic heatmap data for visualization.
    """

    def __init__(
        self,
        positional_analogue_scanning_params: dict | None = {},
        nitrogen_walk_params: dict | None = {},
        lime_descriptor_set: list[str] | None = None,
        lime_fingerprint_params: dict | None = None,
        lime_scale_coeff: bool = True,
        lime_remove_noise: bool = True,
    ):
        """Initialize the MatchaExplainer.

        :param dict | None positional_analogue_scanning_params: Parameters for
            positional analogue scanning. Empty dict uses defaults; None disables.
        :param dict | None nitrogen_walk_params: Parameters for nitrogen walking.
            Empty dict uses defaults; None disables.
        :param list[str] | None lime_descriptor_set: RDKit descriptor names for
            LIME. None uses the default set.
        :param dict | None lime_fingerprint_params: Morgan fingerprint parameters
            for ECFP-based LIME. None uses defaults.
        :param bool lime_scale_coeff: Whether to normalize LIME coefficients.
            Defaults to True.
        :param bool lime_remove_noise: Whether to filter unreliable coefficients
            in the explanation. Defaults to True.
        """
        ExplainerInputModel(
            positional_analogue_scanning_params=positional_analogue_scanning_params,
            nitrogen_walk_params=nitrogen_walk_params,
            lime_descriptor_set=lime_descriptor_set,
            lime_fingerprint_params=lime_fingerprint_params,
            lime_scale_coeff=lime_scale_coeff,
            lime_remove_noise=lime_remove_noise,
        )

        if positional_analogue_scanning_params == {}:
            self._pos_params = _pos_params
        else:
            self._pos_params = positional_analogue_scanning_params
        if nitrogen_walk_params == {}:
            self._nitrogen_walk_params = _nitrogen_walk_params
        else:
            self._nitrogen_walk_params = nitrogen_walk_params
        self._descriptor_set = lime_descriptor_set
        self._fingerprint_params = lime_fingerprint_params
        self._scale_coeff = lime_scale_coeff
        self._remove_noise = lime_remove_noise

    def _run_lime_desc(self, mols, predictions, bootstrap_num) -> tuple:
        """Run LIME analysis using RDKit descriptors.

        :param list mols: RDKit molecule objects.
        :param np.ndarray predictions: Target values for the molecules.
        :param int bootstrap_num: Number of bootstrap iterations.

        :returns: DataFrame of descriptor coefficients.
        """
        lime_desc = LIME(
            self._descriptor_set, self._fingerprint_params, self._scale_coeff, False
        )
        df_desc = lime_desc.explain(mols, predictions, bootstrap_num)
        return df_desc

    def _run_lime_ecfp(self, mols, predictions, bootstrap_num) -> tuple:
        """Run LIME analysis using ECFP fingerprints.

        :param list mols: RDKit molecule objects.
        :param np.ndarray predictions: Target values for the molecules.
        :param int bootstrap_num: Number of bootstrap iterations.

        :returns: Tuple of (envs, weights) mapping fingerprint bit IDs to
            atom environments and their coefficients.
        """
        lime_ecfp = LIME(
            self._descriptor_set, self._fingerprint_params, self._scale_coeff, True
        )
        df_ecfp = lime_ecfp.explain(mols, predictions, bootstrap_num)
        envs, weights = lime_ecfp.get_envs_and_weights(mols[0], df_ecfp)
        return envs, weights

    def generate_analogues(self, mol: Mol) -> list[Mol]:
        """Generate structural analogues for a molecule.

        :param Mol mol: Query RDKit molecule.

        :returns: List of unique analogue molecules.
        """
        return AnalogueGenerator.generate_analogues(
            mol,
            self._pos_params,
            self._nitrogen_walk_params,
        )

    def decompose(self, mol: Mol) -> list[Mol]:
        """Decompose a molecule into BRICS fragments.

        :param Mol mol: Input RDKit molecule.

        :returns: List of fragment molecules.
        """
        return AnalogueGenerator.decompose(mol)

    def explain(
        self,
        mols: list[Mol],
        predictions: np.ndarray,
        bootstrap_num: int = 25,
    ) -> MatchaExplanation:
        """Run LIME analysis with both RDKit descriptors and ECFP fingerprints.

        Performs two parallel LIME analyses — one using RDKit physicochemical
        descriptors for interpretable coefficients, and one using ECFP
        fingerprints for atomic-level heatmap visualization.

        :param list[Mol] mols: RDKit molecule objects to explain. At least 10
            molecules are recommended for reliable results.
        :param np.ndarray predictions: Target values (e.g., model predictions),
            shape ``(n_molecules,)``.
        :param int bootstrap_num: Number of bootstrap iterations. Defaults to 25.

        :returns: :class:`MatchaExplanation` containing coefficients, atomic
            environments, weights, and analogue SMILES.
        """
        # Run LIME with RDKit descriptors
        if len(mols) < 10:
            logger.warning(
                f"Only {len(mols)} molecules provided. At least 10 analogues are "
                "recommended for reliable LIME explanations. Results may be unstable."
            )

        df_desc = self._run_lime_desc(mols, predictions, bootstrap_num)

        # Run LIME with ECFP fingerprints
        envs, weights = self._run_lime_ecfp(mols, predictions, bootstrap_num)

        # Convert molecules to SMILES
        smiles = [MolToSmiles(mol) for mol in mols]

        return MatchaExplanation(df_desc, envs, weights, mols[0], smiles)
