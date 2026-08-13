from matcha.utils.schemas.base import BaseDataModel
from typing import Literal, Optional
from pydantic import model_validator

### Base classes ###


class CalibrationDataSettings(BaseDataModel):
    """Settings for splitting calibration data by column and ratio."""

    split_column: str = "compound_id"
    split_ratio: float = 0.99


class Dataset(BaseDataModel):
    """Schema for dataset configuration including file path, column keys, and calibration."""

    path: str
    label_key: str
    smiles_key: Optional[str] = None
    operator_key: Optional[str] = None
    calibration: Optional[CalibrationDataSettings] = None
    statistics: Optional[dict] = None


class CalibrationModelSettings(BaseDataModel):
    """Settings for the calibration model algorithm and its parameters."""

    algorithm: str
    params: Optional[dict] = None


class Metadata(BaseDataModel):
    """Schema for model metadata such as name, version, scope, and owner."""

    model_name: str
    model_version: int
    model_scope: str
    model_owner: str
    description: Optional[str] = "no description provided"


class Model(BaseDataModel):
    """Schema for model configuration including architecture, parameters, and metadata."""

    architecture: str
    params: dict
    metadata: Metadata
    ensemble: Optional[int] = None
    config_path: Optional[str] = None
    calibration: Optional[CalibrationModelSettings] = None


class MLFlow(BaseDataModel):
    """Schema for MLflow experiment tracking configuration."""

    experiment_name: str
    tags: Optional[dict] = {}
    log_dir: str
    run_name: Optional[str] = None
    server_uri: Optional[str] = None


class Serialization(BaseDataModel):
    """Schema for model serialization settings including path and quantization."""

    path: str
    quantize: Optional[bool] = False


class Output(BaseDataModel):
    """Schema for training output configuration combining MLflow and serialization."""

    mlflow: Optional[MLFlow] = None
    serialization: Optional[Serialization] = None


class Split(BaseDataModel):
    """Schema for data splitting strategy including method, subsets, and bootstrapping.

    ``n_bootstrap`` and ``frac_bootstrap`` apply uniformly across all split
    methods (cv, time, cluster, file): each split's test set is resampled
    ``n_bootstrap`` times at ``frac_bootstrap`` fraction for uncertainty
    estimation.
    """

    method: str
    n_subset: int
    n_bootstrap: Optional[int] = None
    frac_bootstrap: Optional[float] = None
    method_params: Optional[dict] = None


### train CLI validation ###


class CLITrainInputModel(BaseDataModel):
    """Top-level input schema for the ``train`` CLI command."""

    dataset: Dataset
    model: Model
    output: Output
    magic: Optional[str] = None


### evaluate CLI validation ###


class ModelEvaluation(Model):
    """Model schema variant for evaluation where metadata is optional."""

    metadata: Optional[Metadata] = None


class CLIEvaluationInputModel(BaseDataModel):
    """Top-level input schema for the ``evaluate`` CLI command."""

    dataset: Dataset
    model: ModelEvaluation
    output: Output
    split: Split
    quantize: Optional[bool] = False
    magic: Optional[str] = None


### baseline CLI validation ###
class BaselineModel(BaseDataModel):
    """Schema for scikit-learn baseline model configuration."""

    n_jobs: int = 16
    feature_list: list[str] = ["rdkit_all_descriptors", "ecfp"]
    params: dict = {"n_estimators": 100}
    label_transform: Optional[str] = None
    label_encoder_params: Optional[dict] = None
    algorithm: str = "RandomForestRegressor"


class CLIBaselineInputModel(BaseDataModel):
    """Top-level input schema for the ``baseline`` CLI command."""

    dataset: Dataset
    model: BaselineModel
    output: Output
    split: Split
    magic: Optional[str] = None


### summarize CLI validation ###
class CLISummarizeInputModel(BaseDataModel):
    """Top-level input schema for the ``summarize`` CLI command.

    Supports two mutually exclusive input modes:

    **MLflow mode** — supply ``experiment_name`` and ``mlruns_path``::

        experiment_name: my_experiment
        mlruns_path: ./mlruns
        output_path: ./outputs

    **Directory mode** — supply ``root_dir``; each immediate subdirectory is
    treated as one model run, identified by its folder name::

        root_dir: ./experiment
        output_path: ./outputs

    ``output_path`` is shared by both modes (default: ``./outputs``).
    In MLflow mode it receives the summary files written to disk; in directory
    mode it additionally receives the HTML plot files.
    """

    experiment_name: Optional[str] = None
    mlruns_path: Optional[str] = None
    root_dir: Optional[str] = None
    output_path: Optional[str] = "./outputs"
    statistical_test: Optional[str] = "non-parametric"
    runs: Optional[list[str]] = None
    magic: Optional[str] = None

    @model_validator(mode="after")
    def _check_input_mode(self) -> "CLISummarizeInputModel":
        has_mlflow = self.experiment_name is not None or self.mlruns_path is not None
        has_dir = self.root_dir is not None
        if has_mlflow and has_dir:
            raise ValueError(
                "Provide either 'root_dir' (directory mode) or "
                "'experiment_name'+'mlruns_path' (MLflow mode), not both."
            )
        if not has_mlflow and not has_dir:
            raise ValueError(
                "Must provide either 'root_dir' (directory mode) or "
                "both 'experiment_name' and 'mlruns_path' (MLflow mode)."
            )
        if has_mlflow and (self.experiment_name is None or self.mlruns_path is None):
            raise ValueError(
                "'experiment_name' and 'mlruns_path' must both be provided for MLflow mode."
            )
        return self


### prepare_multitask CLI validation ###


class StitcherInputModel(BaseDataModel):
    """Input schema for the dataset stitcher specifying source files and column keys."""

    folder_path: str
    dataset_names: list[str]
    label_keys: list[str]
    tag: str = "endpoint"
    operator_key: str = "OPERATOR"
    smiles_key: str = "SMILES"
    index_key: Optional[str] = None


class StitcherOutputModel(BaseDataModel):
    """Output schema for the dataset stitcher specifying destination path and filename."""

    folder_path: str = "./outputs"
    filename: str = "df.csv"


class CLIStitcherInputModel(BaseDataModel):
    """Top-level input schema for the ``prepare_multitask`` CLI command."""

    input: StitcherInputModel
    output: StitcherOutputModel
    magic: Optional[str] = None


### autotune CLI validation ###
class AutotuneOptimum(BaseDataModel):
    """Schema for autotune optimum output path and filename."""

    path: str = "./outputs"
    filename: str = "hpo_output"


class AutotuneOutput(Output):
    """Extended output schema for autotune that adds the optimum result location."""

    optimum: AutotuneOptimum


class Search(BaseDataModel):
    """Schema for a hyperparameter search space configuration and budget."""

    config: str | dict = "default"
    budget: int = 70


class Tuning(BaseDataModel):
    """Schema for autotune tuning settings with architecture and optimizer search."""

    architecture_search: Search = Search()
    optimizer_search: Search = Search()


class CLIAutotuneOutput(BaseDataModel):
    """Top-level input schema for the ``autotune`` CLI command."""

    dataset: Dataset
    model: ModelEvaluation
    split: Split
    tuning: Tuning
    output: AutotuneOutput
    magic: Optional[str] = None


### prepare CLI validation ###
class PrepareDatasets(BaseDataModel):
    """Schema for listing dataset files and their task types for pretraining preparation."""

    files: list[str]
    task_type: list[str]
    sparse: bool = True


class PrepareMetadata(BaseDataModel):
    """Schema for metadata settings used during pretraining data preparation."""

    merge_col: str = "SMILES"
    tag_to_add: str = "PRETRAIN"


class PrepareValidation(BaseDataModel):
    """Schema for validation set creation parameters during data preparation."""

    min_compounds: int = 10
    sampling_rate: float = 0.001
    seed: int = 42


class CLIPrepareInputModel(BaseDataModel):
    """Top-level input schema for the ``prepare`` CLI command."""

    datasets: PrepareDatasets
    metadata: PrepareMetadata
    validation: PrepareValidation
    output: str
    magic: Optional[str] = None


### predict CLI validation ###
class PredictDataset(Dataset):
    """Dataset schema variant for prediction where label_key is optional."""

    label_key: Optional[str] = None
    keep_cols: bool = True


class Inference(BaseDataModel):
    """Schema for inference hardware and batching settings."""

    accelerator: str = "cpu"
    devices: int = 1
    batch_size: int = 256


class PredictModel(BaseDataModel):
    """Schema for the model used in prediction, referencing a serialized model path."""

    path: str
    inference: Optional[Inference] = None


class CLIPredictInputModel(BaseDataModel):
    """Top-level input schema for the ``predict`` CLI command."""

    dataset: PredictDataset
    model: PredictModel
    output: str
    magic: Optional[str] = None


### Shared pretraining schemas (used by both pretrain_multitask and pretrain_encoder) ###


class PretrainPipe(BaseDataModel):
    """Hardware / distributed-training knobs shared by all pretraining CLIs."""

    visible_devices: str | None = None
    strategy: dict = {"timeout_seconds": 3600, "find_unused_parameters": False}
    dataloader_num_workers: int = 8
    fit_datamodule_size: int | None = None
    gradient_accumulation_steps: int = 1
    gradient_clip_val: float = 0.0


class PretrainTraining(BaseDataModel):
    """Training-loop parameters shared by all pretraining CLIs."""

    num_epochs: int = 50
    early_stopping: bool = True
    patience: int = 10
    accelerator: str = "gpu"
    devices: int = 1
    seed: int = 0


class PretrainOutput(BaseDataModel):
    """Schema for pretraining output with serialization path and optional MLflow logging."""

    serialization: str
    mlflow: Optional[MLFlow] = None


### pretrain_multitask CLI validation ###


class MultitaskPretrainDataset(BaseDataModel):
    """Dataset config for multitask pretraining.

    ``dataset_dir`` is the base directory produced by the ``prepare`` command.
    By default, files are resolved relative to that directory using the
    standard names (``train_molecules.parquet``, ``val_molecules.parquet``,
    ``train_tasks.npz``, ``val_tasks.npz``, ``task_metadata.json``).
    Each path can be overridden individually to point at a different location.
    """

    dataset_dir: str
    train_smiles: Optional[str] = None
    val_smiles: Optional[str] = None
    train_tasks: Optional[str] = None
    val_tasks: Optional[str] = None
    task_metadata: Optional[str] = None


class Loss(BaseDataModel):
    """Schema for a single loss term in multitask pretraining with weighting schedule."""

    dataset: str
    loss_fn: str
    loss_args: dict
    init_w: float
    final_w: float
    T: float
    warmup: float


class PretrainMultitaskModel(BaseDataModel):
    """Model config for multitask pretraining.

    Mirrors :class:`EncoderPretrainModel` so that neural-network params,
    datamodule params and training params live in separate config keys
    instead of a single flat ``params`` dict.
    """

    architecture: str
    params: dict  # neural-network constructor args only
    datamodule: Optional[dict] = None  # datamodule-specific overrides
    training: Optional[PretrainTraining] = PretrainTraining()


class CLIPretrainMultitaskInputModel(BaseDataModel):
    """Top-level input schema for the ``pretrain_multitask`` CLI command."""

    dataset: MultitaskPretrainDataset
    model: PretrainMultitaskModel
    loss: list[Loss]
    pipe: PretrainPipe
    output: PretrainOutput
    magic: Optional[str] = None


### pretrain_encoder CLI validation ###


class EncoderPretrainDataset(BaseDataModel):
    """Dataset config for encoder pretraining.

    Supports three modes:
    - **MLM**: only ``train_smiles`` and ``val_smiles`` parquet paths are needed.
    - **Graph pretraining**: additionally requires ``train_y_graph``, ``val_y_graph``
      (npz with key ``descriptors``), ``train_y_node``, ``val_y_node``
      (npz files for node-level targets), and ``task_type`` = ``"graph"``.
    - **Graph3D pretraining**: same requirements as ``graph``, plus
      ``train_coords`` / ``val_coords`` npz files carrying precomputed
      per-molecule 3D atomic coordinates; ``task_type`` = ``"graph3d"``.

    Node-level target files (``train_y_node``, ``val_y_node``) use a packed
    flat + offsets layout::

        np.savez_compressed("node_y.npz",
                            flat=np.concatenate(list_of_arrays),
                            offsets=np.cumsum([0] + [len(a) for a in list_of_arrays]))

    Coordinate files (``train_coords``, ``val_coords``) use the same packing,
    with ``flat`` shaped ``(total_atoms, 3)`` and ``offsets`` of length
    ``N + 1``. Each molecule slice becomes an ``(A_i, 3)`` ``float32`` array.
    """

    task_type: Literal["mlm", "graph", "graph3d"]
    train_smiles: str
    val_smiles: str
    train_y_graph: Optional[str] = None
    val_y_graph: Optional[str] = None
    train_y_node: Optional[str] = None
    val_y_node: Optional[str] = None
    train_coords: Optional[str] = None
    val_coords: Optional[str] = None

    @model_validator(mode="after")
    def _check_task_type_requirements(self) -> "EncoderPretrainDataset":
        if self.task_type == "graph3d":
            missing = [
                name
                for name, value in {
                    "train_y_graph": self.train_y_graph,
                    "val_y_graph": self.val_y_graph,
                    "train_y_node": self.train_y_node,
                    "val_y_node": self.val_y_node,
                    "train_coords": self.train_coords,
                    "val_coords": self.val_coords,
                }.items()
                if value is None
            ]
            if missing:
                raise ValueError(
                    "task_type='graph3d' requires all four label paths "
                    "(train_y_graph, val_y_graph, train_y_node, val_y_node) "
                    "plus train_coords and val_coords. Missing: "
                    f"{sorted(missing)}"
                )
        else:  # "mlm" or "graph"
            if self.train_coords is not None or self.val_coords is not None:
                raise ValueError(
                    "train_coords / val_coords are only valid when task_type='graph3d'."
                )
        return self


class EncoderPretrainGraphDatamodule(BaseDataModel):
    """Datamodule parameters for **graph** encoder pretraining.

    Keys like ``laplacian_k``, ``rwse_k``, etc. are also forwarded to the
    model constructor (as ``enc_laplacian_k``, …) because graph encoders
    need them to size their positional-encoding layers.
    """

    laplacian_k: int = 10
    rwse_k: int = 20
    elstatic_k: int = 0
    distmat_k: int = 0
    rrwp_k: int = 20
    compute_distances: bool = True
    num_virtual_nodes: int = 0
    init_virtual_nodes: bool = False
    batch_size: int = 256
    num_workers: int = 0
    augment_resonance: bool = False
    scale_y_graph: bool = False
    scale_y_node: bool = False


class EncoderPretrainMLMDatamodule(BaseDataModel):
    """Datamodule parameters for **MLM** encoder pretraining."""

    mask_rate: float = 0.15
    max_length: int = 128
    num_augmentations: int = 2
    num_test_augmentations: int = 4
    include_canonical: bool = True
    batch_size: int = 256
    num_workers: int = 0
    augment_resonance: bool = False


class EncoderPretrainModel(BaseDataModel):
    """Schema for encoder pretraining model configuration with architecture and datamodule."""

    architecture: str  # e.g. "RoFormerMLM", "GINPretraining", ...
    params: dict  # neural-network constructor args only
    datamodule: Optional[dict] = None  # datamodule params (graph or mlm)
    training: Optional[PretrainTraining] = PretrainTraining()


class CLIEncoderPretrainInputModel(BaseDataModel):
    """Top-level input schema for the ``pretrain_encoder`` CLI command."""

    dataset: EncoderPretrainDataset
    model: EncoderPretrainModel
    pipe: PretrainPipe
    output: PretrainOutput
    magic: Optional[str] = None


# Backward-compatible aliases for old names
Pipe = PretrainPipe
EncoderPretrainPipe = PretrainPipe
EncoderPretrainTraining = PretrainTraining
PretrainMultitaskOutput = PretrainOutput
