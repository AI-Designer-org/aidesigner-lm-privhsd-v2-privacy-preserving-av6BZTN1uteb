"""PrivHSD v2: Privacy-preserving Hate Speech Detection."""

from src.model import (
    PrivHSDConfig,
    PrivHSDModelV2,
    GradientReversalLayer,
    AdaptiveAlphaScheduler,
    AdversaryMLP,
    MultiLevelAdversarialBlock,
    MutualInformationMinimizer,
    HateClassificationHead,
    compute_subspace_orthogonality,
)
from src.data_utils import (
    HateSpeechDataset,
    create_author_labels,
    load_jigsaw_dataset,
    load_hatexplain_dataset,
    create_privacy_augmented_variant,
    get_dataloaders,
)
from src.train import PrivHSDTrainer
from src.evaluate import (
    evaluate_model,
    ParetoFrontierAnalyzer,
    EvaluationResult,
    UtilityMetrics,
    PrivacyMetrics,
    compute_utility_metrics,
)
from src.attacks import (
    MembershipInferenceAttack,
    AttributeInferenceAttack,
    StylometryReidentificationRisk,
    RepresentationPrivacyAudit,
    AttackMetrics,
)
from src.inference import PrivHSDInference

__version__ = "2.0.0"
