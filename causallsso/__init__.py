import warnings

from .config import SolveDeltaConfig
from .reference import (
    SolveDeltaState,
    solvedelta_reference,
    solvedelta_zero_state,
)

# PyTorch 2.13's Inductor import eagerly defines its unused MKLDNN TorchScript
# wrappers when FLA registers @torch.compile callables. PyTorch PR #189914 made
# those wrappers lazy. Keep the exact upstream deprecation local to this import
# window until a stable wheel contains that fix; all other warnings propagate.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(
            r"^`torch\.jit\.script_method` is deprecated\. "
            r"Please switch to `torch\.compile` or `torch\.export`\.$"
        ),
        category=DeprecationWarning,
        module=r"^torch\.jit\._script$",
    )

    from .model import SolveDelta, SolveDeltaLayerState
    from transformers import AutoConfig

    AutoConfig.register(SolveDeltaConfig.model_type, SolveDeltaConfig, exist_ok=True)

    try:
        from transformers import AutoModel, AutoModelForCausalLM

        from .modeling_solvedelta import (
            SolveDeltaBlock,
            SolveDeltaCacheAdapter,
            SolveDeltaForCausalLM,
            SolveDeltaModel,
            SolveDeltaPreTrainedModel,
        )
        from .cuda_graph import SolveDeltaGraphedTrainingStep
    except ModuleNotFoundError as error:
        if error.name is None or not error.name.startswith("fla"):
            raise
    else:
        AutoModel.register(SolveDeltaConfig, SolveDeltaModel, exist_ok=True)
        AutoModelForCausalLM.register(
            SolveDeltaConfig, SolveDeltaForCausalLM, exist_ok=True
        )

__all__ = [
    "SolveDelta",
    "SolveDeltaConfig",
    "SolveDeltaLayerState",
    "SolveDeltaState",
    "solvedelta_reference",
    "solvedelta_zero_state",
]

if "SolveDeltaModel" in globals():
    __all__ += [
        "SolveDeltaBlock",
        "SolveDeltaCacheAdapter",
        "SolveDeltaForCausalLM",
        "SolveDeltaGraphedTrainingStep",
        "SolveDeltaModel",
        "SolveDeltaPreTrainedModel",
    ]
