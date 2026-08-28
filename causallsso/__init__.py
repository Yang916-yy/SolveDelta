from .config import SolveDeltaConfig
from .model import SolveDelta, SolveDeltaLayerState
from .reference import (
    SolveDeltaState,
    solvedelta_reference,
    solvedelta_zero_state,
)

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
        "SolveDeltaModel",
        "SolveDeltaPreTrainedModel",
    ]
