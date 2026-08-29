from __future__ import annotations

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

pytest.importorskip("fla")

from causallsso import (
    SolveDelta,
    SolveDeltaConfig,
    SolveDeltaForCausalLM,
    SolveDeltaGraphedTrainingStep,
    SolveDeltaModel,
)


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native SolveDelta requires CUDA"
)


def _tiny_config(**overrides) -> SolveDeltaConfig:
    values = dict(
        hidden_size=32,
        num_heads=2,
        head_k_dim=16,
        head_v_dim=16,
        num_hidden_layers=2,
        hidden_ratio=2,
        vocab_size=97,
        max_position_embeddings=64,
        use_short_conv=True,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        use_cache=False,
    )
    values.update(overrides)
    return SolveDeltaConfig(**values)


def test_huggingface_auto_registration_and_config_roundtrip():
    config = _tiny_config()
    restored = SolveDeltaConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert isinstance(
        AutoConfig.for_model("solvedelta", hidden_size=32, num_heads=2),
        SolveDeltaConfig,
    )
    assert isinstance(AutoModel.from_config(config), SolveDeltaModel)
    assert isinstance(AutoModelForCausalLM.from_config(config), SolveDeltaForCausalLM)


def test_huggingface_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(2)
    model = SolveDeltaForCausalLM(_tiny_config(num_hidden_layers=1))
    mixer = model.model.layers[0].mixer.mixer
    initial_geometry_write = torch.sigmoid(mixer.geometry_write_bias)
    expected_geometry_write = torch.full_like(
        initial_geometry_write, torch.sigmoid(torch.tensor(-2.0))
    )
    torch.testing.assert_close(
        initial_geometry_write,
        expected_geometry_write,
        rtol=1e-6,
        atol=1e-6,
    )

    single_head = SolveDelta(
        _tiny_config(
            hidden_size=16,
            num_heads=1,
            head_k_dim=16,
            head_v_dim=16,
            num_hidden_layers=1,
        )
    )
    single_head_write = torch.sigmoid(single_head.geometry_write_bias)
    torch.testing.assert_close(
        single_head_write,
        torch.full_like(single_head_write, torch.sigmoid(torch.tensor(-2.0))),
        rtol=1e-6,
        atol=1e-6,
    )

    model.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert isinstance(restored, SolveDeltaForCausalLM)
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_causal_lm_cpu_loss_backward_and_packed_segments():
    torch.manual_seed(3)
    model = SolveDeltaForCausalLM(_tiny_config()).train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]],
        dtype=torch.long,
    )
    labels = input_ids.masked_fill(~attention_mask.bool(), -100)
    result = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert result.logits.shape == (2, 8, model.config.vocab_size)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    model.eval()
    first = torch.randint(0, model.config.vocab_size, (1, 4))
    second = torch.randint(0, model.config.vocab_size, (1, 5))
    packed = torch.cat((first, second), dim=1)
    cu_seqlens = torch.tensor([0, 4, 9], dtype=torch.int32)
    with torch.no_grad():
        packed_logits = model(packed, cu_seqlens=cu_seqlens).logits
        separate_logits = torch.cat(
            (model(first).logits, model(second).logits), dim=1
        )
    assert torch.equal(packed_logits, separate_logits)


def test_recurrent_cache_matches_unsplit_model_and_owns_fp32_state():
    torch.manual_seed(5)
    model = SolveDeltaForCausalLM(_tiny_config(use_cache=True)).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 9))
    with torch.no_grad():
        whole = model(input_ids, use_cache=False).logits
        left = model(input_ids[:, :4], use_cache=True)
        right = model(
            input_ids[:, 4:], past_key_values=left.past_key_values, use_cache=True
        )
    assert torch.equal(right.logits, whole[:, 4:])
    assert left.past_key_values.get_seq_length() == input_ids.shape[1]
    assert len(left.past_key_values) == model.config.num_hidden_layers
    for layer_state in left.past_key_values:
        recurrent = layer_state["recurrent_state"]
        assert len(recurrent) == 2
        assert all(tensor.dtype == torch.float32 for tensor in recurrent)
        assert recurrent[0].shape[-2:] == (
            model.config.resolved_head_k_dim,
            model.config.resolved_head_k_dim,
        )

    with torch.no_grad():
        generated = model.generate(input_ids[:, :4], max_new_tokens=2, do_sample=False)
    assert torch.equal(generated[:, :4], input_ids[:, :4])
    assert 4 < generated.shape[1] <= 6


def test_cuda_graph_training_rejects_fused_linear_loss_before_capture():
    config = _tiny_config(
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=True,
    )
    model = SolveDeltaForCausalLM(config).train()
    sample_ids = torch.randint(0, config.vocab_size, (1, 8))
    with pytest.raises(ValueError, match="fused linear cross entropy"):
        SolveDeltaGraphedTrainingStep(model, sample_ids, sample_ids)


@CUDA_ONLY
def test_causal_lm_cuda_bf16_native_forward_backward(monkeypatch):
    import causallsso.model as solvedelta_model

    torch.manual_seed(7)
    config = _tiny_config(
        hidden_size=128,
        num_heads=1,
        head_k_dim=128,
        head_v_dim=128,
        num_hidden_layers=1,
        vocab_size=256,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=True,
    )
    model = SolveDeltaForCausalLM(config).cuda().train()
    input_ids = torch.randint(0, config.vocab_size, (1, 32), device="cuda")
    native_calls = 0
    native_operator = solvedelta_model.solvedelta_native

    def counted_native(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return native_operator(*args, **kwargs)

    monkeypatch.setattr(solvedelta_model, "solvedelta_native", counted_native)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(input_ids, labels=input_ids)
    assert native_calls == config.num_hidden_layers
    assert result.logits.dtype == torch.bfloat16
    assert torch.isfinite(result.logits).all()
    assert torch.isfinite(result.loss)
    result.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    mixer = model.model.layers[0].mixer.mixer
    assert mixer.geometry_write_bias.dtype == torch.float32
    assert mixer.associative_log_rate.dtype == torch.float32


@CUDA_ONLY
def test_causal_lm_cuda_graph_training_matches_eager():
    torch.manual_seed(11)
    config = _tiny_config(
        hidden_size=128,
        num_heads=1,
        head_k_dim=128,
        head_v_dim=128,
        num_hidden_layers=1,
        vocab_size=256,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=True,
    )
    eager_model = SolveDeltaForCausalLM(config).cuda().train()
    graphed_model = copy.deepcopy(eager_model).train()
    sample_ids = torch.randint(0, config.vocab_size, (1, 32), device="cuda")
    graph_step = SolveDeltaGraphedTrainingStep(
        graphed_model,
        sample_ids,
        sample_ids,
    )

    input_ids = torch.randint(0, config.vocab_size, (1, 32), device="cuda")
    eager_model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        eager_loss = eager_model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
            return_dict=False,
        )[0]
    eager_loss.backward()

    graphed_model.zero_grad(set_to_none=True)
    graphed_loss = graph_step(input_ids, input_ids)
    graphed_loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(graphed_loss, eager_loss, rtol=0, atol=0)
    for (eager_name, eager_parameter), (graph_name, graph_parameter) in zip(
        eager_model.named_parameters(), graphed_model.named_parameters()
    ):
        assert eager_name == graph_name
        assert eager_parameter.grad is not None
        assert graph_parameter.grad is not None
        torch.testing.assert_close(
            graph_parameter.grad,
            eager_parameter.grad,
            rtol=0,
            atol=0,
        )

    eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=1e-3)
    graph_optimizer = torch.optim.SGD(graphed_model.parameters(), lr=1e-3)
    eager_optimizer.step()
    graph_optimizer.step()

    next_ids = torch.randint(0, config.vocab_size, (1, 32), device="cuda")
    eager_model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        next_eager_loss = eager_model(
            input_ids=next_ids,
            labels=next_ids,
            use_cache=False,
            return_dict=False,
        )[0]
    next_eager_loss.backward()

    graphed_model.zero_grad(set_to_none=True)
    next_loss = graph_step(next_ids, next_ids)
    next_loss.backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(next_loss, next_eager_loss, rtol=0, atol=0)
    for eager_parameter, graph_parameter in zip(
        eager_model.parameters(), graphed_model.parameters()
    ):
        assert eager_parameter.grad is not None
        assert graph_parameter.grad is not None
        torch.testing.assert_close(
            graph_parameter.grad,
            eager_parameter.grad,
            rtol=0,
            atol=0,
        )

    with pytest.raises(ValueError, match="requires input shape"):
        graph_step(next_ids[:, :16], next_ids[:, :16])
