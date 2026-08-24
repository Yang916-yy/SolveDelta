import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_abandoned_operator_name_is_absent() -> None:
    forbidden = ("TD-SFD", "Transpose-Dual Solve-Frame Delta")
    for path in [ROOT / "README.md", ROOT / "AGENTS.md", *(ROOT / "docs").glob("*.md")]:
        text = path.read_text()
        assert not any(name in text for name in forbidden), path


def test_reference_is_the_only_operator_math_module() -> None:
    modules = sorted((ROOT / "causallsso").glob("*.py"))
    assert ROOT / "causallsso" / "reference.py" in modules
    assert not (ROOT / "causallsso" / "naive.py").exists()


def test_deleted_backend_abis_do_not_return() -> None:
    forbidden_paths = (
        "causallsso/ops/delta_outer.py",
        "causallsso/ops/packet_frame.py",
        "causallsso/ops/panel_frame.py",
        "causallsso/ops/chunk_frame.py",
        "causallsso/ops/tensorcore_frame.py",
        "causallsso/ops/triton_frame.py",
        "causallsso/ops/triton_bounded_ldu.py",
        "causallsso/ops/resident_frame.py",
        "causallsso/ops/wy.py",
        "causallsso/ops/wy_stage.py",
        "native/solvedelta_c32_backward.cu",
    )
    assert all(not (ROOT / path).exists() for path in forbidden_paths)

    public_ops = (ROOT / "causallsso" / "ops" / "__init__.py").read_text()
    forbidden_symbols = (
        "packet_frame",
        "panel_frame",
        "chunk_frame",
        "tensorcore_frame",
        "triton_frame",
        "cuda_chunk_solve",
        "bounded_ldu_vjp",
        "mathdx",
    )
    assert all(symbol not in public_ops for symbol in forbidden_symbols)

    native_sources = "\n".join(
        (ROOT / path).read_text()
        for path in (
            "native/solvedelta_c32.h",
            "native/solvedelta_c32_forward.cu",
            "native/solvedelta_wy.cu",
            "native/solvedelta_prepare.cu",
        )
    )
    forbidden_native_abi = (
        "c32_frame_forward_cuda",
        "c32_frame_backward_cuda",
        '"c32_frame_forward(',
        '"c32_frame_backward(',
        "c32_frame_resident_forward",
        "c32_frame_resident_action_backward",
        "c32_frame_wy_stage_forward",
        "c32_frame_wy_stage_action_backward",
        "c32_frame_compact_pair",
        "c32_frame_compact_coefficients",
        "c32_frame_compact_leaf",
        "wy_solve_backward_kernel",
        "solution[kChunk]",
        "value_backward_kernel",
    )
    assert all(symbol not in native_sources for symbol in forbidden_native_abi)
    assert re.search(r"\bgrad_(?:d|e|chi)\b", native_sources) is None

    frame_source = (
        ROOT / "native" / "solvedelta_c32_forward.cu"
    ).read_text()
    assert "at::BFloat16* __restrict__ descriptor_bundle" in frame_source
    assert "descriptor_bundle.data_ptr<at::BFloat16>()" in frame_source


def test_single_native_training_path_is_present() -> None:
    required_paths = (
        "causallsso/ops/chunk_wy.py",
        "causallsso/ops/native_chunk.py",
        "causallsso/ops/paired_wy.py",
        "causallsso/ops/chunk_state.py",
        "causallsso/ops/radial_compact.py",
        "causallsso/ops/strict_chart.py",
        "causallsso/ops/triton_geometry.py",
        "native/solvedelta_c32_forward.cu",
        "native/solvedelta_wy.cu",
        "native/solvedelta_prepare.cu",
    )
    assert all((ROOT / path).is_file() for path in required_paths)

    chunk_wy = (ROOT / "causallsso/ops/chunk_wy.py").read_text()
    assert "_CHUNK_SIZE = 32" in chunk_wy
    assert "_RANK = 128" in chunk_wy
    assert "_EDITS = 1" in chunk_wy
    assert "native_chunk_solvedelta" in chunk_wy
    for forbidden in (
        "native_geometry_wy_stage",
        "wy_stage_statistics",
        "wy_associative_staged",
        "qg",
        "kg",
        "ag",
        "A_ad",
    ):
        assert forbidden not in chunk_wy
    module = ast.parse(chunk_wy)
    entrypoint = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "chunk_wy_solvedelta"
    )
    parameters = {
        argument.arg
        for argument in (*entrypoint.args.args, *entrypoint.args.kwonlyargs)
    }
    assert {"backend", "wy_dtype", "chunk_size"}.isdisjoint(parameters)

    native_chunk = (ROOT / "causallsso/ops/native_chunk.py").read_text()
    for symbol in (
        "c32_solvedelta_prepare_forward",
        "c32_solvedelta_prepare_backward",
    ):
        assert symbol in native_chunk
    for symbol in (
        "c32_frame_actions_forward",
        "c32_frame_actions_backward",
        "c32_wy_statistics_forward",
        "c32_wy_solve_forward",
        "c32_wy_solve_backward",
        "c32_wy_pair_backward",
        "c32_frame_wy_stage_forward",
        "c32_frame_wy_stage_action_backward",
        "c32_frame_compact_pair",
        "c32_frame_compact_coefficients",
        "c32_frame_compact_leaf",
    ):
        assert symbol not in native_chunk


def test_mathdx_is_an_optional_oracle_only() -> None:
    cmake = (ROOT / "native" / "CMakeLists.txt").read_text()
    assert "CAUSALLSSO_BUILD_MATHDX_ORACLE" in cmake
    assert "if(CAUSALLSSO_BUILD_MATHDX_ORACLE)" in cmake

    model_source = (ROOT / "causallsso" / "model.py").read_text()
    assert "mathdx" not in model_source.lower()


def test_bf16_observable_numerical_contract_is_current() -> None:
    validation = (ROOT / "docs" / "VALIDATION_PLAN.md").read_text()
    parallelism = (ROOT / "docs" / "PARALLELISM.md").read_text()
    for text in (validation, parallelism):
        normalized = " ".join(text.split())
        assert "nonstructural exact cancellation" in normalized
        assert "A=aZ" in normalized
        assert "bar a=<bar A,Z>" in normalized
        assert "private `q2`" in normalized
        assert "expm1" in normalized
    assert "exact zero must emit an exact zero radial component" not in validation

    paired_wy = (ROOT / "causallsso" / "ops" / "paired_wy.py").read_text()
    paired_wy_test = (ROOT / "tests" / "core" / "test_paired_wy.py").read_text()
    assert "_twofold_bf16_dot" not in paired_wy
    assert "_paired_wy_forward_fp32_diagnostic" not in paired_wy
    assert "_max_eta" not in paired_wy_test
    assert "no private inverse/residual gate" in validation


def test_bounded_private_fp16_contract_is_current() -> None:
    validation = (ROOT / "docs" / "VALIDATION_PLAN.md").read_text()
    parallelism = (ROOT / "docs" / "PARALLELISM.md").read_text()
    for text in (validation, parallelism):
        normalized = " ".join(text.split())
        assert "bounded private FP16" in normalized
        assert "BF16 -> FP16" in normalized
        assert "pseudo-promotion" in normalized
        assert "FP32 producer" in normalized
        assert "FP32 accumulation" in normalized


def test_adapted_upstream_sources_are_attributed() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "NVIDIA cuBLASDx TRSM block example" in notice
    assert "Flash Linear Attention generalized Delta/WY kernels" in notice
    assert "causallsso/ops/paired_wy.py" in notice
    assert "wu_fwd_kernel" in notice
    assert (ROOT / "LICENSES" / "Apache-2.0.txt").is_file()
    assert (ROOT / "LICENSES" / "MIT.txt").is_file()
