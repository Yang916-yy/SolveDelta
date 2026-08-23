import ast
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
            "native/solvedelta_c32_backward.cu",
        )
    )
    forbidden_native_abi = (
        "c32_frame_forward_cuda",
        "c32_frame_backward_cuda",
        '"c32_frame_forward(',
        '"c32_frame_backward(',
    )
    assert all(symbol not in native_sources for symbol in forbidden_native_abi)


def test_single_native_training_path_is_present() -> None:
    required_paths = (
        "causallsso/ops/chunk_wy.py",
        "causallsso/ops/native_chunk.py",
        "causallsso/ops/resident_frame.py",
        "causallsso/ops/triton_geometry.py",
        "causallsso/ops/wy.py",
    )
    assert all((ROOT / path).is_file() for path in required_paths)

    chunk_wy = (ROOT / "causallsso/ops/chunk_wy.py").read_text()
    assert "_CHUNK_SIZE = 32" in chunk_wy
    assert "_RANK = 128" in chunk_wy
    assert "_EDITS = 1" in chunk_wy
    assert "native_geometry_frame" in chunk_wy
    assert "wy_associative" in chunk_wy
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
        "c32_frame_resident_forward",
        "c32_frame_resident_action_backward",
        "c32_frame_compact_pair",
        "c32_frame_compact_coefficients",
        "c32_frame_compact_leaf",
    ):
        assert symbol in native_chunk

    wy = (ROOT / "causallsso/ops/wy.py").read_text()
    assert "_direct_e_fwd_intra_kernel" in wy
    assert "_direct_e_bwd_intra_kernel" in wy


def test_mathdx_is_an_optional_oracle_only() -> None:
    cmake = (ROOT / "native" / "CMakeLists.txt").read_text()
    assert "CAUSALLSSO_BUILD_MATHDX_ORACLE" in cmake
    assert "if(CAUSALLSSO_BUILD_MATHDX_ORACLE)" in cmake

    model_source = (ROOT / "causallsso" / "model.py").read_text()
    assert "mathdx" not in model_source.lower()


def test_adapted_upstream_sources_are_attributed() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "NVIDIA cuBLASDx TRSM block example" in notice
    assert "Flash Linear Attention generalized Delta/WY kernels" in notice
    assert (ROOT / "LICENSES" / "Apache-2.0.txt").is_file()
    assert (ROOT / "LICENSES" / "MIT.txt").is_file()
