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
        "causallsso/ops/triton_bounded_ldu.py",
    )
    assert all(not (ROOT / path).exists() for path in forbidden_paths)

    public_ops = (ROOT / "causallsso" / "ops" / "__init__.py").read_text()
    forbidden_symbols = (
        "packet_frame",
        "panel_frame",
        "cuda_chunk_solve",
        "bounded_ldu_vjp",
        "mathdx",
    )
    assert all(symbol not in public_ops for symbol in forbidden_symbols)


def test_mathdx_is_an_optional_oracle_only() -> None:
    cmake = (ROOT / "native" / "CMakeLists.txt").read_text()
    assert "CAUSALLSSO_BUILD_MATHDX_ORACLE" in cmake
    assert "if(CAUSALLSSO_BUILD_MATHDX_ORACLE)" in cmake

    model_source = (ROOT / "causallsso" / "model.py").read_text()
    assert "mathdx" not in model_source.lower()
