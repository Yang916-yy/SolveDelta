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
