from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pnpm_11_approves_esbuild_install_script() -> None:
    workspace_config = (ROOT / "frontend" / "pnpm-workspace.yaml").read_text(
        encoding="utf-8"
    )

    assert "allowBuilds:" in workspace_config
    assert "  esbuild: true" in workspace_config
    assert "onlyBuiltDependencies:" not in workspace_config
