from pathlib import Path

from shared.config import load_config


def test_load_config_expands_optional_adata_path(temp_dir):
    config_path = temp_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: demo",
                "  expression_path: data/expression.json",
                "  adata_path: data/demo/perturb_processed.h5ad",
                "  gene_names_path: data/gene_names.json",
                "tasks:",
                "  - class",
                "output_root: results/demo",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.dataset.adata_path == (temp_dir / "data/demo/perturb_processed.h5ad").resolve()
    assert cfg.output_root == (temp_dir / "results/demo").resolve()
    assert cfg.dataset.expression_path == (temp_dir / "data/expression.json").resolve()
