## Legacy Embedding Outputs

The embedding parity harness writes or expects legacy script outputs in this directory. Each entry in `validation/embedding_parity_config.yaml` can specify a `legacy.output_path` under this folder. When `legacy.command` is provided, the harness will run it and deposit the resulting TSV here; otherwise, you can manually place the legacy outputs before running `--task validate-embeddings`.

