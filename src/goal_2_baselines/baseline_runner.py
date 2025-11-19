"""
Baseline runner for reproducing all 8 linear baseline models.

This module implements the complete baseline reproduction pipeline:
1. Load data and train/test/val splits
2. Compute Y (pseudobulk expression changes)
3. Construct A (gene embeddings) and B (perturbation embeddings) for each baseline
4. Solve for K (interaction matrix) via ridge regression
5. Make predictions and evaluate
6. Save results
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from shared.linear_model import solve_y_axb
from shared.metrics import compute_metrics

from .baseline_types import BaselineConfig, BaselineType, get_baseline_config
from .split_logic import load_split_config, prepare_perturbation_splits

# Optional import - paper_implementation may have been moved to goal_5_validation
try:
    from .paper_implementation import run_paper_baseline
except ImportError:
    try:
        from goal_5_validation.paper_implementation import run_paper_baseline
    except ImportError:
        run_paper_baseline = None

LOGGER = logging.getLogger(__name__)


def compute_pseudobulk_expression_changes(
    adata: ad.AnnData,
    split_config: Dict[str, List[str]],
    seed: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Compute pseudobulk expression changes (Y matrix).
    
    This is the same for all 8 baselines:
    Y_{i,j} = mean(expression of gene i in perturbation j) - mean(expression of gene i in ctrl)
    
    Args:
        adata: AnnData object with expression data
        split_config: Dictionary with 'train', 'test', 'val' keys mapping to condition lists
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (Y matrix as DataFrame, split labels dictionary)
    """
    # Filter to valid conditions
    all_conditions = []
    for conditions in split_config.values():
        all_conditions.extend(conditions)
    all_conditions = list(set(all_conditions))
    
    adata = adata[adata.obs["condition"].isin(all_conditions)].copy()
    
    # Clean condition names (remove +ctrl suffix)
    adata.obs["clean_condition"] = (
        adata.obs["condition"].astype(str).str.replace(r"\+ctrl", "", regex=True)
    )
    
    # Compute baseline (mean expression in control)
    ctrl_mask = adata.obs["condition"] == "ctrl"
    if ctrl_mask.sum() == 0:
        raise ValueError("No control condition found in data")
    
    baseline = np.asarray(adata.X[ctrl_mask].mean(axis=0)).ravel()
    
    # Pseudobulk by condition
    unique_conditions = adata.obs["clean_condition"].unique()
    
    pseudobulk_data = []
    condition_labels = []
    
    for cond in unique_conditions:
        cond_mask = adata.obs["clean_condition"] == cond
        if cond_mask.sum() == 0:
            continue
        
        # Average expression within condition
        cond_expr = adata.X[cond_mask]
        if hasattr(cond_expr, "toarray"):
            cond_expr = cond_expr.toarray()
        mean_expr = np.asarray(cond_expr.mean(axis=0)).ravel()
        
        # Compute change from baseline
        change = mean_expr - baseline
        
        pseudobulk_data.append(change)
        condition_labels.append(cond)
    
    # Create Y matrix (genes × perturbations)
    Y = np.vstack(pseudobulk_data).T  # Transpose to get genes × perts
    
    # Create DataFrame
    gene_names = adata.var_names.tolist()
    Y_df = pd.DataFrame(Y, index=gene_names, columns=condition_labels)
    
    # Map conditions to splits
    # Clean condition names in split_config (remove +ctrl suffix)
    clean_split_config = {}
    for split_name, conditions in split_config.items():
        clean_split_config[split_name] = [
            cond.replace("+ctrl", "") if "+ctrl" in cond else cond
            for cond in conditions
        ]
    
    split_labels = {}
    for split_name, clean_conditions in clean_split_config.items():
        split_labels[split_name] = [
            cond for cond in condition_labels if cond in clean_conditions
        ]
    
    return Y_df, split_labels


def construct_gene_embeddings(
    source: str,
    train_data: np.ndarray,
    gene_names: List[str],
    pca_dim: int,
    seed: int,
    embedding_args: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct gene embeddings (A matrix).
    
    Args:
        source: "training_data", "random", or embedding loader name
        train_data: Training data matrix (genes × train_perturbations)
        gene_names: List of gene names
        pca_dim: PCA dimension
        seed: Random seed
        embedding_args: Optional arguments for embedding loaders
    
    Returns:
        Tuple of (A matrix, gene labels)
    """
    if source == "training_data":
        # PCA on training data genes
        # train_data is (genes, train_perts)
        # We want A to be (genes, d), so we do PCA on genes (treating genes as observations)
        pca = PCA(n_components=pca_dim, random_state=seed)
        A = pca.fit_transform(train_data)  # genes × d (genes as observations, train_perts as features)
        return A, gene_names
    
    elif source == "random":
        # Random Gaussian matrix
        rng = np.random.default_rng(seed)
        A = rng.normal(0, 1, size=(train_data.shape[0], pca_dim))
        return A, gene_names
    
    elif source == "scgpt":
        # Load scGPT embeddings
        from embeddings.registry import load
        
        if embedding_args is None:
            raise ValueError("scgpt requires embedding_args with checkpoint_dir")
        
        # Resolve relative paths
        embedding_args = embedding_args.copy()
        gene_name_mapping = embedding_args.pop("gene_name_mapping", None)  # Remove from args, handle separately
        if "checkpoint_dir" in embedding_args:
            checkpoint_dir = Path(embedding_args["checkpoint_dir"])
            if not checkpoint_dir.is_absolute():
                # Resolve relative to evaluation_framework root
                eval_framework_root = Path(__file__).parent.parent.parent
                checkpoint_dir = eval_framework_root / checkpoint_dir
            embedding_args["checkpoint_dir"] = str(checkpoint_dir.resolve())
        
        result = load("scgpt_gene", **embedding_args)
        # Align with gene_names
        # result.values is (dims × genes), result.item_labels are gene names (gene symbols)
        # gene_names are var_names (may be Ensembl IDs), use gene_name_mapping if available
        emb_gene_names = result.item_labels  # Gene symbols from scGPT
        
        # Map target gene_names to gene symbols if mapping available
        # gene_name_mapping was popped from embedding_args above
        if gene_name_mapping:
            # Map var_names to gene symbols
            target_gene_symbols = [gene_name_mapping.get(g, g) for g in gene_names]
        else:
            target_gene_symbols = gene_names
        
        common_genes = sorted(set(target_gene_symbols) & set(emb_gene_names))
        if len(common_genes) == 0:
            raise ValueError("No common genes between scGPT embeddings and target dataset")
        
        # Align embeddings to common genes
        emb_gene_idx = [emb_gene_names.index(g) for g in common_genes]
        aligned_emb = result.values[:, emb_gene_idx]  # dims × common_genes
        
        # Align target gene_names to common genes
        target_gene_idx = [i for i, g in enumerate(target_gene_symbols) if g in common_genes]
        common_target_genes = [target_gene_symbols[i] for i in target_gene_idx]
        
        # Create full embedding matrix aligned to target genes
        # For genes not in embeddings, use zeros
        # aligned_emb is (dims × common_genes), we want A_full to be (genes × dims)
        A_full = np.zeros((len(gene_names), aligned_emb.shape[0]))
        # Map common genes back to indices
        for i, g_sym in enumerate(common_target_genes):
            target_idx = target_gene_idx[i]
            emb_idx = common_genes.index(g_sym)
            A_full[target_idx, :] = aligned_emb[:, emb_idx]  # (dims,) -> (1 × dims)
        
        return A_full, gene_names
    
    elif source == "scfoundation":
        # Load scFoundation embeddings
        from embeddings.registry import load
        
        if embedding_args is None:
            raise ValueError("scfoundation requires embedding_args")
        
        # Resolve relative paths
        embedding_args = embedding_args.copy()
        gene_name_mapping = embedding_args.pop("gene_name_mapping", None)  # Remove from args, handle separately
        eval_framework_root = Path(__file__).parent.parent.parent
        if "checkpoint_path" in embedding_args:
            checkpoint_path = Path(embedding_args["checkpoint_path"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = eval_framework_root / checkpoint_path
            embedding_args["checkpoint_path"] = str(checkpoint_path.resolve())
        if "demo_h5ad" in embedding_args:
            demo_h5ad = Path(embedding_args["demo_h5ad"])
            if not demo_h5ad.is_absolute():
                demo_h5ad = eval_framework_root / demo_h5ad
            embedding_args["demo_h5ad"] = str(demo_h5ad.resolve())
        
        result = load("scfoundation_gene", **embedding_args)
        # Align with gene_names
        # result.values is (dims × genes), result.item_labels are gene names (gene symbols)
        # gene_names are var_names (may be Ensembl IDs), use gene_name_mapping if available
        emb_gene_names = result.item_labels  # Gene symbols from scFoundation
        
        # Map target gene_names to gene symbols if mapping available
        # gene_name_mapping was popped from embedding_args above
        if gene_name_mapping:
            # Map var_names to gene symbols
            target_gene_symbols = [gene_name_mapping.get(g, g) for g in gene_names]
        else:
            target_gene_symbols = gene_names
        
        common_genes = sorted(set(target_gene_symbols) & set(emb_gene_names))
        if len(common_genes) == 0:
            raise ValueError("No common genes between scFoundation embeddings and target dataset")
        
        # Align embeddings to common genes
        emb_gene_idx = [emb_gene_names.index(g) for g in common_genes]
        aligned_emb = result.values[:, emb_gene_idx]  # dims × common_genes
        
        # Align target gene_names to common genes
        target_gene_idx = [i for i, g in enumerate(target_gene_symbols) if g in common_genes]
        common_target_genes = [target_gene_symbols[i] for i in target_gene_idx]
        
        # Create full embedding matrix aligned to target genes
        # For genes not in embeddings, use zeros
        # aligned_emb is (dims × common_genes), we want A_full to be (genes × dims)
        A_full = np.zeros((len(gene_names), aligned_emb.shape[0]))
        # Map common genes back to indices
        for i, g_sym in enumerate(common_target_genes):
            target_idx = target_gene_idx[i]
            emb_idx = common_genes.index(g_sym)
            A_full[target_idx, :] = aligned_emb[:, emb_idx]  # (dims,) -> (1 × dims)
        
        return A_full, gene_names
    
    else:
        raise ValueError(f"Unknown gene embedding source: {source}")


def construct_pert_embeddings(
    source: str,
    train_data: np.ndarray,
    pert_names: List[str],
    pca_dim: int,
    seed: int,
    embedding_args: Optional[Dict] = None,
    test_data: Optional[np.ndarray] = None,
    test_pert_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Optional[PCA], Optional[np.ndarray], Optional[List[str]]]:
    """
    Construct perturbation embeddings (B matrix).
    
    Args:
        source: "training_data", "random", or embedding loader name
        train_data: Training data matrix (genes × train_perturbations)
        pert_names: List of perturbation names
        pca_dim: PCA dimension
        seed: Random seed
        embedding_args: Optional arguments for embedding loaders
        test_data: Optional test data matrix (genes × test_perturbations)
        test_pert_names: Optional test perturbation names
    
    Returns:
        Tuple of (B_train, train_labels, pca_object, B_test, test_labels)
        pca_object and B_test are None if not applicable
    """
    if source == "training_data":
        # PCA on training data perturbations
        pca = PCA(n_components=pca_dim, random_state=seed)
        pert_emb_train = pca.fit_transform(train_data.T)  # perturbations × d
        B_train = pert_emb_train.T  # d × perturbations
        
        # Transform test data if provided
        B_test = None
        if test_data is not None:
            pert_emb_test = pca.transform(test_data.T)  # test_perturbations × d
            B_test = pert_emb_test.T  # d × test_perturbations
        
        return B_train, pert_names, pca, B_test, test_pert_names
    
    elif source == "random":
        # Random Gaussian matrix
        rng = np.random.default_rng(seed)
        B_train = rng.normal(0, 1, size=(pca_dim, train_data.shape[1]))
        
        # Generate random test embeddings
        B_test = None
        if test_data is not None:
            B_test = rng.normal(0, 1, size=(pca_dim, test_data.shape[1]))
        
        return B_train, pert_names, None, B_test, test_pert_names
    
    elif source == "gears":
        # Load GEARS embeddings
        from embeddings.registry import load
        
        if embedding_args is None:
            raise ValueError("gears requires embedding_args with source_csv")
        
        # Resolve relative paths
        embedding_args = embedding_args.copy()
        if "source_csv" in embedding_args:
            source_csv = Path(embedding_args["source_csv"])
            if not source_csv.is_absolute():
                # Resolve relative to evaluation_framework root
                eval_framework_root = Path(__file__).parent.parent.parent
                source_csv = eval_framework_root / source_csv
            embedding_args["source_csv"] = str(source_csv.resolve())
        
        result = load("gears_go", **embedding_args)
        # Align with pert_names
        # GEARS embeddings use gene symbols, pert_names may have "+ctrl" suffix
        # Clean pert_names to match GEARS gene symbols
        gears_pert_names = result.item_labels  # Gene symbols from GEARS
        cleaned_pert_names = [p.replace("+ctrl", "") for p in pert_names]
        
        # Find common perturbations
        common_perts = sorted(set(cleaned_pert_names) & set(gears_pert_names))
        if len(common_perts) == 0:
            raise ValueError("No common perturbations between GEARS embeddings and target dataset")
        
        # Align GEARS embeddings to common perturbations
        gears_pert_idx = [gears_pert_names.index(p) for p in common_perts]
        aligned_gears_emb = result.values[:, gears_pert_idx]  # dims × common_perts
        
        # Align target perturbations to common perturbations
        target_pert_idx = [i for i, p in enumerate(cleaned_pert_names) if p in common_perts]
        common_target_perts = [cleaned_pert_names[i] for i in target_pert_idx]
        
        # Create full embedding matrix aligned to target perturbations
        # For perturbations not in GEARS, use zeros
        B_full = np.zeros((aligned_gears_emb.shape[0], len(pert_names)))  # dims × perts
        # Map common perturbations back to indices
        for i, p_sym in enumerate(common_target_perts):
            target_idx = target_pert_idx[i]
            gears_idx = common_perts.index(p_sym)
            B_full[:, target_idx] = aligned_gears_emb[:, gears_idx]  # (dims,) -> (dims × 1)
        
        # For test data, do the same alignment
        B_test = None
        if test_pert_names is not None:
            cleaned_test_pert_names = [p.replace("+ctrl", "") for p in test_pert_names]
            common_test_perts = sorted(set(cleaned_test_pert_names) & set(gears_pert_names))
            if len(common_test_perts) > 0:
                gears_test_idx = [gears_pert_names.index(p) for p in common_test_perts]
                aligned_gears_test_emb = result.values[:, gears_test_idx]  # dims × common_test_perts
                
                test_pert_idx = [i for i, p in enumerate(cleaned_test_pert_names) if p in common_test_perts]
                B_test = np.zeros((aligned_gears_test_emb.shape[0], len(test_pert_names)))  # dims × test_perts
                for i, p_sym in enumerate([cleaned_test_pert_names[j] for j in test_pert_idx]):
                    target_idx = test_pert_idx[i]
                    gears_idx = common_test_perts.index(p_sym)
                    B_test[:, target_idx] = aligned_gears_test_emb[:, gears_idx]
        
        return B_full, pert_names, None, B_test, test_pert_names
    
    elif source in ["k562_pca", "rpe1_pca"]:
        # Cross-dataset PCA: Use PCA fitted on K562/RPE1 to transform target dataset perturbations
        import anndata as ad
        from scipy import sparse
        
        # Load source dataset - use direct path for both K562 and RPE1
        if embedding_args is None or "source_adata_path" not in embedding_args:
            raise ValueError(f"{source} requires source_adata_path in embedding_args (path to K562/RPE1 h5ad)")
        
        source_adata_path = Path(embedding_args["source_adata_path"])
        if not source_adata_path.exists():
            raise FileNotFoundError(f"Source data file not found: {source_adata_path}")
        
        source_adata = ad.read_h5ad(source_adata_path)
        
        # Load source dataset and compute PCA on it
        # For cross-dataset, we need to fit PCA on source data, then transform target data
        
        # Compute pseudobulk on source data (same logic as pca_perturbation loader)
        source_adata.obs["clean_condition"] = (
            source_adata.obs["condition"].astype(str).str.replace(r"\+ctrl", "", regex=True)
        )
        source_adata.obs["_key"] = (
            source_adata.obs["condition"].astype(str) + "|" + source_adata.obs["clean_condition"]
        )
        unique_keys = source_adata.obs["_key"].unique().tolist()
        
        source_aggregated = []
        for key in unique_keys:
            idx = np.where(source_adata.obs["_key"] == key)[0]
            if len(idx) == 0:
                continue
            data = source_adata.X[idx]
            if sparse.issparse(data):
                mean_vec = np.asarray(data.mean(axis=0)).ravel()
            else:
                mean_vec = data.mean(axis=0)
            source_aggregated.append(np.asarray(mean_vec).ravel())
        
        source_matrix = np.vstack(source_aggregated)  # source_perts × genes
        source_gene_names = source_adata.var_names.tolist()
        
        # Align genes between source and target
        # train_data is (genes × train_perts), we need gene_names to align
        # Get target gene names from the calling context (passed via embedding_args or use var_names)
        target_gene_names = embedding_args.get("target_gene_names", None)
        if target_gene_names is None:
            # Try to infer from train_data shape - but we don't have gene names here
            # We need to pass gene_names through embedding_args
            raise ValueError("target_gene_names must be provided in embedding_args for cross-dataset PCA")
        
        # Find common genes
        common_genes = sorted(set(source_gene_names) & set(target_gene_names))
        if len(common_genes) == 0:
            raise ValueError("No common genes between source and target datasets")
        
        # Align source matrix to common genes
        source_gene_idx = [source_gene_names.index(g) for g in common_genes]
        source_matrix_aligned = source_matrix[:, source_gene_idx]  # source_perts × common_genes
        
        # Fit PCA on source perturbations (aligned to common genes)
        pca = PCA(n_components=pca_dim, random_state=seed)
        pca.fit(source_matrix_aligned)  # source_matrix is (source_perts × common_genes)
        
        # Align target data to common genes
        target_gene_idx = [target_gene_names.index(g) for g in common_genes]
        train_data_aligned = train_data[target_gene_idx, :]  # common_genes × train_perts
        
        # Transform target training perturbations
        pert_emb_train = pca.transform(train_data_aligned.T)  # train_perts × d
        B_train = pert_emb_train.T  # d × train_perts
        
        # Transform test perturbations if provided
        B_test = None
        if test_data is not None:
            test_data_aligned = test_data[target_gene_idx, :]  # common_genes × test_perts
            pert_emb_test = pca.transform(test_data_aligned.T)  # test_perts × d
            B_test = pert_emb_test.T  # d × test_perts
        
        return B_train, pert_names, pca, B_test, test_pert_names
    
    else:
        raise ValueError(f"Unknown perturbation embedding source: {source}")


def run_mean_response_baseline(
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
) -> Dict:
    """
    Run mean-response baseline (always predicts mean expression).
    
    This is a simple baseline that always predicts the mean expression
    across training perturbations for each gene.
    
    Args:
        Y_train: Training Y matrix (genes × train_perturbations)
        Y_test: Test Y matrix (genes × test_perturbations)
    
    Returns:
        Dictionary with predictions and metrics
    """
    LOGGER.info("Running mean-response baseline")
    
    # Mean response = mean across training perturbations for each gene
    mean_response = Y_train.values.mean(axis=1, keepdims=True)
    
    # Predict mean for all test perturbations
    Y_test_np = Y_test.values
    Y_pred_test = np.tile(mean_response, (1, Y_test_np.shape[1]))
    
    # Compute metrics
    test_pert_names = Y_test.columns.tolist()
    metrics = {}
    for i, pert_name in enumerate(test_pert_names):
        y_true = Y_test_np[:, i]
        y_pred = Y_pred_test[:, i]
        pert_metrics = compute_metrics(y_true, y_pred)
        metrics[pert_name] = pert_metrics
    
    return {
        "baseline_type": "mean_response",
        "predictions": Y_pred_test,
        "metrics": metrics,
        "mean_response": mean_response,
    }


def run_single_baseline(
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
    config: BaselineConfig,
    gene_names: List[str],
    gene_name_mapping: Optional[Dict[str, str]] = None,
    use_paper_implementation: bool = False,
    adata_path: Optional[Path] = None,
    split_config: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Run a single baseline model.
    
    Args:
        Y_train: Training Y matrix (genes × train_perturbations)
        Y_test: Test Y matrix (genes × test_perturbations)
        config: Baseline configuration
        gene_names: List of gene names
    
    Returns:
        Dictionary with predictions and metrics
    """
    LOGGER.info(f"Running baseline: {config.baseline_type.value}")
    
    # Use paper's implementation if requested
    if use_paper_implementation and adata_path and split_config:
        LOGGER.info("Using paper's validated Python implementation")
        
        # Map baseline type to paper's embedding arguments
        gene_emb_map = {
            BaselineType.SELFTRAINED: "training_data",
            BaselineType.RANDOM_PERT_EMB: "training_data",
            BaselineType.RANDOM_GENE_EMB: "random",
            BaselineType.K562_PERT_EMB: "training_data",
            BaselineType.RPE1_PERT_EMB: "training_data",
        }
        
        pert_emb_map = {
            BaselineType.SELFTRAINED: "training_data",
            BaselineType.RANDOM_PERT_EMB: "random",
            BaselineType.RANDOM_GENE_EMB: "training_data",
            BaselineType.K562_PERT_EMB: "file",
            BaselineType.RPE1_PERT_EMB: "file",
        }
        
        gene_emb = gene_emb_map.get(config.baseline_type, "training_data")
        pert_emb = pert_emb_map.get(config.baseline_type, "training_data")
        
        # Get perturbation embedding path for cross-dataset baselines
        pert_emb_path = None
        if config.baseline_type == BaselineType.K562_PERT_EMB:
            # Use precomputed embedding if available
            precomputed_path = Path(__file__).parent.parent.parent / "results" / "replogle_k562_pert_emb_pca10_seed1.tsv"
            if precomputed_path.exists():
                pert_emb_path = precomputed_path
            else:
                # Fall back to source_adata_path if precomputed not available
                pert_emb_path = config.pert_embedding_args.get("source_adata_path")
        elif config.baseline_type == BaselineType.RPE1_PERT_EMB:
            # Use precomputed embedding if available
            precomputed_path = Path(__file__).parent.parent.parent / "results" / "replogle_rpe1_pert_emb_pca10_seed1.tsv"
            if precomputed_path.exists():
                pert_emb_path = precomputed_path
            else:
                # Fall back to source_adata_path if precomputed not available
                pert_emb_path = config.pert_embedding_args.get("source_adata_path")
        
        result = run_paper_baseline(
            adata_path=adata_path,
            split_config=split_config,
            gene_embedding=gene_emb,
            pert_embedding=pert_emb if pert_emb != "file" else "file",
            pca_dim=config.pca_dim,
            ridge_penalty=config.ridge_penalty,
            seed=config.seed,
            pert_embedding_path=pert_emb_path,
        )
        
        return result
    
    # Convert to numpy
    Y_train_np = Y_train.values
    Y_test_np = Y_test.values
    
    train_pert_names = Y_train.columns.tolist()
    test_pert_names = Y_test.columns.tolist()
    
    # For cross-dataset embeddings, we need to align to common genes
    # Get common genes if using cross-dataset embeddings
    common_genes = None
    if config.pert_embedding_source in ["k562_pca", "rpe1_pca"]:
        # Load source data to get gene names
        source_adata_path = config.pert_embedding_args.get("source_adata_path")
        if source_adata_path:
            import anndata as ad
            source_adata = ad.read_h5ad(source_adata_path)
        else:
            raise ValueError("source_adata_path required for cross-dataset baseline")
        
        source_gene_names = source_adata.var_names.tolist()
        common_genes = sorted(set(gene_names) & set(source_gene_names))
        if len(common_genes) == 0:
            raise ValueError("No common genes between source and target datasets")
        
        # Align Y and gene_names to common genes
        common_gene_idx = [gene_names.index(g) for g in common_genes]
        Y_train_np = Y_train_np[common_gene_idx, :]
        Y_test_np = Y_test_np[common_gene_idx, :] if not Y_test.empty else Y_test_np
        gene_names = common_genes
    
    # Construct A (gene embeddings)
    # For embeddings that use gene symbols, use gene_name_mapping if available
    embedding_args_with_mapping = config.gene_embedding_args.copy() if config.gene_embedding_args else {}
    if gene_name_mapping and config.gene_embedding_source in ["scgpt", "scfoundation"]:
        embedding_args_with_mapping["gene_name_mapping"] = gene_name_mapping
    
    A, gene_labels = construct_gene_embeddings(
        source=config.gene_embedding_source,
        train_data=Y_train_np,
        gene_names=gene_names,
        pca_dim=config.pca_dim,
        seed=config.seed,
        embedding_args=embedding_args_with_mapping,
    )
    
    # Construct B (perturbation embeddings) - for training and test perturbations
    # Pass gene_names for cross-dataset alignment
    pert_embedding_args = config.pert_embedding_args.copy() if config.pert_embedding_args else {}
    if config.pert_embedding_source in ["k562_pca", "rpe1_pca"]:
        pert_embedding_args["target_gene_names"] = gene_names
    
    B_train, pert_labels_train, pert_pca, B_test, pert_labels_test = construct_pert_embeddings(
        source=config.pert_embedding_source,
        train_data=Y_train_np,
        pert_names=train_pert_names,
        pca_dim=config.pca_dim,
        seed=config.seed,
        embedding_args=pert_embedding_args,
        test_data=Y_test_np if not Y_test.empty else None,
        test_pert_names=test_pert_names if not Y_test.empty else None,
    )
    
    # Solve for K
    center = Y_train_np.mean(axis=1, keepdims=True)
    Y_centered = Y_train_np - center
    
    solution = solve_y_axb(
        Y=Y_centered,
        A=A,
        B=B_train,
        ridge_penalty=config.ridge_penalty,
    )
    K = solution["K"]
    
    # Make predictions on test set
    if Y_test.empty or B_test is None:
        # No test data or test embeddings available
        Y_pred_test = np.array([]).reshape(Y_train_np.shape[0], 0)
        metrics = {}
    else:
        # Compute predictions: Y_pred = A @ K @ B_test + center
        Y_pred_test = A @ K @ B_test + center
        
        # Compute baseline (mean expression in control) for adding back
        # Note: In the paper, predictions are typically: Y_pred = A @ K @ B + center
        # But we may need to add baseline back depending on how Y was computed
        # For now, Y_pred_test is the change from baseline, which matches Y_test
        
        # Compute metrics
        metrics = {}
        for i, pert_name in enumerate(test_pert_names):
            y_true = Y_test_np[:, i]
            y_pred = Y_pred_test[:, i]
            pert_metrics = compute_metrics(y_true, y_pred)
            metrics[pert_name] = pert_metrics
    
    return {
        "baseline_type": config.baseline_type.value,
        "predictions": Y_pred_test,
        "metrics": metrics,
        "K": K,
        "A": A,
        "B_train": B_train,
    }


def run_all_baselines(
    adata_path: Path,
    split_config_path: Path,
    output_dir: Path,
    baseline_types: Optional[List[BaselineType]] = None,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
    seed: int = 1,
    use_paper_implementation: bool = False,
) -> pd.DataFrame:
    """
    Run all baseline models and save results.
    
    Args:
        adata_path: Path to perturb_processed.h5ad
        split_config_path: Path to train/test/val split JSON
        output_dir: Directory to save results
        baseline_types: List of baseline types to run (None = all 8)
        pca_dim: PCA dimension
        ridge_penalty: Ridge penalty
        seed: Random seed
    
    Returns:
        DataFrame with results summary
    """
    LOGGER.info("Starting baseline reproduction")
    
    # Load data
    LOGGER.info(f"Loading data from {adata_path}")
    adata = ad.read_h5ad(adata_path)
    
    # Load splits
    LOGGER.info(f"Loading splits from {split_config_path}")
    split_config = load_split_config(split_config_path)
    
    # Compute Y matrix (pseudobulk expression changes)
    LOGGER.info("Computing pseudobulk expression changes")
    Y_df, split_labels = compute_pseudobulk_expression_changes(adata, split_config, seed)
    
    # Split Y into train/test/val
    train_perts = split_labels.get("train", [])
    test_perts = split_labels.get("test", [])
    val_perts = split_labels.get("val", [])
    
    Y_train = Y_df[train_perts]
    Y_test = Y_df[test_perts] if test_perts else pd.DataFrame()
    Y_val = Y_df[val_perts] if val_perts else pd.DataFrame()
    
    LOGGER.info(f"Train: {len(train_perts)} perturbations")
    LOGGER.info(f"Test: {len(test_perts)} perturbations")
    LOGGER.info(f"Val: {len(val_perts)} perturbations")
    
    # Default to all 8 baselines + mean_response
    if baseline_types is None:
        baseline_types = [
            BaselineType.SELFTRAINED,
            BaselineType.RANDOM_PERT_EMB,
            BaselineType.RANDOM_GENE_EMB,
            BaselineType.SCGPT_GENE_EMB,
            BaselineType.SCFOUNDATION_GENE_EMB,
            BaselineType.GEARS_PERT_EMB,
            BaselineType.K562_PERT_EMB,
            BaselineType.RPE1_PERT_EMB,
            BaselineType.MEAN_RESPONSE,
        ]
    
    # Run each baseline
    results = []
    gene_names = Y_df.index.tolist()  # These are var_names (Ensembl IDs for Adamson)
    
    # Get gene_name mapping if available (for alignment with embeddings that use gene symbols)
    gene_name_mapping = None
    if "gene_name" in adata.var.columns:
        gene_name_mapping = dict(zip(adata.var_names, adata.var["gene_name"]))
    
    for baseline_type in baseline_types:
        try:
            # Handle mean-response baseline separately
            if baseline_type == BaselineType.MEAN_RESPONSE:
                result = run_mean_response_baseline(
                    Y_train=Y_train,
                    Y_test=Y_test,
                )
            else:
                config = get_baseline_config(
                    baseline_type,
                    pca_dim=pca_dim,
                    ridge_penalty=ridge_penalty,
                    seed=seed,
                )
                
                result = run_single_baseline(
                    Y_train=Y_train,
                    Y_test=Y_test,
                    config=config,
                    gene_names=gene_names,
                    gene_name_mapping=gene_name_mapping,
                    use_paper_implementation=use_paper_implementation,
                    adata_path=adata_path,
                    split_config=split_config,
                )
            
            # Aggregate metrics
            all_metrics = result["metrics"]
            if all_metrics:
                mean_r = np.mean([m["pearson_r"] for m in all_metrics.values()])
                mean_l2 = np.mean([m["l2"] for m in all_metrics.values()])
            else:
                mean_r = np.nan
                mean_l2 = np.nan
            
            results.append({
                "baseline": baseline_type.value,
                "mean_pearson_r": mean_r,
                "mean_l2": mean_l2,
                "n_test_perturbations": len(test_perts),
            })
            
            # Save predictions for this baseline
            baseline_output_dir = output_dir / baseline_type.value
            baseline_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions as JSON (matching format for comparison)
            if "predictions" in result and result["predictions"].size > 0:
                predictions_dict = {}
                pred_matrix = result["predictions"]  # genes × test_perturbations
                test_pert_names = Y_test.columns.tolist() if not Y_test.empty else []
                
                for i, pert_name in enumerate(test_pert_names):
                    # Clean perturbation name (remove +ctrl)
                    clean_pert_name = pert_name.replace("+ctrl", "")
                    predictions_dict[clean_pert_name] = pred_matrix[:, i].tolist()
                
                import json
                predictions_path = baseline_output_dir / "predictions.json"
                with open(predictions_path, "w") as f:
                    json.dump(predictions_dict, f)
                
                # Save gene names
                gene_names_path = baseline_output_dir / "gene_names.json"
                with open(gene_names_path, "w") as f:
                    json.dump(gene_names, f)
                
                LOGGER.info(f"Saved predictions for {baseline_type.value} to {baseline_output_dir}")
            
        except Exception as e:
            LOGGER.error(f"Failed to run {baseline_type.value}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "baseline_results_reproduced.csv"
    results_df.to_csv(results_path, index=False)
    LOGGER.info(f"Saved results to {results_path}")
    
    return results_df

