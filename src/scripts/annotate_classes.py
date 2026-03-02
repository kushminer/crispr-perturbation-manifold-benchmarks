#!/usr/bin/env python3
"""
Generate functional class annotations from gene lists using GO terms, Reactome pathways,
or manual curation mappings.

This script addresses the need to enrich sparse annotations (e.g., all genes in "general")
with biologically meaningful functional classes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("annotate_classes")


# Manual curation mapping for Adamson dataset (ER stress/UPR focused)
ADAMSON_MANUAL_MAPPING = {
    # ERAD (ER-associated degradation)
    "ERAD": [
        "DERL2", "SEL1L", "SYVN1", "UFL1", "UFM1", "PSMD4", "NEDD8",
    ],
    # UPR (Unfolded protein response)
    "UPR": [
        "HSPA5", "HSPA9", "HYOU1", "DDIT3", "EIF2S1", "EIF2B2", "EIF2B3", "EIF2B4",
        "IER3IP1", "MANF",
    ],
    # Protein folding and chaperones
    "Chaperone": [
        "DNAJC19", "P4HB", "PDIA6",
    ],
    # ER transport and translocation
    "ER_Transport": [
        "SEC61A1", "SEC61B", "SEC61G", "SEC63", "SRP68", "SRP72", "SRPRB",
        "SPCS2", "SPCS3", "OST4", "STT3A", "TMED10", "TMED2", "TMEM167A",
    ],
    # Translation
    "Translation": [
        "AARS", "CARS", "DARS", "FARSB", "HARS", "IARS2", "MARS", "QARS", "SARS", "TARS",
        "EIF2B2", "EIF2B3", "EIF2B4", "EIF2S1",
    ],
    # ER/Golgi transport
    "ER_Golgi_Transport": [
        "COPB1", "COPZ1", "GBF1", "SCYL1", "YIPF5",
    ],
    # Other ER-related
    "ER_Other": [
        "DDOST", "DAD1", "DDRGK1", "DHDDS", "GNPNAT1", "OST4", "SLC35B1", "SLC39A7",
        "PTDSS1", "SAMM50", "TIMM44",
    ],
    # Metabolic/Other
    "Metabolic": [
        "ATP5B", "CAD", "FECH", "GMPPB", "HSD17B12", "IDH3A", "MTHFD1",
    ],
    # Transcription/Regulation
    "Transcription": [
        "BHLHE40", "CREB1", "CHERP", "SOCS1", "ZNF326",
    ],
    # Other
    "Other": [
        "ARHGAP22", "ASCC3", "CCND3", "MRGBP", "MRPL39", "PPWD1", "TELO2",
        "TTI1", "TTI2", "UFL1", "XRN1", "ctrl",
    ],
}


def load_gene_list(input_path: Path) -> List[str]:
    """Load gene list from file (one per line or TSV with gene column)."""
    input_path = Path(input_path)
    
    if input_path.suffix == ".tsv" or input_path.suffix == ".csv":
        df = pd.read_csv(input_path, sep="\t" if input_path.suffix == ".tsv" else ",")
        # Try common column names
        for col in ["gene", "target", "gene_name", "symbol", "Gene"]:
            if col in df.columns:
                return df[col].astype(str).tolist()
        # If no match, use first column
        return df.iloc[:, 0].astype(str).tolist()
    else:
        # Plain text, one per line
        with open(input_path) as f:
            return [line.strip() for line in f if line.strip()]


def map_genes_to_classes_manual(
    genes: List[str],
    mapping: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Map genes to functional classes using manual curation.
    
    Args:
        genes: List of gene symbols
        mapping: Dictionary mapping class names to lists of genes
    
    Returns:
        DataFrame with 'target' and 'class' columns
    """
    # Create reverse mapping: gene -> class
    gene_to_class = {}
    for class_name, class_genes in mapping.items():
        for gene in class_genes:
            gene_upper = gene.upper()
            gene_to_class[gene_upper] = class_name
            # Also map original case
            gene_to_class[gene] = class_name
    
    # Map genes
    results = []
    unmapped = []
    for gene in genes:
        gene_upper = gene.upper()
        if gene_upper in gene_to_class or gene in gene_to_class:
            class_name = gene_to_class.get(gene_upper) or gene_to_class.get(gene)
            results.append({"target": gene, "class": class_name})
        else:
            unmapped.append(gene)
            # Default to "Other" for unmapped
            results.append({"target": gene, "class": "Other"})
    
    if unmapped:
        LOGGER.warning("Unmapped genes (assigned to 'Other'): %d", len(unmapped))
        LOGGER.debug("Unmapped: %s", unmapped[:10])
    
    df = pd.DataFrame(results)
    class_counts = df["class"].value_counts()
    LOGGER.info("Class distribution:")
    for cls, count in class_counts.items():
        LOGGER.info("  %s: %d genes", cls, count)
    
    return df


def map_genes_to_classes_go(
    genes: List[str],
    method: str = "gseapy",
    organism: str = "human",
    min_class_size: int = 3,
) -> pd.DataFrame:
    """
    Map genes to functional classes using GO terms.
    
    Uses gseapy to perform GO enrichment and groups genes by enriched GO terms.
    
    Args:
        genes: List of gene symbols
        method: Method to use (currently only "gseapy" supported)
        organism: Organism name for GO database (default: "human")
        min_class_size: Minimum genes per class (default: 3)
    
    Returns:
        DataFrame with 'target' and 'class' columns
    """
    try:
        import mygene
    except ImportError:
        LOGGER.error("mygene package required for GO mapping. Install with: pip install mygene")
        raise ImportError("mygene package required. Install with: pip install mygene")
    
    LOGGER.info("Mapping %d genes to GO terms using mygene.info API...", len(genes))
    
    try:
        mg = mygene.MyGeneInfo()
        LOGGER.info("Querying mygene.info for GO annotations...")
        
        # Query genes in batches (mygene handles batching internally, but we'll do it explicitly for progress)
        batch_size = 1000
        all_results = []
        
        for batch_start in range(0, len(genes), batch_size):
            batch_end = min(batch_start + batch_size, len(genes))
            batch_genes = genes[batch_start:batch_end]
            
            if batch_start > 0:
                LOGGER.info("Processing batch %d-%d of %d genes...", batch_start + 1, batch_end, len(genes))
            
            # Query mygene for GO terms
            gene_info = mg.querymany(
                batch_genes,
                scopes="symbol",
                fields="go",
                species="human",
                returnall=True,
            )
            
            all_results.extend(gene_info["out"])
        
        # Map genes to GO terms
        gene_to_go = {}
        for result in all_results:
            if "query" in result:
                gene = result["query"]
                go_terms = result.get("go", {})
                
                # Extract GO Biological Process terms
                bp_terms = go_terms.get("BP", [])
                if bp_terms:
                    gene_to_go[gene] = bp_terms
        
        LOGGER.info("Found GO terms for %d/%d genes", len(gene_to_go), len(genes))
        
        # Define high-level biological process categories for grouping
        # These keywords will help group related GO terms together
        category_keywords = {
            "Transcription": ["transcription", "rna", "rna polymerase", "transcriptional", "gene expression"],
            "Translation": ["translation", "ribosome", "protein synthesis", "translational"],
            "Cell_Cycle": ["cell cycle", "mitosis", "meiosis", "cell division", "dna replication"],
            "DNA_Repair": ["dna repair", "dna damage", "dna recombination", "dna replication"],
            "Metabolism": ["metabolic", "metabolism", "biosynthesis", "catabolic", "glycolysis", "oxidation"],
            "Protein_Folding": ["protein folding", "chaperone", "unfolded protein", "protein refolding"],
            "Protein_Degradation": ["protein degradation", "ubiquitin", "proteasome", "protein catabolic"],
            "Signal_Transduction": ["signal transduction", "signaling pathway", "signal", "receptor"],
            "Transport": ["transport", "vesicle", "endocytosis", "exocytosis", "secretion"],
            "Immune_Response": ["immune", "immune response", "inflammatory", "defense response"],
            "Apoptosis": ["apoptosis", "programmed cell death", "cell death"],
            "Development": ["development", "differentiation", "morphogenesis", "organogenesis"],
            "Stress_Response": ["stress response", "heat shock", "oxidative stress", "er stress"],
            "Chromatin": ["chromatin", "histone", "chromosome", "nucleosome"],
            "Other": []  # Catch-all for terms that don't match
        }
        
        # Group genes by GO term categories using keyword matching
        go_to_genes = {}
        gene_to_best_category = {}  # Track best category for each gene
        
        for gene, go_list in gene_to_go.items():
            gene_categories = {}  # Count matches per category
            
            for go_term in go_list:
                # Handle both dict and string formats
                if isinstance(go_term, dict):
                    # mygene returns GO terms with "term" key, not "name"
                    go_name = go_term.get("term", go_term.get("name", "")).lower()
                elif isinstance(go_term, str):
                    go_name = go_term.lower()
                else:
                    continue
                
                if go_name:
                    # Match GO term to category using keywords
                    matched_category = None
                    max_matches = 0
                    
                    for category, keywords in category_keywords.items():
                        if category == "Other":
                            continue
                        # Check if any keyword appears in the GO term name
                        matches = sum(1 for keyword in keywords if keyword in go_name)
                        if matches > 0 and matches > max_matches:
                            max_matches = matches
                            matched_category = category
                    
                    # If no match, use "Other"
                    if matched_category is None:
                        matched_category = "Other"
                    
                    # Count this match for the gene
                    gene_categories[matched_category] = gene_categories.get(matched_category, 0) + 1
            
            # Assign gene to its most frequent category
            if gene_categories:
                best_category = max(gene_categories.items(), key=lambda x: x[1])[0]
                gene_to_best_category[gene] = best_category
                go_to_genes.setdefault(best_category, []).append(gene)
            else:
                # No GO terms matched, assign to Other
                gene_to_best_category[gene] = "Other"
                go_to_genes.setdefault("Other", []).append(gene)
        
        # Create results DataFrame
        # First, assign all genes with GO terms to their categories
        results = []
        for gene in genes:
            if gene in gene_to_best_category:
                category = gene_to_best_category[gene]
                results.append({"target": gene, "class": category})
            else:
                # Gene had no GO terms, assign to Other
                results.append({"target": gene, "class": "Other"})
        
        # Filter classes by minimum size
        class_counts = {}
        for r in results:
            class_counts[r["class"]] = class_counts.get(r["class"], 0) + 1
        
        # Reassign genes from small classes to "Other"
        filtered_results = []
        for r in results:
            if class_counts[r["class"]] >= min_class_size:
                filtered_results.append(r)
            else:
                filtered_results.append({"target": r["target"], "class": "Other"})
        
        results = filtered_results
        
        df = pd.DataFrame(results)
        
        LOGGER.info("Mapped %d genes to %d GO-based classes", len(genes), df["class"].nunique())
        
        # Log class distribution
        class_counts = df["class"].value_counts()
        LOGGER.info("Class distribution:")
        for cls, count in class_counts.head(20).items():  # Show top 20 classes
            LOGGER.info("  %s: %d genes", cls, count)
        if len(class_counts) > 20:
            LOGGER.info("  ... and %d more classes", len(class_counts) - 20)
        
        return df
    
    except Exception as e:
        LOGGER.error("Error in GO mapping: %s", e)
        LOGGER.error("Falling back to manual mapping. Check mygene installation: pip install mygene")
        raise


def map_genes_to_classes_reactome(
    genes: List[str],
    min_class_size: int = 3,
) -> pd.DataFrame:
    """
    Map genes to functional classes using Reactome pathways.
    
    Uses Reactome REST API to query pathway annotations for genes.
    
    Args:
        genes: List of gene symbols
        min_class_size: Minimum genes per class (default: 3)
    
    Returns:
        DataFrame with 'target' and 'class' columns
    """
    import requests
    
    LOGGER.info("Mapping %d genes to Reactome pathways...", len(genes))
    
    # Reactome REST API endpoint
    BASE_URL = "https://reactome.org/ContentService"
    
    # Map genes to pathways
    gene_to_pathways = {}
    pathway_to_genes = {}
    
    LOGGER.info("Querying Reactome API for pathway annotations...")
    LOGGER.info("Note: This may take several minutes for large gene lists...")
    
    # Query genes individually using Reactome's pathway participant endpoint
    # This is the most reliable method
    for i, gene in enumerate(genes):
        if (i + 1) % 100 == 0:
            LOGGER.info("Processed %d/%d genes...", i + 1, len(genes))
        
        try:
            # Reactome API: query pathways containing this gene
            # First, try to find the gene in Reactome
            # Use the query/ids endpoint to map gene symbol to Reactome ID
            query_url = f"{BASE_URL}/query/ids"
            
            # Try querying by gene symbol
            response = requests.post(
                query_url,
                json={"ids": [gene]},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Get pathways for this gene using the mapped ID
                    # Actually, let's use the simpler approach: query by gene symbol directly
                    pathway_url = f"{BASE_URL}/query/pathways/participant/{gene}"
                    pathway_response = requests.get(pathway_url, timeout=10)
                    
                    if pathway_response.status_code == 200:
                        pathways = pathway_response.json()
                        if isinstance(pathways, list) and len(pathways) > 0:
                            gene_pathways = []
                            for pathway in pathways:
                                pathway_name = pathway.get("displayName", "")
                                if pathway_name:
                                    # Extract meaningful category from pathway name
                                    words = pathway_name.split()
                                    if len(words) >= 2:
                                        # Use first 2 words as class name
                                        class_name = "_".join(words[:2]).replace(",", "").replace("(", "").replace(")", "").replace("/", "_")
                                        gene_pathways.append(class_name)
                                        pathway_to_genes.setdefault(class_name, []).append(gene)
                            
                            if gene_pathways:
                                # Use the first (most significant) pathway
                                gene_to_pathways[gene] = gene_pathways
                            else:
                                gene_to_pathways[gene] = []
                        else:
                            gene_to_pathways[gene] = []
                    else:
                        gene_to_pathways[gene] = []
                else:
                    gene_to_pathways[gene] = []
            else:
                gene_to_pathways[gene] = []
        
        except Exception as e:
            LOGGER.debug("Error querying Reactome for %s: %s", gene, e)
            gene_to_pathways[gene] = []
    
    # Ensure all genes are in the mapping
    for gene in genes:
        if gene not in gene_to_pathways:
            gene_to_pathways[gene] = []
    
    # Create results DataFrame
    results = []
    
    # Assign each gene to its first (most significant) pathway class
    for gene, pathways in gene_to_pathways.items():
        if pathways:
            # Use the first pathway class
            results.append({"target": gene, "class": pathways[0]})
        else:
            results.append({"target": gene, "class": "Other"})
    
    # Filter classes by minimum size
    class_counts = {}
    for r in results:
        class_counts[r["class"]] = class_counts.get(r["class"], 0) + 1
    
    # Reassign genes from small classes to "Other"
    filtered_results = []
    for r in results:
        if class_counts[r["class"]] >= min_class_size:
            filtered_results.append(r)
        else:
            filtered_results.append({"target": r["target"], "class": "Other"})
    
    df = pd.DataFrame(filtered_results)
    
    LOGGER.info("Mapped %d genes to %d Reactome-based classes", len(genes), df["class"].nunique())
    
    # Log class distribution
    class_counts = df["class"].value_counts()
    LOGGER.info("Class distribution:")
    for cls, count in class_counts.items():
        LOGGER.info("  %s: %d genes", cls, count)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate functional class annotations from gene lists"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file: gene list (one per line) or TSV with gene column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TSV file path",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["manual", "go", "reactome"],
        default="manual",
        help="Annotation method (default: manual)",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Custom mapping file (JSON or TSV). If not provided, uses built-in Adamson mapping.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adamson",
        help="Dataset name (for built-in mappings)",
    )
    
    args = parser.parse_args()
    
    # Load gene list
    LOGGER.info("Loading gene list from %s", args.input)
    genes = load_gene_list(args.input)
    LOGGER.info("Loaded %d genes", len(genes))
    
    # Map to classes
    if args.method == "manual":
        if args.mapping:
            # Load custom mapping
            mapping_path = Path(args.mapping)
            if mapping_path.suffix == ".json":
                import json
                with open(mapping_path) as f:
                    mapping = json.load(f)
            else:
                # TSV format: class, gene
                df_map = pd.read_csv(mapping_path, sep="\t")
                mapping = {}
                for _, row in df_map.iterrows():
                    cls = row["class"]
                    gene = row["target"] if "target" in row else row["gene"]
                    mapping.setdefault(cls, []).append(gene)
        elif args.dataset == "adamson":
            mapping = ADAMSON_MANUAL_MAPPING
        else:
            LOGGER.error("No built-in mapping for dataset '%s'. Provide --mapping file.", args.dataset)
            return 1
        
        df = map_genes_to_classes_manual(genes, mapping)
    
    elif args.method == "go":
        df = map_genes_to_classes_go(genes, organism="human", min_class_size=3)
    
    elif args.method == "reactome":
        df = map_genes_to_classes_reactome(genes, min_class_size=3)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    LOGGER.info("Saved annotations to %s", output_path)
    LOGGER.info("Total: %d genes, %d classes", len(df), df["class"].nunique())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

