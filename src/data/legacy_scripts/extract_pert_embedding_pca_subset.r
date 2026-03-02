library(SingleCellExperiment)
library(tidyverse)
library(reticulate)
Sys.setenv("BASILISK_EXTERNAL_CONDA"="/g/easybuild/x86_64/Rocky/8/haswell/software/Miniforge3/24.1.2-0")

pa <- argparser::arg_parser("Run PCA on data")
pa <- argparser::add_argument(pa, "--dataset_name", type = "character", help = "The name of the dataset") 
pa <- argparser::add_argument(pa, "--pca_dim", type = "integer", default = 10, nargs = 1, help = "The number of PCA dimensions")

pa <- argparser::add_argument(pa, "--working_dir", type = "character", help = "The directory that contains the params, results, scripts etc.")
pa <- argparser::add_argument(pa, "--result_id", type = "character", help = "The result_id")
pa <- argparser::parse_args(pa)
# pa <- argparser::parse_args(pa, argv = r"(
#                             --dataset_name adamson
#                             --pca_dim 4
#                             --working_dir /scratch/ahlmanne/perturbation_prediction_benchmark
# )" |> stringr::str_trim() |> stringr::str_split("\\s+"))

print(pa)
print(getwd())
set.seed(pa$seed)

out_file <- file.path(pa$working_dir, "results/", pa$result_id)
# ---------------------------------------


# Configure reticulate to use system Python (avoid pyenv autoinstall)
Sys.setenv(
  RETICULATE_PYENV = "~/.pyenv",
  PYENV_ROOT = "~/.pyenv",
  PYENV_VERSION = "3.13.3"
)
reticulate::use_python("~/.pyenv/versions/3.13.3/bin/python3", required = TRUE)

# Load data
folder <- "data/gears_pert_data"
data_path <- normalizePath(file.path(folder, pa$dataset_name, "perturb_processed.h5ad"))
sce <- zellkonverter::readH5AD(data_path)

# Clean up the colData(sce) a bit
sce$condition <- droplevels(sce$condition)
sce$clean_condition <- stringr::str_remove(sce$condition, "\\+ctrl")

gene_names <- rowData(sce)[["gene_name"]]
rownames(sce) <- gene_names

baseline <- MatrixGenerics::rowMeans2(assay(sce, "X")[,sce$condition == "ctrl",drop=FALSE])

# Pseudobulk everything
psce <- glmGamPoi::pseudobulk(sce, group_by = vars(condition, clean_condition))
assay(psce, "change") <- assay(psce, "X") - baseline


pca <- irlba::prcomp_irlba(t(as.matrix(assay(psce, "X"))), n = pa$pca_dim)
pert_emb <- t(pca$x)
colnames(pert_emb) <- psce$clean_condition

# Store output
as.data.frame(pert_emb) |>
  write_tsv(out_file)


#### Session Info
sessionInfo()


