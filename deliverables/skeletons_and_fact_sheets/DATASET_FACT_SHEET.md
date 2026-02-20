
# Raw Dataset Fact Sheet (Corrected)

**Generated:** 2025-11-21
**Status:** Updated for Manifold Law Poster
**Datasets:** Adamson (2016) and Replogle (2022)

-----

## 1\. Adamson Dataset (The "High-Signal" Baseline)

### Publication

  - **Authors:** Adamson, B., et al.
  - **Title:** "A Multiplexed Single-Cell CRISPR Screening Platform Enables Systematic Dissection of the Unfolded Protein Response"
  - **Journal:** *Cell*, 167(7), 1867-1882
  - **Year:** 2016
  - **DOI:** 10.1016/j.cell.2016.11.048

### Dataset Characteristics

  - **Cell Line:** K562 (Chronic Myelogenous Leukemia)
  - **Perturbation Method:** CRISPR interference (CRISPRi)
  - **Biological Focus:** **Unfolded Protein Response (UPR)**
  - **"Hardness" Rating:** **Low.** The UPR is a massive, coordinated stress response. It generates a high signal-to-noise ratio, making it easier for linear models (PCA) to capture global structure without deep learning.

### Data Dimensions

  - **Total Conditions:** 87
      - Train: 61 | Test: 12 | Val: 14
  - **Cells:** \~68,603 total
  - **Test Set:** Small ($N=12$), leading to wider confidence intervals in bootstrapping.

-----

## 2\. Replogle Dataset (The "Genome-Scale" Challenge)

### Publication **(CORRECTED)**

  - **Authors:** Replogle, J.M., et al.
  - **Title:** "Mapping Information-Rich Genotype-Phenotype Landscapes with Genome-Scale Perturb-seq"
  - **Journal:** *Cell*, 185(14), 2559-2575
  - **Year:** **2022** (Previously listed incorrectly as 2020)
  - **Note:** This is the dataset integrated into GEARS (often called "Replogle 2022" or "K562 Essential"). The 2020 paper described the *method*, but this paper produced the *data*.

### Replogle K562 Essential

  - **Cell Line:** K562 (Cancer/Leukemia - highly plastic)
  - **Gene Set:** Essential Genes (survival-related)
  - **Dimensions:** 1,093 Conditions (163 Test)
  - **Context:** Perturbing essential genes often triggers generic failure modes (e.g., cell cycle arrest), creating a "dense" perturbation manifold where neighbors share functional similarities.

### Replogle RPE1 Essential

  - **Cell Line:** RPE1 (Retinal Pigment Epithelium - diploid/non-cancer)
  - **Gene Set:** Essential Genes
  - **Dimensions:** 1,544 Conditions (231 Test)
  - **Context:** RPE1 cells have stricter regulation than K562. This dataset is often considered "harder" or "cleaner" because the cells are not transformed, meaning the biological signal is more subtle and less dominated by cancer dysregulation.

-----

## 3\. Dataset Comparison Table

| Characteristic | Adamson | Replogle K562 | Replogle RPE1 |
| :--- | :--- | :--- | :--- |
| **Year** | 2016 | **2022** | **2022** |
| **Cell Line** | K562 (Cancer) | K562 (Cancer) | RPE1 (Normal) |
| **Perturbation** | CRISPRi (Knockdown) | CRISPR-Cas9 (KO) | CRISPR-Cas9 (KO) |
| **Test Size** | 12 (Small) | 163 (Large) | 231 (Largest) |
| **Biological Focus** | **UPR Pathway** | **Essential Genes** | **Essential Genes** |
| **Signal Strength** | **High (Strong Pathway)** | **Mixed (Viability)** | **Subtle (Diploid)** |
| **Manifold Law Role**| **Proof of Mechanism** | **Statistical Power** | **Generalization** |

-----

## 4\. Data Processing & Model Implications

### Pseudobulking

  - **Method:** `Y = mean(perturbation_cells) - mean(control_cells)`
  - **Critical Implication:** This step removes single-cell stochasticity (noise).
  - **Effect on Results:** By averaging hundreds of cells into one vector, the problem becomes **linearized**. This explains why **PCA (linear)** performs so competitively against **scGPT (non-linear transformer)**. The complexity that transformers excel at (sparse, noisy single-cell counts) is removed during preprocessing.

### Functional Annotations (for LOGO)

  - **Adamson:** 10 manual classes (e.g., "Chaperones").
  - **Replogle:** GO-based classes.
  - **Holdout Strategy:**
      * **Transcription Factors:** Selected as the holdout class because they have distinct, hierarchical regulatory roles that are difficult to extrapolate if unseen during training.

-----

## 5\. Data Access (GEARS API)

The standard access method via the GEARS framework remains:

```python
from gears import PertData

# Set data path
data_path = "./paper/benchmark/data/gears_pert_data/"
pert_data = PertData(data_path)

# Download corrected datasets
pert_data.download_dataset("adamson")             # 2016 UPR
pert_data.download_dataset("replogle_k562_essential") # 2022 Genome-wide
pert_data.download_dataset("replogle_rpe1_essential") # 2022 Genome-wide
```