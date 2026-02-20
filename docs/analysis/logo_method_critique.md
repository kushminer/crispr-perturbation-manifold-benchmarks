# LOGO Method Critique: Impact of 'Other' Class Distribution

## Executive Summary

**Critical Issue Identified**: The current LOGO implementation includes ALL non-holdout classes in training, including a potentially large and heterogeneous "Other" class. This may be **inflating performance** on Transcription prediction by:

1. **Data leakage risk**: "Other" may contain transcription-related genes that are misclassified
2. **Distribution bias**: "Other" may dominate the training set, skewing learned embeddings
3. **Unrealistic evaluation**: Real-world scenarios wouldn't have such a large catch-all class

## Current LOGO Implementation

### Split Logic (from `src/goal_3_prediction/functional_class_holdout/logo.py`)

```python
# Lines 122-126
# Train = all non-holdout class perturbations
# Test = holdout class perturbations only
holdout_targets_available = [t for t in holdout_targets if t in available_targets]
train_targets = [t for t in Y_df.columns if t not in holdout_targets_available]
```

**Key Issue**: `train_targets` includes **ALL** perturbations not in the holdout class, with no filtering of "Other" or other potentially problematic classes.

## Potential Problems

### 1. **"Other" Class May Contain Transcription-Related Genes**

**Hypothesis**: The "Other" class is a catch-all for unannotated or poorly annotated genes. Some of these may actually be transcription-related but were:
- Not captured by GO/Reactome annotations
- Misclassified due to incomplete pathway knowledge
- Functionally related but not formally categorized

**Impact**: If "Other" contains transcription-related genes, the model is effectively "cheating" by seeing similar biology during training.

**Evidence Needed**:
- Check if "Other" genes have transcription-related GO terms
- Analyze expression patterns: Do "Other" genes correlate with Transcription genes?
- Functional similarity: Are "Other" genes co-expressed with Transcription genes?

### 2. **"Other" May Dominate Training Set**

**Hypothesis**: If "Other" is the largest class, it may:
- Dominate the PCA space (for self-trained embeddings)
- Skew the learned K matrix toward "Other" patterns
- Create a bias that helps predict Transcription if "Other" and Transcription share common expression patterns

**Example Scenario**:
- Transcription (test): 5 perturbations
- Translation (train): 10 perturbations  
- Metabolic (train): 8 perturbations
- **Other (train): 30 perturbations** ← Dominates training

**Impact**: The model learns primarily from "Other", which may be a poor proxy for biological structure.

### 3. **Heterogeneity of "Other" Class**

**Hypothesis**: "Other" is likely highly heterogeneous, containing:
- Genes from multiple biological processes
- Genes with no clear functional annotation
- Genes that should belong to other classes but were missed

**Impact**: 
- Heterogeneous training data may create a "mean" embedding that accidentally helps with Transcription
- The model may learn general patterns that happen to work for Transcription without true biological understanding

### 4. **Unrealistic Evaluation Scenario**

**Hypothesis**: In real-world applications, you wouldn't have such a large "Other" class. The evaluation is testing:
- "Can we predict Transcription using Translation + Metabolic + Other?"
- But it should test: "Can we predict Transcription using only well-defined, non-Transcription classes?"

**Impact**: Results may be **overly optimistic** compared to realistic deployment scenarios.

## Recommended Analyses

### Analysis 1: Class Distribution Check

```python
# Check class sizes
class_counts = annotations["class"].value_counts()
print("Class distribution:")
print(class_counts)
print(f"\nOther as % of non-Transcription: {class_counts['Other'] / (len(annotations) - class_counts['Transcription']) * 100:.1f}%")
```

**Expected Output**: If "Other" > 30% of training data, this is a concern.

### Analysis 2: Expression Similarity Analysis

**Question**: Do "Other" genes have similar expression patterns to Transcription genes?

```python
# Compute correlation between "Other" and "Transcription" perturbations
other_perts = annotations[annotations["class"] == "Other"]["target"].tolist()
transcription_perts = annotations[annotations["class"] == "Transcription"]["target"].tolist()

# Compute mean expression correlation
other_expr = Y_df[other_perts].mean(axis=1)
transcription_expr = Y_df[transcription_perts].mean(axis=1)
correlation = np.corrcoef(other_expr, transcription_expr)[0, 1]
```

**Interpretation**: 
- High correlation (>0.5) suggests "Other" may contain transcription-related genes
- Low correlation suggests "Other" is truly different

### Analysis 3: Ablation Study: Remove "Other" from Training

**Critical Test**: Re-run LOGO excluding "Other" from training:

```python
# Modified split logic
train_targets = [
    t for t in Y_df.columns 
    if t not in holdout_targets_available 
    and annotations[annotations["target"] == t]["class"].values[0] != "Other"
]
```

**Expected Outcome**:
- If performance **drops significantly** (e.g., r drops from 0.88 to 0.70), "Other" was helping
- If performance **stays similar** (e.g., r stays at 0.85-0.88), "Other" wasn't critical

### Analysis 4: Functional Annotation Check

**Question**: Do "Other" genes have any transcription-related GO terms?

```python
# Check GO term overlap
from goatools import GOEnrichmentStudy
# Or use gseapy to check if "Other" genes are enriched for transcription terms
```

**Interpretation**: If "Other" genes are enriched for transcription GO terms, this confirms leakage.

## Recommended Fixes

### Option 1: Exclude "Other" from Training (Conservative)

**Rationale**: "Other" is too heterogeneous and may contain misclassified genes.

**Implementation**:
```python
# In logo.py, line 126
train_targets = [
    t for t in Y_df.columns 
    if t not in holdout_targets_available 
    and annotations[annotations["target"] == t]["class"].values[0] != "Other"
]
```

**Pros**: 
- More conservative evaluation
- Tests true extrapolation to well-defined classes
- Removes potential leakage

**Cons**:
- May reduce training set size significantly
- May make evaluation too hard (unrealistic)

### Option 2: Stratified Sampling (Balanced)

**Rationale**: Balance training classes to prevent "Other" from dominating.

**Implementation**:
```python
# Sample equal numbers from each class (excluding holdout)
max_per_class = min([len(annotations[annotations["class"] == c]) 
                     for c in annotations["class"].unique() 
                     if c != class_name])
```

**Pros**:
- Prevents class imbalance
- More realistic (each class contributes equally)

**Cons**:
- Throws away data
- May not reflect real-world distribution

### Option 3: Multi-Class Holdout (Comprehensive)

**Rationale**: Hold out multiple classes, not just Transcription.

**Implementation**: Hold out Transcription + Other, train on well-defined classes only.

**Pros**:
- Tests extrapolation to both specific and general classes
- More comprehensive evaluation

**Cons**:
- More complex
- May reduce test set too much

### Option 4: Report Both (Transparent)

**Rationale**: Report results with and without "Other" for transparency.

**Implementation**: Run LOGO twice:
1. Standard (with "Other")
2. Conservative (without "Other")

**Pros**:
- Full transparency
- Readers can judge for themselves
- Shows robustness of findings

**Cons**:
- More computation
- More complex reporting

## Recommended Next Steps

1. **Immediate**: Run Analysis 1 (class distribution) to quantify the problem
2. **Short-term**: Run Analysis 3 (ablation study) to measure impact
3. **Medium-term**: Implement Option 4 (report both) for transparency
4. **Long-term**: Improve annotations to reduce "Other" class size

## Conclusion

The current LOGO implementation **may be inflating performance** on Transcription prediction by including a potentially large and heterogeneous "Other" class in training. This is a **methodological concern** that should be:

1. **Quantified**: Measure how much "Other" affects results
2. **Addressed**: Either exclude "Other" or report both versions
3. **Documented**: Clearly state in methods what classes are included in training

**Recommendation**: Run the ablation study (Analysis 3) to determine if this is a real problem, then implement Option 4 (report both) for maximum transparency.

