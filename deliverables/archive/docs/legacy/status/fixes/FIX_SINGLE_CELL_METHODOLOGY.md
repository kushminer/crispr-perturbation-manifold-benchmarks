# Fix for Single Cell Prediction Methodology - Self Train vs GEARS Identical Outputs

## Problem Identified

Self train (SELFTRAINED) and GEARS (GEARS_PERT_EMB) were producing **identical outputs** across all datasets (Adamson, K562, RPE1) in:
- Baseline predictions
- LSFT results  
- LOGO results

This indicated a critical bug where GEARS embeddings were not being used correctly.

## Root Cause Analysis

The issue was likely caused by:
1. **Silent fallback**: If GEARS CSV file didn't exist or failed to load, the code might have silently fallen back to training_data embeddings
2. **Insufficient error handling**: Errors during GEARS loading might have been caught and ignored
3. **Lack of validation**: No checks to ensure different baselines produce different embeddings

## Fixes Implemented

### 1. Enhanced Diagnostic Logging (`baseline_runner_single_cell.py`)
- Added comprehensive logging throughout embedding construction:
  - Which code path is executed for each baseline
  - Pseudobulk computation steps
  - GEARS CSV file existence and path resolution
  - Embedding construction results and statistics
  - Comparison with training_data embeddings

### 2. Validation Checks (`baseline_runner_single_cell.py`)
- Added validation that compares embeddings against training_data for non-training_data baselines
- **Raises error** if embeddings are identical (max_diff < 1e-6)
- **Warns** if embeddings are very similar (max_diff < 1e-3)
- Prevents silent fallbacks to training_data

### 3. Improved GEARS Embedding Loader (`baseline_runner.py`)
- Enhanced error handling with detailed logging:
  - Validates CSV file exists before loading
  - Logs path resolution steps
  - Reports statistics on loaded embeddings
  - Warns when few perturbations have GEARS embeddings
  - Provides clear error messages if loading fails

### 4. Path Resolution Fix (`baseline_runner_single_cell.py`)
- Fixed path resolution to match `baseline_runner.py` logic
- Ensures consistent path resolution across both files
- Prevents false negatives when checking file existence

### 5. Validation Script (`scripts/validate_single_cell_baselines.py`)
- Created standalone script to validate baseline differences
- Compares embeddings and predictions between two baselines
- Can be run independently to verify fixes

## Files Modified

1. `src/goal_2_baselines/baseline_runner_single_cell.py`
   - Added diagnostic logging
   - Added embedding validation
   - Fixed path resolution

2. `src/goal_2_baselines/baseline_runner.py`
   - Enhanced GEARS loader with better error handling
   - Added comprehensive logging

3. `scripts/validate_single_cell_baselines.py` (new)
   - Validation script for testing baseline differences

## How to Verify the Fix

1. **Run baselines with logging enabled**:
   ```bash
   python3 -m goal_2_baselines.baseline_runner_single_cell \
     --adata_path <path> \
     --split_config_path <path> \
     --output_dir <output>
   ```

2. **Check logs for**:
   - GEARS CSV file existence confirmation
   - Embedding construction paths
   - Validation messages showing embeddings differ
   - Any warnings about similar embeddings

3. **Use validation script**:
   ```bash
   python3 scripts/validate_single_cell_baselines.py \
     --adata_path <path> \
     --split_config_path <path> \
     --baseline_a lpm_selftrained \
     --baseline_b lpm_gearsPertEmb
   ```

## Expected Behavior After Fix

- **If GEARS CSV doesn't exist**: Code will raise `FileNotFoundError` with clear message
- **If GEARS embeddings are identical to training_data**: Code will raise `ValueError` with detailed comparison
- **If GEARS embeddings are very similar**: Code will log warning but continue
- **If everything works correctly**: Logs will show different embedding statistics and validation will pass

## Prevention

The validation check will **automatically catch** this issue in the future:
- Runs for all non-training_data baselines
- Compares against training_data embeddings
- Fails loudly if embeddings are identical
- Provides detailed diagnostics for debugging

## Next Steps

1. Re-run single-cell baselines to verify GEARS and SELFTRAINED now produce different outputs
2. Check logs to ensure GEARS embeddings are being loaded correctly
3. Verify that metrics are no longer identical between the two baselines
4. If issues persist, use the diagnostic logs to identify the specific failure point

