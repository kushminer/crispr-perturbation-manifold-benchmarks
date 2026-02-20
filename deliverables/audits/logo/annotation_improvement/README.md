# Annotation Improvement Framework

## Goal

Reduce "Other" class size by improving functional class annotations using comprehensive GO/Reactome mappings and validation strategies.

## Key Principle

**Never modify original annotation files** - always create backups and new improved versions.

## Validation Strategy

To ensure appropriate class annotation with certainty, we use multiple validation approaches:

### 1. GO Term Consistency
- Check if gene has GO annotations
- Verify GO terms are consistent with assigned class
- Flag conflicting GO terms

### 2. Expression Similarity
- Compute mean expression profile for each class
- Correlate each gene with class mean
- Flag genes with low correlation (potential misclassification)

### 3. Cross-Dataset Consistency
- For genes present in multiple datasets, check consistency
- Flag inconsistencies
- Suggest most common assignment

### 4. Statistical Validation
- Check class size distribution
- Verify gene coverage
- Assess class diversity

### 5. Literature/Domain Knowledge
- Manual curation for ambiguous cases
- Expert review of assignments
- Reference to published classifications

## Implementation Status

### Completed
- [x] Framework structure
- [x] Backup mechanism
- [x] Validation framework

### In Progress
- [ ] GO term queries (requires GO API/database)
- [ ] Expression similarity analysis
- [ ] Cross-dataset consistency check

### To Do
- [ ] Implement GO term to class mapping
- [ ] Generate improved annotations
- [ ] Validate improved annotations
- [ ] Re-run LOGO with improved annotations

## Files

- `analyze_other_class.py` - Analyze what's in "Other" class
- `validate_annotations.py` - Validation framework
- `improve_annotations.py` - Create improved annotations
- `improved_annotations/` - New annotation files (backups of originals)
- `validation/` - Validation reports

## Usage

```bash
# Analyze "Other" class
python analyze_other_class.py

# Validate annotations
python validate_annotations.py

# Improve annotations (framework - implement assignment logic)
python improve_annotations.py
```

## Next Steps

1. **Implement GO queries**: Use GO API or local database to query GO terms for "Other" genes
2. **Expression analysis**: Compute expression similarity to assign genes to classes
3. **Generate assignments**: Combine GO and expression data to create new assignments
4. **Validate**: Run validation framework on improved annotations
5. **Test**: Re-run LOGO with improved annotations to measure impact

