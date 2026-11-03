# Review Scripts

This directory contains scripts for reviewing and resolving disagreements between model predictions and intended labels during the synthetic data generation process.

## Overview

During synthetic data generation, the labeling model may disagree with the intended class (especially for hard negatives and edge cases). These scripts provide a systematic workflow to:

1. **Automatically resolve** high-confidence cases using taxonomy rules
2. **Manually review** uncertain cases through an interactive interface
3. **Merge** all reviewed labels into a final training dataset

## Scripts

### 1. `review_disagreements.py` - Interactive Review Tool

**Purpose:** Review model-human disagreements interactively.

**Usage:**
```bash
# Run from project root (scripts auto-navigate to correct directory)
cd /path/to/pii-column-classifier

# List available categories
python scripts/review/review_disagreements.py --list-categories

# Review a specific category (interactive mode)
python scripts/review/review_disagreements.py --category high_confidence_model_likely_right

# Batch review mode (show all records at once)
python scripts/review/review_disagreements.py --category clear_cut_cases --batch
```

**Note:** Scripts automatically change to the project root directory, so they work regardless of your current working directory.

**Categories:**
- `high_confidence_model_likely_right` - Model is probably correct (40 items)
- `high_confidence_model_likely_wrong` - Model is probably wrong (16 items)
- `ambiguous_needs_context` - Requires human judgment (29 items)
- `clear_cut_cases` - Obvious cases (112 items)

**Interactive Commands:**
- `1-10`: Select class by number
- `i`: Accept the intended class
- `m`: Accept the model's predicted class
- `?`: Show all class definitions from taxonomy
- `s`: Skip this record
- `q`: Quit and save progress

**Batch Mode Commands:**
- `all-intended`: Accept all intended labels
- `all-model`: Accept all model labels
- `interactive`: Switch to interactive review mode
- `q`: Quit

**Output:** Creates `data/reviewed_<category>.csv` with human-verified labels.

**Example Interactive Session:**
```
================================================================================
RECORD 1 of 40
================================================================================
Table:  customer_prefs
Column: dob_hint_txt
Type:   varchar(255)
Mode:   hard_negative

Intended class:  non_pii
Model labeled:   date_of_birth
Model confidence: 0.8663

Decision [non_pii]: i
✓ Kept intended: non_pii
```

---

### 2. `auto_review.py` - Automated Review Engine

**Purpose:** Automatically resolve disagreements using taxonomy rules and pattern matching.

**Features:**
- **Rule-based classification:** Uses domain knowledge to auto-classify edge cases
- **Pattern detection:** Identifies birth dates, session IDs, account numbers, etc.
- **Confidence scoring:** Flags uncertain cases for manual review
- **Segment vs. individual demographics:** Distinguishes aggregate metadata from PII

**Key Rules:**
- Birth date detection: `birth`, `dob`, `bday`, `birthday`, `born`
- Session identifiers: `session`, `sess` + UUID/ID patterns
- Account numbers: `acct` + `nbr`, `num`, `key`, `id`
- Segment metadata: Aggregate demographics vs. individual PII

**Usage:**
```bash
# Run from anywhere in the project
python scripts/review/auto_review.py
```

**Output:** Auto-resolved labels with confidence scores.

---

### 3. `manual_final_review.py` - Manual Decision Repository

**Purpose:** Codifies manual review decisions for the remaining uncertain cases.

**Format:**
```python
MANUAL_DECISIONS = {
    (table_name, column_name): (decision, confidence, reason)
}
```

**Example Decisions:**
```python
# Language code - demographic attribute
('ms_contact_extra', 'native_language_cd'):
    ('demographic_attribute', 'high', 'Language/nationality is demographic'),

# DOB flag - metadata ABOUT DOB, not the DOB itself
('CustomerPrefs', 'DobFlg'):
    ('non_pii', 'high', 'Flag/indicator about DOB, not the DOB value itself'),

# Screen name - online_identifier
('don_form_audit', 'col_scrn_nm'):
    ('online_identifier', 'high', 'Screen name is an online identifier'),
```

**Contains 65 manual decisions** covering edge cases like:
- Metadata indicators (flags, codes) vs. actual PII values
- Segment-level demographics vs. individual demographics
- Ambiguous name fields (owner names, authorized users)
- Context-dependent identifiers

**Usage:** This is a reference file consumed by the review workflow. Decisions are applied automatically when merging reviews.

---

### 4. `merge_reviews.py` - Review Merger

**Purpose:** Combine all reviewed labels into a final training dataset.

**Workflow:**
1. Loads all `reviewed_*.csv` files from the `data/` directory
2. Combines them with the original disagreements
3. Creates a final dataset with human-verified labels
4. Generates statistics on review decisions

**Usage:**
```bash
# Run from anywhere in the project
python scripts/review/merge_reviews.py
```

**Output:** `data/synthetic_pii_columns_disagreements_resolved.csv`

**Resolution Types:**
- `agreed_with_model`: Human chose the model's prediction
- `agreed_with_intended`: Human chose the originally intended class
- `manual_override`: Human selected a different class entirely

**Example Output:**
```
Found 4 review file(s):
  - reviewed_high_confidence_model_likely_right.csv
  - reviewed_high_confidence_model_likely_wrong.csv
  - reviewed_ambiguous_needs_context.csv
  - reviewed_clear_cut_cases.csv

Total reviewed records: 197

Review decisions:
  batch_kept_intended: 85
  kept_intended: 45
  kept_model: 52
  manual_override: 15

✓ Updated 197 records with human labels
✓ Saved to data/synthetic_pii_columns_disagreements_resolved.csv

Resolution breakdown:
  agreed_with_intended          :   130 ( 66.0%)
  agreed_with_model             :    52 ( 26.4%)
  manual_override               :    15 (  7.6%)

Total disagreements: 197
Resolved by human review: 197 (100.0%)
Still unresolved: 0 (0.0%)
```

---

## Workflow

### Complete Review Workflow

```bash
# Step 1: List available review categories
python scripts/review/review_disagreements.py --list-categories

# Step 2: Review each category
# Start with high-confidence cases (easy decisions)
python scripts/review/review_disagreements.py --category clear_cut_cases --batch

# Review model-likely-right cases
python scripts/review/review_disagreements.py --category high_confidence_model_likely_right

# Review model-likely-wrong cases
python scripts/review/review_disagreements.py --category high_confidence_model_likely_wrong

# Review ambiguous cases (most challenging)
python scripts/review/review_disagreements.py --category ambiguous_needs_context

# Step 3: Merge all reviews into final dataset
python scripts/review/merge_reviews.py

# Step 4: Verify the output
head data/synthetic_pii_columns_disagreements_resolved.csv
```

---

## File Dependencies

**Required Files:**
- `data/synthetic_pii_columns_disagreements.csv` - Original disagreements file
- `data/review_categories.json` - Categorized disagreements
- `src/taxonomy.py` - PII taxonomy definitions

**Generated Files:**
- `data/reviewed_<category>.csv` - Human-reviewed labels for each category
- `data/synthetic_pii_columns_disagreements_resolved.csv` - Final merged dataset

---

## Review Guidelines

### Taxonomy Reference

When reviewing, refer to the PII taxonomy:

1. **person_name** - Full names, first/last names, initials
2. **email** - Email addresses
3. **phone** - Phone numbers, mobile numbers
4. **physical_address** - Street addresses, city, state, zip, country
5. **government_id** - SSN, passport, driver's license, national ID
6. **date_of_birth** - Birth dates, age calculations based on DOB
7. **financial_account** - Credit card, bank account, payment info
8. **online_identifier** - Usernames, device IDs, session IDs, cookies
9. **demographic_attribute** - Race, ethnicity, gender, religion, nationality
10. **non_pii** - Everything else (metadata, flags, codes, internal IDs)

### Edge Case Decision Tree

**Is it metadata ABOUT PII, or the PII itself?**
- `dob_flag` → **non_pii** (indicator/flag)
- `date_of_birth` → **date_of_birth** (actual value)

**Is it individual-level or aggregate/segment-level?**
- `customer_gender` → **demographic_attribute** (individual)
- `segment_target_gender` → **non_pii** (aggregate targeting metadata)

**Is it a system identifier or personal identifier?**
- `user_id` (internal system key) → **non_pii**
- `username` (chosen by user) → **online_identifier**
- `session_id` → **online_identifier** (tracks user behavior)

**Is it public or personally identifiable?**
- `company_name` → **non_pii** (business entity)
- `owner_name` → **person_name** (individual)

---

## Statistics

From the complete review process:

**Total Disagreements:** 197 cases where model disagreed with intended labels

**Review Effort:**
- Clear-cut cases: 112 (56.9%) - Batch reviewed
- High-confidence model likely right: 40 (20.3%) - Interactive review
- High-confidence model likely wrong: 16 (8.1%) - Interactive review
- Ambiguous cases: 29 (14.7%) - Detailed interactive review

**Final Resolutions:**
- Agreed with intended: 66%
- Agreed with model: 26%
- Manual override: 8%

**Confidence Distribution:**
- High confidence: 75%
- Medium confidence: 20%
- Low confidence: 5%

---

## Troubleshooting

### "No reviewed files found!"
**Problem:** Running `merge_reviews.py` before reviewing any categories.

**Solution:** Review at least one category first:
```bash
python scripts/review/review_disagreements.py --category clear_cut_cases --batch
```

---

### "Unknown category"
**Problem:** Typo in category name or category doesn't exist.

**Solution:** List available categories:
```bash
python scripts/review/review_disagreements.py --list-categories
```

---

### "ModuleNotFoundError: No module named 'taxonomy'"
**Problem:** Scripts can't find the taxonomy module.

**Solution:** Ensure `src/taxonomy.py` exists. Scripts automatically navigate to the project root, but if you still see this error:
```bash
# Verify the file exists
ls src/taxonomy.py

# Run the script
python scripts/review/review_disagreements.py
```

---

## Best Practices

1. **Start with batch mode for clear cases:** Use `--batch` for high-confidence categories to save time.

2. **Use taxonomy reference liberally:** Press `?` in interactive mode to see class definitions.

3. **Document reasoning for overrides:** If you select a class different from both intended and model, note why in your external documentation.

4. **Review in order of confidence:** Start with clear-cut cases, end with ambiguous ones.

5. **Take breaks on ambiguous cases:** Reviewing 29 edge cases in one sitting leads to decision fatigue. Review in batches of 10-15.

6. **Cross-check similar patterns:** If you see `_id` fields, make consistent decisions (system IDs vs. government IDs).

---

## Integration with Training Pipeline

After completing the review workflow, the resolved disagreements can be integrated back into the training data:

```bash
# 1. Complete the review workflow
python scripts/review/merge_reviews.py

# 2. Use the resolved dataset in training
# Update generate_synthetic_data.py to incorporate human-reviewed labels

# 3. Retrain models with corrected labels
python src/train_models.py --run-name human_verified_v1
```

---

## Contributing

When adding new review categories or rules:

1. Update `auto_review.py` with new pattern detection logic
2. Document decision rules in comments
3. Add test cases for edge cases
4. Update this README with new categories

---

## References

- **Taxonomy Definition:** `src/taxonomy.py`
- **Disagreement Generation:** `generate_synthetic_data.py`
- **Model Training:** `src/train_models.py`
- **Evaluation Reports:** `MODEL_EVALUATION_REPORT.md`, `ML_SCIENTIFIC_ANALYSIS.md`
