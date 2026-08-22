# Step 2.1: Yiyuan Patient / Case / Slide Identity Audit

> Candidate groups in this report are explainable naming heuristics only. They are not confirmed patient IDs.

## Dataset identity

- Total records: 968
- Unique slide_id: 968
- Unique case_id: 968
- Literal case_id != slide_id rows: 5
- Step 2.0 inventory population/order match: yes (`analysis/stage2_yiyuan_data_audit/00_inventory/dataset_inventory.csv`)
- No higher-level Yiyuan patient identifier or mapping was found in the repository paths searched.

## Grouping rule and confidence

A slide is grouping-eligible only when its full ID parses as a leading numeric core followed by an explicit B/block suffix. Slides form a candidate group only when the complete numeric core is identical and at least two records exist. No edit distance or partial-number similarity is used.

`medium` means a standard B/block suffix and consistent labels/case-slide spelling; `low` means a nonstandard sub-block suffix, label conflict, or literal case/slide mismatch. No group is assigned `high`, because the repository does not contain hospital semantics proving patient identity. Every group requires manual confirmation.

## ID patterns

### slide_id

- `numeric_B`: 81
- `numeric_B_hyphen_number`: 1
- `numeric_B_number`: 494
- `numeric_B_number_letter`: 1
- `numeric_hyphen_B`: 59
- `numeric_hyphen_B_number`: 332
- Unparsed slide IDs: 0

### case_id

- `numeric_B`: 81
- `numeric_B_hyphen_number`: 1
- `numeric_B_number`: 494
- `numeric_B_number_letter`: 1
- `numeric_hyphen_B`: 54
- `numeric_hyphen_B_number`: 332
- `numeric_hyphen_B_trailing_punctuation`: 5

## Candidate multi-slide groups

- Candidate groups: 71
- Slides involved: 157
- Maximum group size: 5
- size 2: 61 groups
- size 3: 7 groups
- size 4: 1 groups
- size 5: 2 groups

## Label consistency

- Label-consistent candidate groups: 60
- Label-conflicting candidate groups: 11
- Conflicts are high-priority review items; no label was automatically selected.
- `2484791`: 2484791-B3 (Adenocarcinoma), 2484791-B2 (NonAdenocarcinoma)
- `25013062`: 25013062B3 (Adenocarcinoma), 25013062B2 (NonAdenocarcinoma)
- `25013189`: 25013189B2 (Adenocarcinoma), 25013189B3 (Adenocarcinoma), 25013189B4 (NonAdenocarcinoma)
- `25017341`: 25017341B4 (Adenocarcinoma), 25017341B3 (NonAdenocarcinoma)
- `25017355`: 25017355B1 (Adenocarcinoma), 25017355B4 (NonAdenocarcinoma)
- `25020714`: 25020714B2 (Adenocarcinoma), 25020714B4 (NonAdenocarcinoma)
- `25023687`: 25023687B2 (Adenocarcinoma), 25023687B3 (NonAdenocarcinoma), 25023687B4 (NonAdenocarcinoma)
- `25024855`: 25024855B3 (Adenocarcinoma), 25024855B4 (NonAdenocarcinoma)
- `25024982`: 25024982B2 (Adenocarcinoma), 25024982B1 (NonAdenocarcinoma)
- `25025567`: 25025567B3 (Adenocarcinoma), 25025567B4 (NonAdenocarcinoma)
- `25026103`: 25026103B3 (Adenocarcinoma), 25026103B2 (NonAdenocarcinoma)

## Strict split audit

- Experiment config: `trained_models/adenocarcinoma_strict5_new/adenocarcinoma_biomedclip_dual_strict5_new_s1/experiment_adenocarcinoma_biomedclip_dual_strict5_new.txt`
- Authoritative split directory from config: `splits/strict/task_adenocarcinoma_100_k5_s1`
- Configured folds: 5
- Rule confirmed from generator code: test=i, val=(i+1)%k, train=remaining folds.
- All fold files cover all 968 slides exactly once per fold: yes
- strict_fold_assignments.csv agrees with test membership: True
- Model-directory split copies are semantically identical: True
- Confirmed case_id leakage groups: 0
- Potential candidate-group leakage groups (unique): 59
- Potential candidate-group leakage fold/group events: 213
- Folds involved: 0, 1, 2, 3, 4
- Candidate slides involved in potential leakage: 133

Fold coverage details:

- fold 0: train=580, val=194, test=194, coverage=968
- fold 1: train=580, val=194, test=194, coverage=968
- fold 2: train=581, val=193, test=194, coverage=968
- fold 3: train=582, val=193, test=193, coverage=968
- fold 4: train=581, val=194, test=193, coverage=968

## Manual confirmation

- Manual confirmation rows: 76
- High-priority rows: 62
- Primary question: does the leading numeric core represent a patient, encounter/accession, specimen, or another entity, and do differing B suffixes identify related tissue blocks?
- The 11 label-conflicting groups should be reviewed first because they may indicate an invalid grouping heuristic, true within-patient heterogeneity, or annotation issues.
- The five case_id values with a trailing period not present in slide_id also require clarification before any normalization policy is adopted.

## Current conclusion

- State: **C**
- 发现命名候选组且存在跨 split：如果医院确认这些组属于同一患者/病例/送检，当前 strict5 存在 patient-level leakage 风险。
- No confirmed literal case_id leakage was found.
- The existing strict5 can remain a provisional baseline, but it cannot yet be described as proven patient-level strict.
- Hospital ID semantics should be confirmed before deciding whether a regrouped strict5 is necessary.

## Scope

This step did not modify the dataset CSV, data files, split files, or model artifacts, and did not execute Step 2.2.
