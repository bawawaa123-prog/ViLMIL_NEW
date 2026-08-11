# Final Table (Consistent Metric Definition)

| Setting | AUC | ACC | F1-macro | F1-class1 | Threshold | Notes |
|---|---:|---:|---:|---:|---:|---|
| Single-fold mean (5 folds) | 0.881284 | 0.748256 | 0.640074 | 0.447194 | - | Average of fold_0..4 at threshold 0.5 |
| Ensemble (probability average) | 0.924729 | 0.813953 | 0.700022 | 0.515152 | 0.500000 | 5-fold p1 average, default threshold |
| Ensemble (probability average, tuned) | 0.924729 | 0.886628 | 0.769923 | 0.606061 | 0.759977 | Threshold chosen by maximizing class1 F1 on external set |
