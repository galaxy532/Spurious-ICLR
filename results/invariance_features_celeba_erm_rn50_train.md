
### Invariance rule (Breslow-Day)  features_celeba_erm_rn50_train

d = 2048, bins = 8 (+ zero atom = True), min_cell = 10, FDR q = 0.05
tau_rel = 0.2, tau_het = 0.2

- BH-rejected (heterogeneous relation): **820** of 2048
- n_r = **475**, n_s = **608**, n_weak = 965
- coordinates with < 2 usable bins (untestable): 917

| quantile | strength (log-odds) | het (log-odds) |
|---|---|---|
| 0.50 | 0.330 | 0.062 |
| 0.75 | 1.120 | 0.242 |
| 0.90 | 1.568 | 0.407 |
| 0.99 | 2.289 | 0.716 |
| 1.00 | 2.741 | 1.276 |

Threshold sensitivity (tau_rel = tau_het = tau):

| tau | n_r | n_s | n_weak |
|---|---|---|---|
| 0.05 | 311 | 815 | 922 |
| 0.10 | 326 | 789 | 933 |
| 0.15 | 385 | 717 | 946 |
| 0.20 | 475 | 608 | 965 |
| 0.30 | 698 | 341 | 1009 |
| 0.40 | 785 | 204 | 1059 |
