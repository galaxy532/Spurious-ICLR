# Long-horizon diagnostics

All quantities are defined in words in the docstring of `long_horizon.py`. Short version: z = h * (number of GD steps); beta = decay exponent of a group's soft error in z, fitted on the window [z/2, 2z]; kappa = exponent of a group's soft error in eps at that single z, fitted across the eps grid.

## celeba_erm_rn50_n5000_train

status: finished, z reached: 1e+06, full-split margin (lower bound, separability_check): 0.0257

### eps-exponents at each z

Each group is fitted against its OWN weight share, as the theorem writes it: err_min ~ c/(eps z) gives kappa_min -> 1 (regression of log err_min on log eps), err_maj ~ c/((1-eps) z) gives kappa_maj -> 1 (regression of log err_maj on log(1-eps)). Both targets are 1 and neither depends on the eps grid. Each fit uses the 5 eps values at that single z.

'max resid' is the largest residual of that 5-point fit: it says whether the points are a straight line at all. Below ~0.02 the exponent means something; at 0.1 or more the curve is bent and the slope is just a number. A group whose beta has not reached its limit has no reason to show a clean power law in eps.

kappa_maj_eps is the superseded definition (majority fitted against log eps), whose grid-dependent target is -0.157 for this grid; it is kept only to compare with reports written before 18 Sept 2026. For a group whose beta -> alpha > 1 the theorem gives no eps-dependence at all.

| z | kappa_min (-> 1) | max resid | kappa_maj (-> 1) | max resid | kappa_maj_eps (legacy) |
|---|---|---|---|---|---|
| 1 | 0.028 | 0.036 | 1.143 | 0.177 | -0.218 |
| 3.15 | 0.228 | 0.193 | 0.775 | 0.045 | -0.134 |
| 10 | 0.329 | 0.272 | 0.468 | 0.171 | -0.100 |
| 31.6 | 0.373 | 0.208 | 0.627 | 0.085 | -0.119 |
| 99.8 | 0.433 | 0.205 | 0.780 | 0.166 | -0.163 |
| 316 | 0.481 | 0.171 | 1.045 | 0.281 | -0.230 |
| 998 | 0.496 | 0.171 | 1.296 | 0.381 | -0.291 |
| 3.15e+03 | 0.515 | 0.166 | 1.567 | 0.449 | -0.350 |
| 9.98e+03 | 0.539 | 0.154 | 1.760 | 0.449 | -0.382 |
| 3.15e+04 | 0.623 | 0.155 | 1.690 | 0.394 | -0.357 |
| 9.98e+04 | 0.790 | 0.149 | 1.255 | 0.253 | -0.250 |
| 3.15e+05 | 0.961 | 0.101 | 0.809 | 0.070 | -0.129 |
| 1e+06 | 1.047 | 0.060 | 0.672 | 0.145 | -0.071 |

### decay exponents in z (one column per eps)

Theory: alpha < 1: both groups -> 1. alpha > 1: the larger-r-margin group -> alpha, the other -> 1.

beta_min:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.064 | 0.073 | 0.084 | 0.114 | 0.499 |
| 3.15 | 0.033 | 0.053 | 0.115 | 0.265 | 0.223 |
| 10 | 0.005 | 0.036 | 0.151 | 0.311 | 0.068 |
| 31.6 | 0.011 | 0.058 | 0.160 | 0.187 | 0.291 |
| 99.8 | 0.022 | 0.082 | 0.141 | 0.172 | 0.218 |
| 316 | 0.046 | 0.123 | 0.126 | 0.137 | 0.174 |
| 998 | 0.110 | 0.160 | 0.159 | 0.146 | 0.164 |
| 3.15e+03 | 0.184 | 0.218 | 0.230 | 0.242 | 0.236 |
| 9.98e+03 | 0.235 | 0.286 | 0.333 | 0.369 | 0.405 |
| 3.15e+04 | 0.297 | 0.391 | 0.484 | 0.653 | 0.715 |
| 9.98e+04 | 0.401 | 0.525 | 0.784 | 0.946 | 0.985 |
| 3.15e+05 | 0.537 | 0.856 | 1.024 | 1.017 | 1.020 |
| 1e+06 (half window) | 0.793 | 1.019 | 1.042 | 1.010 | 1.014 |

A row marked (half window) is fitted on [z/2, z] only, because the run stops at that z; it is biased and should not be read as the limit.

beta_maj:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.267 | 0.247 | 0.196 | 0.140 | 0.371 |
| 3.15 | 0.186 | 0.169 | 0.164 | 0.262 | 0.171 |
| 10 | 0.202 | 0.205 | 0.232 | 0.347 | 0.071 |
| 31.6 | 0.265 | 0.226 | 0.209 | 0.153 | 0.228 |
| 99.8 | 0.323 | 0.233 | 0.194 | 0.171 | 0.159 |
| 316 | 0.364 | 0.250 | 0.199 | 0.174 | 0.144 |
| 998 | 0.363 | 0.281 | 0.229 | 0.190 | 0.152 |
| 3.15e+03 | 0.375 | 0.344 | 0.311 | 0.265 | 0.200 |
| 9.98e+03 | 0.393 | 0.420 | 0.434 | 0.412 | 0.369 |
| 3.15e+04 | 0.443 | 0.529 | 0.577 | 0.680 | 0.651 |
| 9.98e+04 | 0.549 | 0.632 | 0.836 | 0.943 | 0.948 |
| 3.15e+05 | 0.653 | 0.912 | 1.029 | 1.014 | 1.001 |
| 1e+06 (half window) | 0.867 | 1.032 | 1.046 | 1.008 | 1.001 |

A row marked (half window) is fitted on [z/2, z] only, because the run stops at that z; it is biased and should not be read as the limit.

### scaled errors (theory: constant in z and eps for a group whose beta -> 1)

eps_z_err_min:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.00572 | 0.0172 | 0.0459 | 0.11 | 0.254 |
| 3.15 | 0.0172 | 0.0502 | 0.121 | 0.219 | 0.344 |
| 10 | 0.0525 | 0.149 | 0.342 | 0.501 | 0.726 |
| 31.6 | 0.166 | 0.456 | 0.918 | 1.39 | 1.96 |
| 99.8 | 0.513 | 1.32 | 2.46 | 3.75 | 4.69 |
| 316 | 1.57 | 3.69 | 6.36 | 9.42 | 12 |
| 998 | 4.57 | 9.92 | 17.4 | 26.2 | 31.9 |
| 3.15e+03 | 12.1 | 25.4 | 44.1 | 65.1 | 78.4 |
| 9.98e+03 | 30.2 | 60.2 | 101 | 149 | 178 |
| 3.15e+04 | 69.5 | 129 | 203 | 266 | 298 |
| 9.98e+04 | 149 | 242 | 308 | 336 | 346 |
| 3.15e+05 | 275 | 348 | 347 | 328 | 339 |
| 1e+06 | 394 | 362 | 325 | 324 | 333 |

one_minus_eps_z_err_maj:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.205 | 0.218 | 0.249 | 0.284 | 0.24 |
| 3.15 | 0.464 | 0.476 | 0.493 | 0.484 | 0.409 |
| 10 | 1.22 | 1.32 | 1.57 | 1.23 | 0.94 |
| 31.6 | 2.96 | 3.16 | 3.34 | 3.15 | 2.44 |
| 99.8 | 6.66 | 7.73 | 8.58 | 8.5 | 6.43 |
| 316 | 14.1 | 18.6 | 22.1 | 22.9 | 17.8 |
| 998 | 29.2 | 43.3 | 54.9 | 58.2 | 47.3 |
| 3.15e+03 | 60.4 | 96.4 | 128 | 145 | 123 |
| 9.98e+03 | 123 | 197 | 265 | 312 | 287 |
| 3.15e+04 | 239 | 361 | 471 | 536 | 510 |
| 9.98e+04 | 434 | 584 | 659 | 665 | 630 |
| 3.15e+05 | 681 | 765 | 719 | 655 | 636 |
| 1e+06 | 883 | 767 | 672 | 648 | 635 |

### iterate margin and training accuracy

margin = min over training points of y w.x / ||w|| (negative: some point misclassified). Rises to the maximum margin at rate O(1/log t) (Soudry et al. 2018, Thm 5).

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | -39.5 | -35.4 | -25.2 | -17.3 | -18.3 |
| 3.15 | -31.6 | -26.1 | -14.3 | -10.5 | -12.5 |
| 10 | -17.3 | -11.4 | -8.11 | -8.87 | -9.18 |
| 31.6 | -10.6 | -6.93 | -5.9 | -6.85 | -7.12 |
| 99.8 | -9.14 | -4.87 | -4.8 | -5.52 | -5.78 |
| 316 | -7.82 | -4.12 | -3.77 | -4.22 | -4.36 |
| 998 | -5.36 | -2.83 | -2.72 | -2.84 | -2.83 |
| 3.15e+03 | -3.1 | -2.02 | -1.89 | -1.75 | -1.55 |
| 9.98e+03 | -2 | -1.63 | -1.3 | -0.975 | -0.555 |
| 3.15e+04 | -1.61 | -1.21 | -0.817 | -0.443 | -0.116 |
| 9.98e+04 | -1.24 | -0.795 | -0.39 | -0.00881 | -0.00324 |
| 3.15e+05 | -0.841 | -0.381 | -0.00308 | 0.0117 | 0.0118 |
| 1e+06 | -0.419 | -0.00105 | 0.0111 | 0.0174 | 0.0171 |

minority training accuracy:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.413 | 0.397 | 0.371 | 0.365 | 0.431 |
| 3.15 | 0.471 | 0.488 | 0.546 | 0.723 | 0.884 |
| 10 | 0.488 | 0.523 | 0.605 | 0.872 | 0.949 |
| 31.6 | 0.485 | 0.537 | 0.687 | 0.887 | 0.957 |
| 99.8 | 0.492 | 0.571 | 0.739 | 0.898 | 0.980 |
| 316 | 0.504 | 0.627 | 0.798 | 0.928 | 0.982 |
| 998 | 0.549 | 0.691 | 0.826 | 0.939 | 0.989 |
| 3.15e+03 | 0.624 | 0.752 | 0.867 | 0.957 | 0.991 |
| 9.98e+03 | 0.705 | 0.823 | 0.915 | 0.977 | 0.997 |
| 3.15e+04 | 0.796 | 0.896 | 0.964 | 0.992 | 1.000 |
| 9.98e+04 | 0.877 | 0.959 | 0.992 | 0.999 | 1.000 |
| 3.15e+05 | 0.948 | 0.992 | 1.000 | 1.000 | 1.000 |
| 1e+06 | 0.989 | 1.000 | 1.000 | 1.000 | 1.000 |

## celeba_under_rn50_n5000_train

status: finished, z reached: 1e+06, full-split margin (lower bound, separability_check): 0.00828

### eps-exponents at each z

Each group is fitted against its OWN weight share, as the theorem writes it: err_min ~ c/(eps z) gives kappa_min -> 1 (regression of log err_min on log eps), err_maj ~ c/((1-eps) z) gives kappa_maj -> 1 (regression of log err_maj on log(1-eps)). Both targets are 1 and neither depends on the eps grid. Each fit uses the 5 eps values at that single z.

'max resid' is the largest residual of that 5-point fit: it says whether the points are a straight line at all. Below ~0.02 the exponent means something; at 0.1 or more the curve is bent and the slope is just a number. A group whose beta has not reached its limit has no reason to show a clean power law in eps.

kappa_maj_eps is the superseded definition (majority fitted against log eps), whose grid-dependent target is -0.157 for this grid; it is kept only to compare with reports written before 18 Sept 2026. For a group whose beta -> alpha > 1 the theorem gives no eps-dependence at all.

| z | kappa_min (-> 1) | max resid | kappa_maj (-> 1) | max resid | kappa_maj_eps (legacy) |
|---|---|---|---|---|---|
| 1 | 0.248 | 0.312 | 0.537 | 0.029 | -0.091 |
| 3.15 | 0.187 | 0.158 | 0.840 | 0.192 | -0.173 |
| 10 | 0.369 | 0.273 | 0.458 | 0.055 | -0.086 |
| 31.6 | 0.458 | 0.230 | 0.369 | 0.106 | -0.085 |
| 99.8 | 0.519 | 0.244 | 0.388 | 0.133 | -0.091 |
| 316 | 0.531 | 0.247 | 0.499 | 0.166 | -0.114 |
| 998 | 0.589 | 0.237 | 0.593 | 0.209 | -0.139 |
| 3.15e+03 | 0.611 | 0.232 | 0.718 | 0.265 | -0.171 |
| 9.98e+03 | 0.627 | 0.238 | 0.885 | 0.340 | -0.214 |
| 3.15e+04 | 0.644 | 0.234 | 1.084 | 0.407 | -0.260 |
| 9.98e+04 | 0.690 | 0.176 | 1.259 | 0.397 | -0.282 |
| 3.15e+05 | 0.789 | 0.065 | 1.267 | 0.207 | -0.242 |
| 1e+06 | 0.891 | 0.022 | 1.106 | 0.052 | -0.184 |

### decay exponents in z (one column per eps)

Theory: alpha < 1: both groups -> 1. alpha > 1: the larger-r-margin group -> alpha, the other -> 1.

beta_min:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | -0.026 | -0.016 | 0.005 | 0.055 | 0.303 |
| 3.15 | -0.004 | 0.015 | 0.084 | 0.258 | 0.182 |
| 10 | -0.006 | 0.018 | 0.133 | 0.276 | 0.264 |
| 31.6 | 0.005 | 0.027 | 0.122 | 0.195 | 0.226 |
| 99.8 | 0.017 | 0.047 | 0.090 | 0.108 | 0.146 |
| 316 | 0.021 | 0.076 | 0.111 | 0.135 | 0.155 |
| 998 | 0.028 | 0.096 | 0.114 | 0.164 | 0.176 |
| 3.15e+03 | 0.078 | 0.110 | 0.107 | 0.101 | 0.165 |
| 9.98e+03 | 0.127 | 0.136 | 0.136 | 0.148 | 0.171 |
| 3.15e+04 | 0.158 | 0.200 | 0.273 | 0.265 | 0.242 |
| 9.98e+04 | 0.206 | 0.365 | 0.509 | 0.517 | 0.437 |
| 3.15e+05 | 0.360 | 0.650 | 0.692 | 0.748 | 0.751 |
| 1e+06 (half window) | 0.553 | 0.743 | 0.789 | 0.833 | 0.920 |

A row marked (half window) is fitted on [z/2, z] only, because the run stops at that z; it is biased and should not be read as the limit.

beta_maj:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.220 | 0.193 | 0.142 | 0.140 | 0.191 |
| 3.15 | 0.227 | 0.228 | 0.228 | 0.267 | 0.183 |
| 10 | 0.170 | 0.142 | 0.131 | 0.169 | 0.218 |
| 31.6 | 0.150 | 0.112 | 0.125 | 0.141 | 0.154 |
| 99.8 | 0.164 | 0.119 | 0.112 | 0.101 | 0.094 |
| 316 | 0.179 | 0.139 | 0.118 | 0.103 | 0.093 |
| 998 | 0.205 | 0.162 | 0.132 | 0.121 | 0.112 |
| 3.15e+03 | 0.246 | 0.181 | 0.141 | 0.128 | 0.116 |
| 9.98e+03 | 0.290 | 0.217 | 0.174 | 0.154 | 0.129 |
| 3.15e+04 | 0.337 | 0.275 | 0.278 | 0.250 | 0.188 |
| 9.98e+04 | 0.347 | 0.413 | 0.498 | 0.468 | 0.348 |
| 3.15e+05 | 0.445 | 0.693 | 0.730 | 0.726 | 0.643 |
| 1e+06 (half window) | 0.613 | 0.823 | 0.828 | 0.838 | 0.855 |

A row marked (half window) is fitted on [z/2, z] only, because the run stops at that z; it is biased and should not be read as the limit.

### scaled errors (theory: constant in z and eps for a group whose beta -> 1)

eps_z_err_min:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.00572 | 0.0165 | 0.0401 | 0.0843 | 0.0955 |
| 3.15 | 0.0184 | 0.0517 | 0.123 | 0.262 | 0.419 |
| 10 | 0.0583 | 0.164 | 0.359 | 0.555 | 0.662 |
| 31.6 | 0.185 | 0.505 | 0.861 | 1.26 | 1.59 |
| 99.8 | 0.576 | 1.53 | 2.39 | 3.28 | 3.98 |
| 316 | 1.78 | 4.51 | 7.58 | 10.2 | 11.3 |
| 998 | 5.5 | 12.9 | 20.5 | 25.1 | 28.3 |
| 3.15e+03 | 16.4 | 36.3 | 57.3 | 73 | 74.8 |
| 9.98e+03 | 46.1 | 100 | 159 | 196 | 197 |
| 3.15e+04 | 124 | 263 | 405 | 499 | 494 |
| 9.98e+04 | 317 | 608 | 823 | 1.02e+03 | 1.09e+03 |
| 3.15e+05 | 748 | 1.08e+03 | 1.28e+03 | 1.53e+03 | 1.74e+03 |
| 1e+06 | 1.34e+03 | 1.48e+03 | 1.69e+03 | 1.89e+03 | 2.01e+03 |

one_minus_eps_z_err_maj:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.2 | 0.2 | 0.199 | 0.19 | 0.147 |
| 3.15 | 0.485 | 0.512 | 0.585 | 0.645 | 0.463 |
| 10 | 1.25 | 1.29 | 1.32 | 1.23 | 0.895 |
| 31.6 | 3.29 | 3.54 | 3.78 | 3.47 | 2.3 |
| 99.8 | 8.69 | 9.82 | 10.3 | 9.47 | 6.31 |
| 316 | 22.6 | 26.8 | 28.4 | 26.2 | 18.2 |
| 998 | 57.3 | 71.2 | 77.6 | 73.2 | 50.8 |
| 3.15e+03 | 140 | 185 | 210 | 200 | 141 |
| 9.98e+03 | 325 | 466 | 558 | 541 | 389 |
| 3.15e+04 | 716 | 1.12e+03 | 1.38e+03 | 1.38e+03 | 1.03e+03 |
| 9.98e+04 | 1.51e+03 | 2.41e+03 | 2.83e+03 | 2.94e+03 | 2.45e+03 |
| 3.15e+05 | 3.16e+03 | 4.08e+03 | 4.35e+03 | 4.63e+03 | 4.44e+03 |
| 1e+06 | 5.22e+03 | 5.19e+03 | 5.47e+03 | 5.75e+03 | 5.64e+03 |

### iterate margin and training accuracy

margin = min over training points of y w.x / ||w|| (negative: some point misclassified). Rises to the maximum margin at rate O(1/log t) (Soudry et al. 2018, Thm 5).

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | -42.8 | -42.6 | -43.4 | -33.3 | -14.7 |
| 3.15 | -13.9 | -11.1 | -11.7 | -14.5 | -15.1 |
| 10 | -19.7 | -15.8 | -11.8 | -8.76 | -9.45 |
| 31.6 | -15.5 | -10.5 | -5.69 | -7.09 | -7.62 |
| 99.8 | -12.6 | -8.06 | -3.87 | -4.69 | -5.42 |
| 316 | -9.69 | -6.13 | -2.47 | -3.03 | -3.51 |
| 998 | -7.01 | -4.77 | -1.64 | -2.03 | -2.34 |
| 3.15e+03 | -5.31 | -3.12 | -0.986 | -1.2 | -1.48 |
| 9.98e+03 | -3.86 | -2.14 | -0.515 | -0.637 | -0.874 |
| 3.15e+04 | -2.76 | -1.54 | -0.213 | -0.27 | -0.431 |
| 9.98e+04 | -2.18 | -0.573 | -0.0974 | -0.0781 | -0.148 |
| 3.15e+05 | -1.41 | -0.106 | -0.0666 | -0.0406 | -0.0296 |
| 1e+06 | -0.198 | -0.0731 | -0.0464 | -0.0064 | -0.000625 |

minority training accuracy:

| z | eps=0.01 | eps=0.03 | eps=0.08 | eps=0.2 | eps=0.5 |
|---|---|---|---|---|---|
| 1 | 0.454 | 0.481 | 0.541 | 0.626 | 0.900 |
| 3.15 | 0.420 | 0.455 | 0.492 | 0.567 | 0.829 |
| 10 | 0.439 | 0.470 | 0.597 | 0.794 | 0.969 |
| 31.6 | 0.431 | 0.483 | 0.741 | 0.925 | 0.973 |
| 99.8 | 0.440 | 0.502 | 0.795 | 0.944 | 0.978 |
| 316 | 0.452 | 0.538 | 0.770 | 0.942 | 0.983 |
| 998 | 0.459 | 0.586 | 0.806 | 0.963 | 0.987 |
| 3.15e+03 | 0.489 | 0.645 | 0.827 | 0.968 | 0.991 |
| 9.98e+03 | 0.549 | 0.694 | 0.850 | 0.975 | 0.993 |
| 3.15e+04 | 0.622 | 0.749 | 0.900 | 0.982 | 0.995 |
| 9.98e+04 | 0.694 | 0.831 | 0.958 | 0.990 | 0.999 |
| 3.15e+05 | 0.782 | 0.937 | 0.987 | 0.997 | 1.000 |
| 1e+06 | 0.914 | 0.986 | 0.996 | 1.000 | 1.000 |
