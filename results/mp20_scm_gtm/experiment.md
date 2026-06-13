# Descriptor experiment — MP-20

6000 structures, frame set 1441, GTM k=16 niter=200. Q2 is held-out (5-fold); higher = more predictive.

## Mean GTM-responsibility Q2 (across properties) + size confound

| descriptor | dims | mean Q2 | n_sites Q2 |
|---|---|---|---|
| scm | 20 | +0.059 | +0.331 |
| composition | 132 | +0.453 | +0.216 |
| scm+composition | 152 | +0.419 | +0.465 |

## Per-property Q2 (GTM-resp / GTM-2D / PCA-2D)

| descriptor | band_gap | formation_energy_per_atom | e_above_hull |
|---|---|---|---|
| scm | +0.06/+0.13/+0.11 | +0.09/+0.14/+0.16 | +0.03/+0.02/+0.02 |
| composition | +0.43/+0.41/+0.39 | +0.82/+0.81/+0.78 | +0.11/+0.09/+0.06 |
| scm+composition | +0.39/+0.36/+0.27 | +0.75/+0.73/+0.35 | +0.12/+0.09/+0.03 |
