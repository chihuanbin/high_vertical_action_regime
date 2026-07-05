# High Vertical Action Regime (VEAR) in Galactic Open Clusters

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-ApJL%20Submission-orange.svg)]()

Official analysis pipeline accompanying the manuscript

> **A Vertically Enhanced Action-space Regime in Galactic Open Clusters**

---

# Overview

This repository contains the complete analysis pipeline used to investigate the action-space structure of Galactic open clusters using the HUNT24 catalogue and Gaia DR3.

Rather than claiming the discovery of a new stellar population, this project demonstrates the existence of a **statistically reproducible Vertically Enhanced Action-space Regime (VEAR)** within the continuous action-space distribution of Galactic open clusters.

The repository includes all scripts used in the paper, from the initial Gaussian-mixture decomposition to the final robustness analyses and physical interpretation.

---

# Scientific Contributions

The project performs the following analyses.

## 1. Action-space decomposition

- Compute orbital actions
- Gaussian Mixture Model decomposition
- Identify the vertically enhanced action-space regime (VEAR)

Script

```
run_v3_action_space_analysis.py
```

---

## 2. Matched-control experiment

Strict matched-control comparison controlling

- age
- Galactocentric radius
- angular momentum
- eccentricity

Scripts

```
run_v4_matched_control_experiment.py
run_v5_lz_eccentricity_matching.py
```

---

## 3. Bootstrap robustness

1000 bootstrap realizations

Evaluate

- optimal mixture number
- component stability
- ARI
- VI
- Jaccard overlap
- component-center uncertainty

Script

```
run_bootstrap_vdc_stability.py
```

---

## 4. Multimodality tests

Non-parametric tests

including

- Hartigan Dip Test
- Silverman Test
- KDE
- Persistent Homology
- Watershed / Morse-Smale proxy

Script

```
run_multimodality_vdc_test.py
```

---

## 5. Orbital-family analysis

Compare

- orbital frequencies
- action ratios
- vertical amplitudes

Script

```
run_orbital_family_analysis.py
```

---

## 6. Galactic physical-origin study

Evaluate possible associations with

- Galactic flare
- warp
- phase-space structure
- bar
- spiral structure
- Sagittarius perturbation proxies

Script

```
run_physical_origin_study.py
```

---

# Main Scientific Result

The analyses consistently support

✔ a statistically reproducible vertically enhanced action-space regime

but do **not** support

✘ an independently resolved density peak

✘ a distinct open-cluster population

✘ a new orbital family

✘ a uniquely identifiable perturbation origin

Instead, the evidence indicates that Galactic open clusters occupy a continuous action-space distribution containing a reproducible high-vertical-action regime.

---

# Repository Structure

```
.
├── run_v3_action_space_analysis.py
├── run_v4_matched_control_experiment.py
├── run_v5_lz_eccentricity_matching.py
├── run_bootstrap_vdc_stability.py
├── run_multimodality_vdc_test.py
├── run_orbital_family_analysis.py
├── run_physical_origin_study.py
├── run_high_latitude_dynamics.py
├── run_vertical_action_outliers.py
├── results/
├── figures/
└── paper/
```

---

# Requirements

Python ≥ 3.11

Main packages

```
numpy
scipy
pandas
astropy
galpy
matplotlib
scikit-learn
joblib
```

---

# Reproducibility

All numerical results reported in the manuscript can be reproduced directly from the scripts provided in this repository.

Each analysis module produces

- figures

- CSV tables

- summary statistics

- JSON reports

used in the manuscript.

---

# Novelty Statement

Previous studies have primarily characterized Galactic open clusters through their spatial distributions, ages, and kinematics, whereas their global organization in orbital action space has remained largely unexplored. This work presents a comprehensive action-space analysis of 1,079 quality-selected Galactic open clusters from the HUNT24 catalogue and Gaia DR3. We identify a statistically reproducible **Vertically Enhanced Action-space Regime (VEAR)** characterized by systematically larger vertical actions and orbital amplitudes than carefully matched reference clusters.

Importantly, we combine Gaussian-mixture modelling with extensive robustness analyses—including 1,000 bootstrap realizations, matched-control experiments, non-parametric multimodality tests, persistent-homology diagnostics, orbital-frequency analyses, and Galactic perturbation association studies. These analyses demonstrate that the VEAR is a reproducible high-vertical-action regime embedded within a continuous action-space distribution, while providing no evidence that it constitutes an independently resolved density peak, a distinct orbital family, or the product of a unique Galactic perturbation.

The primary contribution of this work is therefore not the identification of a new stellar population, but the establishment of a statistically rigorous framework for distinguishing reproducible dynamical structure from over-interpretation of probabilistic clustering in Galactic action space.

---

# Citation

If you use this code, please cite the accompanying manuscript.

```
Huanbin Chi et al.

A Vertically Enhanced Action-space Regime in Galactic Open Clusters

Submitted to The Astrophysical Journal Letters
```

---

# License

MIT License

---

# Contact

Huanbin Chi

School of Artificial Intelligence

Yunnan Open University

Center for Astrophysics, Guangzhou University
