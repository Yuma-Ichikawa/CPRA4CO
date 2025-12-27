# CPRA4CO — Continuous Parallel Relaxation for Diverse Combinatorial Optimization Solutions

This repository provides code associated with the TMLR paper:

**Continuous Parallel Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems**  
Yuma Ichikawa, Hiroaki Iwashita (Transactions on Machine Learning Research; published Aug 11, 2025)  
OpenReview: https://openreview.net/forum?id=ix33zd5zCw

---

## Overview

Classic combinatorial optimization (CO) focuses on finding *one* optimal solution. In many real-world settings, however, practitioners often prefer **diverse solutions** rather than a single optimum—e.g., to explore trade-offs, incorporate domain knowledge that is not fully captured by the formulation, or tolerate small constraint violations when that yields lower cost.

This work targets two important kinds of diversity:

1. **Penalty-diversified solutions**  
   When constraints are moved into the objective as penalty terms, tuning penalty strengths can be time-consuming. Generating solutions across a range of penalty intensities helps users select an appropriate trade-off.

2. **Variation-diversified solutions**  
   Even if a formulation is well-defined, it may oversimplify real-world considerations (implicit preferences, ethics, unmodeled constraints). Obtaining structurally different solutions provides candidates for post-selection.

To address these needs efficiently, the paper introduces **Continual Parallel Relaxation Annealing (CPRA)**:
a computationally efficient framework for *unsupervised-learning (UL)-based CO solvers* that generates diverse solutions **within a single training run**, leveraging **representation learning** and **parallelization** to discover shared representations and accelerate the search.

---

## What is CPRA?

**CPRA (Continual Parallel Relaxation Annealing)** is designed to:
- produce multiple diverse candidate solutions in one run,
- support both penalty- and variation-diversified solution discovery,
- reduce compute cost compared to repeatedly training UL-based solvers.

For full details, please see the paper on OpenReview.

---

## Reproducing Paper Results

The paper reports numerical experiments demonstrating that CPRA can generate diverse solutions more efficiently than existing UL-based solvers while reducing computational cost.

Use `experiment.ipynb` as the entry point; it is intended to demonstrate how to run the method and evaluate diversity and objective values for the considered CO tasks.

---

## License

* **Code**: BSD-3-Clause-Clear (see `LICENSE.txt` in this repository).
* **Paper**: CC BY 4.0 (as indicated on OpenReview).

---

## Citation

If you use this repository or the CPRA method in your research, please cite the TMLR paper:

```bibtex
@article{ichikawa2025cpra,
  title   = {Continuous Parallel Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems},
  author  = {Ichikawa, Yuma and Iwashita, Hiroaki},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  month   = {aug},
  url     = {https://openreview.net/forum?id=ix33zd5zCw},
  note    = {Published: 11 Aug 2025}
}
```
---

## Contact

For questions, please use the OpenReview discussion thread or open an issue in this repository.
