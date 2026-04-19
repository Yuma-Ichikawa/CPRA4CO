# CPRA4CO — Continuous Parallel Relaxation for Diverse Combinatorial Optimization Solutions

> [!IMPORTANT]
> ## This repository has moved → use [**QQA4CO**](https://github.com/Yuma-Ichikawa/QQA4CO)
>
> **CPRA is now developed and maintained inside the unified
> [QQA4CO](https://github.com/Yuma-Ichikawa/QQA4CO) toolkit.** The full
> algorithm of this paper — multi-head GCN backbone, per-replica QUBO loss,
> CRA penalty annealing, and the optional inter-replica diversity term —
> has been ported there with **the same numerics**, plus best-so-far
> tracking, batched per-replica evaluation, a CLI, a Streamlit dashboard,
> and Blackwell (`sm_100`) GPU support via PyTorch Geometric.
>
> ### One install, three solvers
>
> | Solver | Paper | API in QQA4CO |
> | --- | --- | --- |
> | **CPRA** *(this paper)* | TMLR 2025 | `qqa.pignn.train_cpra_pi_gnn` |
> | CRA-PI-GNN | NeurIPS 2024 | `qqa.pignn.train_cra_pi_gnn` |
> | PQQA | ICLR 2025 | `qqa.anneal` |
>
> ### Quick start (replaces this repo's `experiment.ipynb`)
>
> ```bash
> pip install "qqa[pignn]"
> ```
>
> ```python
> import qqa
> from qqa.pignn import train_cpra_pi_gnn
>
> g = ...  # any networkx graph
> base = qqa.MaximumIndependentSet(g, penalty=2.0)
>
> # Penalty diversification — one solution per penalty in a single run.
> result = train_cpra_pi_gnn(
>     base,
>     num_replicas=4,
>     replica_problems=[qqa.MaximumIndependentSet(g, penalty=p)
>                       for p in [1.5, 2.0, 2.5, 3.0]],
> )
> for rec in result.score["extra"]["replicas"]:
>     print(rec["replica"], rec["obj"], rec["sol"].sum().item())
> ```
>
> ```bash
> # Same thing from the command line:
> qqa solve --problem mis --backend cpra --size 200 \
>           --cpra-num-replicas 4 --cpra-penalty-levels 1.5,2.0,2.5,3.0 \
>           --epochs 5000 --learning-rate 1e-3 \
>           --pignn-init-reg-param -2 --pignn-annealing-rate 5e-4
> ```
>
> A walkthrough notebook lives at
> [`QQA4CO/notebooks/cpra_pignn_example.ipynb`](https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/notebooks/cpra_pignn_example.ipynb).
>
> ### What stays here
>
> This standalone repository is **frozen** as the canonical reference for
> the published TMLR 2025 experiments — `experiment.ipynb` and the
> reported numbers reproduce exactly as in the paper. It remains usable
> as-is, but **is not the recommended starting point for new work**.

---

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
