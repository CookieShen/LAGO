# LAGO: LLM-Augmented Grid Optimization for Constrained Auto-Bidding in FinTech

[//]: # ([![Venue: CIKM 2026]&#40;https://img.shields.io/badge/Venue-CIKM%202026-blue.svg&#41;]&#40;https://cikm2026.diag.uniroma1.it/&#41;)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

> This repository contains the official offline simulation environment and algorithmic implementation for the paper **"LAGO: LLM-Augmented Grid Optimization for Constrained Auto-Bidding in FinTech"** (Under Review at CIKM 2026 Applied Research Track).

*(Note: Due to commercial confidentiality, the online deployment pipeline and proprietary FinTech user profiles have been masked or replaced with this semi-synthetic configuration. However, this repository accurately reproduces the core mathematical dynamics and evaluation logic presented in the paper.)*

## 📖 Overview

In FinTech real-time bidding (RTB), traditional auto-bidding agents often fall into the **"FinTech Winner's Curse"**—aggressively acquiring high-intent users who are subsequently rejected by backend risk controls. This structural misalignment (the "Risk Cliff") leads to severe budget exhaustion and Target CPA constraint violations.

**LAGO** is a novel *neuro-symbolic* auto-bidding framework designed to bridge the gap between mathematical optimality and complex industrial risk management. It structurally circumvents the Winner's Curse without relying on hard-coded heuristics.

### ✨ Key Features

* **Dual-Axis Valuation Grid:** Geometrically decouples frontend borrowing intent from posterior credit qualification, resolving unidimensional feature collapse.
* **Elasticity-Driven OR Solver:** Employs primal-dual gradient descent under Lagrangian dynamics to mathematically execute "steep truncation" on toxic traffic quadrants.
* **LLM Agentic Guardrail:** Integrates Large Language Models (LLMs) via an **Axiomatic Chain-of-Thought (CoT)** paradigm to semantically audit bidding matrices and intercept out-of-distribution (OOD) market noise.

## 📂 Repository Structure

The codebase is refactored into an object-oriented, extensible simulation framework:

```text
├── run_simulation.py       # Main entry point for the offline simulation
├── README.md               # This documentation file
└── requirements.txt        # Python dependencies

```

## ⚙️ Installation

1. Clone this repository:
```bash
git clone [https://github.com/YourUsername/LAGO.git](https://github.com/YourUsername/LAGO.git)
cd LAGO

```


2. Install the required dependencies:
```bash
pip install -r requirements.txt

```



## 🚀 Quick Start

To reproduce the offline evaluation results (RQ1) from the paper, simply execute the main simulation script. You can customize the simulation using command-line arguments.

### Basic Execution

```bash
python run_simulation.py

```

### Advanced Configuration

You can control the scale of the simulation and the optimization constraints:

```bash
python run_simulation.py --samples 200000 --target_cpa 250.0 --epochs 20

```

* `--samples`: Number of simulated users (default: 100,000).
* `--target_cpa`: The global Target CPA constraint (default: 200.0).
* `--epochs`: Number of optimization iterations for the dynamic solvers (default: 15).

## 📊 Evaluation Metrics

The simulation evaluates agents across four critical FinTech dimensions:

1. **Actual CPA:** The true acquisition cost per valid credit (must satisfy `target_cpa`).
2. **Vol ($V_{\text{credit}}$):** The absolute number of acquired users who pass backend risk assessments.
3. **$R_{\text{approve}}$:** The Application Approval Rate (higher indicates better toxic traffic filtering).
4. **SCR (Spend-to-Credit Ratio):** Ad Spend divided by Granted Credit Line. A crucial metric for FinTech capital leverage and asset quality.

## 🧠 Algorithmic Implementations Included

The simulation framework includes the following agents (inheriting from a unified `BaseAgent`):

* `FlatBidAgent`: A traditional unidimensional baseline.
* `UnconstrainedDRLAgent`: Simulates aggressive volume acquisition without risk constraints.
* `PIDAgent`: A strong industry baseline using a proportional-integral-derivative controller bounded by Target CPA.
* `LAGOSolver`: **Our proposed OR engine** executing elasticity-driven truncation via Lagrangian dynamics on the Dual-Axis Grid.

## 📈 Sample Output

Upon successful execution, the script outputs the steady-state performance matrix, demonstrating LAGO's Pareto domination:

```text
============================================================
=== Final Steady-State Performance (Target CPA = 200.0) ===
============================================================
           Strategy  Actual CPA  Vol (V_credit) R_approve    SCR
Flat-Bid (Baseline)      186.89          2798.0     6.98%  1.36%
  Unconstrained DRL      176.57          4073.0     7.55%  1.24%
    PID-Constrained      183.28          3617.0     7.25%  1.29%
        LAGO (Ours)      154.95          4831.0     8.62%  0.99%
============================================================
```
Note: Due to the stochastic nature of the auction simulation (np.random.binomial and np.random.uniform), the exact absolute numbers may fluctuate slightly across different runs, Python versions, or hardware. However, the relative performance ranking and the Pareto dominance of LAGO remain strictly consistent.