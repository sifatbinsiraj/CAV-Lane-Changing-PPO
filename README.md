# A Behavior-Aware, Lane-Semantic, and Geometry-Informed Multi-Agent Reinforcement Learning Framework for Cooperative Lane-Changing in CAVs

**Author:** Md Sifat Bin Siraj  
**Affiliation:** Southern Illinois University Edwardsville  
**Target Journal:** IEEE Transactions on Intelligent Transportation Systems (Q1)  
**Date:** April 2026

---

## Overview

This repository contains the official implementation of a three-component PPO framework for cooperative CAV lane-changing that simultaneously addresses three structural limitations in the existing literature:

- **N1 — Motivation-Aware Policy Disentanglement:** Separate π_DLC and π_MLC policy heads for discretionary and mandatory lane changes
- **N2 — Lane-Semantic Gate:** 3-dimensional binary encoding of lane position as a structural policy modulator
- **N3 — Geometry-Aware Safety Filter:** Yaw-aware bounding box overlap detection for collision-free lateral actions

**Key Results:**
- Lane-change success rate: **0.745**
- Collision rate: **0.0023**
- Collision reduction in high-density (S3): **66%** (p < 0.001)

---

## Dataset

This study uses the **CitySim FreewayC** naturalistic trajectory dataset collected on Interstate 4 (I-4 Freeway) in Orlando, Florida.

- **Source:** [UCF-SST CitySim Dataset](https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset)
- **Size:** ~6.7 million trajectory records, 25 Hz sampling
- **Lane-change events extracted:** 13,040 (DLC: 8,621 | MLC: 4,419)
- **Corridor:** 8-lane freeway, 3.5 km segment

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Phase 1: N1 only
python cav_lane_change_full.py --mode train --step 1

# Phase 2: N1 + N2
python cav_lane_change_full.py --mode train --step 2

# Phase 3: N1 + N2 + N3 (Full)
python cav_lane_change_full.py --mode train --step 3

# Evaluate
python cav_lane_change_full.py --mode evaluate

# Ablation
python cav_lane_change_full.py --mode ablation
```

---

## Results

| Model | LC Success (%) | Collision (%) |
|---|---|---|
| MOBIL | 72 ± 5 | 9 ± 2 |
| DQN | 82 ± 4 | 6 ± 2 |
| **N1+N2+N3 (ours)** | **91 ± 2** | **2 ± 1** |

---

## Citation

```bibtex
@article{mahmud2026cav,
  title={A Behavior-Aware, Lane-Semantic, and Geometry-Informed Multi-Agent 
         Reinforcement Learning Framework for Cooperative Lane-Changing in CAVs},
  author={Mahmud, Saifullah},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  note={Under review}
}
```

---

## License

MIT License

## Acknowledgments

- CitySim dataset: Zheng et al. (2024), UCF SST Lab
- highway-env: Leurent (2018)
- PPO: Schulman et al. (2017)
