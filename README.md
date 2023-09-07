# LRGB: Long Range Graph Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2206.08164-b31b1b.svg)](https://arxiv.org/abs/2206.08164)

The repo is based on [LRGB](https://github.com/vijaydwivedi75/lrgb), it is built for DGL library.

<img src="https://i.imgur.com/2LKoGbu.png" align="right" width="275"/>



We present the **Long Range Graph Benchmark (LRGB)** with 5 graph learning datasets that arguably require
long-range reasoning to achieve strong performance in a given task. 
- PascalVOC-SP
- COCO-SP
- PCQM-Contact 
- Peptides-func
- Peptides-struct 

### Overview of Datasets

|  Dataset | Domain  |  Task | Node Feat. (dim)  | Edge Feat. (dim) | Perf. Metric | 
|---|---|---|---|---|---|
| PascalVOC-SP| Computer Vision | Node Prediction | Pixel + Coord (14) | Edge Weight (1 or 2) | macro F1 |
| COCO-SP | Computer Vision | Node Prediction | Pixel + Coord (14) | Edge Weight (1 or 2) | macro F1 |
| PCQM-Contact | Quantum Chemistry | Link Prediction | Atom Encoder (9) | Bond Encoder (3) | Hits@K, MRR
| Peptides-func | Chemistry | Graph Classification | Atom Encoder (9) | Bond Encoder (3) | AP
| Peptides-struct | Chemistry | Graph Regression | Atom Encoder (9) | Bond Encoder (3) | MAE |


### Statistics of Datasets

|  Dataset | # Graphs  |  # Nodes | μ Nodes  | μ Deg. | # Edges | μ Edges | μ Short. Path | μ Diameter 
|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| PascalVOC-SP| 11,355 | 5,443,545 | 479.40 | 5.65 | 30,777,444 | 2,710.48 | 10.74±0.51 | 27.62±2.13 |
| COCO-SP | 123,286 | 58,793,216 | 476.88 | 5.65 | 332,091,902 | 2,693.67 | 10.66±0.55 | 27.39±2.14 |
| PCQM-Contact | 529,434 | 15,955,687 | 30.14 | 2.03 | 32,341,644 | 61.09 |4.63±0.63 | 9.86±1.79 |
| Peptides-func | 15,535 | 2,344,859 | 150.94 | 2.04 | 4,773,974 | 307.30 | 20.89±9.79 | 56.99±28.72 |
| Peptides-struct | 15,535 | 2,344,859 | 150.94 | 2.04 | 4,773,974 | 307.30 | 20.89±9.79 | 56.99±28.72 |
