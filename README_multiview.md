# Multiview with AMINE (Active Module Identification through Network Embedding)

**Active Module Identification through Node Embedding (AMINE)**  
A method for detecting relevant subnetworks from biological interaction networks.

AMINE is a flexible and efficient method for detecting subnetworks (modules) that are relevant with respect to a given biological signal, e.g., p-values associated with gene expression data. 
---

## Multiview Extension (Added in this Fork)

This fork introduces a *multiview* extension to the original AMINE framework. It enables the integration of multiple complementary graph views to improve the detection of active modules in biological networks.

###  Key Features:
- Support for **dual-view embeddings**, where:
  - **View 1** = original PPI topology
  - **View 2** = graph reconstructed from biological signals (e.g., gene expression p-values)
- Ability to compare, combine, and cluster embeddings from both views
- Flexible fusion strategies: `union`, `intersection`, or `average`
- Modular design for extending to more than two views

This extension follows the principles of SIMBA but allows explicit graph-level construction for each view before combining them.

---

## Installation

### Virtual environemet installation (recommended environment setup)

We recommend using a virtual environment for isolation. Here are two options:

#### Option 1: Using Python built-in `venv`

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
python3 -m venv multiview-env
source multiview-env/bin/activate
```

</details>

<details>
<summary><strong>Windows (CMD)</strong></summary>

```bash
python -m venv multiview-env
multiview-env\Scripts\activate
```

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```bash 
python -m venv multiview-env
multiview-env\Scripts\Activate.ps1

```

</details>

#### Option 2: Using `conda` (recommended)

```bash
conda create -n multiview-env python=3.10
conda activate multiview-env
```

---
### dependecies 
```bash
pip install -r requirements.txt
```


## Usage

### On artificial data

To run the AMINE algorithm on artificial data:

```bash
python -m amine.process_artificial_network
```

### On real biological networks

Use:

```bash
python -m amine.process_real_network
```

Input format is described in the `data/` directory.

### On multiview data (new)

To run AMINE with the multiview extension:

```bash
python -m amine.process_artificial_network --multiview --views 2 --fusion union
```

This executes module detection over two constructed graph views and fuses the neighborhood similarities using the specified strategy.

Available fusion modes:
- `union`: merge neighbor lists from both views
- `inter`: keep only shared neighbors
- `average`: combine similarity scores by averaging

---

## Directory Structure

```
amine/
├── datasets.py                   # Synthetic and real data loading
├── models.py                     # Node2Vec and multiview models
├── module_detection.py           # Core detection logic
├── process_artificial_network.py # CLI for artificial graphs
├── process_real_network.py       # CLI for real data
├── dimension_reduction/          # Node2vec implementation for Amine without Multiview 
└── graph_generation/             # Graph construction utilities
```

---

## Citation

If you use this software in your research, please cite the original AMINE publication:

Pasquier, C. (2020). **AMINE: Active Module Identification through Network Embedding**. Bioinformatics.

---

## License

MIT License.

---

## Acknowledgments

This project is a fork of [claudepasquier/amine](https://github.com/claudepasquier/amine),  
with multiview support added by [your name or GitHub profile].
