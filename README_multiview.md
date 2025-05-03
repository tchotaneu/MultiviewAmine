# Multiview with AMINE (Active Module Identification through Network Embedding)

## About AMINE

AMINE (Active Module Identification through Node Embedding) is a method designed to identify biologically relevant subnetworks in large interaction graphs. It combines network topology and biological signal (e.g., p-values) using node embeddings and a clustering strategy.

---

## Multiview Extension (Added in this Fork)

### Why Multiview?

This fork introduces a *multiview* extension to the original AMINE framework.  
Rather than embedding biological signals directly into the interaction network topology, this version treats them as an independent source of information, creating a separate graph view from the signal.




### Key Concepts:

- Support for **dual-view embeddings**, where:
  - **View 1** preserves the original protein-protein interaction (PPI) topology.
  - **View 2** is constructed based solely on biological signals (e.g., p-values), where edges reflect functional similarity or biological coherence between nodes.
- Ability to compare, combine, and cluster embeddings from both views
- Flexible fusion strategies: `union`, `intersection`, or `ponderation`
- Modular design for extending to more than two views

This multiview modeling better captures the complexity of biological systems, where multiple, overlapping biological processes can coexist and interact.  
By decoupling structure and signal, this approach enables more flexible, modular, and biologically meaningful detection of active modules.


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
python -m amine.process_artificial_network --model multiview --views 2 --fusion union
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
