# Multiview with AMINE (Active Module Identification through Network Embedding)

## 1. About AMINE

AMINE (Active Module Identification through Node Embedding) is a method designed to identify biologically relevant subnetworks in large interaction graphs. It combines network topology and biological signal (e.g., p-values) using node embeddings and a clustering strategy.

---

## 2. Multiview Extension (Added in this Fork)

### 2.1 Why Multiview?

This fork introduces a *multiview* extension to the original AMINE framework.  
Rather than embedding biological signals directly into the interaction network topology, this version treats them as an independent source of information, creating a separate graph view from the signal.




### 2.2 Key Concepts:

- Support for **dual-view embeddings**, where:
  - **View 1** preserves the original protein-protein interaction (PPI) topology.
  - **View 2** is constructed based solely on biological signals (e.g., p-values), where edges reflect functional similarity or biological coherence between nodes.
- Ability to compare, combine, and cluster embeddings from both views
- Flexible fusion strategies: `union`, `intersection`, or `ponderation`
- Modular design for extending to more than two views

This multiview modeling better captures the complexity of biological systems, where multiple, overlapping biological processes can coexist and interact.  
By decoupling structure and signal, this approach enables more flexible, modular, and biologically meaningful detection of active modules.


---

## 3. Installation

### 3.1 Download repository

Clone the repository and change to the project directory
```bash
git clone https://github.com/tchotaneu/MultiviewAmine.git
cd MultiviewAmine
```
### 3.2 Virtual environemet installation (recommended environment setup)

We recommend using a virtual environment for isolation. Here are two options:

#### 3.2.1 Option 1: Using Python built-in `venv`

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

#### 3.2.2 Option 2: Using `conda` (recommended)

```bash
conda create -n multiview-env python=3.10
conda activate multiview-env
```

---
### 3.3 Dependecies 

```bash
pip install -r requirements.txt
```


## 4. Usage

### 4.1 On artificial data with multiview 

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

### 4.2 On real biological networks with multiview 

Use:

```bash
python -m amine.process_real_network
```
### On artificial data without multiview 

To run the AMINE algorithm on artificial data:

```bash
python -m amine.process_artificial_network
```

### 4.3  On real biological networks without multiview 

Use:

```bash
python -m amine.process_real_network
```
## 5. Citation

If you use this software in your research, please cite the following works:

- Tchotaneu, G. (2024). **Multiview Extension of Active Module Detection Methods for Biological Networks: Toward Enhanced Integration of Expression Signals and Network Topology**. Ongoing research project and internship report: *Integration of an Artificial Intelligence Approach for the Detection of Active Modules in Biological Interaction Networks through Multi-View Data to Assess the Role of the SigmaR1 Protein in Pancreatic Cancer*,  
  Laboratory of Computer Science, Signals and Systems of Sophia Antipolis (I3S), University Côte d’Azur, France.

- Pasquier, C. (2020). **AMINE: Active Module Identification through Network Embedding**. *Bioinformatics*.


---

## 6. License

MIT License.

---

## 7. Acknowledgments

This project is a fork of [claudepasquier/amine](https://github.com/claudepasquier/amine),  
with multiview support added by [Tchotaneu Ngatcha Giresse].
