# Mapping Uncertainty Using Differentiable Programming

Welcome to the repository supporting the paper **"Mapping Uncertainty Using Differentiable Programming."** This repository contains all the necessary code and data to reproduce the case studies and results presented in the manuscript.

## 📂 Repository Contents

Below is an overview of the files included in this repository:

### 🔹 Models & Simulations
- **[`DMA_MR_ss.py`](DMA_MR_ss.py)** – Python script for the steady-state model (ODE system) of the **Membrane Reactor System (DMA-MR)**.
- **[`cstr_example.ipynb`](cstr_example.ipynb)** – Jupyter Notebook demonstrating a **Continuous Stirred Tank Reactor (CSTR) example/case study**.
- **[`dma_mr_example_forward.ipynb`](dma_mr_example_forward.ipynb)** – Jupyter Notebook showcasing the **forward uncertainty propagation** case study for the **DMA-MR system**.
- **[`dma_mr_inverse_uncertainty_mapping.py`](dma_mr_inverse_uncertainty_mapping.py)** – Python script implementing the **inverse uncertainty propagation** for the DMA-MR case study.

### 📊 Data & Visualization
- **[`montecarlo.npz`](montecarlo.npz)** – NumPy data file containing results from **10,000 Monte Carlo simulations**, used for comparison with the proposed approach.
- **[`generate_plots_dma_mr.py`](generate_plots_dma_mr.py)** – Python script for **generating all figures** for the DMA-MR inverse uncertainty propagation case study using the Monte Carlo data.
- **[`time_montecarlo_dma_mr.png`](time_montecarlo_dma_mr.png)** – Screenshot of the **computational time required** to run all **10,000 MC simulations**, executed on a **Lenovo Legion 2021 (32GB RAM, Ryzen 7 5800H, Windows 11)**.

## 🔧 Getting Started
To run the models and generate results, ensure you have the required Python dependencies installed. You can set up the environment using:

```sh
pip install -r requirements.txt  
```


## 📜 Usage
- Open **Jupyter notebooks** (`.ipynb` files) to explore forward and inverse uncertainty propagation.
- Run or import the functions from the **Python scripts** (`.py` files) to simulate models and generate plots.

## 📖 Citation
If you use this code in your research, please cite our paper:
> **"Mapping Uncertainty Using Differentiable Programming"**

*(Full citation details will be added here once available)*

## 🤝 Contributing
Feel free to open issues or submit pull requests if you find any bugs or improvements!

---
📌 **Maintainer:** *Victor Alves*  
📧 Contact: *[GitHub](https://github.com/victoraalves)*
