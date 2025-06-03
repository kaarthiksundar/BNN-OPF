# BNN-OPF: Bayesian Neural Networks for Optimal Power Flow

## License
BNN-OPF is provided under a BSD-3 license as part of the Optimization and Machine Learning Toolbox project, O4806.
See [LICENSE.md](https://github.com/lanl-ansi/MathOptAI.jl/blob/main/LICENSE.md) for details.

## Information 
This repository contains the source code for the paper:
**"Optimization Proxies using Limited Labeled Data and Training Time – A Semi-Supervised Bayesian Neural Network Approach"**  
Parikshit Pareek, Abhijith Jayakumar, Kaarthik Sundar, Deepjyoti Deka, Sidhant Misra  
[arXiv:2410.03085](https://arxiv.org/pdf/2410.03085)

---

## 📘 Overview

The BNN-OPF framework addresses the challenge of learning optimization proxies for constrained problems like Optimal Power Flow (OPF) under limited labeled data and training time. It employs a semi-supervised Bayesian Neural Network (BNN) that alternates between supervised learning (minimizing cost) and unsupervised learning (enforcing constraint feasibility). This approach provides probabilistic confidence bounds on performance, making it suitable for safety-critical applications.

---

## 🗂️ Repository Structure

- `src/`: Core implementation of the BNN model and training routines.
- `scripts/`: Scripts for training, evaluation, and plotting.
- `data/`: Contains datasets used for training and testing.
- `logs/`: Training logs and checkpoints.
- `plots/`: Generated plots from experiments.
- `output/`: Model outputs and evaluation results.
- `main.py`: Entry point for training and evaluation with configurable CLI arguments.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kaarthiksundar/BNN-OPF.git
   cd BNN-OPF
   ```

2. Install dependencies
  ```bash 
   poetry install 
  ```

## ⚙️ CLI Arguments in main.py 

Run the command for details on the CLI arguments. 
```python 
python main.py --help 
``` 

📬 Contact

For questions or collaborations, please reach out to:

Parikshit Pareek: pareek@ee.iitr.ac.in
