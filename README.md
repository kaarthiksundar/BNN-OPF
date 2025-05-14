# BNN-OPF
Bayesian Neural Networks for the Optimal Power Flow Problem

#DC3 dataset

\[
\begin{aligned}
\min_{y\in\mathbb{R}^{n}}\quad & \tfrac12\,y^{\!\top} Q\,y \;+\; p^{\!\top}\!\sin(y) \\
\text{s.t.}\quad & A\,y \;=\; x, \\
                 & G\,y \;\le\; h.
\end{aligned}
\]

Data generation can be found in `dc3data/datagen'

