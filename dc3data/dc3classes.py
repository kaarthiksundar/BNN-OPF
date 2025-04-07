# problemdata.py
from dataclasses import dataclass
from typing import Dict, Any
import jax.numpy as jnp

@dataclass
class ProblemData:
    # ─── optimisation constants ────────────────────────────────────────────────
    Q: jnp.ndarray          # (n, n)
    A: jnp.ndarray          # (m, n)
    G: jnp.ndarray          # (k, n)
    p: jnp.ndarray          # (n,)
    h: jnp.ndarray          # (k,)

    # ─── data splits (already normalised / raw) ───────────────────────────────
    X_train_norm: jnp.ndarray   # (N_u, m)
    X_train:       jnp.ndarray  # (N_u, m)
    X_val_norm:           jnp.ndarray  # (N_v, m)
    X_val:                jnp.ndarray  # (N_v, m)
    Y_train: jnp.ndarray
    Y_val: jnp.ndarray


    # ─── normalisation statistics for Y (if you standardise outputs) ─────────
    Y_mean: jnp.ndarray    # (n,)
    Y_std:  jnp.ndarray    # (n,)

    # ─── training hyper‑params / meta info ───────────────────────────────────
    batch_size: int = 128
    meta: Dict[str, Any] = None        # see helper below

    # helper to build the default meta‑dict
    def init_meta(self, hidden_width: int = 128, num_hidden_layers: int = 2):
        n = self.Q.shape[0]
        self.meta = {
            "output_block_dim": {"y": n},   # one output group called "y"
            "hidden_width": hidden_width,
            "num_hidden_layers": num_hidden_layers,
        }
