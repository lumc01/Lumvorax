import numpy as np


def normalize_state(psi: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(psi)
    return psi if norm <= 1e-15 else psi / norm


def advance_state(psi: np.ndarray, dt: float, force_fn) -> np.ndarray:
    psi_next = psi + dt * force_fn(psi)
    return normalize_state(psi_next)


def step_observables(psi: np.ndarray, hamiltonian_fn, pairing_fn, n_sites: int):
    energy = hamiltonian_fn(psi)
    pairing = pairing_fn(psi)
    if n_sites > 0:
        energy /= n_sites
        pairing /= n_sites
    return energy, pairing
