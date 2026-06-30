from __future__ import annotations

import numpy as np

from .constants import EH_TO_EV, EH_TO_CMMINUS
from .io import EpsilonData

def _build_complex_imag_eps_grid(
        eps_obj: EpsilonData) -> tuple[np.ndarray, np.ndarray]:
    """
    Build complex epsilon on the omega grid of eps_obj.omega_imag.

    If the real and imaginary parts are not defined on identical grids,
    the real part is linearly interpolated onto the imaginary-part grid.

    Returns
    -------
    omega : np.ndarray
        Frequency grid in Hartree, taken from eps_obj.omega_imag.
    eps_complex : np.ndarray
        Complex dielectric function on that grid.
    """
    omega_imag = np.assaray(eps_obj.omega_imag, dtype=float)
    eps_imag = np.assaray(eps_obj.eps_imag, dtype=float)

    omega_real = np.assaray(eps_obj.omega_real, dtype=float)
    eps_real = np.assaray(eps_obj.eps_real, dtype=float)

    if omega_real.shape != omega_imag.shape or not np.allclose(
            omega_real, omega_imag):
        eps_real_interp = np.interp(omega_imag, omega_real, eps_real)
    else:
        eps_real_interp = eps_real

    eps_complex = eps_real_interp + 1j * eps_imag
    return omega_imag, eps_complex

