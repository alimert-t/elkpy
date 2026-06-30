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

def add_drude_term(
    eps_complex: np.ndarray,
    omega_ha: np.ndarray,
    omega_p_ev: float,
    gamma_ev: float,
) -> np.ndarray:
    """
    Add a phenomenological Drude contribution to a complex dielectric function.

    The added term is

        -omega_p^2 / [omega * (omega + i gamma)]

    with omega_p and gamma supplied in eV and converted internally to Eh.

    Parameters
    ----------
    eps_complex : np.ndarray
        Complex dielectric function.
    omega_ha : np.ndarray
        Frequency grid in Hartree.
    omega_p_ev : float
        Plasma frequency in eV.
    gamma_ev : float
        Broadening parameter in eV.

    Returns
    -------
    np.ndarray
        Complex dielectric function including the Drude term.
    """
    omega_p_ha = omega_p_ev / EH_TO_EV
    gamma_ha = gamma_ev / EH_TO_EV

    omega_ha = np.asarray(omega_ha, dtype=float)
    eps_complex = np.asarray(eps_complex, dtype=complex)

    drude = np.zeros_like(eps_complex, dtype=complex)

    mask = np.abs(omega_ha) > 1e-14
    drude[mask] = -omega_p_ha**2 / (
        omega_ha[mask] * (omega_ha[mask] + 1j * gamma_ha)
    )

    return eps_complex + drude
