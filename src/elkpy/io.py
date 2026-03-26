from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass 
class EpsilonData:
    omega_real: np.ndarray
    eps_real: np.ndarray 
    omega_imag: np.ndarray 
    eps_imag: np.ndarray 

@dataclass
class BandSegment:
    x: np.ndarray
    energy: np.ndarray 

@dataclass
class BandStructure:
    bands: list[BandSegment]

@dataclass
class BandLines:
    lines: list[np.ndarray]

def _read_numeric_blocks(
        filepath: str | Path, min_columns: int = 2
        ) -> list[np.ndarray]:
    """
    Read numeric data blocks of outputs file 
    separated by blank lines. 
    Each block is returned as individual 2D 
    numpy arrays.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    blocks = []
    current = []

    with filepath.open("r") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped:
                if current:
                    blocks.append(np.array(current, dtype=float))
                    current = []
                continue

            parts = stripped.split()
            if len(parts) < min_columns:
                raise ValueError(
                        f"{filepath}:{lineno}: expected at least {min_columns} columns, got {len(parts)}."
                )

            try: 
                row = [float(x) for x in parts]
            except ValueError as e:
                raise ValueError(f"{filepath}:{lineno}: could not parse floats.") from e 

            current.append(row)

        if current:
            blocks.append(np.array(current, dtype=float))

    return blocks

def read_epsilon(file_path: str | Path) -> EpsilonData:
    """
    Reads EPSILON_*.OUT output from Elk tasks. 
    Expects two blocks in EPSILON_*.OUT file:
        1. Real part 
        2. Imaginary part
    """
    
    blocks = _read_numeric_blocks(file_path, min_columns=2)

    if len(blocks) != 2:
        raise ValueError(f"{file_path}: expected 2 data blocks, found {len(blocks)}.")

    real_block, imag_block = blocks 

    return EpsilonData(
            omega_real = real_block[:,0],
            eps_real = real_block[:,1],
            omega_imag = imag_block[:,0],
            eps_imag = imag_block[:,1],
            )

def read_bandlines(file_path: str | Path) -> BandLines:
    """
    Reads BANDLINES.OUT.
    """

    blocks = _read_numeric_blocks(file_path, min_columns=2)
    
    return BandLines(lines=blocks)

def read_bands(file_path: str | Path) -> BandStructure:
    """
    Reads BAND.OUT. 
    Returns one BandSegment per band. 
    """
    
    blocks = _read_numeric_blocks(file_path, min_columns=2)
    
    bands = [
            BandSegment(x=block[:,0], energy=block[:,1])
            for block in blocks
        ]

    return BandStructure(bands=bands)
