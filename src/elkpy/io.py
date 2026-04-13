from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

@dataclass
class ProjectedBandSegment:
    x: np.ndarray
    energy: np.ndarray
    characters: np.ndarray
    # characters.shape = (nk, nchar)
    # raw character columns after x and energy

    @property
    def nchar(self) -> int:
        return self.characters.shape[1]

@dataclass
class ProjectedBandStructure:
    bands: list[ProjectedBandSegment]

    @property
    def nchar(self) -> int:
        if not self.bands:
            raise ValueError("ProjectedBandStructure contains no bands.")
        return self.bands[0].nchar

@dataclass(frozen=True)
class ProjectionSelection:
    bands: ProjectedBandStructure
    columns: tuple[int, ...]
    label: str

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

# To-do: this is only as good as the column assignment.
#        for something less brittle, the better long-term
#        move is to add a helper that reads ELMIREP.OUT, when I have time.

def read_projected_bands(file_path: str | Path) -> ProjectedBandStructure:
    """
    Read BAND_Sss_Aaaaa.OUT from Elk task 21 or 22.

    Expected columns:
        1: x
        2: energy relative to E_F
        3...: raw character channels

    No assumption is made that Elk writes a precomputed total weight
    as a separate column. The total can be reconstructed by summing
    the character channels.
    """

    blocks = _read_numeric_blocks(file_path, min_columns=3)

    bands = [
        ProjectedBandSegment(
            x=block[:, 0],
            energy=block[:, 1],
            characters=block[:, 2:],
        )
        for block in blocks
    ]
    return ProjectedBandStructure(bands=bands)

def _normalize_columns(
    nchar: int,
    columns: slice | Sequence[int],
) -> tuple[int, ...]:
    """
    Convert a slice or sequence into a validated tuple of column indices.
    """

    if isinstance(columns, slice):
        idx = tuple(range(*columns.indices(nchar)))
    else:
        idx = tuple(int(i) for i in columns)

    if not idx:
        raise ValueError("No projection columns selected.")

    for i in idx:
        if i < 0 or i >= nchar:
            raise IndexError(
                f"Projection column {i} out of range for nchar={nchar}."
            )

    return idx

def projection_sum(
    segment: ProjectedBandSegment,
    columns: slice | Sequence[int],
) -> np.ndarray:
    """
    Sum selected character columns for one band segment.
    """

    idx = _normalize_columns(segment.nchar, columns)
    return segment.characters[:, idx].sum(axis=1)

def validate_projected_mesh(
    ref: ProjectedBandStructure,
    other: ProjectedBandStructure,
    *,
    check_energy: bool = False,
) -> None:
    """
    Validate that two projected band datasets are compatible.

    By default, checks that band count and x-grids match.
    Optionally also checks energy arrays.
    """

    if len(ref.bands) != len(other.bands):
        raise ValueError(
            "Projected band files do not contain the same number of bands."
        )

    for iband, (a, b) in enumerate(zip(ref.bands, other.bands), start=1):
        if a.x.shape != b.x.shape or not np.allclose(a.x, b.x):
            raise ValueError(
                f"Band {iband}: x grids do not match between files."
            )

        if a.energy.shape != b.energy.shape:
            raise ValueError(
                f"Band {iband}: energy array shapes do not match."
            )

        if check_energy and not np.allclose(a.energy, b.energy):
            raise ValueError(
                f"Band {iband}: energy arrays do not match."
            )
