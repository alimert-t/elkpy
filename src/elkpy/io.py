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

@dataclass
class TDOSBlock:
    energy: np.ndarray
    tdos: np.ndarray


@dataclass
class TDOSData:
    blocks: list[TDOSBlock]

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> TDOSBlock:
        return self.blocks[idx]

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

def read_tdos(file_path: str | Path) -> TDOSData:
    """
    Reads TDOS.OUT from Elk.

    Elk can write multiple TDOS blocks separated by blank lines,
    for example positive/negative spin or SOC-resolved channels.
    Each block is returned as a TDOSBlock with:

        block.energy
        block.tdos
    """

    blocks = _read_numeric_blocks(file_path, min_columns=2)

    if not blocks:
        raise ValueError(f"{file_path}: no TDOS data blocks found.")

    tdos_blocks = [
        TDOSBlock(
            energy=block[:, 0],
            tdos=block[:, 1],
        )
        for block in blocks
    ]

    return TDOSData(blocks=tdos_blocks)

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

def total_character(segment: ProjectedBandSegment) -> np.ndarray:
    """
    Sum all character channels for one band segment.
    """

    return segment.characters.sum(axis=1)

def validate_projected_mesh(
    ref: ProjectedBandStructure,
    other: ProjectedBandStructure,
    *,
    check_energy: bool = False,
) -> None:
    """
    Validate that two projected band datasets are compatible.

    By default, checks:
        - same number of bands
        - same x-grid in each band
        - same number of character columns

    Optionally also checks energy arrays.
    """

    if len(ref.bands) != len(other.bands):
        raise ValueError(
            "Projected band files do not contain the same number of bands."
        )

    for iband, (a, b) in enumerate(zip(ref.bands, other.bands), start=1):
        if a.nchar != b.nchar:
            raise ValueError(
                f"Band {iband}: different numbers of character columns "
                f"({a.nchar} vs {b.nchar})."
            )

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

def combine_projected_bands(
    parts: Sequence[ProjectedBandStructure],
    *,
    check_energy: bool = True,
) -> ProjectedBandStructure:
    """
    Sum several projected band datasets channel-by-channel.

    Typical use:
        combine the two Se atoms into one Se-projected band structure.
    """

    if not parts:
        raise ValueError("No projected band datasets were provided.")

    ref = parts[0]
    for other in parts[1:]:
        validate_projected_mesh(ref, other, check_energy=check_energy)

    out_bands: list[ProjectedBandSegment] = []
    for iband in range(len(ref.bands)):
        x = ref.bands[iband].x
        energy = ref.bands[iband].energy
        chars = np.zeros_like(ref.bands[iband].characters)

        for part in parts:
            chars += part.bands[iband].characters

        out_bands.append(
            ProjectedBandSegment(
                x=x,
                energy=energy,
                characters=chars,
            )
        )

    return ProjectedBandStructure(bands=out_bands)


def infer_orbital_slices(nchar: int) -> dict[str, slice]:
    """
    Infer contiguous angular-momentum channel slices.

    Assumes the character columns are ordered as
        s (1), p (3), d (5), f (7), g (9), ...

    Example:
        nchar = 16  ->  s: [0:1], p: [1:4], d: [4:9], f: [9:16]
    """

    labels = ["s", "p", "d", "f", "g", "h", "i", "j"]
    out: dict[str, slice] = {}

    start = 0
    l = 0
    remaining = nchar

    while remaining > 0:
        width = 2 * l + 1
        if remaining < width:
            raise ValueError(
                f"Cannot infer orbital slices from nchar={nchar}. "
                "It is not a complete sum of odd-number blocks."
            )
        if l >= len(labels):
            raise ValueError(
                "Too many angular-momentum channels for built-in labels."
            )

        out[labels[l]] = slice(start, start + width)
        start += width
        remaining -= width
        l += 1

    return out


def select_projection(
    bands: ProjectedBandStructure,
    columns: slice | Sequence[int],
    label: str,
) -> ProjectionSelection:
    """
    Create a named projection selection.
    """

    idx = _normalize_columns(bands.nchar, columns)
    return ProjectionSelection(bands=bands, columns=idx, label=label)


def select_orbital(
    bands: ProjectedBandStructure,
    orbital: str,
) -> ProjectionSelection:
    """
    Select a whole orbital channel, e.g. 's', 'p', 'd', or 'f'.

    The mapping is inferred from the number of character columns.
    """

    slices = infer_orbital_slices(bands.nchar)

    if orbital not in slices:
        available = ", ".join(slices.keys())
        raise KeyError(
            f"Orbital {orbital!r} not available. Available: {available}"
        )

    idx = _normalize_columns(bands.nchar, slices[orbital])
    return ProjectionSelection(bands=bands, columns=idx, label=orbital)


def selected_weights(selection: ProjectionSelection) -> list[np.ndarray]:
    """
    Return one weight array per band for a selection.
    """

    return [
        projection_sum(segment, selection.columns)
        for segment in selection.bands.bands
    ]


def total_weights(bands: ProjectedBandStructure) -> list[np.ndarray]:
    """
    Return one total-character array per band.
    """

    return [total_character(segment) for segment in bands.bands]


def dominant_orbital_labels(
    bands: ProjectedBandStructure,
) -> list[np.ndarray]:
    """
    For each band, return the dominant orbital label at each k-point.
    """

    slices = infer_orbital_slices(bands.nchar)
    labels = tuple(slices.keys())

    out: list[np.ndarray] = []
    for segment in bands.bands:
        weights = np.column_stack(
            [projection_sum(segment, slices[label]) for label in labels]
        )
        argmax = np.argmax(weights, axis=1)
        out.append(np.array([labels[i] for i in argmax], dtype=object))

    return out
