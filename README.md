# elkpy

`elkpy` is a small Python package for reading and working with output files from the [Elk](http://elk.sourceforge.net/) all-electron DFT code.

It aims to make it easier to load Elk outputs into Python for analysis, plotting, and post-processing.

The first version of `elkpy` includes small set of readers:

- `EPSILON_*.OUT`
- `BAND.OUT`
- `BANDLINES.OUT`

The functions of `elkpy` is planned to be extended over time.

## Installation

Clone the repository and install it in editable mode:

```bash
pip install -e .
````

## Requirements

* Python 3.10 or newer
* NumPy

## Usage

### Read EPSILON outputs (dielectric function)

```python
from elkpy import read_epsilon

eps = read_epsilon("EPSILON_11.OUT")

print(eps.omega_real)
print(eps.eps_real)
print(eps.omega_imag)
print(eps.eps_imag)
```

`read_epsilon()` expects an `EPSILON_*.OUT` file with two data blocks:

1. real part
2. imaginary part

It returns an `EpsilonData` object.

### Read band structure

```python
from elkpy import read_bands

bands = read_bands("BAND.OUT")

for band in bands.bands:
    print(band.x)
    print(band.energy)
```

`read_bands()` reads `BAND.OUT` and returns a `BandStructure` object containing one `BandSegment` per band.

### Read band line positions

```python
from elkpy import read_bandlines

lines = read_bandlines("BANDLINES.OUT")

for line in lines.lines:
    print(line)
```

`read_bandlines()` reads `BANDLINES.OUT` and returns a `BandLines` object.

## Returned data structures

### `EpsilonData`

Holds dielectric function data:

* `omega_real`
* `eps_real`
* `omega_imag`
* `eps_imag`

### `BandSegment`

Represents one band:

* `x`
* `energy`

### `BandStructure`

Container for a list of band segments:

* `bands`

### `BandLines`

Container for band path marker lines:

* `lines`

## Note 

`elkpy` is currently focused on small set of readers for a few common Elk output files. It is in still early-stage and the interface may change as more functionality is added.

## Contributing

This project is still small and under active development. Bug reports, suggestions, and improvements are very welcome.

## License

MIT License.
