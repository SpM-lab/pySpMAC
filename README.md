# pyspmac

## usage

Python 3.8 or higher is required.

``` bash
# install pyspmac
python3 -m pip install .

# run a sample
cd sample/multiple
spmac input.toml
```

## parameters

The names of parameters are case-insensitive.

- output: str
    - Output directory name (default: output)
- num_flavor: int
    - Number of flavors (orbitals)
- filein_g: str
    - Filename storing G(τ)
    - When `num_flavor > 1`, "{filein_g}.{a}_{b}" are read for G_{ab}, where a,b = 0,1,...,num_flavor-1
- column: int
    - Index of column storing G(τ) (0-origin)
- beta: float
    - Inverse temperature β
- max_omega: float
    - Upper bound of ω
- min_omega: float
    - Lower bound of ω
- num_omega: int
    - Number of ωs
- nonnegative: bool
    - Impose non-negativity (single flavor) or semi-positive definiteness (multiple flavor) (default:true)
- sumrule: bool
    - Impose sum-rule (default: true)
- min_sv: float
    - Cutoff in singular value (default: 1e-10)
- max_iteration: int
    - Maximum number of iterations of ADMM (default: 1000)
- optimize: bool
    - Optimize λ by the elbow method (default: false)
- max_loglambda: float
    - Maximum value of log10(λ)
    - If optimize
- min_loglambda: float
    - Minimum value of log10(λ)
    - If optimize
- loglambda: float
    - log10(λ)
    - If not optimize

## small test

Run `pytest` at the root directory:

``` bash
python3 -m pytest
```

## Authors

- Yuichi Motoyama
- Hiroshi Shinaoka

## Paper

- Sparse Modeling Analytic Continuation
    - Junya Otsuki, Masayuki Ohzeki, Hiroshi Shinaoka, and Kazuyoshi Yoshimi, "Sparse modeling approach to analytical continuation of imaginary-time quantum Monte Carlo data", [Phys. Rev. E 95, 061302(R)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.061302).
- SpM AC + Pade approximation
    - Yuichi Motoyama, Kazuyoshi Yoshimi, and Junya Otsuki, "Robust analytic continuation combining the advantages of the sparse modeling approach and the Padé approximation", [Phys. Rev. B 105, 035139](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.105.035139).
- SpM AC for multi-orbital data
    - Yuichi Motoyama, Hiroshi Shinaoka, Junya Otsuki, and Kazuyoshi Yoshimi, "Robust analytic continuation using sparse modeling approach imposed by semi-positive definiteness for multi-orbital systems", [arXiv:2409.01509](https://arxiv.org/abs/2409.01509).

## License

PySpMAC is distributed under the Mozilla Public License 2.0
