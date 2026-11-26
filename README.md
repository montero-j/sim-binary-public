# sim-binary-public

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
<!-- Replace XXXXXXX with the Zenodo concept DOI once the first release is archived. -->

Simulation code, data-generation workflow and analysis scripts accompanying the
manuscript

> J. Montero, L. A. Pugnaloni and C. M. Carlevaro,
> *Effect of small circular grains on the clogging dynamics of polygonal particles*
> (submitted).

The repository contains a two-dimensional discrete-element (DEM) simulator,
built on [Box2D](https://box2d.org) 2.4.2, of the discharge of a flat-bottom
silo filled with a binary mixture of large grains (disks, triangles, squares or
pentagons) and small disks, together with the Python pipeline that turns the
raw simulation output into the quantities and figures reported in the paper
and its Supplemental Material.

## How to cite

If you use this code, please cite both the article and the archived version of
the software:

- **Article:** J. Montero, L. A. Pugnaloni and C. M. Carlevaro, *Effect of small
  circular grains on the clogging dynamics of polygonal particles* (submitted).
  <!-- Add journal, volume, pages and DOI once published. -->
- **Software:** J. Montero, L. A. Pugnaloni and C. M. Carlevaro,
  *sim-binary-public*, Zenodo, https://doi.org/10.5281/zenodo.XXXXXXX.

Citation metadata are also provided in [`CITATION.cff`](CITATION.cff) (GitHub
shows a "Cite this repository" button) and [`.zenodo.json`](.zenodo.json).

## Repository layout

```text
box2d/                     Box2D 2.4.2 physics engine (unmodified, MIT license)
simulation/                DEM simulator (C++17): `make` -> bin/eo-silo
  src/eo-silo.cpp          main loop: filling, discharge, clog detection, arch removal
  src/globalSetup.*        reading of params.in
  src/siloAux.*            silo geometry, grain creation, auxiliary routines
  src/jam_output.*         contacts and per-grain stress tensor of every clog
run_scripts/
  params_template.in       commented template of the input file
  run_local.sh             creates and runs a parameter sweep on a local machine
  preparar_video.sh        re-runs a simulation and renders a movie (needs external renderer)
analysis/
  rutas.py                 input/output paths (see "Where data are read and written")
  estilo_figuras.py        common figure style
  especies.py              large/small grain classification
  1_avalanchas_y_arcos/    avalanches, arches, segregation, flow statistics
  2_fuerzas/               clog snapshot with the contact-force network
  3_presion_y_mapas/       per-grain stress tensor, pressure in the arch and ROI
requirements.txt           Python dependencies of the analysis
```

## Workflow overview

```text
params_template.in ──run_local.sh──► <sweep>/Ratio_<Ra>_Chi_<chi>_W_<W>/params.in
                                              │  eo-silo
                                              ▼
                          flx.dat + frames_<dirID>/atasco_*.{xy,dat}   (raw data)
                                              │  analysis/1_.../main_pipeline.py
                                              ▼
                 master_sim_index.csv + processed_data/1_avalanche_data/*.csv
                                              │  analysis/3_.../presion_tensor_arch.py
                                              ▼
                       processed_data/presion/tensor_nuevo/*.{csv,parquet}
                                              │  plotting scripts
                                              ▼
                              processed_data/5_graficos_finales/*.pdf
```

## 1. Build the simulator

Requirements: a C++17 compiler (tested with GCC), CMake ≥ 3.8, GNU Make and the
[GNU Scientific Library](https://www.gnu.org/software/gsl/) (`libgsl-dev` on
Debian/Ubuntu, `gsl-devel` on Fedora, `gsl` on Homebrew).

```bash
cd box2d && ./build.sh && cd ..          # -> box2d/build/bin/libbox2d.a
cd simulation && make && cd ..           # -> simulation/bin/eo-silo
```

`build.sh` also builds the Box2D testbed and unit tests, which need OpenGL
development headers. On a headless machine build only the library:

```bash
cmake -S box2d -B box2d/build -DBOX2D_BUILD_TESTBED=OFF -DBOX2D_BUILD_UNIT_TESTS=OFF -DBOX2D_BUILD_DOCS=OFF
cmake --build box2d/build
```

## 2. Generate the simulation data

Each simulation lives in its own folder containing a `params.in` file. The
meaning of every parameter is documented in
[`run_scripts/params_template.in`](run_scripts/params_template.in). A single run is

```bash
cd /path/to/simulation_data/BARRIDO_TRIANGLE_DISK/Ratio_0p1_Chi_0p2_W_2p0
/path/to/sim-binary-public/simulation/bin/eo-silo params.in > sim.log
```

### Parameter sweeps

[`run_scripts/run_local.sh`](run_scripts/run_local.sh) builds the folders and
`params.in` files of a sweep and runs them in parallel. Edit the arrays at the
top of the script (`SHAPES_BIG`, `CHIS`, `RATIOS`, `WIDTHS`, `MAX_JOBS`) and run

```bash
OUT_DIR=/path/to/simulation_data bash run_scripts/run_local.sh
```

`EJECUTABLE` can be set to use a simulator binary other than
`simulation/bin/eo-silo`. A random seed is drawn for every run and stored in
its `params.in` (`RandomSeed`), so any run can be reproduced exactly from its
input file.

The analysis reads the small-grain concentration `chi`, the area ratio `R_a`
and the orifice from the **folder name**, so keep the naming
`Ratio_<Ra>_Chi_<chi>_W_<W>` (with `p` as decimal point, e.g.
`Ratio_0p1_Chi_0p2_W_2p0`). Pure systems use `Chi_0p0_W_<W>`.

### Parameters used in the paper

| quantity | value |
|---|---|
| silo width × height | 20.5 *d* × 120 *d* (`Radio_silo = 10.25`, `H = 120`) |
| orifice | `Radio_salida = 2.0` |
| total grain mass | equal to that of 2000 large grains (`N_REF = 2000`) |
| grain friction / restitution | 0.5 / 0.1 |
| wall friction / restitution | 0.5 / 0.05 |
| time step | 0.001 s (20 position and velocity iterations) |
| filling time | `tFill = 20` s |
| clog waiting time | `tEsperaAtasco = 5` s |
| avalanches per run | ≥ 400 (`maxAvalanchas`) |

<!-- TODO (authors): list the shapes, chi and R_a values of the sweeps in the paper. -->

### Raw output

Each run writes `flx.dat` (time and grain type of every grain that leaves the
silo) and, inside `frames_<dirID>/`, one set of files per clog `k`:

| file | content |
|---|---|
| `atasco_<k>.xy` | positions and shapes of all the grains |
| `atasco_contacts_<k>.dat` | contact points and impulses |
| `atasco_contacts_full_<k>.dat` | the same, with the Box2D normals and forces |
| `atasco_tensor_<k>.dat` | stress tensor of each grain computed in the simulation |

## 3. Analysis

Python ≥ 3.9 with the packages in `requirements.txt`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Where data are read and written

All paths come from [`analysis/rutas.py`](analysis/rutas.py):

- `SIM_DATA_DIR`: folder(s) with the simulations (several separated by `:`).
  Default: `datos_simulaciones/simulation_data/R_2.0` inside the repository.
- `RESULTADOS_DIR`: where `master_sim_index.csv` and `processed_data/` are
  written. Default: the repository root.

```bash
export SIM_DATA_DIR=/path/to/simulation_data
export RESULTADOS_DIR=/path/to/results     # optional
```

### Pipeline

1. **Index, avalanches, arches and flow statistics**

   ```bash
   python3 analysis/1_avalanchas_y_arcos/main_pipeline.py
   ```

   Writes `master_sim_index.csv` and, in `processed_data/1_avalanche_data/`,
   `resultados_completos_esqueleto.csv` (one row per avalanche),
   `estadisticas_flujo_agregadas.csv` (one row per configuration) and
   `resultados_completos_arcos.csv` (one row per arch).

2. **Stress tensor and pressure of each clog**

   ```bash
   python3 analysis/3_presion_y_mapas/presion_tensor_arch.py \
       --dir "$SIM_DATA_DIR" --master master_sim_index.csv \
       --out processed_data/presion/tensor_nuevo
   ```

   Writes `resumen_presion_por_atasco.csv` (one row per clog) and
   `presion_por_particula_<shape>.parquet`.

3. **Figures** (written to `processed_data/5_graficos_finales/`)

   ```bash
   python3 analysis/1_avalanchas_y_arcos/graficos_finales.py
   python3 analysis/3_presion_y_mapas/comparar_presion_por_chi.py \
       --csv processed_data/presion/tensor_nuevo/resumen_presion_por_atasco.csv \
       --magnitud P --out processed_data/5_graficos_finales/comparacion_chi_0p4.pdf
   python3 analysis/3_presion_y_mapas/correlacion_s_vs_P.py --s_en_x --logy
   python3 analysis/1_avalanchas_y_arcos/figuras_suplementario.py
   python3 analysis/1_avalanchas_y_arcos/correlacion_avalanchas_formas.py
   python3 analysis/1_avalanchas_y_arcos/ccdf_avalanchas.py
   ```

Every script accepts `--help`.

### Notes on the analysis

- `s` (avalanche size) is the number of **large** grains discharged between
  two consecutive clogs; small grains are not counted.
- The error bars of ⟨s⟩ are standard errors over the avalanches of a single
  long run; `correlacion_avalanchas_formas.py` checks that consecutive
  avalanches are uncorrelated.
- Variable, column and file names in the code are in Spanish: `atasco` = clog,
  `arco` = arch, `avalancha` = avalanche, `grande`/`chico` = large/small,
  `tamano_big` = number of large grains in an avalanche, `presion` = pressure,
  `forma` = shape, `barrido` = sweep, `mezcla` = mixture.

## Data availability

The raw simulation output is not stored in this repository because of its size.
<!-- TODO (authors): state where the raw/processed data are available
     (e.g. a Zenodo dataset DOI, or "available from the authors upon reasonable request"). -->
All of it can be regenerated with the workflow above from the `params.in`
files, which fix every parameter including the random seed.

## License

The code written for this project (`simulation/`, `analysis/`, `run_scripts/`)
is distributed under the MIT License, see [`LICENSE`](LICENSE).
Box2D (`box2d/`) is © Erin Catto and is redistributed unmodified under its own
MIT License, see [`box2d/LICENSE`](box2d/LICENSE).

## Contact

Questions about the code: open an
[issue](https://github.com/montero-j/sim-binary-public/issues) or write to the
corresponding author of the article.
