# sim-binary-public

Simulation and analysis code for

> J. Montero, L. A. Pugnaloni and C. M. Carlevaro,
> *Effect of small circular grains on the clogging dynamics of polygonal particles*.

Two-dimensional DEM simulations (Box2D 2.4.2) of the discharge of a flat-bottom
silo filled with a binary mixture of large grains (disks, triangles, squares or
pentagons) and small disks, and the Python scripts that produce the figures of
the paper and of the Supplemental Material from the raw simulation output.

## Contents

```text
box2d/                     Box2D 2.4.2 (physics engine)
simulation/                Simulator (C++): make -> bin/eo-silo
  src/eo-silo.cpp          main loop: discharge, clog detection, arch removal
  src/jam_output.cpp       contacts and per-grain stress tensor of each clog
run_scripts/               parameter template and launch scripts
analysis/
  rutas.py                 paths (see "Where data are read and written")
  estilo_figuras.py        common figure style
  especies.py              large/small grain classification
  1_avalanchas_y_arcos/    avalanches, arches, segregation, flow statistics
  2_fuerzas/               clog snapshot with the contact-force network
  3_presion_y_mapas/       per-grain stress tensor, pressure in the arch and ROI
```

## 1. Build the simulator

Requirements: a C++17 compiler, CMake and GSL (`libgsl-dev` on Debian/Ubuntu,
`gsl-devel` on Fedora).

```bash
cd box2d && ./build.sh && cd ..          # -> box2d/build/bin/libbox2d.a
cd simulation && make && cd ..           # -> simulation/bin/eo-silo
```

## 2. Run the simulations

Each simulation is a folder with a `params.in` file (see
`run_scripts/params_template.in` for the meaning of every parameter):

```bash
cd /path/to/simulation_data/BARRIDO_X/Ratio_0p1_Chi_0p2_W_2p0
/path/to/simulation/bin/eo-silo params.in > sim.log
```

`run_scripts/run_local.sh` creates the folders and `params.in` files of a sweep
(large-grain shape, concentration `chi`, area ratio `R_a`) and runs them.
The analysis reads `chi`, `R_a` and the orifice from the folder name, so keep
the naming `Ratio_<Ra>_Chi_<chi>_W_<W>` (with `p` as decimal point, e.g.
`Ratio_0p1_Chi_0p2_W_2p0`); pure systems use `Chi_0p0_W_<W>`.

Parameters of the paper: silo width 20.5 d, height 120 d, orifice radius
`Radio_salida = 2.0`, total mass equal to that of 2000 large grains, friction
0.5, restitution 0.1, time step 0.001 s, clog waiting time
`tEsperaAtasco = 5` s and at least 400 avalanches per run (`maxAvalanchas`).

Each run writes `flx.dat` (time and grain type of every grain that leaves the
silo) and, inside `frames_<dirID>/`, one set of files per clog `k`:

| file | content |
|---|---|
| `atasco_<k>.xy` | positions and shapes of all the grains |
| `atasco_contacts_<k>.dat` | contact points and impulses |
| `atasco_contacts_full_<k>.dat` | the same, with the Box2D normals and forces |
| `atasco_tensor_<k>.dat` | stress tensor of each grain computed in the simulation |

## 3. Analysis

Python 3.9+ with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Where data are read and written

All paths come from `analysis/rutas.py`:

- `SIM_DATA_DIR`: folder(s) with the simulations (several separated by `:`).
  Default: `datos_simulaciones/simulation_data/R_2.0` inside the repository.
- `RESULTADOS_DIR`: where `master_sim_index.csv` and `processed_data/` are
  written. Default: the repository.

```bash
export SIM_DATA_DIR=/path/to/simulation_data
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

## Notes

- `s` (avalanche size) is the number of **large** grains discharged between
  two consecutive clogs; small grains are not counted.
- The error bars of <s> are standard errors over the avalanches of a single
  long run; `correlacion_avalanchas_formas.py` checks that consecutive
  avalanches are uncorrelated.
- Variable, column and file names in the code are in Spanish: `atasco` = clog,
  `arco` = arch, `grande`/`chico` = large/small, `tamano_big` = number of large
  grains in an avalanche, `presion` = pressure, `forma` = shape.
