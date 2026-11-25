"""Project paths, computed from the location of this file (analysis/rutas.py),
so the scripts work wherever the repository is cloned.

Usage from a script in analysis/<folder>/:

    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    import rutas

    pd.read_csv(rutas.MASTER_INDEX)

Raw simulation data may live outside the repository: in that case set the
environment variable SIM_DATA_DIR before running, e.g.:

    export SIM_DATA_DIR=/path/to/simulation_data

Several folders can be given, separated by ':'.

The same applies to the results (master_sim_index.csv and processed_data/):
if RESULTADOS_DIR is set, they are read from and written to that folder
instead of the repository:

    export RESULTADOS_DIR=/path/to/results
      -> $RESULTADOS_DIR/master_sim_index.csv
      -> $RESULTADOS_DIR/processed_data/...
"""
import os

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SIM_DATA_DIRS = [d for d in os.environ.get(
    "SIM_DATA_DIR",
    os.path.join(REPO_DIR, "datos_simulaciones", "simulation_data", "R_2.0"),
).split(os.pathsep) if d]
SIM_DATA_DIR = SIM_DATA_DIRS[0]

RESULTADOS_DIR = os.path.abspath(os.environ.get("RESULTADOS_DIR", REPO_DIR))

MASTER_INDEX = os.path.join(RESULTADOS_DIR, "master_sim_index.csv")

PROCESSED_DIR = os.path.join(RESULTADOS_DIR, "processed_data")
AVALANCHE_DIR = os.path.join(PROCESSED_DIR, "1_avalanche_data")
SPATIAL_MAPS_DIR = os.path.join(PROCESSED_DIR, "2_spatial_maps")
ARCH_HIST_DIR = os.path.join(PROCESSED_DIR, "4_arch_histograms")
GRAFICOS_FINALES_DIR = os.path.join(PROCESSED_DIR, "5_graficos_finales")
FUERZAS_DIR = os.path.join(PROCESSED_DIR, "fuerzas")
PRESION_DIR = os.path.join(PROCESSED_DIR, "presion")
MAPAS_CALOR_DIR = os.path.join(PROCESSED_DIR, "mapas_calor")
