"""Large/small ('B'/'S') classification of the grains of a clog.

It uses the `tipo` column of the .xy files (the exact species of each grain,
i.e. the index of its grain type in params.in) instead of a radius threshold:
radius thresholds fail for mixtures of shapes, because the inradius or
circumradius of a polygon cannot be compared directly with the radius of a
disk.

Rule:
  - If chi == 0 or there is a single species in the clog: everything is 'B'.
  - Otherwise, the species with the largest median area is 'B' and the rest 'S'.
  - If the areas are equal (difference < 1 %, area ratio 1.0) there is no
    large or small species: everything is 'B'.

Usage from a script in analysis/<folder>/:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from especies import especie_por_tipo
"""
import numpy as np
import pandas as pd

TOL_AREA_RELATIVA = 0.01


def especie_por_tipo(tipos, areas, chi=None):
    """tipos, areas: one value per grain. Returns {tipo: 'B' | 'S'}."""
    df = pd.DataFrame({'tipo': np.asarray(tipos), 'area': np.asarray(areas, dtype=float)})
    if df.empty:
        return {}
    area_med = df.groupby('tipo')['area'].median()
    todo_B = {t: 'B' for t in area_med.index}
    if len(area_med) < 2 or (chi is not None and pd.notna(chi) and float(chi) == 0.0):
        return todo_B
    a_max = area_med.max()
    if (a_max - area_med.min()) / a_max < TOL_AREA_RELATIVA:
        return todo_B
    grande = area_med.idxmax()
    return {t: ('B' if t == grande else 'S') for t in area_med.index}


def area_grano(p):
    """Area of a grain from a particle dict with 'r_eff' (disks) or
    'coords' (polygons, vertices of shape (n, 2)).
    """
    coords = p.get('coords')
    if coords is None:
        return float(np.pi * p['r_eff'] ** 2)
    x, y = coords[:, 0], coords[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
