#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import json
import ast
import numpy as np
import pandas as pd
import networkx as nx
import gc
import time
import collections
import multiprocessing as mp
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from especies import especie_por_tipo

ROI_EXTRA = 0.0

CHI_REGEX = re.compile(r'(?:Chi_|_X)([0-9]+(?:p[0-9]+)?)')

def poly_area(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def nombre_forma(n_sides):
    if pd.isna(n_sides): return "Ninguna"
    n = int(n_sides)
    if n == 1: return "Disco"
    if n == 3: return "Triangulo"
    if n == 4: return "Cuadrado"
    if n == 5: return "Pentagono"
    if n == 6: return "Hexagono"
    if n == 8: return "Octogono"
    return f"Poli{n}"

def parse_xy(filename):
    particles = {}
    with open(filename, "r") as f:
        for line in f:
            p = line.split()
            if not p: continue
            pid = int(p[0])
            if pid < 0: continue
            n_sides = int(p[1])
            shape_name = nombre_forma(n_sides)
            if n_sides == 1:
                x, y, r, ptype = np.float32(p[2]), np.float32(p[3]), np.float32(p[4]), int(p[5])
                area = np.pi * (r**2)
                particles[pid] = {"pos_x": x, "pos_y": y, "r_eff": r, "type": ptype, "shape": shape_name, "area": area, "coords": None}
            else:
                coords = np.array(p[2:-1], dtype=np.float32).reshape(-1, 2)
                center = coords.mean(axis=0)
                ptype = int(p[-1])
                r_eff = np.float32(np.max(np.linalg.norm(coords - center, axis=1)))
                area = poly_area(coords)
                particles[pid] = {"pos_x": center[0], "pos_y": center[1], "r_eff": r_eff, "type": ptype, "shape": shape_name, "area": area, "coords": coords}
    return particles

def parse_paredes(filename):
    """Segments of the silo walls and base (lines with id < 0 in the .xy file):
    'id n x1 y1 ... xn yn tipo' is an open chain of n vertices (or a single
    edge if n = 2). Returns an array (k, 2, 2) with the k consecutive segments
    (the chain is not closed).
    """
    segs = []
    with open(filename, "r") as f:
        for line in f:
            p = line.split()
            if len(p) < 2 or int(p[0]) >= 0:
                continue
            n = int(p[1])
            if n < 2 or len(p) < 2 + 2 * n:
                continue
            v = np.array(p[2:2 + 2 * n], dtype=np.float64).reshape(-1, 2)
            segs.extend(np.stack([v[:-1], v[1:]], axis=1))
    return np.array(segs, dtype=np.float64).reshape(-1, 2, 2)


def parse_radio_salida(sim_path, default=2.5):
    """Radio_salida from the params.in of the simulation folder, or the
    default value if the file or the field is missing.
    """
    params_file = os.path.join(sim_path, "params.in")
    r_salida = default
    try:
        with open(params_file, 'r') as f:
            for line in f:
                if 'Radio_salida:' in line:
                    r_salida = float(line.split(':')[1].split('#')[0].strip())
                    break
    except Exception:
        pass
    return r_salida


def parse_g(sim_path, default=9.8):
    """|g| from the params.in of the simulation folder, to express the pressure
    in reduced units (P/g). params.in stores g with a negative sign
    (e.g. 'g: -9.8').
    """
    params_file = os.path.join(sim_path, "params.in")
    g = default
    try:
        with open(params_file, 'r') as f:
            for line in f:
                if line.strip().startswith('g:'):
                    g = float(line.split(':')[1].split('#')[0].strip())
                    break
    except Exception:
        pass
    return abs(g) if g != 0 else default


def detectar_arco_primario(df_particles, df_contacts, r_salida):
    """Detects the clogging arch as the minimum-cost path between a grain resting
    on the base to the left of the orifice and one resting on the base to the
    right, with edge cost 10**y_max (which strongly favors the lowest paths),
    and then removes from both ends the grains that touch the base (they are
    foundations, not part of the bridge that holds the load).
    """
    if df_particles.empty or df_contacts.empty:
        return []

    valid_c = df_contacts[(df_contacts['id_a'] >= 0) & (df_contacts['id_b'] >= 0)]
    if valid_c.empty:
        return []
    pares = pd.DataFrame({
        'pair_a': np.minimum(valid_c['id_a'].to_numpy(), valid_c['id_b'].to_numpy()),
        'pair_b': np.maximum(valid_c['id_a'].to_numpy(), valid_c['id_b'].to_numpy()),
    }).drop_duplicates().sort_values(['pair_a', 'pair_b'])

    idx = df_particles.index
    pos_y = df_particles['pos_y']
    ya = pos_y.reindex(pares['pair_a'].to_numpy()).to_numpy(dtype=np.float64)
    yb = pos_y.reindex(pares['pair_b'].to_numpy()).to_numpy(dtype=np.float64)
    ok = ~np.isnan(ya) & ~np.isnan(yb)
    costos = [10.0 ** float(y) for y in np.maximum(ya[ok], yb[ok])]

    G = nx.Graph()
    G.add_nodes_from(idx)
    G.add_weighted_edges_from(zip(pares['pair_a'].to_numpy()[ok].tolist(),
                                  pares['pair_b'].to_numpy()[ok].tolist(), costos),
                              weight='costo_arco')

    r_col = 'r_eff'

    def es_cimiento_estable(pid, path, i):
        x = df_particles.at[pid, 'pos_x']
        y = df_particles.at[pid, 'pos_y']
        r = df_particles.at[pid, r_col]
        if abs(x) < (r_salida - 0.2):
            return False
        if (y - r) < 0.2:
            return True
        vecinos_arco = []
        if i > 0: vecinos_arco.append(path[i - 1])
        if i < len(path) - 1: vecinos_arco.append(path[i + 1])
        apoyos_inferiores = 0
        for vecino in G.neighbors(pid):
            if vecino not in vecinos_arco and df_particles.at[vecino, 'pos_y'] < (y - 0.05):
                apoyos_inferiores += 1
        return apoyos_inferiores >= 2

    def purgar_cimientos(path):
        if not path:
            return []
        iL = 0
        while iL < len(path) and es_cimiento_estable(path[iL], path, iL):
            iL += 1
        iR = len(path) - 1
        while iR >= iL and es_cimiento_estable(path[iR], path, iR):
            iR -= 1
        if iL > iR:
            return []
        return path[iL:iR + 1]

    apoyado = ((df_particles['pos_y'] - df_particles[r_col]) < 0.2).to_numpy()
    xs = df_particles['pos_x'].to_numpy()
    nleft = idx[apoyado & (xs < 0.0)].tolist()
    nright = idx[apoyado & (xs > 0.0)].tolist()


    mejor_path = None
    min_costo_total = float('inf')
    for l in nleft:
        dist, paths = nx.single_source_dijkstra(G, l, weight='costo_arco')
        for rr in nright:
            costo_path = dist.get(rr)
            if costo_path is not None and costo_path < min_costo_total:
                min_costo_total = costo_path
                mejor_path = paths[rr]

    return purgar_cimientos(mejor_path) if mejor_path is not None else []

def obtener_vector_normal_poly(row):
    px, py = row['px'], row['py']
    cx, cy = row['pos_x_A'], row['pos_y_A']
    coords = row['coords']
    n_vertices = len(coords)
    min_dist = float('inf')
    mejor_normal = (0.0, 0.0)
    for i in range(n_vertices):
        x1, y1 = coords[i]
        x2, y2 = coords[(i+1) % n_vertices]
        dx, dy = x2 - x1, y2 - y1
        l2 = dx*dx + dy*dy
        if l2 == 0: continue
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / l2))
        proy_x, proy_y = x1 + t * dx, y1 + t * dy
        dist2 = (px - proy_x)**2 + (py - proy_y)**2
        if dist2 < min_dist:
            min_dist = dist2
            nx_v, ny_v = -dy, dx
            norma = np.hypot(nx_v, ny_v)
            if norma > 0:
                nx_v, ny_v = nx_v/norma, ny_v/norma
            if (nx_v * (proy_x - cx) + ny_v * (proy_y - cy)) < 0:
                nx_v, ny_v = -nx_v, -ny_v
            mejor_normal = (nx_v, ny_v)
    return mejor_normal

def obtener_vector_normal_poly_lado(row, lado):
    """Same as obtener_vector_normal_poly, but it can read side A or B of a row
    already merged with suffixes (_A/_B), so that the appropriate side can be
    used when the other one is a wall (no geometry data).
    """
    px, py = row['px'], row['py']
    cx, cy = row[f'pos_x_{lado}'], row[f'pos_y_{lado}']
    coords = row[f'coords_{lado}']
    n_vertices = len(coords)
    min_dist = float('inf')
    mejor_normal = (0.0, 0.0)
    for i in range(n_vertices):
        x1, y1 = coords[i]
        x2, y2 = coords[(i+1) % n_vertices]
        dx, dy = x2 - x1, y2 - y1
        l2 = dx*dx + dy*dy
        if l2 == 0: continue
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / l2))
        proy_x, proy_y = x1 + t * dx, y1 + t * dy
        dist2 = (px - proy_x)**2 + (py - proy_y)**2
        if dist2 < min_dist:
            min_dist = dist2
            nx_v, ny_v = -dy, dx
            norma = np.hypot(nx_v, ny_v)
            if norma > 0:
                nx_v, ny_v = nx_v/norma, ny_v/norma
            if (nx_v * (proy_x - cx) + ny_v * (proy_y - cy)) < 0:
                nx_v, ny_v = -nx_v, -ny_v
            mejor_normal = (nx_v, ny_v)
    return mejor_normal


def normales_poligonos(px, py, cx, cy, coords, con_interior=False):
    """Vectorized version of obtener_vector_normal_poly_lado for many contacts:
    for each contact, finds the polygon edge closest to the contact point and
    returns its outward unit normal (projection clipped to the segment, first
    minimum, orientation from the center), using numpy over all the contacts
    of polygons with the same number of vertices.
    """
    m = len(px)
    out = np.zeros((m, 2))
    interior = np.zeros(m)
    if m == 0:
        return (out, interior) if con_interior else out
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    coords = list(coords)
    nverts = np.array([len(c) for c in coords])
    for n in np.unique(nverts):
        sel = np.flatnonzero(nverts == n)
        V = np.stack([np.asarray(coords[i], dtype=np.float64) for i in sel])
        d = np.roll(V, -1, axis=1) - V
        l2 = (d ** 2).sum(-1)
        P = np.stack([px[sel], py[sel]], -1)[:, None, :]
        with np.errstate(invalid='ignore', divide='ignore'):
            t = np.clip(((P - V) * d).sum(-1) / l2, 0.0, 1.0)
        proy = V + t[..., None] * d
        dist2 = ((P - proy) ** 2).sum(-1)
        dist2 = np.where(l2 == 0, np.inf, dist2)
        i = np.argmin(dist2, axis=1)
        filas = np.arange(len(sel))
        ds = d[filas, i]
        ps = proy[filas, i]
        nv = np.stack([-ds[:, 1], ds[:, 0]], -1)
        norma = np.hypot(nv[:, 0], nv[:, 1])
        nv = np.where(norma[:, None] > 0, nv / np.where(norma > 0, norma, 1.0)[:, None], nv)
        C = np.stack([cx[sel], cy[sel]], -1)
        flip = ((nv * (ps - C)).sum(-1)) < 0
        nv[flip] *= -1
        nv[np.isinf(dist2).all(axis=1)] = 0.0
        out[sel] = nv
        ts = t[filas, i]
        interior[sel] = np.minimum(ts, 1.0 - ts) * np.sqrt(l2[filas, i])
    return (out, interior) if con_interior else out


def normales_paredes(px, py, cx, cy, segs):
    """For each contact with a wall: wall segment closest to the contact point,
    its unit normal oriented towards the body with center (cx, cy), and the
    'interiority' of the point on that segment (as in normales_poligonos).
    """
    m = len(px)
    out = np.zeros((m, 2))
    interior = np.zeros(m)
    if m == 0 or segs is None or len(segs) == 0:
        return out, np.full(m, -np.inf)
    P = np.stack([np.asarray(px, float), np.asarray(py, float)], -1)[:, None, :]
    V = segs[None, :, 0, :]
    d = (segs[:, 1, :] - segs[:, 0, :])[None]
    l2 = (d ** 2).sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = np.clip(((P - V) * d).sum(-1) / l2, 0.0, 1.0)
    proy = V + t[..., None] * d
    dist2 = np.where(l2 == 0, np.inf, ((P - proy) ** 2).sum(-1))
    i = np.argmin(dist2, axis=1)
    filas = np.arange(m)
    ds = d[0, i]
    nv = np.stack([-ds[:, 1], ds[:, 0]], -1)
    nv /= np.where(np.hypot(nv[:, 0], nv[:, 1]) > 0, np.hypot(nv[:, 0], nv[:, 1]), 1.0)[:, None]
    C = np.stack([np.asarray(cx, float), np.asarray(cy, float)], -1)
    flip = ((nv * (C - proy[filas, i])).sum(-1)) < 0
    nv[flip] *= -1
    out[:] = nv
    ts = t[filas, i]
    interior[:] = np.minimum(ts, 1.0 - ts) * np.sqrt(l2[0, i])
    return out, interior


def normales_contacto(df, paredes=None, metodo='nuevo'):
    """Unit normal of each contact, oriented from A to B (Box2D convention). df is
    the merge of the contacts with the particles (columns _A/_B); a wall
    (id < 0) has shape NaN.

    metodo='nuevo': if either body is a disk, the radial direction of the disk
      (exact). Otherwise (polygon-polygon or polygon-wall), the surface normal
      of each side is computed and the one of the side on which the contact
      point falls inside a face (reference face), not on a vertex, is used.
    metodo='viejo': side A if A is a grain, otherwise side B; for polygons,
      always the closest edge of that body.
    """
    m = len(df)
    px = df['px'].to_numpy(np.float64)
    py = df['py'].to_numpy(np.float64)
    ax_, ay_ = df['pos_x_A'].to_numpy(np.float64), df['pos_y_A'].to_numpy(np.float64)
    bx_, by_ = df['pos_x_B'].to_numpy(np.float64), df['pos_y_B'].to_numpy(np.float64)
    shA, shB = df['shape_A'], df['shape_B']
    a_ok, b_ok = shA.notna().to_numpy(), shB.notna().to_numpy()
    discoA = (shA == 'Disco').to_numpy()
    discoB = (shB == 'Disco').to_numpy()
    polyA = a_ok & ~discoA
    polyB = b_ok & ~discoB

    def radial(cx, cy, hacia_afuera):
        vx, vy = (px - cx, py - cy) if hacia_afuera else (cx - px, cy - py)
        nr = np.hypot(vx, vy)
        nr = np.where(nr > 0, nr, 1.0)
        return np.stack([vx / nr, vy / nr], -1)

    nA = np.zeros((m, 2)); iA = np.full(m, -np.inf)
    nB = np.zeros((m, 2)); iB = np.full(m, -np.inf)
    if discoA.any():
        nA[discoA] = radial(ax_, ay_, True)[discoA]; iA[discoA] = np.inf
    if discoB.any():
        nB[discoB] = radial(bx_, by_, False)[discoB]; iB[discoB] = np.inf
    for lado, mask, cx, cy, signo, nn, ii in (('A', polyA, ax_, ay_, +1.0, nA, iA),
                                              ('B', polyB, bx_, by_, -1.0, nB, iB)):
        if mask.any():
            nv, it = normales_poligonos(px[mask], py[mask], cx[mask], cy[mask],
                                        df[f'coords_{lado}'].to_numpy()[mask], con_interior=True)
            nn[mask] = signo * nv
            ii[mask] = it
    pared_A = ~a_ok & b_ok
    pared_B = a_ok & ~b_ok
    if paredes is not None and len(paredes) > 0:
        if pared_A.any():
            nv, it = normales_paredes(px[pared_A], py[pared_A], bx_[pared_A], by_[pared_A], paredes)
            nA[pared_A] = nv; iA[pared_A] = it
        if pared_B.any():
            nv, it = normales_paredes(px[pared_B], py[pared_B], ax_[pared_B], ay_[pared_B], paredes)
            nB[pared_B] = -nv; iB[pared_B] = it

    if metodo == 'viejo':
        usar_A = a_ok
        n = np.where(usar_A[:, None], nA, nB)
        return n[:, 0], n[:, 1]
    usar_A = iA >= iB
    n = np.where(usar_A[:, None], nA, nB)
    return n[:, 0], n[:, 1]


def leer_normales_box2d(contact_file, df_contacts):
    """Box2D contact normals (from A to B) from atasco_contacts_full_<jam>.dat,
    written by the simulator together with atasco_contacts_<jam>.dat (same
    rows, same order).

    Returns (normals, reason): normals is an (m, 2) array, or None if the file
    does not exist or does not match df_contacts row by row (in that case the
    normal inferred from the geometry is used); reason says which case
    applies.
    """
    d, nombre = os.path.split(contact_file)
    full = os.path.join(d, nombre.replace('atasco_contacts_', 'atasco_contacts_full_', 1))
    if not os.path.exists(full):
        return None, 'inferred (no atasco_contacts_full)'
    cols = None
    with open(full) as f:
        for linea in f:
            if not linea.startswith('#'):
                break
            cols = linea[1:].split()
    df_full = pd.read_csv(full, sep=r'\s+', comment='#', names=cols,
                          usecols=['ID_A', 'ID_B', 'nx', 'ny'], engine='c')
    if (len(df_full) != len(df_contacts)
            or not np.array_equal(df_full['ID_A'].to_numpy(), df_contacts['id_a'].to_numpy())
            or not np.array_equal(df_full['ID_B'].to_numpy(), df_contacts['id_b'].to_numpy())):
        return None, 'inferred (atasco_contacts_full does not match row by row)'
    return df_full[['nx', 'ny']].to_numpy(np.float64), 'box2d'


def fuerzas_contacto(df_particles, df_contacts, dt, paredes=None, metodo='nuevo',
                     signo_tan=1.0, dividir_dt=True, normales=None):
    """Force of each contact on body A (Fx_A, Fy_A), with the normal from A to B.
    The force on B is -F_A. Box2D applies on B the impulse fn*n + ft*t_b2d,
    t_b2d = (n_y, -n_x) = -t with t = (-n_y, n_x); on A:
    -fn*n - ft*t_b2d = -fn*n + ft*t. This was checked against the static
    equilibrium of each grain.
    normales: (m, 2) array with the Box2D normal of each row of df_contacts
    (see leer_normales_box2d); if None, it is inferred from the geometry with
    normales_contacto (the inference fails for corner-to-corner contacts
    between polygons).
    signo_tan and dividir_dt are only used for diagnostics.
    """
    part_cols = df_particles[['pos_x', 'pos_y', 'shape', 'coords']]
    df = df_contacts.merge(part_cols, left_on='id_a', right_index=True, how='left')
    df = df.merge(part_cols, left_on='id_b', right_index=True, how='left', suffixes=('_A', '_B'))
    if normales is not None:
        nx, ny = normales[:, 0], normales[:, 1]
    else:
        nx, ny = normales_contacto(df, paredes, metodo)
    tx, ty = -ny, nx
    fn = df['f_norm'].to_numpy(np.float64)
    ft = df['f_tan'].to_numpy(np.float64)
    esc = 1.0 / dt if dividir_dt else 1.0
    df['Fx_A'] = (-fn * nx + signo_tan * ft * tx) * esc
    df['Fy_A'] = (-fn * ny + signo_tan * ft * ty) * esc
    df['a_valida'] = df['shape_A'].notna()
    df['b_valida'] = df['shape_B'].notna()
    return df


def calcular_tensor_y_presion_vectorizado(df_particles, df_contacts, arco_path, dt, umbral_r, g=9.8,
                                          paredes=None, normales=None):
    df = fuerzas_contacto(df_particles, df_contacts, dt, paredes=paredes, normales=normales)
    a_valida = df['a_valida'].to_numpy()
    b_valida = df['b_valida'].to_numpy()
    Fx, Fy = df['Fx_A'].to_numpy(), df['Fy_A'].to_numpy()
    rax, ray = (df['px'] - df['pos_x_A']).to_numpy(), (df['py'] - df['pos_y_A']).to_numpy()
    rbx, rby = (df['px'] - df['pos_x_B']).to_numpy(), (df['py'] - df['pos_y_B']).to_numpy()
    for nombre, (ra, rb, F) in {'S00': (rax, rbx, Fx), 'S11': (ray, rby, Fy),
                                'S01': (rax, rbx, Fy), 'S10': (ray, rby, Fx)}.items():
        df[f'{nombre}_A'] = np.where(a_valida, ra * F, 0.0).astype(np.float32)
        df[f'{nombre}_B'] = np.where(b_valida, -rb * F, 0.0).astype(np.float32)

    comps_A = {'S00_A': 'S00', 'S11_A': 'S11', 'S01_A': 'S01', 'S10_A': 'S10'}
    comps_B = {'S00_B': 'S00', 'S11_B': 'S11', 'S01_B': 'S01', 'S10_B': 'S10'}
    tensor_A = df.groupby('id_a')[list(comps_A)].sum().rename(columns=comps_A)
    tensor_B = df.groupby('id_b')[list(comps_B)].sum().rename(columns=comps_B)
    tensor_total = tensor_A.add(tensor_B, fill_value=0)
    
    del df, tensor_A, tensor_B
    
    p_df = df_particles.join(tensor_total).fillna({'S00': 0, 'S11': 0, 'S01': 0, 'S10': 0})
    p_df['P'] = np.where(p_df['area'] > 0, -0.5 * (p_df['S00'] + p_df['S11']) / p_df['area'], 0.0)
    p_df['P_reducida'] = p_df['P'] / g

    area_ok = p_df['area'] > 0
    area_seg = p_df['area'].where(area_ok, 1.0)
    sxx = np.where(area_ok, -p_df['S00'] / area_seg, 0.0)
    syy = np.where(area_ok, -p_df['S11'] / area_seg, 0.0)
    sxy = np.where(area_ok, -0.5 * (p_df['S01'] + p_df['S10']) / area_seg, 0.0)
    p_df['tau'] = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    p_df['tau_reducida'] = p_df['tau'] / g
    p_df['sigma_xx_reducida'] = sxx / g
    p_df['sigma_yy_reducida'] = syy / g
    p_df['sigma_xy_reducida'] = sxy / g
    
    y_arco = p_df.loc[p_df.index.isin(arco_path), 'pos_y'] if arco_path else []
    roi_ymin = y_arco.min() if len(y_arco) > 0 else 0
    roi_ymax = y_arco.max() + ROI_EXTRA if len(y_arco) > 0 else 20.0
    
    p_df['region'] = np.where(p_df.index.isin(arco_path), 'arco',
                     np.where((p_df['pos_y'] >= roi_ymin) & (p_df['pos_y'] <= roi_ymax), 'roi', 'bulk'))
    
    if 'tipo' not in p_df.columns:
        r_col = 'r_nominal' if 'r_nominal' in p_df.columns else 'r_eff'
        p_df['tipo'] = np.where(p_df[r_col] >= umbral_r, 'B', 'S')
    p_df['P_A'] = p_df['P'] * p_df['area']
    p_df['P_A_reducida'] = p_df['P_reducida'] * p_df['area']
    p_df['T_A'] = p_df['tau'] * p_df['area']
    p_df['T_A_reducida'] = p_df['tau_reducida'] * p_df['area']

    res = {}
    for region in ['arco', 'roi', 'bulk']:
        mask_reg = p_df['region'] == region
        sub_df = p_df[mask_reg]
        
        res[f"SumP_{region}"] = sub_df['P_A'].sum()
        res[f"SumP_reducida_{region}"] = sub_df['P_A_reducida'].sum()
        res[f"SumTau_{region}"] = sub_df['T_A'].sum()
        res[f"SumTau_reducida_{region}"] = sub_df['T_A_reducida'].sum()
        res[f"Area_{region}"] = sub_df['area'].sum()
        
        for tipo in ['B', 'S']:
            mask_tipo = sub_df['tipo'] == tipo
            res[f"SumP_{region}_{tipo}"] = sub_df.loc[mask_tipo, 'P_A'].sum()
            res[f"SumP_reducida_{region}_{tipo}"] = sub_df.loc[mask_tipo, 'P_A_reducida'].sum()
            res[f"SumTau_{region}_{tipo}"] = sub_df.loc[mask_tipo, 'T_A'].sum()
            res[f"SumTau_reducida_{region}_{tipo}"] = sub_df.loc[mask_tipo, 'T_A_reducida'].sum()
            res[f"Area_{region}_{tipo}"] = sub_df.loc[mask_tipo, 'area'].sum()

    particulas = p_df[['pos_x', 'pos_y', 'shape', 'tipo', 'region', 'area', 'P', 'P_reducida',
                       'tau', 'tau_reducida', 'sigma_xx_reducida', 'sigma_yy_reducida',
                       'sigma_xy_reducida', 'coords']].reset_index()
    particulas['coords'] = particulas['coords'].apply(
        lambda c: json.dumps(np.asarray(c, dtype=float).round(5).tolist()) if c is not None else ''
    )

    del p_df
    return res, particulas

def _fijar_roi_extra(valor):
    global ROI_EXTRA
    ROI_EXTRA = float(valor)


def worker_wrapper(args):
    return process_one_jam_task(*args)

def process_one_jam_task(xy_file, contact_file, jam_id, config_id, dt, r_large_m, r_small_m, sides_large_m, sides_small_m, master_chi, sim_path):
    try:
        particles = parse_xy(xy_file)
        df_particles = pd.DataFrame.from_dict(particles, orient='index')
        df_particles.index.name = 'id'
        
        especie = especie_por_tipo(df_particles['type'], df_particles['area'], master_chi)
        df_particles['tipo'] = df_particles['type'].map(especie)
        hay_chicas = (df_particles['tipo'] == 'S').any()
        if not hay_chicas:
            r_s_val = np.nan
        elif pd.notna(r_small_m):
            r_s_val = r_small_m
        else:
            r_s_val = float(np.sqrt(df_particles.loc[df_particles['tipo'] == 'S', 'area'].median() / np.pi))
        umbral_r = 0.0

        cols = ['id_a', 'id_b', 'px', 'py', 'f_norm', 'f_tan']
        dtypes_dict = {'id_a': np.int32, 'id_b': np.int32, 'px': np.float32, 'py': np.float32, 'f_norm': np.float32, 'f_tan': np.float32}
        df_contacts = pd.read_csv(contact_file, sep=r'\s+', names=cols, comment='#', dtype=dtypes_dict, engine='c')
        
        r_salida = parse_radio_salida(sim_path)
        g_abs = parse_g(sim_path)
        arco_path = detectar_arco_primario(df_particles, df_contacts, r_salida)
        if not arco_path:
            return {"ok": False, "error": "No arch detected"}
            
        normales, fuente_normales = leer_normales_box2d(contact_file, df_contacts)
        summary, particulas = calcular_tensor_y_presion_vectorizado(df_particles, df_contacts, arco_path, dt, umbral_r, g=g_abs,
                                                                    paredes=parse_paredes(xy_file),
                                                                    normales=normales)
        summary['normales'] = fuente_normales
        
        if pd.notna(sides_large_m):
            forma_grande = nombre_forma(sides_large_m)
        else:
            grandes_shapes = df_particles.loc[df_particles['tipo'] == 'B', 'shape']
            forma_grande = grandes_shapes.mode()[0] if not grandes_shapes.empty else "Ninguna"

        if not hay_chicas:
            forma_chica = "Ninguna"
        elif pd.notna(sides_small_m):
            forma_chica = nombre_forma(sides_small_m)
        else:
            chicas_shapes = df_particles.loc[df_particles['tipo'] == 'S', 'shape']
            forma_chica = chicas_shapes.mode()[0] if not chicas_shapes.empty else "Ninguna"
        
        if pd.notna(master_chi):
            chi_val = float(master_chi)
        else:
            chi_match = CHI_REGEX.search(config_id)
            chi_val = float(chi_match.group(1).replace('p', '.')) if chi_match else np.nan

        summary['jam_id'] = jam_id
        summary['config_id'] = config_id
        summary['sim_path'] = sim_path 
        summary['forma_grande'] = forma_grande
        summary['forma_chica'] = forma_chica
        summary['chi'] = chi_val
        summary['r_chica'] = r_s_val 

        particulas['jam_id'] = jam_id
        particulas['config_id'] = config_id
        particulas['sim_path'] = sim_path
        particulas['chi'] = chi_val
        particulas['forma_grande'] = forma_grande
        particulas['forma_chica'] = forma_chica
        
        del df_particles, df_contacts
        return {"ok": True, "summary": summary, "particulas": particulas}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def parse_grain_column(val):
    if pd.isna(val):
        return {}
    val_str = str(val).strip()
    if val_str.startswith('"') and val_str.endswith('"'):
        val_str = val_str[1:-1]
    val_str = val_str.replace('""', '"')
    try:
        return json.loads(val_str)
    except:
        try:
            return ast.literal_eval(val_str)
        except:
            return {}


def cargar_master(path_master):
    df_master = pd.read_csv(path_master)
    df_master = df_master.drop_duplicates(subset=['path'])
    master_dict = {}
    for _, row in df_master.iterrows():
        path_val = row.get('path')
        if pd.isna(path_val):
            continue
        grain_data = parse_grain_column(row.get('grain_json'))
        large_radius = large_sides = small_radius = small_sides = np.nan
        if isinstance(grain_data, list) and len(grain_data) > 0:
            large_radius = grain_data[0].get('radius', np.nan)
            large_sides = grain_data[0].get('sides', np.nan)
            if len(grain_data) > 1:
                small_radius = grain_data[1].get('radius', np.nan)
                small_sides = grain_data[1].get('sides', np.nan)
        if pd.isna(large_radius) and pd.notna(row.get('r_large')):
            large_radius = row.get('r_large')
        if pd.isna(small_radius) and pd.notna(row.get('r_small')):
            small_radius = row.get('r_small')
        info = {'r_large': large_radius, 'r_small': small_radius,
                'sides_large': large_sides, 'sides_small': small_sides, 'chi': row.get('chi')}
        path_abs = os.path.abspath(os.path.normpath(str(path_val)))
        master_dict[path_abs] = info
        parts = path_abs.split(os.sep)
        if len(parts) >= 2:
            master_dict[os.path.join(parts[-2], parts[-1])] = info
    return master_dict


def listar_atascos(dirs, incluir=None):
    """Single walk over the directory tree (os.walk): collects the atasco_*.xy
    files and their atasco_contacts_*.dat from the listing of each folder,
    without an os.path.exists() call per file.
    """
    pares = []
    for d in dirs:
        for root, _, files in os.walk(d):
            nombres = set(files)
            for f in files:
                if f.startswith('atasco_') and f.endswith('.xy') and not f.startswith('atasco_contacts_'):
                    c = f.replace('atasco_', 'atasco_contacts_', 1)[:-3] + '.dat'
                    if c in nombres:
                        pares.append((os.path.join(root, f), os.path.join(root, c)))
    return _filtrar_y_ordenar(pares, incluir)


def leer_lista(path_lista, incluir=None):
    """Reads a list of paths (one per line) with the atasco_*.xy and
    atasco_contacts_*.dat files. Windows paths (E:...) are mapped to
    /mnt/e/... (WSL).
    """
    xys, contactos = [], set()
    with open(path_lista, encoding='utf-8', errors='replace') as f:
        for linea in f:
            p = linea.strip().replace('\r', '')
            if not p:
                continue
            m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
            if m:
                p = f"/mnt/{m.group(1).lower()}/" + m.group(2).replace('\\', '/')
            nombre = os.path.basename(p)
            if nombre.startswith('atasco_contacts_') and nombre.endswith('.dat'):
                contactos.add(p)
            elif nombre.startswith('atasco_') and nombre.endswith('.xy'):
                xys.append(p)
    pares = []
    for xy in xys:
        d, n = os.path.split(xy)
        c = os.path.join(d, n.replace('atasco_', 'atasco_contacts_', 1)[:-3] + '.dat')
        if c in contactos:
            pares.append((xy, c))
    return _filtrar_y_ordenar(pares, incluir)


def _filtrar_y_ordenar(pares, incluir):
    if incluir:
        rx = re.compile(incluir)
        pares = [p for p in pares if rx.search(p[0])]
    return sorted(pares)


def generar_tareas(pares, master_dict, dt):
    for xy_file, contact_file in pares:
        jam_id = os.path.basename(xy_file).split('_')[-1].replace('.xy', '')
        config_id = os.path.basename(os.path.dirname(xy_file))
        abs_xy = os.path.abspath(os.path.normpath(xy_file))
        sim_path = os.path.dirname(os.path.dirname(abs_xy))
        config_info = master_dict.get(sim_path)
        if not config_info:
            parts = sim_path.split(os.sep)
            if len(parts) >= 2:
                config_info = master_dict.get(os.path.join(parts[-2], parts[-1]), {})
            else:
                config_info = {}
        yield (xy_file, contact_file, jam_id, config_id, dt,
               config_info.get('r_large', np.nan), config_info.get('r_small', np.nan),
               config_info.get('sides_large', np.nan), config_info.get('sides_small', np.nan),
               config_info.get('chi', np.nan), sim_path)


META_PART = ['forma_grande', 'forma_chica', 'config_id', 'sim_path', 'jam_id', 'chi', 'id']
META_ATASCO = ['forma_grande', 'forma_chica', 'chi', 'r_chica', 'config_id', 'sim_path', 'jam_id']


class Registro:
    def __init__(self, out_dir, reanudar):
        self.out = out_dir
        self.dir_partes = os.path.join(out_dir, 'partes')
        self.f_ok = os.path.join(out_dir, 'lotes_ok.txt')
        self.f_proc = os.path.join(out_dir, 'procesados.txt')
        self.f_resumen = os.path.join(out_dir, 'resumen_presion_por_atasco.csv')
        os.makedirs(self.dir_partes, exist_ok=True)
        if not reanudar:
            for f in (self.f_ok, self.f_proc, self.f_resumen):
                if os.path.exists(f):
                    os.remove(f)
            for f in os.listdir(self.dir_partes):
                os.remove(os.path.join(self.dir_partes, f))
        self.lotes_ok = self._leer_ok()
        for f in os.listdir(self.dir_partes):
            lote = f.rsplit('__', 1)[-1].replace('.parquet', '')
            if lote not in self.lotes_ok:
                os.remove(os.path.join(self.dir_partes, f))
        self.run = 1 + max([int(l.split('_')[0]) for l in self.lotes_ok] or [0])
        self.n_lote = 0

    def _leer_ok(self):
        if not os.path.exists(self.f_ok):
            return set()
        with open(self.f_ok) as f:
            return {l.strip() for l in f if l.strip()}

    def ya_procesados(self):
        hechos = set()
        if os.path.exists(self.f_proc):
            with open(self.f_proc) as f:
                for l in f:
                    lote, _, ruta = l.rstrip('\n').partition('\t')
                    if lote in self.lotes_ok:
                        hechos.add(ruta)
        return hechos

    def guardar_lote(self, particulas, resumenes, xy_files):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.n_lote += 1
        lote = f"{self.run:03d}_{self.n_lote:05d}"
        if particulas:
            df = pd.concat(particulas, ignore_index=True)
            df = df[[c for c in META_PART if c in df.columns] +
                    [c for c in df.columns if c not in META_PART]]
            for forma, sub in df.groupby('forma_grande', sort=False):
                tmp = os.path.join(self.dir_partes, f".tmp_{forma}__{lote}.parquet")
                pq.write_table(pa.Table.from_pandas(sub, preserve_index=False), tmp, compression='zstd')
                os.replace(tmp, os.path.join(self.dir_partes, f"{forma}__{lote}.parquet"))
            del df
        if resumenes:
            dr = pd.DataFrame(resumenes)
            dr['lote'] = lote
            dr = dr[[c for c in META_ATASCO if c in dr.columns] +
                    [c for c in dr.columns if c not in META_ATASCO]]
            dr.to_csv(self.f_resumen, mode='a', index=False, header=not os.path.exists(self.f_resumen))
        with open(self.f_proc, 'a') as f:
            f.writelines(f"{lote}\t{x}\n" for x in xy_files)
        with open(self.f_ok, 'a') as f:
            f.write(lote + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.lotes_ok.add(lote)


def combinar_partes(out_dir, borrar=False):
    """Merges out/partes/<shape>__*.parquet into one
    presion_por_particula_<shape>.parquet per shape, reading and writing one
    partial file at a time.
    """
    import pyarrow.parquet as pq
    dir_partes = os.path.join(out_dir, 'partes')
    por_forma = collections.defaultdict(list)
    for f in sorted(os.listdir(dir_partes)):
        if f.endswith('.parquet') and not f.startswith('.tmp_'):
            por_forma[f.rsplit('__', 1)[0]].append(os.path.join(dir_partes, f))
    for forma, archivos in por_forma.items():
        destino = os.path.join(out_dir, f"presion_por_particula_{forma}.parquet")
        tmp = destino + ".tmp"
        escritor, esquema, filas = None, None, 0
        for a in archivos:
            t = pq.read_table(a)
            if escritor is None:
                esquema = t.schema
                escritor = pq.ParquetWriter(tmp, esquema, compression='zstd')
            elif not t.schema.equals(esquema):
                t = t.cast(esquema)
            escritor.write_table(t)
            filas += t.num_rows
        escritor.close()
        os.replace(tmp, destino)
        print(f"  {os.path.basename(destino)}: {filas:,} rows ({len(archivos)} parts)", flush=True)
        if borrar:
            for a in archivos:
                os.remove(a)


def resumen_por_configuracion(out_dir, lotes_ok):
    atasco_csv_path = os.path.join(out_dir, "resumen_presion_por_atasco.csv")
    config_csv_path = os.path.join(out_dir, "resumen_presion_por_configuracion.csv")
    if not os.path.exists(atasco_csv_path):
        return
    df_all = pd.read_csv(atasco_csv_path, dtype={'lote': str, 'jam_id': str})
    if 'lote' in df_all.columns:
        df_all = df_all[df_all['lote'].isin(lotes_ok)]
    group_cols = ['forma_grande', 'forma_chica', 'chi', 'r_chica', 'config_id', 'sim_path']
    grouped = df_all.groupby(group_cols, dropna=False)
    df_config = grouped.size().reset_index(name='n_atascos')
    variables_sumar = [c for c in df_all.columns
                       if c.startswith('SumP_') or c.startswith('SumTau_') or c.startswith('Area_')]
    cols_to_agg = variables_sumar
    if not cols_to_agg:
        return
    df_sums = grouped[cols_to_agg].sum().reset_index(drop=True)
    df_config = pd.concat([df_config, df_sums], axis=1)
    for c in cols_to_agg:
        for pref, nombre in (('SumP_', 'mean_P_'), ('SumTau_', 'mean_tau_')):
            if c.startswith(pref):
                resto = c[len(pref):]
                area = 'Area_' + resto.replace('reducida_', '', 1)
                if area in df_config.columns:
                    df_config[nombre + resto] = df_config[c] / df_config[area]
    df_config = df_config.drop(columns=cols_to_agg)
    df_config.to_csv(config_csv_path, index=False)
    print(f"Summary per configuration: {config_csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Stress tensor, pressure and shear stress of each particle in each clog.")
    parser.add_argument("--dir", nargs='+', default=None,
                        help="root folder(s) of the simulations (scanned recursively)")
    parser.add_argument("--lista", default=None,
                        help="file with the list of atasco_*.xy and atasco_contacts_*.dat files (one path "
                             "per line); avoids scanning "
                             "the disk")
    parser.add_argument("--incluir", default=None,
                        help="regex on the path: only matching clogs are processed "
                             "(e.g. 'Ratio_0p1_Chi_0p(0|2)_W')")
    parser.add_argument("--out", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--roi_extra", type=float, default=ROI_EXTRA,
                        help="extra height of the ROI above the highest arch grain "
                             "(simulation units; default 0: only the band spanned by the arch)")
    parser.add_argument("--filas_lote", type=int, default=500_000,
                        help="particle rows per saved batch (default 500000)")
    parser.add_argument("--reanudar", action="store_true",
                        help="resume an interrupted run: skip the clogs already saved in --out")
    parser.add_argument("--solo_combinar", action="store_true",
                        help="process nothing: only merge out/partes/ into one parquet per shape")
    parser.add_argument("--borrar_partes", action="store_true",
                        help="delete out/partes/ after merging")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.solo_combinar:
        print("Merging partial files...", flush=True)
        combinar_partes(args.out, borrar=args.borrar_partes)
        return

    if not args.dir and not args.lista:
        raise SystemExit("--dir or --lista is required.")

    registro = Registro(args.out, args.reanudar)

    print("Loading the master index and the grain definitions...", flush=True)
    master_dict = cargar_master(args.master)

    t_busq = time.time()
    if args.lista:
        print(f"Reading the file list {args.lista}...", flush=True)
        pares = leer_lista(args.lista, args.incluir)
    else:
        print("Scanning folders for atasco_*.xy...", flush=True)
        pares = listar_atascos(args.dir, args.incluir)
    print(f"Clogs found: {len(pares)}  (search: {time.time() - t_busq:.0f} s)", flush=True)

    if args.reanudar:
        hechos = registro.ya_procesados()
        antes = len(pares)
        pares = [p for p in pares if p[0] not in hechos]
        print(f"--reanudar: {antes - len(pares)} already processed, {len(pares)} left", flush=True)

    total = len(pares)
    part_buf, res_buf, xy_buf = [], [], []
    filas_buf = 0
    procesados = 0
    errores = collections.Counter()
    t0 = time.time()

    def guardar():
        nonlocal part_buf, res_buf, xy_buf, filas_buf
        registro.guardar_lote(part_buf, res_buf, xy_buf)
        part_buf, res_buf, xy_buf = [], [], []
        filas_buf = 0
        gc.collect()

    tareas = generar_tareas(pares, master_dict, args.dt)
    _fijar_roi_extra(args.roi_extra)
    print(f"ROI: band of the arch + {args.roi_extra:g} above the highest grain", flush=True)
    with mp.Pool(processes=args.workers, maxtasksperchild=50,
                 initializer=_fijar_roi_extra, initargs=(args.roi_extra,)) as pool:
        for xy, res in pool.imap_unordered(worker_wrapper_con_ruta, tareas, chunksize=4):
            procesados += 1
            xy_buf.append(xy)
            if res["ok"]:
                res_buf.append(res["summary"])
                part_buf.append(res["particulas"])
                filas_buf += len(res["particulas"])
            else:
                errores[res.get("error", "?")] += 1
            if filas_buf >= args.filas_lote:
                guardar()
            if procesados % 200 == 0:
                dtr = time.time() - t0
                ritmo = procesados / dtr
                resta = (total - procesados) / ritmo if ritmo > 0 else float('nan')
                print(f"Processed {procesados} of {total}  ({ritmo:.1f} clogs/s, "
                      f"~{resta / 60:.0f} min left, {sum(errores.values())} failed)", flush=True)
    if xy_buf:
        guardar()

    print(f"\n{procesados} clogs in {(time.time() - t0) / 60:.1f} min, "
          f"{sum(errores.values())} failed.", flush=True)
    for msg, n in errores.most_common(10):
        print(f"  {n:>6}  {msg}")

    print("\nMerging the partial files into one parquet per forma_grande...", flush=True)
    combinar_partes(args.out, borrar=args.borrar_partes)
    resumen_por_configuracion(args.out, registro.lotes_ok)


def worker_wrapper_con_ruta(args):
    return args[0], process_one_jam_task(*args)


if __name__ == "__main__":
    mp.freeze_support()
    main()