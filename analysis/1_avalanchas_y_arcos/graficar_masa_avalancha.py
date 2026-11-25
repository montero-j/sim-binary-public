#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mean avalanche size in MASS (instead of number of grains) vs chi, split
into large grains, small grains and total, for several systems (by default
square-disk and disk-disk).

Mass of each species (as computed by Box2D, density * area):
    disk    (nLados = 1): m = dens * pi r^2
    polygon (nLados = n): m = dens * (n/2) r^2 sin(2 pi / n)
      (the n vertices lie on the circle of radius r)
with 'N radius nLados dens fric rest' from each grain line of params.in.

Per avalanche (resultados_completos_esqueleto.csv from processor.py):
    M_large = tamano_big * m_large
    M_small = (total_mix - tamano_big) * m_small
    M_total = M_large + M_small
Then: mean and SEM over all the avalanches of each (system, chi).

Usage:
    python3 analysis/1_avalanchas_y_arcos/graficar_masa_avalancha.py
    python3 analysis/1_avalanchas_y_arcos/graficar_masa_avalancha.py --solo_con_arco
    python3 analysis/1_avalanchas_y_arcos/graficar_masa_avalancha.py --sistemas Disco-Disco Cuad-Disco Tri-Disco Pent-Disco --area_ratio 0.1 --unidad m_grande
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rutas  # noqa: E402
from estilo_figuras import (aplicar_estilo, estilo_sistema, guardar_figura,  # noqa: E402
                            ERRORBAR_KW, SYSTEM_NAME_MAP)

DENS_DEFAULT = 1.27324

COMPONENTES = {
    'M_grandes': dict(etiqueta='large', fill='full', ls='-', marker=None),
    'M_chicos':  dict(etiqueta='small', fill='none', ls='--', marker=None),
    'M_total':   dict(etiqueta='total', fill='full', ls=':', marker='*'),
}


def area_grano(radio, n_lados):
    if int(n_lados) == 1:
        return np.pi * radio ** 2
    n = int(n_lados)
    return 0.5 * n * radio ** 2 * np.sin(2.0 * np.pi / n)


def leer_granos_params(sim_path):
    """[(radius, nLados, dens), ...] in the grain-type order of params.in, or None."""
    try:
        with open(os.path.join(sim_path, "params.in")) as f:
            lineas = [l.split('#')[0].strip() for l in f]
    except OSError:
        return None
    for i, l in enumerate(lineas):
        if l.replace(' ', '').lower().startswith('notipogranos:'):
            n = int(l.split(':')[1])
            granos = [x.split() for x in lineas[i + 1:] if x][:n]
            return [(float(g[1]), int(g[2]), float(g[3])) for g in granos]
    return None


def masas_por_simulacion(row):
    """m_large, m_small (large = grain type with the largest radius, as in
    processor.py). Uses the params.in of the simulation; if missing, grain_json
    and the default density.
    """
    granos = leer_granos_params(row['path']) if isinstance(row.get('path'), str) else None
    fuente = 'params.in'
    if not granos:
        fuente = 'grain_json'
        try:
            granos = [(float(g['radius']), int(g['sides']), DENS_DEFAULT) for g in json.loads(row['grain_json'])]
        except Exception:
            return pd.Series({'m_grande': np.nan, 'm_chico': np.nan, 'fuente_masa': 'sin_datos'})
    masas = [d * area_grano(r, n) for r, n, d in granos]
    i_g = int(np.argmax([r for r, _, _ in granos]))
    otros = [m for i, m in enumerate(masas) if i != i_g]
    return pd.Series({'m_grande': masas[i_g], 'm_chico': otros[0] if otros else np.nan, 'fuente_masa': fuente})


def preparar(idx_path, av_path, solo_con_arco):
    av = pd.read_csv(av_path)
    if solo_con_arco:
        if 'atasco_id' not in av.columns:
            sys.exit("[!] The avalanche file has no atasco_id column: run processor.py --emparejar.")
        n0 = len(av)
        av = av[av['atasco_id'].notna()].copy()
        print(f"--solo_con_arco: {len(av)} of {n0} avalanches have an associated clog (arch).")

    idx = pd.read_csv(idx_path)
    if 'area_ratio_nominal' not in idx.columns and 'area_ratio' in idx.columns:
        idx['area_ratio_nominal'] = idx['area_ratio'].round(3)
    idx = idx[idx['sim_hash'].isin(av['sim_hash'].unique())].copy()
    idx = pd.concat([idx, idx.apply(masas_por_simulacion, axis=1)], axis=1)
    print("Grain masses from: " + ", ".join(f"{k}: {v}" for k, v in idx['fuente_masa'].value_counts().items())
          + " simulations")
    if (idx['fuente_masa'] == 'grain_json').any():
        print(f"  [!] params.in missing for some simulations: dens = {DENS_DEFAULT} was used (check SIM_DATA_DIR)")

    cols = [c for c in ('sim_hash', 'system_type', 'chi', 'area_ratio_nominal', 'Radio_salida',
                        'm_grande', 'm_chico') if c in idx.columns]
    df = av.merge(idx[cols], on='sim_hash', how='inner')
    df['sistema'] = df['system_type'].map(lambda s: SYSTEM_NAME_MAP.get(str(s), str(s)))
    n_chicos = (df['total_mix'] - df['tamano_big']).clip(lower=0)
    df['M_grandes'] = df['tamano_big'] * df['m_grande']
    df['M_chicos'] = n_chicos * df['m_chico'].fillna(0.0)
    df['M_total'] = df['M_grandes'] + df['M_chicos']
    return df


def agregar(df):
    g = df.groupby(['sistema', 'chi'])
    res = g[list(COMPONENTES)].agg(['mean', 'sem'])
    res.columns = [f"{c}_{s}" for c, s in res.columns]
    res['n_avalanchas'] = g.size()
    res['n_simulaciones'] = g['sim_hash'].nunique()
    res['m_grande'] = g['m_grande'].first()
    res['m_chico'] = g['m_chico'].first()
    return res.reset_index()


def graficar(res, sistemas, out, xlim, una_figura, unidad_lbl, logy):
    aplicar_estilo()
    n = len(sistemas)
    if una_figura:
        fig, axes = plt.subplots(1, 1, figsize=(7.5, 5.5))
        axes = [axes] * n
    else:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), sharey=True, squeeze=False)
        axes = list(axes[0])

    margen = 0.025 * (xlim[1] - xlim[0])
    for ax, sist in zip(axes, sistemas):
        d = res[res['sistema'] == sist].sort_values('chi')
        color, marker, nombre = estilo_sistema(sist)
        if d.empty:
            print(f"  [!] no data for {sist}")
        for comp, st in COMPONENTES.items():
            if comp == 'M_chicos':
                d_c = d[d['chi'] > 0]
            else:
                d_c = d
            lbl = f"{nombre}, {st['etiqueta']}" if una_figura else st['etiqueta']
            ax.errorbar(d_c['chi'], d_c[f'{comp}_mean'], yerr=d_c[f'{comp}_sem'], color=color,
                        marker=st['marker'] or marker, fillstyle=st['fill'],
                        markersize=11 if st['marker'] == '*' else None,
                        ls=st['ls'], label=lbl, **ERRORBAR_KW)
        if not una_figura:
            ax.set_title(nombre)
        ax.set_xlim(xlim[0] - margen, xlim[1] + margen)
        ax.set_xlabel(r'$\chi$')
        if logy:
            ax.set_yscale('log')
    axes[0].set_ylabel(rf'$\langle M \rangle$ {unidad_lbl}')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    guardar_figura(out, fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--av', default=os.path.join(rutas.AVALANCHE_DIR, 'resultados_completos_esqueleto.csv'))
    ap.add_argument('--idx', default=rutas.MASTER_INDEX)
    ap.add_argument('--sistemas', nargs='+', default=['Cuad-Disco', 'Disco-Disco'],
                    help="system_type of the index (Disco-Disco, Cuad-Disco, Tri-Disco, Pent-Disco, ...)")
    ap.add_argument('--area_ratio', type=float, default=None,
                    help="filter by area_ratio_nominal (e.g. 0.1); default: all")
    ap.add_argument('--radio_salida', type=float, default=None, help="filter by Radio_salida; default: all")
    ap.add_argument('--chi_max', type=float, default=0.4)
    ap.add_argument('--xlim', nargs=2, type=float, default=[0.0, 0.4])
    ap.add_argument('--unidad', choices=['masa', 'm_grande'], default='masa',
                    help="masa: simulation units; m_grande: in units of the mass of a large grain")
    ap.add_argument('--logy', action='store_true')
    ap.add_argument('--una_figura', action='store_true', help="all systems in a single panel")
    ap.add_argument('--solo_con_arco', action='store_true',
                    help="only avalanches that ended in a clog with a saved snapshot (arch)")
    ap.add_argument('--out', default=None, help="output figure (no extension -> pdf)")
    args = ap.parse_args()

    df = preparar(args.idx, args.av, args.solo_con_arco)
    sistemas = [SYSTEM_NAME_MAP.get(s, s) for s in args.sistemas]
    puras = df[~df['sistema'].str.contains('-')]
    if not puras.empty:
        extra = [puras[puras['sistema'] == s.split('-')[0]].assign(sistema=s) for s in sistemas]
        df = pd.concat([df[df['sistema'].str.contains('-')]] + extra, ignore_index=True)
    df = df[df['sistema'].isin(sistemas) & (df['chi'] <= args.chi_max + 1e-9)]
    if args.area_ratio is not None:
        df = df[np.isclose(df['area_ratio_nominal'], args.area_ratio) | (df['chi'] == 0)]
    if args.radio_salida is not None and 'Radio_salida' in df.columns:
        df = df[np.isclose(df['Radio_salida'], args.radio_salida)]
    if df.empty:
        sys.exit("[!] No avalanches left after filtering.")
    for r in ('area_ratio_nominal', 'Radio_salida'):
        if r in df.columns and df.loc[df['chi'] > 0, r].nunique() > 1:
            print(f"  [!] several values of {r} are mixed: {sorted(df.loc[df['chi'] > 0, r].dropna().unique())} "
                  f"(filter with --{'area_ratio' if r.startswith('area') else 'radio_salida'})")

    unidad_lbl = ''
    if args.unidad == 'm_grande':
        for c in COMPONENTES:
            df[c] = df[c] / df['m_grande']
        unidad_lbl = r'$/\, m_{\rm large}$'

    res = agregar(df)
    sufijo = '_solo_con_arco' if args.solo_con_arco else ''
    if args.area_ratio is not None:
        sufijo += f"_Ar{args.area_ratio:g}".replace('.', 'p')
    base = os.path.join(rutas.GRAFICOS_FINALES_DIR, f"masa_avalancha{sufijo}")
    os.makedirs(rutas.GRAFICOS_FINALES_DIR, exist_ok=True)
    res.to_csv(base + '.csv', index=False)
    print(f"Table: {base}.csv")
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(res[['sistema', 'chi', 'M_grandes_mean', 'M_chicos_mean', 'M_total_mean',
                   'n_avalanchas', 'n_simulaciones']].to_string(index=False, float_format='%.3f'))

    out = args.out or base
    graficar(res, sistemas, out, args.xlim, args.una_figura, unidad_lbl, args.logy)
    print(f"Figure: {out if os.path.splitext(out)[1] else out + '.pdf'}")


if __name__ == '__main__':
    main()
