#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pressure (and shear stress) vs chi, one figure per large-grain shape,
comparing the arch region with the ROI (large grains: solid lines, small
grains: dashed lines), from resumen_presion_por_atasco.csv (one row per
clog, written by presion_tensor_arch.py). Two quantities, in separate
figures:
  - Reduced mean pressure      <P>d/mg    -> <out>_nuevo_<Shape>.pdf
  - Reduced mean shear stress  <tau>d/mg  -> <out>_nuevo_tau_<Shape>.pdf
    (tau = radius of the Mohr circle of the stress tensor of each particle;
    area-weighted average, as for P)

All the clogs with the same (shape, chi) are pooled into a single point,
even if they come from different runs.

The data are filtered by area ratio (small/large) before aggregating. The
ratio is read from "Ratio_X" in the folder name; folders without that tag
("Chi_X_W_Y") have R_a = 0.1.

Error bars: spread BETWEEN CLOGS (each clog is one sample; the particles of
the same clog are not independent). For each clog, its mean in the region
(Sum/Area of that clog) is computed and, for each shape and chi, the
spread of these means weighted by the area of the clog in the region (the
same weight used by the central value, Sum total / Area total):
  --error sem (default): standard error = std / sqrt(n_eff),
                          n_eff = (sum w)^2 / sum w^2 (effective number of clogs)
  --error std           : weighted standard deviation
  --error ninguno       : no error bars
<out>_datos.csv stores the mean, std, sem, n and n_eff of each point.

x axis (--xlim): same chi range in all figures, default 0 to 0.4.
Legend (--leyenda): afuera (default, to the right of the axes), arriba
(above, 2 columns), adentro (loc='best') or auto.

If the csv comes from presion_tensor_arch.py with resumable batches (column
'lote'), only the batches listed in lotes_ok.txt (same folder as the csv)
are used.

Usage:
    python3 comparar_presion_por_chi.py --csv resumen_presion_por_atasco.csv --out comparacion_chi.pdf
    # -> comparacion_chi_nuevo_<Shape>.pdf (pressure) and comparacion_chi_nuevo_tau_<Shape>.pdf (shear)

    # A single quantity, another area ratio, other shapes or another chi range:
    python3 comparar_presion_por_chi.py --csv resumen_presion_por_atasco.csv --magnitud tau --ratio 0.3 --formas Disco Cuadrado --chi_max 0.4 --out comparacion_chi.pdf
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from estilo_figuras import (aplicar_estilo, estilo_forma, guardar_figura,
                            REGION_STYLE, ERRORBAR_KW)

REGIONES = ('arco', 'roi')

SUFIJO_NUEVO = '_nuevo'

MAGNITUDES = {
    'P':   ('SumP_reducida',   r'$\langle P\rangle d / mg$', ''),
    'tau': ('SumTau_reducida', r'$\langle \tau\rangle d / mg$', '_tau'),
}

RATIO_REGEX = re.compile(r'Ratio_([0-9]+p[0-9]+)')
RATIO_SIN_PREFIJO = 0.1


def extraer_ratio_area(sim_path):
    m = RATIO_REGEX.search(str(sim_path))
    if m:
        return float(m.group(1).replace('p', '.'))
    return RATIO_SIN_PREFIJO


def filtrar_lotes_confirmados(df, csv_path):
    """If the csv has a 'lote' column, keeps only the batches listed in
    lotes_ok.txt (those that were completely written).
    """
    if 'lote' not in df.columns:
        return df
    f_ok = os.path.join(os.path.dirname(os.path.abspath(csv_path)), 'lotes_ok.txt')
    if not os.path.exists(f_ok):
        print(f"WARNING: the csv has a 'lote' column but {f_ok} was not found; all rows are used.")
        return df
    with open(f_ok) as f:
        ok = {l.strip() for l in f if l.strip()}
    antes = len(df)
    df = df[df['lote'].astype(str).isin(ok)]
    if len(df) < antes:
        print(f"{antes - len(df)} rows of unconfirmed batches are discarded (interrupted run).")
    return df


def _estadistica_pesada(x, w):
    """Mean, standard deviation, standard error and n of the per-clog means x,
    weighted by w (area of the clog in the region). Clogs with no area in the
    region (w = 0, e.g. no small grains in the arch) do not count.
    """
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0, 0.0
    sw = w.sum()
    media = (w * x).sum() / sw
    n_eff = sw ** 2 / (w ** 2).sum()
    if n < 2 or n_eff <= 1:
        return media, np.nan, np.nan, n, n_eff
    var = (w * (x - media) ** 2).sum() / (sw - (w ** 2).sum() / sw)
    std = np.sqrt(var)
    return media, std, std / np.sqrt(n_eff), n, n_eff


def agregar_por_forma_y_chi(df, magnitudes, regiones=REGIONES):
    """One point per (forma_grande, chi), pooling all the clogs that share that
    chi. The central value is Sum total / Area total (area-weighted mean); the
    spread is that of the per-clog means, with the same weight.
    """
    cols_necesarias = []
    for region in regiones:
        for t in ('B', 'S'):
            cols_necesarias.append(f'Area_{region}_{t}')
            for mag in magnitudes:
                cols_necesarias.append(f'{MAGNITUDES[mag][0]}_{region}_{t}')
    faltantes = [c for c in cols_necesarias if c not in df.columns]
    if faltantes:
        pista = (" The SumTau_* columns are written by presion_tensor_arch.py "
                 "(shear stress): run it again."
                 if any(c.startswith('SumTau') for c in faltantes) else "")
        raise SystemExit(f"The csv lacks the columns {faltantes}.{pista}")

    filas = []
    for (forma, chi), g in df.groupby(['forma_grande', 'chi'], dropna=False):
        fila = {'forma_grande': forma, 'chi': chi, 'n_atascos': len(g)}
        for mag in magnitudes:
            pref = MAGNITUDES[mag][0]
            for region in regiones:
                for t in ('B', 'S'):
                    w = g[f'Area_{region}_{t}'].to_numpy(dtype=float)
                    suma = g[f'{pref}_{region}_{t}'].to_numpy(dtype=float)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        x = np.where(w > 0, suma / w, np.nan)
                    media, std, sem, n, n_eff = _estadistica_pesada(x, w)
                    clave = f'{mag}_{region}_{t}'
                    fila[f'mean_{clave}'] = media
                    fila[f'std_{clave}'] = std
                    fila[f'sem_{clave}'] = sem
                    fila[f'n_{clave}'] = n
                    fila[f'neff_{clave}'] = n_eff
        filas.append(fila)
    return pd.DataFrame(filas)


def limites_y(df_agg, formas, mags, error, regiones=REGIONES, margen=0.05):
    """(ymin, ymax) covering all the points and error bars of the given shapes
    and quantities, with a margin; never below 0.
    """
    lo, hi = [], []
    sub = df_agg[df_agg['forma_grande'].isin(formas)]
    for mag in mags:
        for region in regiones:
            for t in ('B', 'S'):
                y = sub[f'mean_{mag}_{region}_{t}'].to_numpy(dtype=float)
                e = (np.zeros_like(y) if error == 'ninguno'
                     else np.nan_to_num(sub[f'{error}_{mag}_{region}_{t}'].to_numpy(dtype=float)))
                ok = np.isfinite(y)
                if ok.any():
                    lo.append(np.min(y[ok] - e[ok]))
                    hi.append(np.max(y[ok] + e[ok]))
    if not lo:
        return None
    ymin, ymax = min(lo), max(hi)
    pad = margen * (ymax - ymin if ymax > ymin else abs(ymax) or 1.0)
    return max(0.0, ymin - pad), ymax + pad


def _tapados(ax, leg, puntos):
    """Number of points (data, error-bar ends, lines) inside the legend box."""
    fig = ax.figure
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer()).expanded(1.04, 1.08)
    pts = ax.transData.transform(np.asarray([p for p in puntos if np.all(np.isfinite(p))], dtype=float))
    return int(np.sum((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)))


def leyenda_sin_tapar(ax, puntos, titulo, max_pasos=12):
    """Legend inside the axes, in two columns, at the position that covers no
    data (nor error bars or lines). If no position is free, the upper y limit
    is raised step by step until there is room on top.
    """
    posiciones = ['upper right', 'upper left', 'upper center', 'center right', 'center left',
                  'center', 'lower right', 'lower left', 'lower center']
    kw = dict(title=titulo, ncol=2, columnspacing=1.0,
              handlelength=1.6, framealpha=0.9)
    y0, y1 = ax.get_ylim()
    for paso in range(max_pasos + 1):
        mejor = None
        for loc in posiciones:
            leg = ax.legend(loc=loc, **kw)
            n = _tapados(ax, leg, puntos)
            if mejor is None or n < mejor[0]:
                mejor = (n, loc)
            if n == 0:
                return leg
        ax.set_ylim(y0, y1 + (paso + 1) * 0.08 * (y1 - y0))
    return ax.legend(loc=mejor[1], **kw)


def graficar_una_forma(df_agg, forma, mag, etiqueta_chi, error='sem', desplazar=0.0,
                       ylim=None, regiones=REGIONES, leyenda='afuera', xlim=(0.0, 0.4)):
    """One figure for one large-grain shape and one quantity: arch vs ROI, large
    vs small grains, as a function of chi, with error bars.
    Color = region (arch vermillion, ROI blue). Large grains: marker of the
    grain shape, solid line. Small grains (always disks): open circle, dashed
    line.
    """
    sub = df_agg[df_agg['forma_grande'] == forma].sort_values('chi')
    chi = sub['chi'].to_numpy(dtype=float)
    paso = desplazar * (np.nanmax(chi) - np.nanmin(chi) if len(chi) > 1 else 1.0)
    label_forma, marker_forma = estilo_forma(forma)

    fig, ax = plt.subplots()
    series = [(region, t) for region in regiones for t in ('B', 'S')]
    puntos = []
    for k, (region, t) in enumerate(series):
        color, label_region = REGION_STYLE[region]
        clave = f'{mag}_{region}_{t}'
        y = sub[f'mean_{clave}'].to_numpy(dtype=float)
        yerr = None if error == 'ninguno' else sub[f'{error}_{clave}'].to_numpy(dtype=float)
        dx = (k - (len(series) - 1) / 2) * paso
        if t == 'B':
            estilo = dict(fmt=f'{marker_forma}-', color=color)
            label = f'{label_region}, large'
        else:
            estilo = dict(fmt='o--', color=color, markerfacecolor='white')
            label = f'{label_region}, small'
        ax.errorbar(chi + dx, y, yerr=yerr, label=label, **estilo, **ERRORBAR_KW)
        e = np.zeros_like(y) if yerr is None else np.nan_to_num(np.asarray(yerr, dtype=float))
        for xs in (chi + dx,):
            puntos += [(xi, yi) for xi, yi in zip(xs, y)]
            puntos += [(xi, yi + ei) for xi, yi, ei in zip(xs, y, e)]
            puntos += [(xi, yi - ei) for xi, yi, ei in zip(xs, y, e)]
            ok = np.isfinite(y)
            xo, yo = xs[ok], y[ok]
            for i in range(len(xo) - 1):
                t = np.linspace(0, 1, 12)
                puntos += list(zip(xo[i] + t * (xo[i + 1] - xo[i]), yo[i] + t * (yo[i + 1] - yo[i])))

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        margen = 0.025 * (xlim[1] - xlim[0])
        ax.set_xlim(xlim[0] - margen, xlim[1] + margen)
    ax.set_xlabel(etiqueta_chi)
    ax.set_ylabel(MAGNITUDES[mag][1])
    if leyenda == 'afuera':
        ax.legend(title=label_forma, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    elif leyenda == 'arriba':
        ax.legend(title=label_forma, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
                  borderaxespad=0.0)
    elif leyenda == 'auto':
        leyenda_sin_tapar(ax, puntos, label_forma)
    else:
        ax.legend(title=label_forma, loc='best')
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=None, help='resumen_presion_por_atasco.csv')
    parser.add_argument('--datos', default=None,
                        help='<out>_datos.csv of a previous run (values already aggregated by '
                             'shape and chi): plot from it without --csv. --ratio is not '
                             'applied (the file has a single area ratio); --formas, '
                             '--chi_min and --chi_max are.')
    parser.add_argument('--out', default='comparacion_chi.pdf',
                        help='base name; the extension sets the format (default pdf)'
                             '')
    parser.add_argument('--magnitud', choices=['P', 'tau', 'ambas'], default='ambas',
                        help="P (pressure), tau (shear stress) or ambas (both, default), "
                             "each in its own figure")
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='Area ratio (small/large) to include (default: 0.1)')
    parser.add_argument('--formas', nargs='+', default=None,
                        help='forma_grande values to include (default: all in the csv)')
    parser.add_argument('--chi_min', type=float, default=None)
    parser.add_argument('--chi_max', type=float, default=None)
    parser.add_argument('--error', choices=['std', 'sem', 'ninguno'], default='sem',
                        help="Error bars between clogs: sem = standard error of the mean (default), "
                             "std = standard deviation, ninguno = no error bars")
    parser.add_argument('--xlim', type=float, nargs=2, default=[0.0, 0.4], metavar=('CHIMIN', 'CHIMAX'),
                        help='chi range of the x axis, the same in all figures (default 0 0.4)')
    parser.add_argument('--leyenda', choices=['afuera', 'arriba', 'adentro', 'auto'], default='afuera',
                        help="Legend position: afuera (default, to the right of the axes), "
                             "arriba (above, 2 columns) or adentro (wherever matplotlib finds room)")
    parser.add_argument('--desplazar', type=float, default=0.0,
                        help="Horizontal offset between series, as a fraction of the chi range "
                             "(e.g. 0.004), so that the error bars do not overlap. Default 0.")
    parser.add_argument('--escala_y', choices=['por_magnitud', 'comun', 'libre'],
                        default='por_magnitud',
                        help="por_magnitud (default): all the P figures share the y axis, and "
                             "all the tau figures share another one; comun: P and tau share the y axis; "
                             "libre: each figure has its own y axis")
    parser.add_argument('--ylim_P', type=float, nargs=2, default=None, metavar=('YMIN', 'YMAX'),
                        help='y limits of the pressure figures')
    parser.add_argument('--ylim_tau', type=float, nargs=2, default=None, metavar=('YMIN', 'YMAX'),
                        help='y limits of the shear-stress figures')
    parser.add_argument('--etiqueta_chi', default=r'Concentration $\chi$',
                        help='x-axis label')
    args = parser.parse_args()
    aplicar_estilo(verbose=True)

    magnitudes = ['P', 'tau'] if args.magnitud == 'ambas' else [args.magnitud]

    if (args.csv is None) == (args.datos is None):
        parser.error("pass either --csv (per-clog summary) or --datos (aggregated values), not both.")

    if args.datos is not None:
        df = pd.read_csv(args.datos)
        faltan = [f'mean_{m}_{r}_{t}' for m in magnitudes for r in REGIONES for t in ('B', 'S')
                  if f'mean_{m}_{r}_{t}' not in df.columns]
        if faltan:
            raise SystemExit(f"{args.datos} lacks the columns {faltan}: check --magnitud.")
        print(f"Plotting aggregated values from {args.datos} (--ratio is not applied).")
    else:
        df = pd.read_csv(args.csv, dtype={'lote': str, 'jam_id': str})
        df = filtrar_lotes_confirmados(df, args.csv)
        df['ratio_area'] = df['sim_path'].apply(extraer_ratio_area)
        df = df[np.isclose(df['ratio_area'], args.ratio, atol=1e-3)]
        if df.empty:
            raise SystemExit(f"No rows with ratio_area ~= {args.ratio}: check --ratio.")
        print(f"Area ratio = {args.ratio}: {df['sim_path'].nunique()} folders included "
              f"({df['config_id'].nunique()} distinct config_id).")

    if args.formas is not None:
        df = df[df['forma_grande'].isin(args.formas)]
    if args.chi_min is not None:
        df = df[df['chi'] >= args.chi_min]
    if args.chi_max is not None:
        df = df[df['chi'] <= args.chi_max + 1e-9]
    if df.empty:
        raise SystemExit("No rows left after filtering: check --formas/--chi_min/--chi_max.")

    df_agg = df.reset_index(drop=True) if args.datos is not None else agregar_por_forma_y_chi(df, magnitudes)
    formas = args.formas if args.formas is not None else sorted(df_agg['forma_grande'].dropna().unique())
    print("Clogs per shape and chi:")
    print(df_agg[['forma_grande', 'chi', 'n_atascos']].to_string(index=False))

    base, ext = os.path.splitext(args.out)
    base = base + SUFIJO_NUEVO
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    datos_path = f"{base}_datos.csv"
    df_agg.to_csv(datos_path, index=False)
    print(f"Plotted values (mean, std, sem, n, n_eff): {datos_path}")
    print(f"Error bars: {args.error}")

    formas_ok = [f for f in formas if f in df_agg['forma_grande'].values]
    ylims = {}
    if args.escala_y == 'comun':
        lim = limites_y(df_agg, formas_ok, magnitudes, args.error)
        ylims = {m: lim for m in magnitudes}
    elif args.escala_y == 'por_magnitud':
        ylims = {m: limites_y(df_agg, formas_ok, [m], args.error) for m in magnitudes}
    if args.ylim_P is not None:
        ylims['P'] = tuple(args.ylim_P)
    if args.ylim_tau is not None:
        ylims['tau'] = tuple(args.ylim_tau)
    for m, lim in ylims.items():
        if lim is not None and m in magnitudes:
            print(f"y axis {m}: [{lim[0]:.4g}, {lim[1]:.4g}]  "
                  f"(to fix it: --ylim_{m} {lim[0]:.4g} {lim[1]:.4g})")
    for mag in magnitudes:
        sufijo = MAGNITUDES[mag][2]
        for forma in formas:
            if forma not in df_agg['forma_grande'].values:
                print(f"(no data for {forma!r}, skipped)")
                continue
            fig = graficar_una_forma(df_agg, forma, mag, args.etiqueta_chi,
                                     error=args.error, desplazar=args.desplazar,
                                     ylim=ylims.get(mag), leyenda=args.leyenda,
                                     xlim=tuple(args.xlim))
            out_path = f"{base}{sufijo}_{forma}{ext}"
            guardar_figura(out_path, fig)
            print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()