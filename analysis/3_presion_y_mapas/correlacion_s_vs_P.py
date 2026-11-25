#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correlation between the mean avalanche size <s> and the mean pressure <P>
of the clogging arches.

One point per (system, chi), as in comparar_presion_por_chi.py: if several
simulations share the same chi (e.g. the pure system used as chi = 0), their
clogs (area-weighted mean pressure, error between clogs) and their
avalanches (<s> and standard error over all of them) are pooled. The same
simulation indexed under two paths (identical data) is counted once.
--por_simulacion: one point per simulation instead.

Figure: a single panel, color and marker per system (estilo_figuras.py),
points without lines. A single orifice W (from _W_ in the folder name, --W,
default 2.0) is used so that <s> is comparable. --logy / --logx / --loglog:
logarithmic y axis, x axis or both (the pdf gets the suffix -logy, -logx or
-loglog).
--paneles: one panel per system, colored by chi.
--s_en_x: <s> on the x axis and the pressure on the y axis. --logx / --logy
still refer to the pressure and to <s>, respectively; the pdf is named
16-P-vs-s-... with suffix -logs, -logP or -loglog.

Inputs:
  resumen_presion_por_atasco.csv     (presion_tensor_arch.py)
  resultados_completos_esqueleto.csv (processor.py)
  master_sim_index.csv               (indexer.py: sim_hash <-> folder)
Simulations are matched by the last two folders of their path
(BARRIDO_.../Chi_..._W_...), so the index and the pressure file may have
been computed with the data at different locations.

Output (in processed_data/5_graficos_finales/ by default):
  16-s-vs-P-<region>-ratio-<r>-W-<W>_nuevo.pdf and ..._datos.csv with the plotted values.
The legend shows the area ratio R_a; the Pearson coefficient r of each
system is only printed (with --logy, computed with log<s>).

Usage (Fig. 16 of the paper):
    python3 analysis/3_presion_y_mapas/correlacion_s_vs_P.py --s_en_x --logy
    python3 analysis/3_presion_y_mapas/correlacion_s_vs_P.py --region roi --W 2.0 --radio_salida 1.0
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rutas  # noqa: E402
from estilo_figuras import (aplicar_estilo, estilo_sistema, guardar_figura,  # noqa: E402
                            ERRORBAR_KW, SYSTEM_NAME_MAP)
from comparar_presion_por_chi import filtrar_lotes_confirmados, _estadistica_pesada  # noqa: E402

COLUMNA_PRESION = {'P': 'SumP_reducida', 'tau': 'SumTau_reducida'}
SUFIJO_NUEVO = '_nuevo'
ETIQUETA_X = {'P': r'$\langle P\rangle d / mg$',
              'tau': r'$\langle \tau\rangle d / mg$'}
ETIQUETA_Y = {'big': r'Mean Avalanche Size $\langle s \rangle$',
              'total': r'Mean Avalanche Size $\langle s \rangle$ (all grains)'}
COLUMNA_S = {'big': 'tamano_big', 'total': 'total_mix'}
NOMBRE_REGION = {'arco': 'arch', 'roi': 'ROI', 'bulk': 'bulk'}


W_REGEX = re.compile(r'_W_([0-9]+(?:p[0-9]+)?)')
MARCADORES = ['o', 's', '^', 'D', 'v', 'p', 'h', '<', '>']


def ancho_W(clave):
    m = W_REGEX.search(str(clave))
    return float(m.group(1).replace('p', '.')) if m else np.nan


def clave_carpeta(ruta):
    """The last two folders of a path: 'BARRIDO_X/Chi_0p2_W_2p0'."""
    partes = os.path.normpath(str(ruta)).replace('\\', '/').rstrip('/').split('/')
    return '/'.join(partes[-2:])


def presion_por_simulacion(path_csv, magnitud, region):
    df = pd.read_csv(path_csv, dtype={'lote': str, 'jam_id': str})
    df = filtrar_lotes_confirmados(df, path_csv)
    col_sum, col_area = f"{COLUMNA_PRESION[magnitud]}_{region}", f"Area_{region}"
    faltan = [c for c in (col_sum, col_area) if c not in df.columns]
    if faltan:
        sys.exit(f"[!] The pressure csv lacks the columns {faltan}")
    filas, atascos, claves = [], [], set()
    for sim_path, g in df.groupby('sim_path'):
        if clave_carpeta(sim_path) in claves:
            print(f"  [warning] {sim_path}: same folder as another path already read, counted once")
            continue
        claves.add(clave_carpeta(sim_path))
        w = g[col_area].to_numpy(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            x = g[col_sum].to_numpy(float) / w
        media, std, sem, n, _ = _estadistica_pesada(x, w)
        clave = clave_carpeta(sim_path)
        filas.append(dict(clave=clave, sim_path=sim_path, P=media, P_sem=sem, n_atascos=n))
        atascos.append(pd.DataFrame({'clave': clave, 'x': x, 'w': w}))
    return pd.DataFrame(filas), (pd.concat(atascos, ignore_index=True) if atascos
                                 else pd.DataFrame(columns=['clave', 'x', 'w']))


def avalanchas_por_simulacion(path_av, path_idx, col_s, solo_con_arco):
    av = pd.read_csv(path_av)
    if solo_con_arco:
        if 'atasco_id' not in av.columns:
            sys.exit("[!] The avalanche file has no atasco_id column: run processor.py --emparejar.")
        av = av[av['atasco_id'].notna()]
    idx = pd.read_csv(path_idx)
    if 'area_ratio_nominal' not in idx.columns and 'area_ratio' in idx.columns:
        idx['area_ratio_nominal'] = idx['area_ratio'].round(3)
    idx = idx.drop_duplicates(subset=['sim_hash'])
    idx['clave'] = idx['path'].map(clave_carpeta)
    g = av.groupby('sim_hash')[col_s].agg(['mean', 'sem', 'count']).reset_index()
    g.columns = ['sim_hash', 's', 's_sem', 'n_avalanchas']
    cols = [c for c in ('sim_hash', 'clave', 'system_type', 'chi', 'area_ratio_nominal', 'Radio_salida')
            if c in idx.columns]
    return g.merge(idx[cols], on='sim_hash', how='inner'), av[['sim_hash', col_s]].rename(columns={col_s: 's'})


def agrupar_por_chi(df, atascos, avalanchas):
    """One point per (system, chi) from the per-simulation rows of df.
    Pressure: area-weighted mean of ALL the clogs of those simulations, with
    the error between clogs (_estadistica_pesada, as in
    comparar_presion_por_chi.py). <s>: mean and standard error of all their
    avalanches. Simulations with identical data (the same folder indexed
    twice) are counted once.
    """
    filas = []
    for (sis, chi), g in df.groupby(['sistema', 'chi'], sort=True):
        vistos, sims = set(), []
        for _, f in g.iterrows():
            a = avalanchas.loc[avalanchas['sim_hash'] == f['sim_hash'], 's'].to_numpy(float)
            p = atascos.loc[atascos['clave'] == f['clave'], 'x'].to_numpy(float)
            firma = (tuple(a), tuple(np.round(p, 10)))
            if firma in vistos:
                print(f"  [warning] {sis} chi = {chi:g}: {f['clave']} repeats the data of another folder, counted once")
                continue
            vistos.add(firma)
            sims.append(f)
        g = pd.DataFrame(sims)
        at = atascos[atascos['clave'].isin(g['clave'])]
        x, w = at['x'].to_numpy(float), at['w'].to_numpy(float)
        P, _, P_sem, n_at, _ = _estadistica_pesada(x, w)
        s_all = avalanchas.loc[avalanchas['sim_hash'].isin(g['sim_hash']), 's'].to_numpy(float)
        s_sem = s_all.std(ddof=1) / np.sqrt(len(s_all)) if len(s_all) > 1 else np.nan
        fila = dict(sistema=sis, system_type=g['system_type'].iloc[0], chi=chi, P=P, P_sem=P_sem,
                    n_atascos=n_at, s=s_all.mean(), s_sem=s_sem, n_avalanchas=len(s_all),
                    n_simulaciones=len(g), clave=' | '.join(g['clave']))
        for c in ('area_ratio_nominal', 'Radio_salida', 'W'):
            if c in g.columns:
                fila[c] = g[c].iloc[0] if g[c].nunique(dropna=False) == 1 else np.nan
        filas.append(fila)
    out = pd.DataFrame(filas)
    multi = out[out['n_simulaciones'] > 1]
    if len(multi):
        print("Points that pool several simulations (same system and chi):")
        for _, f in multi.iterrows():
            print(f"  {f['sistema']:20s} chi = {f['chi']:<6g} {f['n_simulaciones']} sims: {f['clave']}")
    return out


def agregar_puros(df):
    """Pure simulations (chi = 0, system_type without '-', e.g. 'Cuad') are
    added as the chi = 0 point of each mixture with the same large grain
    ('Cuad-Disco').
    """
    st = df['system_type'].astype(str)
    puros, mezclas = df[~st.str.contains('-')], df[st.str.contains('-')]
    extra = []
    for sis in mezclas['system_type'].unique():
        base = str(sis).split('-')[0]
        p = puros[puros['system_type'] == base]
        if len(p) and not ((mezclas['system_type'] == sis) & (mezclas['chi'] == 0)).any():
            extra.append(p.assign(system_type=sis))
    return pd.concat([mezclas] + extra, ignore_index=True) if len(mezclas) else df


def ejes_log(ax, logx, logy):
    """Log scale with plain numbers on the ticks (4, 10, 100 instead of 4x10^0).
    Ticks depend on the range: if it spans less than ~1.5 decades, 1..9 per
    decade (e.g. 50, 60, 70); otherwise 1, 2, 5.
    """
    from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator
    for usar, eje, set_scale, get_lim in ((logx, ax.xaxis, ax.set_xscale, ax.get_xlim),
                                          (logy, ax.yaxis, ax.set_yscale, ax.get_ylim)):
        if not usar:
            continue
        set_scale('log')
        lo, hi = get_lim()
        decadas = np.log10(hi / lo) if lo > 0 and hi > lo else 2.0
        subs = np.arange(1, 10) if decadas < 1.5 else (1.0, 2.0, 5.0)
        eje.set_major_locator(LogLocator(base=10, subs=subs))
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        eje.set_major_formatter(fmt)
        eje.set_minor_formatter(NullFormatter())


def pearson(x, y, logy):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if logy:
        ok = y > 0
        x, y = x[ok], np.log(y[ok])
    return np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan


def ejes_xy(magnitud, region, tipo_s, s_en_x):
    """(column, error, label) of the x and y axes."""
    pres = ('P', 'P_sem', ETIQUETA_X[magnitud])
    aval = ('s', 's_sem', ETIQUETA_Y[tipo_s])
    return (aval, pres) if s_en_x else (pres, aval)


def graficar_paneles(df, out, magnitud, region, tipo_s, logx, logy, error, chi_max, area_ratio,
                     s_en_x=False):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    aplicar_estilo()
    (xc, xe, xl), (yc, ye, yl) = ejes_xy(magnitud, region, tipo_s, s_en_x)
    if s_en_x:
        logx, logy = logy, logx
    sistemas = sorted(df['sistema'].unique(), key=lambda s: (s != 'Disco(G)-Disco(C)', s))
    ncol = 2 if len(sistemas) > 1 else 1
    nfil = int(np.ceil(len(sistemas) / ncol))
    fig, axes = plt.subplots(nfil, ncol, figsize=(6.2 * ncol, 4.6 * nfil), sharex=True, sharey=True,
                             squeeze=False)
    norm, cmap = Normalize(0, chi_max), plt.cm.viridis
    familias = sorted(df[['Radio_salida', 'W']].drop_duplicates().itertuples(index=False, name=None),
                      key=lambda t: (np.nan_to_num(t[0]), np.nan_to_num(t[1])))
    marcador = {f: MARCADORES[i % len(MARCADORES)] for i, f in enumerate(familias)}
    vistos = set()
    for ax, sis in zip(axes.flat, sistemas):
        sub = df[df['sistema'] == sis]
        for fam, g in sub.groupby(['Radio_salida', 'W'], dropna=False):
            fam = tuple(fam)
            g = g.sort_values('chi')
            m = marcador.get(fam, 'o')
            if error:
                ax.errorbar(g[xc], g[yc], xerr=g[xe], yerr=g[ye], fmt='none', ecolor='0.55',
                            elinewidth=0.8, capsize=2, zorder=2)
            lbl = None
            if len(familias) > 1 and fam not in vistos:
                lbl = f"$R_{{out}}$ = {fam[0]:g}, W = {fam[1]:g}"
                vistos.add(fam)
            ax.scatter(g[xc], g[yc], c=g['chi'], cmap=cmap, norm=norm, marker=m, s=55,
                       edgecolors='black', linewidths=0.6, zorder=3, label=lbl)
        ax.set_title(rf"{estilo_sistema(sis)[2]}  ($R_{{\rm a}} = {area_ratio:g}$)", fontsize='medium')
        ejes_log(ax, logx, logy)
    for ax in axes.flat[len(sistemas):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel(xl)
    for ax in axes[:, 0]:
        ax.set_ylabel(yl)
    if len(familias) > 1:
        fig.legend(loc='lower center', ncol=min(len(familias), 4), frameon=False, fontsize='small',
                   bbox_to_anchor=(0.45, -0.02))
    fig.tight_layout(rect=(0, 0.05 if len(familias) > 1 else 0, 0.9, 1))
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=r'$\chi$')
    return guardar_figura(out, fig)


FS_LEYENDA = 'large'


def orden_sistema(sis):
    """Legend order: triangle, square, pentagon, disk."""
    for k, clave in enumerate(('Tri', 'Cuad', 'Pent', 'Disco')):
        if str(sis).startswith(clave):
            return (k, str(sis))
    return (9, str(sis))


def graficar(df, out, magnitud, region, tipo_s, logx, logy, error, area_ratio, s_en_x=False):
    aplicar_estilo()
    (xc, xe, xl), (yc, ye, yl) = ejes_xy(magnitud, region, tipo_s, s_en_x)
    if s_en_x:
        logx, logy = logy, logx
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sistemas = sorted(df['sistema'].unique(), key=orden_sistema)
    for sis in sistemas:
        sub = df[df['sistema'] == sis].sort_values('chi')
        color, marker, etiqueta = estilo_sistema(sis)
        lbl = etiqueta
        xerr = sub[xe] if error else None
        yerr = sub[ye] if error else None
        ax.errorbar(sub[xc], sub[yc], xerr=xerr, yerr=yerr, color=color, marker=marker, ls='none',
                    ms=7, label=lbl, **ERRORBAR_KW)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ejes_log(ax, logx, logy)
    ax.legend(title=rf"$R_{{\rm a}} = {area_ratio:g}$", frameon=False,
              loc='best', fontsize=FS_LEYENDA, title_fontsize=FS_LEYENDA)
    return guardar_figura(out, fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--presion', default=os.path.join(rutas.PRESION_DIR, 'tensor_nuevo',
                                                      'resumen_presion_por_atasco.csv'),
                    help='resumen_presion_por_atasco.csv from presion_tensor_arch.py')
    ap.add_argument('--av', default=os.path.join(rutas.AVALANCHE_DIR, 'resultados_completos_esqueleto.csv'))
    ap.add_argument('--idx', default=rutas.MASTER_INDEX)
    ap.add_argument('--region', choices=['arco', 'roi', 'bulk'], default='arco')
    ap.add_argument('--magnitud', choices=['P', 'tau'], default='P', help='P (pressure) or tau (shear stress)')
    ap.add_argument('--s', choices=['big', 'total'], default='big',
                    help="big: large grains per avalanche (tamano_big, default); total: total_mix")
    ap.add_argument('--area_ratio', type=float, default=0.1, help='nominal area ratio (default 0.1)')
    ap.add_argument('--radio_salida', type=float, default=None, help='filter by Radio_salida; default: all')
    ap.add_argument('--chi_max', type=float, default=0.4)
    ap.add_argument('--sistemas', nargs='+', default=None,
                    help="system_type of the index (e.g. Disco-Disco Cuad-Disco); default: all")
    ap.add_argument('--solo_con_arco', action='store_true',
                    help='only avalanches that ended in a saved clog (with an arch)')
    ap.add_argument('--W', type=float, default=2.0,
                    help='orifice W from the folder name (_W_2p0 -> 2.0, default); one orifice per figure')
    ap.add_argument('--logx', action='store_true', help='logarithmic x axis (pressure)')
    ap.add_argument('--logy', action='store_true', help='logarithmic y axis (<s>; r computed with log<s>)')
    ap.add_argument('--loglog', action='store_true', help='both axes logarithmic (= --logx --logy)')
    ap.add_argument('--paneles', action='store_true',
                    help='one panel per system colored by chi (default: a single panel, '
                         'color and marker per system)')
    ap.add_argument('--s_en_x', action='store_true',
                    help='<s> on the x axis and the pressure on the y axis; '
                         '--logy still means logarithmic <s>')
    ap.add_argument('--sin_error', action='store_true', help='no error bars')
    ap.add_argument('--por_simulacion', action='store_true',
                    help='one point per simulation, without pooling those with the same system and chi')
    ap.add_argument('--datos', default=None,
                    help='<figure>_datos.csv of a previous run (one point per system and chi, '
                         'columns sistema, chi, P, P_sem, s, s_sem): plot from it without '
                         '--presion/--av/--idx. --chi_max is applied.')
    ap.add_argument('--fs_leyenda', default='large',
                    help="legend font size: small, medium, large (default), or a number in pt")
    ap.add_argument('--out', default=None,
                    help='output pdf (default: 5_graficos_finales/16-s-vs-<P|tau>-<region>-ratio-<r>-W-<W>.pdf)')
    args = ap.parse_args()
    global FS_LEYENDA
    try:
        FS_LEYENDA = float(args.fs_leyenda)
    except ValueError:
        FS_LEYENDA = args.fs_leyenda
    if args.loglog:
        args.logx = args.logy = True

    if args.datos is not None:
        df = pd.read_csv(args.datos)
        faltan = [c for c in ('sistema', 'chi', 'P', 'P_sem', 's', 's_sem') if c not in df.columns]
        if faltan:
            sys.exit(f"[!] {args.datos} lacks the columns {faltan}")
        df = df[df['chi'] <= args.chi_max + 1e-9].dropna(subset=['P', 's'])
        print(f"Plotting from {args.datos}: {len(df)} points.")
        for col in ('Radio_salida', 'W'):
            if col not in df.columns:
                df[col] = np.nan
        return _salida(df, args)

    pres, atascos = presion_por_simulacion(args.presion, args.magnitud, args.region)
    aval, avalanchas = avalanchas_por_simulacion(args.av, args.idx, COLUMNA_S[args.s], args.solo_con_arco)
    df = aval.merge(pres, on='clave', how='inner')
    print(f"Simulations: {len(pres)} with pressure, {len(aval)} with avalanches, {len(df)} with both.")
    if df.empty:
        print("  example keys (pressure):  ", pres['clave'].head(3).tolist())
        print("  example keys (avalanches):", aval['clave'].head(3).tolist())
        sys.exit("[!] No simulation matches between the pressure and the avalanche files.")
    if 'area_ratio_nominal' in df.columns and args.area_ratio is not None:
        df = df[np.isclose(df['area_ratio_nominal'].fillna(1.0), args.area_ratio, atol=1e-3)
                | (df['chi'] == 0)]
    df['W'] = df['clave'].map(ancho_W)
    if args.radio_salida is not None:
        df = df[np.isclose(df['Radio_salida'], args.radio_salida)]
    df = df[np.isclose(df['W'], args.W)]
    df = df[df['chi'] <= args.chi_max]
    if args.sistemas:
        df = df[df['system_type'].isin(args.sistemas)]
    df = df.dropna(subset=['P', 's'])
    df = agregar_puros(df)
    if df.empty:
        sys.exit("[!] No simulation left after filtering (--W, --area_ratio, --radio_salida, --chi_max).")
    df['sistema'] = df['system_type'].map(lambda s: SYSTEM_NAME_MAP.get(str(s), str(s)))
    if not args.por_simulacion:
        n_antes = len(df)
        df = agrupar_por_chi(df, atascos, avalanchas)
        print(f"{n_antes} simulations -> {len(df)} points (one per system and chi).")
    _salida(df, args)


def _salida(df, args):
    """Output name, csv with the points, Pearson coefficients and figure."""

    if args.s_en_x:
        nombre = f"16-{args.magnitud}-vs-s"
        sufijo = ('-loglog' if args.logx and args.logy else f'-log{args.magnitud}' if args.logx
                  else '-logs' if args.logy else '')
    else:
        nombre = f"16-s-vs-{args.magnitud}"
        sufijo = '-loglog' if args.logx and args.logy else '-logx' if args.logx else '-logy' if args.logy else ''
    out = args.out or os.path.join(
        rutas.GRAFICOS_FINALES_DIR,
        f"{nombre}-{args.region}-ratio-{args.area_ratio:g}-W-{args.W:g}{sufijo}{SUFIJO_NUEVO}.pdf")
    datos = os.path.splitext(out)[0] + '_datos.csv'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cols = ['sistema', 'chi', 'area_ratio_nominal', 'Radio_salida', 'W', 'n_simulaciones', 'clave', 'P', 'P_sem',
            'n_atascos', 's', 's_sem', 'n_avalanchas']
    df.sort_values(['sistema', 'chi'])[[c for c in cols if c in df.columns]].to_csv(datos, index=False)
    print("Points per system and family (Radio_salida, W): values of chi")
    for (sis, rs, w), g in df.groupby(['sistema', 'Radio_salida', 'W'], dropna=False):
        print(f"  {sis:20s} R_out={rs:<5g} W={w:<5g} chi={sorted(g['chi'].round(3).tolist())}")
    print("Pearson r (printed only, not shown in the figure):")
    for sis, g in df.groupby('sistema'):
        print(f"  {sis:20s} r = {pearson(g['P'], g['s'], args.logy):+.2f}")
    if args.paneles:
        path = graficar_paneles(df, out, args.magnitud, args.region, args.s, args.logx, args.logy,
                                not args.sin_error, args.chi_max, args.area_ratio, args.s_en_x)
    else:
        path = graficar(df, out, args.magnitud, args.region, args.s, args.logx, args.logy,
                        not args.sin_error, args.area_ratio, args.s_en_x)
    print(f"Figure: {path}\nData:   {datos}")


if __name__ == '__main__':
    main()
