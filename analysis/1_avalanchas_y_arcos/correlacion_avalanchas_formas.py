#!/usr/bin/env python3
"""Correlation between consecutive avalanches for the four mixtures with
R_a = 0.1 (triangle-, square-, pentagon- and disk-disk), including the pure
system (chi = 0) of each large-grain shape.

For each run:
  - first-return map s_{i+1} vs s_i and Spearman coefficient r_S with its
    p-value, also corrected for multiple comparisons (Benjamini-Hochberg over
    all the runs in the table);
  - autocorrelation function rho(k), k = 0..20, with the 95 % band expected
    for uncorrelated data, +-1.96/sqrt(N), using the N of each run;
  - block averaging: standard error of <s> with blocks of 1, 2, 4, 8 and 16
    consecutive avalanches.

For each shape and chi a single run is used (the one with most avalanches if
the index has more than one; a warning is printed). Avalanches are sorted
by t_inicio, i.e. in temporal order. s = tamano_big (number of large grains
discharged between two consecutive clogs; small grains are not counted).
The chi = 0 point is the pure system ('Tri', 'Cuad', ... in the index) with
the same orifice (w_salida) as the mixtures.

Outputs (in --out_dir, default processed_data/5_graficos_finales/SM_SII_correlacion/;
existing files are not overwritten unless --sobrescribir is given):
  primer_retorno_<Shape>-Disco_Ra0.1.pdf   s_{i+1} vs s_i (log-log), ~6 values of chi
  autocorrelacion_<Shape>-Disco_Ra0.1.pdf  rho(k), k = 0..20, 95 % band
  bloques_<Shape>-Disco_Ra0.1.pdf          SEM(b)/SEM(1) vs b
  tabla_correlacion_Ra0.1.csv / .md        one row per (shape, chi)
  resumen_correlacion_Ra0.1.md             summary (which cases are significant)

Usage:
  python3 correlacion_avalanchas_formas.py
  python3 correlacion_avalanchas_formas.py --formas Tri-Disco --chi_fig 0 0.05 0.1 0.2 0.3 0.4
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rutas  # noqa: E402
from estilo_figuras import aplicar_estilo, guardar_figura, ERRORBAR_KW, REF_LINE_KW, SEQ_STYLE  # noqa: E402
from sem_bloques import autocorrelacion  # noqa: E402

FORMAS = ['Tri-Disco', 'Cuad-Disco', 'Pent-Disco', 'Disco-Disco']
BLOQUES = [1, 2, 4, 8, 16]
Z95 = 1.96
CHI_FIG_DISCO = [0.01, 0.04, 0.08, 0.12, 0.16, 0.2]


def leer(av_path, index_path):
    av = pd.read_csv(av_path)
    faltan = [c for c in ('sim_hash', 'tamano_big') if c not in av.columns]
    if faltan:
        sys.exit(f"[!] {av_path} lacks the columns {faltan}")
    if 't_inicio' not in av.columns:
        print("[warning] the csv has no t_inicio column: the row order of each simulation is used "
              "(processor.py writes them in temporal order).")
        av['t_inicio'] = av.groupby('sim_hash').cumcount().astype(float)
    idx = pd.read_csv(index_path)
    meta = [c for c in ('sim_hash', 'system_type', 'chi', 'area_ratio_nominal', 'w_salida', 'path')
            if c in idx.columns]
    av = av.drop(columns=[c for c in meta if c != 'sim_hash' and c in av.columns]).merge(
        idx[meta].drop_duplicates('sim_hash'), on='sim_hash', how='left')
    return av


def secuencias(av):
    """{sim_hash: (metadata row, s in temporal order)}, without duplicates
    (the same simulation indexed twice gives the same sequence).
    """
    sims, vistas = {}, set()
    for h, g in av.groupby('sim_hash'):
        x = g.sort_values('t_inicio', kind='stable')['tamano_big'].to_numpy(float)
        firma = hash(tuple(x))
        if firma in vistas or len(x) < 10:
            continue
        vistas.add(firma)
        sims[h] = (g.iloc[0], x)
    return sims


def corridas_de_forma(av, forma, area_ratio, tol_ra):
    """One run per chi for the mixture `forma` (e.g. 'Tri-Disco') with the given
    R_a, plus the pure system of that large-grain shape as chi = 0.
    Returns a list of (chi, sim_hash, meta, x) sorted by chi.
    """
    base = forma.split('-')[0]
    mez = av[(av['system_type'] == forma)
             & np.isclose(av['area_ratio_nominal'].astype(float), area_ratio, atol=tol_ra)
             & (av['chi'].astype(float) > 0)]
    puro = av[(av['system_type'] == base) & np.isclose(av['chi'].astype(float), 0.0)]
    if 'w_salida' in av.columns and not mez.empty and not puro.empty:
        w = mez['w_salida'].dropna()
        if len(w):
            w0 = w.mode().iloc[0]
            puro_w = puro[np.isclose(puro['w_salida'].astype(float), float(w0))]
            if puro_w.empty:
                print(f"  [warning] {base}: no pure system with w_salida = {w0:g}; all pure runs are used")
            else:
                puro = puro_w
    sims = secuencias(pd.concat([mez, puro]))
    por_chi = {}
    for h, (meta, x) in sims.items():
        c = round(float(meta['chi']), 6)
        por_chi.setdefault(c, []).append((h, meta, x))
    out = []
    for c in sorted(por_chi):
        cand = por_chi[c]
        h, meta, x = max(cand, key=lambda t: len(t[2]))
        if len(cand) > 1:
            print(f"  [warning] {forma} chi = {c:g}: {len(cand)} runs, the longest one is used "
                  f"({h}, N = {len(x)}; the others have N = {sorted(len(t[2]) for t in cand)[:-1]})")
        out.append((c, h, meta, x))
    return out


def spearman_p(a, b):
    """(r_S, two-sided p-value). Uses scipy if available; otherwise the
    Student t approximation with N - 2 degrees of freedom (as scipy does).
    """
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(a, b)
        return float(r), float(p)
    except ImportError:
        from math import erfc, sqrt
        ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
        r = float(np.corrcoef(ra, rb)[0, 1])
        n = len(a)
        t = r * sqrt((n - 2) / max(1e-300, 1 - r * r))
        return r, erfc(abs(t) / sqrt(2))


def sem_bloques(x, bloques=BLOQUES):
    """{b: (n_blocks, SEM(b), error of SEM(b))}. Blocks of b consecutive
    avalanches; SEM(b) = std(block means)/sqrt(n_blocks). The error of
    SEM(b) is SEM(b)/sqrt(2 (n_blocks - 1)).
    """
    out = {}
    for b in bloques:
        nb = len(x) // b
        if nb < 4:
            continue
        m = x[:nb * b].reshape(nb, b).mean(axis=1)
        sem = m.std(ddof=1) / np.sqrt(nb)
        out[b] = (nb, sem, sem / np.sqrt(2 * (nb - 1)))
    return out


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    n = len(p)
    orden = np.argsort(p)
    q = p[orden] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[orden] = np.minimum(q, 1.0)
    return out


def analizar(forma, chi, h, x, k_max):
    n = len(x)
    r_s, p = spearman_p(x[:-1], x[1:])
    rho = autocorrelacion(x, k_max)
    banda = Z95 / np.sqrt(n)
    bl = sem_bloques(x)
    sem1 = bl[1][1]
    fila = dict(forma=forma, chi=chi, sim_hash=h, N=n, s_medio=x.mean(), sem=sem1,
                r_S=r_s, p_valor=p, rho1=rho[0], banda_95=banda,
                lags_fuera=int(np.sum(np.abs(rho) > banda)), k_max=len(rho),
                cociente_b8_b1=bl[8][1] / sem1 if 8 in bl else np.nan,
                err_cociente_b8=bl[8][2] / sem1 if 8 in bl else np.nan)
    for b in BLOQUES:
        fila[f'sem_b{b}'] = bl[b][1] if b in bl else np.nan
    return fila, rho, bl


def elegir_chi(chis, n=6, pedidos=None):
    """~n values of chi covering the range: the requested ones (nearest
    available), or chi = 0 (if present) and n-1 values spread between the
    smallest and largest chi > 0.
    """
    chis = np.array(sorted(chis))
    if pedidos:
        sel = [chis[np.argmin(np.abs(chis - c))] for c in pedidos]
    else:
        pos = chis[chis > 0]
        sel = [0.0] if np.any(np.isclose(chis, 0)) else []
        m = n - len(sel)
        if len(pos) <= m:
            sel += list(pos)
        else:
            for t in np.linspace(pos.min(), pos.max(), m):
                libres = [c for c in pos if c not in sel]
                sel.append(min(libres, key=lambda c: abs(c - t)))
    return sorted(set(float(c) for c in sel))


def figura_primer_retorno(corridas, stats, out):
    aplicar_estilo()
    n = len(corridas)
    ncol = min(3, n)
    nfil = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nfil, ncol, figsize=(3.6 * ncol, 3.5 * nfil), squeeze=False)
    for ax, (chi, h, meta, x) in zip(axes.flat, corridas):
        a, b = x[:-1], x[1:]
        ok = (a > 0) & (b > 0)
        ax.loglog(a[ok], b[ok], 'o', ms=3, alpha=0.5, color='black', mec='none')
        lo, hi = min(a[ok].min(), b[ok].min()), max(a[ok].max(), b[ok].max())
        ax.plot([lo, hi], [lo, hi], **REF_LINE_KW)
        ax.set_title(rf"$\chi = {chi:g}$,  $r_S = {stats[h]['r_S']:+.2f}$", fontsize='medium')
        ax.set_xlabel(r"$s_i$")
        ax.set_ylabel(r"$s_{i+1}$")
        ax.set_aspect('equal', adjustable='datalim')
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return guardar_figura(out, fig)


def figura_autocorrelacion(corridas, rhos, out, k_max):
    """rho(k) with the band +-1.96/sqrt(N): dark gray up to the band of the
    largest N, light gray up to that of the smallest N (if they differ). Points
    outside the band of their own run get a black edge.
    """
    aplicar_estilo()
    colores, marcadores = SEQ_STYLE(len(corridas))
    fig, ax = plt.subplots()
    ns = [len(x) for _, _, _, x in corridas]
    b_ang, b_anch = Z95 / np.sqrt(max(ns)), Z95 / np.sqrt(min(ns))
    if b_anch - b_ang > 1e-3:
        ax.axhspan(-b_anch, b_anch, color='0.92', lw=0)
    ax.axhspan(-b_ang, b_ang, color='0.85', lw=0)
    ax.axhline(0.0, **REF_LINE_KW)
    for (chi, h, meta, x), c, m in zip(corridas, colores, marcadores):
        rho = np.concatenate([[1.0], rhos[h]])
        k = np.arange(len(rho))
        ax.plot(k, rho, f'{m}-', color=c, ms=4, label=rf"$\chi = {chi:g}$")
        fuera = np.abs(rho) > Z95 / np.sqrt(len(x))
        fuera[0] = False
        if fuera.any():
            ax.plot(k[fuera], rho[fuera], m, ms=7, mfc='none', mec='black', mew=1.0, zorder=5)
    ax.set_xlim(-0.5, k_max + 0.5)
    ax.set_xlabel(r"Lag $k$ (avalanches)")
    ax.set_ylabel(r"$\rho(k)$")
    ax.legend(loc='best', fontsize='small', ncol=2)
    return guardar_figura(out, fig)


def figura_bloques(corridas, bloques, out):
    aplicar_estilo()
    colores, marcadores = SEQ_STYLE(len(corridas))
    fig, ax = plt.subplots()
    for (chi, h, meta, x), c, m in zip(corridas, colores, marcadores):
        bl = bloques[h]
        b = np.array(sorted(bl))
        s1 = bl[1][1]
        ax.errorbar(b, [bl[i][1] / s1 for i in b], yerr=[bl[i][2] / s1 for i in b],
                    fmt=f'{m}-', color=c, label=rf"$\chi = {chi:g}$", **ERRORBAR_KW)
    ax.axhline(1.0, **REF_LINE_KW)
    ax.set_xscale('log', base=2)
    ax.set_xticks(BLOQUES)
    ax.set_xticklabels([str(b) for b in BLOQUES])
    ax.set_xlabel(r"Block size $b$ (consecutive avalanches)")
    ax.set_ylabel(r"$\mathrm{SEM}(b)/\mathrm{SEM}(1)$")
    ax.legend(loc='best', fontsize='small', ncol=2)
    return guardar_figura(out, fig)


COLS_MD = [('forma', 'Shape', '{}'), ('chi', 'chi', '{:g}'), ('N', 'N', '{:d}'),
           ('s_medio', '<s>', '{:.1f}'), ('sem', 'SEM', '{:.2f}'), ('r_S', 'r_S', '{:+.3f}'),
           ('p_valor', 'p', '{:.3f}'), ('p_BH', 'p (BH)', '{:.3f}'), ('rho1', 'rho(1)', '{:+.3f}'),
           ('banda_95', '1.96/sqrt(N)', '{:.3f}'), ('lags_fuera', 'lags outside band (k=1..20)', '{:d}'),
           ('cociente_b8_b1', 'SEM(8)/SEM(1)', '{:.2f}')]


def tabla_md(df, cols):
    def fmt(v, f):
        try:
            return f.format(int(v) if f == '{:d}' else v) if pd.notna(v) else '-'
        except (ValueError, TypeError):
            return str(v)
    lineas = ['| ' + ' | '.join(t for _, t, _ in cols) + ' |',
              '|' + '|'.join('---' for _ in cols) + '|']
    for _, f in df.iterrows():
        lineas.append('| ' + ' | '.join(fmt(f[c], fm) for c, _, fm in cols) + ' |')
    return '\n'.join(lineas) + '\n'


def resumen(df, alpha=0.05):
    L = ["# Correlation between consecutive avalanches (R_a = 0.1)\n",
         f"{len(df)} runs ({', '.join(f'{f}: {n}' for f, n in df['forma'].value_counts().items())}), "
         f"N between {df['N'].min()} and {df['N'].max()} avalanches.\n"]
    sig = df[df['p_valor'] < alpha]
    sig_bh = df[df['p_BH'] < alpha]
    fuera = df[df['rho1'].abs() > df['banda_95']]
    esperado = alpha * len(df)
    L.append(f"- Spearman (s_i, s_i+1) with p < {alpha}: {len(sig)} of {len(df)} runs "
             f"(~{esperado:.1f} expected by chance).")
    for _, f in sig.iterrows():
        L.append(f"    - {f['forma']} chi = {f['chi']:g}: r_S = {f['r_S']:+.3f}, p = {f['p_valor']:.3g}, "
                 f"p(BH) = {f['p_BH']:.3g}, SEM(8)/SEM(1) = {f['cociente_b8_b1']:.2f} "
                 f"+- {f['err_cociente_b8']:.2f}")
    L.append(f"- After correcting for multiple comparisons (Benjamini-Hochberg over the "
             f"{len(df)} runs): {len(sig_bh)} with p(BH) < {alpha}"
             + (": " + ', '.join(f"{f['forma']} chi = {f['chi']:g}" for _, f in sig_bh.iterrows())
                if len(sig_bh) else "."))
    L.append(f"- |rho(1)| > 1.96/sqrt(N): {len(fuera)} of {len(df)}"
             + (": " + ', '.join(f"{f['forma']} chi = {f['chi']:g} (rho1 = {f['rho1']:+.3f}, band "
                                 f"{f['banda_95']:.3f})" for _, f in fuera.iterrows()) if len(fuera) else "."))
    tot_lags = df['k_max'].sum()
    L.append(f"- Lags k = 1..20 outside the band: {df['lags_fuera'].sum()} of {tot_lags} "
             f"(~{0.05 * tot_lags:.0f} expected by chance, 5 %).")
    c = df['cociente_b8_b1'].dropna()
    L.append(f"- SEM(8)/SEM(1): median {c.median():.2f}, range {c.min():.2f}-{c.max():.2f}; "
             f"expected noise for independent avalanches ~ +-{df['err_cociente_b8'].median():.2f} (1 sigma). "
             f"Runs with ratio > 1 + 2 sigma: "
             + (', '.join(f"{f['forma']} chi = {f['chi']:g} ({f['cociente_b8_b1']:.2f})"
                          for _, f in df[df['cociente_b8_b1'] > 1 + 2 * df['err_cociente_b8']].iterrows())
                or 'none') + '.')
    return '\n'.join(L) + '\n'


def no_pisar(path, sobrescribir):
    if os.path.exists(path) and not sobrescribir:
        sys.exit(f"[!] {path} already exists and is not overwritten (use another --out_dir or --sobrescribir).")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--av', default=os.path.join(rutas.AVALANCHE_DIR, 'resultados_completos_esqueleto.csv'))
    ap.add_argument('--index', default=rutas.MASTER_INDEX)
    ap.add_argument('--formas', nargs='+', default=FORMAS, help=f'system_type of the mixtures (default {FORMAS})')
    ap.add_argument('--area_ratio', type=float, default=0.1)
    ap.add_argument('--tol_ra', type=float, default=0.02)
    ap.add_argument('--chi_fig', type=float, nargs='+', default=None,
                    help='chi values shown in the figures (default: ~6 covering the range; disk-disk: CHI_FIG_DISCO)')
    ap.add_argument('--n_fig', type=int, default=6)
    ap.add_argument('--k_max', type=int, default=20)
    ap.add_argument('--out_dir', default=os.path.join(rutas.GRAFICOS_FINALES_DIR, 'SM_SII_correlacion'))
    ap.add_argument('--sobrescribir', action='store_true')
    args = ap.parse_args()

    for p in (args.av, args.index):
        if not os.path.exists(p):
            sys.exit(f"[!] {p} not found (run main_pipeline.py or pass --av / --index).")
    av = leer(args.av, args.index)
    os.makedirs(args.out_dir, exist_ok=True)
    ra = f"Ra{args.area_ratio:g}"

    filas = []
    for forma in args.formas:
        corr = corridas_de_forma(av, forma, args.area_ratio, args.tol_ra)
        if not corr:
            print(f"[!] {forma}: no runs with R_a = {args.area_ratio:g}")
            continue
        stats, rhos, bloques = {}, {}, {}
        for chi, h, meta, x in corr:
            f, rho, bl = analizar(forma, chi, h, x, args.k_max)
            stats[h], rhos[h], bloques[h] = f, rho, bl
            filas.append(f)
            if len(x) < 400:
                print(f"  [warning] {forma} chi = {chi:g}: only {len(x)} avalanches")
        pedidos = args.chi_fig or (CHI_FIG_DISCO if forma == 'Disco-Disco' else None)
        sel = elegir_chi([c for c, *_ in corr], args.n_fig, pedidos)
        fig_corr = [t for t in corr if any(np.isclose(t[0], s) for s in sel)]
        print(f"[+] {forma}: {len(corr)} values of chi in the table ({', '.join(f'{c:g}' for c, *_ in corr)}); "
              f"figures with chi = {', '.join(f'{c:g}' for c in sel)}")
        et = f"{forma}_{ra}"
        print("   ", figura_primer_retorno(fig_corr, stats, no_pisar(
            os.path.join(args.out_dir, f"primer_retorno_{et}.pdf"), args.sobrescribir)))
        print("   ", figura_autocorrelacion(fig_corr, rhos, no_pisar(
            os.path.join(args.out_dir, f"autocorrelacion_{et}.pdf"), args.sobrescribir), args.k_max))
        print("   ", figura_bloques(fig_corr, bloques, no_pisar(
            os.path.join(args.out_dir, f"bloques_{et}.pdf"), args.sobrescribir)))

    if not filas:
        sys.exit("[!] No runs to analyze.")
    df = pd.DataFrame(filas)
    df['p_BH'] = benjamini_hochberg(df['p_valor'])
    df['forma'] = pd.Categorical(df['forma'], [f for f in args.formas if f in set(df['forma'])], ordered=True)
    df = df.sort_values(['forma', 'chi']).reset_index(drop=True)
    df['forma'] = df['forma'].astype(str)

    csv = no_pisar(os.path.join(args.out_dir, f"tabla_correlacion_{ra}.csv"), args.sobrescribir)
    df.to_csv(csv, index=False)
    md = no_pisar(os.path.join(args.out_dir, f"tabla_correlacion_{ra}.md"), args.sobrescribir)
    with open(md, 'w') as fh:
        fh.write(f"# Consecutive avalanches, R_a = {args.area_ratio:g}\n\n"
                 "SEM = std(s)/sqrt(N). p (BH): p-value corrected with Benjamini-Hochberg over all the "
                 "rows. 95 % band = 1.96/sqrt(N). SEM(8)/SEM(1): blocks of 8 consecutive avalanches.\n\n")
        fh.write(tabla_md(df, COLS_MD))
    res = resumen(df)
    rpath = no_pisar(os.path.join(args.out_dir, f"resumen_correlacion_{ra}.md"), args.sobrescribir)
    with open(rpath, 'w') as fh:
        fh.write(res)
    print()
    print(tabla_md(df, COLS_MD))
    print(res)
    print(f"Tables: {csv}\n        {md}\nSummary: {rpath}")


if __name__ == '__main__':
    main()
