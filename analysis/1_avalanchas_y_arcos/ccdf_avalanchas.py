#!/usr/bin/env python3
"""Avalanche-size distributions: empirical complementary cumulative
distribution P(S >= s) and exponential fit of its tail, for the four
mixtures with R_a = 0.1 (triangle-, square-, pentagon- and disk-disk) at
chi = 0 (pure system, with the same orifice as the mixtures) and chi = 0.2.
Data are read as in figuras_suplementario.py (resultados_completos_esqueleto.csv
and master_sim_index.csv; duplicated simulations are counted once).

s = tamano_big: number of LARGE grains discharged between two clogs.

  - Empirical CCDF, without binning: for each distinct value of s,
    (number of avalanches with size >= s) / N. The avalanches of all the runs
    with the same (shape, chi) are pooled.
  - Tail fit, s > frac_cola <s> (default <s>/2), to P(S >= s) = A exp(-s/s*):
    least squares of ln P vs s over the distinct values of s with
    P >= p_min_n / N. Error of s* by bootstrap over avalanches. As a check,
    the maximum-likelihood estimator of an exponential (geometric)
    distribution above the cut: s*_ML = 1 / ln(1 + 1/<s - s_c>).
  - Small avalanches:
      excess = 1 - A exp(-s_min/s*): fraction of avalanches not accounted for
               by the tail exponential extrapolated down to s_min = 1
               (0 for a pure exponential; > 0: excess of small avalanches).
      f(s < <s>/2), to compare with 1 - exp(-1/2) = 0.39 for a pure
               exponential with the same mean.
    An excess of small avalanches lowers <s> without changing the tail, so
    it also shows up as s*/<s> > 1.

Outputs (in --out_dir, default processed_data/5_graficos_finales/; existing
files are not overwritten unless --sobrescribir is given):
  SM-avalanche-ccdf.pdf           (a) P(S>=s) vs s, chi = 0.2; (b) vs s/<s>, chi = 0 and 0.2, with exp(-x)
  SM-avalanche-ccdf_tabla.csv/.md shape, chi, N, <s>, s*, s*/<s>, ...

Usage:
  python3 ccdf_avalanchas.py
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
from figuras_suplementario import gf, CFG, AR_COL, SISTEMAS, con_ra, color, etiqueta  # noqa: E402

CHIS = (0.0, 0.2)
CHI_TOL = 1e-3


def leer_avalanchas(av_path, index_path):
    av = pd.read_csv(av_path)
    idx = pd.read_csv(index_path)
    meta = [c for c in ('sim_hash', 'system_type', 'chi', AR_COL, 'w_salida') if c in idx.columns]
    av = av.drop(columns=[c for c in meta if c != 'sim_hash' and c in av.columns]).merge(
        idx[meta].drop_duplicates('sim_hash'), on='sim_hash', how='left')
    av['sistema'] = av['system_type'].map(gf.SYSTEM_NAME_MAP)
    chi = av['chi'].astype(float)
    mez = con_ra(av[av['sistema'].isin(SISTEMAS) & (chi > 0)])
    partes = [mez[np.isclose(mez['chi'].astype(float), c, atol=CHI_TOL)] for c in CHIS if c > 0]
    for sis in SISTEMAS:
        base = sis.split('-')[0]
        p = av[(av['sistema'] == base) & np.isclose(chi, 0.0)]
        if p.empty:
            print(f"  [!] no pure system {base} (chi = 0)")
            continue
        w = mez.loc[mez['sistema'] == sis, 'w_salida'].dropna() if 'w_salida' in av.columns else []
        if len(w):
            pw = p[np.isclose(p['w_salida'].astype(float), float(w.mode().iloc[0]))]
            p = pw if not pw.empty else p
        partes.append(p.assign(sistema=sis))
    av = pd.concat(partes, ignore_index=True)
    firmas = av.groupby('sim_hash')['tamano_big'].apply(lambda x: hash(tuple(x)))
    av = av[~av['sim_hash'].isin(firmas[firmas.duplicated()].index)]
    av['chi_nom'] = np.where(np.isclose(av['chi'].astype(float), 0.0), 0.0, av['chi'].astype(float).round(3))
    return av[av['tamano_big'] > 0]


def ccdf(s):
    """Distinct values of s and the empirical P(S >= s) (no binning)."""
    s = np.sort(np.asarray(s, float))
    u, primero = np.unique(s, return_index=True)
    return u, (len(s) - primero) / len(s)


def ajuste_cola(s, frac_cola=0.5, p_min_n=5):
    """(A, s*) of ln P(S>=s) = ln A - s/s* for s > frac_cola <s> and P >= p_min_n/N."""
    u, P = ccdf(s)
    sc = frac_cola * np.mean(s)
    ok = (u > sc) & (P >= p_min_n / len(s))
    if ok.sum() < 3:
        return np.nan, np.nan
    pend, ord0 = np.polyfit(u[ok], np.log(P[ok]), 1)
    return (np.exp(ord0), -1.0 / pend) if pend < 0 else (np.nan, np.nan)


def s_star_mv(s, frac_cola=0.5):
    s = np.asarray(s, float)
    sc = frac_cola * s.mean()
    exc = s[s > sc] - s[s > sc].min()
    m = exc.mean() if len(exc) else np.nan
    return 1.0 / np.log1p(1.0 / m) if m > 0 else np.nan


def analizar(s, frac_cola, p_min_n, n_boot, rng):
    s = np.asarray(s, float)
    media = s.mean()
    A, ss = ajuste_cola(s, frac_cola, p_min_n)
    boot = []
    for _ in range(n_boot):
        _, b = ajuste_cola(rng.choice(s, len(s)), frac_cola, p_min_n)
        boot.append(b)
    boot = np.array(boot)
    err = np.nanstd(boot, ddof=1) if np.isfinite(boot).sum() > 2 else np.nan
    sc = frac_cola * media
    f_emp = np.mean(s < sc)
    exceso = 1.0 - A * np.exp(-s.min() / ss) if np.isfinite(ss) else np.nan
    return dict(N=len(s), s_medio=media, sem=s.std(ddof=1) / np.sqrt(len(s)),
                s_star=ss, err_s_star=err, s_star_sobre_s=ss / media, err_cociente=err / media,
                s_star_MV=s_star_mv(s, frac_cola), A=A, s_corte=sc,
                f_chicas_emp=f_emp, f_chicas_exp=1 - np.exp(-frac_cola),
                exceso_chicas=exceso, cv=s.std(ddof=1) / media)


def figura(av, out, presentes):
    gf.aplicar_estilo()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.8))
    for sis in presentes:
        c, m = color(sis)
        for chi, ls, lleno in ((0.2, '-', True), (0.0, '--', False)):
            s = av.loc[(av['sistema'] == sis) & np.isclose(av['chi_nom'], chi), 'tamano_big'].to_numpy(float)
            if len(s) == 0:
                continue
            u, P = ccdf(s)
            kw = dict(color=c, marker=m, ms=5, markevery=max(1, len(u) // 12),
                      mfc=c if lleno else 'none', mec=c, lw=1.5, ls=ls)
            if chi == 0.2:
                axa.plot(u, P, label=etiqueta(sis), **kw)
            axb.plot(u / s.mean(), P, label=f"{etiqueta(sis)}, $\\chi = {chi:g}$", **kw)
    x_max = min(axb.get_xlim()[1], 8)
    xx = np.linspace(0, x_max, 200)
    axb.plot(xx, np.exp(-xx), color='black', ls=':', lw=2, label=r'$e^{-x}$', zorder=5)
    n_max = av.groupby(['sistema', 'chi_nom']).size().max()
    for ax in (axa, axb):
        ax.set_yscale('log')
        ax.set_ylabel(r"$P(S \geq s)$")
        ax.set_ylim(0.5 / n_max, 1.3)
        ax.set_xlim(left=0)
    axb.set_xlim(right=x_max)
    axa.set_xlabel(r"Avalanche Size $s$")
    axb.set_xlabel(r"$s/\langle s \rangle$")
    axa.legend(title=r'$\chi = 0.2$', loc='lower left', fontsize='small', title_fontsize='small')
    axb.legend(loc='lower left', fontsize='x-small', ncol=2)
    for ax, let in ((axa, '(a)'), (axb, '(b)')):
        ax.text(0.97, 0.97, let, transform=ax.transAxes, ha='right', va='top', fontsize='large')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


COLS_MD = [('forma', 'Shape', '{}'), ('chi', 'chi', '{:g}'), ('N', 'N', '{:d}'),
           ('s_medio', '<s>', '{:.1f}'), ('s_star', 's*', '{:.1f}'), ('err_s_star', 'err s*', '{:.1f}'),
           ('s_star_sobre_s', 's*/<s>', '{:.2f}'), ('s_star_MV', 's* (ML)', '{:.1f}'), ('A', 'A', '{:.2f}'),
           ('exceso_chicas', 'excess small', '{:+.2f}'), ('f_chicas_emp', 'f(s<<s>/2)', '{:.2f}'),
           ('cv', 'std/<s>', '{:.2f}')]


def tabla_md(df):
    def fmt(v, f):
        if pd.isna(v):
            return '-'
        return f.format(int(v)) if f == '{:d}' else f.format(v)
    L = ['| ' + ' | '.join(t for _, t, _ in COLS_MD) + ' |', '|' + '|'.join('---' for _ in COLS_MD) + '|']
    L += ['| ' + ' | '.join(fmt(f[c], fm) for c, _, fm in COLS_MD) + ' |' for _, f in df.iterrows()]
    return '\n'.join(L) + '\n'


def no_pisar(path, sobrescribir):
    if os.path.exists(path) and not sobrescribir:
        sys.exit(f"[!] {path} already exists and is not overwritten (use another --out_dir or --sobrescribir).")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--av', default=CFG.raw_flow_path)
    ap.add_argument('--index', default=gf.rutas.MASTER_INDEX)
    ap.add_argument('--frac_cola', type=float, default=0.5, help='the tail is s > frac_cola * <s> (default 0.5)')
    ap.add_argument('--p_min_n', type=float, default=5,
                    help='fit only points with P >= p_min_n / N (default 5)')
    ap.add_argument('--n_boot', type=int, default=500)
    ap.add_argument('--out_dir', default=CFG.out_dir)
    ap.add_argument('--sobrescribir', action='store_true')
    args = ap.parse_args()

    for p in (args.av, args.index):
        if not os.path.exists(p):
            sys.exit(f"[!] {p} not found (run main_pipeline.py or pass --av / --index).")
    pdf = no_pisar(os.path.join(args.out_dir, "SM-avalanche-ccdf.pdf"), args.sobrescribir)
    av = leer_avalanchas(args.av, args.index)
    presentes = [s for s in SISTEMAS if s in set(av['sistema'])]
    if not presentes:
        sys.exit("[!] no avalanches with R_a = 0.1 at chi = 0 / 0.2")

    rng = np.random.default_rng(12345)
    filas = []
    for sis in presentes:
        for chi in CHIS:
            g = av[(av['sistema'] == sis) & np.isclose(av['chi_nom'], chi)]
            if g.empty:
                print(f"  [!] {etiqueta(sis)}: no avalanches at chi = {chi:g}")
                continue
            f = analizar(g['tamano_big'].to_numpy(float), args.frac_cola, args.p_min_n, args.n_boot, rng)
            filas.append(dict(forma=etiqueta(sis), chi=chi, n_corridas=g['sim_hash'].nunique(), **f))
    df = pd.DataFrame(filas)

    print(f"[+] {figura(av, pdf, presentes)}")
    base = os.path.join(args.out_dir, "SM-avalanche-ccdf")
    csv = no_pisar(base + "_tabla.csv", args.sobrescribir)
    df.to_csv(csv, index=False)
    md = no_pisar(base + "_tabla.md", args.sobrescribir)
    with open(md, 'w') as fh:
        fh.write(f"# Avalanche-size distributions, R_a = 0.1\n\nTail fit P(S>=s) = A exp(-s/s*) for "
                 f"s > {args.frac_cola:g}<s> and P >= {args.p_min_n:g}/N; err s* from {args.n_boot} bootstrap "
                 f"resamples. s* (ML): maximum-likelihood exponential above the same cut. "
                 f"excess small = 1 - A exp(-s_min/s*): fraction of avalanches not accounted for by the "
                 f"tail exponential extrapolated down to s_min (0 for a pure exponential). "
                 f"f(s<<s>/2): measured fraction below the cut (pure exponential: "
                 f"{1 - np.exp(-args.frac_cola):.2f}). std/<s> = 1 for an exponential.\n\n")
        fh.write(tabla_md(df))
    print()
    print(tabla_md(df))
    print(f"Tables: {csv}\n        {md}")


if __name__ == '__main__':
    main()
