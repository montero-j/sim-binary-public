#!/usr/bin/env python3
"""Block-averaged standard error of <s> (Flyvbjerg-Petersen) and
autocorrelation of consecutive avalanches, to check whether the naive SEM
(std/sqrt(n), used by flow_analysis.py) underestimates the error because of
correlations within a run.

For each simulation the avalanches are sorted by t_inicio and grouped in
blocks of b = 1, 2, 4, 8, ... consecutive avalanches.
SEM(b) = std of the block means / sqrt(number of blocks). If the avalanches
are independent, SEM(b) = SEM(1) for every b (within noise); if they are
correlated, SEM(b) grows up to a plateau, which is the correct error. Only
block sizes with at least --min_bloques blocks are used.

Also: autocorrelation rho(k) of s at a distance of k avalanches and
integrated autocorrelation time tau_int = 1/2 + sum_k rho(k) (up to the
first rho <= 0); for independent avalanches tau_int = 1/2 and the SEM
correction factor sqrt(2 tau_int) is 1.

Outputs (in 5_graficos_finales/ by default):
  sem_bloques_<label>.csv   one row per simulation: n, naive SEM,
                            block SEM (plateau), ratio, rho(1), tau_int
  sem_bloques_<label>.pdf   SEM(b)/SEM(1) vs b, one curve per chi

Usage (disk-disk, R_a = 0.1):
  python3 sem_bloques.py --sistema Disco-Disco --area_ratio 0.1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rutas  # noqa: E402
from estilo_figuras import aplicar_estilo, guardar_figura, ERRORBAR_KW, REF_LINE_KW, SEQ_STYLE  # noqa: E402


def sem_por_bloques(x, min_bloques=16):
    """List of (b, n_blocks, SEM(b), error of SEM(b)) for b = 1, 2, 4, ...
    The error of SEM(b) is SEM(b) / sqrt(2 (n_blocks - 1)).
    """
    x = np.asarray(x, dtype=float)
    out, b = [], 1
    while len(x) // b >= min_bloques:
        nb = len(x) // b
        medias = x[:nb * b].reshape(nb, b).mean(axis=1)
        sem = medias.std(ddof=1) / np.sqrt(nb)
        out.append((b, nb, sem, sem / np.sqrt(2 * (nb - 1))))
        b *= 2
    return out


def autocorrelacion(x, k_max=50):
    """rho(k) for k = 1..k_max (estimator normalized by the variance)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    var = np.mean(x * x)
    if var == 0:
        return np.zeros(0)
    k_max = min(k_max, len(x) // 4)
    return np.array([np.mean(x[:-k] * x[k:]) / var for k in range(1, k_max + 1)])


def tau_integrado(rho):
    """1/2 + sum of rho(k) up to the first rho <= 0 (simple window cut)."""
    tau = 0.5
    for r in rho:
        if r <= 0:
            break
        tau += r
    return tau


def analizar(df_av, min_bloques):
    filas, curvas = [], {}
    for h, g in df_av.groupby('sim_hash'):
        x = g.sort_values('t_inicio', kind='stable')['tamano_big'].to_numpy(float)
        if len(x) < 2 * min_bloques:
            continue
        bl = sem_por_bloques(x, min_bloques)
        sem1 = bl[0][2]
        sem_pl, err_pl = max(bl, key=lambda t: t[0])[2:4]
        rho = autocorrelacion(x)
        tau = tau_integrado(rho)
        meta = g.iloc[0]
        filas.append(dict(sim_hash=h, system_type=meta.get('system_type'), chi=meta.get('chi'),
                          area_ratio_nominal=meta.get('area_ratio_nominal'), n_avalanchas=len(x),
                          s_medio=x.mean(), sem_ingenuo=sem1, sem_bloques=sem_pl,
                          err_sem_bloques=err_pl, cociente=sem_pl / sem1,
                          b_max=max(t[0] for t in bl),
                          n_bloques=max(bl, key=lambda t: t[0])[1],
                          rho1=rho[0] if len(rho) else np.nan, tau_int=tau,
                          sem_tau=sem1 * np.sqrt(2 * tau)))
        curvas[h] = bl
    return pd.DataFrame(filas), curvas


def imprimir_tabla(res, curvas):
    """One row per simulation: chi, <s>, SEM(b) for each block size, rho_1."""
    bs = sorted({t[0] for bl in curvas.values() for t in bl})
    filas = []
    for _, f in res.sort_values(['system_type', 'chi']).iterrows():
        sem = {t[0]: t[2] for t in curvas[f['sim_hash']]}
        fila = {'sistema': f['system_type'], 'chi': f'{f["chi"]:g}', '<s>': f'{f["s_medio"]:.1f}'}
        fila.update({f'SEM(b={b})': (f'{sem[b]:.2f}' if b in sem else '-') for b in bs})
        fila['rho_1'] = f'{f["rho1"]:+.3f}'
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    if tabla['sistema'].nunique() == 1:
        tabla = tabla.drop(columns='sistema')
    print(tabla.to_string(index=False))


def graficar_resumen(res, out):
    """Two panels, one point per simulation vs chi, with the noise band
    expected for independent avalanches:
      (a) SEM(b_max)/SEM(1), band 1 +- 2/sqrt(2 (n_blocks - 1))  (95 %)
      (b) rho(1),            band +- 2/sqrt(n_avalanches) (95 %)
    """
    aplicar_estilo()
    res = res.sort_values('chi')
    chi = res['chi'].to_numpy(float)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.4))
    pad = 0.02 * (chi.max() - chi.min() or 1)
    x_banda = np.array([chi.min() - pad, chi.max() + pad])

    ancho = np.median(2.0 / np.sqrt(2 * (res['n_bloques'].to_numpy(float) - 1)))
    ax1.fill_between(x_banda, 1 - ancho, 1 + ancho, color='0.85', lw=0)
    ax1.axhline(1.0, **REF_LINE_KW)
    ax1.plot(chi, res['cociente'], 'o', color='black')
    lim = max(1.6 * ancho, 1.1 * np.nanmax(np.abs(res['cociente'] - 1)))
    ax1.set_ylim(1 - lim, 1 + lim)
    b = int(res['b_max'].median())
    ax1.set_ylabel(rf"$\mathrm{{SEM}}({b})/\mathrm{{SEM}}(1)$")

    ancho_rho = np.median(2.0 / np.sqrt(res['n_avalanchas'].to_numpy(float)))
    ax2.fill_between(x_banda, -ancho_rho, ancho_rho, color='0.85', lw=0)
    ax2.axhline(0.0, **REF_LINE_KW)
    ax2.plot(chi, res['rho1'], 'o', color='black')
    lim = max(1.6 * ancho_rho, 1.1 * np.nanmax(np.abs(res['rho1'])))
    ax2.set_ylim(-lim, lim)
    ax2.set_ylabel(r"$\rho_1$")
    ax2.set_xlabel(r"Concentration $\chi$")
    ax2.set_xlim(*x_banda)
    for ax, letra in ((ax1, '(a)'), (ax2, '(b)')):
        ax.text(0.98, 0.95, letra, transform=ax.transAxes, ha='right', va='top')
    fig.tight_layout()
    return guardar_figura(out, fig)


def graficar(res, curvas, out):
    aplicar_estilo()
    fig, ax = plt.subplots()
    res = res.sort_values('chi')
    colores, marcadores = SEQ_STYLE(len(res))
    for (_, f), c, m in zip(res.iterrows(), colores, marcadores):
        bl = curvas[f['sim_hash']]
        b = np.array([t[0] for t in bl])
        sem1 = bl[0][2]
        ax.errorbar(b, [t[2] / sem1 for t in bl], yerr=[t[3] / sem1 for t in bl],
                    fmt=f'{m}-', color=c, label=rf"$\chi = {f['chi']:g}$", **ERRORBAR_KW)
    ax.axhline(1.0, **REF_LINE_KW)
    ax.set_xscale('log', base=2)
    ax.set_xlabel(r"Block size $b$ (consecutive avalanches)")
    ax.set_ylabel(r"$\mathrm{SEM}(b)/\mathrm{SEM}(1)$")
    ax.legend(loc='best', fontsize='small', ncol=2)
    return guardar_figura(out, fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--av', default=os.path.join(rutas.AVALANCHE_DIR, 'resultados_completos_esqueleto.csv'),
                    help='avalanches, one per row (processor.py): sim_hash, t_inicio, tamano_big')
    ap.add_argument('--index', default=rutas.MASTER_INDEX)
    ap.add_argument('--sistema', nargs='+', default=None, help='system_type (e.g. Disco-Disco Cuad-Disco)')
    ap.add_argument('--area_ratio', type=float, nargs='+', default=None)
    ap.add_argument('--chi', type=float, nargs='+', default=None)
    ap.add_argument('--min_bloques', type=int, default=16,
                    help='minimum number of blocks to use a block size (default 16)')
    ap.add_argument('--tabla', action='store_true',
                    help='only print the table (SEM for each block size); nothing is saved')
    ap.add_argument('--curvas', action='store_true',
                    help='also the figure SEM(b)/SEM(1) vs b with one curve per simulation')
    ap.add_argument('--etiqueta', default=None, help='suffix of the output files')
    ap.add_argument('--out_dir', default=rutas.GRAFICOS_FINALES_DIR)
    args = ap.parse_args()

    av = pd.read_csv(args.av)
    faltan = [c for c in ('sim_hash', 'tamano_big') if c not in av.columns]
    if faltan:
        sys.exit(f"[!] {args.av} lacks the columns {faltan}")
    if 't_inicio' not in av.columns:
        if not args.tabla:
            print("[warning] the csv has no t_inicio column: the row order within each simulation is used.")
        av['t_inicio'] = av.groupby('sim_hash').cumcount().astype(float)
    idx = pd.read_csv(args.index)
    meta = [c for c in ('sim_hash', 'system_type', 'chi', 'area_ratio_nominal') if c in idx.columns]
    av = av.drop(columns=[c for c in meta if c != 'sim_hash' and c in av.columns]).merge(
        idx[meta].drop_duplicates('sim_hash'), on='sim_hash', how='left')
    if args.sistema:
        av = av[av['system_type'].isin(args.sistema)]
    if args.area_ratio:
        av = av[np.isclose(av['area_ratio_nominal'].to_numpy(float)[:, None], args.area_ratio).any(axis=1)]
    if args.chi:
        av = av[np.isclose(av['chi'].to_numpy(float)[:, None], args.chi).any(axis=1)]
    if av.empty:
        sys.exit("[!] No avalanches left after filtering (--sistema / --area_ratio / --chi).")

    firmas = av.sort_values(['sim_hash', 't_inicio'], kind='stable').groupby('sim_hash')['tamano_big'] \
               .apply(lambda x: hash(tuple(x)))
    dup = firmas[firmas.duplicated()].index
    if len(dup):
        if not args.tabla:
          print(f"[warning] {len(dup)} sim_hash with the same avalanche sequence as another one (duplicates "
              f"in the index): counted once: {', '.join(dup)}")
        av = av[~av['sim_hash'].isin(dup)]

    res, curvas = analizar(av, args.min_bloques)
    if res.empty:
        sys.exit(f"[!] No simulation has at least {2 * args.min_bloques} avalanches.")
    if args.tabla:
        imprimir_tabla(res, curvas)
        return
    etiqueta = args.etiqueta or '_'.join(filter(None, [
        '-'.join(args.sistema) if args.sistema else 'todos',
        f"Ra{'-'.join(f'{r:g}' for r in args.area_ratio)}" if args.area_ratio else None]))
    os.makedirs(args.out_dir, exist_ok=True)
    csv = os.path.join(args.out_dir, f"sem_bloques_{etiqueta}.csv")
    res.sort_values(['system_type', 'chi']).to_csv(csv, index=False)

    cols = ['sim_hash', 'system_type', 'chi', 'n_avalanchas', 's_medio', 'sem_ingenuo', 'sem_bloques',
            'cociente', 'b_max', 'rho1', 'tau_int']
    with pd.option_context('display.width', 200):
        print(res.sort_values(['system_type', 'chi'])[cols].to_string(index=False, float_format='%.3g'))
    print(f"\nRatio SEM(blocks)/SEM(naive): median {res['cociente'].median():.2f}, "
          f"range {res['cociente'].min():.2f}-{res['cociente'].max():.2f}")
    print(f"rho(1): median {res['rho1'].median():.3f}, range {res['rho1'].min():.3f} to {res['rho1'].max():.3f}")
    print(f"Table: {csv}")
    pdf = graficar_resumen(res, os.path.join(args.out_dir, f"sem_bloques_resumen_{etiqueta}.pdf"))
    print(f"Summary figure: {pdf}")
    if args.curvas:
        pdf = graficar(res, curvas, os.path.join(args.out_dir, f"sem_bloques_{etiqueta}.pdf"))
        print(f"SEM(b) figure per simulation: {pdf}")


if __name__ == '__main__':
    main()
