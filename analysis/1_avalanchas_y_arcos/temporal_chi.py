import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import rutas
from estilo_figuras import aplicar_estilo, estilo_sistema, guardar_figura, REF_LINE_KW

FUENTES = {'axes.labelsize': 26, 'xtick.labelsize': 22, 'ytick.labelsize': 22,
           'legend.fontsize': 18, 'legend.title_fontsize': 18}


def _estilo():
    aplicar_estilo()
    plt.rcParams.update(FUENTES)
    plt.rcParams['axes.grid'] = False


_estilo()

COLORES_CASOS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#000000']
SUAVIZADO_BATCHES = 5

INDEX_FILE = rutas.MASTER_INDEX
WINDOW_SIZE = 10.0
CPU_CORES = cpu_count()
FORMATO = 'pdf'
PREFIJO_VIEJO = None
PREFIJO_NUEVO = None
T_MIN = None
T_MAX = None


def area_grano(radio, n_lados):
    """Area of a grain of radius r: disk (n_lados = 1) or regular polygon with
    n_lados vertices on the circle of radius r.
    """
    n = int(n_lados) if n_lados else 1
    if n <= 2:
        return np.pi * radio ** 2
    return 0.5 * n * radio ** 2 * np.sin(2 * np.pi / n)


def especie_chica(areas_unitarias):
    """Index of the additive species: type 1, i.e. the second grain line of
    params.in (as written by the sweep scripts). It is not chosen by size,
    which would fail for equal radii (disk-disk, R_a = 1).
    """
    return 1 if 1 in areas_unitarias else min(areas_unitarias)


def calcular_chi_temporal(row):
    """Discharge chi (area fraction of small grains) per time window of
    WINDOW_SIZE s. Returns (info, df), with info['status'] = 'ok' or the reason
    why it could not be computed; df has columns tiempo, chi_eff (or None).
    """
    path = row['path']
    if PREFIJO_VIEJO and str(path).startswith(PREFIJO_VIEJO):
        path = PREFIJO_NUEVO + str(path)[len(PREFIJO_VIEJO):]
    flux_filename = row['fluxFile']
    sim_hash = row['sim_hash']
    tirada = row['tirada']
    base = {'sim_hash': sim_hash, 'tirada': tirada, 'path': path}

    if not os.path.isdir(path):
        return {**base, 'status': 'dir_no_existe'}, None

    try:
        grain_info = json.loads(row['grain_json'])
    except Exception as e:
        return {**base, 'status': 'json_invalido', 'detail': str(e)}, None

    chi_nominal = row['chi']

    if len(grain_info) < 2:
        return {**base, 'status': 'monodisperso_omitido'}, None

    radios = [g['radius'] for g in grain_info]
    areas_unitarias = {i: area_grano(g['radius'], g.get('sides', 1)) for i, g in enumerate(grain_info)}
    tipo_chica = especie_chica(areas_unitarias)

    flux_path = os.path.join(path, flux_filename)
    if not os.path.exists(flux_path):
        return {**base, 'status': 'flux_no_existe', 'detail': flux_path}, None

    try:
        df_flux = pd.read_csv(flux_path, sep=r'\s+', comment='#', header=None, usecols=[1, 2])
    except Exception as e:
        return {**base, 'status': 'error_lectura_flux', 'detail': str(e)}, None

    if df_flux.empty:
        return {**base, 'status': 'flux_vacio'}, None
    df_flux.columns = ['tipo', 'tiempo']
    df_flux = df_flux.apply(pd.to_numeric, errors='coerce')
    n_total = len(df_flux)
    df_flux = df_flux.dropna()
    lineas_descartadas = n_total - len(df_flux)
    if df_flux.empty:
        return {**base, 'status': 'flux_vacio', 'detail': f'{lineas_descartadas} invalid lines'}, None
    df_flux['tipo'] = df_flux['tipo'].astype(int)
    if lineas_descartadas:
        base['lineas_descartadas'] = lineas_descartadas

    tipos_en_flux = set(df_flux['tipo'].unique())
    tipos_esperados = set(areas_unitarias.keys())
    if not tipos_en_flux.issubset(tipos_esperados):
        return {**base, 'status': 'tipo_mismatch',
                'detail': f'flux has {sorted(tipos_en_flux)}, expected a subset of {sorted(tipos_esperados)}'}, None

    if T_MIN is not None:
        df_flux = df_flux[df_flux['tiempo'] >= T_MIN]
    if T_MAX is not None:
        df_flux = df_flux[df_flux['tiempo'] <= T_MAX]
    if df_flux.empty:
        return {**base, 'status': 'sin_datos_en_intervalo', 'detail': f'[{T_MIN}, {T_MAX}] s'}, None
    df_flux = df_flux.copy()

    t_start = T_MIN if T_MIN is not None else df_flux['tiempo'].min()
    t_end = df_flux['tiempo'].max()
    bins = np.arange(t_start, t_end + WINDOW_SIZE, WINDOW_SIZE)
    if len(bins) < 2:
        return {**base, 'status': 'pocos_bins', 'detail': f't_start={t_start}, t_end={t_end}'}, None

    df_flux['batch'] = pd.cut(df_flux['tiempo'], bins=bins, labels=bins[:-1])

    batch_results = []
    for label, group in df_flux.groupby('batch', observed=True):
        counts = group['tipo'].value_counts()
        a_chicas_out = counts.get(tipo_chica, 0) * areas_unitarias[tipo_chica]
        a_total_out = sum(counts.get(t, 0) * areas_unitarias[t] for t in areas_unitarias.keys())
        if a_total_out > 0:
            batch_results.append({'tiempo': float(label) + WINDOW_SIZE / 2,
                                  'chi_eff': a_chicas_out / a_total_out})

    df_plot = pd.DataFrame(batch_results)
    if df_plot.empty:
        return {**base, 'status': 'sin_batches_validos'}, None
    return {**base, 'status': 'ok', 'chi_nominal': chi_nominal}, df_plot


def medir_segregacion(df_plot, chi_nominal):
    """Deviation of chi_eff from the nominal value, on the curve smoothed with
    a moving average of SUAVIZADO_BATCHES windows (so that the noise of each
    window does not count). Relative to chi_nominal:
      desvio_medio_rel  mean of |chi_smooth - chi_nom| / chi_nom
      desvio_max_rel    maximum of the same
      deriva_rel        (last 20 % - first 20 % of the time) / chi_nom
    """
    chi = df_plot['chi_eff'].to_numpy()
    suave = pd.Series(chi).rolling(SUAVIZADO_BATCHES, center=True, min_periods=1).mean().to_numpy()
    n = max(1, len(chi) // 5)
    dev = np.abs(suave - chi_nominal) / chi_nominal
    return {'desvio_medio_rel': float(dev.mean()), 'desvio_max_rel': float(dev.max()),
            'deriva_rel': float((chi[-n:].mean() - chi[:n].mean()) / chi_nominal),
            'chi_eff_medio': float(chi.mean()), 'n_ventanas': len(chi)}


def analizar_flujo_temporal(row):
    """Plot of chi_eff(t) for one simulation (saved in its folder). Returns a
    dict with 'status' (ok, or the reason why no figure was made) and the
    segregation measures.
    """
    info, df_plot = calcular_chi_temporal(row)
    if df_plot is None:
        return info
    base = {k: info[k] for k in ('sim_hash', 'tirada', 'path', 'lineas_descartadas') if k in info}
    path, sim_hash, tirada, chi_nominal = info['path'], info['sim_hash'], info['tirada'], info['chi_nominal']
    if chi_nominal > 0:
        base.update(medir_segregacion(df_plot, chi_nominal))

    try:
        color, marker, label_en = estilo_sistema(row.get('system_type', ''))

        plt.figure(figsize=(10, 5))
        plt.axhline(y=chi_nominal, **REF_LINE_KW,
                    label=rf'$\chi = {chi_nominal:.3g}$')
        plt.plot(df_plot['tiempo'], df_plot['chi_eff'], '-', color=color, lw=1.2,
                 label=rf'$\chi_{{out}}$ ({WINDOW_SIZE:g} s windows)', zorder=2)

        last_row = df_plot.iloc[-1]
        t_final, chi_final = last_row['tiempo'], last_row['chi_eff']
        plt.plot([t_final, t_final], [chi_final, chi_nominal], color='gray', linestyle=':',
                 linewidth=1.5, zorder=1)
        if chi_nominal > 0:
            diff_pct = (chi_nominal - chi_final) / chi_nominal * 100
            plt.annotate(f"{diff_pct:.1f}%", xy=(t_final, (chi_final + chi_nominal) / 2),
                         xytext=(6, 0), textcoords='offset points', va='center',
                         color='gray', fontsize=14)

        plt.grid(False)
        plt.xlabel(r"Simulation Time (s)")
        plt.ylabel(r"Outflow Concentration $\chi_{\rm eff}$")
        plt.ylim(0, min(1.0, max(chi_nominal * 2.5, df_plot['chi_eff'].max() * 1.3, 0.05)))
        _xlim_intervalo(plt.gca())
        plt.legend(title=f"{label_en}, run {tirada}", loc='upper right')

        graph_name = f"dinamica_flujo_final_{sim_hash}_T{tirada}.{FORMATO}"
        plt.savefig(os.path.join(path, graph_name), bbox_inches='tight')
        plt.close()
    except Exception as e:
        plt.close('all')
        return {**base, 'status': 'error_al_graficar', 'detail': str(e)}

    return {**base, 'status': 'ok'}


def _xlim_intervalo(ax):
    """If a time interval was requested, the x axis spans exactly T_MIN to T_MAX."""
    if T_MIN is not None or T_MAX is not None:
        lo, hi = ax.get_xlim()
        ax.set_xlim(T_MIN if T_MIN is not None else lo, T_MAX if T_MAX is not None else hi)


def figura_comparativa(df_idx, casos, out, normalizar=False, cortar=False):
    """chi_eff(t) of several simulations in the same figure (lines only), each
    with its nominal chi as a dashed line of the same color.
    casos: list of sim_hash (prefixes accepted), or ['todos'] for all the
    simulations left after --sistema / --chi / --area_ratio.
    """
    if list(casos) == ['todos']:
        casos = df_idx['sim_hash'].astype(str).tolist()
        print(f"--casos todos: {len(casos)} simulations")
    _estilo()
    fig, ax = plt.subplots(figsize=(10, 5))
    y_max = 0.0
    datos = []
    for k, h in enumerate(casos):
        fila = df_idx[df_idx['sim_hash'].astype(str).str.startswith(str(h))]
        if fila.empty:
            print(f"  [!] sim_hash {h} is not in the index")
            continue
        row = fila.iloc[0]
        info, d = calcular_chi_temporal(row)
        if d is None:
            print(f"  [!] {h}: {info['status']} {info.get('detail', '')}")
            continue
        datos.append((k, row, info, d))
    t_corte = None
    if cortar and datos:
        t_corte = min(float(d['tiempo'].max()) for *_, d in datos)
        datos = [(k, row, info, d[d['tiempo'] <= t_corte]) for k, row, info, d in datos]
        print(f"--cortar_al_mas_corto: all curves up to t = {t_corte:.0f} s")
    for k, row, info, d in datos:
        chi_nom = info['chi_nominal']
        y = d['chi_eff'] / chi_nom if normalizar else d['chi_eff']
        color = COLORES_CASOS[k % len(COLORES_CASOS)]
        _, _, nombre = estilo_sistema(row.get('system_type', ''))
        extra = ''
        if 'area_ratio_nominal' in row and pd.notnull(row['area_ratio_nominal']):
            extra = rf', $R_{{\rm a}} = {row["area_ratio_nominal"]:g}$'
        ax.plot(d['tiempo'], y, '-', color=color, lw=1.2, zorder=2,
                label=rf'{nombre}, $\chi = {chi_nom:.3g}${extra}')
        ax.axhline(1.0 if normalizar else chi_nom, color=color, ls='--', lw=1.5, zorder=3)
        y_max = max(y_max, float(y.max()))
        if chi_nom > 0:
            seg = medir_segregacion(d, chi_nom)
            print(f"  {row['sim_hash']}: {nombre} chi={chi_nom:g}  desvio medio {100 * seg['desvio_medio_rel']:.1f} %"
                  f"  max {100 * seg['desvio_max_rel']:.1f} %  deriva {100 * seg['deriva_rel']:+.1f} %")
    if not ax.lines:
        plt.close(fig)
        print("[!] No case has data: the figure is not saved.")
        return
    ax.grid(False)
    ax.set_xlabel(r'Simulation Time (s)')
    ax.set_ylabel(r'$\chi_{\rm eff} / \chi$' if normalizar else r'Outflow Concentration $\chi_{\rm eff}$')
    n_casos = len(ax.get_legend_handles_labels()[1])
    ax.set_ylim(0, y_max * (1.2 + 0.12 * n_casos) if y_max > 0 else 1)
    _xlim_intervalo(ax)
    if t_corte is not None:
        ax.set_xlim(T_MIN if T_MIN is not None else 0, t_corte)
    ax.legend(loc='upper right')
    guardar_figura(out, fig)
    plt.close(fig)
    print(f"Figure: {out if os.path.splitext(out)[1] else out + '.' + FORMATO}")


def _inicializar_worker(viejo, nuevo, ventana, formato, t_min=None, t_max=None):
    global PREFIJO_VIEJO, PREFIJO_NUEVO, WINDOW_SIZE, FORMATO, T_MIN, T_MAX
    PREFIJO_VIEJO, PREFIJO_NUEVO, WINDOW_SIZE, FORMATO = viejo, nuevo, ventana, formato
    T_MIN, T_MAX = t_min, t_max


def _segregacion_fila(row):
    info, d = calcular_chi_temporal(row)
    if d is None or not info.get('chi_nominal'):
        return None
    meta = {k: row[k] for k in ('system_type', 'chi', 'area_ratio_nominal', 'w_salida') if k in row}
    return {**{k: info[k] for k in ('sim_hash', 'tirada', 'path', 'lineas_descartadas') if k in info}, **meta,
            **medir_segregacion(d, info['chi_nominal'])}


def main():
    parser = argparse.ArgumentParser(
        description="Plots the discharge chi as a function of time for each simulation of the index.")
    parser.add_argument('--index', default=INDEX_FILE, help='master_sim_index.csv')
    parser.add_argument('--dir', nargs='+', default=None,
                        help="simulation folder(s) to scan; "
                             "builds the index on the fly with indexer.py instead of using --index")
    parser.add_argument('--incluir', default=None,
                        help="regex on the simulation path: only matching simulations")
    parser.add_argument('--out_csv', default=None,
                        help="output csv of --ranking (default: 5_graficos_finales/chi_temporal_ranking_segregacion.csv)")
    parser.add_argument('--top', type=int, default=20, help="how many to print with --ranking")
    parser.add_argument('--ventana', type=float, default=WINDOW_SIZE,
                        help='time window in seconds (default 10)')
    parser.add_argument('--formato', choices=['pdf', 'svg', 'png'], default=FORMATO,
                        help='output format (default pdf)')
    parser.add_argument('--prefijo_viejo', default=None,
                        help="path prefix in the index to replace (if the data were moved)")
    parser.add_argument('--prefijo_nuevo', default=None,
                        help="new path prefix")
    parser.add_argument('--workers', type=int, default=CPU_CORES)
    parser.add_argument('--casos', nargs='+', default=None,
                        help="sim_hash (or prefixes) for ONE comparison figure with all the curves; "
                             "'todos' = all left after --sistema/--chi/--area_ratio. "
                             "Per-simulation figures are not made")
    parser.add_argument('--normalizar', action='store_true',
                        help="comparison figure with chi_eff / nominal chi (reference at 1)")
    parser.add_argument('--t_min', type=float, default=None,
                        help="start of the time interval (s); all modes")
    parser.add_argument('--t_max', type=float, default=None,
                        help="end of the time interval (s); all modes")
    parser.add_argument('--out', default=os.path.join(rutas.GRAFICOS_FINALES_DIR, 'chi_temporal_comparacion'),
                        help="comparison figure (no extension -> --formato)")
    parser.add_argument('--cortar_al_mas_corto', action='store_true',
                        help="comparison figure: all the curves (and the x axis) end at the last "
                             "point of the shortest simulation")
    parser.add_argument('--ranking', action='store_true',
                        help="no plots: computes the segregation of each simulation and lists the most segregated")
    parser.add_argument('--sistema', nargs='+', default=None,
                        help="filter by system_type (e.g. Disco-Disco Cuad-Disco)")
    parser.add_argument('--chi', type=float, nargs='+', default=None, help="filter by nominal chi")
    parser.add_argument('--area_ratio', type=float, nargs='+', default=None,
                        help="filter by area_ratio_nominal (small/large area ratio, e.g. 0.1)")
    args = parser.parse_args()
    if (args.prefijo_viejo is None) != (args.prefijo_nuevo is None):
        raise SystemExit("--prefijo_viejo and --prefijo_nuevo must be given together.")

    if args.dir:
        from indexer import indexar
        partes = []
        for d in args.dir:
            print(f"Indexing {d} ...")
            parte = indexar(d)
            n = 0 if parte is None else len(parte)
            print(f"  {n} simulations found.")
            if parte is not None:
                partes.append(parte)
        if not partes:
            raise SystemExit(f"No simulations (folders with params.in) found in {args.dir}")
        df_idx = pd.concat(partes, ignore_index=True).drop_duplicates(subset=['path'])
    else:
        if not os.path.exists(args.index):
            print(f"Error: {args.index} not found.")
            return
        df_idx = pd.read_csv(args.index)
    if args.incluir:
        df_idx = df_idx[df_idx['path'].astype(str).str.contains(args.incluir, regex=True)]
        print(f"--incluir {args.incluir!r}: {len(df_idx)} simulations left.")
        if df_idx.empty:
            raise SystemExit("No simulation matches --incluir.")
    _inicializar_worker(args.prefijo_viejo, args.prefijo_nuevo, args.ventana, args.formato, args.t_min, args.t_max)
    if args.sistema:
        df_idx = df_idx[df_idx['system_type'].isin(args.sistema)]
    if args.chi:
        df_idx = df_idx[np.isclose(df_idx['chi'].to_numpy()[:, None], args.chi).any(axis=1)]
    if args.area_ratio:
        if 'area_ratio_nominal' not in df_idx.columns:
            raise SystemExit("The index has no area_ratio_nominal column: run indexer.py again.")
        df_idx = df_idx[np.isclose(df_idx['area_ratio_nominal'].to_numpy(float)[:, None],
                                   args.area_ratio).any(axis=1)]
    if args.sistema or args.chi or args.area_ratio:
        print(f"Filters: {len(df_idx)} simulations left.")
        if df_idx.empty:
            raise SystemExit("No simulation passes the filters (check --sistema / --chi / --area_ratio).")

    if args.casos:
        figura_comparativa(df_idx, args.casos, args.out, normalizar=args.normalizar,
                           cortar=args.cortar_al_mas_corto)
        return

    if args.ranking:
        tasks = [row for _, row in df_idx.iterrows()]
        with Pool(args.workers, initializer=_inicializar_worker,
                  initargs=(args.prefijo_viejo, args.prefijo_nuevo, args.ventana, args.formato,
                            args.t_min, args.t_max)) as p:
            res = p.map(_segregacion_fila, tasks)
        r = pd.DataFrame([x for x in res if x is not None])
        if r.empty:
            print("No simulation could be analyzed (check the paths / --prefijo_viejo).")
            return
        r = r.sort_values('desvio_medio_rel', ascending=False)
        out_csv = args.out_csv or os.path.join(rutas.GRAFICOS_FINALES_DIR, 'chi_temporal_ranking_segregacion.csv')
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        r.to_csv(out_csv, index=False)
        cols = [c for c in ('sim_hash', 'system_type', 'chi', 'area_ratio_nominal', 'tirada',
                            'desvio_medio_rel', 'desvio_max_rel', 'deriva_rel') if c in r.columns]
        print(f"Segregation of chi_eff with respect to the nominal value (moving average of {SUAVIZADO_BATCHES} windows), "
              f"top {args.top} of {len(r)}:")
        with pd.option_context('display.width', 200):
            print(r[cols].head(args.top).to_string(index=False, float_format='%.3f'))
        print(f"\nFull table: {out_csv}")
        print("Median of desvio_medio_rel: " f"{r['desvio_medio_rel'].median():.3f}")
        return

    print(f"Analyzing chi(t) for {len(df_idx)} simulations...")

    tasks = [row for _, row in df_idx.iterrows()]
    with Pool(args.workers, initializer=_inicializar_worker,
              initargs=(args.prefijo_viejo, args.prefijo_nuevo, args.ventana, args.formato,
                            args.t_min, args.t_max)) as p:
        resultados = p.map(analizar_flujo_temporal, tasks)

    conteo = Counter(r['status'] for r in resultados)
    print("\n--- Summary by status ---")
    for status, n in conteo.most_common():
        print(f"  {status}: {n}")

    print("\n--- Examples per failure status (up to 3 each) ---")
    for status in conteo:
        if status == 'ok':
            continue
        ejemplos = [r for r in resultados if r['status'] == status][:3]
        print(f"\n[{status}]")
        for e in ejemplos:
            print(f"  sim_hash={e['sim_hash']} tirada={e['tirada']} detail={e.get('detail', '')}")

    con_descarte = [r for r in resultados if r.get('lineas_descartadas')]
    if con_descarte:
        print(f"\n--- flx.dat with discarded invalid lines ({len(con_descarte)} simulations) ---")
        for r in con_descarte:
            print(f"  sim_hash={r['sim_hash']} tirada={r['tirada']}: {r['lineas_descartadas']} lineas  ({r['path']})")

    print(f"\n[+] Done. {conteo.get('ok', 0)} figures made out of {len(df_idx)} simulations.")


if __name__ == "__main__":
    main()