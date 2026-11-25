import os
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import rutas
from estilo_figuras import (SYSTEM_STYLE, ENGLISH_LABELS, SYSTEM_NAME_MAP,
                            set_plot_fonts, aplicar_estilo, ERRORBAR_KW,
                            REF_LINE_KW, SEQ_STYLE, anotar_chi)


@dataclass
class AnalysisConfig:
    """Paths and styles shared by all the figures."""
    data_path: str = os.path.join(rutas.AVALANCHE_DIR, "estadisticas_flujo_agregadas.csv")
    arcos_path: str = os.path.join(rutas.AVALANCHE_DIR, "resultados_completos_arcos.csv")
    raw_flow_path: str = os.path.join(rutas.AVALANCHE_DIR, "resultados_completos_esqueleto.csv")
    out_dir: str = rutas.GRAFICOS_FINALES_DIR

    system_style: Dict[str, Tuple[str, str]] = field(default_factory=lambda: dict(SYSTEM_STYLE))
    english_labels: Dict[str, str] = field(default_factory=lambda: dict(ENGLISH_LABELS))

CFG = AnalysisConfig()

AR_COL = "area_ratio_nominal"
RR_COL = "r_ratio_nominal"

SUFIJO_NUEVO = "_nuevo"


def save_plot(filename: str):
    os.makedirs(CFG.out_dir, exist_ok=True)
    plt.savefig(f"{CFG.out_dir}/{filename}.pdf", bbox_inches='tight')
    plt.close()


def require_columns(df: pd.DataFrame, cols: List[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in {context}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def prepare_flow_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Validates estadisticas_flujo_agregadas.csv (written by flow_analysis.py).

    Expected columns:
        sistema, chi, r_ratio_nominal, area_ratio_nominal, D_d_sq, grano_base,
        S_mean, S_sem, log_S, log_S_err, ...
    """
    require_columns(
        df,
        ['sistema', 'chi', AR_COL, RR_COL, 'D_d_sq', 'grano_base', 'S_mean', 'S_sem', 'log_S', 'log_S_err'],
        'estadisticas_flujo_agregadas.csv',
    )
    return df.copy()


def prepare_arcos(df_arcos: pd.DataFrame) -> pd.DataFrame:
    """Validates and normalizes resultados_completos_arcos.csv (written by
    arch_consolidator.py). The file carries both the real and the nominal
    size ratios from master_sim_index.csv; only the nominal ones are used for
    grouping and plotting.
    """
    require_columns(
        df_arcos,
        ['sim_hash', 'system_type', 'chi', AR_COL, RR_COL,
         'Total_Particulas_Arco', 'Particulas_Tipo_0', 'Particulas_Tipo_1',
         'Tortuosidad', 'Relacion_Aspecto', 'Altura', 'Span', 'Longitud'],
        'resultados_completos_arcos.csv',
    )

    out = df_arcos.copy()
    altura = out['Altura'].astype(float)
    out['Relacion_Aspecto'] = np.where(altura > 0, 0.5 * out['Span'].astype(float) / altura.where(altura > 0, 1.0),
                                       np.nan)
    out['sistema'] = out['system_type'].map(SYSTEM_NAME_MAP)
    out = out.dropna(subset=['sistema'])
    return out


def get_pure_ref(df: pd.DataFrame, system_name: str, orifice: float, col: str = 'log_S'):
    """Reference value of the pure system of the corresponding large-grain shape."""
    matrix_type = system_name.split('(')[0]
    ref = df[(df['chi'] == 0.0) &
             (df['sistema'].str.startswith(matrix_type)) &
             (df['D_d_sq'] == orifice)]
    if ref.empty:
        return None, None

    if col == 'log_S':
        err_col = 'log_S_err'
    elif col == 'S_mean':
        err_col = 'S_sem'
    else:
        err_col = col + '_err'

    if err_col not in ref.columns:
        return ref[col].mean(), None

    return ref[col].mean(), ref[err_col].mean()


def filter_standard_size(df: pd.DataFrame, target_ar: float, atol: float, system_col: str = 'sistema') -> pd.DataFrame:
    """Selects the mixtures used to compare systems at a fixed size ratio:
    disk-disk with R_a = 0.1, the other systems with R_a = target_ar +- atol.
    Uses area_ratio_nominal (the real area ratios differ slightly, e.g.
    0.0998 and 0.0999).
    """
    require_columns(df, [system_col, AR_COL], 'filter_standard_size')

    mask_disco = (
        (df[system_col] == 'Disco(G)-Disco(C)') &
        np.isclose(df[AR_COL].astype(float), 0.1, atol=0.03)
    )
    mask_otros = (
        (df[system_col] != 'Disco(G)-Disco(C)') &
        np.isclose(df[AR_COL].astype(float), target_ar, atol=atol)
    )
    return df[mask_disco | mask_otros].copy()


def select_by_nominal_ar(df: pd.DataFrame, target: float, atol: float = 0.05) -> pd.DataFrame:
    require_columns(df, [AR_COL], 'select_by_nominal_ar')
    return df[np.isclose(df[AR_COL].astype(float), target, atol=atol)].copy()


def plot_puros_vs_orificio(df: pd.DataFrame):
    print("  -> Plotting 1: Pure Systems vs Orifice...")
    df_puros = df[(df['chi'] == 0.0) & (df['sistema'].str.startswith('Disco'))].sort_values('D_d_sq')
    if df_puros.empty:
        return

    plt.figure()
    plt.errorbar(
        df_puros['D_d_sq'], df_puros['log_S'], yerr=df_puros['log_S_err'],
        fmt='o-', color='black', label='Pure Disks', **ERRORBAR_KW
    )
    plt.xlabel(r"Orifice Ratio $(D/d)^2$")
    plt.ylabel(r"$\ln(\langle s \rangle + 1)$")
    plt.legend(title="System")
    save_plot("1-pure-vs-orifice")


def plot_disco_vs_area_ratio(df: pd.DataFrame):
    print("  -> Plotting 2: Disco(G)-Disco(C) vs Area Ratio...")

    df_sys = df[df['sistema'] == 'Disco(G)-Disco(C)'].copy()
    if df_sys.empty:
        return

    main_dd = df_sys['D_d_sq'].value_counts().idxmax()
    df_sys = df_sys[df_sys['D_d_sq'] == main_dd]

    plt.figure()

    val_ref, _ = get_pure_ref(df, 'Disco(G)-Disco(C)', main_dd, 'log_S')
    if val_ref is not None:
        plt.axhline(y=val_ref, **REF_LINE_KW, label=r'$\chi = 0$ (Pure)')

    target_chis = [0.01, 0.1, 0.14, 0.2]
    colors, markers = SEQ_STYLE(len(target_chis))

    for chi, color, marker in zip(target_chis, colors, markers):
        grp = df_sys[np.isclose(df_sys['chi'], chi, atol=0.005)].sort_values(AR_COL)
        if grp.empty:
            continue

        plot_data = (
            grp.groupby(AR_COL, dropna=False)
               .agg(log_S=('log_S', 'mean'), log_S_err=('log_S_err', 'mean'))
               .reset_index()
               .sort_values(AR_COL)
        )

        plt.errorbar(
            plot_data[AR_COL], plot_data['log_S'], yerr=plot_data['log_S_err'],
            fmt=f'{marker}-', color=color, label=rf'$\chi = {chi}$',
            **ERRORBAR_KW,
        )

    plt.xlim(left=0.0)
    plt.xlabel(r"Area Ratio $R_{\rm a}$")
    plt.ylabel(r"$\ln(\langle s \rangle + 1)$")

    plt.legend(loc='best')

    save_plot("2-discog-discoc-vs-area-ratio" + SUFIJO_NUEVO)


def plot_disco_vs_area_ratio_norm(df: pd.DataFrame):
    """Disk-disk <s>/<s>_0 (pure disk system, same orifice) vs R_a in
    logarithmic scale, with the reference at 1. Error: S_sem / <s>_0 (the
    error of the pure system is not propagated).
    """
    print("  -> Plotting 2B: Disco(G)-Disco(C) vs Area Ratio, <s>/<s>_0 (log)...")

    df_sys = df[df['sistema'] == 'Disco(G)-Disco(C)'].copy()
    if df_sys.empty:
        return

    main_dd = df_sys['D_d_sq'].value_counts().idxmax()
    df_sys = df_sys[df_sys['D_d_sq'] == main_dd]

    s0, _ = get_pure_ref(df, 'Disco(G)-Disco(C)', main_dd, 'S_mean')
    if s0 is None:
        print("     [!] no pure disk system: cannot normalize")
        return
    print(f"     <s>_0 = {s0:.2f}; minimum <s> in the csv = {df['S_mean'].min():.2f}, "
          f"in disk-disk = {df_sys['S_mean'].min():.2f}")

    plt.figure()
    plt.axhline(y=1.0, **REF_LINE_KW, label=r'$\chi = 0$ (Pure)')

    target_chis = [0.01, 0.1, 0.14, 0.2]
    colors, markers = SEQ_STYLE(len(target_chis))

    for chi, color, marker in zip(target_chis, colors, markers):
        grp = df_sys[np.isclose(df_sys['chi'], chi, atol=0.005)].sort_values(AR_COL)
        if grp.empty:
            continue

        plot_data = (
            grp.groupby(AR_COL, dropna=False)
               .agg(S_mean=('S_mean', 'mean'), S_sem=('S_sem', 'mean'))
               .reset_index()
               .sort_values(AR_COL)
        )

        plt.errorbar(
            plot_data[AR_COL], plot_data['S_mean'] / s0, yerr=plot_data['S_sem'] / s0,
            fmt=f'{marker}-', color=color, label=rf'$\chi = {chi}$',
            **ERRORBAR_KW,
        )

    plt.yscale('log')
    ax = plt.gca()
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 20, 30, 50) if lo <= t <= hi]
    ax.yaxis.set_major_locator(plt.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    plt.xlim(left=0.0)
    plt.xlabel(r"Area Ratio $R_{\rm a}$")
    plt.ylabel(r"$\langle s \rangle / \langle s \rangle_0$")

    plt.legend(loc='best')

    save_plot("2-discog-discoc-vs-area-ratio_B")


def _puntos_de(curvas, n_seg=30):
    """Points of the curves (R_a, x, y, err): the data, the ends of the error
    bars and intermediate points on the lines joining them.
    """
    puntos = []
    for _, x, y, err in curvas:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        e = np.nan_to_num(np.asarray(err, dtype=float), nan=0.0)
        for xi, yi, ei in zip(x, y, e):
            puntos += [(xi, yi), (xi, yi - ei), (xi, yi + ei)]
        t = np.linspace(0.0, 1.0, n_seg)
        for i in range(len(x) - 1):
            puntos += list(zip(x[i] + t * (x[i + 1] - x[i]), y[i] + t * (y[i + 1] - y[i])))
    return np.asarray(puntos, dtype=float)


def _puntos_tapados(ax, puntos, titulo, margen_px=10.0):
    """Number of points (data coordinates) covered by the legend in the upper
    right corner, with a margin in pixels for markers and error bars.
    """
    leg = ax.legend(loc='upper right', title=titulo)
    ax.figure.canvas.draw()
    caja = leg.get_window_extent()
    disp = ax.transData.transform(puntos)
    return int(np.sum((disp[:, 0] >= caja.x0 - margen_px) & (disp[:, 0] <= caja.x1 + margen_px) &
                      (disp[:, 1] >= caja.y0 - margen_px) & (disp[:, 1] <= caja.y1 + margen_px)))


def margen_para_leyenda(ax, curvas, titulo, paso=0.05, max_extra=1.5):
    """Fraction of the y range to add on top so that the legend fits in the
    upper right corner without covering any curve (0 if it already fits). The
    axes are not changed.
    """
    puntos = _puntos_de(curvas)
    y0, y1 = ax.get_ylim()
    extra = 0.0
    while extra <= max_extra:
        ax.set_ylim(y0, y1 + extra * (y1 - y0))
        if _puntos_tapados(ax, puntos, titulo) == 0:
            break
        extra += paso
    ax.set_ylim(y0, y1)
    return min(extra, max_extra)


def plot_sweeps_by_system(
    df: pd.DataFrame,
    target_systems: List[Tuple[str, str]],
    zoom: bool = False,
    xlim: Tuple[float, float] = (0.05, 0.15),
):
    if zoom:
        print("  -> Plotting 4 zoom: Clogging Curves (chi 0.05 - 0.15)...")
    else:
        print("  -> Plotting 4: Clogging Curves (Sweeps)...")

    excluded_chis_no_zoom = [0.07, 0.075, 0.085, 0.09, 0.095, 0.105, 0.11, 0.15]

    datos_por_sistema = {}
    x_all, y_lo_all, y_hi_all = [], [], []

    for system_name, _ in target_systems:
        df_mix = df[(df['sistema'] == system_name) & (df['chi'] > 0.0)].copy()

        if df_mix.empty:
            continue

        if not zoom:
            mask_excluded = np.zeros(len(df_mix), dtype=bool)
            for chi_excl in excluded_chis_no_zoom:
                mask_excluded |= np.isclose(df_mix['chi'].astype(float), chi_excl, atol=1e-6)

            df_mix = df_mix[~mask_excluded].copy()

        if df_mix.empty:
            continue

        main_dd = df_mix['D_d_sq'].value_counts().idxmax()
        p0, e0 = get_pure_ref(df, system_name, main_dd, 'log_S')

        if p0 is None:
            continue

        targets = [0.1, 0.2, 0.4]
        curvas = []

        for target in targets:
            grp_raw = select_by_nominal_ar(df_mix, target, atol=0.06)

            if grp_raw.empty:
                continue

            available = np.sort(grp_raw[AR_COL].dropna().unique())
            chosen = available[np.argmin(np.abs(available - target))]
            grp_raw = grp_raw[np.isclose(grp_raw[AR_COL], chosen, atol=1e-12)]

            grp = (
                grp_raw.groupby('chi')
                       .agg(log_S=('log_S', 'mean'), log_S_err=('log_S_err', 'mean'))
                       .reset_index()
                       .sort_values('chi')
            )

            if grp.empty:
                continue

            x = [0.0] + grp['chi'].tolist()
            y = [p0] + grp['log_S'].tolist()
            err = [e0] + grp['log_S_err'].tolist()
            curvas.append((chosen, x, y, err))

            print(f"\nSystem: {system_name}\n  R_a {target} -> {AR_COL} real {chosen:.4f}")

            x_all.extend(x)
            y_arr = np.asarray(y, dtype=float)
            err_arr = np.nan_to_num(np.asarray(err, dtype=float), nan=0.0)
            y_lo_all.extend((y_arr - err_arr).tolist())
            y_hi_all.extend((y_arr + err_arr).tolist())

        if curvas:
            datos_por_sistema[system_name] = curvas

    if not datos_por_sistema:
        return

    if zoom:
        x_range = xlim
    else:
        x_min, x_max = min(x_all), max(x_all)
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 0.02
        x_range = (x_min - pad_x, x_max + pad_x)

    y_min, y_max = min(y_lo_all), max(y_hi_all)
    pad_y = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    y_range = (y_min - pad_y, y_max + pad_y)

    figuras = []
    for system_name, _ in target_systems:
        if system_name not in datos_por_sistema:
            continue
        system_id = system_name.lower().replace('(', '').replace(')', '').replace('_', '-').replace(' ', '-')

        fig = plt.figure()
        ax = plt.gca()
        curvas = datos_por_sistema[system_name]
        colors, markers = SEQ_STYLE(len(curvas))
        for (chosen, x, y, err), color, marker in zip(curvas, colors, markers):
            plt.errorbar(x, y, yerr=err, fmt=f'{marker}-', color=color,
                         label=rf"$R_{{\rm a}} = {chosen:.3g}$", **ERRORBAR_KW)

        plt.xlabel(r"Concentration $\chi$")
        plt.ylabel(r"$\ln(\langle s \rangle + 1)$")
        plt.xlim(*x_range)
        plt.ylim(*y_range)
        titulo = CFG.english_labels.get(system_name, system_name)
        nombre = (f"4-{system_id}-clogging-curves-zoom-chi-0p05-0p15" if zoom
                  else f"4-{system_id}-clogging-curves")
        figuras.append((fig, ax, curvas, titulo, nombre))

    extra = max(margen_para_leyenda(ax, curvas, titulo) for _, ax, curvas, titulo, _ in figuras)
    for fig, ax, curvas, titulo, nombre in figuras:
        ax.set_ylim(y_range[0], y_range[1] + extra * (y_range[1] - y_range[0]))
        ax.legend(loc='upper right', title=titulo)
        plt.figure(fig.number)
        save_plot(nombre)


RA_FIG6, RA_FIG6_TOL = 0.1, 0.02
RA_ADITIVAS, RA_ADITIVAS_TOL = 0.15, 0.07


def plot_geometric_effects(df: pd.DataFrame, target_ar: float = RA_FIG6, total: bool = False, masa: bool = False,
                           atol: float = RA_FIG6_TOL):
    """5a/5b: mean avalanche size normalized by that of the pure system.
    5b (large polygons or disks + small disks) uses R_a = target_ar +- atol
    (0.1 +- 0.02); 5a (large disks + small polygons) uses R_a ~ 0.15 +- 0.07.
    total=False: <s> = large grains per avalanche (S_mean).
    total=True:  <s_T> = all grains, large + small (ST_mean); 5b only,
                 file 5b-matrix-normalized-total.
    masa=True:   the same in mass (MB_mean / MT_mean from flow_analysis.py); 5b
                 only, files 5b-matrix-normalized-mass and
                 5b-matrix-normalized-total-mass. For the large grains,
                 M/M0 = s/s0 (the mass of a large grain cancels out).
    """
    if masa:
        mean_col, sem_col = ('MT_mean', 'MT_sem') if total else ('MB_mean', 'MB_sem')
    else:
        mean_col, sem_col = ('ST_mean', 'ST_sem') if total else ('S_mean', 'S_sem')
    if mean_col not in df.columns:
        print(f"  [!] {mean_col} is not in estadisticas_flujo_agregadas.csv: run flow_analysis.py again.")
        return
    print(f"  -> Plotting 5{'b' if total or masa else 'a/b'}{' total' if total else ''}{' mass' if masa else ''}: "
          f"Normalized Effects (matrix R_a = {target_ar} +- {atol}; additive R_a ~ {RA_ADITIVAS})...")

    df_mix = df[df['chi'] > 0].copy()
    if df_mix.empty:
        return

    main_dd = df_mix['D_d_sq'].value_counts().idxmax()
    df_puros = (
        df[df['chi'] == 0][['grano_base', 'D_d_sq', mean_col, sem_col]]
        .rename(columns={mean_col: 'S0', sem_col: 'S0_err'})
    )

    df_norm = pd.merge(df_mix[df_mix['D_d_sq'] == main_dd], df_puros, on=['grano_base', 'D_d_sq'])
    if df_norm.empty:
        return

    df_norm['log_S_norm'] = np.log((df_norm[mean_col] / df_norm['S0']) + 1)
    df_norm['log_S_norm_err'] = df_norm[sem_col] / (df_norm[mean_col] + df_norm['S0'])
    v = 'M' if masa else 's'
    ylabel = (rf"$\ln(\langle {v}_{{\rm T}} \rangle / \langle {v}_{{\rm T,0}} \rangle + 1)$" if total
              else rf"$\ln(\langle {v} \rangle / \langle {v}_0 \rangle + 1)$")

    def draw(sistemas: List[str], filename: str, ra: float, tol: float):
        plt.figure()
        for sis in sistemas:
            grp = filter_standard_size(df_norm[df_norm['sistema'] == sis], ra, tol)
            if grp.empty:
                continue
            grp_sis = (
                grp.groupby('chi')
                   .agg(log_S_norm=('log_S_norm', 'mean'), log_S_norm_err=('log_S_norm_err', 'mean'))
                   .reset_index()
                   .sort_values('chi')
            )
            if grp_sis.empty:
                continue

            color, marker = CFG.system_style.get(sis, ('black', 'o'))
            label_en = CFG.english_labels.get(sis, sis)
            plt.errorbar(
                [0.0] + grp_sis['chi'].tolist(),
                [np.log(2.0)] + grp_sis['log_S_norm'].tolist(),
                yerr=[0.0] + grp_sis['log_S_norm_err'].tolist(),
                fmt=f'{marker}-', color=color, label=label_en, **ERRORBAR_KW,
            )

        plt.xlabel(r"Concentration $\chi$")
        plt.ylabel(ylabel)
        if not total and not masa:
            plt.ylim(0.4, 2.5)
        plt.legend(loc='upper left')
        save_plot(filename)

    matriz = ['Disco(G)-Disco(C)', 'Cuad(G)-Disco(C)', 'Pent(G)-Disco(C)', 'Tri(G)-Disco(C)', 'Oct(G)-Disco(C)']
    if total or masa:
        draw(matriz, "5b-matrix-normalized" + ("-total" if total else "") + ("-mass" if masa else ""),
             target_ar, atol)
        return
    draw(['Disco(G)-Disco(C)', 'Disco(G)-Cuad(C)', 'Disco(G)-Pent(C)', 'Disco(G)-Tri(C)', 'Disco(G)-Oct(C)'],
         "5a-additive-normalized", RA_ADITIVAS, RA_ADITIVAS_TOL)
    draw(matriz, "5b-matrix-normalized", target_ar, atol)


def plot_geometric_effects_raw(df: pd.DataFrame, target_ar: float = RA_FIG6, atol: float = RA_FIG6_TOL):
    """5c: ln(<s>+1) vs chi, one curve per shape, R_a = target_ar +- atol."""
    print(f"  -> Plotting 5c: Raw Log-Size Matrix Effect (R_a = {target_ar} +- {atol})...")
    main_dd = df['D_d_sq'].value_counts().idxmax()

    plt.figure()
    for sis in ['Disco(G)-Disco(C)', 'Cuad(G)-Disco(C)', 'Pent(G)-Disco(C)', 'Tri(G)-Disco(C)', 'Oct(G)-Disco(C)']:
        grp = filter_standard_size(df[df['D_d_sq'] == main_dd], target_ar, atol)
        grp_sis = (
            grp[grp['sistema'] == sis]
            .groupby('chi')
            .agg(log_S=('log_S', 'mean'), log_S_err=('log_S_err', 'mean'))
            .reset_index()
            .sort_values('chi')
        )
        if grp_sis.empty:
            continue

        p0, e0 = get_pure_ref(df, sis, main_dd, 'log_S')
        if p0 is None:
            continue

        color, marker = CFG.system_style.get(sis, ('black', 'o'))
        label_en = CFG.english_labels.get(sis, sis)
        plt.errorbar(
            [0.0] + grp_sis['chi'].tolist(),
            [p0] + grp_sis['log_S'].tolist(),
            yerr=[e0] + grp_sis['log_S_err'].tolist(),
            fmt=f'{marker}-', color=color, label=label_en, **ERRORBAR_KW,
        )

    plt.xlabel(r"Concentration $\chi$")
    plt.ylabel(r"$\ln(\langle s \rangle + 1)$")
    plt.ylim(2.2, 6.5)
    plt.legend()
    save_plot("5c-matrix-raw-log")


def plot_geometric_effects_linear(df: pd.DataFrame, target_ar: float = 0.15):
    print(f"  -> Plotting 11: Linear Scale (R_a ~ {target_ar})...")
    main_dd = df['D_d_sq'].value_counts().idxmax()

    def draw_lin(sistemas: List[str], filename: str):
        plt.figure()
        for sis in sistemas:
            grp = filter_standard_size(df[df['D_d_sq'] == main_dd], target_ar, 0.07)
            grp_sis = (
                grp[grp['sistema'] == sis]
                .groupby('chi')
                .agg(S_mean=('S_mean', 'mean'), S_sem=('S_sem', 'mean'))
                .reset_index()
                .sort_values('chi')
            )
            if grp_sis.empty:
                continue

            p0, e0 = get_pure_ref(df, sis, main_dd, 'S_mean')
            if p0 is None:
                continue

            color, marker = CFG.system_style.get(sis, ('black', 'o'))
            label_en = CFG.english_labels.get(sis, sis)
            plt.errorbar(
                [0.0] + grp_sis['chi'].tolist(),
                [p0] + grp_sis['S_mean'].tolist(),
                yerr=[e0] + grp_sis['S_sem'].tolist(),
                fmt=f'{marker}-', color=color, label=label_en, **ERRORBAR_KW,
            )

        plt.xlabel(r"Concentration $\chi$")
        plt.ylabel(r"Mean Avalanche Size $\langle s \rangle$")
        plt.legend(loc='upper left')
        save_plot(filename)

    draw_lin(['Disco(G)-Disco(C)', 'Disco(G)-Cuad(C)', 'Disco(G)-Pent(C)', 'Disco(G)-Tri(C)', 'Disco(G)-Oct(C)'], "11a-additive-linear")
    draw_lin(['Disco(G)-Disco(C)', 'Cuad(G)-Disco(C)', 'Pent(G)-Disco(C)', 'Tri(G)-Disco(C)', 'Oct(G)-Disco(C)'], "11b-matrix-linear")


def plot_arch_composition_B(df_arcos: pd.DataFrame, matrix_systems: List[str], ar_comp: float = 0.1):
    """6-arch-composition_B: NUMBER fraction of small grains in the arch, all
    curves with the same R_a, with the reference of a homogeneous mixture in
    number. chi is an AREA fraction:
        n_S = (chi/R_a) / (chi/R_a + 1 - chi) = chi / (chi + R_a (1 - chi)).
    """
    df = df_arcos[(df_arcos['chi'] > 0) & (df_arcos['chi'] <= 0.4)
                  & np.isclose(df_arcos[AR_COL].astype(float), ar_comp, atol=0.03)].copy()
    if df.empty:
        print(f"  [!] 6-arch-composition_B: no arches with R_a ~ {ar_comp}")
        return
    df['frac_arco'] = df['Particulas_Tipo_1'] / df['Total_Particulas_Arco']
    plt.figure()
    for sys_name in matrix_systems:
        group = df[df['sistema'] == sys_name]
        if group.empty:
            continue
        grp = group.groupby('chi')['frac_arco'].agg(['mean', 'std', 'count']).reset_index()
        grp['sem'] = grp['std'] / np.sqrt(grp['count'])
        color, marker = CFG.system_style.get(sys_name, ('black', 'o'))
        plt.errorbar(
            grp['chi'], grp['mean'], yerr=grp['sem'],
            fmt=f'{marker}-', color=color,
            label=CFG.english_labels.get(sys_name, sys_name), **ERRORBAR_KW,
        )

    chi_ref = np.linspace(0, 0.45, 200)
    plt.plot(chi_ref, chi_ref / (chi_ref + ar_comp * (1 - chi_ref)), **REF_LINE_KW,
             label=rf'Homogeneous ($R_{{\rm a}} = {ar_comp:g}$)')
    plt.xlim(0, 0.45)
    plt.xlabel(r"Global Concentration $\chi$")
    plt.ylabel(r"Arch Small-Grain Fraction $n_{\rm S}$")
    plt.legend(loc='lower right')
    save_plot("6-arch-composition_B")


def plot_arch_analysis(df_stats: pd.DataFrame):
    """Arch composition, tortuosity and avalanche sizes at chi = 0.2."""
    if not os.path.exists(CFG.arcos_path):
        return

    print("\n[+] Starting Arch Analysis (Target chi = 0.2)...")
    df_arcos = prepare_arcos(pd.read_csv(CFG.arcos_path))

    matrix_systems = ['Disco(G)-Disco(C)', 'Tri(G)-Disco(C)', 'Cuad(G)-Disco(C)', 'Pent(G)-Disco(C)', 'Oct(G)-Disco(C)']
    df_filtered = df_arcos[df_arcos['sistema'].isin(matrix_systems)].copy()

    df_mix = filter_standard_size(df_filtered[df_filtered['chi'] > 0], 0.15, 0.1, system_col='sistema')
    if df_mix.empty:
        print("  [!] No arch data for the standard size filter.")
        return

    plot_arch_composition(df_arcos, matrix_systems)

    plot_arch_composition_B(df_arcos[df_arcos['sistema'].isin(matrix_systems)], matrix_systems)

    target_chi = 0.2
    df_target = df_mix[np.isclose(df_mix['chi'], target_chi, atol=0.05)].copy()
    if df_target.empty:
        print(f"  [!] Warning: no arch data for chi ~ {target_chi}")
        return

    color_map = {sys: CFG.system_style[sys][0] for sys in matrix_systems}
    present = [s for s in matrix_systems if s in df_target['sistema'].unique()]
    new_ticks = [CFG.english_labels.get(s, s) for s in present]

    plt.figure()
    ax = sns.violinplot(
        data=df_target, x='sistema', y='Tortuosidad', hue='sistema',
        palette=color_map, inner='quartile', order=present, legend=False,
    )
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(new_ticks, rotation=15)
    anotar_chi(ax, target_chi)
    plt.xlabel(None)
    plt.ylabel("Tortuosity")
    save_plot(f"9-arch-tortuosity-chi-{target_chi}")

    if os.path.exists(CFG.raw_flow_path):
        df_raw = pd.read_csv(CFG.raw_flow_path)
        require_columns(df_raw, ['sim_hash', 'tamano_big'], 'resultados_completos_esqueleto.csv')
        df_plot = pd.merge(
            df_raw,
            df_target[['sim_hash', 'sistema', 'chi']].drop_duplicates(),
            on='sim_hash',
            how='inner',
        )
        if not df_plot.empty:
            plt.figure()
            ax_box = sns.boxplot(
                data=df_plot, x='sistema', y='tamano_big', hue='sistema',
                palette=color_map, order=present, legend=False,
            )
            ax_box.set_xticks(range(len(present)))
            ax_box.set_xticklabels(new_ticks, rotation=15)
            anotar_chi(ax_box, target_chi)
            plt.xlabel(None)
            plt.yscale('log')
            plt.ylabel(r"Avalanche Size $s$")
            save_plot(f"10-avalanche-boxplot-chi-{target_chi}")


AR_COMPOSICION = 0.1
AR_COMPOSICION_TOL = 0.02


def n_s_homogeneo(chi, r_a):
    """Number fraction of small grains in a homogeneous mixture with mass
    fraction chi, with m_S = R_a m (same density): n_S = (chi/R_a) / (chi/R_a + 1 - chi).
    """
    chi = np.asarray(chi, dtype=float)
    return (chi / r_a) / (chi / r_a + 1.0 - chi)


def plot_arch_composition(df_arcos: pd.DataFrame, sistemas: List[str]):
    """Arch composition in two versions:
      6-arch-composition-number: x_S^arch = N_S/(N_S+N_L) in the arch, with the
          homogeneous-mixture curve n_S(chi; R_a).
      6-arch-composition-mass:   chi_arch = R_a N_S/(R_a N_S + N_L), mass
          fraction of small grains in the arch, with the homogeneous line
          chi_arch = chi.
    Both with R_a = AR_COMPOSICION for all systems, chi in (0, 0.4].
    """
    df = df_arcos[df_arcos['sistema'].isin(sistemas) & (df_arcos['chi'] > 0) & (df_arcos['chi'] <= 0.4)].copy()
    ra = df[AR_COL].astype(float)
    print("  -> Arch composition: nominal R_a available per system (chi > 0):")
    for sis in sistemas:
        vals = sorted(float(v) for v in ra[df["sistema"] == sis].round(3).unique())
        if vals:
            print(f"     {sis}: {vals}")
    df = df[np.isclose(ra, AR_COMPOSICION, atol=AR_COMPOSICION_TOL)]
    df = df[df['Total_Particulas_Arco'] > 0]
    if df.empty:
        print(f"  [!] No arches with R_a ~ {AR_COMPOSICION}.")
        return
    r = df[AR_COL].astype(float)
    n_s, n_l = df['Particulas_Tipo_1'].astype(float), df['Particulas_Tipo_0'].astype(float)
    df['frac_num'] = n_s / (n_s + n_l)
    df['frac_masa'] = r * n_s / (r * n_s + n_l)
    df['frac_masa_rel'] = df['frac_masa'] / df['chi'].astype(float)

    chi_ref = np.linspace(0.0, 0.45, 200)
    versiones = [
        ('frac_num', r"$x_{\rm S}^{\rm arch}$",
         n_s_homogeneo(chi_ref, AR_COMPOSICION), rf"Homogeneous, $n_{{\rm S}}(\chi)$", "6-arch-composition-number"),
        ('frac_masa', r"$\chi_{\rm arch}$",
         chi_ref, r"Homogeneous, $\chi_{\rm arch} = \chi$", "6-arch-composition-mass"),
        ('frac_masa_rel', r"$\chi_{\rm arch}/\chi$",
         np.ones_like(chi_ref), r"Homogeneous, $\chi_{\rm arch} = \chi$", "6-arch-composition-mass-ratio"),
    ]
    for col, ylabel, y_ref, label_ref, nombre in versiones:
        plt.figure()
        plt.plot(chi_ref, y_ref, **REF_LINE_KW, label=label_ref)
        for sis in sistemas:
            g = df[df['sistema'] == sis]
            if g.empty:
                continue
            grp = g.groupby('chi')[col].agg(['mean', 'std', 'count']).reset_index()
            grp['sem'] = grp['std'] / np.sqrt(grp['count'])
            color, marker = CFG.system_style.get(sis, ('black', 'o'))
            plt.errorbar(grp['chi'], grp['mean'], yerr=grp['sem'], fmt=f'{marker}-', color=color,
                         label=CFG.english_labels.get(sis, sis), **ERRORBAR_KW)
            if col == 'frac_num':
                fila = grp.iloc[(grp['chi'] - 0.2).abs().argmin()]
                print(f"     {sis}: chi = {fila['chi']:.3g} -> x_S^arch = {fila['mean']:.3f} "
                      f"(n_S = {n_s_homogeneo(fila['chi'], AR_COMPOSICION):.3f}), "
                      f"chi_arch = {g[np.isclose(g['chi'], fila['chi'])]['frac_masa'].mean():.4f}")
        plt.xlim(0, 0.45)
        if col == 'frac_num':
            plt.ylim(0, 1)
        if col == 'frac_masa_rel':
            plt.yscale('log')
            y_min = df.groupby(['sistema', 'chi'])[col].mean().min()
            plt.ylim(0.6 * y_min, 1.6)
        plt.xlabel(r"Concentration $\chi$")
        plt.ylabel(ylabel)
        if col == 'frac_masa_rel':
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.82), ncol=2, fontsize='small')
        else:
            plt.legend(loc='upper left')
        save_plot(nombre + SUFIJO_NUEVO)


def plot_arch_geometry_deep_dive(df_stats: pd.DataFrame):
    """Arch geometry figures (12, 13, 14) at chi = 0.2."""
    if not os.path.exists(CFG.arcos_path):
        return

    print("[+] Arch geometry (chi = 0.2)...")
    df_arcos = prepare_arcos(pd.read_csv(CFG.arcos_path))

    matrix_systems = ['Disco(G)-Disco(C)', 'Tri(G)-Disco(C)', 'Cuad(G)-Disco(C)', 'Pent(G)-Disco(C)', 'Oct(G)-Disco(C)']
    target_chi = 0.2
    df_plot = df_arcos[
        (df_arcos['sistema'].isin(matrix_systems)) &
        (np.isclose(df_arcos['chi'], target_chi, atol=0.05))
    ].copy()
    if df_plot.empty:
        return

    color_map = {sys: CFG.system_style[sys][0] for sys in matrix_systems}
    present = [s for s in matrix_systems if s in df_plot['sistema'].unique()]
    new_ticks = [CFG.english_labels.get(s, s) for s in present]

    plt.figure()
    for sys_name in present:
        data = df_plot[df_plot['sistema'] == sys_name]
        sns.regplot(
            data=data, x='Span', y='Longitud', label=CFG.english_labels[sys_name],
            color=color_map[sys_name], scatter_kws={'alpha': 0.05, 's': 2},
            line_kws={'lw': 2.5},
        )
    plt.xlabel("Orifice Span ($L$)")
    plt.ylabel("Arch Length ($s$)")
    plt.legend(loc='upper left')
    save_plot(f"12-arch-length-vs-span-chi-{target_chi}")

    plt.figure()
    ax = sns.violinplot(
        data=df_plot, x='sistema', y='Relacion_Aspecto', hue='sistema',
        palette=color_map, inner='quartile', order=present, legend=False,
    )
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(new_ticks, rotation=15)
    plt.xlabel(None)
    plt.ylabel("Arch Aspect Ratio $L/2H$")
    anotar_chi(ax, target_chi)
    save_plot(f"13-arch-aspect-ratio-chi-{target_chi}")

    plt.figure()
    ax = sns.boxplot(
        data=df_plot, x='sistema', y='Altura', hue='sistema',
        palette=color_map, order=present, legend=False,
    )
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(new_ticks, rotation=15)
    plt.xlabel(None)
    plt.ylabel("Arch Height ($H$)")
    anotar_chi(ax, target_chi)
    save_plot(f"14-arch-height-boxplot-chi-{target_chi}")


def aspecto_medio(g, n_boot=500, seed=0):
    """Aspect ratio of the mean arch, <L>/(2<H>), and its bootstrap standard
    error over arches (resampling with replacement). Uses Span (L) and
    Altura (H) of each arch.
    """
    L = g['Span'].to_numpy(float)
    H = g['Altura'].to_numpy(float)
    ok = np.isfinite(L) & np.isfinite(H)
    L, H = L[ok], H[ok]
    n = len(L)
    if n == 0 or H.mean() <= 0:
        return {'mean': np.nan, 'sem': np.nan, 'count': n}
    valor = L.mean() / (2.0 * H.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = L[idx].mean(axis=1) / (2.0 * H[idx].mean(axis=1))
    return {'mean': valor, 'sem': float(np.std(boot, ddof=1)), 'count': n}


W_REGEX = re.compile(r'_W_([0-9]+(?:p[0-9]+)?)')


def _w_de_path(path) -> float:
    """Orifice from the folder name (..._W_2p0 -> 2.0); NaN if absent."""
    m = W_REGEX.search(str(path))
    return float(m.group(1).replace('p', '.')) if m else np.nan


def puros_como_mezcla(df_arcos: pd.DataFrame, df_mezclas: pd.DataFrame, sistemas: List[str]) -> pd.DataFrame:
    """Arches of the pure systems (chi = 0) relabelled as the mixture with the
    same large grain ('Tri(G)' -> 'Tri(G)-Disco(C)'), to plot them as the
    chi = 0 point of that curve. If the csv has a 'path' column, only pure
    systems with the same orifice (_W_ in the folder name) as the mixtures of
    that system are used.
    """
    partes = []
    for sis in sistemas:
        base = sis.split('-')[0]
        p = df_arcos[(df_arcos['sistema'] == base) & np.isclose(df_arcos['chi'].astype(float), 0.0)]
        if p.empty:
            print(f"  [!] No arches of the pure system {base} for the chi = 0 point of {sis}.")
            continue
        if 'path' in df_arcos.columns:
            w_mezcla = df_mezclas.loc[df_mezclas['sistema'] == sis, 'path'].map(_w_de_path).dropna().unique()
            if len(w_mezcla):
                p = p[p['path'].map(_w_de_path).isin(w_mezcla)]
        print(f"  -> {sis}: {len(p)} arches of the pure system {base} as the chi = 0 point")
        if not p.empty:
            partes.append(p.assign(sistema=sis))
    return pd.concat(partes, ignore_index=True) if partes else df_arcos.iloc[0:0]


def plot_arch_evolution_vs_chi(df_stats: pd.DataFrame):
    """Figure 15: arch aspect ratio and tortuosity vs chi."""
    if not os.path.exists(CFG.arcos_path):
        return

    print("[+] Arch shape vs chi...")
    df_arcos = prepare_arcos(pd.read_csv(CFG.arcos_path))

    matrix_systems = ['Disco(G)-Disco(C)', 'Tri(G)-Disco(C)', 'Cuad(G)-Disco(C)', 'Pent(G)-Disco(C)', 'Oct(G)-Disco(C)']
    df_filtered = filter_standard_size(
        df_arcos[df_arcos['sistema'].isin(matrix_systems)],
        0.15,
        0.1,
        system_col='sistema',
    )
    
    df_filtered = pd.concat([df_filtered, puros_como_mezcla(df_arcos, df_filtered, matrix_systems)],
                            ignore_index=True)

    df_filtered = df_filtered[(df_filtered['chi'] >= 0) & (df_filtered['chi'] <= 0.4)]

    metrics = {
        'Relacion_Aspecto': r'$L_{\rm arch}/2h_{\rm arch}$',
        'Tortuosidad': 'Arch Tortuosity',
    }

    for metric, ylabel in metrics.items():
        plt.figure()
        for sys_name in matrix_systems:
            group = df_filtered[df_filtered['sistema'] == sys_name]
            if group.empty:
                continue

            if metric == 'Relacion_Aspecto':
                summary = (group.groupby('chi')
                           .apply(lambda g: pd.Series(aspecto_medio(g)), include_groups=False)
                           .reset_index())
            else:
                summary = group.groupby('chi')[metric].agg(['mean', 'std', 'count']).reset_index()
                summary['sem'] = summary['std'] / np.sqrt(summary['count'])
            summary = summary[summary['count'] >= 20]
            if summary.empty:
                continue

            color, marker = CFG.system_style.get(sys_name, ('black', 'o'))
            plt.errorbar(
                summary['chi'], summary['mean'], yerr=summary['sem'],
                fmt=f'{marker}-', color=color,
                label=CFG.english_labels.get(sys_name, sys_name), **ERRORBAR_KW,
            )

        plt.xlim(-0.01, 0.45)
        
        plt.xlabel(r"Concentration $\chi$")
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        save_plot(f"15-evolution-{metric.lower()}{SUFIJO_NUEVO}")


def solo_composicion():
    """Only the arch composition figures:
        python3 graficos_finales.py --solo_composicion
    """
    aplicar_estilo()
    sistemas = ['Disco(G)-Disco(C)', 'Tri(G)-Disco(C)', 'Cuad(G)-Disco(C)', 'Pent(G)-Disco(C)', 'Oct(G)-Disco(C)']
    plot_arch_composition(prepare_arcos(pd.read_csv(CFG.arcos_path)), sistemas)
    print(f"[+] Arch composition figures in: {CFG.out_dir}")


def main():
    print("[+] Starting integrated PDF pipeline...")

    aplicar_estilo(verbose=True)

    os.makedirs(CFG.out_dir, exist_ok=True)

    df = pd.read_csv(CFG.data_path)
    df = prepare_flow_stats(df)

    df = df[df['sistema'] != 'Cuad(G)-Cuad(C)']

    sistemas_meta = [
        ('Tri(G)-Disco(C)', 'Triangle'),
        ('Cuad(G)-Disco(C)', 'Square'),
        ('Pent(G)-Disco(C)', 'Pentagon'),
        ('Oct(G)-Disco(C)', 'Octagon'),
        ('Disco(G)-Disco(C)', 'Disk'),
    ]

    plot_disco_vs_area_ratio(df)
    plot_disco_vs_area_ratio_norm(df)
    plot_sweeps_by_system(df, sistemas_meta)
    plot_sweeps_by_system(df, sistemas_meta, zoom=True, xlim=(0.05, 0.15))
    plot_geometric_effects(df)
    plot_geometric_effects(df, total=True)
    plot_geometric_effects(df, masa=True)
    plot_geometric_effects(df, total=True, masa=True)
    plot_geometric_effects_raw(df)
    plot_geometric_effects_linear(df)

    plot_arch_analysis(df)
    plot_arch_geometry_deep_dive(df)
    plot_arch_evolution_vs_chi(df)

    print("\n[+] Pipeline finished.")
    print(f"[+] All plots generated successfully in: {CFG.out_dir}")


if __name__ == "__main__":
    import sys as _sys_main
    if '--solo_composicion' in _sys_main.argv:
        solo_composicion()
        raise SystemExit
    main()