import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as pltPolygon
import matplotlib as mpl
import os
import glob
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from especies import especie_por_tipo  # noqa: E402

mpl.use('Agg') 
mpl.rcParams['svg.fonttype'] = 'none'

ESCALA_GROSOR_FUERZA = 0.4
NOMBRE_COLORMAP = 'inferno' 
GROSOR_BORDE_FUERZAS = 0.4
COLOR_BORDE_FUERZAS = '#777777'
GROSOR_BORDE_SIMPLE = 0.3

COLOR_RELLENO_GRANO = '#DCE3EA'
COLOR_BORDE_GRANO = '#7F8C99'
COLOR_FUERZA = '#4D4D4D'
COLOR_ARCO_RELLENO = '#FFC864'
COLOR_GRANDE = '#D62728'
COLOR_CHICO = '#1F77B4'
ALPHA_ARCO = 0.9
COLOR_ARCO_LINEA = 'black'
GROSOR_LINEA_ARCO = 0.75

def parse_params(filename):
    params = {'Radio_silo': 10.25, 'Radio_salida': 2.5}
    try:
        with open(filename, 'r') as f:
            for line in f:
                if 'Radio_silo:' in line:
                    params['Radio_silo'] = float(line.split(':')[1].split('#')[0].strip())
                if 'Radio_salida:' in line:
                    params['Radio_salida'] = float(line.split(':')[1].split('#')[0].strip())
    except Exception: pass
    return params

def parse_xy(filename):
    particles = {}
    with open(filename, 'r') as f:
        for line in f:
            p = line.split()
            if not p: continue
            pid = int(p[0])
            if pid < 0: continue
            n_sides = int(p[1])
            if n_sides == 1:
                x, y, r, ptype = float(p[2]), float(p[3]), float(p[4]), int(p[5])
                particles[pid] = {'pos': (x, y), 'r': r, 'type': ptype, 'shape': 'Disco'}
            else:
                coords = np.array(p[2:-1], dtype=float).reshape(-1, 2)
                center = np.mean(coords, axis=0)
                particles[pid] = {'pos': tuple(center), 'coords': coords, 'type': int(p[-1]), 'shape': 'Poligono'}
    return particles

def obtener_radio(p_data):
    if p_data['shape'] == 'Disco': return p_data['r']
    return np.max(np.linalg.norm(p_data['coords'] - p_data['pos'], axis=1))

def es_cimiento_estable(pid, particles, G, path, idx, r_salida):
    p = particles[pid]
    x, y = p['pos']
    r = obtener_radio(p)
    if abs(x) < (r_salida - 0.2): return False
    if (y - r) < 0.2: return True
    vecinos_arco = []
    if idx > 0: vecinos_arco.append(path[idx-1])
    if idx < len(path) - 1: vecinos_arco.append(path[idx+1])
    apoyos_inferiores = 0
    for vecino in G.neighbors(pid):
        if vecino not in vecinos_arco:
            y_vecino = particles[vecino]['pos'][1]
            if y_vecino < (y - 0.05):
                apoyos_inferiores += 1
    return apoyos_inferiores >= 2

def purgar_cimientos(path, particles, G, r_salida):
    if not path: return []
    iL = 0
    while iL < len(path) and es_cimiento_estable(path[iL], particles, G, path, iL, r_salida):
        iL += 1
    iR = len(path) - 1
    while iR >= iL and es_cimiento_estable(path[iR], particles, G, path, iR, r_salida):
        iR -= 1
    if iL > iR: return []
    return path[iL:iR+1]

def detectar_arco_primario(G, particles, params):
    r_salida = params.get('Radio_salida', 2.5)
    nleft, nright = [], []
    for pid, data in particles.items():
        r = obtener_radio(data)
        x, y = data['pos']
        if (y - r) < 0.2:
            if x < 0.0: nleft.append(pid)
            elif x > 0.0: nright.append(pid)
    mejor_arco = []
    min_costo_total = float('inf')
    for l in nleft:
        for r in nright:
            if nx.has_path(G, l, r):
                try:
                    path = nx.shortest_path(G, source=l, target=r, weight='costo_arco')
                    costo_path = sum(G[path[i]][path[i+1]]['costo_arco'] for i in range(len(path)-1))
                    if costo_path < min_costo_total:
                        min_costo_total = costo_path
                        mejor_arco = purgar_cimientos(path, particles, G, r_salida)
                except nx.NetworkXNoPath: continue
    return mejor_arco

def area_particula(p):
    if p['shape'] == 'Disco':
        return np.pi * p['r'] ** 2
    x, y = p['coords'][:, 0], p['coords'][:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def graficar_arco_y_fuerzas(sim_folder, jam_id, estilo='fuerzas', formato='svg',
                            fuerzas='arco', escala_fuerza=ESCALA_GROSOR_FUERZA,
                            grosor_arco=GROSOR_LINEA_ARCO, colores='especie', salida=None,
                            color_fondo=COLOR_BORDE_FUERZAS, grosor_fondo=GROSOR_BORDE_FUERZAS,
                            recorte='roi', margen=4.0):
    """Snapshot of a clog with the detected arch and the contact forces.
    estilo='fuerzas': forces colored with inferno, grains without fill.
    estilo='simple': grains filled with a light color, forces in a single color.
    fuerzas='arco': only the contacts involving an arch grain;
    fuerzas='todas': the force network of the whole clog.
    colores='especie': large arch grains in red, small ones in blue;
    colores='arco': arch in orange/green.
    """
    simple = (estilo == 'simple')
    frames_dir = glob.glob(os.path.join(sim_folder, "frames_*"))
    if not frames_dir: 
        print(f"Error: no 'frames_' folder in {sim_folder}")
        return
    frames_dir = frames_dir[0]

    xy_file = os.path.join(frames_dir, f"atasco_{jam_id}.xy")
    contacts_file = os.path.join(frames_dir, f"atasco_contacts_{jam_id}.dat")
    params_file = os.path.join(sim_folder, "params.in")

    if not os.path.exists(xy_file):
        print(f"Error: file not found: {xy_file}")
        return

    params = parse_params(params_file)
    particles = parse_xy(xy_file)
    r_salida = params.get('Radio_salida', 2.5)
    r_silo = params.get('Radio_silo', 10.25)

    try:
        contacts = pd.read_csv(contacts_file, sep=r'\s+', comment='#', names=['id_a', 'id_b', 'px', 'py', 'f_norm', 'f_tan'])
    except Exception: 
        print("Error: the contact file could not be read.")
        return

    valid_c = contacts[(contacts['id_a'] >= 0) & (contacts['id_b'] >= 0)].copy()
    valid_c['pair_a'] = np.minimum(valid_c['id_a'], valid_c['id_b'])
    valid_c['pair_b'] = np.maximum(valid_c['id_a'], valid_c['id_b'])
    
    grouped_contacts = valid_c.groupby(['pair_a', 'pair_b']).agg({
        'px': 'mean',
        'py': 'mean',
        'f_norm': 'sum'
    }).reset_index()

    dict_contactos = {}
    for _, row in grouped_contacts.iterrows():
        dict_contactos[(int(row['pair_a']), int(row['pair_b']))] = (row['px'], row['py'])

    G = nx.Graph()
    for pid, attr in particles.items(): G.add_node(pid, **attr)
    
    for _, row in grouped_contacts.iterrows():
        u, v = int(row['pair_a']), int(row['pair_b'])
        if u in particles and v in particles:
            y_max = max(particles[u]['pos'][1], particles[v]['pos'][1])
            costo = 10.0 ** y_max 
            G.add_edge(u, v, weight=row['f_norm'], costo_arco=costo)

    arco_path = detectar_arco_primario(G, particles, params)

    y_coords = [p['pos'][1] for p in particles.values()]
    y_max_plot = max(y_coords) + 2
    y_min_plot = -2
    if recorte == 'roi' and arco_path:
        y_arco_max = max(particles[i]['pos'][1] for i in arco_path)
        y_max_plot = y_arco_max + margen
        y_min_plot = -1
    elif recorte == 'roi':
        print("Warning: no arch detected; the whole silo is shown.")
    
    ancho_silo = r_silo * 2
    alto_silo = y_max_plot - y_min_plot
    ratio = alto_silo / ancho_silo
    fig, ax = plt.subplots(figsize=(10, 10 * ratio))

    por_especie = (colores == 'especie')
    if por_especie:
        pids = list(particles)
        especie = especie_por_tipo([particles[i]['type'] for i in pids],
                                   [area_particula(particles[i]) for i in pids])
        color_de = lambda p: COLOR_GRANDE if especie[p['type']] == 'B' else COLOR_CHICO

    fc, ec, lw_borde = ((COLOR_RELLENO_GRANO, COLOR_BORDE_GRANO, GROSOR_BORDE_SIMPLE) if simple
                        else ('none', color_fondo, grosor_fondo))
    for pid, p in particles.items():
        if p['pos'][1] > y_max_plot + 2:
            continue
        if p['shape'] == 'Disco':
            ax.add_patch(Circle(p['pos'], p['r'], fc=fc, ec=ec, lw=lw_borde, zorder=1))
        else:
            ax.add_patch(pltPolygon(p['coords'], fc=fc, ec=ec, lw=lw_borde, zorder=1))

    if arco_path:
        for pid in arco_path:
            p = particles[pid]
            if por_especie:
                kw = dict(fc=color_de(p), alpha=ALPHA_ARCO, ec='none', zorder=2)
            elif simple:
                kw = dict(fc=COLOR_ARCO_RELLENO, ec='none', zorder=2)
            else:
                c = 'orange' if p['type'] == 0 else 'lightgreen'
                kw = dict(fc=c, alpha=0.5, ec='none', zorder=2)
            if p['shape'] == 'Disco':
                ax.add_patch(Circle(p['pos'], p['r'], **kw))
            else:
                ax.add_patch(pltPolygon(p['coords'], **kw))
        
        arch_x, arch_y = [], []
        arch_x.append(particles[arco_path[0]]['pos'][0])
        arch_y.append(particles[arco_path[0]]['pos'][1])
        
        for i in range(len(arco_path) - 1):
            u, v = arco_path[i], arco_path[i+1]
            pair = (min(u, v), max(u, v))
            
            if pair in dict_contactos:
                px, py = dict_contactos[pair]
                arch_x.append(px)
                arch_y.append(py)
                
            arch_x.append(particles[v]['pos'][0])
            arch_y.append(particles[v]['pos'][1])
            
        ax.plot(arch_x, arch_y, color=COLOR_ARCO_LINEA, lw=grosor_arco, zorder=5, alpha=0.6, solid_capstyle='round')

    f_max_scale = grouped_contacts['f_norm'].quantile(0.95) if not grouped_contacts.empty else 1.0
    norm = mpl.colors.Normalize(vmin=0, vmax=f_max_scale)
    cmap = mpl.colormaps[NOMBRE_COLORMAP]

    en_arco = set(arco_path)
    for _, row in grouped_contacts.iterrows():
        id_a, id_b = int(row['pair_a']), int(row['pair_b'])
        if fuerzas == 'arco' and id_a not in en_arco and id_b not in en_arco:
            continue
        if id_a in particles and id_b in particles:
            pos_a = particles[id_a]['pos']
            pos_b = particles[id_b]['pos']
            
            px, py = row['px'], row['py']
            f_val = row['f_norm']
            
            ax.plot([pos_a[0], px, pos_b[0]], [pos_a[1], py, pos_b[1]], 
                    color=COLOR_FUERZA if simple else cmap(norm(f_val)), 
                    lw=f_val * escala_fuerza, 
                    solid_capstyle='round', alpha=0.85, zorder=4)

    ax.plot([-r_silo, -r_silo], [0, y_max_plot], color='black', lw=2, zorder=7)
    ax.plot([r_silo, r_silo], [0, y_max_plot], color='black', lw=2, zorder=7)
    ax.plot([-r_silo, -r_salida], [0, 0], color='black', lw=4, zorder=7)
    ax.plot([r_salida, r_silo], [0, 0], color='black', lw=4, zorder=7)

    ax.set_xlim(-r_silo - 1, r_silo + 1)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.set_aspect('equal')
    ax.axis('off')

    sufijo = '_simple' if simple else ''
    output_path = os.path.join(salida or sim_folder, f'arco_y_fuerzas_contacto_{jam_id}{sufijo}.{formato}')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, format=formato, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)
    print(f"[+] Image saved to:\n   {output_path}")

if __name__ == "__main__":
    import rutas
    import argparse
    ap = argparse.ArgumentParser(description="Clog snapshot with the force network and the detected arch.")
    ap.add_argument('--sim', default=os.path.join(rutas.SIM_DATA_DIR, "BARRIDO_WIDTHS_DISK_DISK", "Chi_0p0_W_2p0"),
                    help="simulation folder (the one with params.in and frames_*)")
    ap.add_argument('--jam_id', nargs='+', default=["00095"], help="one or more clog ids (00095 or 95)")
    ap.add_argument('--estilo', choices=['fuerzas', 'simple'], default='fuerzas',
                    help="fuerzas: inferno colormap; simple: filled grains and single-color forces")
    ap.add_argument('--formato', default='svg', help="svg, pdf or png")
    ap.add_argument('--fuerzas', choices=['arco', 'todas'], default='arco',
                    help="arco: only contacts of the arch grains; todas: the whole force network")
    ap.add_argument('--escala_fuerza', type=float, default=ESCALA_GROSOR_FUERZA,
                    help=f"line width per unit force (default {ESCALA_GROSOR_FUERZA})")
    ap.add_argument('--grosor_arco', type=float, default=GROSOR_LINEA_ARCO,
                    help=f"width of the arch line (default {GROSOR_LINEA_ARCO})")
    ap.add_argument('--colores', choices=['especie', 'arco'], default='especie',
                    help="especie: large arch grains red, small ones blue; arco: arch in orange/green")
    ap.add_argument('--color_fondo', default=COLOR_BORDE_FUERZAS,
                    help=f"edge color of the grains outside the arch (fuerzas style; default {COLOR_BORDE_FUERZAS})")
    ap.add_argument('--grosor_fondo', type=float, default=GROSOR_BORDE_FUERZAS,
                    help=f"edge width of the grains outside the arch (default {GROSOR_BORDE_FUERZAS})")
    ap.add_argument('--recorte', choices=['roi', 'silo'], default='roi',
                    help="roi: window from the base to the top of the arch plus a margin (default); silo: whole silo")
    ap.add_argument('--margen', type=float, default=4.0,
                    help="height (in d) shown above the highest arch center (default 4)")
    ap.add_argument('--salida', default=None,
                    help="output folder (default: the simulation folder)")
    args = ap.parse_args()
    for j in args.jam_id:
        graficar_arco_y_fuerzas(args.sim, f"{int(j):05d}", estilo=args.estilo, formato=args.formato,
                                fuerzas=args.fuerzas, escala_fuerza=args.escala_fuerza,
                                grosor_arco=args.grosor_arco, colores=args.colores, salida=args.salida,
                                color_fondo=args.color_fondo, grosor_fondo=args.grosor_fondo,
                                recorte=args.recorte, margen=args.margen)