import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.use('Agg')

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import rutas
POSTPRO_DIR = rutas.SPATIAL_MAPS_DIR

def plot_segregation_analysis(npz_path, chi_global=0.1, output_folder=POSTPRO_DIR):
    """Heat map of the spatial distribution of the additive grains."""
    try:
        data = np.load(npz_path)
        s_hash = os.path.basename(npz_path).replace(".npz", "")
        
        d1 = data['d1']
        extent = [-10.25, 10.25, 0, 120]

        fig, ax = plt.subplots(figsize=(6, 10), constrained_layout=True)

        im0 = ax.imshow(d1, origin='lower', extent=extent, 
                        cmap='viridis', interpolation='gaussian', 
                        aspect='auto')

        ax.axvline(-10.25, color='black', linewidth=2, alpha=0.7)
        ax.axvline(10.25, color='black', linewidth=2, alpha=0.7)

        ax.set_title(f"Spatial distribution of the additive\nHash: {s_hash}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Horizontal position (x)", fontsize=11)
        ax.set_ylabel("Height (y)", fontsize=11)

        cbar = plt.colorbar(im0, ax=ax, shrink=0.8)
        cbar.set_label('Mean density (particles/cell)', fontsize=10)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        save_path = os.path.join(output_folder, f"{s_hash}_segregacion.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[+] Density map for {s_hash}")
        return True

    except Exception as e:
        print(f"[!] density_map error for {npz_path}: {e}")
        return False

if __name__ == "__main__":
    pass