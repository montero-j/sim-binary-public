import os
import shutil
import multiprocessing
import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import rutas
import indexer
import processor
import flow_analysis
import density_map
import temporal_chi
import generar_csv_arcos
import arch_consolidator
import histogram_arch


def main():
    print("=" * 60)
    print("   AVALANCHE AND ARCH ANALYSIS PIPELINE   ")
    print("=" * 60)

    idx_path = rutas.MASTER_INDEX
    if not indexer.main() or not os.path.exists(idx_path):
        print("[-] Error: the simulation index could not be built. Aborting.")
        return False

    processor.main()

    print("\n[+] Extracting arch statistics...")
    df_idx = pd.read_csv(idx_path)

    for _, row in df_idx.iterrows():
        sim_path = row.get('path', '')
        if os.path.exists(sim_path):
            print(f"  -> Detecting arches in {row.get('sim_hash', 'unk')}...", end='\r')
            generar_csv_arcos.procesar_directorio(sim_path)

    print("\n  -> Arch extraction finished.")

    exito_cons = arch_consolidator.consolidate_arcs(idx_path=idx_path)

    print("\n[+] Running flow and avalanche statistics...")
    flow_analysis.main(idx_path=idx_path)


    print("\n[+] PIPELINE FINISHED.")
    return True


if __name__ == "__main__":
    import sys
    multiprocessing.freeze_support()
    sys.exit(0 if main() else 1)
