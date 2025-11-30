
import json, csv, numpy as np

def electrode_positions(geom_json):
    g = json.load(open(geom_json,"r"))
    pts = [(int(e["index"]), float(e["x"]), float(e["y"]), float(e["z"])) for e in g["electrodes"]]
    A  = float(g["tip"]["area_m2"]); r = float(g["tip"]["radius_m"])
    return pts, A, r

def spherical_patch_mask(x, y, z, center, radius):
    cx, cy, cz = center
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return ((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) <= radius**2

def neumann_flux_series(pulse_csv, electrode_index, A_tip=None):
    """Yield (t_start, t_end, q, note) with q in A/m^2 for a given electrode index."""
    with open(pulse_csv,"r") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["electrode_index"]) != int(electrode_index): continue
            if row.get("current_density_A_per_m2"):
                q = float(row["current_density_A_per_m2"])
            else:
                if A_tip is None: raise ValueError("A_tip required to derive current density")
                q = float(row["current_A"])/float(A_tip)
            yield (float(row["t_start_s"]), float(row["t_end_s"]), q, row.get("note",""))
