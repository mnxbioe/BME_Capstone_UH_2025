# build_vpl_stim_assets.py
import json, csv, math
import numpy as np

# ---------------------- Geometry (from the paper) ----------------------
n_rows, n_cols = 2, 8
row_spacing = 500e-6        # 500 µm between rows
col_spacing = 250e-6        # 250 µm within row
tip_area = 1250e-12         # ≈1250 µm^2 conducting tip area
tip_radius = float(np.sqrt(tip_area/np.pi))

# tip plane depth (just metadata for convenience)
Lz = 2.5e-3
tip_z = 0.5 * Lz

def rc_to_index(r, c): return r*n_cols + c

row_offsets = np.array([-row_spacing/2, +row_spacing/2], dtype=float)
col_offsets = (np.arange(n_cols) - (n_cols - 1)/2.0) * col_spacing

electrodes = []
for r in range(n_rows):
    for c in range(n_cols):
        electrodes.append({
            "index": rc_to_index(r,c), "r": int(r), "c": int(c),
            "x": float(col_offsets[c]), "y": float(row_offsets[r]), "z": float(tip_z)
        })

# Adjacent bipolar pairs (horizontal + vertical)
pairs = []
for r in range(n_rows):
    for c in range(n_cols-1):  # horizontal neighbor
        iL, iR = rc_to_index(r,c), rc_to_index(r,c+1)
        pairs.append({"type":"horizontal","left":iL,"right":iR})
for c in range(n_cols):        # vertical neighbor
    iT, iB = rc_to_index(0,c), rc_to_index(1,c)
    pairs.append({"type":"vertical","top":iT,"bottom":iB})

# ---------------------- Choose 1–3 active bipolar pairs ----------------------
# Use adjacent electrodes (as in the paper). Here: two horizontal pairs (one per row).
# Add or change entries to use 1, 2, or 3 pairs.
active_pairs_rc = [(0,3), (1,3)]   # (row, left_col) -> pair is (left_col, left_col+1)

active_pairs_indices = []
for (r, cL) in active_pairs_rc:
    active_pairs_indices.append((rc_to_index(r,cL), rc_to_index(r,cL+1)))  # (pos, neg) order is arbitrary; sign is set below

# ---------------------- Pulse pattern (Choi et al. style) ----------------------
# Symmetric biphasic, 200 µs/phase. Two bursts: one near onset, one near release.
pulse_uA       = 30.0     # amplitude per phase (typical 7–40 µA in paper)
phase_us       = 200.0    # per phase
interphase_us  = 0.0      # inter-phase delay
burst1 = dict(start_ms=6.0,   duration_ms=20.0, rate_hz=300.0)
burst2 = dict(start_ms=170.0, duration_ms=15.0, rate_hz=300.0)

class PulseEvent:
    __slots__=("t_start_s","t_end_s","electrode_index","pair_id","current_A","current_density_A_per_m2","note")
    def __init__(self,t0,t1,eidx,pid,I,J,note):
        self.t_start_s=t0; self.t_end_s=t1; self.electrode_index=int(eidx)
        self.pair_id=int(pid); self.current_A=I; self.current_density_A_per_m2=J; self.note=note

def biphasic_events(t0_s, I_A, phase_s, interphase_s, elec_pos, elec_neg, pair_id):
    J = I_A / tip_area
    ev=[]
    # phase 1: +I on "pos", -I on "neg"
    ev.append(PulseEvent(t0_s, t0_s+phase_s, elec_pos, pair_id, +I_A, +J, "phase1_pos(+I)"))
    ev.append(PulseEvent(t0_s, t0_s+phase_s, elec_neg, pair_id, -I_A, -J, "phase1_neg(-I)"))
    # phase 2 (flip)
    t1 = t0_s + phase_s + interphase_s
    ev.append(PulseEvent(t1, t1+phase_s, elec_pos, pair_id, -I_A, -J, "phase2_pos(-I)"))
    ev.append(PulseEvent(t1, t1+phase_s, elec_neg, pair_id, +I_A, +J, "phase2_neg(+I)"))
    return ev

def generate_burst_events(start_ms, duration_ms, rate_hz, I_uA, phase_us, interphase_us, pair_list):
    t = start_ms*1e-3; end_t=(start_ms+duration_ms)*1e-3; dt=1.0/rate_hz
    I_A = I_uA*1e-6; phase_s=phase_us*1e-6; interphase_s=interphase_us*1e-6
    events=[]; pid=0; tk=t
    while tk < end_t - 1e-12:
        for (i_pos,i_neg) in pair_list:
            events += biphasic_events(tk, I_A, phase_s, interphase_s, i_pos, i_neg, pid)
            pid += 1
        tk += dt
    return events

events  = generate_burst_events(**burst1, I_uA=pulse_uA, pair_list=active_pairs_indices, phase_us=phase_us, interphase_us=interphase_us)
events += generate_burst_events(**burst2, I_uA=pulse_uA, pair_list=active_pairs_indices, phase_us=phase_us, interphase_us=interphase_us)
events.sort(key=lambda e: (e.t_start_s, e.electrode_index))

# ---------------------- Write outputs ----------------------
geom = {
  "array":{"n_rows":n_rows,"n_cols":n_cols,"row_spacing_m":row_spacing,"col_spacing_m":col_spacing},
  "tip":{"area_m2":tip_area,"radius_m":tip_radius},
  "electrodes":electrodes,
  "tip_plane_z_m":tip_z,
  "notes":"Coordinates centered laterally (x,y). z increases with depth."
}
with open("vpl_array_geometry.json","w") as f: json.dump(geom,f,indent=2)

with open("stim_pairs.json","w") as f:
    json.dump({
        "pairs":pairs,
        "active_pairs_default":[{"left":int(iL),"right":int(iR)} for (iL,iR) in active_pairs_indices],
        "pattern_notes":"Default uses two horizontal adjacent pairs (one per row). Use 1–3 pairs total."
    }, f, indent=2)

with open("pulse_schedule.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["t_start_s","t_end_s","electrode_index","pair_id","current_A","current_density_A_per_m2","note"])
    for ev in events:
        w.writerow([f"{ev.t_start_s:.9f}", f"{ev.t_end_s:.9f}", ev.electrode_index, ev.pair_id,
                    f"{ev.current_A:.9e}", f"{ev.current_density_A_per_m2:.9e}", ev.note])

# Tiny helper module for your FEM code
with open("bc_helpers.py","w") as f:
    f.write(r'''
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
''')

print("Wrote vpl_array_geometry.json, stim_pairs.json, pulse_schedule.csv, bc_helpers.py")