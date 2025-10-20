import tdt
import os
               #vvvvvvvvvv     REPLACE PATH WITH YOUR FILE STRUCTURE   vvvvvvvvv
block_path = r"C:\Users\Melvi\02.1_Coding_projects\BME_Capstone_UH_2025\BME_Capstone_UH_2025_Github\data\raw\Exp_1\ExperimentBL-230918-004353"

if not os.path.exists(block_path):
    raise FileNotFoundError(f"Block path not found:\n{block_path}")

print(f"\n📂 Reading block: {block_path}")
blk = tdt.read_block(block_path)

# --- List what TDT sees ---
print("\nEpoc stores:", list(blk.epocs.keys()))
print("Stream stores:", list(blk.streams.keys()))

# --- Inspect all streams systematically ---
print("\n🔍 Stream summaries:")
for name, stream in blk.streams.items():
    data = stream.data
    print(f"  {name:<6} shape={data.shape}, fs={stream.fs:.2f} Hz, "
          f"duration={data.shape[-1]/stream.fs:.2f}s")

# --- Inspect all epocs ---
print("\n🔍 Epoc summaries:")
for name, ep in blk.epocs.items():
    print(f"  {name:<6} n={len(ep.onset)}, first_onset={ep.onset[0] if len(ep.onset)>0 else 'None'}")

# --- Pick main signals manually ---
lfp = blk.streams.get("Wav1") or blk.streams.get("Wav2")
stim = blk.streams.get("IZn1") or blk.streams.get("sSig") or blk.streams.get("sOut")

if lfp:
    print(f"\n✅ Using {lfp} as LFP stream: shape={lfp.data.shape}, fs={lfp.fs}")
else:
    print("\n⚠️ No LFP-like stream found (try checking Wav1/Wav2 Above ^^^).")

if stim:
    print(f"✅ Using {stim} as stim stream: shape={stim.data.shape}, fs={stim.fs}")
else:
    print("⚠️ No stim stream found (check for sSsig ^^^^).")
