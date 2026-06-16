import h5py, os
data_dir = "examples/cfd/darcy_nuno/data/darcy_master_normalized/train"
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
print(f"Found {len(files)} files")
for f in files[:5]:
    with h5py.File(os.path.join(data_dir, f)) as h:
        print(f"  {f}: keys={list(h.keys())}, shapes={[(k, h[k].shape) for k in h.keys()]}")