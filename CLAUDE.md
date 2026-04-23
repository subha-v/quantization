Use uv to do package management

## Remote server access (tambe-server-1)

- SSH target: `subha2@tambe-server-1` (key-based auth configured from this Mac)
- **Scope constraint:** Only operate inside `/data/subha2/` on the remote server. Do not read, write, or modify anything under `/home/`, `/etc/`, `/usr/`, or any other user's directory. The `/home` partition is NFS quota-limited and shared; `/data/subha2` is the user's private workspace.
- Key subdirectories under `/data/subha2/`:
  - `openpi/` — openpi repo + venv (install deps here via `cd openpi && uv pip install ...`)
  - `quantization/` — this repo's clone (pull latest with `git pull`)
  - `experiments/` — `$EXPERIMENT_DIR`; scripts get copied here, results written to `experiments/results/`, plots to `experiments/plots/`
  - `pi05_libero_pytorch/` — converted checkpoint
  - `libero_raw/` — LIBERO static parquet dataset
- GPU: Check nvidia-smi for devices others are using and make sure to not use a device someone else is using
- Destructive ops (killing processes, uninstalling packages, editing shared configs) require explicit user confirmation. Read-only diagnostics (`ps`, `ls`, `nvidia-smi`, reading logs/results) are OK to run directly.

