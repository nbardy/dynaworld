# CUDA next

After validating the Metal version, port these exact stages to CUDA in the same order:

1. `snugbox_count`
2. `emit_pairs`
3. `tile_histogram`
4. `tile_forward`
5. `tile_backward`

## Keep identical array contracts

- `means2d`: `[G, 2]`
- `conics`: `[G, 3]`
- `colors`: `[G, 3]`
- `opacities`: `[G]`
- `depths`: `[G]`
- `tile_ids`: `[N]`
- `gauss_ids`: `[N]`
- `tile_ranges`: `[T + 1]`

## CUDA-only upgrades to add after parity

1. true row/column AccuTile sweep from Speedy-Splat
2. stronger shared-memory staging and warp reductions
3. optional per-Gaussian backward kernel
4. fused optimizer step if it proves worth the complexity

## Do not change first

Keep these choices the same until the two backends numerically match:

- exact alpha compositing
- same alpha threshold
- same bbox logic
- same no-gradient-through-sort rule
- same tile size
- same chunk size
