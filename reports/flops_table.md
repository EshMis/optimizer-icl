## Training compute per step

| width | optimizer | GFLOPs per step | Newton-Schulz share |
|---|---|---|---|
| 64 | adamw | 21.0 | 0.0% |
| 64 | muon | 21.6 | 2.9% |
| 128 | adamw | 80.6 | 0.0% |
| 128 | muon | 85.6 | 5.9% |
| 256 | adamw | 315.8 | 0.0% |
| 256 | muon | 356.1 | 11.3% |
| 512 | adamw | 1250.1 | 0.0% |
| 512 | muon | 1572.2 | 20.5% |

## Compute to reach a threshold (first evaluation where the seed-mean M* reaches it)

| width | AdamW to M*=32 | Muon to M*=32 | AdamW / Muon | AdamW to M*=48 | Muon to M*=48 | AdamW / Muon | AdamW to M*=64 | Muon to M*=64 | AdamW / Muon |
|---|---|---|---|---|---|---|---|---|---|
| 64 | not reached | 1.43e+14 | - | not reached | 5.44e+14 | - | not reached | not reached | - |
| 128 | 1.55e+15 | 2.56e+14 | 6.0x | not reached | 6.61e+14 | - | not reached | 1.39e+15 | - |
| 256 | 3.20e+15 | 7.08e+14 | 4.5x | 6.31e+15 | 2.10e+15 | 3.0x | not reached | 3.42e+15 | - |
| 512 | 8.25e+15 | 2.73e+15 | 3.0x | 1.88e+16 | 7.55e+15 | 2.5x | 3.27e+16 | 1.51e+16 | 2.2x |
