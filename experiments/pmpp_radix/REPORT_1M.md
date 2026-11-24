# One-million-Gaussian comparison — 2026-11-09

Same RTX 4090, native Dawn/Vulkan, WGSL implementations and measurement protocol as REPORT.md. Each input contains exactly 1,000,000 Gaussian records, constructed by cycling through the original PLY records without jitter or shuffling. These are scaled workloads derived from Luigi (14,526 unique source records) and Black Jacket (41,470), not new million-unique-Gaussian reconstructions. Repeated positions create equal-depth ties, whose stable original-index order is validated.

| Source asset | Count | Masks median, ms | PMPP-style median, ms | Local / masks |
|---|---:|---:|---:|---:|
| luigi | 1,000,000 | 0.250624 | 0.466432 | 1.86x |
| black_jacket | 1,000,000 | 0.255232 | 0.459520 | 1.80x |

Cells are medians of 15 trial medians: three view orientations × five alternating-order trials. Each trial uses 64 timestamp samples of four consecutive complete sorts, normalized to milliseconds/sort, following eight warmup sorts. Includes key generation and final order output; excludes uploads, pipeline compilation, CPU work, readback, and rendering. Both methods still use eight four-bit passes and 32 dispatches. Mask scratch is 4 bytes/key; this PMPP-style implementation stores locally reordered pairs plus rank, 12 bytes/key. This compares these implementations, not an optimal possible PMPP port.

All 12 method/asset/view stable-order checks matched the CPU reference exactly. At this size, 3,907 key workgroups produce 16 histogram scan chunks per digit, exercising multiple scan chunks unlike the original-size assets. The CPU reference sorts GPU-generated keys with original index tie breaking. Production renderer sources remain unchanged. No rendering or new full edge-case-suite validation is claimed.

The GPU remained attached to the desktop. Before this run nvidia-smi reported 5% utilization and 15,556 MiB free; background activity was not eliminated. Measurements were serialized by this benchmark. Raw trial summaries: timings_1m.csv; all 3,840 timestamp samples: timings_1m.csv.samples.csv; validation log: run_1m.log.

```sh
cmake --build build --target pmpp_radix_benchmark -j 8
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/pmpp_radix_benchmark experiments/pmpp_radix/timings_1m.csv 1000000
```

The optional second argument selects the cyclically expanded/truncated record count. Omitting it retains the original asset sizes. Existing original-size results were preserved.

Benchmark source SHA-256: `f493f9943c7c4b4efc794e2c59171cc6ca5e0a1ae1bcce97136f960167f740bb`. Shader/input hashes are unchanged from REPORT.md.
