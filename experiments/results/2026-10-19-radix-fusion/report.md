# Radix fusion

The first histogram generates depth keys. The last scatter writes Gaussian indices directly to the render order buffer. This reduces six pipelines and 34 dispatches to four pipelines and 32 dispatches. The 4-bit radix, both scan stages, and stable ordering are unchanged.

Before/after measurements use validate_radix_sort on the RTX 4090, 32 GPU timestamp samples after four warmups per size. These are sort compute-pass timings, including key generation and final order output, not frame times. Each version was measured once; small differences may include clock or background-load variation. CPU measurements are recorded but are not part of the fusion comparison.

All 25 sort cases and six independent rendering checks passed after the change. Historical benchmark results are preserved separately.

Final pre-commit recheck on 2026-10-19: the native build and all six rendering cases passed. The sort rerun passed every case through 1,000,000 items, then could not allocate the next case with only 163 MiB VRAM reported free. The earlier complete 25-case pass is retained in validation.txt; the interrupted recheck is not a new complete-suite pass.
