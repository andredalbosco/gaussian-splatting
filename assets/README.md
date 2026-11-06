# Scene assets

## Luigi

- File: `luigi.ply`
- Source: https://huggingface.co/datasets/dylanebert/3dgs/blob/main/luigi/luigi.ply
- 14,526 Gaussians in binary little-endian PLY.
- Fields: position, normal, SH DC color, opacity, scale, and quaternion rotation.
- No asset license was specified in the dataset metadata inspected at download time.

Luigi contains DC color only, with no higher-order SH coefficients. It is retained as a DC-only test scene.

## Black Jacket

- File: `black_jacket.ply`
- Source: https://geofront.eu/hosted/geosplat/data/blackjacket_geofront.ply
- Attribution: [Geofront Gaussian Splatting Library](https://geofront.eu/splatting_guide/models.html), geofront e.U. (2026).
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- 41,470 Gaussians; 10,286,090 bytes; binary little-endian float PLY.
- Full degree-3 SH: `f_dc_0..2` and `f_rest_0..44`.
- Downloaded unchanged. This is the default native and browser scene.
- The app uses negative Y as camera-up for this asset, preserving its geometry and world-space SH basis.
