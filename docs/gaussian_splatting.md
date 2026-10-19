# Gaussian splatting

This viewer renders a scene as overlapping, transparent 3D Gaussians using quads.
The tiled renderer is a separate experiment under `experiments/`. Each Gaussian becomes
an elliptical footprint on the image. The vertex shader computes its projected shape;
the fragment shader evaluates its opacity; blending combines it with the other splats.

The current implementation loads Luigi, uses fixed per-Gaussian RGB colors, and sorts
Gaussian centers on the GPU. It does not evaluate view-dependent spherical harmonics.

## 1. Data representation

A Gaussian stores four vectors in [`Gaussian`](../src/gaussian.hpp):

| Field | Meaning |
|---|---|
| Position | World-space center, $\mu$ |
| Scale | Three positive standard deviations, $s_x,s_y,s_z$ |
| Rotation | Unit quaternion, stored as $(x,y,z,w)$ |
| Color | Fixed RGB and peak opacity |

[`readPlyFile`](../src/gaussian.cpp) decodes logarithmic scales with `exp`, opacity logits
with the sigmoid function, and normalizes the quaternion. It converts the PLY's existing
DC color coefficients to RGB once. Higher-order spherical harmonics are not used.

Let $R$ be the rotation matrix corresponding to the quaternion and

$$
S=\operatorname{diag}(s_x,s_y,s_z).
$$

The world-space covariance is

$$
\Sigma=RS^2R^T=(RS)(RS)^T.
$$

A Gaussian weight at world position $x$ is

$$
G(x)=\exp\left[-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right].
$$

The surface $(x-\mu)^T\Sigma^{-1}(x-\mu)=1$ is the ellipsoid whose semi-axes are the
three stored scales. It is one contour of the Gaussian, not a solid boundary.

## 2. Projecting the center

Write the world-to-camera transformation as

$$
x_c=Wx+t,
$$

where $W$ is the camera rotation. The center becomes $\mu_c=W\mu+t$.
Our camera looks along positive camera-space $Z$.

Perspective projection, expressed in pixels, is

$$
\pi(X,Y,Z)=
\begin{pmatrix}
f_xX/Z+c_x\\
f_yY/Z+c_y
\end{pmatrix}.
$$

The projected center is $\mu_{2D}=\pi(\mu_c)$. The implementation obtains the same
center by multiplying by the view-projection matrix; rasterization performs the divide by $w$.

## 3. Projecting the differential vectors

Projecting the center does not tell us the footprint's size and orientation. We also need
to know how a displacement around the center changes the image position.

The perspective Jacobian, evaluated at $\mu_c=(X,Y,Z)$, is

$$
J=
\begin{pmatrix}
f_x/Z & 0 & -f_xX/Z^2\\
0 & f_y/Z & -f_yY/Z^2
\end{pmatrix}.
$$

For a small displacement $\delta x_c$,

$$
\pi(\mu_c+\delta x_c)\approx\pi(\mu_c)+J\delta x_c.
$$

The complete transformation of the three local differential vectors is

$$
\boxed{B=JWRS}.
$$

Read this from right to left: scale the local axes, rotate them into world space, rotate
into camera space, then apply the projection differential. Translation affects the center
but does not act on the differential vectors.

$B$ has two rows and three columns. Its columns are the three projected, scaled axes.
All three contribute: variation in depth can move an image point because of perspective.

The quad shader in [`src/shaders.wgsl`](../src/shaders.wgsl) uses the same
factorization in clip-space footprint units. For a Gaussian center with clip $w=Z$,
let $D=\operatorname{diag}(2w/\text{width},2w/\text{height})$. Then
$J_{\mathrm{clip}}=DJ_{\mathrm{pixel}}$:

$$
J_{\mathrm{clip}}=
\begin{pmatrix}
P_{00} & 0 & -P_{00}X/Z\\
0 & P_{11} & -P_{11}Y/Z
\end{pmatrix}.
$$

`camera.projection_scale` stores $(P_{00},P_{11})$, the projection matrix diagonal.
`quat_to_matrix` constructs $R$, and scaling its columns constructs $RS$.
The shader evaluates `B = J * W * M` and `B * transpose(B)` in these clip units.
This describes offsets on the quad's constant-$w$ plane; the depth derivative remains
necessary. The tiled compute renderer retains its separate pixel-space representation.

## 4. Constructing the 2D covariance

For a linear transformation, covariance transforms as $A\Sigma A^T$. Here this gives

$$
\boxed{\Sigma_{2D}=J\Sigma_cJ^T=BB^T},
\qquad \Sigma_c=W\Sigma W^T.
$$

With column-vector conventions, $BB^T$ is $2\times2$. The opposite product $B^TB$
would be $3\times3$ and is not the image covariance.

If the projected axes are $b_1,b_2,b_3$, then

$$
\Sigma_{2D}=b_1b_1^T+b_2b_2^T+b_3b_3^T
=\begin{pmatrix}a&b\\b&c\end{pmatrix}.
$$

The shader constructs the covariance directly as a matrix product:

```wgsl
let pixel_to_clip = 2.0 * clip_pos.w / camera.viewport;
let blur = 0.3 * pixel_to_clip * pixel_to_clip;
let cov = B * transpose(B) + mat2x2f(blur.x, 0, 0, blur.y);
```

The added filter is still 0.3 pixel² of variance along each image axis, transformed
into clip units before addition. It keeps tiny footprints from collapsing. The renderer
does not apply amplitude compensation, so filtering changes the integrated contribution
of small splats.

## 5. Constructing the bounding quad

For displacement $d$ from the projected center, expressed in the same units as the covariance, define

$$
q=d^T\Sigma_{2D}^{-1}d.
$$

A contour $q=k^2$ is an ellipse. The Gaussian extends infinitely, but this viewer draws
only $q\leq9$: a 3-sigma footprint.

The tight axis-aligned rectangle around this contour has half-extents

$$
h_x=3\sqrt a,\qquad h_y=3\sqrt c.
$$

Four vertices at $\mu_{2D}+(\pm h_x,\pm h_y)$ form a triangle strip. An eigendecomposition
is unnecessary for this rectangle. Eigenvectors would instead give the ellipse's principal
directions, with semi-axis lengths $3\sqrt{\lambda_1}$ and $3\sqrt{\lambda_2}$.

The shader adds the clip-space offsets directly to `clip_pos.xy`, preserving the center's
`clip_pos.zw`. Every corner has the same $w$, so interpolation of the offsets is linear
across the image. The inverse covariance and fragment offsets both use these clip units.

This quad covers candidate pixels. The fragment shader removes the corners outside the
ellipse; no ray–ellipsoid intersection is performed.

## 6. Evaluating Gaussian opacity

The inverse covariance is

$$
\Sigma_{2D}^{-1}
=\frac{1}{ac-b^2}\begin{pmatrix}c&-b\\-b&a\end{pmatrix}.
$$

The vertex shader stores its three independent entries as `conic`. Each fragment evaluates

$$
q=A d_x^2+2B d_xd_y+C d_y^2,
$$

where $(A,B,C)$ are those inverse-covariance entries. The footprint's opacity is

$$
\alpha(d)=\min\left(0.99,\;o\exp(-q/2)\right),
$$

with $o$ the Gaussian's stored opacity. The shader discards $q>9$ and contributions below
$1/255$. It outputs premultiplied color $(\alpha r,\alpha g,\alpha b,\alpha)$.

There is no probability-density normalization factor here: stored opacity controls the
peak contribution, rather than requiring each splat's integral to equal one.

## 7. Sorting and blending

Transparent contributions depend on their order. `GaussianSplattingApp::recordRadixSort` records a compute
pass before the render pass. It generates a center-depth key for each Gaussian using
$f\cdot\mu$, where $f$ is the camera's forward direction, then sorts indices from far to
near. Subtracting the camera position would add the same constant to every depth and
would not change their order.

Signed float depths are converted to unsigned keys by flipping the sign bit for
nonnegative values and complementing negative values, then complementing the result for
descending depth order. Signed zero is canonicalized. Eight stable four-bit passes use
workgroup membership bitmaps, hierarchical histogram prefixes, and scatter into alternating
key/index buffers. The first histogram also generates keys, and the final scatter writes
indices directly into `order`, for 32 dispatches across four compute pipelines.

The Gaussian storage buffer remains unchanged. Compute writes the sorted indices directly,
and the vertex shader fetches `gaussians[order[instance_index]]`. No CPU sort, sorted-index
upload, or GPU readback is needed. `GaussianSplattingApp::recordCommands` compares the view direction with the last sorted
direction and records a sort when it changes; rebuilding buffers resets the cached direction.
FOV, resize, and translation along the orbit direction preserve the mathematical depth order.

Premultiplied-alpha blending uses

$$
C_{out}=\alpha_s C_s+(1-\alpha_s)C_{destination}.
$$

Far splats are already in the destination when a nearer splat is blended. The pipeline
uses source factor `One` and destination factor `OneMinusSrcAlpha`, because the fragment
shader has already multiplied source RGB by alpha.

Splats do not write surface depth, and this viewer does not use a depth attachment.
The opaque background is cleared first; ImGui is drawn after the scene.

## 8. Approximations and clipping

Perspective division is nonlinear. A perspective-projected 3D Gaussian is generally not
exactly a 2D Gaussian. Using $J$ replaces projection with a local affine approximation
around the center. Large footprints, substantial depth extent, and close views can differ
from the exact projected ellipsoid.

Center-depth sorting also approximates visibility between extended, overlapping Gaussians.
It does not establish a different order for every pixel. This is a conventional splat
compositing approximation, not the same operation as exact volume integration.

The shader rejects centers behind the camera or outside the near/far depth range. It does
not clip the underlying 3D Gaussian volume against those planes. The rasterizer clips the
projected quad at the image edges.

## 9. Implementation and validation

| File | Responsibility |
|---|---|
| [`src/gaussian.hpp`](../src/gaussian.hpp) | GPU-compatible Gaussian data |
| [`src/gaussian.cpp`](../src/gaussian.cpp) | Gaussian PLY loading |
| [`src/app.cpp`](../src/app.cpp) | Base application setup, frame loop, render pass, presentation |
| [`src/glfw_utils.cpp`](../src/glfw_utils.cpp) | Window creation and framebuffer size |
| [`src/imgui_utils.cpp`](../src/imgui_utils.cpp) | ImGui initialization, DPI scaling, shutdown |
| [`src/wgpu_utils.cpp`](../src/wgpu_utils.cpp) | Device, surface, resizing, and shader loading |
| [`src/utils.cpp`](../src/utils.cpp) | General file loading |
| [`src/camera.cpp`](../src/camera.cpp) | View and perspective matrices |
| [`src/main.cpp`](../src/main.cpp) | Gaussian application, radix sorting, draw commands, controls, and program entry point |
| [`src/shaders.wgsl`](../src/shaders.wgsl) | GPU radix sort, projected covariance, quad, Gaussian evaluation |
| [`experiments/validate_splats.cpp`](../experiments/validate_splats.cpp) | Native numerical shader checks |

`App` owns the window, device, frame loop, and presentation. Its `frame()` calls
`buildUi()`, `update()`, and `recordCommands(encoder, target)` on the derived app. The derived app
records its render passes, then `App` draws ImGui in a separate pass that loads the
existing target contents. `GaussianSplattingApp::init()` initializes
the base app and Gaussian pipeline without loading any Gaussians.
`main()` loads the initial Gaussians with `loadGaussians(path)`, then calls the argument-free
`run()` to start the frame loop. A later `loadGaussians(path)`
replaces the Gaussian data, refreshes the scene buffers and bind group, reframes the camera,
and invalidates the cached sort direction so the next frame sorts the new scene.

The validation program compares GPU output with a CPU reference that estimates the
projection differential using finite differences. It checks single-splat falloff, a known
two-layer blend, rotated anisotropic splats, an off-axis center, a non-square viewport,
and behind/near/far rejection. The six cases consume the GPU-generated order and passed with at most 1/255 channel error
on the local RTX 4090.

```sh
cmake --build build --target gaussian_splatting validate_splats -j 16
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/validate_splats
```

The [experiment report](../experiments/BENCHMARKS.md) compares exact ellipsoid proxies,
projected hard ellipses, and Gaussian splats. Its opacity controls were unsorted; their
timings are not end-to-end timings of this sorted viewer.

## References

- [Original 3D Gaussian Splatting project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
- [Official rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu): covariance projection and Gaussian evaluation.
- [WebGPU Fundamentals: transparency and blending](https://webgpufundamentals.org/webgpu/lessons/webgpu-transparency.html).

## 10. Experimental tiled GPU path

The standalone `tiled_viewer` experiment provides a `GPU tiles` comparison checkbox.
Its `TiledRenderer` lives in `experiments/tiled.cpp` and `experiments/tiled.wgsl`;
the main app does not link or load it.
Projection runs once per Gaussian. Offscreen and negligible-opacity primitives are
culled, visible projected data is atomically compacted, and the shader emits one
record for each intersected tile. Opacity-aware bounds retain the existing 3-sigma
cutoff. An ellipse–rectangle test rejects tiles outside the contributing ellipse.

The portable radix-16 implementation uses stable workgroup partitions, per-bin
prefix offsets over workgroups, and global scatter. Least-significant-digit passes
sort reverse source ID, positive camera-space depth, then tile ID. Reverse IDs make
equal-depth compositing agree with the quad renderer's source-index tie order.
WGSL represents the keys as separate 32-bit words; it requires no 64-bit integers,
subgroups, CUDA, or CPU sorting. GPU-written dispatch arguments limit sorting work
to the generated records.

Each 16×16 workgroup shares batches of projected splats and accumulates front-to-back:
`color += T * alpha * rgb`, `T *= 1 - alpha`. A pixel stops at `T < 1e-4`; an entire
tile stops when every active pixel is finished. The output is presented by an ordinary
three-vertex fullscreen draw before the base app's ImGui pass.

Source records pack RGB into 8-bit channels (48 bytes total); projected records are
32 bytes and tile records are 16 bytes. This trades color precision for bandwidth.
The comparison renderer keeps its original full-precision buffers resident, so the
combined comparison app does not achieve a 25% reduction in total allocated GPU memory.

Tile-pair overflow preserves the previous complete output while an asynchronous
counter readback grows storage. If device binding/dispatch limits prevent further
growth, the UI reports the limit; reduce scene coverage or use the quad mode.

Validation and initial timing scopes are documented in `experiments/TILED.md`.
