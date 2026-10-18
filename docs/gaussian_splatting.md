# Gaussian splatting

This viewer renders a scene as overlapping, transparent 3D Gaussians. Each Gaussian becomes
an elliptical footprint on the image. The vertex shader computes its projected shape;
the fragment shader evaluates its opacity; blending combines it with the other splats.

The current implementation loads Luigi, uses fixed per-Gaussian RGB colors, and sorts
Gaussian centers on the CPU. It does not evaluate view-dependent spherical harmonics.

## 1. Data representation

A Gaussian stores four vectors in [`Gaussian`](../src/gaussian.hpp):

| Field | Meaning |
|---|---|
| Position | World-space center, $\mu$ |
| Scale | Three positive standard deviations, $s_x,s_y,s_z$ |
| Rotation | Unit quaternion, stored as $(x,y,z,w)$ |
| Color | Fixed RGB and peak opacity |

[`readPlyFile`](../src/utils.cpp) decodes logarithmic scales with `exp`, opacity logits
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
center by multiplying by the view-projection matrix and dividing clip-space $xy$ by $w$.

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

In [`projected_axis`](../src/shaders.wgsl), we differentiate the homogeneous divide
directly. If $c$ is the center in clip space and $a$ is a transformed axis, then

$$
\delta\left(\frac{c_{xy}}{c_w}\right)
=\frac{a_{xy}c_w-c_{xy}a_w}{c_w^2}.
$$

The shader multiplies this NDC displacement by half the framebuffer width and height,
expressing the projected differential in pixel units. Its vertical coordinate points up;
the interpolated fragment offsets use that same convention.

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

Writing their horizontal components as $x=(x_1,x_2,x_3)$ and vertical components as
$y=(y_1,y_2,y_3)$, the shader computes

```wgsl
a = dot(x, x) + 0.3;
b = dot(x, y);
c = dot(y, y) + 0.3;
```

The added diagonal variance is a small screen-space filter, measured in pixel². It keeps
tiny footprints from collapsing. The renderer does not apply an amplitude compensation
for this filter, so filtering changes the integrated contribution of small splats.

## 5. Constructing the bounding quad

For pixel displacement $d$ from the projected center, define

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

The shader converts the pixel offsets back to NDC for rasterization. The quad's clip $w$
is one, so interpolation of the pixel offsets is linear across the image.

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

Transparent contributions depend on their order. [`GSplatApp::updateCamera`](../src/main.cpp)
computes a center-depth key for each Gaussian and sorts indices from far to near. The key
can use $f\cdot\mu$, where $f$ is the camera's forward direction: subtracting the camera
position would add the same constant to every key and would not change the order.

The Gaussian storage buffer remains unchanged. The viewer uploads the sorted indices,
and the vertex shader fetches `gaussians[order[instance_index]]`. Sorting is reused while
the view direction stays unchanged; FOV, resize, and translation along the orbit direction
do not change the mathematical depth order.

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
| [`src/utils.cpp`](../src/utils.cpp) | PLY decoding and window/WebGPU setup |
| [`src/camera.cpp`](../src/camera.cpp) | View and perspective matrices |
| [`src/main.cpp`](../src/main.cpp) | Buffers, sorting, draw loop, blending, controls |
| [`src/shaders.wgsl`](../src/shaders.wgsl) | Projected covariance, quad, Gaussian evaluation |
| [`experiments/validate_splats.cpp`](../experiments/validate_splats.cpp) | Native numerical shader checks |

`init()` creates the window, device, rendering pipeline, and UI without loading a scene.
`main()` loads the initial Gaussians with `loadGaussians(path)`, then calls the argument-free
`run()` to start the frame loop. A later `loadGaussians(path)`
replaces the Gaussian data, refreshes the scene buffers and bind group, reframes the camera,
and invalidates the cached sort direction so the next frame sorts the new scene.

The validation program compares GPU output with a CPU reference that estimates the
projection differential using finite differences. It checks single-splat falloff, a known
two-layer blend, rotated anisotropic splats, an off-axis center, a non-square viewport,
and behind/near/far rejection. The five cases passed with at most 1/255 channel error
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
