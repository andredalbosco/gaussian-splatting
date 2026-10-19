# Predictions before measurement

Compare four equivalent vertex paths, holding quaternion/scale input, camera, projected support, 0.3 pixel-squared blur, blending, and Gaussian exponent fixed:

1. Pixel covariance, pixel fragment offsets, manual NDC vertex output (w = 1): current convention.
2. Pixel covariance and fragment offsets, homogeneous clip output: remove explicit center perspective divide; multiply NDC corner offset by center w.
3. NDC covariance and fragment offsets, homogeneous clip output: use dimensionless projection diagonal / z instead of pixel focal / z. Transform the blur into NDC units, and multiply corner offset by center w.
4. Homogeneous clip covariance and fragment offsets, homogeneous clip output: use a center-w-scaled perspective Jacobian, add blur scaled by (2w/viewport)^2, and add corner offsets directly to clip xy.

The last option is a per-Gaussian footprint on the constant-w quad plane; projecting only the linear xy rows of the projection matrix would omit the depth derivative and would be incorrect.

Prediction: homogeneous output alone might save a few scalar operations, but vertex/raster overhead should dominate at 10k. Changing covariance units does not change the Gaussian calculation fundamentally; no large speedup is expected. Clip units remove the 1/z scale from the first two Jacobian columns but require depth-scaled filtering. NDC/clip covariance may trade different floating-point rounding for similar compiled cost. Larger splats should conceal arithmetic differences behind fragments/blending. Off-screen vertex tests should expose differences more clearly, but still include primitive processing.

All predictions concern complete matched paths. Count formulas are source arithmetic, not verified GPU instruction counts. Background GPU applications remain running; small differences need per-trial stability.
