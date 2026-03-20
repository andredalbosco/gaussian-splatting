/**
 * Keeps the canvas drawing buffer matched to its displayed size.
 * Uses device-pixel measurements when the browser exposes them.
 */
export function observeCanvasResize(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
) {
    const maxTextureDimension2D = device.limits.maxTextureDimension2D;
    const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
            const width = entry.devicePixelContentBoxSize?.[0].inlineSize
                ?? entry.contentBoxSize[0].inlineSize * window.devicePixelRatio;
            const height = entry.devicePixelContentBoxSize?.[0].blockSize
                ?? entry.contentBoxSize[0].blockSize * window.devicePixelRatio;

            canvas.width = Math.max(1, Math.min(width, maxTextureDimension2D));
            canvas.height = Math.max(1, Math.min(height, maxTextureDimension2D));
        }
    });

    try {
        resizeObserver.observe(canvas, { box: "device-pixel-content-box" });
    } catch {
        resizeObserver.observe(canvas, { box: "content-box" });
    }
}
