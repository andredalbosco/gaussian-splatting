const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
    throw new Error("App container not found");
}

if (!navigator.gpu) {
    throw new Error("WebGPU not available");
}

const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
});

if (!adapter) {
    throw new Error("WebGPU adapter not available");
}

const device = await adapter.requestDevice();
const format = navigator.gpu.getPreferredCanvasFormat();

const canvas = document.createElement("canvas");
canvas.width = window.innerWidth * window.devicePixelRatio;
canvas.height = window.innerHeight * window.devicePixelRatio;
canvas.style.width = "100%";
canvas.style.height = "100%";
app.append(canvas);

const context = canvas.getContext("webgpu");

if (!context) {
    throw new Error("WebGPU canvas context not available");
}

context.configure({
    device,
    format,
    alphaMode: "opaque",
});

import shaders from './shaders.wgsl?raw';
const shader = device.createShaderModule({ code: shaders });

const depthFormat: GPUTextureFormat = 'depth24plus';

device.createRenderPipeline({
    layout: 'auto',
    vertex: {
        module: shader,
    },
    depthStencil: {
        format: depthFormat,
    },
    fragment: {
        module: shader,
        targets: []
    },
})
