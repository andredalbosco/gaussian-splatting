import { observeCanvasResize } from "./utils";
import shaders from './shaders.wgsl?raw';

const canvas = document.querySelector<HTMLCanvasElement>("canvas")!;

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
const context = canvas.getContext("webgpu");

if (!context) {
    throw new Error("WebGPU canvas context not available");
}

context.configure({
    device,
    format,
});

observeCanvasResize(canvas, device);
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
