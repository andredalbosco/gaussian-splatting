import { mat3, vec3 } from 'wgpu-matrix';
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
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
const context = canvas.getContext("webgpu")!;

if (!context) {
    throw new Error("WebGPU canvas context not available");
}

context.configure({
    device,
    format: canvasFormat,
})!;

const shader = device.createShaderModule({ code: shaders });

const depthFormat: GPUTextureFormat = 'depth24plus';
let depthTexture: GPUTexture;

function createDepthTexture() {
    depthTexture = device.createTexture({
        format: depthFormat,
        size: {width: canvas.width, height: canvas.height},
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
}

observeCanvasResize(canvas, device, createDepthTexture);
createDepthTexture();

const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
        module: shader,
    },
    depthStencil: {
        format: depthFormat,
        depthWriteEnabled: true,
        depthCompare: 'less-equal',
    },
    fragment: {
        module: shader,
        targets: [{ format: canvasFormat }],
    },
})

const gaussMeans = [
    vec3.create(0.0, 0.0, 0.01),
    vec3.create(0.0, 0.0, 0.02),
    vec3.create(0.0, 0.0, 0.04),
];

const gaussCov = [
    mat3.create(
        0.2, 0.0, 0.0,
        0.0, 0.2, 0.0,
        0.0, 0.0, 0.2,
    ),
    mat3.create(
        1.0, 0.0, 0.0,
        0.0, 0.05, 0.0,
        0.0, 0.0, 0.1,
    ),
    mat3.create(
        0.1, 0.0, 0.0,
        0.0, 0.1, 0.0,
        0.0, 0.0, 0.1,
    ),
];

const gaussOpacities = [0.7, 0.7, 0.7];

const gaussColors = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

function renderFrame() {
    const encoder = device.createCommandEncoder();

    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: [0, 0, 0, 1],
            view: context.getCurrentTexture().createView(),
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
            depthClearValue: 1.0,
        }
    });
    pass.setPipeline(pipeline);
    pass.draw(3, 1);
    pass.end();

    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(renderFrame);
}

requestAnimationFrame(renderFrame);
