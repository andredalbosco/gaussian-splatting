import { mat3, mat4, vec3 } from 'wgpu-matrix';
import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { observeCanvasResize } from "./utils";
import shaderCode from './shaders.wgsl?raw';

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

const shaderModule = device.createShaderModule({ code: shaderCode });
const shaderDefs = makeShaderDataDefinitions(shaderCode);
const globalsData = makeStructuredView(shaderDefs.uniforms.globals);
const globalsBuffer = device.createBuffer({
    size: globalsData.arrayBuffer.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const depthFormat: GPUTextureFormat = 'depth24plus';
let depthTexture: GPUTexture;

function createDepthTexture() {
    depthTexture = device.createTexture({
        format: depthFormat,
        size: {width: canvas.width, height: canvas.height},
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
}

const globalsBindGroupLayout = device.createBindGroupLayout({
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: {type: 'uniform'},
    }]
});
const globalsBindGroup = device.createBindGroup({
    layout: globalsBindGroupLayout,
    entries: [{
        binding: 0,
        resource: globalsBuffer,
    }]
});

const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [globalsBindGroupLayout],
})
const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
        module: shaderModule,
    },
    depthStencil: {
        format: depthFormat,
        depthWriteEnabled: true,
        depthCompare: 'less-equal',
    },
    fragment: {
        module: shaderModule,
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

const eye = vec3.create(0, 0, 1);
const target = vec3.create(0, 0, 0);
const up = vec3.create(0, 1, 0);
const viewMatrix = mat4.lookAt(eye, target, up);

function updateProjectionMatrix() {
    const fovy = 60.0 * Math.PI / 180;
    const aspect = canvas.width / canvas.height;
    const zNear = 0.01;
    const zFar = 1000.0;
    const projectionMatrix = mat4.perspective(fovy, aspect, zNear, zFar);
    globalsData.set({ projectionMatrix });
}

function handleResize() {
    createDepthTexture();
    updateProjectionMatrix();
}

globalsData.set({
    viewMatrix,
});
observeCanvasResize(canvas, device, handleResize);
handleResize();

device.queue.writeBuffer(globalsBuffer, 0, globalsData.arrayBuffer);

function renderFrame() {
    device.queue.writeBuffer(globalsBuffer, 0, globalsData.arrayBuffer);

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
    pass.setBindGroup(0, globalsBindGroup);
    pass.draw(3, 1);
    pass.end();

    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(renderFrame);
}

requestAnimationFrame(renderFrame);
