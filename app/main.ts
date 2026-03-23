import Stats from 'stats.js';
import { mat4, vec3 } from 'wgpu-matrix';
import { getSizeAndAlignmentOfUnsizedArrayElement, makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { loadScene } from './loader';
import { observeCanvasResize } from './utils';
import shaderCode from './shaders.wgsl?raw';

const SCENE = 'fur_armchair';

const canvas = document.querySelector<HTMLCanvasElement>('canvas')!;
const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

if (!navigator.gpu) {
    throw new Error('WebGPU not available');
}

const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
});

if (!adapter) {
    throw new Error('WebGPU adapter not available');
}

const device = await adapter.requestDevice();
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
const context = canvas.getContext('webgpu');

if (!context) {
    throw new Error('WebGPU canvas context not available');
}

const webgpuContext = context;

webgpuContext.configure({
    device,
    format: canvasFormat,
    alphaMode: 'premultiplied',
});

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
    depthTexture?.destroy();
    depthTexture = device.createTexture({
        format: depthFormat,
        size: { width: canvas.width, height: canvas.height },
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
}

const globalsBindGroupLayout = device.createBindGroupLayout({
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' },
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
    }, {
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
    }, {
        binding: 3,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
    }],
});
const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
        bindGroupLayouts: [globalsBindGroupLayout],
    }),
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
        targets: [{
            format: canvasFormat,
            blend: {
                color: {
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha',
                },
                alpha: {
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha',
                },
            },
        }],
    },
});

const scene = await loadScene(SCENE);
const splatStdDev = Math.max(scene.radius / 150, 0.002);
const variance = splatStdDev * splatStdDev;

const { size: gaussian3dSize } = getSizeAndAlignmentOfUnsizedArrayElement(shaderDefs.storages.gaussians3d);
const gaussian3dData = makeStructuredView(
    shaderDefs.storages.gaussians3d,
    new ArrayBuffer(scene.count * gaussian3dSize),
);
gaussian3dData.set(Array.from({ length: scene.count }, (_, i) => {
    const base = i * 3;
    return {
        mean3d: scene.means.subarray(base, base + 3),
        cov3d: [
            variance, 0, 0,
            0, variance, 0,
            0, 0, variance,
        ],
        color: [0.95, 0.9, 0.82],
        opacity: 0.9,
    };
}));

const gaussians3dBuffer = device.createBuffer({
    size: gaussian3dData.arrayBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const { size: gaussian2dSize } = getSizeAndAlignmentOfUnsizedArrayElement(shaderDefs.storages.gaussians2d);
const gaussians2dBuffer = device.createBuffer({
    size: gaussian2dSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const gaussianIndexData = new Uint32Array(scene.count);
for (let i = 0; i < gaussianIndexData.length; i++) {
    gaussianIndexData[i] = i;
}

const gaussianIndexBuffer = device.createBuffer({
    size: gaussianIndexData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const globalsBindGroup = device.createBindGroup({
    layout: globalsBindGroupLayout,
    entries: [{
        binding: 0,
        resource: { buffer: globalsBuffer },
    }, {
        binding: 1,
        resource: { buffer: gaussians3dBuffer },
    }, {
        binding: 2,
        resource: { buffer: gaussians2dBuffer },
    }, {
        binding: 3,
        resource: { buffer: gaussianIndexBuffer },
    }],
});

const target: [number, number, number] = [...scene.center];
let orbitRadius = scene.radius * 4;
let azimuth = 0;
let polar = 1.2;
let dragging = false;
let lastX = 0;
let lastY = 0;
let orderEnabled = false;

function updateViewMatrix() {
    const sinPolar = Math.sin(polar);
    const eye = [
        target[0] + orbitRadius * sinPolar * Math.sin(azimuth),
        target[1] + orbitRadius * Math.cos(polar),
        target[2] + orbitRadius * sinPolar * Math.cos(azimuth),
    ] as const;
    globalsData.set({
        viewMatrix: mat4.lookAt(eye, target, [0, 1, 0]),
    });
}

function updateProjectionMatrix() {
    globalsData.set({
        projectionMatrix: mat4.perspective(60 * Math.PI / 180, canvas.width / canvas.height, 0.01, 1000),
    });
}

canvas.addEventListener('pointerdown', (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener('pointermove', (event) => {
    if (!dragging) {
        return;
    }
    azimuth -= (event.clientX - lastX) * 0.01;
    polar = Math.max(0.1, Math.min(Math.PI - 0.1, polar - (event.clientY - lastY) * 0.01));
    lastX = event.clientX;
    lastY = event.clientY;
});

canvas.addEventListener('pointerup', (event) => {
    dragging = false;
    canvas.releasePointerCapture(event.pointerId);
});

canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    orbitRadius = Math.max(0.1, orbitRadius * Math.exp(event.deltaY * 0.001));
}, { passive: false });

window.addEventListener('keydown', (event) => {
    if (event.repeat || event.key.toLowerCase() !== 'o') {
        return;
    }
    orderEnabled = !orderEnabled;
    if (!orderEnabled) {
        for (let i = 0; i < gaussianIndexData.length; i++) {
            gaussianIndexData[i] = i;
        }
        device.queue.writeBuffer(gaussianIndexBuffer, 0, gaussianIndexData);
    }
});

function handleResize() {
    createDepthTexture();
    updateProjectionMatrix();
}

observeCanvasResize(canvas, device, handleResize);
handleResize();
updateViewMatrix();

device.queue.writeBuffer(globalsBuffer, 0, globalsData.arrayBuffer);
device.queue.writeBuffer(gaussians3dBuffer, 0, gaussian3dData.arrayBuffer);
device.queue.writeBuffer(gaussianIndexBuffer, 0, gaussianIndexData);

function orderGaussians() {
    const gaussianPosition = vec3.create();
    const viewProjection = mat4.mul(globalsData.views.projectionMatrix, globalsData.views.viewMatrix);
    const depths = new Float32Array(scene.count);
    for (let i = 0; i < scene.count; i++) {
        const base = i * 3;
        vec3.set(scene.means[base + 0], scene.means[base + 1], scene.means[base + 2], gaussianPosition);
        depths[i] = vec3.transformMat4(gaussianPosition, viewProjection)[2];
    }
    gaussianIndexData.sort((a, b) => depths[b] - depths[a]);
    device.queue.writeBuffer(gaussianIndexBuffer, 0, gaussianIndexData);
}

function renderFrame() {
    stats.begin();
    updateViewMatrix();
    if (orderEnabled) {
        orderGaussians();
    }
    device.queue.writeBuffer(globalsBuffer, 0, globalsData.arrayBuffer);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: [0.03, 0.03, 0.04, 1],
            view: webgpuContext.getCurrentTexture().createView(),
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
            depthClearValue: 1,
        },
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, globalsBindGroup);
    pass.draw(6, scene.count);
    pass.end();

    device.queue.submit([encoder.finish()]);
    stats.end();
    requestAnimationFrame(renderFrame);
}

requestAnimationFrame(renderFrame);
