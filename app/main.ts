import Stats from 'stats.js';
import { Pane } from 'tweakpane';
import { mat4, vec3 } from 'wgpu-matrix';
import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';
import { createGaussian3dData, createGaussianIndexData, DEFAULT_SCENE, getScene, sceneOptions, type Scene } from './loader';
import { observeCanvasResize } from './utils';
import shaderCode from './shaders.wgsl?raw';

const canvas = document.querySelector<HTMLCanvasElement>('canvas')!;
const overlay = document.createElement('div');
overlay.className = 'overlay';
document.body.appendChild(overlay);

const stats = new Stats();
stats.showPanel(0);
stats.dom.style.position = 'relative';
stats.dom.style.left = '0';
stats.dom.style.top = '0';
overlay.appendChild(stats.dom);

const pane = new Pane({ container: overlay });
pane.element.style.width = '220px';
const params = { scene: DEFAULT_SCENE };
const sceneBinding = (pane as any).addBinding(params, 'scene', { label: 'scene', options: sceneOptions });

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

const gaussianPosition = vec3.create();
const target: [number, number, number] = [0, 0, 0];
let scene = await getScene(params.scene);
let gaussians3dBuffer: GPUBuffer;
let gaussianIndexData: Uint32Array;
let gaussianIndexBuffer: GPUBuffer;
let globalsBindGroup: GPUBindGroup;
let orbitRadius = 1;
let azimuth = 0;
let polar = 1.2;
let dragging = false;
let lastX = 0;
let lastY = 0;
let orderEnabled = false;
let sceneRequest = 0;

function applyScene(nextScene: Scene) {
    const gaussian3dData = createGaussian3dData(nextScene, shaderDefs.storages.gaussians3d);

    const nextGaussians3dBuffer = device.createBuffer({
        size: gaussian3dData.arrayBuffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const nextGaussianIndexData = createGaussianIndexData(nextScene.count);
    const nextGaussianIndexBuffer = device.createBuffer({
        size: nextGaussianIndexData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const nextGlobalsBindGroup = device.createBindGroup({
        layout: globalsBindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: globalsBuffer },
        }, {
            binding: 1,
            resource: { buffer: nextGaussians3dBuffer },
        }, {
            binding: 2,
            resource: { buffer: nextGaussianIndexBuffer },
        }],
    });

    device.queue.writeBuffer(nextGaussians3dBuffer, 0, gaussian3dData.arrayBuffer);
    device.queue.writeBuffer(nextGaussianIndexBuffer, 0, nextGaussianIndexData);
    gaussians3dBuffer?.destroy();
    gaussianIndexBuffer?.destroy();
    scene = nextScene;
    gaussians3dBuffer = nextGaussians3dBuffer;
    gaussianIndexData = nextGaussianIndexData;
    gaussianIndexBuffer = nextGaussianIndexBuffer;
    globalsBindGroup = nextGlobalsBindGroup;
    target[0] = nextScene.center[0];
    target[1] = nextScene.center[1];
    target[2] = nextScene.center[2];
    orbitRadius = nextScene.radius * 4;
}

async function setScene(name: string) {
    const request = ++sceneRequest;
    const nextScene = await getScene(name);
    if (request !== sceneRequest) {
        return;
    }
    applyScene(nextScene);
}

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
        device.queue.writeBuffer(gaussianIndexBuffer, 0, gaussianIndexData.buffer as ArrayBuffer);
    }
});

function handleResize() {
    createDepthTexture();
    updateProjectionMatrix();
}

observeCanvasResize(canvas, device, handleResize);
applyScene(scene);
handleResize();
updateViewMatrix();

device.queue.writeBuffer(globalsBuffer, 0, globalsData.arrayBuffer);

function orderGaussians() {
    const viewProjection = mat4.mul(globalsData.views.projectionMatrix, globalsData.views.viewMatrix);
    const depths = new Float32Array(scene.count);
    for (let i = 0; i < scene.count; i++) {
        const base = i * 3;
        vec3.set(scene.means[base + 0], scene.means[base + 1], scene.means[base + 2], gaussianPosition);
        depths[i] = vec3.transformMat4(gaussianPosition, viewProjection)[2];
    }
    gaussianIndexData.sort((a, b) => depths[b] - depths[a]);
    device.queue.writeBuffer(gaussianIndexBuffer, 0, gaussianIndexData.buffer as ArrayBuffer);
}

sceneBinding.on('change', (event: { value: string }) => {
    void setScene(event.value);
});

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
