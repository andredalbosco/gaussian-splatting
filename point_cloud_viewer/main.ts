import Stats from 'stats.js';
import { Pane } from 'tweakpane';
import { OrbitCamera } from './camera';
import { loadGSplatData, validateGSplatData } from './loader';
import { Renderer, type PointCloud } from './renderer';
import { MemoryReadFileSystem, WebPCodec } from '@playcanvas/splat-transform';
import { observeCanvasResize } from '../app/utils';
import webpWasmUrl from '@playcanvas/splat-transform/lib/webp.wasm?url';

const SH_C0 = 0.28209479177387814;
const DEFAULT_SCENE = 'fur_armchair';
const SCENE_FILES = [
    'meta.json',
    'means_l.webp',
    'means_u.webp',
    'quats.webp',
    'scales.webp',
    'sh0.webp',
    'shN_centroids.webp',
    'shN_labels.webp',
] as const;
const sceneAssets = Object.entries(import.meta.glob<string>('../assets/*/*', {
    eager: true,
    query: '?url',
    import: 'default',
})).reduce((scenes, [path, url]) => {
    const [, scene, file] = path.match(/\.\.\/assets\/([^/]+)\/([^/]+)$/)!;
    (scenes[scene] ??= {})[file] = url;
    return scenes;
}, {} as Record<string, Record<string, string>>);
const sceneNames = Object.keys(sceneAssets).sort((a, b) =>
    a === DEFAULT_SCENE ? -1 : b === DEFAULT_SCENE ? 1 : a.localeCompare(b));
const sceneOptions = Object.fromEntries(sceneNames.map((name) => [
    name.replaceAll('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase()),
    name,
]));

function sigmoid(x: number) {
    return x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

function rotatePoint(x: number, y: number, z: number): [number, number, number] {
    return [x, -y, -z];
}

function gsplatToPointCloud(gsplatData: import('playcanvas').GSplatData, maxPoints = 50000): PointCloud {
    const x = gsplatData.getProp('x') as Float32Array;
    const y = gsplatData.getProp('y') as Float32Array;
    const z = gsplatData.getProp('z') as Float32Array;
    const scale0 = gsplatData.getProp('scale_0') as Float32Array;
    const scale1 = gsplatData.getProp('scale_1') as Float32Array;
    const scale2 = gsplatData.getProp('scale_2') as Float32Array;
    const fdc0 = gsplatData.getProp('f_dc_0') as Float32Array;
    const fdc1 = gsplatData.getProp('f_dc_1') as Float32Array;
    const fdc2 = gsplatData.getProp('f_dc_2') as Float32Array;
    const opacity = gsplatData.getProp('opacity') as Float32Array;

    const count = Math.min(gsplatData.numSplats, maxPoints);
    const points = new Float32Array(count * 8);
    let minX = x[0];
    let minY = y[0];
    let minZ = z[0];
    let maxX = x[0];
    let maxY = y[0];
    let maxZ = z[0];

    for (let j = 0; j < count; j++) {
        const i = Math.floor(j * gsplatData.numSplats / count);
        const out = j * 8;
        const [px, py, pz] = rotatePoint(x[i], y[i], z[i]);
        const s = Math.exp(Math.max(scale0[i], scale1[i], scale2[i]));
        const a = sigmoid(opacity[i]);

        minX = Math.min(minX, px);
        minY = Math.min(minY, py);
        minZ = Math.min(minZ, pz);
        maxX = Math.max(maxX, px);
        maxY = Math.max(maxY, py);
        maxZ = Math.max(maxZ, pz);

        points[out + 0] = px;
        points[out + 1] = py;
        points[out + 2] = pz;
        points[out + 3] = Math.max(1, Math.min(8, s * 200));
        points[out + 4] = Math.max(0, Math.min(1, (0.5 + SH_C0 * fdc0[i]) * a));
        points[out + 5] = Math.max(0, Math.min(1, (0.5 + SH_C0 * fdc1[i]) * a));
        points[out + 6] = Math.max(0, Math.min(1, (0.5 + SH_C0 * fdc2[i]) * a));
        points[out + 7] = 1;
    }

    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;

    return {
        points,
        count,
        center: [
            0.5 * (minX + maxX),
            0.5 * (minY + maxY),
            0.5 * (minZ + maxZ),
        ],
        radius: Math.max(sizeX, sizeY, sizeZ) * 1.5,
    };
}

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

const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
});

if (!adapter) {
    throw new Error('WebGPU adapter not available');
}

const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');

if (!context) {
    throw new Error('WebGPU canvas context not available');
}

const webgpuContext = context;

WebPCodec.wasmUrl = webpWasmUrl;
const sceneCache = new Map<string, Promise<PointCloud>>();

async function loadScenePointCloud(scene: string) {
    const fileSystem = new MemoryReadFileSystem();
    await Promise.all(SCENE_FILES.map(async (name) => {
        const url = sceneAssets[scene][name];
        fileSystem.set(name, new Uint8Array(await (await fetch(url)).arrayBuffer()));
    }));
    const gsplatData = await loadGSplatData('meta.json', fileSystem);
    validateGSplatData(gsplatData);
    return gsplatToPointCloud(gsplatData, 50000);
}

function getScenePointCloud(scene: string) {
    let pointCloud = sceneCache.get(scene);
    if (!pointCloud) {
        pointCloud = loadScenePointCloud(scene);
        sceneCache.set(scene, pointCloud);
    }
    return pointCloud;
}

const pointCloud = await getScenePointCloud(params.scene);
const renderer = new Renderer(device, webgpuContext, navigator.gpu.getPreferredCanvasFormat(), pointCloud);
const camera = new OrbitCamera(canvas);
camera.setScene(pointCloud.center, pointCloud.radius);

function handleResize() {
    camera.update(canvas.width / canvas.height);
    renderer.resize(canvas.width, canvas.height);
}

observeCanvasResize(canvas, device, handleResize);
handleResize();

let sceneRequest = 0;

async function setScene(scene: string) {
    const request = ++sceneRequest;
    const pointCloud = await getScenePointCloud(scene);
    if (request !== sceneRequest) {
        return;
    }
    renderer.setPointCloud(pointCloud);
    camera.setScene(pointCloud.center, pointCloud.radius);
}

sceneBinding.on('change', (event: { value: string }) => {
    void setScene(event.value);
});

function renderFrame() {
    stats.begin();
    camera.update(canvas.width / canvas.height);
    renderer.render(camera.viewProjection, canvas.width, canvas.height);
    stats.end();
    requestAnimationFrame(renderFrame);
}

requestAnimationFrame(renderFrame);
