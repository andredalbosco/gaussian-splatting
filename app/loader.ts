/**
 * Unified loader for all splat file formats using splat-transform.
 */

import {
    MemoryReadFileSystem,
    WebPCodec,
    getInputFormat,
    readFile,
    sortMortonOrder,
    Column,
    DataTable,
    ZipReadFileSystem
} from '@playcanvas/splat-transform';
import type { ColumnType, Options, ReadFileSystem } from '@playcanvas/splat-transform';
import { GSplatData } from 'playcanvas';
import webpWasmUrl from '@playcanvas/splat-transform/lib/webp.wasm?url';

const SCENE_FILES = [
    'meta.json',
    'means_l.webp',
    'means_u.webp',
    'quats.webp',
    'scales.webp',
    'sh0.webp',
    'shN_centroids.webp',
    'shN_labels.webp'
] as const;
const sceneAssets = Object.entries(import.meta.glob<string>('../assets/*/*', {
    eager: true,
    query: '?url',
    import: 'default'
})).reduce((scenes, [path, url]) => {
    const [, scene, file] = path.match(/\.\.\/assets\/([^/]+)\/([^/]+)$/)!;
    (scenes[scene] ??= {})[file] = url;
    return scenes;
}, {} as Record<string, Record<string, string>>);

type Scene = {
    means: Float32Array,
    count: number,
    center: [number, number, number],
    radius: number,
};

/**
 * Default options for readFile.
 */
const defaultOptions: Options = {
    iterations: 10,
    lodSelect: [0],
    unbundled: false,
    lodChunkCount: 512,
    lodChunkExtent: 16
};

/**
 * Map splat-transform column types to GSplatData property types.
 */
const columnTypeToGSplatType = (colType: ColumnType | null): string => {
    switch (colType) {
        case 'int8': return 'char';
        case 'uint8': return 'uchar';
        case 'int16': return 'short';
        case 'uint16': return 'ushort';
        case 'int32': return 'int';
        case 'uint32': return 'uint';
        case 'float32': return 'float';
        case 'float64': return 'double';
        default: return 'float';
    }
};

/**
 * Convert a splat-transform DataTable to PlayCanvas GSplatData.
 */
const dataTableToGSplatData = (dataTable: DataTable): GSplatData => {
    const properties = dataTable.columns.map((col: Column) => ({
        type: columnTypeToGSplatType(col.dataType),
        name: col.name,
        storage: col.data,
        byteSize: col.data.BYTES_PER_ELEMENT
    }));

    const gsplatData = new GSplatData([{
        name: 'vertex',
        count: dataTable.numRows,
        properties
    }]);

    // Support loading 2D splats by adding scale_2 property with almost 0 scale
    if (gsplatData.getProp('scale_0') && gsplatData.getProp('scale_1') && !gsplatData.getProp('scale_2')) {
        const scale2 = new Float32Array(gsplatData.numSplats).fill(Math.log(1e-6));
        gsplatData.addProp('scale_2', scale2);

        // Place the new scale_2 property just after scale_1
        const props = gsplatData.getElement('vertex').properties;
        props.splice(props.findIndex((prop: any) => prop.name === 'scale_1') + 1, 0, props.splice(props.length - 1, 1)[0]);
    }

    return gsplatData;
};

/**
 * Load a file using splat-transform and convert to GSplatData.
 * @param filename - The filename to load
 * @param fileSystem - The file system to read from
 * @param skipReorder - Skip morton reordering (for files already in morton order or animation playback)
 */
const loadGSplatData = async (filename: string, fileSystem: ReadFileSystem, skipReorder?: boolean): Promise<GSplatData> => {
    const inputFormat = getInputFormat(filename);
    const lowerFilename = filename.toLowerCase();

    // Handle bundled SOG (.sog extension) - wrap with ZipReadFileSystem
    if (inputFormat === 'sog' && lowerFilename.endsWith('.sog')) {
        const source = await fileSystem.createSource(filename);
        const zipFs = new ZipReadFileSystem(source);
        try {
            const tables = await readFile({
                filename: 'meta.json',
                inputFormat: 'sog',
                options: defaultOptions,
                params: [],
                fileSystem: zipFs
            });
            return dataTableToGSplatData(tables[0]);
        } finally {
            zipFs.close();
        }
    }

    // Read the file using splat-transform
    const tables = await readFile({
        filename,
        inputFormat,
        options: defaultOptions,
        params: [],
        fileSystem
    });

    // Reorder data into morton order for better render performance.
    // Skip reordering for:
    // - SOG format (already in morton order)
    // - Compressed PLY (already in morton order from write-compressed-ply)
    // - When skipReorder is true (ssproj files are already ordered, animation frames need speed)
    const isCompressedPly = lowerFilename.endsWith('.compressed.ply');
    if (inputFormat !== 'sog' && !isCompressedPly && !skipReorder) {
        const indices = new Uint32Array(tables[0].numRows);
        for (let i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        sortMortonOrder(tables[0], indices);
        tables[0].permuteRowsInPlace(indices);
    }

    // Convert to GSplatData (use first table, as most formats return single table)
    // LCC may return multiple tables for different LOD levels - we use the first (highest detail)
    return dataTableToGSplatData(tables[0]);
};

/**
 * Validate that GSplatData contains required properties.
 */
const validateGSplatData = (gsplatData: GSplatData): void => {
    const required = [
        'x', 'y', 'z',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'
    ];

    const missing = required.filter(x => !gsplatData.getProp(x));
    if (missing.length > 0) {
        throw new Error(`This file does not contain gaussian splatting data. The following properties are missing: ${missing.join(', ')}`);
    }
};

const rotatePoint = (x: number, y: number, z: number): [number, number, number] => [x, -y, -z];

const loadScene = async (scene: string): Promise<Scene> => {
    WebPCodec.wasmUrl = webpWasmUrl;

    const fileSystem = new MemoryReadFileSystem();
    await Promise.all(SCENE_FILES.map(async (name) => {
        const url = sceneAssets[scene][name];
        fileSystem.set(name, new Uint8Array(await (await fetch(url)).arrayBuffer()));
    }));

    const gsplatData = await loadGSplatData('meta.json', fileSystem);
    const x = gsplatData.getProp('x') as Float32Array;
    const y = gsplatData.getProp('y') as Float32Array;
    const z = gsplatData.getProp('z') as Float32Array;
    const means = new Float32Array(gsplatData.numSplats * 3);

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    for (let i = 0; i < gsplatData.numSplats; i++) {
        const [px, py, pz] = rotatePoint(x[i], y[i], z[i]);
        const offset = i * 3;
        means[offset + 0] = px;
        means[offset + 1] = py;
        means[offset + 2] = pz;
        minX = Math.min(minX, px);
        minY = Math.min(minY, py);
        minZ = Math.min(minZ, pz);
        maxX = Math.max(maxX, px);
        maxY = Math.max(maxY, py);
        maxZ = Math.max(maxZ, pz);
    }

    return {
        means,
        count: gsplatData.numSplats,
        center: [
            0.5 * (minX + maxX),
            0.5 * (minY + maxY),
            0.5 * (minZ + maxZ),
        ],
        radius: 0.5 * Math.max(maxX - minX, maxY - minY, maxZ - minZ),
    };
};

export {
    loadScene,
    loadGSplatData,
    type Scene,
    validateGSplatData
};
