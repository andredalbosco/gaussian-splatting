import shaderCode from './shaders.wgsl?raw';

export type PointCloud = {
    points: Float32Array,
    count: number,
    center: [number, number, number],
    radius: number,
};

export class Renderer {
    private device: GPUDevice;
    private context: GPUCanvasContext;
    private format: GPUTextureFormat;
    private pipeline: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup;
    private pointBuffer: GPUBuffer | null = null;
    private count = 0;
    private uniforms = new Float32Array(20);
    private depthFormat: GPUTextureFormat = 'depth24plus';
    private depthTexture: GPUTexture | null = null;
    private width = 0;
    private height = 0;
    readonly pointSize = 1;

    constructor(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat, pointCloud: PointCloud) {
        this.device = device;
        this.context = context;
        this.format = format;
        this.uniformBuffer = device.createBuffer({
            size: this.uniforms.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const shaderModule = device.createShaderModule({ code: shaderCode });
        this.pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                buffers: [{
                    arrayStride: 32,
                    stepMode: 'instance',
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x4' },
                        { shaderLocation: 1, offset: 16, format: 'float32x4' },
                    ],
                }],
            },
            fragment: {
                module: shaderModule,
                targets: [{ format }],
            },
            primitive: {
                topology: 'triangle-list',
            },
            depthStencil: {
                format: this.depthFormat,
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
        });

        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer },
            }],
        });

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied',
        });

        this.setPointCloud(pointCloud);
    }

    resize(width: number, height: number) {
        if (width === this.width && height === this.height) {
            return;
        }

        this.width = width;
        this.height = height;
        this.depthTexture?.destroy();
        this.depthTexture = this.device.createTexture({
            size: { width, height },
            format: this.depthFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    setPointCloud(pointCloud: PointCloud) {
        this.count = pointCloud.count;
        this.pointBuffer?.destroy();
        this.pointBuffer = this.device.createBuffer({
            size: pointCloud.points.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.pointBuffer, 0, new Float32Array(pointCloud.points));
    }

    render(viewProjection: Float32Array, width: number, height: number) {
        if (!this.depthTexture) {
            this.resize(width, height);
        }

        this.uniforms.set(viewProjection, 0);
        this.uniforms[16] = width;
        this.uniforms[17] = height;
        this.uniforms[18] = this.pointSize;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniforms);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: [0.03, 0.03, 0.04, 1],
            }],
            depthStencilAttachment: {
                view: this.depthTexture!.createView(),
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1,
            },
        });

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.setVertexBuffer(0, this.pointBuffer!);
        pass.draw(6, this.count);
        pass.end();

        this.device.queue.submit([encoder.finish()]);
    }
}
