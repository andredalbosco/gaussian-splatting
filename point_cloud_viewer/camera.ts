import { mat4, vec3 } from 'wgpu-matrix';

export class OrbitCamera {
    private target = vec3.create(0, 0, 0);
    private radius = 5;
    private azimuth = 0;
    private polar = 1.2;
    private dragging = false;
    private lastX = 0;
    private lastY = 0;
    readonly viewProjection = mat4.identity();

    constructor(canvas: HTMLCanvasElement) {
        canvas.addEventListener('pointerdown', (event) => {
            this.dragging = true;
            this.lastX = event.clientX;
            this.lastY = event.clientY;
            canvas.setPointerCapture(event.pointerId);
        });

        canvas.addEventListener('pointermove', (event) => {
            if (!this.dragging) {
                return;
            }
            this.azimuth -= (event.clientX - this.lastX) * 0.01;
            this.polar = Math.max(0.1, Math.min(Math.PI - 0.1, this.polar - (event.clientY - this.lastY) * 0.01));
            this.lastX = event.clientX;
            this.lastY = event.clientY;
        });

        canvas.addEventListener('pointerup', (event) => {
            this.dragging = false;
            canvas.releasePointerCapture(event.pointerId);
        });

        canvas.addEventListener('wheel', (event) => {
            event.preventDefault();
            this.radius = Math.max(0.1, this.radius * Math.exp(event.deltaY * 0.001));
        }, { passive: false });
    }

    setScene(center: [number, number, number], radius: number) {
        this.target = vec3.create(...center);
        this.radius = radius * 1.5;
    }

    update(aspect: number) {
        const sinPolar = Math.sin(this.polar);
        const eye = vec3.create(
            this.target[0] + this.radius * sinPolar * Math.sin(this.azimuth),
            this.target[1] + this.radius * Math.cos(this.polar),
            this.target[2] + this.radius * sinPolar * Math.cos(this.azimuth),
        );
        const view = mat4.lookAt(eye, this.target, vec3.create(0, 1, 0));
        const projection = mat4.perspective(60 * Math.PI / 180, aspect, 0.01, 1000);
        mat4.mul(projection, view, this.viewProjection);
    }
}
