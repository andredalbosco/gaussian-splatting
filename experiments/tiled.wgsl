@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> projected: array<Projected>;
@group(0) @binding(3) var<storage, read_write> pairs: array<vec4u>;
@group(0) @binding(4) var<storage, read_write> scratch: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> histogram: array<u32>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> ranges: array<vec2u>;
@group(1) @binding(1) var<storage, read_write> dispatch: array<u32>;
@group(0) @binding(9) var image: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(0) var<uniform> radix: vec4u;

struct Camera {
    view_projection: mat4x4f,
    size: vec4u, // width, height, source count, pair capacity
    background: vec4f,
}

// 48 bytes, with RGB packed into one word. Opacity retains full precision.
struct Gaussian {
    position: vec3f,
    opacity: f32,
    rotation: vec4f,
    scale: vec3f,
    color: u32,
}

// 32 bytes per visible Gaussian, computed once and shared by its tiles.
struct Projected {
    center: vec2f,
    depth: f32,
    opacity: f32,
    conic: vec3f,
    color: u32,
}

fn rotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn project_axis(g: Gaussian, clip_pos: vec4f, axis: vec3f) -> vec2f {
    let a = camera.view_projection * vec4f(rotate(g.rotation, axis), 0);
    return (a.xy * clip_pos.w - clip_pos.xy * a.w) / (clip_pos.w * clip_pos.w)
        * vec2f(camera.size.xy) * vec2f(0.5, -0.5);
}

fn quadratic(p: vec2f, conic: vec3f) -> f32 {
    return conic.x*p.x*p.x + 2.0*conic.y*p.x*p.y + conic.z*p.y*p.y;
}

fn tile_intersects(lo: vec2f, hi: vec2f, conic: vec3f, cutoff: f32) -> bool {
    if (all(lo <= vec2f(0)) && all(hi >= vec2f(0))) { return true; }
    let a = quadratic(vec2f(lo.x, clamp(-conic.y*lo.x/conic.z, lo.y, hi.y)), conic);
    let b = quadratic(vec2f(hi.x, clamp(-conic.y*hi.x/conic.z, lo.y, hi.y)), conic);
    let c = quadratic(vec2f(clamp(-conic.y*lo.y/conic.x, lo.x, hi.x), lo.y), conic);
    let d = quadratic(vec2f(clamp(-conic.y*hi.y/conic.x, lo.x, hi.x), hi.y), conic);
    return min(min(a,b), min(c,d)) <= cutoff;
}

@compute @workgroup_size(256) fn project(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= camera.size.z) { return; }
    let g = gaussians[id.x];
    if (g.opacity < 1.0/255.0) { return; }
    let clip_pos = camera.view_projection * vec4f(g.position, 1);
    if (clip_pos.w <= 0 || clip_pos.z < 0 || clip_pos.z > clip_pos.w) { return; }
    let ax = project_axis(g, clip_pos, vec3f(g.scale.x,0,0));
    let ay = project_axis(g, clip_pos, vec3f(0,g.scale.y,0));
    let az = project_axis(g, clip_pos, vec3f(0,0,g.scale.z));
    let x = vec3f(ax.x,ay.x,az.x);
    let y = vec3f(ax.y,ay.y,az.y);
    let a = dot(x,x)+0.3;
    let b = dot(x,y);
    let c = dot(y,y)+0.3;
    let det = a*c-b*b;
    if (det <= 0) { return; }
    let conic = vec3f(c,-b,a)/det;
    let cutoff = min(9.0, 2.0*log(g.opacity*255.0));
    let extent = sqrt(cutoff*vec2f(a,c));
    let center = (clip_pos.xy/clip_pos.w*vec2f(0.5,-0.5)+0.5)*vec2f(camera.size.xy);
    if (any(center+extent < vec2f(0)) || any(center-extent >= vec2f(camera.size.xy))) { return; }
    let tiles = (camera.size.xy+15u)/16u;
    let lo = vec2u(clamp(floor((center-extent)/16.0), vec2f(0), vec2f(tiles-1)));
    let hi = vec2u(clamp(floor((center+extent)/16.0), vec2f(0), vec2f(tiles-1)));
    let index = atomicAdd(&counters[0], 1);
    projected[index] = Projected(center, clip_pos.w, g.opacity, conic, g.color);
    for (var ty=lo.y; ty<=hi.y; ty++) {
        for (var tx=lo.x; tx<=hi.x; tx++) {
            let tile = vec2u(tx,ty);
            if (!tile_intersects(vec2f(tile*16u)-center, vec2f((tile+1u)*16u)-center, conic, cutoff)) { continue; }
            let slot = atomicAdd(&counters[1],1);
            if (slot < camera.size.w) {
                pairs[slot] = vec4u(bitcast<u32>(clip_pos.w), ty*tiles.x+tx, index, ~id.x);
            } else {
                atomicStore(&counters[2],1);
            }
        }
    }
}

@compute @workgroup_size(1) fn prepare_dispatch() {
    dispatch[0] = (min(atomicLoad(&counters[1]), camera.size.w)+255u)/256u;
    dispatch[1] = 1;
    dispatch[2] = 1;
}

var<workgroup> items: array<vec4u,256>;
var<workgroup> temporary: array<vec4u,256>;
var<workgroup> prefix: array<u32,256>;
var<workgroup> bins: array<atomic<u32>,16>;
var<workgroup> totals: array<u32,16>;

fn digit(item: vec4u) -> u32 {
    return (item[radix.y] >> radix.x) & 15u;
}

// Stable four-bit local radix partition. No subgroup or forward-progress assumptions.
@compute @workgroup_size(256) fn radix_local(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3u
) {
    let n = min(atomicLoad(&counters[1]),camera.size.w);
    let index = group.x*256u+lane;
    items[lane] = vec4u(0xffffffffu);
    if (index < n) { items[lane] = pairs[index]; }
    workgroupBarrier();
    for (var bit=0u; bit<4u; bit++) {
        let zero = 1u-((digit(items[lane]) >> bit)&1u);
        prefix[lane] = zero;
        workgroupBarrier();
        for (var step=1u; step<256u; step*=2u) {
            var value = prefix[lane];
            if (lane >= step) { value += prefix[lane-step]; }
            workgroupBarrier();
            prefix[lane] = value;
            workgroupBarrier();
        }
        let zeros_before = prefix[lane]-zero;
        let destination = select(prefix[255]+lane-zeros_before, zeros_before, zero==1u);
        temporary[destination] = items[lane];
        workgroupBarrier();
        items[lane] = temporary[lane];
        workgroupBarrier();
    }
    if (lane < 16u) { atomicStore(&bins[lane],0); }
    workgroupBarrier();
    if (index < n) {
        scratch[index] = items[lane];
        atomicAdd(&bins[digit(items[lane])],1);
    }
    workgroupBarrier();
    if (lane < 16u) { histogram[lane*((n+255u)/256u)+group.x] = atomicLoad(&bins[lane]); }
}

// One independent lane per bin: prefix over workgroups, with a global bin base.
@compute @workgroup_size(16) fn radix_prefix(@builtin(local_invocation_index) bin: u32) {
    let groups = (min(atomicLoad(&counters[1]),camera.size.w)+255u)/256u;
    var total=0u;
    for (var g=0u; g<groups; g++) { total += histogram[bin*groups+g]; }
    totals[bin]=total;
    workgroupBarrier();
    var offset=0u;
    for (var b=0u; b<bin; b++) { offset += totals[b]; }
    for (var g=0u; g<groups; g++) {
        let count=histogram[bin*groups+g];
        histogram[bin*groups+g]=offset;
        offset += count;
    }
}

@compute @workgroup_size(256) fn radix_scatter(@builtin(global_invocation_id) id: vec3u) {
    let n=min(atomicLoad(&counters[1]),camera.size.w);
    if (id.x >= n) { return; }
    let group=id.x/256u;
    let item=scratch[id.x];
    let bin=digit(item);
    var lo=group*256u;
    var hi=id.x;
    while (lo<hi) {
        let mid=(lo+hi)/2u;
        if (digit(scratch[mid]) < bin) { lo=mid+1u; } else { hi=mid; }
    }
    let rank=id.x-lo;
    pairs[histogram[bin*((n+255u)/256u)+group]+rank]=item;
}

@compute @workgroup_size(256) fn tile_ranges(@builtin(global_invocation_id) id: vec3u) {
    let n=min(atomicLoad(&counters[1]),camera.size.w);
    let i=id.x;
    if (i>=n) { return; }
    let tile=pairs[i].y;
    if (i==0u || pairs[i-1u].y!=tile) { ranges[tile].x=i; }
    if (i+1u==n || pairs[i+1u].y!=tile) { ranges[tile].y=i+1u; }
}

var<workgroup> tile_range: vec2u;
var<workgroup> cache: array<Projected,64>;
var<workgroup> finished: array<u32,256>;
var<workgroup> all_finished: u32;

@compute @workgroup_size(16,16) fn rasterize(
    @builtin(local_invocation_index) lane: u32,
    @builtin(global_invocation_id) pixel: vec3u,
    @builtin(workgroup_id) tile: vec3u
) {
    let tiles=(camera.size.xy+15u)/16u;
    if (lane==0u) { tile_range=ranges[tile.y*tiles.x+tile.x]; }
    let range=workgroupUniformLoad(&tile_range);
    let inside=all(pixel.xy<camera.size.xy);
    var color=vec3f(0);
    var transmission=1.0;
    for (var base=range.x; base<range.y; base+=64u) {
        if (lane<64u && base+lane<range.y) { cache[lane]=projected[pairs[base+lane].z]; }
        workgroupBarrier();
        if (inside && transmission>=0.0001) {
            for (var j=0u; j<min(64u,range.y-base); j++) {
                let g=cache[j];
                let q=quadratic(vec2f(pixel.xy)+0.5-g.center,g.conic);
                if (q>9.0) { continue; }
                let alpha=min(0.99,g.opacity*exp(-0.5*q));
                if (alpha<1.0/255.0) { continue; }
                color += transmission*alpha*unpack4x8unorm(g.color).rgb;
                transmission *= 1.0-alpha;
                if (transmission<0.0001) { break; }
            }
        }
        finished[lane]=select(0u,1u,!inside || transmission<0.0001);
        workgroupBarrier();
        if (lane==0u) {
            var count=0u;
            for (var j=0u;j<256u;j++) { count+=finished[j]; }
            all_finished=count;
        }
        if (workgroupUniformLoad(&all_finished)==256u) { break; }
    }
    if (inside) {
        var result=vec4f(color+transmission*camera.background.rgb,1);
        // An exhausted pair buffer is explicit, never silently missing geometry.
        if (atomicLoad(&counters[2])!=0u) { return; }
        textureStore(image,pixel.xy,result);
    }
}
