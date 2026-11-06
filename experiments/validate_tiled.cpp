#include "camera.hpp"
#include "colored_gaussian.hpp"
#include "tiled.hpp"
#include "wgpu_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>

#include <Eigen/Geometry>

int main() {
    auto instance = createInstance();
    auto adapter = requestAdapter(instance, {});
    wgpu::Device device;
    wgpu::FeatureName timestampFeature = wgpu::FeatureName::TimestampQuery;
    wgpu::DeviceDescriptor deviceDesc;
    deviceDesc.requiredFeatureCount = 1;
    deviceDesc.requiredFeatures = &timestampFeature;
    instance.WaitAny(adapter.RequestDevice(&deviceDesc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus, wgpu::Device result, wgpu::StringView) { device = result; }), UINT64_MAX);
    auto queue = device.GetQueue();
    device.PushErrorScope(wgpu::ErrorFilter::Validation);
    auto shader = readWgslFile(device, SRC_DIR "/shaders.wgsl");
    wgpu::BlendState blend {
        .color = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
        .alpha = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha}};
    wgpu::ColorTargetState target {.format = wgpu::TextureFormat::RGBA16Float, .blend = &blend};
    wgpu::FragmentState fragment {.module = shader, .entryPoint = "fs_main", .targetCount = 1, .targets = &target};
    wgpu::RenderPipelineDescriptor desc {.vertex = {.module = shader, .entryPoint = "vs_main"},
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleStrip}, .fragment = &fragment};
    auto baseline = device.CreateRenderPipeline(&desc);
    target.format = wgpu::TextureFormat::RGBA8Unorm;
    auto quadPipeline = device.CreateRenderPipeline(&desc);
    target.format = wgpu::TextureFormat::RGBA16Float;
    TiledRenderer tiled;
    tiled.init(device, wgpu::TextureFormat::RGBA8Unorm);
    auto buffer = [&](uint64_t size, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor d {.usage = usage, .size = std::max(uint64_t(16), size)};
        return device.CreateBuffer(&d);
    };
    auto map = [&](wgpu::Buffer b) {
        bool ok = false;
        instance.WaitAny(b.MapAsync(wgpu::MapMode::Read, 0, b.GetSize(), wgpu::CallbackMode::WaitAnyOnly,
            [&](wgpu::MapAsyncStatus status, wgpu::StringView) { ok = status == wgpu::MapAsyncStatus::Success; }), UINT64_MAX);
        if (!ok) std::abort();
    };
    int failures = 0;
    std::mt19937 rng(20261019);
    std::uniform_real_distribution<float> unit(0, 1);
    for (int test = 0; test < 11; test++) {
        uint32_t w = test == 9 ? 1280 : (test == 7 || test == 8) ? 640 : 257;
        uint32_t h = test == 9 ? 720 : (test == 7 || test == 8) ? 360 : 129;
        Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f projection = projectionMatrix(60, float(w)/h, 0.1f, 20);
        Eigen::Matrix4f vp = projection;
        std::vector<ColoredGaussian> data {{{0,0,2,1},{0.2f,0.2f,0.2f,0},{0,0,0,1},{0.8f,0.2f,0.1f,0.5f}}};
        if (test == 0) data.clear();
        if (test == 2 || test == 3) {
            data.push_back({{0,0,test == 2 ? 3.0f : 2.0f,1},{0.3f,0.3f,0.3f,0},{0,0,0,1},{0,0.2f,1,0.6f}});
        }
        if (test == 4) {
            data[0].scale = {0.5f,0.025f,0.1f,0};
            data[0].rotation = Eigen::Quaternionf(Eigen::AngleAxisf(0.8f, Eigen::Vector3f::UnitZ())).coeffs();
        }
        if (test == 5) data[0].position.z() = -2;
        if (test == 6) {
            for (int i=0;i<800;i++) data.push_back({{unit(rng)*4-2,unit(rng)*2-1,1+unit(rng)*5,1},
                {0.02f+unit(rng)*0.1f,0.015f+unit(rng)*0.08f,0.05f,0},{0,0,0,1},
                {unit(rng),unit(rng),unit(rng),unit(rng)}});
            data.push_back({{100,0,2,1},{0.1f,0.1f,0.1f,0},{0,0,0,1},{1,1,1,1}});
            data.push_back({{0,0,2,1},{0.1f,0.1f,0.1f,0},{0,0,0,1},{1,1,1,0.001f}});
        }
        if (test == 7) {
            data.clear();
            for (int i=0;i<10000;i++) data.push_back({{unit(rng)*10-5,unit(rng)*6-3,0.5f+unit(rng)*8,1},
                {0.01f+unit(rng)*0.04f,0.01f+unit(rng)*0.04f,0.02f,0},{0,0,0,1},
                {unit(rng),unit(rng),unit(rng),unit(rng)*0.8f}});
        }
        if (test == 8) {
            data.clear();
            for (int i=0;i<400;i++) data.push_back({{0,0,2+0.002f*i,1},{1.2f,1.2f,1.2f,0},
                {0,0,0,1},{0.3f,0.6f,0.9f,0.8f}});
        }
        if (test == 9) {
            data = readColoredPlyFile(ASSETS_DIR "/luigi.ply");
            OrbitCamera camera;
            Eigen::AlignedBox3f bounds;
            for (const auto& g : data) bounds.extend(g.position.head<3>());
            camera.center = bounds.center();
            camera.distance = 1.75f * bounds.sizes().maxCoeff();
            view = viewMatrix(camera.position(), camera.center, Eigen::Vector3f::UnitY());
            vp = camera.viewProjection(float(w)/h);
        }
        if (test == 10) tiled.capacity = 8;
        tiled.load(data);
        if (test == 10) {
            for (int frame=0;frame<5;frame++) {
                auto e=device.CreateCommandEncoder();
                tiled.record(e,vp,w,h);
                auto c=e.Finish(); queue.Submit(1,&c);
                instance.WaitAny(queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                    [](wgpu::QueueWorkDoneStatus,wgpu::StringView) {}), UINT64_MAX);
                instance.ProcessEvents();
            }
            if (tiled.capacity < 24) failures++;
        }
        std::vector<uint32_t> order(data.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            float za = (vp * Eigen::Vector4f(data[a].position.x(), data[a].position.y(), data[a].position.z(), 1)).w();
            float zb = (vp * Eigen::Vector4f(data[b].position.x(), data[b].position.y(), data[b].position.z(), 1)).w();
            return za == zb ? a < b : za > zb;
        });
        auto gpuData = toGaussians(data);
        auto source = buffer(std::max(uint64_t(sizeof(Gaussian)), uint64_t(gpuData.size()*sizeof(Gaussian))),
                             wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        auto indices = buffer(order.size()*4, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        auto camera = buffer(144, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        queue.WriteBuffer(source, 0, gpuData.data(), gpuData.size()*sizeof(Gaussian));
        queue.WriteBuffer(indices, 0, order.data(), order.size()*4);
        queue.WriteBuffer(camera, 0, vp.data(), 64);
        Eigen::Vector4f viewport {float(w), float(h),projection(0,0),projection(1,1)};
        queue.WriteBuffer(camera, 64, view.data(), 64);
        queue.WriteBuffer(camera, 128, viewport.data(), 16);
        wgpu::BindGroupEntry entries[] {{.binding=0,.buffer=source,.size=source.GetSize()},
            {.binding=1,.buffer=camera,.size=camera.GetSize()},{.binding=2,.buffer=indices,.size=indices.GetSize()}};
        wgpu::BindGroupDescriptor gd {.layout=baseline.GetBindGroupLayout(0),.entryCount=3,.entries=entries};
        auto group=device.CreateBindGroup(&gd);
        wgpu::TextureDescriptor td {.usage=wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
            .size={w,h,1},.format=target.format};
        auto reference=device.CreateTexture(&td);
        td.format=wgpu::TextureFormat::RGBA8Unorm;
        auto quadImage=device.CreateTexture(&td);
        td.format=wgpu::TextureFormat::RGBA16Float;
        gd.layout=quadPipeline.GetBindGroupLayout(0);
        auto quadGroup=device.CreateBindGroup(&gd);
        uint32_t row=((w*8+255)/256)*256;
        auto expected=buffer(uint64_t(row)*h,wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
        auto actual=buffer(uint64_t(row)*h,wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
        auto counts=buffer(16,wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
        auto records=buffer(tiled.pairs.GetSize(),wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
        auto encoder=device.CreateCommandEncoder();
        tiled.record(encoder,vp,w,h);
        wgpu::RenderPassColorAttachment color {.view=reference.CreateView(),.loadOp=wgpu::LoadOp::Clear,
            .storeOp=wgpu::StoreOp::Store,.clearValue={0.05,0.05,0.07,1}};
        wgpu::RenderPassDescriptor pd {.colorAttachmentCount=1,.colorAttachments=&color};
        auto pass=encoder.BeginRenderPass(&pd);
        pass.SetPipeline(baseline); pass.SetBindGroup(0,group); pass.Draw(4,data.size()); pass.End();
        wgpu::TexelCopyTextureInfo ti {.texture=reference};
        wgpu::TexelCopyBufferInfo bi {.layout={.bytesPerRow=row,.rowsPerImage=h},.buffer=expected};
        encoder.CopyTextureToBuffer(&ti,&bi,&td.size);
        ti.texture=tiled.image; bi.buffer=actual;
        encoder.CopyTextureToBuffer(&ti,&bi,&td.size);
        encoder.CopyBufferToBuffer(tiled.counters,0,counts,0,16);
        encoder.CopyBufferToBuffer(tiled.pairs,0,records,0,tiled.pairs.GetSize());
        auto commands=encoder.Finish(); queue.Submit(1,&commands);
        map(expected); map(actual); map(counts); map(records);
        auto counters=static_cast<const uint32_t*>(counts.GetConstMappedRange());
        auto pairs=static_cast<const uint32_t*>(records.GetConstMappedRange());
        uint32_t badOrder=0;
        for (uint32_t i=0;i<counters[1];i++) {
            auto key=[&](uint32_t n) {return std::tuple(pairs[n*4+1],pairs[n*4],pairs[n*4+3]);};
            if (i && key(i)<key(i-1)) badOrder++;
            if (pairs[i*4+2]>=counters[0]) badOrder++;
        }
        auto a=static_cast<const uint8_t*>(actual.GetConstMappedRange());
        auto e=static_cast<const uint8_t*>(expected.GetConstMappedRange());
        float maxError=0; uint32_t mismatches=0;
        for (uint32_t y=0;y<h;y++) for (uint32_t x=0;x<w;x++) for (uint32_t c=0;c<3;c++) {
            Eigen::half half;
            std::memcpy(&half,e+y*row+x*8+c*2,2);
            float error=std::abs(float(half)-a[y*row+x*4+c]/255.0f);
            maxError=std::max(error,maxError);
            if (error>4.0f/255) mismatches++;
        }
        std::cout<<"case "<<test<<": visible="<<counters[0]<<" pairs="<<counters[1]
            <<" overflow="<<counters[2]<<" bad-order="<<badOrder<<" max-error="<<maxError
            <<" mismatched-channels="<<mismatches<<'\n';
        failures += counters[2]!=0 || badOrder!=0 || mismatches!=0;
        expected.Unmap(); actual.Unmap(); counts.Unmap(); records.Unmap();
        color.view=quadImage.CreateView();
        if (test >= 7 && test <= 9) for (bool useTiles : {false, true}) {
            std::vector<double> timings;
            std::vector<double> gpuTimings, recordTimings, finishTimings;
            wgpu::QuerySetDescriptor qd {.type = wgpu::QueryType::Timestamp, .count = 2};
            auto queries = device.CreateQuerySet(&qd);
            auto resolve = buffer(16, wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc);
            auto times = buffer(16, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
            tiled.timestampQueries = useTiles ? queries : nullptr;
            for (int iteration=0; iteration<9; iteration++) {
                auto start=std::chrono::steady_clock::now();
                auto e=device.CreateCommandEncoder();
                if (useTiles) tiled.record(e,vp,w,h);
                else {
                    wgpu::PassTimestampWrites writes {.querySet = queries, .beginningOfPassWriteIndex = 0,
                                                       .endOfPassWriteIndex = 1};
                    pd.timestampWrites = &writes;
                    auto p=e.BeginRenderPass(&pd);
                    p.SetPipeline(quadPipeline); p.SetBindGroup(0,quadGroup); p.Draw(4,data.size()); p.End();
                    pd.timestampWrites = nullptr;
                }
                auto recorded=std::chrono::steady_clock::now();
                e.ResolveQuerySet(queries, 0, 2, resolve, 0);
                e.CopyBufferToBuffer(resolve, 0, times, 0, 16);
                auto commands=e.Finish();
                auto finished=std::chrono::steady_clock::now();
                queue.Submit(1,&commands);
                if (iteration>=2) {
                    recordTimings.push_back(std::chrono::duration<double,std::milli>(recorded-start).count());
                    finishTimings.push_back(std::chrono::duration<double,std::milli>(finished-recorded).count());
                }
                instance.WaitAny(queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                    [](wgpu::QueueWorkDoneStatus,wgpu::StringView) {}), UINT64_MAX);
                if (iteration>=2) timings.push_back(std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-start).count());
                map(times);
                auto ticks=static_cast<const uint64_t*>(times.GetConstMappedRange());
                if (iteration>=2) gpuTimings.push_back((ticks[1]-ticks[0])*1e-6);
                times.Unmap();
                instance.ProcessEvents();
            }
            std::sort(timings.begin(),timings.end());
            std::sort(gpuTimings.begin(),gpuTimings.end());
            std::sort(recordTimings.begin(),recordTimings.end());
            std::sort(finishTimings.begin(),finishTimings.end());
            std::cout<<"completed-frame case="<<test<<" renderer="<<(useTiles?"tiles":"quad")
                <<" median-ms="<<timings[timings.size()/2]<<" gpu-ms="<<gpuTimings[gpuTimings.size()/2]
                <<" record-ms="<<recordTimings[recordTimings.size()/2]<<" finish-ms="<<finishTimings[finishTimings.size()/2]<<'\n';
            tiled.timestampQueries = nullptr;
        }
        instance.ProcessEvents();
    }
    instance.WaitAny(device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::PopErrorScopeStatus, wgpu::ErrorType type, wgpu::StringView message) {
            if (type != wgpu::ErrorType::NoError) {
                std::cerr << std::string_view(message.data, message.length) << '\n';
                failures++;
            }
        }), UINT64_MAX);
    return failures != 0;
}
