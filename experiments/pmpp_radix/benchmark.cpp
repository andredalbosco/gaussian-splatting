#include "camera.hpp"
#include "gaussian.hpp"
#include "radix_sort.hpp"
#include "wgpu_utils.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

struct CameraData { Eigen::Matrix4f vp, view; Eigen::Vector4f viewport; };
struct Key { uint32_t depth, index; };

int main(int argc, char** argv) {
    auto instance = createInstance();
    auto adapter = requestAdapter(instance, {});
    wgpu::AdapterInfo info; adapter.GetInfo(&info);
    std::cout << "GPU=" << std::string_view(info.device.data, info.device.length)
              << " backend=" << int(info.backendType) << std::endl;
    wgpu::FeatureName feature = wgpu::FeatureName::TimestampQuery;
    const char* disabled[] {"timestamp_quantization"};
    wgpu::DawnTogglesDescriptor toggles;
    toggles.disabledToggleCount = 1; toggles.disabledToggles = disabled;
    wgpu::Limits limits {};
    limits.maxStorageBufferBindingSize = 320 * 1024 * 1024;
    limits.maxBufferSize = 320 * 1024 * 1024;
    wgpu::DeviceDescriptor desc {};
    desc.requiredLimits = &limits;
    desc.nextInChain = &toggles; desc.requiredFeatureCount = 1; desc.requiredFeatures = &feature;
    desc.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
        std::cerr << std::string_view(message.data, message.length) << '\n'; std::exit(2);
    });
    wgpu::Device device;
    instance.WaitAny(adapter.RequestDevice(&desc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus, wgpu::Device result, wgpu::StringView) { device = result; }), UINT64_MAX);
    auto queue = device.GetQueue();

    auto buffer = [&](uint64_t bytes, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor d {.usage = usage, .size = std::max(uint64_t(64), bytes)};
        return device.CreateBuffer(&d);
    };
    auto submit = [&](const wgpu::CommandEncoder& encoder) {
        auto commands = encoder.Finish(); queue.Submit(1, &commands);
    };
    auto read = [&](const wgpu::Buffer& source, uint64_t bytes) {
        auto staging = buffer(bytes, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
        auto encoder = device.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(source, 0, staging, 0, bytes); submit(encoder);
        instance.WaitAny(staging.MapAsync(wgpu::MapMode::Read, 0, bytes, wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::MapAsyncStatus status, wgpu::StringView) { if (status != wgpu::MapAsyncStatus::Success) std::exit(2); }), UINT64_MAX);
        std::vector<uint8_t> result(bytes);
        std::memcpy(result.data(), staging.GetConstMappedRange(0, bytes), bytes); staging.Unmap();
        return result;
    };
    std::vector<std::string> names {"masks", "pmpp_local"};
    std::vector<RadixSort> sorters(2);
    sorters[0].init(device, readWgslFile(device, SRC_DIR "/shaders.wgsl"));
    sorters[1].init(device, readWgslFile(device, EXPERIMENT_SHADER));
    CameraData camera {Eigen::Matrix4f::Identity(), Eigen::Matrix4f::Identity(), {1920,1080,1,1}};
    auto cameraBuffer = buffer(sizeof(camera), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    std::ofstream csv(argc > 1 ? argv[1] : "pmpp_timings.csv");
    std::ofstream rawCsv(std::string(argc > 1 ? argv[1] : "pmpp_timings.csv") + ".samples.csv");
    rawCsv << "scene,count,view,trial,method,sample,ms_per_sort\n";
    csv << "scene,count,view,trial,method,median_ms,p90_ms,errors\n";
    for (auto scene : {"luigi", "black_jacket"}) {
        auto data = readPlyFile((std::string(ASSETS_DIR) + "/" + scene + ".ply").c_str());
        if (argc > 2) {
            const auto source = std::move(data);
            data.resize(std::strtoul(argv[2], nullptr, 10));
            for (size_t i = 0; i < data.size(); i++) data[i] = source[i % source.size()];
        }
        const uint32_t count = data.size();
        auto input = buffer(uint64_t(count)*sizeof(Gaussian), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        queue.WriteBuffer(input, 0, data.data(), uint64_t(count)*sizeof(Gaussian));
        std::vector<wgpu::Buffer> outputs;
        for (int m=0;m<2;m++) {
            auto& sorter=sorters[m];
            sorter.createBuffers(count);
            if (m==1) sorter.radixLocalRanksBuffer=buffer(uint64_t(count)*12, wgpu::BufferUsage::Storage);
            outputs.push_back(buffer(uint64_t(count)*4, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
            sorter.createBindGroups(input,cameraBuffer,outputs.back());
        }
        for (int view=0;view<3;view++) {
            camera.view=Eigen::Matrix4f::Identity();
            if(view) camera.view.block<3,3>(0,0)=(Eigen::AngleAxisf(0.4f*view,Eigen::Vector3f::UnitY())*
                Eigen::AngleAxisf(0.7f*view,Eigen::Vector3f::UnitX())).toRotationMatrix();
            queue.WriteBuffer(cameraBuffer,0,&camera,sizeof(camera));
            auto encoder=device.CreateCommandEncoder();
            auto pass=encoder.BeginComputePass();
            auto& base=sorters[0];
            pass.SetPipeline(base.radixHistogramRankPipeline); pass.SetBindGroup(0,base.radixHistogramBindGroup);
            pass.SetBindGroup(1,base.radixHistogramBindGroups[0]); pass.DispatchWorkgroups(base.radixGroupCount); pass.End(); submit(encoder);
            auto raw=read(base.radixKeysBuffer[0],uint64_t(count)*8);
            const auto* keys=reinterpret_cast<const Key*>(raw.data());
            std::vector<uint32_t> expected(count); std::iota(expected.begin(),expected.end(),0u);
            std::sort(expected.begin(),expected.end(),[&](uint32_t a,uint32_t b){
                return keys[a].depth==keys[b].depth ? a<b : keys[a].depth<keys[b].depth;
            });
            for(int m=0;m<2;m++) {
                encoder=device.CreateCommandEncoder(); sorters[m].recordCommands(encoder); submit(encoder);
                auto rawOrder=read(outputs[m],uint64_t(count)*4);
                const auto* order=reinterpret_cast<const uint32_t*>(rawOrder.data());
                uint32_t errors=0; for(uint32_t i=0;i<count;i++) errors+=order[i]!=expected[i];
                std::cout<<scene<<" count="<<count<<" view="<<view<<" "<<names[m]<<" errors="<<errors<<std::endl;
                if(errors) return 1;
            }
            constexpr uint32_t samples=64, batch=4;
            wgpu::QuerySetDescriptor qd {.type=wgpu::QueryType::Timestamp,.count=samples*2};
            auto queries=device.CreateQuerySet(&qd);
            auto resolve=buffer(samples*16,wgpu::BufferUsage::QueryResolve|wgpu::BufferUsage::CopySrc);
            for(int trial=0;trial<5;trial++) for(int j=0;j<2;j++) {
                int m=(j+trial)%2;
                encoder=device.CreateCommandEncoder();
                for(int warm=0;warm<8;warm++) sorters[m].recordCommands(encoder);
                submit(encoder);
                encoder=device.CreateCommandEncoder();
                for(uint32_t sample=0;sample<samples;sample++) {
                    for(uint32_t repeat=0;repeat<batch;repeat++) {
                        wgpu::PassTimestampWrites writes {.querySet=queries,
                            .beginningOfPassWriteIndex=repeat==0 ? sample*2 : wgpu::kQuerySetIndexUndefined,
                            .endOfPassWriteIndex=repeat+1==batch ? sample*2+1 : wgpu::kQuerySetIndexUndefined};
                        sorters[m].recordCommands(encoder,(repeat==0 || repeat+1==batch) ? &writes : nullptr);
                    }
                }
                encoder.ResolveQuerySet(queries,0,samples*2,resolve,0); submit(encoder);
                auto rawTime=read(resolve,samples*16);
                auto* time=reinterpret_cast<const uint64_t*>(rawTime.data());
                std::vector<double> ms(samples);
                for(uint32_t i=0;i<samples;i++) {
                    ms[i]=(time[i*2+1]-time[i*2])/(1e6*batch);
                    rawCsv<<scene<<','<<count<<','<<view<<','<<trial<<','<<names[m]<<','<<i<<','<<ms[i]<<'\n';
                }
                std::sort(ms.begin(),ms.end());
                csv<<scene<<','<<count<<','<<view<<','<<trial<<','<<names[m]<<','<<ms[samples/2]<<','<<ms[samples*9/10]<<",0\n";
                csv.flush();
                std::cout<<"trial="<<trial<<" "<<names[m]<<" ms="<<ms[samples/2]<<std::endl;
            }
        }
    }
}
