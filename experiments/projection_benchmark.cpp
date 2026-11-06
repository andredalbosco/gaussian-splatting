#include "camera.hpp"
#include "colored_gaussian.hpp"
#include "wgpu_utils.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string>

constexpr uint32_t samples = 32;
constexpr uint32_t trials = 3;
constexpr uint32_t width = 1920, height = 1080;
constexpr std::array names {
    "baseline_axes", "shared_rows_matrix", "shared_rows_quaternion", "sparse_camera_axes",
    "gsplat_constructed", "gsplat_precomputed", "playcanvas_constructed", "playcanvas_precomputed",
    "symmetric_constructed", "symmetric_precomputed", "sparse_covariance_precomputed",
};
struct CameraData {
    Eigen::Matrix4f vp, view;
    Eigen::Vector4f viewportFocal, limitsCount;
};
struct Projected {
    Eigen::Vector4f clip, covariance, color;
};
struct Variant { wgpu::ComputePipeline compute; wgpu::RenderPipeline render, math; };
struct Workload { const char* name; uint32_t count; float radius; bool render; bool math = false; };
struct Resources {
    wgpu::Buffer source, packed, output, camera;
    wgpu::BindGroup group, packedGroup, renderGroup, renderPackedGroup, preparedGroup;
    std::vector<ColoredGaussian> gaussians;
    CameraData cameraData;
};

wgpu::Buffer buffer(const wgpu::Device& device, uint64_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor desc {.usage = usage, .size = size};
    return device.CreateBuffer(&desc);
}
void submit(const wgpu::Device& device, const wgpu::CommandEncoder& encoder) {
    auto commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
}
std::vector<uint8_t> readBuffer(const wgpu::Instance& instance, const wgpu::Device& device,
                               const wgpu::Buffer& source, uint64_t size) {
    auto readback = buffer(device,size,wgpu::BufferUsage::CopyDst|wgpu::BufferUsage::MapRead);
    auto encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(source,0,readback,0,size);
    submit(device,encoder);
    instance.WaitAny(readback.MapAsync(wgpu::MapMode::Read,0,size,wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus status,wgpu::StringView) { if (status != wgpu::MapAsyncStatus::Success) std::exit(2); }),UINT64_MAX);
    auto data = static_cast<const uint8_t*>(readback.GetConstMappedRange());
    std::vector<uint8_t> result(data,data+size);
    readback.Unmap();
    return result;
}
Resources makeResources(const wgpu::Device& device, const wgpu::BindGroupLayout& layout,
                        const wgpu::BindGroupLayout& renderLayout, const wgpu::BindGroupLayout& preparedLayout, Workload work, bool offAxis = false) {
    Resources r;
    auto& cam = r.cameraData;
    cam.view = Eigen::Matrix4f::Identity();
    cam.view.block<3,3>(0,0) = (Eigen::AngleAxisf(0.47f,Eigen::Vector3f::UnitY())*
                              Eigen::AngleAxisf(-0.31f,Eigen::Vector3f::UnitX())).toRotationMatrix();
    cam.view.block<3,1>(0,3) = Eigen::Vector3f(0.2f,-0.4f,0.7f);
    auto projection = projectionMatrix(60,float(width)/height,0.1f,100.f);
    cam.vp = projection*cam.view;
    float fx = projection(0,0)*width*0.5f, fy = projection(1,1)*height*0.5f;
    cam.viewportFocal = {float(width),float(height),fx,fy};
    cam.limitsCount = {1.3f*width*0.5f/fx,1.3f*height*0.5f/fy,float(work.count),0};
    std::mt19937 rng(20261019);
    std::uniform_real_distribution<float> uniform(0,1), normal(-1,1);
    r.gaussians.resize(work.count);
    std::vector<ColoredGaussian> packed(work.count);
    Eigen::Matrix3f W = cam.view.block<3,3>(0,0);
    Eigen::Vector3f translation = cam.view.block<3,1>(0,3);
    for (auto& g : r.gaussians) {
        float z = 2+18*uniform(rng);
        float spread = offAxis ? 2.2f : 0.85f;
        Eigen::Vector3f p(normal(rng)*spread*width*0.5f/fx*z,
                          normal(rng)*spread*height*0.5f/fy*z,z);
        g.position.head<3>() = W.transpose()*(p-translation);
        g.position.w() = 1;
        g.scale = {work.radius*(0.2f+uniform(rng))*z/fx,
                   work.radius*(0.2f+uniform(rng))*z/fy,
                   work.radius*(0.2f+uniform(rng))*z/fx,0};
        // Uniform random orientation (Shoemake).
        float u = uniform(rng), a = 6.28318530718f*uniform(rng), b = 6.28318530718f*uniform(rng);
        g.rotation = {std::sqrt(1-u)*std::sin(a),std::sqrt(1-u)*std::cos(a),std::sqrt(u)*std::sin(b),std::sqrt(u)*std::cos(b)};
        g.rotation.normalize();
        g.color = {uniform(rng),uniform(rng),uniform(rng),0.15f+0.7f*uniform(rng)};
    }
    std::sort(r.gaussians.begin(),r.gaussians.end(),[&](const ColoredGaussian& a,const ColoredGaussian& b) {
        return (W*a.position.head<3>()).z() > (W*b.position.head<3>()).z();
    });
    for (size_t i=0;i<packed.size();i++) {
        const auto& g = r.gaussians[i];
        Eigen::Quaternionf q(g.rotation.w(),g.rotation.x(),g.rotation.y(),g.rotation.z());
        Eigen::Matrix3f M = q.toRotationMatrix()*g.scale.head<3>().asDiagonal();
        Eigen::Matrix3f C = M*M.transpose();
        packed[i] = {g.position,{C(0,0),C(0,1),C(0,2),0},{C(1,1),C(1,2),C(2,2),0},g.color};
    }
    auto usage = wgpu::BufferUsage::Storage|wgpu::BufferUsage::CopyDst;
    r.source = buffer(device,work.count*sizeof(ColoredGaussian),usage);
    r.packed = buffer(device,r.source.GetSize(),usage);
    r.output = buffer(device,work.count*sizeof(Projected),usage|wgpu::BufferUsage::CopySrc);
    r.camera = buffer(device,sizeof(CameraData),wgpu::BufferUsage::Uniform|wgpu::BufferUsage::CopyDst);
    auto queue = device.GetQueue();
    queue.WriteBuffer(r.source,0,r.gaussians.data(),r.source.GetSize());
    queue.WriteBuffer(r.packed,0,packed.data(),r.packed.GetSize());
    queue.WriteBuffer(r.camera,0,&cam,sizeof(cam));
    auto makeGroup = [&](const wgpu::Buffer& input, bool render) {
        wgpu::BindGroupEntry entries[] {{.binding=0,.buffer=input,.size=input.GetSize()},
            {.binding=1,.buffer=r.camera,.size=r.camera.GetSize()},
            {.binding=2,.buffer=r.output,.size=r.output.GetSize()}};
        wgpu::BindGroupDescriptor desc {.layout=render?renderLayout:layout,.entryCount=render?2u:3u,.entries=entries};
        return device.CreateBindGroup(&desc);
    };
    r.group = makeGroup(r.source,false); r.packedGroup = makeGroup(r.packed,false);
    r.renderGroup = makeGroup(r.source,true); r.renderPackedGroup = makeGroup(r.packed,true);
    wgpu::BindGroupEntry entry {.binding=0,.buffer=r.output,.size=r.output.GetSize()};
    wgpu::BindGroupDescriptor desc {.layout=preparedLayout,.entryCount=1,.entries=&entry};
    r.preparedGroup = device.CreateBindGroup(&desc);
    return r;
}
bool precomputed(uint32_t method) { return method==5 || method==7 || method==9 || method==10; }
void compute(const wgpu::CommandEncoder& encoder, const Variant& v, const Resources& r, uint32_t method,
             uint32_t count, const wgpu::PassTimestampWrites* writes = nullptr) {
    wgpu::ComputePassDescriptor desc {.timestampWrites=writes};
    auto pass = encoder.BeginComputePass(&desc);
    pass.SetPipeline(v.compute); pass.SetBindGroup(0,precomputed(method)?r.packedGroup:r.group);
    pass.DispatchWorkgroups((count+127)/128); pass.End();
}
std::vector<double> measure(const wgpu::Instance& instance, const wgpu::Device& device,
    const std::function<void(const wgpu::CommandEncoder&,const wgpu::PassTimestampWrites*)>& record) {
    auto encoder = device.CreateCommandEncoder();
    for (int i=0;i<8;i++) record(encoder,nullptr);
    submit(device,encoder);
    instance.WaitAny(device.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus,wgpu::StringView) {}),UINT64_MAX);
    wgpu::QuerySetDescriptor desc {.type=wgpu::QueryType::Timestamp,.count=2*samples};
    auto queries = device.CreateQuerySet(&desc);
    auto resolve = buffer(device,16*samples,wgpu::BufferUsage::QueryResolve|wgpu::BufferUsage::CopySrc);
    encoder = device.CreateCommandEncoder();
    for (uint32_t i=0;i<samples;i++) {
        wgpu::PassTimestampWrites writes {.querySet=queries,.beginningOfPassWriteIndex=2*i,.endOfPassWriteIndex=2*i+1};
        record(encoder,&writes);
    }
    encoder.ResolveQuerySet(queries,0,2*samples,resolve,0);
    submit(device,encoder);
    auto bytes = readBuffer(instance,device,resolve,resolve.GetSize());
    auto values = reinterpret_cast<const uint64_t*>(bytes.data());
    std::vector<double> times(samples);
    for (uint32_t i=0;i<samples;i++) times[i]=(values[2*i+1]-values[2*i])/1000.0;
    return times;
}

int main(int argc,char** argv) {
    std::filesystem::path out = argc>1?argv[1]:"experiments/results/2026-10-19-projection";
    std::filesystem::create_directories(out);
    auto instance = createInstance(); auto adapter = requestAdapter(instance,{});
    wgpu::AdapterInfo info; adapter.GetInfo(&info);
    std::ofstream metadata(out/"adapter.txt");
    metadata << "GPU: " << std::string_view(info.device.data,info.device.length)
             << "\ndriver: " << std::string_view(info.description.data,info.description.length)
             << "\nbackend: " << int(info.backendType) << "\n32 samples x 3 shuffled trials, 8 warmup frames per batch; microseconds\n";
    std::cout << "GPU: " << std::string_view(info.device.data,info.device.length) << std::endl;
    const char* disabled[] {"timestamp_quantization"};
    wgpu::DawnTogglesDescriptor toggles; toggles.disabledToggleCount=1; toggles.disabledToggles=disabled;
    wgpu::FeatureName feature=wgpu::FeatureName::TimestampQuery;
    wgpu::DeviceDescriptor deviceDesc {};
    deviceDesc.nextInChain=&toggles; deviceDesc.requiredFeatureCount=1; deviceDesc.requiredFeatures=&feature;
    deviceDesc.SetUncapturedErrorCallback([](const wgpu::Device&,wgpu::ErrorType,wgpu::StringView message) {
        std::cerr << std::string_view(message.data,message.length) << std::endl; std::exit(2);
    });
    wgpu::Device device;
    auto future = adapter.RequestDevice(&deviceDesc,wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::RequestDeviceStatus,wgpu::Device value,wgpu::StringView,wgpu::Device* result) { *result=std::move(value); },&device);
    instance.WaitAny(future,UINT64_MAX);
    if (!device) return 2;
    auto shader = readWgslFile(device,SRC_DIR "/../experiments/projection_benchmark.wgsl");
    wgpu::BindGroupLayoutEntry entries[] {
        {.binding=0,.visibility=wgpu::ShaderStage::Vertex|wgpu::ShaderStage::Compute,.buffer={.type=wgpu::BufferBindingType::ReadOnlyStorage}},
        {.binding=1,.visibility=wgpu::ShaderStage::Vertex|wgpu::ShaderStage::Compute,.buffer={.type=wgpu::BufferBindingType::Uniform}},
        {.binding=2,.visibility=wgpu::ShaderStage::Compute,.buffer={.type=wgpu::BufferBindingType::Storage}},
    };
    wgpu::BindGroupLayoutDescriptor layoutDesc {.entryCount=3,.entries=entries};
    auto layout = device.CreateBindGroupLayout(&layoutDesc);
    layoutDesc.entryCount=2;
    auto renderLayout=device.CreateBindGroupLayout(&layoutDesc);
    wgpu::PipelineLayoutDescriptor computeLayoutDesc {.bindGroupLayoutCount=1,.bindGroupLayouts=&layout};
    auto computeLayout=device.CreatePipelineLayout(&computeLayoutDesc);
    wgpu::BindGroupLayoutEntry preparedEntry {.binding=0,.visibility=wgpu::ShaderStage::Vertex,.buffer={.type=wgpu::BufferBindingType::ReadOnlyStorage}};
    layoutDesc.entryCount=1; layoutDesc.entries=&preparedEntry;
    auto preparedLayout=device.CreateBindGroupLayout(&layoutDesc);
    wgpu::BindGroupLayout layouts[] {renderLayout,preparedLayout};
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc {.bindGroupLayoutCount=2,.bindGroupLayouts=layouts};
    auto pipelineLayout=device.CreatePipelineLayout(&pipelineLayoutDesc);
    wgpu::BlendState blend {.color={.srcFactor=wgpu::BlendFactor::One,.dstFactor=wgpu::BlendFactor::OneMinusSrcAlpha},
                           .alpha={.srcFactor=wgpu::BlendFactor::One,.dstFactor=wgpu::BlendFactor::OneMinusSrcAlpha}};
    wgpu::ColorTargetState target {.format=wgpu::TextureFormat::RGBA8Unorm,.blend=&blend};
    wgpu::FragmentState fragment {.module=shader,.entryPoint="fs_main",.targetCount=1,.targets=&target};
    wgpu::RenderPipelineDescriptor renderDesc {.layout=pipelineLayout,.vertex={.module=shader,.entryPoint="vs_direct"},
        .primitive={.topology=wgpu::PrimitiveTopology::TriangleStrip},.fragment=&fragment};
    std::vector<Variant> variants(names.size());
    for (uint32_t i=0;i<names.size();i++) {
        wgpu::ConstantEntry constant {.key="method",.value=double(i)};
        wgpu::ComputePipelineDescriptor desc {.layout=computeLayout,
            .compute={.module=shader,.entryPoint="cs_main",.constantCount=1,.constants=&constant}};
        variants[i].compute=device.CreateComputePipeline(&desc);
        renderDesc.vertex.constantCount=1; renderDesc.vertex.constants=&constant;
        variants[i].render=device.CreateRenderPipeline(&renderDesc);
        renderDesc.vertex.entryPoint="vs_math";
        variants[i].math=device.CreateRenderPipeline(&renderDesc);
        renderDesc.vertex.entryPoint="vs_direct";
    }
    renderDesc.vertex.entryPoint="vs_prepared"; renderDesc.vertex.constantCount=0; renderDesc.vertex.constants=nullptr;
    auto preparedPipeline=device.CreateRenderPipeline(&renderDesc);
    renderDesc.vertex.entryPoint="vs_math_prepared";
    auto preparedMathPipeline=device.CreateRenderPipeline(&renderDesc);
    std::ofstream validation(out/"validation.csv");
    validation << "scene,method,max_scaled_error,max_absolute_error,failures\n";
    int failures=0;
    for (bool offAxis : {false,true}) {
        auto r=makeResources(device,layout,renderLayout,preparedLayout,{"validation",10000,6,false},offAxis);
        for (uint32_t method=0;method<names.size();method++) {
            auto encoder=device.CreateCommandEncoder(); compute(encoder,variants[method],r,method,r.gaussians.size()); submit(device,encoder);
            auto bytes=readBuffer(instance,device,r.output,r.output.GetSize());
            auto values=reinterpret_cast<const Projected*>(bytes.data());
            double maxScaled=0,maxAbsolute=0; int bad=0;
            for (uint32_t i=0;i<r.gaussians.size();i++) {
                const auto& g=r.gaussians[i];
                Eigen::Quaterniond q(g.rotation.w(),g.rotation.x(),g.rotation.y(),g.rotation.z());
                Eigen::Matrix3d M=q.toRotationMatrix()*g.scale.head<3>().cast<double>().asDiagonal();
                Eigen::Matrix3d W=r.cameraData.view.block<3,3>(0,0).cast<double>();
                Eigen::Vector3d p=(r.cameraData.view.cast<double>()*g.position.cast<double>()).head<3>();
                Eigen::Vector2d uv=p.head<2>()/p.z();
                if (method==4 || method==5) for (int j=0;j<2;j++) uv[j]=std::clamp(uv[j],-double(r.cameraData.limitsCount[j]),double(r.cameraData.limitsCount[j]));
                Eigen::Vector2d f=r.cameraData.viewportFocal.tail<2>().cast<double>()/p.z();
                Eigen::Matrix<double,2,3> J; J << f.x(),0,-f.x()*uv.x(),0,f.y(),-f.y()*uv.y();
                Eigen::Matrix<double,2,3> B=J*W*M;
                Eigen::Matrix2d C=B*B.transpose()+0.3*Eigen::Matrix2d::Identity();
                double expected[] {C(0,0),C(0,1),C(1,1)};
                for (int j=0;j<3;j++) {
                    double error=std::abs(values[i].covariance[j]-expected[j]);
                    double scaled=error/(1+std::sqrt(C(0,0)*C(1,1)));
                    maxAbsolute=std::max(maxAbsolute,error); maxScaled=std::max(maxScaled,scaled);
                    bad+=!std::isfinite(values[i].covariance[j]) || scaled>2e-5;
                }
            }
            failures+=bad;
            validation << (offAxis?"off_axis":"in_frustum") << ',' << names[method] << ',' << maxScaled << ',' << maxAbsolute << ',' << bad << '\n';
        }
    }
    validation.flush();
    std::cout << "CPU covariance validation failures: " << failures << std::endl;
    if (failures) return 1;
    wgpu::TextureDescriptor textureDesc {.usage=wgpu::TextureUsage::RenderAttachment|wgpu::TextureUsage::CopySrc,
        .size={width,height,1},.format=target.format};
    auto texture=device.CreateTexture(&textureDesc);
    wgpu::RenderPassColorAttachment color {.view=texture.CreateView(),.loadOp=wgpu::LoadOp::Clear,.storeOp=wgpu::StoreOp::Store,.clearValue={0.02,0.03,0.04,1}};
    auto draw=[&](const wgpu::CommandEncoder& encoder,const Resources& r,const wgpu::RenderPipeline& pipeline,
                  uint32_t method,uint32_t count,const wgpu::PassTimestampWrites* writes) {
        wgpu::RenderPassDescriptor desc {.colorAttachmentCount=1,.colorAttachments=&color,.timestampWrites=writes};
        auto pass=encoder.BeginRenderPass(&desc); pass.SetPipeline(pipeline);
        pass.SetBindGroup(0,precomputed(method)?r.renderPackedGroup:r.renderGroup); pass.SetBindGroup(1,r.preparedGroup);
        pass.Draw(4,count); pass.End();
    };
    // Compare complete rendered images, including the prepared-buffer path.
    std::ofstream images(out/"images.csv"); images << "method,path,max_channel_error,channels_over_2\n";
    auto r=makeResources(device,layout,renderLayout,preparedLayout,{"image",10000,8,true});
    auto image=[&](uint32_t method,bool prepared) {
        auto encoder=device.CreateCommandEncoder();
        if (prepared) compute(encoder,variants[method],r,method,r.gaussians.size());
        draw(encoder,r,prepared?preparedPipeline:variants[method].render,method,r.gaussians.size(),nullptr);
        uint32_t rowBytes=(width*4+255)&~255u;
        auto pixels=buffer(device,uint64_t(rowBytes)*height,wgpu::BufferUsage::CopyDst|wgpu::BufferUsage::CopySrc);
        wgpu::TexelCopyTextureInfo source {.texture=texture};
        wgpu::TexelCopyBufferInfo dest {.layout={.bytesPerRow=rowBytes,.rowsPerImage=height},.buffer=pixels};
        encoder.CopyTextureToBuffer(&source,&dest,&textureDesc.size); submit(device,encoder);
        return readBuffer(instance,device,pixels,pixels.GetSize());
    };
    auto reference=image(0,false);
    for (uint32_t method=0;method<names.size();method++) for (bool prepared : {false,true}) {
        auto pixels=image(method,prepared); int maxError=0; size_t bad=0;
        for (size_t i=0;i<pixels.size();i++) {
            int error=std::abs(int(pixels[i])-int(reference[i])); maxError=std::max(maxError,error); bad+=error>2;
        }
        images << names[method] << ',' << (prepared?"prepared":"direct") << ',' << maxError << ',' << bad << '\n';
        failures+=bad!=0;
    }
    images.flush();
    std::cout << "Image validation failures: " << failures << std::endl;
    if (failures) return 1;
    if (argc>2 && std::string(argv[2])=="--validate-only") return 0;
    std::array workloads {Workload{"projection_10k",10000,2,false},Workload{"projection_100k",100000,2,false},
        Workload{"render_10k_small",10000,2,true},
        Workload{"render_10k_large",10000,8,true},Workload{"render_100k_small",100000,2,true},
        Workload{"vertex_math_10k",10000,2,true,true}, Workload{"vertex_math_100k",100000,2,true,true}};
    std::ofstream raw(out/"samples.csv"); raw << "workload,count,method,path,trial,sample,gpu_us\n";
    std::mt19937 shuffle(719);
    for (auto work : workloads) {
        if (argc>2 && std::string(argv[2])=="--math-only" && !work.math) continue;
        std::cout << work.name << std::endl;
        auto resources=makeResources(device,layout,renderLayout,preparedLayout,work);
        std::vector<std::pair<uint32_t,bool>> jobs;
        for (uint32_t i=0;i<names.size();i++) { jobs.push_back({i,false}); if (work.render) jobs.push_back({i,true}); }
        for (uint32_t trial=0;trial<trials;trial++) {
            std::shuffle(jobs.begin(),jobs.end(),shuffle);
            for (auto [method,prepared] : jobs) {
                auto times=measure(instance,device,[&](const wgpu::CommandEncoder& encoder,const wgpu::PassTimestampWrites* writes) {
                    if (!work.render) { compute(encoder,variants[method],resources,method,work.count,writes); return; }
                    if (!prepared) { draw(encoder,resources,work.math?variants[method].math:variants[method].render,method,work.count,writes); return; }
                    wgpu::PassTimestampWrites begin,end;
                    if (writes) {
                        begin={.querySet=writes->querySet,.beginningOfPassWriteIndex=writes->beginningOfPassWriteIndex};
                        end={.querySet=writes->querySet,.endOfPassWriteIndex=writes->endOfPassWriteIndex};
                    }
                    compute(encoder,variants[method],resources,method,work.count,writes?&begin:nullptr);
                    draw(encoder,resources,work.math?preparedMathPipeline:preparedPipeline,method,work.count,writes?&end:nullptr);
                });
                for (size_t sample=0;sample<times.size();sample++) raw << work.name << ',' << work.count << ',' << names[method] << ','
                    << (!work.render?"compute_only":prepared?"compute_render":"vertex_render") << ',' << trial << ',' << sample << ','
                    << std::setprecision(9) << times[sample] << '\n';
            }
            raw.flush(); std::cout << "  trial " << trial+1 << '/' << trials << std::endl;
        }
    }
    return 0;
}
