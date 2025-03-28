use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer,
    BufferUsages, ComputePipeline, Device, PushConstantRange, ShaderModule, ShaderStages,
    util::DeviceExt,
};

pub async fn init_wgpu() -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 8,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .unwrap();

    (instance, adapter, device, queue)
}

pub fn create_buffer<T: Pod + Zeroable>(
    device: &Device,
    usage: BufferUsages,
    data: &[T],
) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

pub fn create_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                count: None,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                count: None,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
        ],
    })
}

/// matrix_buffer binding is located in position 0
/// solution binding is located in position 1,
pub fn create_bind_group(
    device: &Device,
    matrix_buffer: &Buffer,
    solution_buffer: &Buffer,
) -> (BindGroupLayout, BindGroup) {
    let layout = create_bind_group_layout(device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: matrix_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: solution_buffer.as_entire_binding(),
            },
        ],
    });

    (layout, bind_group)
}

pub fn create_compute_pipelin(
    device: &Device,
    layout: &BindGroupLayout,
    shader_module: &ShaderModule,
    entry_point: &str,
) -> ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[layout],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..8,
                }],
            }),
        ),
        module: shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}
