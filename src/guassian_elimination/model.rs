use bytemuck::{Pod, Zeroable};
use futures::channel::oneshot;
use std::borrow::Cow;
use wgpu::{
    BufferAsyncError, BufferSize, BufferSlice, CommandEncoderDescriptor, ComputePipeline,
    util::DeviceExt,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PushConstants {
    pivot: u32,
    matrix_size: u32,
    // _padding: [u32; 2],
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("New Device"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 8,
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
        )
        .await?;

    let n = 1024;

    let mut a = vec![0.0f32; n * n];
    let mut b = vec![0.0f32; n];

    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = ((i + j) % 10 + 1) as f32;
        }

        b[i] = i as f32;
    }

    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix A buffer"),
        contents: bytemuck::cast_slice(&a),
        usage: wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix B Buffer"),
        contents: bytemuck::cast_slice(&b),
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Results Buffer"),
        size: (n * n * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let result_vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Vector Buffer"),
        mapped_at_creation: false,
        size: (n * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader to compute gaussian elimination"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                count: None,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                count: None,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..8,
        }],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gaussian Elimination Pipelin"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
        ],
    });

    for k in 0..n - 1 {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Guassian Elimination Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Push constants for current pivot
        let push_constants = PushConstants {
            pivot: k as u32,
            matrix_size: n as u32,
            // _padding: [0, 0],
        };

        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[push_constants]));

        let workgroup_count = (n + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);

        drop(compute_pass);

        queue.submit(Some(encoder.finish()));
    }

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });

    encoder.copy_buffer_to_buffer(
        &a_buffer,
        0,
        &result_buffer,
        0,
        (n * n * std::mem::size_of::<f32>()) as u64,
    );
    encoder.copy_buffer_to_buffer(
        &b_buffer,
        0,
        &result_vector_buffer,
        0,
        (n * std::mem::size_of::<f32>()) as u64,
    );

    queue.submit(Some(encoder.finish()));

    //Map buffers to read results
    let a_buffer_slice = result_buffer.slice(..);

    let b_buffer_slice = result_vector_buffer.slice(..);

    let a_buffer_future = map_buffer_slice(a_buffer_slice, None);
    let b_buffer_future = map_buffer_slice(b_buffer_slice, None);

    if let (Ok(()), Ok(())) = (a_buffer_future.await, b_buffer_future.await) {
        let a_data = a_buffer_slice.get_mapped_range();
        let b_data = b_buffer_slice.get_mapped_range();

        let result_a: Vec<f32> = bytemuck::cast_slice(&a_data).to_vec();
        let result_b: Vec<f32> = bytemuck::cast_slice(&b_data).to_vec();

        let mut x = vec![0.0f32; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += result_a[i * n + j] * x[j];
            }

            x[i] = (result_b[i] - sum) / result_a[i * n + 1];
        }

        println!(
            "Solution vector (first 10 elements): {:?}",
            &x[0..std::cmp::min(10, n)]
        );
    }

    Ok(())
}

async fn map_buffer_slice(
    slice: BufferSlice<'_>,
    map_mode: Option<wgpu::MapMode>,
) -> Result<(), BufferAsyncError> {
    let (tx, rx) = oneshot::channel();

    let map_mode = map_mode.unwrap_or(wgpu::MapMode::Read);
    slice.map_async(map_mode, move |result| {
        tx.send(result);
    });

    rx.await.unwrap()
}
