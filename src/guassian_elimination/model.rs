use bytemuck::{Pod, Zeroable};
use futures::channel::oneshot;
use rand::Rng;
use std::borrow::Cow;
use wgpu::{
    BufferAsyncError, BufferSize, BufferSlice, CommandEncoderDescriptor, ComputePipeline,
    UncapturedErrorHandler, core::device::DeviceError, naga::compact::compact, util::DeviceExt,
};

use crate::wgpu_helpers::{self, helpers::create_buffer};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PushConstants {
    pivot: u32,
    matrix_size: u32,
}

const MATRIX_SIZE: usize = 4;
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AugmentedMatrix {
    data: [[f32; MATRIX_SIZE + 1]; MATRIX_SIZE], // Augmented matrix [A|B]
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PushConstantsV2 {
    pivot_row: u32,
    pivot_col: u32,
}

async fn gaussian_elimination_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    initial_matrix: AugmentedMatrix,
) -> Vec<f32> {
    let matrix_size = MATRIX_SIZE as u64;
    let augmented_size = matrix_size + 1;

    let num_elements = MATRIX_SIZE * (augmented_size as usize);

    let matrix_buffer = wgpu_helpers::helpers::create_buffer(
        device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        &[initial_matrix],
    );

    let solution_buffer = create_buffer(
        device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        &[0.0; MATRIX_SIZE],
    );

    let (bind_group_layout, bind_group) =
        wgpu_helpers::helpers::create_bind_group(device, &matrix_buffer, &solution_buffer);

    let elimination_shader_str_formatted = include_str!("elimination.wgsl")
        .replace("{{MATRIX_SIZE}}", &MATRIX_SIZE.to_string())
        .replace("{{MATRIX_SIZE + 1}}", &(MATRIX_SIZE + 1).to_string())
        .replace("{{WORKGROUP_SIZE}}", &WORKGROUP_SIZE.to_string());

    let back_substitution_shader_str_formatted = include_str!("back_substitution.wgsl")
        .replace("{{MATRIX_SIZE}}", &MATRIX_SIZE.to_string())
        .replace("{{MATRIX_SIZE + 1}}", &(MATRIX_SIZE + 1).to_string())
        .replace("{{WORKGROUP_SIZE}}", &WORKGROUP_SIZE.to_string());

    let elimination_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("elimination_shader"),
        source: wgpu::ShaderSource::Wgsl(elimination_shader_str_formatted.into()),
    });

    let back_substitution_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("back_substitution_shader"),
        source: wgpu::ShaderSource::Wgsl(back_substitution_shader_str_formatted.into()),
    });

    let elimination_pipeline = wgpu_helpers::helpers::create_compute_pipelin(
        device,
        &bind_group_layout,
        &elimination_shader,
        "main",
    );

    let backback_substitution_pipeline = wgpu_helpers::helpers::create_compute_pipelin(
        device,
        &bind_group_layout,
        &back_substitution_shader,
        "main",
    );

    let num_workgroups = (MATRIX_SIZE as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    for pivot_row in 0..MATRIX_SIZE {
        let mut encoder = device.create_command_encoder(&Default::default());

        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&elimination_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.set_push_constants(
                0,
                bytemuck::bytes_of(&PushConstantsV2 {
                    pivot_row: pivot_row as u32,
                    pivot_col: pivot_row as u32, // Assuming we are eliminating based on the current pivot rows column
                }),
            );

            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&backback_substitution_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let solution_read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Solution Read Buffer"),
        size: (MATRIX_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());

    encoder.copy_buffer_to_buffer(
        &solution_buffer,
        0,
        &solution_read_buffer,
        0,
        solution_read_buffer.size(),
    );
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = solution_read_buffer.slice(..);

    let (tx, rx) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    device.poll(wgpu::MaintainBase::Wait);

    rx.await.unwrap().unwrap();
    let result: Vec<f32>;
    {
        let data = buffer_slice.get_mapped_range();
        result = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
    }

    solution_read_buffer.unmap();
    result
}

pub async fn run() {
    let (instance, adapter, device, queue) = wgpu_helpers::helpers::init_wgpu().await;

    let mut rng = rand::rng();
    let mut initial_data = [[0.0; MATRIX_SIZE + 1]; MATRIX_SIZE];

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            initial_data[i][j] = rng.random_range(-5.0..5.0);
        }
        initial_data[i][MATRIX_SIZE] = rng.random_range(-10.0..10.0);
    }

    let initial_matrix = AugmentedMatrix { data: initial_data };

    println!("Initial Augmented Matrix: ");
    for row in &initial_matrix.data {
        println!("{:?}", row);
    }

    let solution = gaussian_elimination_gpu(&device, &queue, initial_matrix).await;

    println!("\nSolution:");
    println!("{:?}", solution);
}
