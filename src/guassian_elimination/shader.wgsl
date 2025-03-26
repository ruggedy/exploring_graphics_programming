struct PushConstants {
    pivot_row: u32,
    matrix_size: u32,
}
;

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;

var<push_constant> pc: PushConstants;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let k = pc.pivot_row;
    let n = pc.matrix_size;

    // Skip pivot row and processed rows
    if (row <= k || row >= n) {
        return;
    }

    // Get pivot element
    let pivot = a[k * n + k];
    if (abs(pivot) < 0.0000001) {
        return; // Avoid division by near-zero
    }

    // Calculate factor
    let factor = a[row * n + k] / pivot;

    // Update b vector
    b[row] = b[row] - factor * b[k];

    // Update matrix row
    for (var j: u32 = k; j < n; j = j + 1u) {
        a[row * n + j] = a[row * n + j] - factor * a[k * n + j];
    }
}
