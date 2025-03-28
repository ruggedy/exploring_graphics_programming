struct AugmentedMatrix {
    data: array<array<f32, {{MATRIX_SIZE + 1}}>, {{MATRIX_SIZE}}>,
}
;

@group(0) @binding(0)
var<storage, read_write> matrix: AugmentedMatrix;


struct PushConstants {
    pivot_row: u32,
    pivot_col: u32,
}

var<push_constant> push_constants: PushConstants;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_index = global_id.x;
    let n = {{MATRIX_SIZE}}u;

    //Only process rows below tho pivot row
    if (row_index > push_constants.pivot_row && row_index < n) {
        let pivot_value = matrix.data[push_constants.pivot_row][push_constants.pivot_col];

        if (pivot_value == 0.0) {
            // Handle zero pivot (for simplicity this would be done later)
            return;
        }

        let factor = matrix.data[row_index][push_constants.pivot_col] / pivot_value;

        // Parallelize the substraction across all columns in the row
        for (var col_index: u32 = push_constants.pivot_col; col_index < n + 1; col_index++) {
            matrix.data[row_index][col_index] = matrix.data[row_index][col_index] - factor * matrix.data[push_constants.pivot_row][col_index];
        }

    }
}
