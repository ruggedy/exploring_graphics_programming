struct AugmentedMatrix {
    data: array<array<f32, {{MATRIX_SIZE + 1}}>, {{MATRIX_SIZE}}>, //
};

@group(0) @binding(0)
var<storage, read_write> matrix: AugmentedMatrix;

struct Solution {
    data: array<f32, {{MATRIX_SIZE}}>,//
}
;

@group(0) @binding(1)
var<storage, read_write> solution: Solution;

@compute @workgroup_size({{WORKGROUP_SIZE}})//
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_index = global_id.x;
    let n ={{MATRIX_SIZE}}u; //

    if (row_index < n) {
        let i = n - 1 - row_index;
        var sum = 0.0;
        for (var j: u32 = i + 1; j < n; j++) {
            sum = sum + matrix.data[i][j] * solution.data[j];
        }
        solution.data[i] = (matrix.data[i][n] - sum) / matrix.data[i][i];
    }
}
