mod guassian_elimination;
mod wgpu_helpers;

fn main() {
    env_logger::init();
    futures::executor::block_on(guassian_elimination::model::run())
}
