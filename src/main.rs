mod guassian_elimination;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    futures::executor::block_on(guassian_elimination::model::run())
}
