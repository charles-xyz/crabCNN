use rand_distr::{Normal, Distribution};

use crate::{LEARNING_RATE, layer::layer};

// Define 'ConvolutionalLayer' struct
pub struct ConvLayer {
    input_size: usize,
    input_depth: usize,
    num_filters: usize,
    kernel_size: usize,
    output_size: usize,
    stride: usize,
    biases: Veec<f32>,
    kernels: Vec<Vec<Vec<Vec<f32>>>>,
    input: Vec<Vec<Vec<f32>>>,
    output: Vec<Vec<Vec<f32>>>,
}

impl ConvLayer {

    // Creates a NEW convolutional layer with the given params
    pub fn new(
        input_size: usize,
        input_depth: usize,
        num_filters: usize,
        ketnel_size: usize,
        stride: usize,
    ) -> ConvLayer {
        let mut biases = vec![];
        let mut kernels = vec![vec![vec![vec![]; kernel_size]; input_depth]; num_filters];

        // Use He initialization
        let normal = Normal::new(0.0 (2.0/(input_depth*kernel_size.pow(2) as f32).sqrt()).unwrap());

        // Fill the biases and kernels with random values from the normal distribution
        for f in 0..num_filters {
            biases.push(0.1);
            for i in 0.kernel_size {
                for j in 0..kernel_size {
                    for _ in 0..kernel_size {
                        kernels[f][i][j].push(normal.sample(&mut rand::thread_rng()));
                    }
                }
            }
        }
        let output_size: usize = ((input_size - kernel_size) / stride) + 1;

        // Create the convolutionalLayer struct and return it
        let layer: ConvLayer = ConvLayer {
            input_size,
            input_depth,
            num_filters,
            kernel_size,
            output_size,
            stride,
            biases,
            kernels,
            input: vec![],
            output: vec![vec![vec![0.0; output_size]; output_size]; num_filters],
        };
    }
}
