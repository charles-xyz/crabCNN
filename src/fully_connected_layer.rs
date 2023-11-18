use rand_distr::{Distribution, Normal};

use crate::{LEARNING_RATE, layer::Layer};

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct FullyConnectedLayer {
    input_size: usize,
    input_width: usize,
    input_depth: usize,
    output_size: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    input: Vec<f32>,
    pub output: Vec<f32>
}

impl FullyConnectedLayer {
    // Create a fully cnnected layer
    pub fn new(
        input_width: usize,
        input_depth: usize,
        output_size: usize
    ) -> FullyConnectedLayer {
        // calculate the input size from the input width and depth
        let input_size: usize = input_depth * (input_width * input_width);
        
        // Initialize empty vectors for the biases and weights
        let mut biases: Vec<f32> = vec![];
        let mut weights: Vec<Vec<f32>> = vec![vec![], input_size];

        // He initialization, 0 mean
        let normal = Normal::new(0.0, (2.0/(input_size.pow(2) * input_depth) as f32).sqrt()).unwrap();

        for _ in 0..output_size {
            // Initialize the biases with a small positive value
            biases.push(0.1);
            for i in 0..input_size {
                // Initialize the weights with random values drawn from the normal distribution
                weights[i].push(normal.sample(&mut rand::thread_rng()));
            }
        }
        // Create and return a new fully conneted layer with the initializated values
        let layer: FullyConnectedLayer = FullyConnectedLayer {
            input_size,
            input_width,
            input_depth,
            output_size,
            weights,
            biases,
            input: vec![],
            output: vec![0.0; output_size],
        };

        layer
    }
}

// flatten data to a 1d vector
fn flatten(squares: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    let mut flat_data: Vec<f32> = vec![];

    for square in squares {
        for row in square {
            flat_data.extend(row);
        }
    }
    flat_data
}

impl layer for FullyConnectedLayer {
    // Calculate the output layer by forward propagating the input layer
    fn forward_propagate(&mut self, matrix_input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        
    }
}