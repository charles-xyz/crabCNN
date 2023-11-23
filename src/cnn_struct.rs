use crate::{
    conv_layer::ConvLayer, fully_connected_layer::FullyConnectedLayer, layer::Layer, max_pooling_layer::MaxPoolingLayer
};

pub struct CNN {
    // the structure of the CNN is represented by a vector
    // of Layer objects
    // We gave our 3 types of layers that trait 
    layers: Vec<Box<dyn Layer>>,
}

impl CNN {

    pub fn new() -> CNN {
        let layers: Vec<Box<dyn Layer>> = Vec::new();

        let cnn: CNN = CNN{ layers };

        cnn
    }

    // Add a conv layer to the NN
    pub fn add_conv_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
    ) {
        // Create a new convolutional layer with the specified parameters
        let conv_layer: ConvLayer =
            ConvLayer::new(input_size, input_depth, num_filters, kernel_size, stride);

        let conv_layer_ptr = Box::new(conv_layer) as Box<dyn Layer>;

        // Push the layer onto the list of layers in the nueral network
        self.layers.push(conv_layer_ptr)
    }

    pub fn add_mxpl_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
        kernel_size: usize,
        stride: usize
    ) {
        // Create a new max pooling layer with the specified parameters
        let mxpl_layer: MaxPoolingLayer =
        MaxPoolingLayer::new(input_size, input_depth, kernel_size, stride);
        let mxpl_layer_ptr = Box::new(mxpl_layer) as Box<dyn Layer>;
        // Push the layer onto the list of layers in the nueral network
        self.layers.push(mxpl_layer_ptr)
    }

    // add in the FC layer
    pub fn add_fc_layer(
        &mut self, 
        input_width: usize, 
        input_depth: usize, 
        output_size: usize,
        ){
        let fc_layer: FullyConnectedLayer =
            FullyConnectedLayer::new(input_width, input_depth, output_size);
        let fc_layer_ptr = Box::new(fc_layer) as Box<dyn Layer>;
            // push it up
        self.layers.push(fc_layer_ptr)
    }

    // forward propagates an input matrix through the CNN
    pub fn forward_propagate(&mut self, image: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
        let mut output: Vec<Vec<Vec<f32>>> = image;

        // Forward propagate through each layer of the network
        for i in 0..self.layers.len() {
            output = self.layers[i].forward_propagate(output);
        }

        // Flatten and return the output of the final layer
        output[0][0].clone()
    }
}