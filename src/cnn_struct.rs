use crate::{
    conv_layer::ConvLayer, fully_connected_layer::FullyConnectedLayer, layer::Layer
};

pub struct CNN {
    // the structure of the CNN is represented by a vector
    // of Layer objects
    // We gave our 3 types of layers that trait 
    layers: Vec<Box<dyn Layer>>,
}