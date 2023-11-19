// Traits for a layer
pub trait Layer {
    // Forward pass input through layer
    fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>>;

    // Backprop gradient through layer
    fn back_propagate(&mut self, error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>>;

    // Returns the output value at a specific index
    fn get_output(&mut self, index: usize) -> f32;
}