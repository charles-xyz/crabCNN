use mnist::{Mnist, MnistBuilder};
use rand::Rng;

use crate::cnn_struct::CNN;

pub fn run() {
    // load the MNIST dataset
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize();
    
    let train_data: Vec<Vec<Vec<f32>>> = format_images(trn_img, 50_000);
    let train_labels: Vec<u8> = trn_lbl;

    let _test_data: Vec<Vec<Vec<f32>>> = format_images(tst_img, 10_000);
    let _test_labels: Vec<u8> = tst_lbl;

    // Create a new CNN and specify its layers
    let mut cnn: CNN = CNN::new();
    cnn.add_conv_layer(28, 1, 6, 5, 1);
    
}