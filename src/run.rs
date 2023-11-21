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
}