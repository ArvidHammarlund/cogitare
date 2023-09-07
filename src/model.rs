use burn::{
    module::Module,
    nn::{
        generate_sinusoids,
        loss::CrossEntropyLoss,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, Lstm, LstmConfig,
    },
    tensor::{
        activation::softmax,
        backend::{ADBackend, Backend},
        Distribution, Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use itertools::Itertools;
use strum::EnumCount;

use crate::{data::TrainingBatch, gru::GRU};
use crate::{lstm::LSTM, transformers::Transformer};

#[derive(Module, Debug)]
pub struct GenreModel<B: Backend> {}

impl<B: Backend> GenreModel<B> {
    pub fn new(vocab_size: usize, weights: Vec<f32>) -> Self {
        Self {}
    }

    pub fn forward(&self, input: TrainingBatch<B>) -> ClassificationOutput<B> {
        // Move to Device.

        // Input noise.

        // Transformer

        // Decoder

        // println!("{}", x.clone().argmax(1));
        // Loss
        let loss = CrossEntropyLoss::new(None).forward(x.clone(), labels.clone());
        // Return
        ClassificationOutput {
            loss,
            output: x,
            targets: labels.clone(),
        }
    }
}

impl<B: ADBackend> TrainStep<TrainingBatch<B>, ClassificationOutput<B>> for GenreModel<B> {
    fn step(&self, item: TrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TrainingBatch<B>, ClassificationOutput<B>> for GenreModel<B> {
    fn step(&self, item: TrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }
}
