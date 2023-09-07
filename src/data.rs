use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Bool, Int, Tensor},
};
use derive_new::new;
use itertools::Itertools;
use strum::EnumCount;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct DataItem {
    pub features: Vec<f32>,
    pub label: i32,
}

#[derive(new)]
pub struct DataBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Debug, Clone)]
pub struct TrainingBatch<B: Backend> {
    pub features: Tensor<B, 2>,
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<GenreItem, TrainingBatch<B>> for GenreBatcher<B> {
    fn batch(&self, items: Vec<GenreItem>) -> TrainingBatch<B> {
        let (features, labels): (Vec<_>, Vec<_>) =
            items.iter().map(|e| (e.features.clone(), e.label)).unzip();
        let features = features
            .iter()
            .map(|e| Tensor::<B, 1>::from_floats(e.as_slice()).unsqueeze())
            .collect_vec();
        let features = Tensor::cat(features, 0);
        let labels = Tensor::<B, 1, Int>::from_ints(labels.as_slice());
        TrainingBatch { labels, features }
    }
}
