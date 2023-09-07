use std::collections::HashMap;

use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::{decay::WeightDecayConfig, AdamConfig},
    tensor::backend::ADBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use tokenizers::Tokenizer;

use crate::{
    data::{GenreBatcher, GenreItem},
    lr::PeriodicLr,
    model::GenreModel,
};

#[derive(Config)]
pub struct GenreTrainingConfig {
    pub num_epochs: usize,

    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,
}

impl GenreTrainingConfig {
    pub fn train<B: ADBackend, D: Dataset<GenreItem> + 'static>(
        &self,
        device: B::Device,
        train: D,
        validate: D,
        weights: Vec<f32>,
        output_dir: &str,
    ) -> GenreModel<B> {
        // Config
        let optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-7)));
        B::seed(self.seed);

        // Data
        let batcher_train = GenreBatcher::<B>::new(device.clone());
        let batcher_valid = GenreBatcher::<B::InnerBackend>::new(device.clone());

        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(self.batch_size)
            .shuffle(self.seed)
            .num_workers(self.num_workers)
            .build(train);
        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(self.batch_size)
            .shuffle(self.seed)
            .num_workers(self.num_workers)
            .build(validate);

        let lr = PeriodicLr::new(1);

        // Model
        let learner = LearnerBuilder::new(output_dir)
            .metric_train(LossMetric::new())
            .metric_train(AccuracyMetric::new())
            .metric_valid(LossMetric::new())
            .metric_valid(AccuracyMetric::new())
            .devices(vec![device.clone()])
            .num_epochs(self.num_epochs)
            .build(GenreModel::new(weights), optimizer.init(), 1e-3);

        let model_trained = learner.fit(dataloader_train, dataloader_valid);
        model_trained
    }
}
