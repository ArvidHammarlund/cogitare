use burn::tensor::Int;

use {
    burn::{
        config::Config,
        module::Module,
        nn::{
            transformer::{
                TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput,
                TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
            },
            Embedding, EmbeddingConfig,
        },
        tensor::{backend::Backend, Bool, Distribution, Shape, Tensor},
    },
    derive_builder::Builder,
};

#[derive(Config)]
pub struct CogitareConfig {
    pub masking_rate_encoder: f64,
    pub masking_rate_decoder: f64,
    pub d_hidden: usize,
    pub nr_layers: usize,
    pub nr_heads: usize,
}

impl CogitareConfig {
    pub fn init<B: Backend>(&self) -> Cogitare<B> {
        Cogitare {
            encoder: TransformerEncoderConfig::new(
                self.d_hidden,
                self.d_hidden * 4,
                self.nr_heads,
                self.nr_layers,
            )
            .with_norm_first(true)
            .init(),
            decoder: TransformerDecoderConfig::new(
                self.d_hidden,
                self.d_hidden * 4,
                self.nr_heads,
                1,
            )
            .with_norm_first(true)
            .init(),
            encoder_masking_rate: self.masking_rate_encoder,
            decoder_masking_rate: self.masking_rate_decoder,
            d_hidden: self.d_hidden,
            cls_embedding: EmbeddingConfig::new(1, self.d_hidden).init(),
            mask_embedding: EmbeddingConfig::new(1, self.d_hidden).init(),
        }
    }
}

#[derive(Clone, Builder)]
#[builder(setter(strip_option))]
pub struct CogitareInput<B: Backend> {
    pub continuous: Tensor<B, 3>,
    pub continuous_pos: Tensor<B, 3>,

    #[builder(default)]
    pub continuous_pad: Option<Tensor<B, 2, Bool>>,

    pub categorical: Tensor<B, 3>,
    pub categorical_pos: Tensor<B, 3>,

    #[builder(default)]
    pub categorical_pad: Option<Tensor<B, 2, Bool>>,

    pub entity_embeddings: Tensor<B, 2>,
}

#[derive(Clone, Builder)]
pub struct CogitareOutput<B: Backend> {
    pub embedding: Tensor<B, 2>,
    pub encoder_output: Tensor<B, 3>,
    pub continuous: Tensor<B, 3>,
    pub categorical: Tensor<B, 3>,
    pub distangle_loss: Tensor<B, 1>,
}

#[derive(Module, Debug)]
pub struct Cogitare<B: Backend> {
    pub cls_embedding: Embedding<B>,
    pub mask_embedding: Embedding<B>,
    pub encoder: TransformerEncoder<B>,
    pub decoder: TransformerDecoder<B>,
    pub encoder_masking_rate: f64,
    pub decoder_masking_rate: f64,
    pub d_hidden: usize,
}

impl<B: Backend> Cogitare<B> {
    pub fn forward_inference(&self, input: CogitareInput<B>) -> CogitareOutput<B> {
        let features = self.build_feature_tensor(&input);

        let pad_mask = self.build_pad_mask(&input);

        let (cls, encoder_output) = self.encoder_forward(features.clone(), pad_mask.clone(), false);
        let cls = cls + input.entity_embeddings.clone();
        let [batch, _, dim] = encoder_output.dims();
        let encoder_output = Tensor::cat(
            vec![cls.clone().reshape([batch, 1, dim]), encoder_output],
            1,
        );

        let distangle_loss = self.calc_distangle(cls.clone());

        let output = self.decoder_forward(&input, cls.clone(), pad_mask, false);

        let (continuous, categorical) = self.split_output(output, &input);

        CogitareOutputBuilder::default()
            .embedding(cls)
            .encoder_output(encoder_output)
            .distangle_loss(distangle_loss)
            .continuous(continuous)
            .categorical(categorical)
            .build()
            .expect("Should be able to build cogitare output.")
    }

    pub fn forward(&self, input: CogitareInput<B>) -> CogitareOutput<B> {
        let features = self.build_feature_tensor(&input);

        let pad_mask = self.build_pad_mask(&input);

        let (cls, encoder_output) = self.encoder_forward(features.clone(), pad_mask.clone(), true);
        let cls = cls + input.entity_embeddings.clone();
        let [batch, _, dim] = encoder_output.dims();
        let encoder_output = Tensor::cat(
            vec![cls.clone().reshape([batch, 1, dim]), encoder_output],
            1,
        );

        let distangle_loss = self.calc_distangle(cls.clone());

        let output = self.decoder_forward(&input, cls.clone(), pad_mask, true);

        let (continuous, categorical) = self.split_output(output, &input);

        CogitareOutputBuilder::default()
            .embedding(cls)
            .encoder_output(encoder_output)
            .distangle_loss(distangle_loss)
            .continuous(continuous)
            .categorical(categorical)
            .build()
            .expect("Should be able to build cogitare output.")
    }

    fn build_pad_mask(&self, input: &CogitareInput<B>) -> Tensor<B, 2, Bool> {
        let device = &self.devices()[0];
        let [_, cat_len, _] = input.categorical.dims();
        let [batch, con_len, _] = input.continuous.dims();

        let continuous = match input.continuous_pad.clone() {
            Some(pad) => pad,
            None => Tensor::<B, 2>::zeros([batch, con_len]).greater_equal_elem(0.5),
        };

        let categorical = match input.categorical_pad.clone() {
            Some(pad) => pad,
            None => Tensor::<B, 2>::zeros_device([batch, cat_len], device).greater_equal_elem(0.5),
        };

        let cls = Tensor::<B, 2>::zeros_device([batch, 1], device).greater_equal_elem(0.5);

        Tensor::cat(vec![cls, continuous, categorical], 1)
    }

    fn build_feature_tensor(&self, input: &CogitareInput<B>) -> Tensor<B, 3> {
        let device = &self.devices()[0];
        let [batch, _, _] = input.continuous.dims();

        let continuous = input.continuous.clone();
        let categorical = input.categorical.clone();

        let cls = Tensor::<B, 2>::zeros_device([batch, 1], device).int();
        let cls = self.cls_embedding.forward(cls);

        Tensor::cat(vec![cls, continuous, categorical], 1)
    }

    fn build_attn_mask(&self, shape: Shape<3>, masking_rate: f64) -> Tensor<B, 3, Bool> {
        let device = &self.devices()[0];
        let [batch, length, dim] = shape.dims;

        let diagonal = Tensor::<B, 2>::diagonal(length)
            .unsqueeze()
            .repeat(0, batch)
            .greater_equal_elem(0.5)
            .to_device(device);

        Tensor::<B, 3>::random_device(shape.dims, Distribution::Bernoulli(masking_rate), device)
            .slice_assign(
                [0..batch, 0..1, 0..dim],
                Tensor::zeros_device([batch, 1, dim], device),
            )
            .mask_fill(diagonal.clone(), 1)
            .greater_equal_elem(0.5)
    }

    fn encoder_forward(
        &self,
        features: Tensor<B, 3>,
        pad_mask: Tensor<B, 2, Bool>,
        mask: bool,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let device = &self.devices()[0];
        let [batch, len, dim] = features.dims();

        let tensor = if mask {
            let mask = Tensor::<B, 3>::random_device(
                [batch, len, 1],
                Distribution::Bernoulli(self.encoder_masking_rate),
                device,
            )
            .mask_fill(pad_mask.clone().reshape([batch, len, 1]), 0)
            .greater_equal_elem(0.5);
            let mask_embedd = self
                .mask_embedding
                .forward(Tensor::<B, 2, Int>::zeros_device([1, 1], device));
            let features = features.mask_where(mask, mask_embedd);

            self.encoder
                .forward(TransformerEncoderInput::new(features).mask_pad(pad_mask))
        } else {
            self.encoder
                .forward(TransformerEncoderInput::new(features).mask_pad(pad_mask))
        };

        (
            tensor.clone().slice([0..batch, 0..1, 0..dim]).flatten(1, 2),
            tensor.slice([0..batch, 1..len, 0..dim]),
        )
    }

    fn decoder_forward(
        &self,
        input: &CogitareInput<B>,
        encoder_output: Tensor<B, 2>,
        pad_mask: Tensor<B, 2, Bool>,
        mask: bool,
    ) -> Tensor<B, 3> {
        let device = &self.devices()[0];
        let [batch, _, _] = input.categorical.dims();

        let cls = Tensor::<B, 2>::zeros_device([batch, 1], device).int();
        let cls = self.cls_embedding.forward(cls);
        let continuous = input.continuous_pos.clone();
        let categorical = input.categorical_pos.clone();

        let embedd = Tensor::cat(vec![cls, continuous, categorical], 1);
        let [batch, len, dim] = embedd.dims();

        let decoder_input = encoder_output.reshape([batch, 1, dim]).repeat(1, len);
        let attn_mask = self.build_attn_mask(embedd.shape(), self.decoder_masking_rate);

        if mask {
            self.decoder.forward(
                TransformerDecoderInput::new(
                    decoder_input + embedd,
                    self.build_feature_tensor(&input),
                )
                .memory_mask_pad(pad_mask.clone())
                .memory_mask_attn(attn_mask)
                .target_mask_pad(pad_mask),
            )
        } else {
            self.decoder.forward(
                TransformerDecoderInput::new(
                    decoder_input + embedd,
                    self.build_feature_tensor(&input),
                )
                .memory_mask_pad(pad_mask.clone())
                .target_mask_pad(pad_mask),
            )
        }
    }

    fn split_output(
        &self,
        output: Tensor<B, 3>,
        input: &CogitareInput<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, len, dim] = output.dims();
        let split_at = input.continuous.dims()[1] + 1;

        let continuous = output.clone().slice([0..batch, 1..split_at, 0..dim]);
        let categorical = output.slice([0..batch, split_at..len, 0..dim]);
        (continuous, categorical)
    }

    fn calc_distangle(&self, tensor: Tensor<B, 2>) -> Tensor<B, 1> {
        let device = &self.devices()[0];
        let [_, len] = tensor.dims();

        // tensor
        //     .clone()
        //     .cov(0, 1)
        //     .abs()
        //     .mask_fill(
        //         Tensor::<B, 2>::diagonal(len)
        //             .greater_equal_elem(0.5)
        //             .to_device(device),
        //         0,
        //     )
        //     .mean_dim(1)
        //     .mean()
        Tensor::zeros_device([1], device)
    }
}
