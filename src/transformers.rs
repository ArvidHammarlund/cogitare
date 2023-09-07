use burn::{
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{
        activation::{gelu, softmax},
        backend::Backend,
        Tensor,
    },
};

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    layers: Vec<TransformerLayer<B>>,
}

impl<B: Backend> Transformer<B> {
    pub fn new(d_hidden: usize, nr_heads: usize, depth: usize) -> Self {
        let layers = vec![TransformerLayer::new(d_hidden, nr_heads); depth];
        Self { layers }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.layers.iter().fold(input, |res, e| e.forward(res))
    }
}

#[derive(Module, Debug)]
pub struct TransformerLayer<B: Backend> {
    attention: AttentionHeads<B>,
    norm: Vec<LayerNorm<B>>,
    linear: Vec<Linear<B>>,
    dropout: Vec<Dropout>,
}

impl<B: Backend> TransformerLayer<B> {
    pub fn new(d_hidden: usize, nr_heads: usize) -> Self {
        let attention = AttentionHeads::new(d_hidden, nr_heads);
        let norm = vec![
            LayerNormConfig::new(d_hidden * nr_heads).init(),
            LayerNormConfig::new(d_hidden * nr_heads).init(),
        ];
        let linear = vec![LinearConfig::new(d_hidden * nr_heads, d_hidden * nr_heads).init(); 2];
        let dropout = vec![DropoutConfig::new(0.3).init(); 2];
        Self {
            attention,
            norm,
            linear,
            dropout,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let input = self.norm[0].forward(input);
        let x = self.attention.forward(input.clone()) + input;

        let x = self.norm[1].forward(x);

        let x = self.linear[0].forward(x);
        let x = gelu(x);
        let x = self.linear[1].forward(x);
        x
    }
}

#[derive(Module, Debug)]
pub struct AttentionHeads<B: Backend> {
    d_hidden: usize,
    nr_heads: usize,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    merge: Linear<B>,
}

impl<B: Backend> AttentionHeads<B> {
    pub fn new(d_hidden: usize, nr_heads: usize) -> Self {
        let linear = || {
            LinearConfig::new(d_hidden * nr_heads, d_hidden * nr_heads)
                .with_bias(true)
                .init()
        };
        Self {
            query: linear(),
            key: linear(),
            value: linear(),
            d_hidden,
            nr_heads,
            merge: linear(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Component Differentiation by linear project.
        let query = self.query.forward(input.clone());
        let key = self.key.forward(input.clone());
        let value = self.value.forward(input.clone());

        // Reshape to seperate heads.
        let [batch, length, _] = query.dims();
        let query = query.reshape([batch, length, self.nr_heads, self.d_hidden]);
        let key = key.reshape([batch, length, self.nr_heads, self.d_hidden]);
        let value = value.reshape([batch, length, self.nr_heads, self.d_hidden]);

        // Scaled Attention.
        let attention = query
            .matmul(key.transpose())
            .div_scalar(f64::sqrt(self.d_hidden as f64));
        let attention = softmax(attention, 3) * value;

        // Merging of heads.
        let attention = attention.reshape([batch, length, self.nr_heads * self.d_hidden]);
        self.merge.forward(attention)
    }
}
