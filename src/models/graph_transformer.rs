use burn::{
    config::Config,
    module::Module,
    nn::{
        transformer::{PositionWiseFeedForward, PositionWiseFeedForwardConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{
        activation::{sigmoid, softmax},
        backend::Backend,
        Bool, Tensor,
    },
};

#[derive(Config)]
pub struct GraphTransformerConfig {
    layer_config: GraphLayerConfig,
    nr_layers: u32,
}

impl GraphTransformerConfig {
    pub fn init<B: Backend>(&self) -> GraphTransformer<B> {
        let layers = (0..self.nr_layers)
            .map(|_| self.layer_config.init())
            .collect();
        GraphTransformer { layers }
    }
}

#[derive(Module, Debug)]
pub struct GraphTransformer<B: Backend> {
    layers: Vec<GraphLayer<B>>,
}

impl<B: Backend> GraphTransformer<B> {
    pub fn forward(
        &self,
        mut verticies: Tensor<B, 3>,
        mut edges: Tensor<B, 4>,
        padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        for layer in &self.layers {
            let (x, y) = layer.forward(verticies, edges, padding_mask.clone());
            verticies = x;
            edges = y;
        }
        (verticies, edges)
    }
}

#[derive(Config)]
pub struct GraphLayerConfig {
    #[config(default = 0.1)]
    dropout: f64,
    d_verticies: usize,
    d_edges: usize,
    nr_heads: u32,
    len_context: u32,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/libm::sqrt(3.0), fan_out_only:false}"
    )]
    initializer: Initializer,
}

impl GraphLayerConfig {
    pub fn init<B: Backend>(&self) -> GraphLayer<B> {
        GraphLayer {
            verticies_norm_1: LayerNormConfig::new(self.d_verticies).init(),
            verticies_norm_2: LayerNormConfig::new(self.d_verticies).init(),
            verticies_mlp: PositionWiseFeedForwardConfig::new(
                self.d_verticies,
                self.d_verticies * 4,
            )
            .with_dropout(self.dropout)
            .with_initializer(self.initializer.clone())
            .init(),
            edges_norm_1: BatchNormConfig::new(self.len_context as usize).init(),
            edges_norm_2: BatchNormConfig::new(self.len_context as usize).init(),
            edges_mlp: PositionWiseFeedForwardConfig::new(self.d_edges, self.d_edges * 4)
                .with_dropout(self.dropout)
                .with_initializer(self.initializer.clone())
                .init(),
            attn: GraphAttentionConfig::new(
                self.nr_heads as u32,
                self.d_verticies as u32,
                self.d_edges as u32,
            )
            .with_dropout_rate(self.dropout as f32)
            .with_initalizer(self.initializer.clone())
            .init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct GraphLayer<B: Backend> {
    verticies_norm_1: LayerNorm<B>,
    verticies_norm_2: LayerNorm<B>,
    verticies_mlp: PositionWiseFeedForward<B>,

    edges_norm_1: BatchNorm<B, 2>,
    edges_norm_2: BatchNorm<B, 2>,
    edges_mlp: PositionWiseFeedForward<B>,

    attn: GraphAttention<B>,
}

impl<B: Backend> GraphLayer<B> {
    pub fn forward(
        &self,
        verticies: Tensor<B, 3>,
        edges: Tensor<B, 4>,
        padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let verticies = self.verticies_norm_1.forward(verticies);

        let edges = self.edges_norm_1.forward(edges);

        let (verticies_attn, edges_attn) =
            self.attn
                .forward(verticies.clone(), edges.clone(), padding_mask);

        let verticies = verticies.add(verticies_attn);
        let verticies = self.verticies_norm_2.forward(verticies);
        let verticies = self.verticies_mlp.forward(verticies.clone()).add(verticies);

        let edges = edges.add(edges_attn);
        let edges = self.edges_norm_2.forward(edges);
        let edges = self.edges_mlp.forward(edges.clone()).add(edges);

        (verticies, edges)
    }
}

#[derive(Config)]
pub struct GraphAttentionConfig {
    nr_heads: u32,
    d_verticies: u32,
    d_edges: u32,
    #[config(default = 0.1)]
    dropout_rate: f32,
    #[config(default = -1.0e4)]
    min_float: f32,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/libm::sqrt(3.0), fan_out_only:false}"
    )]
    initalizer: Initializer,
}

impl GraphAttentionConfig {
    pub fn init<B: Backend>(&self) -> GraphAttention<B> {
        assert!(
            self.d_verticies % self.nr_heads == 0,
            "d_verticies should be evenly divisable by nr_heads."
        );
        let linear = |d_input: u32, d_output: u32| {
            LinearConfig::new(d_input as usize, d_output as usize)
                .with_initializer(self.initalizer.clone())
                .init()
        };
        GraphAttention {
            key: linear(self.d_verticies, self.d_verticies),
            query: linear(self.d_verticies, self.d_verticies),
            value: linear(self.d_verticies, self.d_verticies),
            output: linear(self.d_verticies, self.d_verticies),
            edge: linear(self.d_edges, self.nr_heads),
            edge_gate: linear(self.d_edges, self.nr_heads),
            edge_output: linear(self.nr_heads, self.d_edges),
            nr_heads: self.nr_heads,
            dropout: DropoutConfig::new(self.dropout_rate as f64).init(),
            min_float: self.min_float,
        }
    }
}

#[derive(Module, Debug)]
pub struct GraphAttention<B: Backend> {
    key: Linear<B>,
    query: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    edge: Linear<B>,
    edge_gate: Linear<B>,
    edge_output: Linear<B>,
    nr_heads: u32,
    dropout: Dropout,
    min_float: f32,
}

impl<B: Backend> GraphAttention<B> {
    pub fn forward(
        &self,
        features: Tensor<B, 3>,
        edges: Tensor<B, 4>,
        padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let key = Self::expand(&self.key, self.nr_heads, features.clone());
        let query = Self::expand(&self.query, self.nr_heads, features.clone());
        let value = Self::expand(&self.value, self.nr_heads, features.clone());
        let new_edge = Self::build_edges(&self.edge, edges.clone());
        // let edge_gate = Self::build_edges(&self.edge_gate, edges.clone());
        // let edge_gate = sigmoid(edge_gate);

        let attn_score = self.attn_scores(query, key).clamp(-5., 5.);
        let attn_merge = attn_score.add(new_edge);

        let attn_weights = self.attn_weights(attn_merge.clone(), padding_mask);

        let attn_output = attn_weights.matmul(value);
        let attn_output = attn_output.swap_dims(1, 2).flatten(2, 3);
        let attn_output = self.output.forward(attn_output);

        let edge_output = attn_merge.swap_dims(1, 3);
        let edge_output = self.edge_output.forward(edge_output);

        (attn_output, edge_output)
    }

    fn attn_weights(
        &self,
        mut attn_score: Tensor<B, 4>,
        padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 4> {
        if let Some(pad) = padding_mask {
            let [batch_size, seq_length] = pad.dims();
            attn_score =
                attn_score.mask_fill(pad.reshape([batch_size, 1, 1, seq_length]), self.min_float);
        }
        softmax(attn_score, 3)
    }

    fn attn_scores(&self, query: Tensor<B, 4>, key: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, _, dim] = query.dims();
        let attn_score = query.matmul(key.transpose());
        let attn_scaled = attn_score.div_scalar((dim as f32).sqrt());
        self.dropout.forward(attn_scaled)
    }

    fn expand(linear: &Linear<B>, nr_heads: u32, tensor: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, len, dim] = tensor.dims();
        let attn_dim = [batch, len, nr_heads as usize, dim / nr_heads as usize];
        linear.forward(tensor).reshape(attn_dim).swap_dims(1, 2)
    }

    fn build_edges(linear: &Linear<B>, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        linear.forward(tensor.clone()).swap_dims(1, 3)
    }
}
