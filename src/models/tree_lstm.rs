use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig},
    tensor::{
        activation::{self, sigmoid, tanh},
        backend::Backend,
        Tensor,
    },
};

#[derive(Config)]
pub struct TreeLstmConfig {
    d_hidden: usize,
    d_cell: usize,
    d_input: usize,
}

impl TreeLstmConfig {
    pub fn init<B: Backend>(&self) -> TreeLstm<B> {
        let linear = |d_input: usize, d_output: usize, gain: f64| {
            LinearConfig::new(d_input, d_output)
                .with_initializer(Initializer::XavierNormal { gain })
                .init()
        };
        // 2 forget gates, 1 update, 1 input, 1 ouput.
        TreeLstm {
            hidden: linear(self.d_hidden * 2, self.d_cell * 5, 1.),
            input: linear(self.d_input * 2, self.d_cell * 5, 1.),
            output: linear(self.d_cell, self.d_hidden, 5. / 3.),
            hidden_init: EmbeddingConfig::new(1, self.d_hidden).init(),
            cell_init: EmbeddingConfig::new(1, self.d_cell).init(),
        }
    }
}

#[derive(derive_new::new)]
pub struct TreeLstmInput<B: Backend> {
    input_advance: Tensor<B, 2>,
    hidden_advance: Option<Tensor<B, 2>>,
    cell_advance: Option<Tensor<B, 2>>,

    input_against: Tensor<B, 2>,
    hidden_against: Option<Tensor<B, 2>>,
    cell_against: Option<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
pub struct TreeLstm<B: Backend> {
    hidden: Linear<B>,
    input: Linear<B>,
    output: Linear<B>,
    hidden_init: Embedding<B>,
    cell_init: Embedding<B>,
}

impl<B: Backend> TreeLstm<B> {
    /// Forward pass
    ///
    /// Output: (Hidden, cell).
    pub fn forward(&self, input: TreeLstmInput<B>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Init
        let device = &self.devices()[0];
        let [batch, _] = input.input_advance.dims();

        let (hidden_advance, cell_advance) =
            self.init_state(input.hidden_advance, input.cell_advance, batch);
        let (hidden_against, cell_against) =
            self.init_state(input.hidden_against, input.cell_against, batch);

        // Batched transforms
        let hidden = Tensor::cat(vec![hidden_advance, hidden_against], 1).to_device(device);
        let hidden = self.hidden.forward(hidden);
        let input = Tensor::cat(vec![input.input_advance, input.input_against], 1);
        let input = self.input.forward(input);

        // Forget
        let forget_against = Self::slice_gate(0, hidden.clone(), input.clone(), sigmoid);
        let forget_advance = Self::slice_gate(1, hidden.clone(), input.clone(), sigmoid);
        let cell = cell_advance * forget_advance - cell_against * forget_against;

        // Update
        let update_factor = Self::slice_gate(2, hidden.clone(), input.clone(), sigmoid);
        let update_value = Self::slice_gate(3, hidden.clone(), input.clone(), tanh);
        let cell = cell + update_factor * update_value;

        // Output
        let output = Self::slice_gate(4, hidden, input, sigmoid);
        let hidden = output * tanh(cell.clone());

        (hidden, cell)
    }

    fn slice_gate<F: Fn(Tensor<B, 2>) -> Tensor<B, 2>>(
        idx: usize,
        hidden: Tensor<B, 2>,
        input: Tensor<B, 2>,
        func: F,
    ) -> Tensor<B, 2> {
        let [batch, dim] = hidden.dims();
        let d_cell = dim / 5;
        let range = idx * d_cell..(idx + 1) * d_cell;
        func(hidden.slice([0..batch, range.clone()]) + input.slice([0..batch, range]))
    }

    fn init_state(
        &self,
        hidden: Option<Tensor<B, 2>>,
        cell: Option<Tensor<B, 2>>,
        batch: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match (hidden, cell) {
            (Some(hidden), Some(cell)) => (hidden, cell),
            (None, None) => (
                self.hidden_init
                    .forward(Tensor::zeros([batch, 1]))
                    .flatten(1, 2),
                self.cell_init
                    .forward(Tensor::zeros([batch, 1]))
                    .flatten(1, 2),
            ),
            _ => unreachable!(),
        }
    }
}
