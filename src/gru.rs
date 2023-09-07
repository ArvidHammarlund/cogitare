use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::sigmoid, backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct GRU<B: Backend> {
    d_hidden: usize,
    forget: Gate<B>,
    update_value: Gate<B>,
    update_factor: Gate<B>,
}

impl<B: Backend> GRU<B> {
    pub fn new(d_input: usize, d_hidden: usize) -> Self {
        let forget = Gate::new(d_input, d_hidden);
        let update_value = Gate::new(d_input, d_hidden);
        let update_factor = Gate::new(d_input, d_hidden);
        Self {
            forget,
            update_value,
            update_factor,
            d_hidden,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>, state: Option<Tensor<B, 2>>) -> Tensor<B, 2> {
        // Init latent state to zeros if not given.
        let [batch, d_input, nr_words] = input.dims();
        let shape = [batch, self.d_hidden];
        let device = &self.devices()[0];
        let mut state = state.unwrap_or(Tensor::zeros_device(shape, device));

        // Time loop.
        for i in 0..nr_words {
            let input = input
                .clone()
                .slice([0..batch, 0..d_input, i..(i + 1)])
                .to_device(device)
                .reshape([batch, d_input]);

            // Forget gate.
            let forget = self.forget.forward(input.clone(), state.clone());
            let forget = state.clone() * sigmoid(forget);

            // Update gates.
            let update_factor = self.update_factor.forward(input.clone(), state.clone());
            let update_factor = sigmoid(update_factor);
            let update_value =
                self.update_value.forward(input.clone(), forget).tanh() * update_factor.clone();

            state = state * -update_factor + update_value
        }
        state
    }
}

#[derive(Module, Debug)]
pub struct Gate<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
}

impl<B: Backend> Gate<B> {
    pub fn new(d_input: usize, d_hidden: usize) -> Self {
        let input = LinearConfig::new(d_input, d_hidden).with_bias(true).init();
        let hidden = LinearConfig::new(d_hidden, d_hidden).with_bias(true).init();
        Self { input, hidden }
    }

    pub fn forward(&self, input: Tensor<B, 2>, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.input.forward(input) + self.hidden.forward(hidden)
    }
}
