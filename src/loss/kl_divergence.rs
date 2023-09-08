use std::marker::PhantomData;

use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Distribution, Int, Tensor},
};

#[derive(Debug)]
pub struct KLDivergence<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> KLDivergence<B> {
    pub fn forward(&self, mean: Tensor<B, 2>, log_var: Tensor<B, 2>) -> Tensor<B, 1> {
        (log_var.clone().add_scalar(1.) - mean.powf(2.) - log_var.exp())
            .sum_dim(1)
            .neg()
            .mean()
    }
}

#[derive(Debug)]
pub struct MaximumMeanDicrepancy<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> MaximumMeanDicrepancy<B> {
    pub fn forward(&self, mean: Tensor<B, 2>, log_variance: Tensor<B, 2>) -> Tensor<B, 1> {
        let variance = log_variance.exp();
        let loss = Self::gaussian_kernel(mean.clone(), mean.clone(), variance.clone()) + 1
            - Self::gaussian_kernel(
                mean.zeros_like(),
                mean,
                variance.add_scalar(1.).div_scalar(2),
            );
        loss.mean()
    }

    fn gaussian_kernel(a: Tensor<B, 2>, b: Tensor<B, 2>, variance: Tensor<B, 2>) -> Tensor<B, 2> {
        (a - b).powf(2.).sum_dim(1).div(variance * 2).neg().exp()
    }
}

#[derive(Debug)]
pub struct DistangledLatent<B: Backend> {
    backend: PhantomData<B>,
    covariance_scaling: f32,
    variance_scaling: f32,
}

impl<B: Backend> DistangledLatent<B> {
    pub fn forward(&self, mean: Tensor<B, 2>) -> Tensor<B, 1> {
        let [_, dim] = mean.dims();
        let device = &mean.device();
        let diagonal = Self::diagonal(dim).to_device(device);

        let covariance_matrix = Self::covariance(mean);

        let lhs = covariance_matrix
            .clone()
            .sub(covariance_matrix.clone() * diagonal.clone())
            .sum()
            .powf(2.)
            .mul_scalar(self.covariance_scaling);

        let rhs = diagonal
            .sub_scalar(1.)
            .mul(covariance_matrix)
            .sum()
            .powf(2.)
            .mul_scalar(self.variance_scaling);

        rhs + lhs
    }

    fn covariance(mean: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, _] = mean.dims();
        let centered = mean.clone() - mean.mean_dim(0);
        centered.clone().transpose().div_scalar(batch as f32 - 1.) * centered
    }

    fn diagonal(size: usize) -> Tensor<B, 2> {
        let indices = (0..size)
            .map(|e| Tensor::<B, 1, Int>::from_ints([e as i32, e as i32]))
            .map(|e| e.unsqueeze())
            .collect::<Vec<Tensor<B, 2, Int>>>();
        let indices = Tensor::cat(indices, 1);
        let ones = Tensor::<B, 1>::from_floats(
            (0..size).map(|e| e as f32).collect::<Vec<f32>>().as_slice(),
        )
        .unsqueeze();
        let diagonal = Tensor::<B, 2>::zeros([size, size]).scatter(0, indices, ones);
        diagonal
    }
}
