use std::marker::PhantomData;

use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Distribution, Int, Tensor},
};

use derive_new::new;

#[derive(Debug, new)]
pub struct KLDivergence {
    // backend: PhantomData<B>,
}

impl KLDivergence {
    pub fn forward<B: Backend>(&self, mean: Tensor<B, 2>, log_var: Tensor<B, 2>) -> Tensor<B, 1> {
        dbg!(
            (log_var.clone().add_scalar(1.) - mean.powf(2.) - log_var.exp())
                .sum_dim(1)
                .mul_scalar(-0.5)
                .mean()
        )
    }
}

#[derive(Debug, new)]
pub struct MaximumMeanDiscrepancy {}

impl MaximumMeanDiscrepancy {
    pub fn forward<B: Backend>(
        &self,
        mean: Tensor<B, 2>,
        log_variance: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let variance = log_variance.exp();
        let loss = Self::gaussian_kernel(
            mean.zeros_like(),
            mean,
            variance.add_scalar(1.).div_scalar(2),
        )
        .neg()
        .add_scalar(2);
        loss.mean()
    }

    fn gaussian_kernel<B: Backend>(
        a: Tensor<B, 2>,
        b: Tensor<B, 2>,
        variance: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        (a - b).powf(2.).sum_dim(1).div(variance * 2).neg().exp()
    }
}

#[derive(Debug, new)]
pub struct DistangledLatent {
    covariance_scaling: f32,
    variance_scaling: f32,
}

impl DistangledLatent {
    pub fn forward<B: Backend>(&self, mean: Tensor<B, 2>) -> Tensor<B, 1> {
        let [_, dim] = mean.dims();
        let device = &mean.device();
        let diagonal = Self::diagonal(dim).to_device(device);

        let covariance_matrix = Self::cov(mean, 0, 1);

        let lhs = covariance_matrix
            .clone()
            .sub(covariance_matrix.clone() * diagonal.clone())
            .sum()
            .powf(2.)
            .mul_scalar(self.covariance_scaling);

        let rhs = covariance_matrix
            .sub_scalar(1.)
            .mul(diagonal)
            .sum()
            .powf(2.)
            .mul_scalar(self.variance_scaling);

        rhs + lhs
    }

    pub fn cov<B: Backend>(
        tensor: Tensor<B, 2>,
        dim: usize,
        degrees_of_feedom: usize,
    ) -> Tensor<B, 2> {
        let n = tensor.dims()[dim];
        let centered = (tensor.clone() - tensor.mean_dim(dim)).swap_dims(dim, 0);
        centered
            .clone()
            .transpose()
            .matmul(centered)
            .div_scalar(n as f32 - degrees_of_feedom as f32)
    }

    fn diagonal<B: Backend>(size: usize) -> Tensor<B, 2> {
        let indices = Tensor::arange(0..size).unsqueeze();
        let ones = Tensor::ones([size]).unsqueeze();
        Tensor::<B, 2, Int>::zeros([size, size])
            .scatter(0, indices, ones)
            .float()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use burn::backend::NdArrayBackend as B;

    #[test]
    fn test_diagonal() {
        let data = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let lhs = Tensor::<B, 2>::from_floats(data);
        let rhs = DistangledLatent::diagonal::<B>(3);
        lhs.to_data().assert_approx_eq(&rhs.to_data(), 3);
    }

    #[test]
    fn test_kl_divergene() {
        let mean = [
            [-0.1, 0.2, 0.3],
            [0.2, 0.1, -0.1],
            [0.3, 0.2, 0.3],
            [0.5, -0.7, 0.3],
        ];
        let log_variance = [
            [0.2, -0.7, 0.5],
            [-0.6, 0.2, 0.1],
            [0.9, 0.2, -0.4],
            [0.3, 0.7, 1.],
        ];
        let mean = Tensor::<B, 2>::from_floats(mean);
        let log_variance = Tensor::<B, 2>::from_floats(log_variance);
        let kl = DistangledLatent {
            covariance_scaling: 10.,
            variance_scaling: 1.,
        };
        kl.forward::<B>(mean);
        panic!();
    }
}
