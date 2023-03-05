use ndarray::{Array1};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU;

impl<'a> Differentiable<'a> for ReLU {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|i| i.max(0.0))
    }
    
    fn diff(&self, x: &Array1<f64>) -> Array1<f64> {
        x.map(|&i| if i > 0.0 { 1.0 } else { 0.0 })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mse(pub Array1<f64>); //TODO

impl<'a> Differentiable<'a> for Mse {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        (0.5 * (x - &self.0)).map(|x| x.powi(2))
    }
    
    fn diff(&self, x: &Array1<f64>) -> Array1<f64> {
        x - &self.0
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Linear;

impl<'a> Differentiable<'a> for Linear {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }

    fn diff(&self, x: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![1.0; x.shape()[0]])
    }
}

pub trait Differentiable<'a>: Clone + Deserialize<'a> + Serialize {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64>;

    fn diff(&self, x: &Array1<f64>) -> Array1<f64>;
}