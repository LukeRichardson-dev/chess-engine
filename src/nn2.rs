use std::cell::RefCell;

use ndarray::{Array2, Array1, array, s, NewAxis, arr1};
use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Activation {
    Linear,
    ReLU,
    LeakyReLU,
}

impl Activation {
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::ReLU => x.map(|x| x.max(0.0)),
            Self::LeakyReLU => x.map(|x| x.max(x * 0.1)),
            Self::Linear => x.clone(),
        }
    }

    pub fn diff(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::ReLU => x.map(|x| if x > &0.0 {1.0} else {0.0}),
            Self::LeakyReLU => x.map(|x| if x > &0.0 {1.0} else {0.01}),
            Self::Linear => Array1::ones(x.shape()[0]),
        }
    }
}


#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Cost {
    Mse,
}

impl Cost {
    pub fn apply(&self, p: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Mse => 0.5 * (p - y).map(|x| x.powi(2)),
        }
    }

    pub fn diff(&self, p: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Mse => p - y,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: Activation,
    child: Option<Box<RefCell<Layer>>>,
}

impl Layer {
    pub fn new(
        weights: Array2<f64>,
        biases: Array1<f64>,
        activation: Activation,
        child: Option<Box<RefCell<Layer>>>,
    ) -> Self {
        Self { weights, biases, activation, child }
    }

    pub fn random(inputs: usize, outputs: usize, activation: Activation) -> Self {
        Self {
            weights: Array2::zeros((outputs, inputs)).map(|_: &f64| rand::random::<f64>() -0.5),
            biases: Array1::zeros(outputs).map(|_: &f64| rand::random::<f64>() - 0.5),
            activation,
            child: None,
        }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        match &self.child {
            None => self.child = Some(Box::new(RefCell::new(layer))),
            Some(x) => x.borrow_mut().add_layer(layer),
        }
    }

    pub fn add_random_layer(&mut self, size: usize, activation: Activation) {
        match &self.child {
            Some(x) => x.borrow_mut().add_random_layer(size, activation),
            None => self.add_layer( //TODO idk
                Layer::random(self.biases.shape()[0], size, activation)
            ),
        }
    }

    fn apply(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let mx = self.weights.dot(inputs);
        self.activation.apply(&(mx + &self.biases))
    }

    pub fn predict(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let a = self.apply(inputs);

        match &self.child {
            Some(x) => x.borrow().predict(&a),
            None => a,
        }
    }

    pub fn train(&mut self, inputs: &Array1<f64>, outputs: &Array1<f64>, cost: &Cost, lr: f64) -> Array1<f64> {
        let da = match &self.child {
            Some(x) => {
                let act = self.apply(inputs);
                x.borrow_mut().train(&act, outputs, cost, lr)
            },
            None => cost.diff(&self.apply(inputs), outputs)
        };
        
        let (dw, db, _) = self.differentiate(inputs, &da);

        self.weights -= &(&dw * lr);
        self.biases -= &(&db * lr);

        let (_, _, di) = self.differentiate(inputs, &da);
        di
    }

    pub fn differentiate(&mut self, inputs: &Array1<f64>, doutput: &Array1<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let z = &self.weights.dot(inputs) + &self.biases;
        let dz = self.activation.diff(&z);
        let rhs = &dz * doutput; 
        
        let dw = &inputs.slice(s![NewAxis, ..]) * &rhs.slice(s![.., NewAxis]);
        let da = rhs.dot(&self.weights);
        let db = doutput.clone();

        (dw, db, da)
    }
}


pub fn test() {
    let mut network = Layer::random(512, 64, Activation::LeakyReLU);
    network.add_random_layer(64, Activation::LeakyReLU);
    network.add_random_layer(8, Activation::LeakyReLU);
    network.add_random_layer(2, Activation::LeakyReLU);

    let res = network.predict(&arr1(&vec![1.0; 512]));
    println!("{}", res);
    for _ in 0..10_000 {
        network.train(&arr1(&vec![1.0; 512]), &array![1.0, 0.0], &Cost::Mse, 0.00005);
    }
    let res = network.predict(&arr1(&vec![1.0; 512]));
    println!("{}", res);
}