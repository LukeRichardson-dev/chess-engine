use std::{path::Path, io::{self, Read, Write}, fs::File};

use crate::maths::{self, Mse};
use maths::{Differentiable, ReLU, Linear};
use ndarray::prelude::{*, s, NewAxis};
use serde::{Deserialize, Serialize};
use anyhow::Result;

pub struct Network<A: Differentiable, B: Differentiable, C: Differentiable> {
    pub layers: Vec<Layer<A>>,
    final_activation: B,
    cost: C,
}

impl<A: Differentiable, B: Differentiable, C: Differentiable> Network<A, B, C> {
    pub fn from_layout(shape: Vec<usize>, cost: C, final_activation: B, activation: A) -> Self {
        Self {
            layers: shape[..shape.len() - 1]
                .iter()
                .zip(shape[1..].iter())
                .map(|(a, b)| {
                    Layer { 
                        weights: Array2::from_shape_vec(
                            (*b, *a),
                            vec![0.0; a * b].iter()
                                .map(|_| (rand::random::<f64>() - 0.5) * 2.0)
                                .collect::<Vec<f64>>()
                        ).unwrap(), 
                        biases: Array1::from_vec(vec![0.0; *b]), 
                        activation: activation.clone(),
                    }
                })
                .collect(),
                final_activation, 
            cost,
        }
    }

    pub fn train(
        &mut self, 
        training_data: (Array2<f64>, Array2<f64>), 
        learning_rate: f64,
        epochs: usize,
    ) {
        
        for _i in 1..=epochs {
            
            for (i, o) in training_data.0.columns().into_iter()
                .zip(training_data.1.columns().into_iter()) {
                    let acts = self.activations(&i.to_owned());
                    let prev = Mse(o.to_owned()).diff(acts.last().unwrap());

                    self.layers.iter_mut().enumerate().rev().fold(prev, |a, (idx, layer)| {
                        // println!("i");   
                        layer.descend(&acts[idx], &a, learning_rate)
                    });
            }
        }

        let binding = self.activations(&training_data.0.column(0).to_owned());
        let res = binding.last().unwrap();
        println!("{:?}", res);
        
    }

    fn activations(&self, input: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut acts = vec![input.clone()];
        self.layers.iter().fold(input.clone(), |a, x| {
            let n = x.apply(&a);
            acts.append(&mut vec![n.clone()]);
            n
        });
        acts
    }

    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        self.activations(input).last().unwrap().clone()
    }
    
}

impl <A: Differentiable, B: Differentiable> Network<A, B, Mse> {
    pub fn load(path: &Path, fa: B) -> Result<Self> {

        let mut data = "".to_owned();
        
        let mut f = File::open(path)?;
        f.read_to_string(&mut data)?;
    
        let layers: Vec<Layer<A>> = serde_json::from_str(&data).unwrap();
        
        Ok(Network {
            layers: layers.into_iter().map(|x| Box::new(x)).collect(),
            final_activation: fa,
            cost: Mse(arr1(&[])),
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string_pretty(&self.layers).unwrap();

        let mut f = File::create(path)?;
        f.write_all(data.as_bytes()).unwrap();

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Layer<'a, A: Differentiable<'a>> {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: A,
}

impl<A: Differentiable + Clone> Layer<A> {
    pub fn apply(&self, inputs: &Array1<f64>) -> Array1<f64> {
        self.activation.apply(&(&self.weights.dot(inputs) + &self.biases))
    }
    
    pub fn differentiate(&self, inputs: &Array1<f64>, doutput: &Array1<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let z = &self.weights.dot(inputs) + &self.biases;
        let dz = self.activation.diff(&z);
        let rhs = &dz * doutput; 
        
        let dw = &inputs.slice(s![NewAxis, ..]) * &rhs.slice(s![.., NewAxis]);
        let da = rhs.dot(&self.weights);
        let db = doutput.clone();

        (dw, db, da)
    }

    pub fn descend(&mut self, inputs: &Array1<f64>, doutput: &Array1<f64>, learning_rate: f64) -> Array1<f64> {
        let (dw, db, da) = self.differentiate(inputs, doutput);
        self.weights -= &(dw * learning_rate);
        self.biases -= &(db * learning_rate);

        da
    }
}


