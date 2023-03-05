
// mod neural_netowrk;
pub mod maths;
mod database;
use std::{path::Path, fs::File, io::{Read, Write}};

use database::{get_batch, init, Game};
use ndarray::{array, arr1, arr2};
// use neural_netowrk::Network;

mod nn2;
// use crate::maths::{Mse, ReLU, Linear};

const WIDTH: usize = 8;
const PEICES: usize = 7;

const BOARD_VEC_SIZE: usize = WIDTH * WIDTH * PEICES;

use nn2::{test, Layer};

fn main_old() {
    todo!();

    // let mut net: Network<_, Mse> = Network::from_layout(
    //     vec![BOARD_VEC_SIZE, WIDTH * WIDTH, WIDTH * WIDTH, WIDTH, PEICES, 2], 
    //     Mse(array![]), 
    //     Linear,
    // );

    // net.layers.last().unwrap().activation = ReLU;

    // let path = Path::new("test.json");

    // let r2 = net.predict(&arr1(&[0.5; BOARD_VEC_SIZE]));
    // println!("{}", r2);

    // net.train(
    //     (
    //         arr2(&[[0.5, 0.2]; BOARD_VEC_SIZE]),
    //         arr2(&[
    //             [0.5, 0.2],
    //             [0.2, 0.5],
    //         ]),
    //     ), 
    //     0.0000005,
    //     100, 
    // );

    // // let r1 = net.predict(&arr1(&[1.0; BOARD_VEC_SIZE]));
    // let r2 = net.predict(&arr1(&[0.5; BOARD_VEC_SIZE]));
    // println!("{}", r2);

    // assert_ne!(r1, r2);
}

static DB_URI: &'static str = "test2.json";

fn main() {
    let conn = init(&"chess.db".to_owned());
    let mut eng = Engine::load(&DB_URI.to_owned());

    for i in 1..=1_000 {
        let batch = get_batch(&conn, 2, 64);
        let c = eng.fit(batch);
        println!("{i} Cost: {c}");

        eng.save(&DB_URI.to_owned());
    }


}

struct Engine {
    network: Layer,
}

impl Engine {
    pub fn random() -> Self {
        Self {
            network: {
                let mut net = Layer::random(512, 64, nn2::Activation::LeakyReLU);

                net.add_random_layer(64, nn2::Activation::LeakyReLU);
                net.add_random_layer(64, nn2::Activation::LeakyReLU);
                net.add_random_layer(64, nn2::Activation::LeakyReLU);
                net.add_random_layer(2,  nn2::Activation::LeakyReLU);

                net
            }
        }
    }

    pub fn load(uri: &String) -> Self {
        let mut f = File::open(uri).unwrap();
        let mut buf = Vec::default();
        f.read_to_end(&mut buf).unwrap();

        Self {
            network: serde_json::from_slice(&buf).unwrap(),
        }
    }

    pub fn save(&self, uri: &String) {
        let mut f = File::create(uri).unwrap();
        f.write_all(
            serde_json::to_string_pretty(&self.network).unwrap().as_bytes()
        ).unwrap();
    }

    pub fn fit(&mut self, batch: Vec<Game>) -> f64 {
        let mut cost = 0.0;
        for i in batch.iter() {
            let (w, l) = i.winrate();
            let buf = i.as_slice();

            cost +=  self.network
                .train(&buf, &arr1(&[w, l]), &nn2::Cost::Mse, 0.00005)
                .map(|x| x.powi(2) * 0.5).sum();
        }

        cost / (batch.len() as f64)
    }
}