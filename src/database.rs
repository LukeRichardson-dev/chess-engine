use ndarray::{Array2, Array3, Array1, arr1};
use rusqlite::{Connection, params};
use base64::prelude::*;
use base64::engine::{GeneralPurpose};
use base64::{Engine as _, engine::general_purpose};   

pub fn init(uri: &String) -> Connection {
    Connection::open(uri).unwrap()
}

#[derive(Debug)]
pub struct Game {
    board: Array3<f64>,
    wins: u64,
    losses: u64,
}

impl Game {
    pub fn from_str(text: &str, wins: u64, losses: u64) -> Self {
        let buf = general_purpose::STANDARD.decode(text).unwrap();
        Self::from_bytes(buf, wins, losses)
    }

    pub fn from_bytes(buf: Vec<u8>, wins: u64, losses: u64) -> Self {
        Self {
            board: Array3::from_shape_vec(
                (8, 8, 8),
                buf.iter()
                    .flat_map(|c| {
                        let c1 = c & 0b1111;

                        vec![
                            if c1 == 1 {1.0} else {0.0},
                            if c1 == 2 {1.0} else {0.0},
                            if c1 == 3 {1.0} else {0.0},
                            if c1 == 4 {1.0} else {0.0},
                            if c1 == 5 {1.0} else {0.0},
                            if c1 == 6 {1.0} else {0.0},
                            if c & 0b1000000 != 0 {1.0} else {0.0},
                            if c & 0b10000000 != 0 {1.0} else {0.0},
                        ]
                    })
                    .collect()
            ).unwrap(),
            wins,
            losses,
        }
    }

    pub fn as_slice(&self) -> Array1<f64> {
        arr1(self.board.as_slice().unwrap())
    }

    pub fn winrate(&self) -> (f64, f64) {
        let t = (self.wins + self.losses) as f64;
        let w = self.wins as f64 / t;
        (w, 1.0 - w)
    }
}

pub fn get_batch(conn: &Connection, min_occurences: usize, size: usize) -> Vec<Game> {
    let mut stmnt = conn.prepare(
        "SELECT * FROM chess_moves 
        WHERE wins + losses > ?1 
        ORDER BY RANDOM() LIMIT ?2", 
    ).unwrap();

    stmnt.query_map(
        params![
            min_occurences, 
            size
        ],
        |row| Ok({
            // let hash: i64 = row.get(0).unwrap();
            let board = &row.get::<_, String>(1).unwrap();
            let wins: u64 = row.get(2).unwrap();
            let losses: u64 = row.get(3).unwrap();
            
            Game::from_str(&board, wins, losses)
        })
    )   
        .unwrap()
        .filter_map(|x| x.ok())
        .collect()
    
}