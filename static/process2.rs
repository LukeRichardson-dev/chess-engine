use std::{fs::File, io::{self, BufRead}, borrow::Borrow, collections::hash_map};
use base64::{engine::general_purpose, Engine as _};
use chess_turn_engine::{ChessTurnEngine, Setup, Gamestate, DisplayOption, encode_as, side::Side};

mod sql;
use sql::init;

use crate::sql::add_state;

const PATH: &'static str = "static/chess/all_with_filtered_anotations_since1998.txt";

enum GameWinner {
    White,
    Black,
    Draw,
}

impl GameWinner {
    pub fn idx(&self) {
        
    }
}


fn main() {
    let db = init("chess.db".to_owned().borrow());

    let file = File::open(PATH).unwrap();
    let lines = io::BufReader::new(file).lines();
    
    for i in lines.skip(5).map(|x| x.unwrap()) {
        
        let mut cte = ChessTurnEngine::new(Setup::Normal).unwrap();

        let mut data = i
            .strip_suffix(" ").unwrap()
            .split(" ### ");

        let metadata: Vec<_> = data.next().unwrap().split(" ").collect();
        
        let winner = match metadata[2] {
            "1-0" => true,
            "0-1" => false,
            _ => continue,
        };

        let moves = match data.last() {
            Some(x) => x.split(" ").collect::<Vec<_>>(),
            None => continue,
        };

        for (t, mv) in moves.iter().enumerate()
            .map(
                |(i, x)| (
                    i & 1 == 0, 
                    x.split(".").last().unwrap()
                )
            ) {
            // println!("{}", cte.display(DisplayOption::BoardView(chess_turn_engine::ViewMode::SimpleAscii)));
            
            match cte.play_turn(mv) {
                Ok(state) => match state {
                    Gamestate::Ongoing => (),
                    Gamestate::Victory(w) => println!("{w}"),
                    _ => {println!("Draw"); break},
                },
                Err(e) => {println!("{e:?}"); break},
            }
            
            let side = if t {
                Side::White
            } else {
                Side::Black
            };

            let hash = cte.game.board.calc_hash();
            let enc = || general_purpose::STANDARD.encode(encode_as(&cte.board_map(), side));

            add_state(&db, hash, enc, !(t ^ winner));
        }

        if let Gamestate::Ongoing = cte.gamestate() {
            println!("{}", match winner {
                true => "White",
                _ => "Black",
            });
        }

    }
}