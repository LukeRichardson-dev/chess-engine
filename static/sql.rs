use std::mem::transmute;
use base64::prelude::*;
use rusqlite::{Connection, params, OptionalExtension};

pub fn init(uri: &String) -> Connection {
    let conn = Connection::open(uri).unwrap();

    conn.execute("create table if not exists chess_moves (
        id integer primary key,
        board text,
        wins integer,
        losses integer
    )", []).unwrap();

    conn
}   

pub fn add_state<T: FnOnce() -> String>(conn: &Connection, hash: u64, board: T, winner: bool) {

    let ihash = unsafe { transmute::<_, i64>(hash) };
    let res: Option<(u32, u32)> = conn.query_row(
        "SELECT wins, losses FROM chess_moves WHERE id = ?", 
        params![ihash],
        |row| {
            row.get(0).map(|x| (x, row.get(1).unwrap()))
        }
    ).optional().unwrap();

    match res {
        Some(x) => {
            conn.execute(
                "UPDATE chess_moves
                SET wins = ?1, losses = ?2
                WHERE id = ?3", 
                params![
                    x.0 + winner as u32,
                    x.1 + !winner as u32,
                    ihash,
                ]
            ).unwrap();
        },
        None => {
            conn.execute(
                "INSERT INTO chess_moves 
                (id, board, wins, losses) 
                values (?1, ?2, ?3, ?4)", 
                params![
                    ihash,
                    board(),
                    if winner {1} else {0},
                    if winner {0} else {1},
                ]
            ).unwrap();
        },
    }
    
    // println!("{:?}", res);
}