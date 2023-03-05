SELECT * FROM chess_moves WHERE wins + losses > 1 ORDER BY RANDOM() LIMIT 64;
