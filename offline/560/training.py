import numpy as np
import csv, os
import sys
sys.path.append("..")

from game2048.expectimax import board_to_move
from game2048.agents import ExpectiMaxAgent
from game2048.game import Game
from game2048.displays import Display

GAME_SIZE = 4
SCORE_TO_WIN = 2048
repeat_times = 4000

game = Game(GAME_SIZE, SCORE_TO_WIN)
board = game.board
agent = ExpectiMaxAgent(game, Display())
direction = agent.step()
board = game.move(direction)

direct={}
i=0
k=0
filename = 'mytrain.csv'

if os.path.exists(filename):
	head = True
else:
	head = False
	os.mknod(filename)

with open(filename, "a") as csvfile:
	writer = csv.writer(csvfile)
	if not head:
		writer.writerow(["11","12","13","14",\
			         "21","22","23","24",\
			         "31","32","33","34",\
			         "41","42","43","44",\
			         "direction"])

	while i < repeat_times:

		game = Game(GAME_SIZE, SCORE_TO_WIN)
		board = game.board
		print('Repeat times:', i)

		while(game.end == 0):
			agent = ExpectiMaxAgent(game, Display())
			direction = agent.step()
			board = game.board
			board[board==0] = 1
			board = np.log2(board).flatten().tolist()

			key = str(k+1)
			k = k + 1
			direct[key] = [np.int32(board), np.int32(direction)]
			data = np.int32(np.append(board, direction))
			writer.writerow(data)

			if k%100 == 0:
				print(data)

			game.move(direction)

		i = i + 1

