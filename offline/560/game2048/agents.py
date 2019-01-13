import numpy as np
import torch

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class myAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        device = torch.device('cuda')
        model = torch.load('/mnt/H/yunyun/560/mymodel.pth')

        model.to('cpu')
        self.search_func = model

    def step(self):
        board = self.preprocess()
        direction = self.search_func(board)
        maxid = np.where(direction == torch.max(direction))[1]
        counts = np.bincount(maxid)
        d = np.argmax(counts)

        return int(d)

    def preprocess(self):
        board = self.game.board
        board[board == 0] = 1
        board = np.log2(board).flatten()
        board = board.reshape(4, 4)
        board = board[:, :np.newaxis]
        board=board/11.0
        board = torch.from_numpy(board)
        board = torch.unsqueeze(board, dim=0)
        board = board.repeat(1, 1, 1, 1)
        board = torch.tensor(board,dtype = torch.float32)
        return board
