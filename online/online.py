import collections
import random
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from game2048.expectimax import board_to_move

OUT_SHAPE = (4, 4)
CAND = 16
map_table = {2 ** i: i for i in range(1, CAND)}
map_table[0] = 0


def grid_ohe(arr):
    ret = np.zeros(shape=(CAND,) + OUT_SHAPE, dtype=bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[map_table[arr[r, c]], r, c] = 1
    return ret


Guide = collections.namedtuple('Guide', ('state', 'action'))


class Guides:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *arg):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Guide(*arg)
        self.position = (self.position + 1) % self.capacity

    def pop(self, *arg):
        if len(self.memory) < self.capacity:
            self.memory.remove(None)
        self.memory[self.position] = Guide(*arg)
        self.position = (self.position - 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def ready(self, batch_size):
        return len(self.memory) >= batch_size

    def _len_(self):
        return len(self.memory)


class ModeWrapper:
    def __init__(self, model, capacity):
        self.model = model
        self.memory = Guides(capacity)
        self.training_step = 0

    def predict(self, board):
        return self.model.predict(np.expand_dims(board, axis=0))

    def move(self, game):
        tmp = game.board
        ohe_board = grid_ohe(tmp)
        ohe_board = ohe_board.astype(np.float32)
        suggest = board_to_move(game.board)
        direction = self.predict(ohe_board).argmax()
        game.move(direction)
        current = game.board
        self.memory.push(ohe_board, suggest)
        #print(ohe_board)
        a=game.score
        print('score: %d' % a)
        return (tmp == current).all()

    def train(self, batch):
        if self.memory.ready(batch):
            guide = self.memory.sample(batch)
            X = []
            Y = []
            for guide in guide:
                X.append(guide.state)
                ohe_action = [0] * 4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            self.training_step += 1
            print('training epoch: %d' % self.training_step)
            training(self.model, X, guide.action, batch)


def training(model, data, target, batch):
    #model = model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss = nn.CrossEntropyLoss()
    #device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

    temp=[0]
    temp[0]=target
    data = torch.tensor(data).float()
    target = torch.tensor(np.array(temp)).long()

    for n in range(batch):
        output = model(data)
        output = output.view(1,4)

        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        print('loss:%.3f ' % (loss_value.item()))
        predict = output.data.max(1)[1]

        print("\t", predict[0:5])
        print("\t", target[0:5])