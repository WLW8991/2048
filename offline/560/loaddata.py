import numpy as np
import pandas as pd 
import torch
from torchvision import transforms

class generator():

	def __init__(self, filename):
		self.file = filename
		self.index = 0

		dataframe = pd.read_csv(self.file)
		dataArray = dataframe.values

		self.x = dataArray[:, 0:16]
		self.y = dataArray[:, 16]
		self.item_num = len(self.x)
		self.idx = 0

	def getitem(self, idx):
		board = self.x[idx].reshape((4,4))
		return board

	def get_next_batch(self, batchSize):
		return

	def dataLoader(self, batchSize=64, shuffle=False):
		batch = self.item_num/batchSize

		data = np.array([batchSize, 1, 4, 4])
		target = np.array()

		for i in range(batchSize):
			board = self.x.reshape((4, 4))


class dataset(torch.utils.data.Dataset):

	def __init__(self, root, transform=None, targetTransform=None):
		super(dataset, self).__init__()
		dataframe = pd.read_csv(root)
		dataArray = dataframe.values

		self.data = dataArray[:, 0:16]
		self.label = dataArray[:, 16]
		self.transform = transform
		self.targetTransform = targetTransform

	def __getitem__(self, index):
		board = self.data[index].reshape((4, 4))
		board = board[:, :, np.newaxis]
		board = board/11.0
		
		label = self.label[index]

		if self.transform is not None:
			board = self.transform(board)
		return board, label

	def __len__(self):
		return len(self.label)
