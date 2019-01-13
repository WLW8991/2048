import torch
import torch.nn as nn
import torch.nn.functional as F

class myNet(nn.Module):
	def __init__(self):
		super(myNet,self).__init__()

		self.conv0=nn.Conv2d(16, 128, kernel_size=[4, 1])
		self.conv1=nn.Conv2d(16, 128, kernel_size=[1, 4])
		self.conv2=nn.Conv2d(16, 128, kernel_size=[2, 2])
		self.conv3=nn.Conv2d(16, 128, kernel_size=[3, 3])
		self.conv4=nn.Conv2d(16, 128, kernel_size=[4, 4])

		self.fc1 = nn.Linear(2816, 128)
		self.fc2 = nn.Linear(128, 4)
import torch
import torch.nn as nn
import torch.nn.functional as F

class myNet(nn.Module):
	def __init__(self):
		super(myNet,self).__init__()

		self.conv0=nn.Conv2d(16, 128, kernel_size=[4, 1])
		self.conv1=nn.Conv2d(16, 128, kernel_size=[1, 4])
		self.conv2=nn.Conv2d(16, 128, kernel_size=[2, 2])
		self.conv3=nn.Conv2d(16, 128, kernel_size=[3, 3])
		self.conv4=nn.Conv2d(16, 128, kernel_size=[4, 4])

		self.fc1 = nn.Linear(2816, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, 4)

	def forward(self, x):
		x=[torch.flatten(self.conv0(x)),torch.flatten(self.conv1(x)),torch.flatten(self.conv2(x)),torch.flatten(self.conv3(x)),torch.flatten(self.conv4(x))]
		x = torch.cat(x)
		x = F.relu(x)
		x = self.fc1(F.relu(x))
		x = self.fc2(F.relu(x))
		return self.fc3(x)

	def predict(self, x):
		x = torch.from_numpy(x)
		with torch.no_grad():
			x = self.forward(x)
			preds = torch.argmax(x, dim=0)
		return preds