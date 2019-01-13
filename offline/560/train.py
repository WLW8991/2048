import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import time
from loaddata import dataset
from Net import myNet

batchSize = 128

def loadData():
    trainingData = dataset(root='mytrain.csv',
                           transform=transforms.Compose([transforms.ToTensor()]))
    training = DataLoader(
        trainingData, batch_size=batchSize, shuffle=True, num_workers=0)

    return training


def training(model, epoch, trainloader, optimizer):
    model.train()
    loss = nn.NLLLoss()

    for n, (data, target) in enumerate(trainloader):
        data = data.type(torch.float)

        if torch.cuda.is_available():
            data = Variable(data).cuda()
            target = Variable(target).cuda()
            model.cuda()
        output = model(data)

        optimizer.zero_grad()
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        if n % 10 == 0:
            print('training epoch: %d loss:%.3f ' % (epoch + 1, loss_value.item()))
            predict = output.data.max(1)[1]
            number = predict.eq(target.data).sum()
            correct = 100 * number / batchSize
            print("\t", predict[0:5])
            print("\t", target[0:5])
            print('Accuracy:%0.2f' % correct, '%')

def train():
    model = myNet()
    trainloader, testloader = loadData()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        training(model, epoch, trainloader, optimizer)
    torch.save(model, 'mymodel.pth')
    #model = torch.load('mymodel.pth')

if __name__ == '__main__':
    train()
