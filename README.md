# 2048
2048大作业
文件夹online下保存的是在线学习模型（由于本地资源限制未能成功训练）。

文件夹offline/560中保存的是离线学习的模型及保存的训练后的model。
其中，Net.py为网络结构；training.py用于训练的类，train.py为用于训练的函数。
训练后的模型保存在mymodel.pth文件中（包括网络模型和权重参数等），若要运行evaluate.py或figerprint.py，需要将agent.py中定义的Class myAgent类中，load.pth的文件路径改为当前mymodel.pth所在路径，运行evaluate.py即可进行测试。
