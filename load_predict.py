from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.pipeline.estimator import *
from bigdl.optim.optimizer import SGD, Adam
from zoo.common.nncontext import *
from zoo.feature.common import FeatureSet
from zoo.pipeline.api.keras.metrics import Accuracy
from torch.autograd import Variable


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.stage1 = nn.Sequential(
#             nn.Conv2d(1, 6, 3, padding=1),
#             nn.BatchNorm2d(6),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(6, 16, 5),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2)
#         )
#
#         self.classfy = nn.Linear(400, 10)
#
#     def forward(self, x):
#         x = self.stage1(x)
#         x = x.view(x.shape[0], -1)
#         x = self.classfy(x)
#         return x
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    result.zero_()
    indexes_rank = len(indexes.size())
    result.scatter_(
        dim=indexes_rank,
        index=indexes.data.unsqueeze(dim=indexes_rank),
        value=1
    )
    return Variable(result)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.fc2_bn = nn.BatchNorm1d(10)

    def forward(self, x, y=None):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2_bn(self.fc2(x))
        return F.log_softmax(x)


# init on yarn when HADOOP_CONF_DIR and ZOO_CONDA_NAME is provided.
if os.environ.get('HADOOP_CONF_DIR') is None:
    sc = init_spark_on_local(cores=1, conf={"spark.driver.memory": "20g"})
else:
    num_executors = 2
    num_cores_per_executor = 4
    hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
    zoo_conda_name = os.environ.get('ZOO_CONDA_NAME')  # The name of the created conda-env
    sc = init_spark_on_yarn(
        hadoop_conf=hadoop_conf_dir,
        conda_name=zoo_conda_name,
        num_executor=num_executors,
        executor_cores=num_cores_per_executor,
        executor_memory="2g",
        driver_memory="10g",
        driver_cores=1,
        spark_conf={"spark.rpc.message.maxSize": "1024",
                    "spark.task.maxFailures": "1",
                    "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

model = Net()
model.load_state_dict(torch.load("bn_model2.pt"))
model.eval()
zoo_model = TorchModel.from_pytorch(model)
# TorchModel predict
samples = []
predict_dataset = datasets.MNIST("/tmp/data", train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

labels = []
for i in range(10000):
    tensor, label = predict_dataset[i]
    temp = tensor.numpy()
    samples.append(Sample.from_ndarray(temp, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    labels.append(label)
predictSet = sc.parallelize(samples)
rdd = zoo_model.predict(predictSet, 1000)
index = 0
acc = 0
for x in rdd.collect():
    # print(x)
    y = x.tolist()
    res = y.index(max(y))
    if res == labels[index]:
        acc += 1
    index += 1
print(acc)
print(index)
