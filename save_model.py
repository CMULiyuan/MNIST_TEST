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

def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    if args.cuda:
        result = result.cuda()
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



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dir', default='/tmp/data', metavar='N',
                    help='the folder store mnist data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training per executor(default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing per executor(default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()

torch.manual_seed(args.seed)


# def data_tf(x):
#     x = np.array(x, dtype='float32') / 255
#     x = (x - 0.5) / 0.5
#     x = torch.from_numpy(x)
#     x = x.unsqueeze(0)
#     return x


train_dataset = datasets.MNIST(args.dir, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size, shuffle=True)

test_dataset = datasets.MNIST(args.dir, train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.test_batch_size, shuffle=False)


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
model.train()
criterion = nn.NLLLoss()
print(len(train_loader))
adam = Adam(args.lr)
zoo_model = TorchModel.from_pytorch(model)
zoo_criterion = TorchLoss.from_pytorch(criterion)
zoo_estimator = Estimator(zoo_model, optim_methods=adam)
train_featureset = FeatureSet.pytorch_dataloader(train_loader)
test_featureset = FeatureSet.pytorch_dataloader(test_loader)
from bigdl.optim.optimizer import MaxEpoch, EveryEpoch

zoo_estimator.train_minibatch(train_featureset, zoo_criterion,
                              end_trigger=MaxEpoch(args.epochs),
                              checkpoint_trigger=EveryEpoch(),
                              validation_set=test_featureset,
                              validation_method=[Accuracy()])

# TorchModel predict
# samples = []
# f = open("/tmp/data/MNIST/raw/t10k-images-idx3-ubyte", "rb")
# for i in range(10000):
#     temp = np.array([])
#     for j in range(28 * 28):
#         temp = np.append(temp, ord(f.read(1)))
#     temp.reshape(28, 28)
#     samples.append(Sample.from_ndarray(temp, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
#
# predictSet = sc.parallelize(samples)
# rdd = zoo_model.predict(predictSet, 1000)
# for x in rdd.collect():
#     print(x)
torch_model = zoo_model.to_pytorch()
torch.save(torch_model.state_dict(), "bn_model2.pt")
