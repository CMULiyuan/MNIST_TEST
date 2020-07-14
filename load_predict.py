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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
model.load_state_dict(torch.load("model.pt"))
model.eval()
zoo_model = TorchModel.from_pytorch(model)
# TorchModel predict
samples = []
f = open("/tmp/data/MNIST/raw/t10k-images-idx3-ubyte", "rb")
f.read(16)
for i in range(10000):
    temp = np.array([])
    for j in range(28 * 28):
        temp = np.append(temp, ord(f.read(1)))
    temp = temp.reshape(28, 28)
    temp = temp[:, :, np.newaxis]
    samples.append(Sample.from_ndarray(temp, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
predictSet = sc.parallelize(samples)
rdd = zoo_model.predict(predictSet, 1000)
for x in rdd.collect():
    print(x)
# print(rdd.shape)
