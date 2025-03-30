# Multi GPU Training with PyTorch Lightning
By default PyTorch utilizes only single GPU for training even if there are multiple GPUs.
This can become severe bottleneck if training is to be done on large datasets.

By using multiple GPUs in training, workloads is distributed across multiple GPUs
leading to efficient utilization of resources and faster training.

## Data Prallelism
Data Prallelism is one of the strategies of Multi-GPU training. Here a big data is broken in chunks let's say 4 chunks, then
each GPU will train the model on the chunks allocated. So, if we have four GPU- GPU1, GPU2, GPU3 and GPU4, then chunk1 will be assigned to GPU1, 
chunk2 will be assigned to GPU2 and so on.

### Data Parallel
#### Algorithm
- Assume 4 GPUs are available on single Machine
- Data is divided into 4 chunks and each GPU uses one chunk for training
- Create 4 replicas of the model to be trained on all the 4 GPUs that means all the weights and biases are copied to all the GPUs
- Each GPU runs one forward pass on the chunk assigned. Since we have 4 chunks, a single forward pass will produce 4 outputs let's say- out1, out2, out3, out4
- The outputs are then sent to the central machine which calculates the gradient
- The Gradient calculated from above step is then again sent to all the four GPUs for model updates

#### Implementation using Pytorch Lightning
```python
import pytorch_lightning as pl
model = MyModel()
trainer = pl.Trainer(num_epochs=5,
                    accelerator='gpu', # Which device to use gpu, cpu or tpu
                    devices=6,  # Number of GPU to utilize
                    strategies='dp'
)
trainer.fit(model)
```
Pytorch Lightning automatically handles everything

#### Problems
- This is not suitable for distributed environment where GPUs are present on multiple machines. It can only utilize GPUs from single machine
- Since a single machine is receiving the outputs (not gradient) from each GPUs, it can become severe bottleneck when the batch size on each GPU is high

### Distributed Data Parallel (DDP)
#### Algorithm
- Assume there are 4 Machines and all can utilized
- The big dataset is divided into 4 chunks and each machine runs the model on a single chunk
- A single machine creates the replica of the original model on each machine
- Each machine then computes output, loss and gradient. This different from DP(Data Parallel) where only output was calculated and then sent to single machine for  gradient calculation.
- The gradient from each GPU is directly sent to other machine for updating the model. For example if GPU1 computes Gradient1, GPU2 computes Gradient2 and so on. So the GPU1 would then send its gradient update to other three GPUs - GPU2, GPU3 and GPU4. Other GPUs would directly add the gradient received from other GPU with their own computed gradient and update their model

#### Implementation using Pytorch Lightning
```python
import pytorch_lightning as pl
model = MyModel()
trainer = pl.Trainer(num_epochs=5,
                    accelerator='gpu', # Which device to use gpu, cpu or tpu
                    devices=6,  # Number of GPU to utilize
                    strategies='ddp'
)
trainer.fit(model)
```
Pytorch Lightning automatically handles everything

### Linear Scaling Rule
When using Data Parallelism, each GPU computes gradients on a mini-batch and then gradients are averaged across all GPUs before applying updates. This effectively means the model is trained on a batch size of k × mini_batch_size instead of mini_batch_size alone.

According to the "Linear Scaling Rule", If you increase the batch size by a factor of k, you should also scale the learning rate by k.
** So, in case of Multi-GPU training the learning rate should be scaled by the number of machines when updating the weights than while training **

#### Why Linear Scaling Rule should be applies?
- With small learning rate on large batch size the update in the weights will be small, leading to slower convergence on the optimal weights
- It could lead to potential underfitting. Since the learning rate is small than it is required the updates in the weight may not be enough. This is because each gradient update has a weaker effect than it should, making optimization inefficient.

### Learning Rate Warm up
Learning Rate Warm up is a strategy where you start with a small learning rate and gradually increase it to the target learning rate.

#### Why Learning Rate Warm up?
- When training batch is large (k x batch_size in multi-GPU training), a high learning rate from start can cause gradient instability
- Warm up prevents gradient from exploding and allowing smooth training

#### How Learning Rate warm is implemeted?
![image](https://github.com/user-attachments/assets/34db99ee-649c-4770-8cf0-162ea13bdf25)
```python
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=0.01)

    # Define the warm-up function (Linear Warm-up)
    def lr_lambda(epoch):
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs  # Gradually increase LR
        return 1.0  # Keep LR constant after warm-up

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
```


