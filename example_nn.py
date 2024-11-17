import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from backend import Model, Tensor, Dataset
from ops import Linear, ReLU, Sigmoid, BCELoss, GradientDescent
from utils import DataLoader, metrics


class MLP(Model):
    def __init__(self):
        self.linear1 = Linear(3, 5)
        self.relu1 = ReLU()

        self.linear2 = Linear(5, 1)
        self.sigmoid2 = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=W0221
        return self.sigmoid2(self.linear2(self.relu1(self.linear1(x))))


class MyDataset(Dataset):
    def __init__(self):
        self.x = np.concatenate(
            [
                np.random.uniform(-1, 0, size=(50, 3)),
                np.random.uniform(0, 1, size=(50, 3)),
            ],
            axis=0,
        )

        self.y = np.concatenate(
            [np.zeros((50, 1)), np.ones((50, 1))],
            axis=0,
        )

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def main():
    # load custom dataset
    ds = MyDataset()

    # make dataloader for the dataset
    dl = DataLoader(ds, 8, shuffle=True)

    # make the custom network
    m = MLP()

    # initialize the loss function and optimizer
    loss_fn = BCELoss()
    optim = GradientDescent(m.get_params(), learn_rate=1e-2)

    # init epoch losses list
    losses = []

    # for each epoch
    for _ in (pbar := tqdm(range(1000))):

        # reset batch losses
        batch_losses = []

        # reset gradients in the optimizer
        optim.reset_grads()

        # for each x, y in the batch
        for x, y in dl:

            # run feed forward in the network to get outputs
            yh = m(x)

            # calculate loss and add it to the batch losses
            loss = loss_fn(y, yh)
            batch_losses.append(loss.val)

            # calculate gradients
            loss.backward()

            # update parameters
            optim.step()

        # calculate epoch loss
        losses.append(sum(batch_losses) / len(batch_losses))

        pbar.set_description_str(f"Loss: {losses[-1]:.4f}")

    # plot the loss over training
    plt.plot(losses)
    plt.show()

    # run feedforward on test data
    # remove last dimension because of metrics
    yh = m(ds.x)[:, -1]
    y = ds.y[:, -1]

    # calculate metrics
    print(metrics.accuracy(metrics.logit_to_cat(yh), y))


if __name__ == "__main__":
    main()
