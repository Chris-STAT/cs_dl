from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data


def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    #raise NotImplementedError('train')

    data_train = load_data('data/train')
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_func = ClassificationLoss()
    epochs = 5

    for epoch in range(epochs):
        model.train()

        for image, labels in data_train:
            optimizer.zero_grad()
            pred_labels = model(image)
            loss = loss_func(pred_labels, labels)
            loss.backward()
            optimizer.step()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
