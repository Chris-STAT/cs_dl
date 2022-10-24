from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torchvision.transforms as T
import torch.utils.tensorboard as tb
import torch.optim as optim


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = ClassificationLoss()
    loss_func.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 55

    trans = T.Compose((T.ToPILImage(), T.ColorJitter(0.8, 0.3), T.RandomHorizontalFlip(), T.RandomCrop(32), T.ToTensor()))
    data_train = load_data('data/train', transform = trans)




    for epoch in range(epochs):
        model.train()

        for image, labels in data_train:
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred_labels = model(image)
            loss = loss_func(pred_labels, labels)
            loss.backward()
            optimizer.step()
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
