import torch
import numpy as np

from .models import FCN, save_model, ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import torchvision.transforms as T
import torch.optim as optim

def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = ClassificationLoss()
    loss_func.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 1e-5)
    epochs = 50

    #trans = T.Compose((T.ToPILImage(), T.ColorJitter(0.8, 0.3), T.RandomHorizontalFlip(), T.RandomCrop(32), T.ToTensor()))
    data_train = load_dense_data('dense_data/train')
    data_val = load_dense_data('dense_data/valid')


    for epoch in range(epochs):
        model.train()

        for image, labels in data_train:
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred_labels = model(image)
            loss = loss_func(pred_labels, labels.long())
            loss.backward()
            optimizer.step()

        model.eval()
        count = 0
        accuracy = 0
        for image, label in data_val:
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            accuracy = accuracy + (pred.argmax(1) == label).float().mean().item()
            count += 1
        print("Epoch: " + str(epoch) + ", Accuracy: " + str(accuracy/count))
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
