import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)


    loss = torch.nn.BCEWithLogitLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-6)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)

    train_data = load_detection_data('dense_data/train')

    for epoch in range(50):
        model.train()
        for image, heatmap, delta in train_data:
            image = image.to(device)
            heatmap = heatmap.to(device)

            pred_heatmap = model(image)

            l = loss(pred_heatmap, heatmap).mean()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
    save_model(model)






    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
