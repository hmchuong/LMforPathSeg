import argparse
import os.path as osp
import yaml
import logging
import gc
# import psutil
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn

from pyseg.models.model_helper import ModelBuilder

import torch.distributed as dist

from pyseg.utils.loss_helper import get_criterion
from pyseg.utils.lr_helper import get_scheduler, get_optimizer

from pyseg.utils.utils import AverageMeter, intersectionAndUnion, init_log, load_trained_model, dice, AUC, Kappa
from pyseg.utils.utils import set_random_seed, get_world_size, get_rank, is_distributed
from pyseg.dataset.builder import get_loader

parser = argparse.ArgumentParser(description="Pytorch Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
logger = init_log('global', logging.INFO)
logger.propagate = 0


def main():
    global args, cfg
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cudnn.enabled = True
    cudnn.benchmark = True

    # logger.info(cfg)

    # Create network.
    model = ModelBuilder(cfg['net'])
    modules_back = [model.encoder]
    modules_head = [model.auxor, model.decoder]

    device = torch.device("cuda")
    model.to(device)

    state_dict = torch.load(cfg['test']['model'], map_location='cpu')['model_state']
    logger.info("Load trained model from ", str(cfg['test']['model']))
    load_trained_model(model, state_dict)

    # logger.info(model)

    testloader = get_loader(cfg, splits=['val'])

    # Start to test the model
    prec = test(model, testloader)


def test(model, data_loader):
    model.eval()

    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    dice_meter = AverageMeter()
    AUC_meter = AverageMeter()
    Kappa_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        with torch.no_grad():
            preds = model(images)

        # get the output produced by model
        # import ipdb;
        # ipdb.set_trace()
        preds = preds[0] if cfg['net'].get('aux_loss', False) else preds
        output = preds.data.max(1)[1].cpu()
        target = labels.cpu()
        auc = AUC(output.ravel(), target.ravel())
        kappa = Kappa(output.ravel(),target.ravel())
        output = output.numpy()
        target = target.numpy()

        # start to calculate miou
        dice_coeff = dice(output, target)
        intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)

        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        dice_meter.update(dice_coeff)
        if kappa == kappa:
            Kappa_meter.update(kappa)
        if auc == auc:
            AUC_meter.update(auc)
        if step % 20 == 0:
            logger.info('iter = {} of {} completed'
                        .format(step, len(data_loader)))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    mDice = dice_meter.avg
    mAUC = AUC_meter.avg
    kappa = Kappa_meter.avg

    logger.info(
        'Eval result: mIoU/mAcc/allAcc/mDice/mAUC/kappaScore {:.4f}/{:.4f}/{:.4f}/{:.4f}./{:.4f}./{:.4f}.'.format(mIoU, mAcc, allAcc,
                                                                                               mDice, mAUC, kappa))
    for i in range(num_classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}'.format(i, iou_class[i], accuracy_class[i]))


if __name__ == '__main__':
    main()
