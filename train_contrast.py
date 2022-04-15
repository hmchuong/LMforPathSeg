import argparse
import os.path as osp
import yaml
import logging
import gc
# import psutil
import time
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn

from pyseg.models.model_helper import ModelBuilder

import torch.distributed as dist

from pyseg.utils.loss_helper import get_criterion
from pyseg.utils.lr_helper import get_scheduler, get_optimizer

from pyseg.utils.utils import AverageMeter, intersectionAndUnion, init_log, load_trained_model, dice
from pyseg.utils.utils import set_random_seed, get_world_size, get_rank, is_distributed
from pyseg.dataset.builder import get_loader

parser = argparse.ArgumentParser(description="Pytorch Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument("--local_rank", type=int, default=0)
logger = init_log('global', logging.INFO)
logger.propagate = 0


def main():
    global args, cfg
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cudnn.enabled = True
    cudnn.benchmark = True

    if is_distributed():
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        # synchronize()
    rank = get_rank()
    world_size = get_world_size()
    print('rank,world_size', rank, world_size)
    if rank == 0:
        logger.info(cfg)
    if args.seed is not None:
        print('set random seed to', args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg['saver']['snapshot_dir']) and rank == 0:
        os.makedirs(cfg['saver']['snapshot_dir'])
    # Create network.
    model = ModelBuilder(cfg['net'])
    modules_back = [model.encoder]
    best_prec = 0

    if hasattr(model, "auxor"):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    # Start: new BCE loss update
    if model.auxor_classifier is not None:
        modules_classifier = [model.auxor_classifier]
    # End: new BCE loss update

    device = torch.device("cuda")
    model.to(device)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            find_unused_parameters=True,
        )
    if cfg['saver']['pretrain']:
        state_dict = torch.load(cfg['saver']['pretrain'], map_location='cpu')
        print("Load trained model from ", str(cfg['saver']['pretrain']))
        load_trained_model(model, state_dict['model_state'])
        best_prec = state_dict['best_IoU']

    # model.cuda()
    if rank == 0:
        logger.info(model)

    criterion = get_criterion(cfg)

    # Start: new BCE loss update
    criterion_bce = get_criterion(cfg, bce=True)
    # End: new BCE loss update

    trainloader, valloader = get_loader(cfg)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg['trainer']
    cfg_optim = cfg_trainer['optimizer']

    params_list = []
    for module in modules_back:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr']))
    for module in modules_head:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr'] * 10))
    # Start: new BCE loss update
    for module in modules_classifier:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr'] * 10))
    # End: new BCE loss update

    optimizer = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(cfg_trainer, len(trainloader), optimizer)  # TODO

    # Start to train model
    for epoch in range(cfg_trainer['start_epochs'], cfg_trainer['epochs']):
        # Training
        gc.collect()
        train(model, optimizer, lr_scheduler, criterion, criterion_bce,  trainloader, epoch)
        gc.collect()
        # import pdb; pdb.set_trace()
        # print('After training: RAM memory % used:', psutil.virtual_memory()[2])
        # Validataion
        state = {'epoch': epoch,
                 'model_state': model.state_dict(),
                 'best_IoU': best_prec,
                 'optimizer_state': optimizer.state_dict()}
        torch.save(state, osp.join(cfg['saver']['snapshot_dir'], 'last.pth'))
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logger.info("start evaluation")
            mIoU, mDice = validate(model, valloader, epoch)
            prec = mDice if cfg_trainer.get('metric', 'mIoU') == 'mDice' else mIoU
            # print('After validate: RAM memory % used:', psutil.virtual_memory()[2])
            # import pdb; pdb.set_trace()
            if rank == 0:
                if prec > best_prec:
                    best_prec = prec
                    state = {'epoch': epoch,
                             'model_state': model.state_dict(),
                             'best_IoU': best_prec,
                             'optimizer_state': optimizer.state_dict()}
                    torch.save(state, osp.join(cfg['saver']['snapshot_dir'], 'best.pth'))
                    logger.info('Currently, the best val result is: {}'.format(best_prec))
        # note we also save the last epoch checkpoint
        if epoch == (cfg_trainer['epochs'] - 1) and rank == 0:
            state = {'epoch': epoch,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(state, osp.join(cfg['saver']['snapshot_dir'], 'epoch_' + str(epoch) + '.pth'))
            logger.info('Save Checkpoint {}'.format(epoch))


def train(model, optimizer, lr_scheduler, criterion, criterion_bce, data_loader, epoch):
    model.train()
    try:
        data_loader.sampler.set_epoch(epoch)
    except:
        pass
    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']
    rank, world_size = get_rank(), get_world_size()

    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    for step, batch in enumerate(data_loader):
        i_iter = epoch * len(data_loader) + step
        lr = lr_scheduler.get_lr()
        lr_scheduler.step()

        images, labels = batch

        # Start: new BCE loss update
        labels_classification = torch.amax(labels, (1,2))
        labels_classification = labels_classification.cuda()
        # End: new BCE loss update
        
        images = images.cuda()
        labels = labels.long().cuda()

        preds = model(images)

        # Start: new BCE loss update
        classification_pred = preds[-2]
        del preds[-2]
        classification_loss = criterion_bce(classification_pred, labels_classification.float().cuda()) / world_size
        logger.info("Classification loss: {}".format(classification_loss))
        # End: new BCE loss update

        contrast_loss = preds[-1] / world_size
        loss = criterion(preds[:-1], labels) / world_size
        #TODO: Check how we can incorporate the classification loss - if included in loss, it blows up
        # loss += classification_loss
        # logger.info("loss: {:.4f}, contrast: {:.4f}".format(loss, contrast_loss))
        loss += cfg['criterion']['contrast_weight'] * contrast_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # get the output produced by model
        output = preds[0] if cfg['net'].get('aux_loss', False) else preds[0]
        output = output.data.max(1)[1].cpu().numpy()
        target = labels.cpu().numpy()

        # calculate miou
        intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)

        # gather all validation information

        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        if is_distributed():
            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

        # gather all loss from different gpus
        reduced_loss = loss.clone()
        if is_distributed():
            dist.all_reduce(reduced_loss)
        # print('rank,reduced_loss',rank,reduced_loss)
        losses.update(reduced_loss.item())

        if i_iter % 50 == 0 and rank == 0:
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class[1:])
            mAcc = np.mean(accuracy_class[1:])
            logger.info('iter = {} of {} completed, LR = {} loss = {}, classification loss = {} mIoU = {}'
                        .format(i_iter, cfg['trainer']['epochs']*len(data_loader), lr, losses.avg, classification_loss, mIoU))
            # logger.info('After training: RAM memory used: {} %'.format(psutil.virtual_memory()[2]))
        del preds, output, target, images, labels
        gc.collect()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class[1:])
    if rank == 0:
        logger.info('=========epoch[{}]=========,Train mIoU = {}'.format(epoch, mIoU))


def validate(model, data_loader, epoch):
    model.eval()
    try:
        data_loader.sampler.set_epoch(epoch)
    except:
        pass

    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']

    rank, world_size = get_rank(), get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    dice_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        with torch.no_grad():
            preds = model(images)

        # get the output produced by model
        # Start: new BCE loss update
        output_classification = preds[-2]
        del preds[-2]
        #TODO: Convert output to 0-1 and get the accuracy
        # End: new BCE loss update
        output = preds[0] if cfg['net'].get('aux_loss', False) else preds
        output = output.data.max(1)[1].cpu().numpy()
        target = labels.cpu().numpy()

        # start to calculate miou
        dice_coeff = dice(output, target)
        intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)

        # gather all validation information
        # reduced_intersection = intersection
        # reduced_union = union
        # reduced_target = target

        # if is_distributed():
        #     reduced_intersection = torch.from_numpy(intersection).cuda()
        #     reduced_union = torch.from_numpy(union).cuda()
        #     reduced_target = torch.from_numpy(target).cuda()
        #     dist.all_reduce(reduced_intersection)
        #     dist.all_reduce(reduced_union)
        #     dist.all_reduce(reduced_target)

        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        dice_meter.update(dice_coeff)
        if step % 20 == 0:
            logger.info('iter = {} of {} completed'
                        .format(step, len(data_loader)))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mDice = dice_meter.avg
    
    if rank == 0:
        logger.info('=========epoch[{}]=========,Val mIoU = {}, Val mDice = {}'.format(epoch, mIoU, mDice))
    # torch.save(mIoU, 'eval_metric.pth.tar')
    return mIoU, float(mDice)


if __name__ == '__main__':
    main()
