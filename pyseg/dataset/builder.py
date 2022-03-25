
import logging
from .cityscapes import build_cityloader
from .camelyon16 import build_camloader

logger = logging.getLogger('global')

def get_loader(cfg):
    cfg_dataset = cfg['dataset']
    dataset_name = cfg_dataset['type']
    if dataset_name == 'cityscapes':
        trainloader = build_cityloader('train', cfg)
        valloader = build_cityloader('val', cfg)
    elif dataset_name == 'camelyon':
        trainloader = build_camloader('train', cfg)
        valloader = build_camloader('val', cfg)
    else:
        raise NotImplementedError("dataset type {} is not supported".format(cfg_dataset))
    logger.info('Get loader Done...')
 
    return trainloader, valloader
