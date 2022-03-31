
import logging
from .cityscapes import build_cityloader
from .camelyon16 import build_camloader
from .hubmap import build_hubmaploader

logger = logging.getLogger('global')

def get_loader(cfg, splits=['train', 'val']):
    loaders = []
    cfg_dataset = cfg['dataset']
    dataset_name = cfg_dataset['type']
    build_fn = None
    if dataset_name == 'cityscapes':
        build_fn = build_cityloader
    elif dataset_name == 'camelyon':
        build_fn = build_camloader
    elif dataset_name == 'hubmap':
        build_fn = build_hubmaploader
    else:
        raise NotImplementedError("dataset type {} is not supported".format(cfg_dataset))
    for split in splits:
        loaders += [build_fn(split, cfg)]
    logger.info('Get loader Done...')
    
    if len(loaders) > 1:
        return tuple(loaders)
    return loaders[0]
