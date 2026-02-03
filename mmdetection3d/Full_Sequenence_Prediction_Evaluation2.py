import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

import torch.distributed as dist
import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import collect_env, get_root_logger
from mmcv.parallel import scatter
if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from os import path as osp
import time

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmseg import __version__ as mmseg_version

from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
)
from mmdet3d.utils import find_latest_checkpoint
from mmdet.core import DistEvalHook as MMDET_DistEvalHook, EvalHook as MMDET_EvalHook
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet.utils import get_root_logger as get_mmdet_root_logger
from mmseg.core import DistEvalHook as MMSEG_DistEvalHook, EvalHook as MMSEG_EvalHook
from mmseg.datasets import build_dataloader as build_mmseg_dataloader
from mmseg.utils import get_root_logger as get_mmseg_root_logger
from mmdet.apis.test import *
from mmcv.runner.utils import get_host_info
import sys
import numpy as np
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_bilateral
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage import filters, morphology
from collections import deque
import pickle
import math
import copy
from collections import OrderedDict
def update_temperature(ep):
    """
    Compute the current temperature based on the epoch.
    
    During the warmup stage (ep < args.epochs_warmup), the temperature remains at args.temp_init.
    Once warmup is done (ep >= args.epochs_warmup), the temperature decays exponentially:
    
        T = temp_init * exp( -eta * (ep - epochs_warmup) )
    
    Args:
        ep (int): Current epoch.
        args: An object (or argparse.Namespace) that contains:
              - epochs_warmup: number of warmup epochs.
              - temp_init: initial temperature (e.g. 5.0).
              - eta: exponential decay rate (e.g. 0.05).
    
    Returns:
        float: The current temperature.
    """
    epochs_warmup=1
    temp_init=1
    eta=0.05
    #if ep < epochs_warmup:
    #    temp = temp_init
    #else:
    #    temp = temp_init * math.exp(-eta * (ep - epochs_warmup))
    temp=1.5
    return temp
def hysteresis_mask_single(
    p: np.ndarray,
    low_frac: float = 0.5,
    min_size: int = 64,
    area_threshold: int = 64
) -> np.ndarray:
    """
    Single-image hysteresis thresholding on a probability map p [H,W].
    Returns uint8 binary mask [H,W].
    """
    # high and low thresholds
    th_high = filters.threshold_otsu(p)
    th_low  = th_high * low_frac

    # seeds and support
    seeds       = (p >= th_high).astype(np.uint8)
    mask_region = (p >= th_low).astype(np.uint8)

    # grow seeds within support
    grown = morphology.reconstruction(seed=seeds, mask=mask_region, method='dilation')

    # cleanup
    clean = morphology.remove_small_objects(grown.astype(bool), min_size=min_size)
    clean = morphology.remove_small_holes(clean, area_threshold=area_threshold)

    return clean.astype(np.uint8)

def batch_hysteresis_mask(
    mask_probs: torch.Tensor,
    low_frac: float = 0.5,
    min_size: int = 64,
    area_threshold: int = 64
) -> torch.Tensor:
    """
    Args:
      mask_probs:     [B,2,H,W] float tensor of soft probabilities.
      low_frac:       fraction of high threshold for low threshold.
      min_size:       min connected-component size.
      area_threshold: min hole size to fill.

    Returns:
      [B,2,H,W] tensor of 0/1 masks, same dtype & device.
    """
    B, C, H, W = mask_probs.shape
    device = mask_probs.device
    dtype  = mask_probs.dtype

    output = torch.zeros((B, C, H, W), dtype=dtype, device=device)

    for b in range(B):
        # get class-1 map as numpy
        p = mask_probs[b, 1].detach().cpu().numpy()
        # single-image hysteresis
        M = hysteresis_mask_single(p, low_frac, min_size, area_threshold)
        # back to tensor
        M_t = torch.from_numpy(M).to(device=device, dtype=dtype)
        # fill channels
        output[b, 1] = M_t
        output[b, 0] = 1.0 - M_t

    return output


def geodesic_distance(seeds: np.ndarray, support: np.ndarray) -> np.ndarray:
    """
    Compute grid-graph geodesic distance within a support region.
    seeds:   bool array [H,W], True where p >= T_high
    support: bool array [H,W], True where p >= T_low
    Returns: float array [H,W], distance (inf outside support)
    """
    H, W = support.shape
    D = np.full((H, W), np.inf, dtype=np.float32)
    q = deque()
    ys, xs = np.nonzero(seeds)
    for y, x in zip(ys, xs):
        D[y, x] = 0.0
        q.append((y, x))
    nbrs = [(-1,0), (1,0), (0,-1), (0,1)]
    while q:
        y, x = q.popleft()
        d0 = D[y, x]
        for dy, dx in nbrs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and support[ny, nx]:
                if D[ny, nx] > d0 + 1:
                    D[ny, nx] = d0 + 1
                    q.append((ny, nx))
    return D

def smart_geodesic_mask_single(
    p: np.ndarray,
    alpha: float = 0.5,
    lam: float = 0.1,
    min_area: int = 64,
    sigma_color: float = 0.1,
    sigma_spatial: float = 3.0
) -> np.ndarray:
    """
    Single-image dynamic-threshold geodesic region-growing.
    p:           numpy array [H,W] of soft class-1 probabilities
    Returns:     uint8 array [H,W] of 0/1 mask
    """
    # 1) bilateral smoothing
    p_s = denoise_bilateral(
        p,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        multichannel=False
    )
    # 2) thresholds
    T_high = threshold_otsu(p_s)
    T_low  = alpha * T_high
    # 3) seeds & support
    seeds   = p_s >= T_high
    support = p_s >= T_low
    # 4) geodesic distance
    D = geodesic_distance(seeds, support)
    # 5) dynamic threshold map
    T_map = T_high - lam * D
    # 6) initial mask
    M = (p_s >= T_map) & support
    # 7) cleanup
    M = remove_small_objects(M, min_size=min_area)
    M = remove_small_holes(M, area_threshold=min_area)
    return M.astype(np.uint8)

def batch_smart_geodesic_mask(
    mask_probs: torch.Tensor,
    alpha: float = 0.5,
    lam: float = 0.1,
    min_area: int = 64,
    sigma_color: float = 0.1,
    sigma_spatial: float = 3.0
) -> torch.Tensor:
    """
    Apply dynamic-geodesic mask to each sample in [B,2,H,W] tensor.
    Returns: [B,2,H,W] binary tensor (0/1) same dtype & device.
    """
    B, _, H, W = mask_probs.shape
    device = mask_probs.device
    dtype  = mask_probs.dtype
    output = torch.zeros((B,2,H,W), device=device, dtype=dtype)
    for b in range(B):
        # extract class-1 probability map as numpy
        p = mask_probs[b,1].detach().cpu().numpy()
        # compute H×W binary mask
        M = smart_geodesic_mask_single(
            p, alpha=alpha, lam=lam,
            min_area=min_area,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial
        )
        # back to tensor
        M_t = torch.from_numpy(M).to(device=device, dtype=dtype)
        output[b,1] = M_t
        output[b,0] = 1.0 - M_t
    return output
def move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [move_to_device(item, device) for item in x]
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x
def compute_sparsity(binary_masks: torch.Tensor) -> torch.Tensor:
    """
    Args:
        binary_masks: torch.Tensor of shape [B, 2, H, W] with values 0 or 1.
    Returns:
        sparsity_per_sample: torch.Tensor of shape [B], where each entry
                             is (# ones in class-1) / (H*W).
    """
    B, _, H, W = binary_masks.shape
    class1 = binary_masks[:, 1, :, :]                   # [B, H, W]
    ones_per_sample = class1.reshape(B, -1).sum(dim=1)   # [B]
    sparsity_per_sample = ones_per_sample.float() / (H * W)
    return sparsity_per_sample
import torch.nn.functional as F
def fill_class1_mask_prob(mask: torch.Tensor,
                          kernel_size: int = 31,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    Takes a soft mask [B,2,H,W] and returns a locally-smoothed soft mask
    of the same shape, then rescales so that the max in channel 1 is 1.

    Args:
      mask:        Tensor [B,2,H,W], soft probabilities in [0,1]
      kernel_size: smoothing window size
      eps:         small constant to avoid div0

    Returns:
      Tensor [B,2,H,W] where channel1 has been box-filtered and then
      divided by its per-sample max (clamped to [0,1]), and channel0=1-channel1.
    """
    # 1) extract class-1 probabilities [B,1,H,W]
    class1 = mask[:, 1:2, :, :]

    # 2) build normalized box filter
    k = kernel_size
    kernel = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype) / (k*k)
    pad = k // 2

    # 3) convolve → local average probability
    prob1 = F.conv2d(class1, kernel, padding=pad)  # [B,1,H,W]

    # 4) rescale so that max per-sample = 1
    #    compute per-sample max over spatial dims
    max_vals = prob1.amax(dim=[2,3], keepdim=True)  # [B,1,1,1]
    prob1 = prob1 / (max_vals + eps)

    # 5) clamp tiny numerical drift
    prob1 = prob1.clamp(0.0, 1.0)

    # 6) re-assemble two channels
    prob0 = 1.0 - prob1
    return torch.cat([prob0, prob1], dim=1)

def batch_soft_bernoulli_mask(
    mask_probs: torch.Tensor,
    offset: float = 0.0,
    quantize_levels: list = None
) -> torch.Tensor:
    """
    Applies advisor’s "soft-masking": uses each block’s probability
    as a sampling rate to randomly keep/drop pixels.

    Args:
        mask_probs:      [B,2,H,W] tensor of block-wise soft probabilities.
                         channel 1 is the keep-confidence p ∈ [0,1].
        offset:          minimum sampling probability (broadcast to all p).
        quantize_levels: optional list of floats (e.g. [0.0625,0.125,0.25,0.5,1.0])
                         to which p is snapped before sampling.

    Returns:
        mask: [B,2,H,W] binary mask tensor (0 or 1), same dtype & device.
    """
    B, C, H, W = mask_probs.shape
    device = mask_probs.device
    dtype = mask_probs.dtype

    # 1) extract and clamp probabilities
    p = mask_probs[:, 1, :, :].clamp(min=offset)  # [B,H,W]

    # 2) optional quantization
    if quantize_levels is not None:
        levels = torch.tensor(quantize_levels, device=device, dtype=dtype)  # [L]
        # compute |p - level| and pick nearest level
        # p.unsqueeze(-1) -> [B,H,W,1], broadcast against [L]
        idx = (p.unsqueeze(-1) - levels).abs().argmin(dim=-1)  # [B,H,W]
        p = levels[idx]  # [B,H,W]

    # 3) sample uniform random and build keep-mask
    rand = torch.rand((B, H, W), device=device, dtype=dtype)
    keep = (rand < p).to(dtype).unsqueeze(1)  # [B,1,H,W]

    # 4) assemble two-channel binary mask
    mask = torch.cat([1.0 - keep, keep], dim=1)  # [B,2,H,W]
    return mask

def geodesic_distance(seeds: np.ndarray, support: np.ndarray) -> np.ndarray:
    H, W = support.shape
    D = np.full((H, W), np.inf, dtype=np.float32)
    q = deque()
    for y, x in zip(*np.nonzero(seeds)):
        D[y, x] = 0.0
        q.append((y, x))
    nbrs = [(-1,0), (1,0), (0,-1), (0,1)]
    while q:
        y, x = q.popleft()
        d0 = D[y, x]
        for dy, dx in nbrs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and support[ny, nx] and D[ny, nx] > d0 + 1:
                D[ny, nx] = d0 + 1
                q.append((ny, nx))
    return D

def smart_geodesic_mask_soft(
    mask_probs: torch.Tensor,
    alpha: float = 0.5,
    lam: float = 0.1,
    sigma_color: float = 0.1,
    sigma_spatial: float = 3.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Continuous (soft) dynamic-threshold geodesic mask.
    Non-support pixels are explicitly set to 0 (no NaNs).
    """
    # 1) extract & smooth class-1 prob
    p = mask_probs[0,1].cpu().numpy()
    p_s = denoise_bilateral(
        p,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        multichannel=False
    )

    # 2) thresholds & support
    T_high = threshold_otsu(p_s)
    seeds   = p_s >= T_high
    support = p_s >= alpha * T_high

    # 3) compute geodesic distance & threshold map
    D     = geodesic_distance(seeds, support)
    T_map = T_high - lam * D

    # 4) linear ramp: only inside support, zeros elsewhere
    M_cont = np.zeros_like(p_s, dtype=np.float32)
    denom = 1.0 - T_map
    # compute ramp for valid pixels
    valid = support
    M_cont[valid] = (p_s[valid] - T_map[valid]) / (denom[valid] + eps)

    # replace any NaN/inf with 0
    M_cont = np.nan_to_num(M_cont, nan=0.0, posinf=1.0, neginf=0.0)

    # clip into [0,1]
    M_cont = np.clip(M_cont, 0.0, 1.0)

    # 5) back to tensor and two-channel format
    M_t = torch.from_numpy(M_cont).to(mask_probs.device).unsqueeze(0).unsqueeze(0)
    out = torch.zeros_like(mask_probs)
    out[0,1] = M_t[0,0]
    out[0,0] = 1.0 - M_t[0,0]
    return out
def smart_geodesic_mask_soft_batch(
    mask_probs: torch.Tensor,
    alpha: float = 0.5,
    lam: float = 0.1,
    sigma_color: float = 0.1,
    sigma_spatial: float = 3.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch version of continuous geodesic mask.
    Input:  mask_probs [B,2,H,W]
    Output: [B,2,H,W], channel1 ∈ [0,1], channel0 = 1-channel1
    """
    B, C, H, W = mask_probs.shape
    device = mask_probs.device
    out = torch.zeros_like(mask_probs)

    for b in range(B):
        # 1) smooth class-1 probability
        p = mask_probs[b,1].detach().cpu().numpy()
        p_s = denoise_bilateral(
            p,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            multichannel=False
        )

        # 2) thresholds & support
        T_high = threshold_otsu(p_s)
        seeds   = p_s >= T_high
        support = p_s >= alpha * T_high

        # 3) distance & threshold map
        D     = geodesic_distance(seeds, support)
        T_map = T_high - lam * D

        # 4) linear ramp within support, zeros outside
        M_cont = np.zeros_like(p_s, dtype=np.float32)
        valid = support
        denom = 1.0 - T_map
        M_cont[valid] = (p_s[valid] - T_map[valid]) / (denom[valid] + eps)
        # Replace NaN/Inf, clip
        M_cont = np.nan_to_num(M_cont, nan=0.0, posinf=1.0, neginf=0.0)
        M_cont = np.clip(M_cont, 0.0, 1.0)

        # 5) write back to tensor
        M_t = torch.from_numpy(M_cont).to(device=device, dtype=mask_probs.dtype)
        out[b,1] = M_t
        out[b,0] = 1.0 - M_t

    return out

def get_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        default=True,
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        default=True,
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        default=True,
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        default=True,
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    
    return parser
sys.argv = ['--config-file','my_config']
args = get_args().parse_args("")
cfg='./Codes/mmdetection3d/CMT/projects/configs/fusion/cmt_lyft_policy.py'
cfg = Config.fromfile(cfg)
if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)
if cfg.get('custom_imports', None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg['custom_imports'])
if hasattr(cfg, 'plugin'):
        sys.path.insert(0, './Codes/mmdetection3d/CMT/')
        #print(_module_path)
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

cfg = compat_cfg(cfg)
cfg.data.samples_per_gpu=1
cfg.data.workers_per_gpu=1
args.launcher='pytorch'
args.diff_seed=True
args.dist_on_itp=True
args.auto_resume=True
args.gpus=1
args.autoscale_lr=0.0001
args.resume_from='./Codes/mmdetection3d/work_dirs/checkpoint/finetuned_lyft_sweep_VoVNet_Mine_voxel_0125_version2.pth'
args.config='./Codes/mmdetection3d/CMT/projects/configs/fusion/cmt_lyft_policy.py'
args.checkpoint=args.resume_from
setup_multi_processes(cfg)
if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
if args.work_dir is not None:
        cfg.work_dir = args.work_dir
elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
if args.resume_from is not None:
        cfg.resume_from = args.resume_from
if args.auto_resume:
        cfg.auto_resume = args.auto_resume
if args.gpus is not None:
    cfg.gpu_ids = range(1)
if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
if args.launcher == 'none':
        distributed = False
else:
        distributed = True
        print("trying to init")
        init_dist(args.launcher, **cfg.dist_params)
        print(" hello init")
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
print("cfg.gpu_ids:")
print(cfg.gpu_ids)
if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
else:
        logger_name = 'mmdet'
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name=logger_name)
meta = dict()
env_info_dict = collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +dash_line)
meta['env_info'] = env_info
meta['config'] = cfg.pretty_text
logger.info(f'Distributed training: {distributed}')
logger.info(f'Config:\n{cfg.pretty_text}')
seed = init_random_seed(args.seed)
seed = seed + dist.get_rank() if args.diff_seed else seed
logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
set_random_seed(seed, deterministic=args.deterministic)
cfg.seed = seed
meta['seed'] = seed
meta['exp_name'] = osp.basename(args.config)
model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
model.init_weights() 
logger.info(f'Model:\n{model}')
cfg.data.train.test_mode=False
datasets = [build_dataset(cfg.data.train)]
if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.val.pipeline
        else:
            val_dataset.pipeline = cfg.data.val.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
model.CLASSES = datasets[0].CLASSES
logger = get_mmdet_root_logger(log_level=cfg.log_level)
dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
data_loaders = [
        build_mmdet_dataloader(
            dataset[i],
            cfg.data.samples_per_gpu if i==0 else 1,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffle=False, 
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for i in range(len(dataset))
]
if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
optimizer = build_optimizer(model, cfg.optimizer)
if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
runner.timestamp = timestamp
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
else:
        optimizer_config = cfg.optimizer_config
runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())
runner.load_checkpoint(args.checkpoint)
validate=True
if validate:
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmdet_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = MMDET_DistEvalHook if distributed else MMDET_EvalHook

        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')
checkpoint_path="./Codes/mmdetection3d/work_dirs/checkpoint/Differentiable_Policy_lowh_4_loww_16_five_frame_loss_weight10_finetune_hysteris_bernouli_start_temp_init_1.3_frac_0.1_47.09.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Adjust map_location if needed
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
new_state_dict = model.state_dict()
# 4. Extract and slice task_heads parameters
sliced_task_heads = OrderedDict()
for key, value in state_dict.items():
    module_key = 'module.' + key
    if module_key in new_state_dict:
        #print(module_key)
        sliced_value = value.clone()
        if sliced_value.size() == new_state_dict[module_key].size():
            sliced_task_heads[module_key] = sliced_value
            #print(f"Sliced and added parameter: {key} with shape {sliced_value.size()}")
        else:
            print(f"Shape mismatch for {key}: sliced shape {sliced_value.size()} vs expected {new_state_dict[module_key].size()}. Skipping.")
    else:
        print(key)
# 5. Update and load the sliced task_heads into the new model's state_dict
new_state_dict.update(sliced_task_heads)
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
# Print missing and unexpected keys for debugging
if missing_keys:
    print("Missing keys:", missing_keys)
if unexpected_keys:
    print("Unexpected keys:", unexpected_keys)  

dataset = data_loaders[1]
gpu_collect=True
tmpdir=None
performance=0

for i, flow in enumerate(cfg.workflow):
            mode, epochs = flow
            if mode == 'train':
                runner._max_iters = runner._max_epochs * len(data_loaders[i])
                break
work_dir = './Code/mmdetection3d/work_dirs/checkpoint/'
runner.logger.info('Start running, host: %s, work_dir: %s',get_host_info(), work_dir)
runner.logger.info('Hooks will be executed in the following order:\n%s',
                         runner.get_hook_info())
runner.logger.info('workflow: %s, max: %d epochs', cfg.workflow, runner._max_epochs)
runner.call_hook('before_run')
dataset = data_loaders[1]
gpu_collect=True
tmpdir=None
performance=0
filename_tmpl='Policy_training_nondiff_voxel.pth'
acc_loss = 0.0
acc_loss_bbox = 0.0
acc_loss_cls = 0.0
acc_count = 0
acc_loss_mask =0
gpu_collect     = True
tmpdir          = None
performance     = 0.0
model.eval()
rank, world_size = get_dist_info()
results = []
mask_fractions = []
local_sparsity_sum   = 0.0
local_sparsity_count = 0
# —— Set up
query_hist    = deque(maxlen=4)    # past 4 query tensors (on CPU) #buffer is 4
pred_hist     = deque(maxlen=4)    # past 4 pred dicts   (on CPU)
current_scene = None
device = next(model.parameters()).device #temporal evaluation with one GPU only
if rank == 0:
    prog_bar = mmcv.ProgressBar(len(dataset))

# —— Main eval loop
for i, data_batch in enumerate(data_loaders[1]):
    # 1) scatter to GPU
    data_batch = scatter(data_batch, [torch.cuda.current_device()])[0]

    # 2) detect scene change → clear histories
    scene_token = data_batch['img_metas'][0][0]['scene_token']
    if scene_token != current_scene:
        current_scene = scene_token
        query_hist.clear()
        pred_hist.clear()
    
    full_query_mat = data_batch['queries'][0].to(device)
    cur_query=full_query_mat[:,0,:,:]
    raw_pred  = data_batch['pred_results'][0]        

    # keep CPU copies for history
    cur_query_cpu = cur_query.detach().cpu()
    cur_pred_cpu  = {
        'boxes':  raw_pred['boxes'][:,0,:,:].detach().cpu(),
        'scores': raw_pred['scores'][:,0,:].detach().cpu(),
        'labels': raw_pred['labels'][:,0,:].detach().cpu(),
    }

    # 4) first FOUR frames of each scene → full-ones mask, no history
    if len(query_hist) < 4:
        mask_fractions.append(1.0)               # mask all-ones
       
        # record history for next iteration
        query_hist.append(cur_query_cpu)
        pred_hist.append(cur_pred_cpu)

    # 5) build the 5-element lists:
    #    [cur, t-1, t-2, t-3, t-4]
    hist_q = list(query_hist) 
    queries_list = [cur_query] + [q.to(device) for q in hist_q]
    queries_tensor = torch.stack(queries_list, dim=1)    # shape: [1,5,256,900,6]

    hist_p = list(pred_hist)

    # 1) pull out each field into its own Python list
    boxes_list = []
    scores_list = []
    labels_list = []
    for p in [cur_pred_cpu] + hist_p:
        boxes_field  = 'boxes'   if 'boxes'   in p else 'boxes_3d'
        scores_field = 'scores'  if 'scores'  in p else 'scores_3d'
        labels_field = 'labels'  if 'labels'  in p else 'labels_3d'

        boxes_list.append( p[boxes_field].to(device) )    # shape [1,900,7]
        scores_list.append(p[scores_field].to(device))   # shape [1,900]
        labels_list.append(p[labels_field].to(device))   # shape [1,900]

    # 2) stack along a new time‐dimension (dim=1) → shape [1,5,900,7], etc.
    pred_list = {
        'boxes':  torch.stack(boxes_list,  dim=1),
        'scores': torch.stack(scores_list, dim=1),
        'labels': torch.stack(labels_list, dim=1),
    }

    # 6) pack into data_new and run
    data_new = {
        'frame_index': i,
        'img_metas':    data_batch['img_metas'],
        'range_image':  data_batch['range_image'],
        'points':       data_batch['points'],
        'img':          data_batch['img'],
        'queries':      data_batch['queries'] if len(query_hist)<4 else [queries_tensor] ,
        'pred_results': data_batch['pred_results'] if len(query_hist)<4  else [pred_list],
    }
    with torch.no_grad():
        mg = model.module.pts_bbox_head.mask_generator.mask_generator
        mg.hard        = False
        mg.temperature = 1

        result, frac, queries_new, soft_mask = model(**data_new, return_loss=False, rescale=False)
   
    # 7) collect results & mask fraction
    results.extend(result)
    mask_fractions.append(frac)

    sparsity_per_sample = compute_sparsity(soft_mask)      # shape [B]
    overall_sparsity   = sparsity_per_sample.mean()           # scalar
    print("Overall sparsity:", 1-overall_sparsity.item())
    local_sparsity_sum   += 1-overall_sparsity.item()
    local_sparsity_count += 1
    
    qn_cpu = queries_new[0].permute(1,3,2,0).detach().cpu()
    query_hist.append(qn_cpu)

    bbox = result[0]['pts_bbox']

    new_pred = {
        'boxes':  bbox['boxes_3d'].tensor.detach().cpu().unsqueeze(0),
        'scores': bbox['scores_3d'].detach().cpu().unsqueeze(0),
        'labels': bbox['labels_3d'].cpu().unsqueeze(0),
    }

    pred_hist.append(new_pred)

    if rank == 0:
        prog_bar.update()

# —— finish up: collect across GPUs and run your evaluation+weighted-mAP exactly as before
if torch.distributed.is_initialized():
    torch.distributed.barrier()
if gpu_collect:
    outputs = collect_results_gpu(results, 3780)
else:
    outputs = collect_results_cpu(results, 3780, tmpdir)

args.out = './Codes/mmdetection3d/CMT/work_dirs/lyft_sweep_ResNet50.pkl'
args.eval = True
if torch.distributed.is_initialized():
    outputs_list = [outputs if outputs is not None else []]
    torch.distributed.broadcast_object_list(outputs_list, src=0)
    outputs = outputs_list[0]
# rank 0 prints & evaluates
if rank == 0:
    avg_frac = sum(mask_fractions) / len(mask_fractions)
    print(f"\nAverage channel-1 mask fraction: {avg_frac:.4f}")

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            #eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'by_epoch']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            kwarg = {'metric': True}

    # run dataset.evaluate → val_result dict
    val_result = dataset.dataset.evaluate(outputs, **eval_kwargs)
    print("\nValidation results:", val_result)
    # compute aggregated mAP (excluding some classes)
    mAP = 0.0
    count = 0
    for k, v in val_result.items():
        if k not in ['pts_bbox_Lyft/emergency_vehicle_AP', 'pts_bbox_Lyft/animal_AP']:
            mAP += v
            count += 1
    agg_map = 100.0 * mAP / count
    print(f"Aggregated mAP: {agg_map:.4f}%")
# final barrier before resuming
#dist.barrier()
sparsity_percent_avg=local_sparsity_sum/local_sparsity_count
print(f"Average Sparsity for one GPU: {sparsity_percent_avg:.5f}%")
# pack into a tensor so we can all_reduce
stat_tensor = torch.tensor(
    [local_sparsity_sum, local_sparsity_count],
    device=next(model.parameters()).device,
    dtype=torch.float64
)
# sum across all ranks
dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
global_sum, global_count = stat_tensor.tolist()
if rank==0:
    raw_dataset = dataset.dataset
    classes     = raw_dataset.CLASSES              # e.g. ['animal','bicycle',...]
    class_counts = {c: 0 for c in classes}
    total_gt     = 0
    for idx in range(len(raw_dataset)):
        ann_info = raw_dataset.get_ann_info(idx)
        # ann_info['labels_3d'] is a 1D array of integer class‐indices
        for label_idx in ann_info['gt_labels_3d']:
            cls_name = classes[label_idx]
            class_counts[cls_name] += 1
            total_gt += 1

    # --- 3. compute weighted mAP ---
    # classes to ignore
    ignore = {'animal', 'emergency_vehicle'}

    filtered_total = sum(
        cnt for cls, cnt in class_counts.items() 
        if cls not in ignore
    )

    weighted_map = 0.0
    for cls_name, count in class_counts.items():
        if cls_name in ignore:
            continue
        ap = val_result.get(f'pts_bbox_Lyft/{cls_name}_AP', 0.0)
        weight = count / filtered_total if filtered_total > 0 else 0.0
        weighted_map += weight * ap

    print(f"Weighted mAP (excluding animal & emergency_vehicle): {100 * weighted_map:.2f}%")
# only rank 0 prints the true global average
if rank == 0 and global_count > 0:
    avg_sparsity = global_sum / global_count
    print(f"\n>>> Global average sparsity across all GPUs: {avg_sparsity:.4f}")
