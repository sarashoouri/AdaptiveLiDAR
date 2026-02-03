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

# train query and results save
import pickle
import math
import copy
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
cfg='./Codes/mmdetection3d/CMT/projects/configs/fusion/nuscenes_policy_hi.py'
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
cfg.data.samples_per_gpu=2
cfg.data.workers_per_gpu=1
args.launcher='pytorch'
args.diff_seed=True
args.dist_on_itp=True
args.auto_resume=True
args.gpus=1
args.autoscale_lr=0.0001
args.resume_from='./Codes/mmdetection3d/work_dirs/checkpoint/voxel0100_r50_800x320_epoch20.pth'
args.config='./Codes/mmdetection3d/CMT/projects/configs/fusion/nuscenes_policy_hi.py'
args.checkpoint=args.resume_from
setup_multi_processes(cfg)

 # set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
if args.work_dir is not None:
        cfg.work_dir = args.work_dir
elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
if args.launcher == 'none':
        distributed = False
else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
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
    # log env info
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
datasets = [build_dataset(cfg.data.train)]
#only for this generation
if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.val.pipeline
        else:
            val_dataset.pipeline = cfg.data.val.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calc ulation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
logger = get_mmdet_root_logger(log_level=cfg.log_level)
dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    #print(dataset)
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
            cfg.data.samples_per_gpu if i==0 else 2,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffle=False if i==1 else True, 
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
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
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
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')
checkpoint_path='./Codes/mmdetection3d/work_dirs/checkpoints_nuscene_hi/retarining_hi_72.7.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Adjust map_location if needed
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
from collections import OrderedDict
new_state_dict = model.state_dict()
# 4. Extract and slice task_heads parameters
sliced_task_heads = OrderedDict()
for key, value in state_dict.items():
    module_key = 'module.' + key
    if module_key in new_state_dict:
        #print(module_key)
        print("from our trained")
        print(module_key)
        sliced_value = value.clone()
        if sliced_value.size() == new_state_dict[module_key].size():
            sliced_task_heads[module_key] = sliced_value
            #print(f"Sliced and added parameter: {key} with shape {sliced_value.size()}")
        else:
            print(f"Shape mismatch for {key}: sliced shape {sliced_value.size()} vs expected {new_state_dict[module_key].size()}. Skipping.")

# 5. Update and load the sliced task_heads into the new model's state_dict
new_state_dict.update(sliced_task_heads)
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
# Print missing and unexpected keys for debugging
if missing_keys:
    print("New Missing keys:", missing_keys)
if unexpected_keys:
    print("Unexpected keys:", unexpected_keys)                

work_dir = './Codes/mmdetection3d/work_dirs/checkpoint/'
runner.logger.info('Start running, host: %s, work_dir: %s',get_host_info(), work_dir)
runner.logger.info('Hooks will be executed in the following order:\n%s',
                         runner.get_hook_info())
runner.logger.info('workflow: %s, max: %d epochs', cfg.workflow, runner._max_epochs)
runner.call_hook('before_run')
missing_grad = []
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is None:
        missing_grad.append(name)

if missing_grad:
    print("The following parameters did not receive gradients:")
    for name in missing_grad:
        print(name)
else:
    print("All parameters received gradients.")
for i, flow in enumerate(cfg.workflow):
            mode, epochs = flow
            if mode == 'train':
                runner._max_iters = runner._max_epochs * len(data_loaders[i])
                break
dataset = data_loaders[1].dataset
print("Before:", dataset.eval_detection_configs.max_boxes_per_sample)  # → 500
dataset.eval_detection_configs.max_boxes_per_sample = 900
print("After: ", dataset.eval_detection_configs.max_boxes_per_sample)  # → 900
gpu_collect=True
tmpdir=None
performance=0
filename_tmpl='retarining_hi.pth'
acc_loss = 0.0
acc_loss_bbox = 0.0
acc_loss_cls = 0.0
acc_count = 0
acc_loss_mask =0
acc_cvar=0
# -------------------- Main Training Loop --------------------
while runner.epoch < runner._max_epochs:
    rank, world_size = get_dist_info()
    if rank == 0:
        print("EPOCH:", runner.epoch)
        print("Training...")
    model.train()
    runner.mode = 'train'
    runner.data_loader = data_loaders[0]
    runner._max_iters = runner._max_epochs * len(runner.data_loader)
    runner.call_hook('before_train_epoch')
    time.sleep(2)
    
    # Update temperature for your gumbel sampler
    current_temp = update_temperature(runner.epoch)
    runner.model.module.pts_bbox_head.mask_generator.mask_generator.temperature = current_temp
    acc_loss = 0.0
    acc_loss_bbox = 0.0
    acc_loss_cls = 0.0
    acc_loss_mask = 0.0
    acc_loss_mask_sparse=0.0
    acc_count = 0
    acc_distill=0.0
    model.module.pts_bbox_head.mask_generator.mask_generator.hard=True

    # Training loop for current epoch.
    for i, data_batch in enumerate(runner.data_loader):
        
        runner.data_batch = data_batch
        runner._inner_iter = i
        runner.call_hook('before_train_iter')
        #if runner.epoch==0 and i<=1486:
           # break
        #    continue
        # Prepare training input. Note the index [0] if data_batch items are lists.
        data_new = {
            'img_metas': data_batch['img_metas'][0],
            'queries': data_batch['queries'][0],
            'pred_results': data_batch['pred_results'][0],
            'gt_bboxes_3d': data_batch['gt_bboxes_3d'][0],
            'gt_labels_3d': data_batch['gt_labels_3d'][0],
            'range_image': data_batch['range_image'][0],
            'points': data_batch['points'][0],
            'img': data_batch['img'][0],
        }
        
        outputs = runner.model.train_step(data_new, runner.optimizer)
        runner.outputs = outputs
        if i % 200 == 0:
            torch.cuda.empty_cache()
        if rank == 0:
            print(f"Iteration {i + 1}")
            log_vars = outputs.get('log_vars', {})
            loss_value = log_vars.get('loss', 0.0)
            loss_bbox = log_vars.get('loss_bbox', 0.0)
            loss_cls = log_vars.get('loss_cls', 0.0)
            loss_mask = log_vars.get('loss_mask', 0.0)
            if log_vars.get('loss_distill', 0.0):
                loss_distill=log_vars.get('loss_distill', 0.0)
            else:
                loss_distill=0
            if log_vars.get('loss_cvar', 0.0):
                loss_cvar=log_vars.get('loss_cvar', 0.0)
            else:
                loss_cvar=0
            if log_vars.get('loss_mask_sparse', 0.0):
                loss_mask_sparse=log_vars.get('loss_mask_sparse', 0.0)
            else:
                loss_mask_sparse=0
    
            # Accumulate losses for logging.
            acc_loss += loss_value
            acc_loss_bbox += loss_bbox
            acc_loss_cls += loss_cls
            acc_loss_mask += loss_mask
            acc_loss_mask_sparse +=loss_mask_sparse
            acc_distill += loss_distill
            acc_cvar+=loss_cvar
            
            acc_count += 1
    
            # Print losses every 50 iterations.
            if i % 50 == 0 and acc_count > 0:
                avg_loss = acc_loss / acc_count
                avg_loss_bbox = acc_loss_bbox / acc_count
                avg_loss_cls = acc_loss_cls / acc_count
                avg_loss_mask = acc_loss_mask / acc_count
                avg_loss_mask_sparse= acc_loss_mask_sparse / acc_count
                avg_distill= acc_distill / acc_count
                avg_cvar=acc_cvar/acc_count
                avg_combined = (avg_loss_cls + avg_loss_bbox) / 2
    
                print(f"Epoch {runner.epoch}: Iteration {i + 1}:")
                print(f"Avg Loss: {avg_loss:.4f}")
                print(f"Avg BBox Loss: {avg_loss_bbox:.4f}")
                print(f"Avg Cls Loss: {avg_loss_cls:.4f}")
                print(f"Avg Mask Loss: {avg_loss_mask:.5f}")
                print(f"Avg Mask Sparse: {avg_loss_mask_sparse:.5f}")
                print(f"Avg Combined: {avg_combined:.4f}")
                print(f"CVar: {avg_cvar:.4f}")
                print(f"Avg Distilliation {avg_distill:.4f}")
    
                # Reset accumulators.
                acc_loss = 0.0
                acc_loss_bbox = 0.0
                acc_loss_cls = 0.0
                acc_loss_mask = 0.0
                acc_loss_mask_sparse=0.0
                acc_distill=0.0
                acc_cvar=0
                acc_count = 0

        runner.call_hook('after_train_iter')
        del runner.data_batch
        runner._iter += 1
    
    runner.call_hook('after_train_epoch')
    
    if runner.epoch % 2 == 0:
        print("entering the dataset of evaluation")
        #model.module.pts_bbox_head.mask_generator.mask_generator.hard=True
        model.eval()
        model.module.pts_bbox_head.mask_generator.mask_generator.hard=False
        model.module.pts_bbox_head.mask_generator.mask_generator.temperature=2.2
        results = []
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        time.sleep(2)

        for i, data_batch1 in enumerate(data_loaders[1]):
            if rank == 0:
                print(f"Processing batch {i}")
            with torch.no_grad():
                data_new = {
                    'img_metas': data_batch1['img_metas'],
                    'queries': data_batch1['queries'],
                    'pred_results': data_batch1['pred_results'],
                    'range_image': data_batch1['range_image'],
                    'points': data_batch1['points'],
                    'img': data_batch1['img'],
                }

                result, queries_new,_= model(**data_new, return_loss=False, rescale=False)
            results.extend(result)

        # Now synchronize all ranks before result collection
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        dataset_length = len(dataset)
        if gpu_collect:
            # This will gather predictions from all ranks onto rank 0
            outputs = collect_results_gpu(results, 6019)
        else:
            outputs = collect_results_cpu(results, 6019, tmpdir)

        args.out = './Codes/mmdetection3d/CMT/work_dirs/nus_sweep_ResNet50.pkl'
        args.eval = True

        # Broadcast outputs so all ranks have the same predictions
        if torch.distributed.is_initialized():
            outputs_list = [outputs if outputs is not None else []]
            torch.distributed.broadcast_object_list(outputs_list, src=0)
            outputs = outputs_list[0]
        # Print overall mask sparsity information
        
        if args.eval:
            kwargs = {} if args.eval_options is None else args.eval_options
            # strip EvalHook args and force metric=True
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval','tmpdir','start','gpu_collect','save_best','rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            kwarg={}
            kwarg['metric']=True
            val_result = dataset.evaluate(outputs, **kwarg)

            # Only rank 0 prints and saves results
            if rank == 0:
                print(val_result)
                
                val_res=val_result['pts_bbox_NuScenes/mAP']
                print(val_res)
                
                if val_res > performance:
                
                    runner.save_checkpoint(work_dir, filename_tmpl)
                    performance = val_res
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    runner._epoch += 1

time.sleep(1)  # wait for some hooks like loggers to finish
runner.call_hook('after_run')

