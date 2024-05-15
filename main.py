# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model, build_mtl_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper

from mtl_loss_schemes import MultiTaskLoss, get_loss
from evaluation.evaluate_utils import PerformanceMeter, get_output


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--ckpt-freq', type=int, default=5,
                        help="checkpoint saving frequency")
    parser.add_argument('--eval-freq', type=int, default=5,
                        help="model evaluation frequency")
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of epochs to train")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--name', type=str, help='override model name')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    # distributed training
    parser.add_argument("--local-rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm',
                        action='store_true', help='Use fused layernorm.')
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # MTL Config
    parser.add_argument('--tasks', type=str, default='depth',
                        help='Enable adaptive MTL, defaults to depth est.')
    parser.add_argument(
        '--nyud', type=str, help='specify the path to load NYUD, replaces --data-path')
    parser.add_argument(
        '--pascal', type=str, help='specify the path to load PASCAL, replaces --data-path and --nyud')
    parser.add_argument('--eval-training-freq', type=int,
                        help='calculate performance score on the training dataset')
    parser.add_argument('--resume-backbone',
                        help='resume checkpoint into the backbone')
    parser.add_argument('--freeze-backbone',
                        action='store_true', help='Freeze encoder layers.')

    parser.add_argument('--skip_initial_validation', action='store_true',
                        help='Skip running validation at the start')
    parser.add_argument('--compute_flops', action='store_true',
                        help='Compute Flops while evaluating.')
    parser.add_argument('--skip_decoder', action='store_true',
                        help='Skip loading decoder head weights')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    teacher = None
    model = build_model(config)
    model = build_mtl_model(model, config)

    model.cuda()

    model_without_ddp = model

    optimizer = build_optimizer(config, model)

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(
            data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.MTL:
        loss_ft = torch.nn.ModuleDict(
            {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
        all_loss_weights = {
            'depth': 1.0,
            'semseg': 1.0,
            'human_parts': 2.0,
            'sal': 5.0,
            'edge': 50.0,
            'normals': 10.0,
        }
        loss_weights = {}
        for t in config.TASKS:
            loss_weights[t] = all_loss_weights[t]

        criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)

        if not config.SKIP_INITIAL_EVAL:
            validate(config, data_loader_val, model)
        if config.EVAL_MODE:
            return

    if config.MODEL.RESUME_BACKBONE:
        max_accuracy = load_checkpoint(
            config, model_without_ddp.backbone, optimizer, lr_scheduler, loss_scaler, logger, True)
        if not config.SKIP_INITIAL_EVAL:
            validate(config, data_loader_val, model)
        if config.EVAL_MODE:
            return

    if config.EVAL_MODE:
        validate(config, data_loader_val, model)
        return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if not config.SKIP_INITIAL_EVAL:
            acc1, _, _ = validate(config, data_loader_val, model)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.EPOCHS):
        if not config.MTL:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, teacher=teacher)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)
        if epoch % config.EVAL_FREQ == 0:
            if config.MTL:
                validate(config, data_loader_val, model)
            else:
                acc1, _, _ = validate(config, data_loader_val, model)
                max_accuracy = max(max_accuracy, acc1)

    # final eval
    validate(config, data_loader_val, model)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, task=None, teacher=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    model_loss_meter = AverageMeter()
    efficiency_loss_meter = AverageMeter()

    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)

    start = time.time()
    end = time.time()

    for idx, batch in enumerate(data_loader):
        samples = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(
            non_blocking=True) for task in config.TASKS}

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            loss, loss_dict = criterion(outputs, targets)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if not config.MTL:
            loss_meter.update(loss.item(), targets.size(0))
        else:
            loss_meter.update(loss.item())

        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    if config.EVAL_TRAINING is not None and (epoch % config.EVAL_TRAINING == 0):
        print("Training Eval:")
        performance_meter.update(
            {t: get_output(outputs[t], t) for t in config.TASKS}, targets)

        _ = performance_meter.get_score(verbose=True)

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = config.TASKS
    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)
    loss_meter = AverageMeter()

    loss_ft = torch.nn.ModuleDict(
        {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
    all_loss_weights = {
        'depth': 1.0,
        'semseg': 1.0,
        'human_parts': 2.0,
        'sal': 5.0,
        'edge': 50.0,
        'normals': 10.0,
    }
    loss_weights = {}
    for t in config.TASKS:
        loss_weights[t] = all_loss_weights[t]
    criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    model.eval()
    num_val_points = 0
    logger.info("Start eval")
    start = time.time()
    outputs_batch = {task: [] for task in config.TASKS}
    labels_batch = {task: [] for task in config.TASKS}
    for i, batch in enumerate(data_loader):
        # Forward pass
        logger.debug(f"Image ID = {batch['meta']['image']}")
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

        output = model(images)
        for t in output:
            outputs_batch[t].append(output[t])
            labels_batch[t].append(targets[t])

        num_val_points += 1

        if len(outputs_batch[config.TASKS[0]]) > 0 and len(outputs_batch[config.TASKS[0]]) % config.DATA.BATCH_SIZE == 0:
            output_batch_tesnor = {task: torch.cat(
                task_batch, dim=0) for task, task_batch in outputs_batch.items()}
            label_batch_tesnor = {task: torch.cat(
                task_batch, dim=0) for task, task_batch in labels_batch.items()}

            # Measure performance
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                loss, loss_dict = criterion(
                    output_batch_tesnor, label_batch_tesnor)
                loss_meter.update(loss.item())
            processed_output = {t: get_output(
                output_batch_tesnor[t], t) for t in tasks}
            performance_meter.update(processed_output, label_batch_tesnor)

            outputs_batch = {task: [] for task in config.TASKS}
            labels_batch = {task: [] for task in config.TASKS}

    if len(outputs_batch[config.TASKS[0]]) > 0:
        output_batch_tesnor = {task: torch.cat(
            task_batch, dim=0) for task, task_batch in outputs_batch.items()}
        label_batch_tesnor = {task: torch.cat(
            task_batch, dim=0) for task, task_batch in labels_batch.items()}

        # Measure performance
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            loss, loss_dict = criterion(
                output_batch_tesnor, label_batch_tesnor)
            loss_meter.update(loss.item())
        processed_output = {t: get_output(
            output_batch_tesnor[t], t) for t in tasks}
        performance_meter.update(processed_output, label_batch_tesnor)

        # save_imgs_mtl(images, targets, processed_output, "adamtl", id=batch['meta']['image'][0])

    logger.info(f"val loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t")

    eval_results = performance_meter.get_score(verbose=True)
    epoch_time = time.time() - start
    logger.info(
        f"eval takes {datetime.timedelta(seconds=int(epoch_time))}")

    return eval_results


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
