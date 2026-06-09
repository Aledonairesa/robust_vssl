import os
import sys
import torch
from torch.optim import *
from torchvision.transforms import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import time
import cv2

from models.model import AVENet

from datasets import GetAudioVideoDataset
from opts import get_arguments
from utils.utils import AverageMeter
import xml.etree.ElementTree as ET
from utils.eval_ import Evaluator
from sklearn.metrics import auc
from tqdm import tqdm

from utils.util import vis_heatmap_bbox, tensor2img
from utils.tf_equivariance_loss import TfEquivarianceLoss
import utils.tensorboard_utils as TB
from utils.training_report import record_epoch
from utils.utils import save_checkpoint, AverageMeter, calc_topk_accuracy, Logger, ProgressMeter, neq_load_customized


class SigmoidContrastiveLoss(nn.Module):
    def __init__(self, t_init=1.0, b_init=0.0):
        super(SigmoidContrastiveLoss, self).__init__()
        if t_init <= 0:
            raise ValueError('sigmoid_t_init must be positive')
        self.t = nn.Parameter(torch.log(torch.tensor(float(t_init))))
        self.b = nn.Parameter(torch.tensor(float(b_init)))

    def forward(self, logits, target=None):
        labels = -torch.ones_like(logits)
        labels[:, 0] = 1

        valid = torch.ones_like(logits, dtype=torch.bool)
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        valid[batch_idx, batch_idx + 1] = False

        logits = self.t.exp() * logits + self.b
        loss = F.softplus(-labels * logits)
        return loss[valid].mean()


def get_logits_a2i(logits):
    B = logits.size(0)
    return torch.cat(
        (logits[:, :1], logits[:, 1:1 + B].T, logits[:, 1 + B:]),
        dim=1,
    )


class SymmetricCrossEntropyLoss(nn.Module):
    def forward(self, logits, target=None):
        B = logits.size(0)
        if target is None:
            target = torch.zeros(B, device=logits.device, dtype=torch.long)

        logits_i2a = logits
        logits_a2i = get_logits_a2i(logits)

        return 0.5 * (
            F.cross_entropy(logits_i2a, target) +
            F.cross_entropy(logits_a2i, target)
        )


def select_atp_cu_embeddings(embeddings, image_embedding):
    if image_embedding == 'positive_mask_mean':
        image_emb = embeddings['image_emb_positive_mask_mean']
    elif image_embedding == 'maxpool':
        image_emb = embeddings['image_emb']
    else:
        raise ValueError('Unknown ATP/CU image embedding: {}'.format(image_embedding))

    audio_emb = embeddings['audio_emb']
    return image_emb, audio_emb


def align_true_pairs_loss(embeddings, image_embedding):
    image_emb, audio_emb = select_atp_cu_embeddings(
        embeddings, image_embedding)
    return (image_emb - audio_emb).pow(2).sum(dim=1).mean()


def centroid_uniformity_loss(embeddings, image_embedding):
    image_emb, audio_emb = select_atp_cu_embeddings(
        embeddings, image_embedding)
    centroids = 0.5 * (image_emb + audio_emb)
    if centroids.size(0) < 2:
        return centroids.new_zeros(())

    dist_sq = torch.pdist(centroids, p=2).pow(2)
    return torch.logsumexp(-2.0 * dist_sq, dim=0) - dist_sq.new_tensor(
        dist_sq.numel()).log()


def build_criterion(args):
    if args.cl_loss == 'ce':
        return nn.CrossEntropyLoss()
    if args.cl_loss == 'ce_sym':
        return SymmetricCrossEntropyLoss()
    if args.cl_loss == 'sigmoid':
        return SigmoidContrastiveLoss(args.sigmoid_t_init, args.sigmoid_b_init)
    raise ValueError('Unknown contrastive loss: {}'.format(args.cl_loss))


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


def cal_auc(iou):
    results = []
    for i in range(21):
        result = np.sum(np.array(iou) >= 0.05 * i)
        result = result / len(iou)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc_ = auc(x, results)

    return auc_


def build_vggss_gt_map(args, name):
    gt_map = np.zeros([224, 224])
    bboxs = []
    gt = ET.parse(args.vggss_test_path + '/anno/' + '%s.xml' % name).getroot()

    for child in gt:
        if child.tag == 'bbox':
            for childs in child:
                bbox_normalized = [float(x.text) for x in childs]
                bbox = [int(x * 224) for x in bbox_normalized]
                bboxs.append(bbox)

    for item in bboxs:
        xmin, ymin, xmax, ymax = item
        gt_map[ymin:ymax, xmin:xmax] = 1

    return gt_map, bboxs


def build_gt_map(args, annotation_type, name):
    if annotation_type == 'vggss':
        return build_vggss_gt_map(args, name)
    raise ValueError('Unknown annotation type: {}'.format(annotation_type))


def select_checkpoint_score(metrics, args):
    metric = args.checkpoint_metric
    if metric == 'auto':
        metric = 'mean_ciou' if metrics.get('mean_ciou') is not None else 'loss'

    value = metrics.get(metric)
    if value is None:
        raise ValueError(
            'Checkpoint metric {} is not available for this validation set.'
            .format(metric))

    score = -value if metric == 'loss' else value
    return metric, score


def set_path(args):
    checkpoint_path = args.resume or args.test
    if checkpoint_path:
        exp_path = os.path.dirname(os.path.dirname(
            os.path.abspath(checkpoint_path)))
    else:
        exp_path = os.path.join(args.output_dir, args.exp_name)

    args.exp_path = exp_path
    args.images_path = os.path.join(exp_path, 'images')
    args.checkpoints_path = os.path.join(exp_path, 'checkpoints')
    args.logs_path = os.path.join(exp_path, 'logs')
    args.metrics_path = os.path.join(exp_path, 'metrics')
    args.tensorboard_path = os.path.join(exp_path, 'tensorboard')

    for path in [
            args.images_path,
            args.checkpoints_path,
            args.logs_path,
            args.metrics_path,
            args.tensorboard_path,
    ]:
        os.makedirs(path, exist_ok=True)


def save_embeddings(exp_path, split, epoch, names, embedding_batches,
                    subset=None):
    path_parts = [exp_path, 'embeddings', split]
    if subset is not None:
        path_parts.append(subset)
    embeddings_dir = os.path.join(*path_parts)
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    embeddings_path = os.path.join(
        embeddings_dir, 'epoch_{:04d}.npz'.format(epoch))
    np.savez_compressed(
        embeddings_path,
        names=np.asarray(names, dtype=str),
        **{
            key: np.concatenate(batches, axis=0)
            for key, batches in embedding_batches.items()
        },
    )
    print('Saved {} embeddings to {}'.format(split, embeddings_path))


def record_epoch_report(args, epoch, optim, train_metrics, val_metrics,
                        checkpoint_metric, checkpoint_score, best_metric,
                        is_best, early_stop_wait):
    row = {
        'epoch': epoch,
        'iteration': args.iteration,
        'learning_rate': optim.param_groups[0]['lr'],
        'checkpoint_metric': checkpoint_metric,
        'checkpoint_score': checkpoint_score,
        'best_metric_score': best_metric,
        'is_best': int(is_best),
        'early_stop_wait': early_stop_wait,
        'validation_ran': int(val_metrics is not None),
    }
    for key, value in train_metrics.items():
        row['train_' + key] = value

    if val_metrics is not None:
        for key in [
                'loss',
                'top1_i2a',
                'top5_i2a',
                'top1_a2i',
                'top5_a2i',
                'epoch_seconds',
        ]:
            row['val_' + key] = val_metrics[key]
        row['val_mean_ciou'] = val_metrics['mean_ciou']
        row['val_mean_auc'] = val_metrics['auc']
        row['val_has_annotations'] = int(val_metrics['has_annotations'])

    record_epoch(args.metrics_path, row)


def train_one_epoch(train_loader, model, criterion, optim, device, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    losses_cl = AverageMeter('Loss',':.4f')
    losses_cl_ts = AverageMeter('Loss',':.4f')
    losses_ts = AverageMeter('Loss',':.4f')
    losses_atp = AverageMeter('Loss_ATP',':.4f')
    losses_atp_ts = AverageMeter('Loss_ATP_ts',':.4f')
    losses_cu = AverageMeter('Loss_CU',':.4f')
    losses_cu_ts = AverageMeter('Loss_CU_ts',':.4f')
    top1_meter_i2a = AverageMeter('acc@1_i2a', ':.4f')
    top5_meter_i2a = AverageMeter('acc@5_i2a', ':.4f')
    top1_meter_a2i = AverageMeter('acc@1_a2i', ':.4f')
    top5_meter_a2i = AverageMeter('acc@5_a2i', ':.4f')
    top1_meter_ts_i2a = AverageMeter('acc@1_ts_i2a', ':.4f')
    top5_meter_ts_i2a = AverageMeter('acc@5_ts_i2a', ':.4f')
    top1_meter_ts_a2i = AverageMeter('acc@1_ts_a2i', ':.4f')
    top5_meter_ts_a2i = AverageMeter('acc@5_ts_a2i', ':.4f')

    progress = ProgressMeter(                             
        len(train_loader),
        [batch_time, data_time, losses, top1_meter_i2a, top5_meter_i2a, top1_meter_a2i, top5_meter_a2i],
        prefix='Epoch:[{}]'.format(epoch))
    model.train()
    end = time.time()
    tic = time.time()

    lambda_trans_equiv = args.trans_equi_weight
    lambda_atp = args.lambda_atp
    lambda_cu = args.lambda_cu
    use_atp_cu = (
        (lambda_atp > 0 or lambda_cu > 0)
        and epoch >= args.atp_cu_start_epoch
    )
    
    for idx, (image, spec, audio, name) in enumerate(train_loader):
        data_time.update(time.time() - end)
        spec = spec.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)
        B = image.size(0)
        if use_atp_cu:
            heatmap, out, Pos, Neg, out_ref, embeddings = model(
                image.float(), spec.float(), return_embeddings=True)
            if lambda_atp > 0:
                loss_atp = align_true_pairs_loss(
                    embeddings, args.atp_cu_image_embedding)
            else:
                loss_atp = torch.zeros((), device=device)
            if lambda_cu > 0:
                loss_cu = centroid_uniformity_loss(
                    embeddings, args.atp_cu_image_embedding)
            else:
                loss_cu = torch.zeros((), device=device)
        else:
            heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float())
            loss_atp = torch.zeros((), device=device)
            loss_cu = torch.zeros((), device=device)

        if args.heatmap_no_grad:
            heatmap = heatmap.detach()

        target = torch.zeros(out.shape[0]).to(device, non_blocking=True).long()        
        loss_cl = criterion(out, target)                          
        logits_i2a = out
        logits_a2i = get_logits_a2i(out)
        top1_i2a, top5_i2a = calc_topk_accuracy(logits_i2a, target, (1,5))
        top1_a2i, top5_a2i = calc_topk_accuracy(logits_a2i, target, (1,5))

        tf_equiv_loss = TfEquivarianceLoss(
                        transform_type='rotation',
                        consistency_type=args.equi_loss_type,
                        batch_size=B,
                        max_angle=args.max_rotation_angle,
                        input_hw=(224, 224),
                        )
        tf_equiv_loss.set_tf_matrices()

        transformed_image = tf_equiv_loss.transform(image)
        
        if use_atp_cu:
            heatmap_ts, out_ts, Pos, Neg, out_ref, embeddings_ts = model(
                transformed_image.float(), spec.float(), return_embeddings=True)
            if lambda_atp > 0:
                loss_atp_ts = align_true_pairs_loss(
                    embeddings_ts, args.atp_cu_image_embedding)
            else:
                loss_atp_ts = torch.zeros((), device=device)
            if lambda_cu > 0:
                loss_cu_ts = centroid_uniformity_loss(
                    embeddings_ts, args.atp_cu_image_embedding)
            else:
                loss_cu_ts = torch.zeros((), device=device)
        else:
            heatmap_ts, out_ts, Pos, Neg, out_ref = model(
                transformed_image.float(), spec.float())
            loss_atp_ts = torch.zeros((), device=device)
            loss_cu_ts = torch.zeros((), device=device)
        loss_cl_ts = criterion(out_ts, target)
        logits_ts_i2a = out_ts
        logits_ts_a2i = get_logits_a2i(out_ts)
        top1_ts_i2a, top5_ts_i2a = calc_topk_accuracy(logits_ts_i2a, target, (1,5))
        top1_ts_a2i, top5_ts_a2i = calc_topk_accuracy(logits_ts_a2i, target, (1,5))

        ts_heatmap = tf_equiv_loss.transform(heatmap)
        loss_ts = tf_equiv_loss(heatmap_ts, ts_heatmap)
        loss_atp_total = 0.5 * (loss_atp + loss_atp_ts)
        loss_cu_total = 0.5 * (loss_cu + loss_cu_ts)
        loss = (
            0.5 * (loss_cl + loss_cl_ts)
            + lambda_trans_equiv * loss_ts
            + lambda_atp * loss_atp_total
            + lambda_cu * loss_cu_total
        )

        losses.update(loss.item(), B)
        losses_cl.update(loss_cl.item(), B)
        losses_cl_ts.update(loss_cl_ts.item(), B)
        losses_ts.update(loss_ts.item(), B)
        losses_atp.update(loss_atp.item(), B)
        losses_atp_ts.update(loss_atp_ts.item(), B)
        losses_cu.update(loss_cu.item(), B)
        losses_cu_ts.update(loss_cu_ts.item(), B)

        top1_meter_i2a.update(top1_i2a.item(), B)
        top5_meter_i2a.update(top5_i2a.item(), B)
        top1_meter_a2i.update(top1_a2i.item(), B)
        top5_meter_a2i.update(top5_a2i.item(), B)

        top1_meter_ts_i2a.update(top1_ts_i2a.item(), B)
        top5_meter_ts_i2a.update(top5_ts_i2a.item(), B)
        top1_meter_ts_a2i.update(top1_ts_a2i.item(), B)
        top5_meter_ts_a2i.update(top5_ts_a2i.item(), B)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)
            args.train_plotter.add_data('local/loss_cl', loss_cl.item(), args.iteration)
            args.train_plotter.add_data('local/loss_cl_ts', loss_cl_ts.item(), args.iteration)
            args.train_plotter.add_data('local/loss_ts', loss_ts.item(), args.iteration)
            args.train_plotter.add_data('local/loss_atp', loss_atp.item(), args.iteration)
            args.train_plotter.add_data('local/loss_atp_ts', loss_atp_ts.item(), args.iteration)
            args.train_plotter.add_data('local/loss_cu', loss_cu.item(), args.iteration)
            args.train_plotter.add_data('local/loss_cu_ts', loss_cu_ts.item(), args.iteration)
            args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1_i2a', top1_meter_i2a.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5_i2a', top5_meter_i2a.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1_a2i', top1_meter_a2i.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5_a2i', top5_meter_a2i.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1_ts_i2a', top1_meter_ts_i2a.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5_ts_i2a', top5_meter_ts_i2a.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1_ts_a2i', top1_meter_ts_a2i.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5_ts_a2i', top5_meter_ts_a2i.local_avg, args.iteration)

        args.iteration += 1

    epoch_seconds = time.time() - tic
    print('Epoch: [{0}][{1}/{2}]\t'
        'T-epoch:{t:.2f}\t'.format(epoch, idx, len(train_loader), t=epoch_seconds))

    sigmoid_metrics = {}
    if isinstance(criterion, SigmoidContrastiveLoss):
        sigmoid_param_msg = (
            'Sigmoid params: t={t:.6f} scale=exp(t)={scale:.6f} b={b:.6f}'
            .format(
                t=criterion.t.item(),
                scale=criterion.t.exp().item(),
                b=criterion.b.item(),
            )
        )
        print(sigmoid_param_msg)
        sigmoid_metrics = {
            'sigmoid_t': criterion.t.item(),
            'sigmoid_scale': criterion.t.exp().item(),
            'sigmoid_b': criterion.b.item(),
        }

    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    args.train_plotter.add_data('global/loss_cl', losses_cl.avg, epoch)
    args.train_plotter.add_data('global/loss_cl_ts', losses_cl_ts.avg, epoch)
    args.train_plotter.add_data('global/loss_ts', losses_ts.avg, epoch)
    args.train_plotter.add_data('global/loss_atp', losses_atp.avg, epoch)
    args.train_plotter.add_data('global/loss_atp_ts', losses_atp_ts.avg, epoch)
    args.train_plotter.add_data('global/loss_cu', losses_cu.avg, epoch)
    args.train_plotter.add_data('global/loss_cu_ts', losses_cu_ts.avg, epoch)
    args.train_plotter.add_data('global/top1_i2a', top1_meter_i2a.avg, epoch)
    args.train_plotter.add_data('global/top5_i2a', top5_meter_i2a.avg, epoch)
    args.train_plotter.add_data('global/top1_a2i', top1_meter_a2i.avg, epoch)
    args.train_plotter.add_data('global/top5_a2i', top5_meter_a2i.avg, epoch)
    args.train_plotter.add_data('global/top1_ts_i2a', top1_meter_ts_i2a.avg, epoch)
    args.train_plotter.add_data('global/top5_ts_i2a', top5_meter_ts_i2a.avg, epoch)
    args.train_plotter.add_data('global/top1_ts_a2i', top1_meter_ts_a2i.avg, epoch)
    args.train_plotter.add_data('global/top5_ts_a2i', top5_meter_ts_a2i.avg, epoch)

    return {
        'loss': losses.avg,
        'loss_cl': losses_cl.avg,
        'loss_cl_ts': losses_cl_ts.avg,
        'loss_ts': losses_ts.avg,
        'loss_atp': losses_atp.avg,
        'loss_atp_ts': losses_atp_ts.avg,
        'loss_cu': losses_cu.avg,
        'loss_cu_ts': losses_cu_ts.avg,
        'top1_i2a': top1_meter_i2a.avg,
        'top5_i2a': top5_meter_i2a.avg,
        'top1_a2i': top1_meter_a2i.avg,
        'top5_a2i': top5_meter_a2i.avg,
        'top1_ts_i2a': top1_meter_ts_i2a.avg,
        'top5_ts_i2a': top5_meter_ts_i2a.avg,
        'top1_ts_a2i': top1_meter_ts_a2i.avg,
        'top5_ts_a2i': top5_meter_ts_a2i.avg,
        'epoch_seconds': epoch_seconds,
        **sigmoid_metrics,
    }


def validate(val_loader, model, criterion, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter_i2a = AverageMeter('acc@1_i2a', ':.4f')
    top5_meter_i2a = AverageMeter('acc@5_i2a', ':.4f')
    top1_meter_a2i = AverageMeter('acc@1_a2i', ':.4f')
    top5_meter_a2i = AverageMeter('acc@5_a2i', ':.4f')

    has_annotations = getattr(val_loader.dataset, 'has_annotations', False)
    annotation_type = getattr(val_loader.dataset, 'annotation_type', None)
    val_ious_meter = []
    sample_names = []
    embedding_batches = {}
    tic = time.time()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (image, spec, audio, name) in tqdm(enumerate(val_loader), total=len(val_loader)):
            spec = spec.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            B = image.size(0)

            if args.save_val_embeddings:
                heatmap, out, Pos, Neg, out_ref, embeddings = model(
                    image.float(), spec.float(), return_embeddings=True)
                sample_names.extend(list(name))
                for key, embedding in embeddings.items():
                    embedding_batches.setdefault(key, []).append(
                        embedding.detach().cpu().numpy().astype(np.float32))
            else:
                heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float())
            target = torch.zeros(out.shape[0]).to(device, non_blocking=True).long()
            loss = criterion(out, target)
            logits_i2a = out
            logits_a2i = get_logits_a2i(out)
            top1_i2a, top5_i2a = calc_topk_accuracy(logits_i2a, target, (1, 5))
            top1_a2i, top5_a2i = calc_topk_accuracy(logits_a2i, target, (1, 5))

            losses.update(loss.item(), B)
            top1_meter_i2a.update(top1_i2a.item(), B)
            top5_meter_i2a.update(top5_i2a.item(), B)
            top1_meter_a2i.update(top1_a2i.item(), B)
            top5_meter_a2i.update(top5_a2i.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()

            if has_annotations:
                heatmap_arr = heatmap.data.cpu().numpy()

                for i in range(spec.shape[0]):
                    heatmap_now = cv2.resize(
                        heatmap_arr[i, 0],
                        dsize=(224, 224),
                        interpolation=cv2.INTER_LINEAR)
                    heatmap_now = normalize_img(-heatmap_now)
                    gt_map, bboxs = build_gt_map(args, annotation_type, name[i])

                    pred = heatmap_now
                    pred = 1 - pred
                    threshold = np.sort(pred.flatten())[
                        int(pred.shape[0] * pred.shape[1] / 2)]
                    pred[pred > threshold] = 1
                    pred[pred < 1] = 0
                    evaluator = Evaluator()
                    ciou, inter, union = evaluator.cal_CIOU(pred, gt_map, 0.5)
                    val_ious_meter.append(ciou)

    mean_ciou = None
    auc_val = None
    if has_annotations:
        mean_ciou = np.sum(np.array(val_ious_meter) >= 0.5) / len(val_ious_meter)
        auc_val = cal_auc(val_ious_meter)

    msg = (
        'Epoch: [{0}]\t Eval '
        'Loss: {loss.avg:.4f} Acc@1_i2a: {top1_i2a.avg:.4f} '
        'Acc@5_i2a: {top5_i2a.avg:.4f} Acc@1_a2i: {top1_a2i.avg:.4f} '
        'Acc@5_a2i: {top5_a2i.avg:.4f}'
        .format(epoch, loss=losses, top1_i2a=top1_meter_i2a,
                top5_i2a=top5_meter_i2a, top1_a2i=top1_meter_a2i,
                top5_a2i=top5_meter_a2i))
    if has_annotations:
        msg += ' MeancIoU: {0:.4f} AUC: {1:.4f}'.format(mean_ciou, auc_val)
    msg += ' \t T-epoch: {0:.2f} \t'.format(time.time() - tic)
    print(msg)

    args.val_plotter.add_data('global/loss', losses.avg, epoch)
    args.val_plotter.add_data('global/top1_i2a', top1_meter_i2a.avg, epoch)
    args.val_plotter.add_data('global/top5_i2a', top5_meter_i2a.avg, epoch)
    args.val_plotter.add_data('global/top1_a2i', top1_meter_a2i.avg, epoch)
    args.val_plotter.add_data('global/top5_a2i', top5_meter_a2i.avg, epoch)
    if has_annotations:
        args.val_plotter.add_data('global/mean_ciou', mean_ciou, epoch)
        args.val_plotter.add_data('global/mean_auc', auc_val, epoch)

    if args.save_val_embeddings:
        save_embeddings(
            args.exp_path, 'val', epoch, sample_names, embedding_batches)

    return {
        'loss': losses.avg,
        'top1_i2a': top1_meter_i2a.avg,
        'top5_i2a': top5_meter_i2a.avg,
        'top1_a2i': top1_meter_a2i.avg,
        'top5_a2i': top5_meter_a2i.avg,
        'mean_ciou': mean_ciou,
        'auc': auc_val,
        'has_annotations': has_annotations,
        'epoch_seconds': time.time() - tic,
    }


def test(test_loader, model, criterion, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter_i2a = AverageMeter('acc@1_i2a', ':.4f')
    top5_meter_i2a = AverageMeter('acc@5_i2a', ':.4f')
    top1_meter_a2i = AverageMeter('acc@1_a2i', ':.4f')
    top5_meter_a2i = AverageMeter('acc@5_a2i', ':.4f')

    # Compute ciou
    val_ious_meter = []

    # dir for saving validationset heatmap images 
    save_dir = os.path.join(args.images_path, "test", str(epoch), args.test_set)
    sample_names = []
    embedding_batches = {}

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (image, spec, audio, name) in tqdm(enumerate(test_loader), total=len(test_loader)):         
            spec = spec.to(device)
            image = image.to(device)
            B = image.size(0)

            if args.save_test_embeddings:
                heatmap, out, Pos, Neg, out_ref, embeddings = model(
                    image.float(), spec.float(), return_embeddings=True)
                sample_names.extend(list(name))
                for key, embedding in embeddings.items():
                    embedding_batches.setdefault(key, []).append(
                        embedding.detach().cpu().numpy().astype(np.float32))
            else:
                heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float())
            target = torch.zeros(out.shape[0]).to(device, non_blocking=True).long()
            loss =  criterion(out, target)
            logits_i2a = out
            logits_a2i = get_logits_a2i(out)
            top1_i2a, top5_i2a = calc_topk_accuracy(logits_i2a, target, (1,5))
            top1_a2i, top5_a2i = calc_topk_accuracy(logits_a2i, target, (1,5))
            losses.update(loss.item(), B)
            top1_meter_i2a.update(top1_i2a.item(), B)
            top5_meter_i2a.update(top5_i2a.item(), B)
            top1_meter_a2i.update(top1_a2i.item(), B)
            top5_meter_a2i.update(top5_a2i.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()

            heatmap_arr =  heatmap.data.cpu().numpy()

            for i in range(spec.shape[0]):
                
                heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_now = normalize_img(-heatmap_now)
                gt_map = np.zeros([224,224])
                bboxs = []

                if args.test_set == 'VGGSS':
                    gt = ET.parse(args.vggss_test_path + '/anno/' + '%s.xml' % name[i]).getroot()
                    
                    for child in gt:                 
                        if child.tag == 'bbox':
                            for childs in child:
                                bbox_normalized = [ float(x.text) for x in childs  ]
                                bbox = [int(x*224) for x in bbox_normalized ]           
                                bboxs.append(bbox)
                
                    for item in bboxs:
                        xmin, ymin, xmax, ymax = item
                        gt_map[ymin:ymax, xmin:xmax] = 1

                else:
                    print('Testing dataset Not Assigned !')

                pred =  heatmap_now
                pred = 1 - pred
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]    # 计算threshold
                pred[pred>threshold]  = 1
                pred[pred<1] = 0
                evaluator = Evaluator()
                ciou, inter, union = evaluator.cal_CIOU(pred, gt_map, 0.5)

                val_ious_meter.append(ciou)  

                heatmap_vis = np.expand_dims(heatmap_arr[i], axis=0)
                # img_vis = img_arrs[i]
                img_vis_tensor = image[i]
                img_vis = tensor2img(img_vis_tensor.data.cpu())

                name_vis = name[i]
                bbox_vis = bboxs
                
                heatmap_img = vis_heatmap_bbox(heatmap_vis, img_vis, name_vis,\
                        bbox=bbox_vis, ciou=ciou, save_dir=save_dir )
            
    mean_ciou = np.sum(np.array(val_ious_meter) >= 0.5)/ len(val_ious_meter)
    auc_val = cal_auc(val_ious_meter)

    print('Test: \t Epoch: [{0}]\t'
          'Loss: {loss.avg:.4f} Acc@1_i2a: {top1_i2a.avg:.4f} Acc@5_i2a: {top5_i2a.avg:.4f} '
          'Acc@1_a2i: {top1_a2i.avg:.4f} Acc@5_a2i: {top5_a2i.avg:.4f} MeancIoU: {ciouAvg:.4f} AUC: {auc:.4f}\t'
          .format(epoch, loss=losses, top1_i2a=top1_meter_i2a, top5_i2a=top5_meter_i2a,
                  top1_a2i=top1_meter_a2i, top5_a2i=top5_meter_a2i,
                  ciouAvg=mean_ciou, auc=auc_val))

    args.test_logger.log('Test Epoch: [{0}]\t'
                    'Loss: {loss.avg:.4f} Acc@1_i2a: {top1_i2a.avg:.4f} Acc@5_i2a: {top5_i2a.avg:.4f} '
                    'Acc@1_a2i: {top1_a2i.avg:.4f} Acc@5_a2i: {top5_a2i.avg:.4f} MeancIoU: {ciouAvg:.4f} AUC:{auc:.4f} \t'
                    .format(epoch, loss=losses, top1_i2a=top1_meter_i2a, top5_i2a=top5_meter_i2a,
                            top1_a2i=top1_meter_a2i, top5_a2i=top5_meter_a2i,
                            ciouAvg=mean_ciou, auc=auc_val))

    if args.save_test_embeddings:
        save_embeddings(
            args.exp_path, 'test', epoch, sample_names, embedding_batches,
            subset=args.test_set)

    sys.exit(0)


def main(args):
    if args.early_stop_patience < 0:
        raise ValueError('early_stop_patience must be non-negative')
    if args.early_stop_min_delta < 0:
        raise ValueError('early_stop_min_delta must be non-negative')
    if args.lambda_atp < 0:
        raise ValueError('lambda_atp must be non-negative')
    if args.lambda_cu < 0:
        raise ValueError('lambda_cu must be non-negative')
    if args.atp_cu_start_epoch < 1:
        raise ValueError('atp_cu_start_epoch must be at least 1')

    # Set GPU IDs
    if args.gpus is None:
        args.gpus = str(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)
        args.gpus = list(range(torch.cuda.device_count()))

    # Set device
    if torch.cuda.is_available() and len(args.gpus) > 0:
        device = torch.device('cuda:1') if len(args.gpus) > 1 else torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    set_path(args)

    best_metric = float('-inf')
    early_stop_wait = 0

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create model and move to device
    model = AVENet(args)
    model = model.to(device)

    # Only wrap the model in DataParallel if we have multiple GPUs
    if device.type == 'cuda' and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpus, output_device=device)  
        model_without_dp = model.module
    else:
        model_without_dp = model

    criterion = build_criterion(args).to(device)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, criterion.parameters()))
    optim = Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[300,700,900], gamma=0.1)
    args.iteration = 1
    
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            
            try: 
                model_without_dp.load_state_dict(state_dict)
            except: 
                neq_load_customized(model_without_dp, state_dict, verbose=True)
            if 'criterion' in checkpoint:
                try:
                    criterion.load_state_dict(checkpoint['criterion'])
                except:
                    print('[WARNING] failed to load criterion state')
        
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.test))
            epoch = 0

        logger_path = os.path.join(args.logs_path, 'test')
        os.makedirs(logger_path, exist_ok=True)

        args.test_logger = Logger(path=logger_path)
        args.test_logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))

        if args.test_set == 'VGGSS':
            test_dataset = GetAudioVideoDataset(args, mode='test')
        else:
            raise ValueError('Unknown test set: {}'.format(args.test_set))

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,\
            num_workers=args.n_threads, pin_memory=True)
        
        test(test_loader, model, criterion, device, epoch, args)
        
    train_dataset = GetAudioVideoDataset(args, mode='train')
    val_dataset = GetAudioVideoDataset(args, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.n_threads, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.n_threads, drop_last=False, pin_memory=True)
    
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']+ 1
            args.iteration = checkpoint['iteration']
            best_metric = checkpoint.get(
                'best_metric',
                checkpoint.get('best_miou', float('-inf')))
            early_stop_wait = checkpoint.get('early_stop_wait', 0)
            state_dict = checkpoint['state_dict']

            try: 
                model_without_dp.load_state_dict(state_dict)
            except:
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_dp, state_dict, verbose=True)
            if 'criterion' in checkpoint:
                try:
                    criterion.load_state_dict(checkpoint['criterion'])
                except:
                    print('[WARNING] failed to load criterion state')
            
            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            
            try:
                optim.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))

    else:
        print('Train the model from scratch on VGGS!')

    torch.backends.cudnn.benchmark = True

    writer_val = SummaryWriter(logdir=os.path.join(args.tensorboard_path, 'val'))
    writer_train = SummaryWriter(logdir=os.path.join(args.tensorboard_path, 'train'))
    args.val_plotter = TB.PlotterThread(writer_val)
    args.train_plotter = TB.PlotterThread(writer_train)

    train_log_path = os.path.join(args.logs_path, 'train')
    os.makedirs(train_log_path, exist_ok=True)

    args.train_logger = Logger(path=train_log_path)

    args.train_logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
    if args.resume:
        args.train_logger.log("Resumed training from '{}'".format(args.resume))
    else:
        args.train_logger.log('Started training from scratch')
    
    print('\n ******************Training Args*************************')
    print('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
    print('******************Training Args*************************')

    for epoch in range(args.start_epoch, args.epochs + 1 ):
        np.random.seed(epoch)
        random.seed(epoch)

        train_metrics = train_one_epoch(
            train_loader, model, criterion, optim, device, epoch, args)

        if epoch >= args.eval_start:
            args.eval_freq = 1

        val_metrics = None
        checkpoint_metric = args.checkpoint_metric
        checkpoint_score = None
        is_best = False
        if epoch % args.eval_freq == 0:
            val_metrics = validate(val_loader, model, criterion, device, epoch, args)
            checkpoint_metric, checkpoint_score = select_checkpoint_score(
                val_metrics, args)
            is_best = checkpoint_score > (
                best_metric + args.early_stop_min_delta)
            if is_best:
                best_metric = checkpoint_score
                early_stop_wait = 0
            else:
                early_stop_wait += 1

            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'criterion': criterion.state_dict(),
                'best_metric': best_metric,
                'checkpoint_metric': checkpoint_metric,
                'early_stop_wait': early_stop_wait,
                'early_stop_patience': args.early_stop_patience,
                'early_stop_min_delta': args.early_stop_min_delta,
                'best_miou': (
                    val_metrics['mean_ciou']
                    if val_metrics['mean_ciou'] is not None
                    else None),
                'optimizer': optim.state_dict(),
                'iteration': args.iteration}

            save_checkpoint(
                save_dict,
                is_best,
                filename=os.path.join(
                    args.checkpoints_path, 'checkpoint_latest.pth.tar'))
            if is_best:
                args.train_logger.log(
                    'Saved new best checkpoint at epoch {}: {} score {:.6f}'
                    .format(epoch, checkpoint_metric, checkpoint_score))

            if (
                    args.early_stop_patience > 0 and
                    early_stop_wait >= args.early_stop_patience):
                msg = (
                    'Early stopping at epoch {}: no {} improvement greater '
                    'than {:.6g} for {} validation checks. Best {} score: '
                    '{:.6f}'
                    .format(
                        epoch,
                        checkpoint_metric,
                        args.early_stop_min_delta,
                        early_stop_wait,
                        checkpoint_metric,
                        best_metric,
                    ))
                print(msg)
                args.train_logger.log(msg)
                record_epoch_report(
                    args, epoch, optim, train_metrics, val_metrics,
                    checkpoint_metric, checkpoint_score, best_metric,
                    is_best, early_stop_wait)
                break
        
        else:
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'criterion': criterion.state_dict(),
                'best_metric': best_metric,
                'checkpoint_metric': args.checkpoint_metric,
                'early_stop_wait': early_stop_wait,
                'early_stop_patience': args.early_stop_patience,
                'early_stop_min_delta': args.early_stop_min_delta,
                'best_miou': None,
                'optimizer': optim.state_dict(),
                'iteration': args.iteration}

            save_checkpoint(
                save_dict,
                is_best=0,
                filename=os.path.join(
                    args.checkpoints_path, 'checkpoint_latest.pth.tar'))

        record_epoch_report(
            args, epoch, optim, train_metrics, val_metrics,
            checkpoint_metric, checkpoint_score, best_metric,
            is_best, early_stop_wait)

        scheduler.step()
    
    finished_msg = 'Training from Epoch %d --> Epoch %d finished' % (
        args.start_epoch, args.epochs)
    print(finished_msg)
    args.train_logger.log(finished_msg)
    writer_train.close()
    writer_val.close()
    
    sys.exit(0)


if __name__ == "__main__":
    args=get_arguments()
    main(args)
