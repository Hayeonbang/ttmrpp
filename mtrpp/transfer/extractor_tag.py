import json
import os
import random
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch.backends.cudnn as cudnn
# backbones
from mtrpp.transfer.dataset_wavs.data_manger import get_dataloader
from mtrpp.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams
from mtrpp.utils.eval_utils import load_ttmr_pp, load_ttmr_atprobing, load_ttmr_probing
from sklearn import metrics
import torch.backends.cudnn as cudnn
from tqdm import tqdm
random.seed(42)
torch.manual_seed(42)
cudnn.deterministic = True

parser = argparse.ArgumentParser(description='')
parser.add_argument('--msu_dir', type=str, default="/home/habang8/music-text-representation-pp/mtrpp/Data")
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument("--eval_dataset", default="annotation", type=str)
parser.add_argument("--probe_type", default="extract", type=str)
parser.add_argument("--model_type", default="tag", type=str)
parser.add_argument("--num_chunks", default=3, type=int)
parser.add_argument("--gpu", default=0, type=int)
args = parser.parse_args()


def main(args) -> None:
    print(args.num_chunks)
    pretrain_dir = "/home/habang8/music-text-representation-pp/mtrpp/exp/ttmrpp/caption/youtube/64_5e-05_128_0.3_1_0.05"
    save_dir = f"/home/habang8/music-text-representation-pp/mtrpp/exp/ttmrpp/{args.model_type}"
    model, sr, duration = load_ttmr_atprobing(pretrain_dir)
    
    embs_dir = os.path.join(save_dir, "embs", args.eval_dataset)
    os.makedirs(embs_dir, exist_ok=True)
    args.sr = sr
    args.duration = duration
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True
    model.eval()
    
    all_loader = get_dataloader(args=args, split="ALL")
    
    tag_embs, audio_embs = {}, {}
    for batch in tqdm(all_loader):
        audio = batch['audio']
        track_id = str(batch['track_id'][0])
        if args.gpu is not None:
            audio = audio.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            z_audio = model.audio_forward(audio.squeeze(0))
        audio_embs[track_id] = z_audio.mean(0).detach().cpu()
    torch.save(audio_embs, os.path.join(save_dir, "embs", "from_probing", "audio_embs_yt_0523-1.pt"))

    for tag in all_loader.dataset.list_of_label:
        with torch.no_grad():
            z_tag = model.text_forward([tag])
        tag_embs[tag] = z_tag.squeeze(0).detach().cpu()
    torch.save(tag_embs, os.path.join(save_dir, "embs", "from_probing", "tag_embs_yt_0523-1.pt"))

if __name__ == "__main__":
    main(args)