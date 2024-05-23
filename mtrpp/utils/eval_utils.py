import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import metrics
from omegaconf import DictConfig, OmegaConf
from mtrpp.model.dual_encoder import DualEncoderModel
from mtrpp.transfer.model_probing_caption_wav import AT_ProbingLayer
from mtrpp.transfer.model_probing_tag import ProbingLayer

def get_query2target_idx(query2target, target2idx):
    query2target_idx = {}
    for query, target_list in query2target.items():
        query2target_idx[query] = [target2idx[i] for i in target_list]
    return query2target_idx

def get_task_predictions(query_features, target_features):
    """Get similarity matrix from model output."""
    query_features = torch.nn.functional.normalize(query_features, dim=-1)
    target_features = torch.nn.functional.normalize(target_features, dim=-1)
    sim_matrix = query_features @ target_features.T
    sim_matrix = sim_matrix.numpy()
    return sim_matrix

def print_model_params(model):
    n_parameters = sum(p.numel() for p in model.parameters())
    train_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("============")
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('number train of params (M): %.2f' % (train_n_parameters / 1.e6))
    print("============")


def load_pretrain_model(args):
    if args.model_type == "random":
        model, sr, duration  = torch.nn.Identity(), 48000, 10
    elif args.model_type == "ttmr_pp":
        model, sr, duration = load_unttim_pp(args)
    else:
        model, sr, duration = load_baselines(args)
    return model, sr, duration

def load_ttmr_pp(save_dir, model_types="last"):
    config = OmegaConf.load('/home/habang8/music-text-representation-pp/mtrpp/Data/pretrain/ttmrpp_resnet_roberta.yaml')
    pretrained_object = torch.load('/home/habang8/music-text-representation-pp/mtrpp/Data/pretrain/ttmrpp_resnet_roberta.pth', map_location='cpu')

    state_dict = pretrained_object['state_dict']
    model = DualEncoderModel(
        text_arch=config.text_arch,
        n_mels=config.n_mels, 
        n_fft=config.n_fft,
        hop_size=config.hop_size,
        width=config.width,
        head =config.n_heads,
        sr=config.sr, 
        duration=config.duration, 
        max_length=config.max_length, 
        audio_dim=config.audio_dim, 
        text_dim=config.text_dim, 
        mlp_dim=config.mlp_dim, 
        temperature=config.temperature
    )
    model.load_state_dict(state_dict)
    return model, config.sr, config.duration

def load_ttmr_probing(pretrain_dir, model_types="last"):
    pth_path = os.path.join(pretrain_dir, 'best.pth')
    config_path = os.path.join(pretrain_dir, 'hparams.yaml')
    pretrained = torch.load(pth_path, map_location='cpu')
    config = OmegaConf.load(config_path)    
    state_dict = pretrained['state_dict']
    
    model = ProbingLayer(
        audio_dim = 128,
        mlp_dim = config.mlp_dim,
        output_dim = 31,
        task_type = "multilabel",
        probe_type = config.probe_type,
        dropout = config.dropout,
        is_norm = config.is_norm,
        loss_fn = nn.BCELoss()
    )
    
    model.load_state_dict(state_dict)
    return model, config.sr, config.duration

def load_ttmr_atprobing(pretrain_dir, model_types="last"):
    pth_path = os.path.join(pretrain_dir, 'best.pth')
    config_path = os.path.join(pretrain_dir, 'hparams.yaml')
    pretrained = torch.load(pth_path, map_location='cpu')
    config = OmegaConf.load(config_path)    
    state_dict = pretrained['state_dict']
    
    model = AT_ProbingLayer(
        audio_dim = 128,
        mlp_dim = config.mlp_dim,
        output_dim = 128,
        dropout = config.dropout,
        is_norm = config.is_norm,
    )
    
    model.load_state_dict(state_dict, strict=False)
    return model, config.sr, config.duration