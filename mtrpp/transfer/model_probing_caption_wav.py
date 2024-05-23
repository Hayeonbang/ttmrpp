import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mtrpp.modules.resnet import ModifiedResNet
from mtrpp.model.loss import InfoNCE
from transformers import AutoModel, AutoTokenizer, set_seed


class AT_ProbingLayer(nn.Module):
    def __init__(self, 
                audio_dim, 
                mlp_dim, 
                output_dim, 
                dropout, 
                is_norm):
        super(AT_ProbingLayer, self).__init__()
        self.audio_dim = audio_dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.is_norm = is_norm
        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.norm_layer = nn.LayerNorm(audio_dim)
        temperature = 0.1
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad=True)
        self.loss_fn = InfoNCE(logit_scale=self.logit_scale)

        self.activation = nn.Identity() ## Identity: 항등함수, y=x 연속적인 값을 그대로 전달 
        
        self.cls_head = nn.Sequential(
            nn.Linear(audio_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, output_dim),
            )

        self.audio_encoder = ModifiedResNet(
            layers=(3, 4, 6, 3),
            output_dim=audio_dim,
            n_mels=128,
            heads=8,
            width=64,
            n_fft=1024,
            hop_size=0.01,
            sr=22050,
            duration=10
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.text_encoder = AutoModel.from_pretrained("roberta-base")
        self.text_encoder.pooler.dense = nn.Identity() # Roberta: remove unused weight

        self.audio_projector = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, mlp_dim, bias=False))
        self.text_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, mlp_dim, bias=False)) # 768: roberta-base hidden size

        self.a_latent = nn.Identity()
        self.t_latent = nn.Identity()

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def audio_forward(self, audio):
        audio_embs = self.audio_encoder(audio)
        h_audio = self.a_latent(audio_embs)
        z_audio = self.audio_projector(h_audio)
        return z_audio
    
    def text_forward(self, text):
        text = self.tokenizer(text,
                            padding='longest',
                            truncation=True,
                            max_length=128,
                            return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)
        text_output = self.text_encoder(input_ids=input_ids,attention_mask=attention_mask)
        h_text = text_output["last_hidden_state"][:,0,:] # sos token
        z_text = self.text_projector(h_text)
        return z_text
    
    def forward(self, audio, text):
        z_audio = self.audio_forward(audio)
        z_text = self.text_forward(text)
        audio_loss = self.loss_fct(z_audio, z_text)
        text_loss = self.loss_fct(z_text, z_audio)
        loss = (audio_loss + text_loss) / 2
        return loss