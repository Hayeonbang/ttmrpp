import torch
import torch.nn as nn
import numpy as np
from mtrpp.model.loss import InfoNCE



class AT_ProbingLayer(nn.Module):
    def __init__(self, audio_dim, mlp_dim, output_dim, dropout, is_norm):
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
            nn.Dropout(dropout),  # 드롭아웃 추가
            nn.Linear(self.mlp_dim, output_dim),
        )



    def forward(self, audio_emb, text_emb):
        if self.is_norm:
            audio_emb = self.norm_layer(audio_emb)
            text_emb = self.norm_layer(text_emb)
        
        audio_emb = self.dropout(audio_emb)  # 드롭아웃 적용
        text_emb = self.dropout(text_emb)    # 드롭아웃 적용

        audio_output = self.cls_head(audio_emb)
        text_output = self.cls_head(text_emb)
        
        audio_loss = self.loss_fn(audio_output, text_output)
        text_loss = self.loss_fn(text_output, audio_output)
        loss = (audio_loss + text_loss) / 2
        return loss

    
    
    # def test_forward(self, x):
    #     output = self.cls_head(x)
    #     logit = self.activation(output)
    #     return logit