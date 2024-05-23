import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset

class Annotation_Cap_Dataset(Dataset):
    def __init__(self, data_path, split, audio_embs, text_embs):
        self.data_path = data_path
        self.split = split
        self.audio_embs = audio_embs
        self.text_embs = text_embs
        self.get_split()
        self.get_file_list()
        
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "Annotation", "Annotation_split.json"), "r"))
        self.train_track = track_split['train_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']

    def get_file_list(self):
        
        with open(os.path.join(self.data_path, "Annotation", "Annotation.json"), 'r') as file:
            annotation_data = json.load(file)
            annotation = {str(item['track_id']): item for item in annotation_data}
        
        if self.split == "TRAIN":
            self.fl = [annotation[str(i)] for i in self.train_track]
        elif self.split == "VALID":
            self.fl = [annotation[str(i)] for i in self.valid_track]
        elif self.split == "TEST":
            self.fl = [annotation[str(i)] for i in self.test_track]
        elif self.split == "ALL":
            self.fl = list(annotation.values())
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
        del annotation
        
    def get_train_item(self, index):
        item = self.fl[index]
        audio_tensor = self.audio_embs[str(item['track_id'])]
        text_tensor = self.text_embs[str(item['track_id'])]
        return {
            "audio_emb": audio_tensor,
            "text_emb": text_tensor
        }
        
    def get_eval_item(self, index):
        item = self.fl[index]
        track_id = item['track_id']
        audio_tensor = self.audio_embs[str(track_id)]
        text_tensor = self.tag_embs[str(track_id)]
        return {
            "audio_emb": audio_tensor,
            "text_emb": text_tensor,
            "track_id": track_id,
            "caption": item['tag']
        }

    def __getitem__(self, index):
        if (self.split=='TRAIN') or (self.split=='VALID'):
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)
            
    def __len__(self):
        return len(self.fl)
