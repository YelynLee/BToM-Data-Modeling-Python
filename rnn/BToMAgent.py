from .BaseAgent import BaseAgent
from .network_models import RNNnet, set_seed

from copy import deepcopy
import torch
import os
import json
import joblib
from path_settings import *
import numpy as np
import torch.nn as nn
from random import random

class BToMAgent(BaseAgent):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 핵심 1: 모델 생성 (Tiny RNN: hidden_dim = 1 or 2)
        # Output Dim 설정 (Desire 3개 + Belief 3개 = 6개)
        # config['output_dim']은 6으로 설정해서 들어와야 함
        self.model = RNNnet(config['input_dim'], config['hidden_dim'], config['output_dim']).double()
        
        # Loss Function 변경 (중요!)
        # 7점 척도 점수를 맞춰야 하므로 '평균 제곱 오차(MSE)' 사용
        self.loss_fn = nn.MSELoss(reduction='none') 

    def forward(self, input_dict):
        # 핵심 2: 데이터 꺼내기
        x = input_dict['input'] # [Seq, Batch, Input_Dim]

        # 핵심 3: 신경망 통과
        # scores: 예측값 (Logits), rnn_out: 내부 상태 (Belief)
        # scores shape: [Seq_Len, Batch, 6]
        # 값의 범위가 1~7이 나오도록 유도해야 함 (보통은 그냥 Linear 출력 쓰고 Target을 Normalize함)
        scores, rnn_out = self.model(x) 
        return {'output': scores, 'internal': rnn_out}

    def compute_loss(self, input_dict):
        # 핵심 4: 정답과 비교
        model_out = self.forward(input_dict)
        pred = model_out['output']     # [Seq, Batch, 6]
        target = input_dict['target']  # [Seq, Batch, 6] (실제 7점 척도 정답)
        mask = input_dict['mask']      # [Seq, Batch]
        
        # Loss 계산 (단일 벡터 처리)        
        # MSELoss 계산 (Target과 Shape를 맞춰서)
        loss = self.loss_fn(pred, target) # 결과: [Seq, Batch, 6]
        
        # 마스킹 (데이터가 없는 부분의 Loss는 0으로)
        # mask는 [Seq, Batch]니까 차원 맞춰서 곱하기
        loss = loss * mask.unsqueeze(-1) 
        
        # 전체 평균 Loss 반환
        return loss.sum() / mask.sum()