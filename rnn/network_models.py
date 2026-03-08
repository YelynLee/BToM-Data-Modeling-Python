"""
Customized RNN networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
# FORCE_USING_TORCH_SCRIPT = False
#
# if FORCE_USING_TORCH_SCRIPT is True or (FORCE_USING_TORCH_SCRIPT is None and platform.system() == 'Windows'):
#     from .custom_lstms import RNNLayer_custom
#     # torchscript can save ~25% time for the current layers
#     # cannot work together with multiprocessing, can be used for single process training
#     # current code will trigger an internal bug in pytorch on Linux
# else:

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


import torch
import torch.nn as nn

class RNNnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_type='GRU', device='cpu'):
        super(RNNnet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # 1. RNN Layer 정의 (기본 GRU 사용)
        # batch_first=False가 원본 코드 기준 (Seq_Len, Batch, Dim)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim)
            
        # 2. Readout Layer (Hidden State -> Belief/Desire)
        self.lin = nn.Linear(hidden_dim, output_dim)
        
        # 3. 초기 Hidden State (h0) - 0으로 초기화
        # 학습 가능한 파라미터로 만들고 싶다면 nn.Parameter로 감싸면 됨
        self.h0 = torch.zeros(1, 1, hidden_dim).to(device)

    def forward(self, x, h0=None):
        # x shape: (Seq_Len, Batch, Input_Dim)
        
        batch_size = x.size(1)
        
        # h0가 없으면 기본값 사용 (Batch 크기에 맞춰 확장)
        if h0 is None:
            h0 = self.h0.repeat(1, batch_size, 1)
            
        # RNN 연산
        # rnn_out: 모든 타임스텝의 Hidden State (Seq_Len, Batch, Hidden_Dim)
        # hn: 마지막 타임스텝의 Hidden State
        rnn_out, hn = self.rnn(x, h0)
        
        # 출력층 통과 (Belief/Desire 예측)
        # scores shape: (Seq_Len, Batch, Output_Dim)
        scores = self.lin(rnn_out)
        
        return scores, rnn_out