import torch.nn as nn

class TinyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TinyRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 1. RNN Layer
        # batch_first=True: 입력이 (Batch, Seq, Feature) 순서임
        # Tiny 실험이므로 가장 기본적인 RNN 사용 (LSTM/GRU 대신)
        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            nonlinearity='tanh' # tanh가 기본, relu로 바꿔 실험 가능
        )
        
        # 2. Output Layer (Hidden State -> Action Probability)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # x shape: (Batch, Seq_Len, Input_Size)
        
        # RNN 통과
        # out shape: (Batch, Seq_Len, Hidden_Size)
        out, hidden = self.rnn(x, hidden)
        
        # 행동 예측
        # out shape: (Batch, Seq_Len, Output_Size)
        out = self.fc(out)
        
        return out, hidden