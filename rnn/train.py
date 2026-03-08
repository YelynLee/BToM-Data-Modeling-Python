import torch
import torch.nn as nn
import torch.optim as optim
from btom_dataset import BToMDataset
from tiny_rnn import TinyRNN
from btom_preprocess import extract_btom_data # 이전에 만든 전처리 파일

# 1. 데이터 로드
mat_path = "stimuli.mat" # 파일 경로 확인
df = extract_btom_data(mat_path)
dataset = BToMDataset(df)

# Batch Size를 1로 하면 패딩(Padding) 없이 학습 가능 (가장 쉬운 구현)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# 2. 하이퍼파라미터 설정 (Tiny RNN의 핵심!)
INPUT_SIZE = 12    # Agent(2)+Trucks(6)+Wall(4)
HIDDEN_SIZE = 4    # ★ Tiny! (이 숫자를 2, 4, 8로 바꿔가며 실험)
OUTPUT_SIZE = 5    # Stay, U, D, L, R
LEARNING_RATE = 0.001
EPOCHS = 100

# 3. 모델 및 설정 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model Created on {device}. Hidden Size: {HIDDEN_SIZE}")

# 4. 학습 루프
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total_steps = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        # x shape: (1, Seq_Len, 12)
        # y shape: (1, Seq_Len) -> Action Indices
        
        # Forward
        outputs, _ = model(x) # (1, Seq_Len, 5)
        
        # Loss 계산을 위해 차원 변경
        # (Batch*Seq, Output_Size) vs (Batch*Seq)
        loss = criterion(outputs.view(-1, OUTPUT_SIZE), y.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 정확도 계산 (선택 사항)
        predictions = torch.argmax(outputs, dim=2)
        correct += (predictions == y).sum().item()
        total_steps += y.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_steps * 100
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

print("Training Finished!")