import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import Rectangle
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import sys
import math

# region  0. 로깅 유틸리티

class TrainingLogger:
    """
    목적: 학습 과정의 모든 로그를 파일과 콘솔에 동시에 기록

    기능:
        - 하이퍼파라미터 및 설정 정보 로깅
        - 학습 과정 및 epoch별 loss 로깅
        - 파일과 콘솔에 동시 출력
    """
    def __init__(self, log_path, mode='a'):
        """
        mode: 'w' (덮어쓰기) 또는 'a' (추가)
        """
        self.log_path = log_path
        
        # 디렉토리가 없으면 생성
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.log_file = open(log_path, mode, encoding='utf-8')
        self.start_time = datetime.now()

    def log(self, message, print_to_console=True):
        """메시지를 파일과 콘솔에 기록"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_file.write(log_message + '\n')
        self.log_file.flush()
        if print_to_console:
            print(message)

    def log_separator(self):
        """구분선 추가"""
        separator = "=" * 80
        self.log_file.write(separator + '\n')
        self.log_file.flush()
        print(separator)

    def log_header(self, title):
        """헤더 추가"""
        self.log_separator()
        self.log(title)
        self.log_separator()

    def log_hyperparameters(self, params_dict):
        """하이퍼파라미터 로깅"""
        self.log_header("하이퍼파라미터 및 설정")
        for key, value in params_dict.items():
            self.log(f"  {key}: {value}")
        self.log_separator()

    def log_epoch(self, epoch, total_epochs, train_loss, test_loss=None):
        """Epoch별 loss 로깅"""
        if test_loss is not None:
            message = f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
        else:
            message = f"Epoch {epoch}/{total_epochs} - Avg Loss: {train_loss:.6f}"
        self.log(message)

    def log_training_summary(self):
        """학습 종료 시 요약 정보"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.log_separator()
        self.log(f"학습 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"학습 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"총 학습 시간: {duration}")
        self.log_separator()

    def close(self):
        """로거 종료"""
        self.log_training_summary()
        self.log_file.close()

# endregion

# region  1. 데이터 처리 모듈 

def load_ohlc_data(csv_path):
    """
    목적: CSV 파일에서 OHLC(Open, High, Low, Close) 시계열 데이터를 로드하고 전처리
    
    입력:
        csv_path (str): CSV 파일 경로
                       예상 컬럼: ['time', 'open', 'high', 'low', 'close', ...]
    
    출력:
        ohlc (np.ndarray): OHLC 데이터, shape=(시간_길이, 4)
                          dtype=float32, 시간 순으로 정렬됨
        time_data (pd.Series): 시간 인덱스 데이터, pd.datetime 형태
    
    처리 과정:
        1. CSV 읽기 및 시간 컬럼 파싱
        2. 시간 순으로 정렬
        3. OHLC 컬럼만 추출하여 numpy 배열로 변환
    """
    print(f"[Function: load_ohlc_data] - CSV 파일 로드 시작: {csv_path}")
    
    # CSV 파일 읽기, time 컬럼을 datetime으로 파싱
    df = pd.read_csv(csv_path, parse_dates=['time'])
    
    # 시간 순으로 정렬 및 인덱스 재설정
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"  - 데이터 로드 완료. 총 {len(df)} 행")
    print(f"  - 로드된 데이터 샘플:\n{df.head()}")
    
    # OHLC 데이터만 추출하여 float32 numpy 배열로 변환
    ohlc = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
    
    print(f"  - OHLC 데이터 추출 완료. Shape: {ohlc.shape}")
    return ohlc, df['time']

def normalize_window(ohlc_window):
    """
    목적: 차트 패턴의 '모양(Shape)'에만 집중하기 위한 2단계 정규화 수행
          가격 절대값과 변동 크기에 무관하게 패턴의 형태만 추출
    
    입력:
        ohlc_window (np.ndarray): 정규화할 OHLC 윈도우 데이터
                                 shape=(시간_길이, 4), 예: (10, 4)
    
    출력:
        scale_invariant_window (np.ndarray): 정규화된 OHLC 데이터
                                           shape=(시간_길이, 4)
                                           값 범위: 대략 [-1, 1] 사이
    
    정규화 과정:
        1단계 - 시작가 기준 정규화: 
            - 시작 open 가격을 0으로 맞춤 (상대적 % 변화로 변환)
            - 공식: (price - start_open) * 100 / start_open
            
        2단계 - 크기(Scale) 정규화:
            - 패턴 내 최대 변동폭으로 나누어 [-1, 1] 범위로 스케일링
            - 공식: percent_change / max_abs_change
            
    예시:
        입력: [[100, 105, 98, 102], [102, 108, 101, 106]]
        1단계 후: [[0, 5, -2, 2], [2, 8, 1, 6]]
        # 2단계 후: [[0, 0.625, -0.25, 0.25], [0.25, 1.0, 0.125, 0.75]]
    """
    if ohlc_window.shape[0] == 0:
        return ohlc_window
    
    epsilon = 1e-8  # 0으로 나누기 방지

    # 1단계: 시작가 기준 정규화 (상대적 % 변화로 변환)
    base = ohlc_window[0, 0]  # 첫 번째 캔들의 시가를 기준점으로 사용
    percent_change_window = (ohlc_window - base) * 100.0 / (base + epsilon)

    # 2단계: 크기(Scale) 정규화 (패턴의 '모양'만 남김)
    # 창 내에서 가장 큰 변동폭(절대값 기준)을 찾습니다.
    max_abs_change = np.max(np.abs(percent_change_window))

    # 만약 변동이 전혀 없다면 (모든 값이 0), 그대로 0을 반환
    if max_abs_change < epsilon:
        print("Warning: 변동이 거의 없는 flat 패턴!")
        return percent_change_window

    # 최대 변동폭으로 나눠주어, 전체 값을 [-1, 1] 범위로 스케일링합니다.
    scale_invariant_window = percent_change_window / max_abs_change
    return scale_invariant_window

    return percent_change_window

def filter_similar_indices(patterns, index_tolerance=2):
    """
    목적: 인덱스가 비슷한 패턴들(±index_tolerance 범위)에서 가장 높은 점수만 남기기
    
    입력:
        patterns (list): 패턴 딕셔너리 리스트
                        각 딕셔너리는 {'idx': int, 'sim': float, ...} 형태
        index_tolerance (int): 인덱스 유사도 허용 범위 (기본값: 2)
    
    출력:
        filtered_patterns (list): 필터링된 패턴 리스트
    
    처리 과정:
        1. 유사도 순으로 정렬 (내림차순)
        2. 각 패턴에 대해 이미 선택된 패턴들과 인덱스 비교
        3. 인덱스 차이가 tolerance 이내면 제외, 아니면 추가
    """
    if not patterns:
        return patterns
    
    # 유사도 순으로 정렬 (높은 순서대로)
    sorted_patterns = sorted(patterns, key=lambda x: x['sim'], reverse=True)
    
    filtered_patterns = []
    
    for pattern in sorted_patterns:
        current_idx = pattern['idx']
        
        # 이미 선택된 패턴들과 인덱스 비교
        is_similar_to_existing = False
        for existing_pattern in filtered_patterns:
            existing_idx = existing_pattern['idx']
            if abs(current_idx - existing_idx) <= index_tolerance:
                is_similar_to_existing = True
                break
        
        # 비슷한 인덱스 패턴이 없으면 추가
        if not is_similar_to_existing:
            filtered_patterns.append(pattern)
    
    # 원래 순서대로 다시 정렬 (유사도 순)
    filtered_patterns.sort(key=lambda x: x['sim'], reverse=True)

    return filtered_patterns

def quantize_embeddings(embeddings):
    """
    목적: float32 임베딩을 int8로 양자화하여 메모리 75% 절감

    입력:
        embeddings (np.ndarray): float32 임베딩, shape=(N, emb_dim)

    출력:
        quantized (np.ndarray): int8 임베딩, shape=(N, emb_dim)
        scale (np.ndarray): 각 차원의 스케일 팩터, shape=(emb_dim,)
        min_vals (np.ndarray): 각 차원의 최소값, shape=(emb_dim,)

    방법:
        각 차원을 독립적으로 [min, max] → [-127, 127] 범위로 매핑
    """
    # 각 차원별 최소/최대값 계산
    min_vals = embeddings.min(axis=0)  # (emb_dim,)
    max_vals = embeddings.max(axis=0)  # (emb_dim,)

    # 스케일 계산 (0으로 나누기 방지)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # 상수 차원 처리
    scale = 254.0 / ranges  # -127~127 범위로 매핑

    # 양자화: [min, max] → [-127, 127]
    normalized = (embeddings - min_vals) * scale - 127
    quantized = np.round(normalized).astype(np.int8)

    return quantized, scale, min_vals

def dequantize_embeddings(quantized, scale, min_vals):
    """
    목적: int8 임베딩을 float32로 역양자화

    입력:
        quantized (np.ndarray): int8 임베딩, shape=(N, emb_dim)
        scale (np.ndarray): 스케일 팩터, shape=(emb_dim,)
        min_vals (np.ndarray): 최소값, shape=(emb_dim,)

    출력:
        embeddings (np.ndarray): 복원된 float32 임베딩, shape=(N, emb_dim)
    """
    # 역양자화: [-127, 127] → [min, max]
    embeddings = (quantized.astype(np.float32) + 127) / scale + min_vals
    return embeddings

# endregion

# region  2. 딥러닝 모델 및 데이터셋 모듈 

class PatternEncoder(nn.Module):
    """
    목적: 가변 길이 OHLC 시퀀스를 1차원으로 펼쳐서 고정 차원 임베딩 벡터로 인코딩
          입력을 플래튼하여 직접적인 패턴 학습 수행
    
    입력:
        x (torch.Tensor): 배치 OHLC 데이터
                         shape=(batch_size, seq_len, 4)
                         예: (32, 15, 4) - 32개 배치, 15개 캔들, OHLC
    
    출력:
        embedding (torch.Tensor): 패턴 임베딩 벡터
                                 shape=(batch_size, emb_dim)
                                 예: (32, 128) - 32개 배치, 128차원 임베딩
    
    아키텍처 구성:
        1. 입력 플래튼: (batch_size, seq_len, 4) -> (batch_size, seq_len * 4)
        2. 적응형 크기 조정: 가변 길이 입력을 고정 크기로 표준화
        3. MLP 계층: 다층 신경망으로 패턴 특징 추출
        4. 최종 임베딩: 고정 차원 임베딩 벡터 생성
    """
    def __init__(self, input_features=4, emb_dim=128, max_len=100):
        super(PatternEncoder, self).__init__()
        
        self.input_features = input_features
        self.max_len = max_len
        self.emb_dim = emb_dim
        
        # 최대 입력 크기 계산 (플래튼된 1D 벡터)
        self.max_flattened_size = max_len * input_features  # 100 * 4 = 400

        # MLP 계층 정의
        self.fc1 = nn.Linear(self.max_flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, emb_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # 배치 정규화 레이어
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        # 입력 x는 collate_fn에서 이미 (batch_size, max_len, 4)로 크기가 고정됨
        batch_size = x.shape[0]

        # 1. Flatten: 각 2D 패턴을 1D 벡터로 펼칩니다.
        # (batch_size, max_len, 4) -> (batch_size, max_len * 4)
        final_input = x.reshape(batch_size, -1)
        
        # 2. MLP 통과
        x = self.fc1(final_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        embedding = self.fc3(x)

        # sigmoid 적용
        embedding = torch.sigmoid(embedding)  # 0~1 범위로 스케일링 (선택적)
        return embedding

class VariableSeqDataset(Dataset):
    """
    목적: 가변 길이 OHLC 시퀀스 데이터셋 생성 (임베딩 사전 계산용)
          모든 가능한 (시작점, 길이) 조합을 생성하여 전체 패턴 커버

    입력:
        ohlc_data (np.ndarray): 전체 OHLC 데이터, shape=(전체_길이, 4)
        min_len (int): 최소 시퀀스 길이 (기본값: 3) - specific_lengths가 None일 때만 사용
        max_len (int): 최대 시퀀스 길이 (기본값: 100) - specific_lengths가 None일 때만 사용
        specific_lengths (list): 특정 길이 리스트 (기본값: None). 제공되면 이 길이들만 사용

    출력 (__getitem__):
        torch.Tensor: 정규화된 OHLC 시퀀스, shape=(seq_len, 4)

    인덱스 생성 예시:
        specific_lengths=[3,5,7]인 경우
        인덱스: [(0,3), (0,5), (0,7), (1,3), (1,5), (1,7), ..., (995,3), (995,5), (995,7)]
    """
    def __init__(self, ohlc_data, min_len=3, max_len=100, specific_lengths=None):
        if specific_lengths is not None:
            print(f"[Function: VariableSeqDataset] - 데이터셋 생성. Specific Lengths: {specific_lengths}")
            self.specific_lengths = specific_lengths
            self.min_len = min(specific_lengths)
            self.max_len = max(specific_lengths)
        else:
            print(f"[Function: VariableSeqDataset] - 데이터셋 생성. Min/Max Length: {min_len}/{max_len}")
            self.specific_lengths = None
            self.min_len = min_len
            self.max_len = max_len

        self.ohlc_data = ohlc_data
        self.indices = self._create_indices()
        print(f"  - 총 {len(self.indices)}개의 가변 길이 샘플 인덱스 생성 완료.")
        print(f"  - 샘플 인덱스 예시 (첫 5개): {self.indices[:5]}")

    def _create_indices(self):
        """
        목적: 모든 가능한 (시작_인덱스, 시퀀스_길이) 조합 생성

        출력:
            indices (list): [(start_idx, seq_len), ...] 형태의 리스트
        """
        indices = []

        # specific_lengths가 제공되었다면 해당 길이들만 사용
        if self.specific_lengths is not None:
            for i in range(len(self.ohlc_data)):
                for length in self.specific_lengths:
                    # 해당 시작점에서 length 길이만큼의 데이터가 있는지 확인
                    if i + length <= len(self.ohlc_data):
                        indices.append((i, length))
        else:
            # 기존 로직: min_len부터 max_len까지 모든 길이 조합 생성
            for i in range(len(self.ohlc_data) - self.min_len):
                # 해당 시작점에서 가능한 최대 길이 계산
                max_possible_len = min(self.max_len, len(self.ohlc_data) - i)

                # min_len부터 max_possible_len까지 모든 길이 조합 생성
                for length in range(self.min_len, max_possible_len + 1):
                    indices.append((i, length))

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        목적: 인덱스로부터 정규화된 OHLC 시퀀스 반환
        
        입력:
            idx (int): 데이터셋 인덱스
        
        출력:
            torch.Tensor: 정규화된 OHLC 시퀀스
        """
        start, length = self.indices[idx]
        window = self.ohlc_data[start : start + length]
        return torch.from_numpy(normalize_window(window))

def collate_fn(batch):
    """
    목적: 가변 길이 시퀀스들을 배치로 묶되, MAX_PATTERN_LEN으로 0패딩 적용
    입력: list[tensor(seq_len, 4)]
    출력: tensor(batch, MAX_PATTERN_LEN, 4)
    """
    MAX_LEN = 100  # 전역 설정 MAX_PATTERN_LEN과 동일하게 유지
    resized = []
    for x in batch:
        # x: (L, 4) 형태의 텐서
        current_len = x.shape[0]

        # MAX_LEN으로 0패딩
        if current_len < MAX_LEN:
            # (MAX_LEN - current_len, 4) 크기의 0 텐서를 생성하여 끝에 붙임
            padding = torch.zeros(MAX_LEN - current_len, 4, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=0)
        else:
            # 이미 MAX_LEN 이상이면 자르기
            x_padded = x[:MAX_LEN]

        resized.append(x_padded)
    return torch.stack(resized, dim=0)

class TripletDataset(Dataset):
    """
    목적: Triplet Loss 학습을 위한 데이터셋
          Anchor, Positive(유사), Negative(비유사) 삼중쌍 생성
          **모든 길이 × 증강 조합을 데이터셋에 포함하여 균등한 학습 보장**

    입력:
        ohlc_data (np.ndarray): 전체 OHLC 데이터
        min_len (int): 최소 시퀀스 길이 (기본값: 3) - specific_lengths가 None일 때만 사용
        max_len (int): 최대 시퀀스 길이 (기본값: 100) - specific_lengths가 None일 때만 사용
        specific_lengths (list): 특정 패턴 길이 리스트 (기본값: None).
                                제공되면 이 길이들 각각에 대해 모든 증강 조합을 생성

    출력 (__getitem__):
        tuple: (anchor_tensor, positive_tensor, negative_tensor)
               각각 shape=(seq_len, 4)인 torch.Tensor

    데이터 증강 방식:
        - 각 (Anchor 위치 × 패턴 길이)에 대해 모든 Positive/Negative 증강 조합을 생성.
        - Positive: ['noise', 'scale']
        - Negative: ['reverse_noise', 'noise', 'pure_random']
        - 총 샘플 수: anchor_indices × len(lengths) × 3 × 3
    """
    def __init__(self, ohlc_data, min_len=3, max_len=100, specific_lengths=None):
        print(f"[Function: TripletDataset] - 데이터셋 생성 (모든 증강 조합 사용). Min/Max Length: {min_len}/{max_len}")
        self.ohlc_data = ohlc_data
        self.min_len = min_len
        self.max_len = max_len
        self.specific_lengths = specific_lengths

        if specific_lengths is not None:
            print(f"  - 특정 패턴 길이 사용: {specific_lengths}")
        
        # 증강 방식 정의
        self.pos_aug_types = ['noise', 'scale']
        # Hard negative mining 추가: 실제로 구별하기 어려운 패턴 학습
        self.neg_aug_types = ['hard_negative', 'semi_hard', 'reverse_noise', 'reverse_scale']
        
        # 길이 목록 결정
        if specific_lengths is not None:
            lengths_to_use = specific_lengths
        else:
            lengths_to_use = list(range(self.min_len, self.max_len + 1))

        # 모든 (anchor_start, anchor_len, pos_type, neg_type) 조합을 인덱스로 생성
        anchor_indices = list(range(len(self.ohlc_data) - self.max_len))
        self.triplet_definitions = []
        for anchor_start in anchor_indices:
            for anchor_len in lengths_to_use:
                for pos_type in self.pos_aug_types:
                    for neg_type in self.neg_aug_types:
                        self.triplet_definitions.append((anchor_start, anchor_len, pos_type, neg_type))

        print(f"  - 총 {len(anchor_indices)}개의 Anchor 샘플로부터")
        print(f"  - {len(lengths_to_use)}개의 길이 × {len(self.pos_aug_types)}개의 Positive 증강 × {len(self.neg_aug_types)}개의 Negative 증강을 조합하여")
        print(f"  - 총 {len(self.triplet_definitions)}개의 Triplet 샘플 생성 완료.")

    def __len__(self):
        return len(self.triplet_definitions)

    def __getitem__(self, idx):
        """
        목적: Triplet (Anchor, Positive, Negative) 생성
        
        처리 과정:
            1. 정의된 Triplet 조합(anchor_start, pos_type, neg_type)을 가져옴
            2. Anchor: 원본 데이터에서 랜덤 길이 패턴 추출
            3. Positive: 지정된 pos_type으로 증강
            4. Negative: 지정된 neg_type으로 증강
        """
        # 1. Triplet 정의 가져오기 (길이 정보 포함)
        anchor_start, anchor_len, pos_type, neg_type = self.triplet_definitions[idx]
        
        # Anchor 샘플 가져오기 (이미 길이가 정의되어 있음)
        anchor_raw = self.ohlc_data[anchor_start : anchor_start + anchor_len]
        anchor = normalize_window(anchor_raw)

        # 2. Positive 샘플 생성 (지정된 타입)
        if pos_type == 'noise':
            # 작은 노이즈 추가로 유사하지만 다른 패턴 생성
            positive = anchor + np.random.uniform(-0.2, 0.2, anchor.shape)
            # print("anchor noise:", anchor, "\n", "anchor noise shape:", anchor.shape, "\n")
            # print("positive noise:", positive, "\n", "positive noise shape:", positive.shape, "\n")
        elif pos_type == 'scale':
            # 스케일 변경으로 크기는 다르지만 모양은 유사한 패턴 생성
            positive = anchor * np.random.uniform(0.9, 1.1, anchor.shape)
            # print("anchor scale:", anchor, "\n", "anchor scale shape:", anchor.shape, "\n")
            # print("positive scale:", positive, "\n", "positive scale shape:", positive.shape, "\n")


        # 3. Negative 샘플 생성 (지정된 타입)
        if neg_type == 'hard_negative':
            # Hard Negative: 비슷해 보이지만 실제로는 다른 패턴 (다른 시간대의 패턴)
            # anchor와 최소 100 캔들 떨어진 위치에서 같은 길이의 패턴 추출
            min_gap = 100
            max_attempts = 10
            found = False

            for _ in range(max_attempts):
                # 전체 데이터 범위에서 랜덤 위치 선택
                neg_start = np.random.randint(0, len(self.ohlc_data) - anchor_len)

                # anchor와 충분히 떨어져 있는지 확인
                if abs(neg_start - anchor_start) >= min_gap:
                    negative_raw = self.ohlc_data[neg_start : neg_start + anchor_len]
                    negative = normalize_window(negative_raw)
                    found = True
                    break

            if not found:
                # fallback: 랜덤 위치 (gap 제약 없이)
                neg_start = np.random.randint(0, len(self.ohlc_data) - anchor_len)
                negative_raw = self.ohlc_data[neg_start : neg_start + anchor_len]
                negative = normalize_window(negative_raw)
        elif neg_type == 'semi_hard':
            # Semi-Hard Negative: anchor와 유사하지만 약간 변형된 패턴
            # 큰 노이즈 또는 다른 길이의 패턴을 같은 길이로 리샘플링
            negative = anchor + np.random.uniform(-0.5, 0.5, anchor.shape)
        elif neg_type == 'reverse_noise':
            # 완전 반전 + 큰 노이즈로 확실하게 다른 패턴 생성
            negative = - anchor - np.random.uniform(-2.0, 2.0, anchor.shape)
        elif neg_type == 'reverse_scale':
            negative = - anchor * np.random.uniform(0.5, 2.0, anchor.shape)

        # # 3. Negative 샘플 생성 (지정된 타입)
        # if neg_type == 'reverse_noise':
        #     # 완전 반전 + 큰 노이즈로 확실하게 다른 패턴 생성
        #     negative = -anchor - np.random.uniform(-0.1, 0.1, anchor.shape)
        # elif neg_type == 'noise':
        #     # 큰 노이즈로 원본과 다른 패턴 생성
        #     negative = anchor.copy()
        #     # negative += np.random.uniform(-1.0, 1.0, anchor.shape)
        #     negative = - negative * np.random.uniform(0.9, 1.1, anchor.shape)
        
        # # 예시 거리 계산
        # anchor_tensor = torch.from_numpy(anchor.astype(np.float32))
        # positive_tensor = torch.from_numpy(positive.astype(np.float32))
        # negative_tensor = torch.from_numpy(negative.astype(np.float32))
        # print(f"Anchor-Positive 거리: {torch.norm(anchor_tensor - positive_tensor, p=2).item()}")
        # print(f"Anchor-Negative 거리: {torch.norm(anchor_tensor - negative_tensor, p=2).item()}")

        return (torch.from_numpy(anchor.astype(np.float32)), 
                torch.from_numpy(positive.astype(np.float32)), 
                torch.from_numpy(negative.astype(np.float32)))

def collate_fn_triplet(batch):
    """
    목적: Triplet 배치를 위한 0패딩 기반 collate
    입력: [(anchor, positive, negative), ...]
    출력: (anchors, positives, negatives) 각각 (B, MAX_LEN, 4)
    """
    MAX_LEN = 100
    anchors, positives, negatives = zip(*batch)

    def _resize_list(tensors):
        out = []
        for x in tensors:
            current_len = x.shape[0]
            # MAX_LEN으로 0패딩
            if current_len < MAX_LEN:
                padding = torch.zeros(MAX_LEN - current_len, 4, dtype=x.dtype)
                x_padded = torch.cat([x, padding], dim=0)
            else:
                x_padded = x[:MAX_LEN]
            out.append(x_padded)
        return torch.stack(out, dim=0)

    padded_anchors = _resize_list(anchors)
    padded_positives = _resize_list(positives)
    padded_negatives = _resize_list(negatives)
    return padded_anchors, padded_positives, padded_negatives

class TripletMarginLoss(nn.Module):
    """
    목적: Triplet Loss 구현 (Anchor-Positive-Negative 쌍을 위한 손실 함수)
    
    입력:
        margin (float): 앵커와 네거티브 간의 최소 거리 마진
        p (int): 거리 계산에 사용할 노름의 차수 (예: 2는 유클리드 거리)
    
    출력:
        loss (torch.Tensor): 계산된 배치 손실 값
    """
    def __init__(self, margin=0.4, p=2):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor, positive, negative):
        """
        목적: 주어진 앵커, 포지티브, 네거티브 임베딩에 대해 손실 계산
        
        입력:
            anchor (torch.Tensor): 앵커 임베딩, shape=(배치_크기, 임베딩_차원)
            positive (torch.Tensor): 포지티브 임베딩, shape=(배치_크기, 임베딩_차원)
            negative (torch.Tensor): 네거티브 임베딩, shape=(배치_크기, 임베딩_차원)
        
        출력:
            loss (torch.Tensor): 계산된 배치 손실 값
        """
        # 앵커-포지티브 거리
        pos_dist = torch.norm(anchor - positive, p=self.p, dim=1)
        # 앵커-네거티브 거리
        neg_dist = torch.norm(anchor - negative, p=self.p, dim=1)
        
        # Triplet Loss 계산
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        return loss

def evaluate_retriever_recall(model, test_data, device, k_values=[10, 50, 100], num_samples=500, max_len=100):
    """
    목적: Retriever 모델의 Recall@K 평가

    입력:
        model (PatternEncoder): 학습된 리트리버 모델
        test_data (np.ndarray): 테스트 OHLC 데이터
        device (torch.device): 계산 장치
        k_values (list): 평가할 K 값들 (예: [10, 50, 100])
        num_samples (int): 평가에 사용할 샘플 수
        max_len (int): 최대 패턴 길이

    출력:
        dict: {k: recall@k} 형태의 딕셔너리

    설명:
        Recall@K = (Top-K 예측 중 실제 유사한 패턴 수) / (전체 유사한 패턴 수)
        여기서는 같은 패턴에 노이즈를 추가한 것을 "유사"로 정의
    """
    print(f"\n[Evaluation] Recall@K 평가 시작 ({num_samples} 샘플)")
    model.eval()

    recalls = {k: [] for k in k_values}
    max_k = max(k_values)

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="  - Recall@K 계산"):
            # 랜덤 anchor 패턴 선택
            anchor_len = np.random.randint(3, min(30, max_len))
            anchor_start = np.random.randint(0, len(test_data) - anchor_len)
            anchor_raw = test_data[anchor_start : anchor_start + anchor_len]
            anchor = normalize_window(anchor_raw)

            # Ground truth: anchor에 작은 노이즈 추가한 positive 샘플 3개 생성
            positives = []
            for _ in range(3):
                pos = anchor + np.random.uniform(-0.15, 0.15, anchor.shape)
                positives.append(pos)

            # 랜덤 negative 샘플들 생성 (max_k개)
            negatives = []
            for _ in range(max_k):
                neg_start = np.random.randint(0, len(test_data) - anchor_len)
                # anchor와 멀리 떨어진 위치에서 샘플링
                while abs(neg_start - anchor_start) < 50:
                    neg_start = np.random.randint(0, len(test_data) - anchor_len)
                neg_raw = test_data[neg_start : neg_start + anchor_len]
                neg = normalize_window(neg_raw)
                negatives.append(neg)

            # 모든 후보 패턴 (positive + negative)
            all_candidates = positives + negatives

            # Anchor 임베딩 계산
            anchor_tensor = torch.from_numpy(anchor.astype(np.float32)).unsqueeze(0).to(device)
            if anchor_tensor.shape[1] < max_len:
                padding = torch.zeros(1, max_len - anchor_tensor.shape[1], 4, dtype=anchor_tensor.dtype, device=device)
                anchor_tensor = torch.cat([anchor_tensor, padding], dim=1)
            anchor_emb = model(anchor_tensor).cpu().numpy().flatten()

            # 모든 후보의 임베딩 계산
            candidate_embs = []
            for cand in all_candidates:
                cand_tensor = torch.from_numpy(cand.astype(np.float32)).unsqueeze(0).to(device)
                if cand_tensor.shape[1] < max_len:
                    padding = torch.zeros(1, max_len - cand_tensor.shape[1], 4, dtype=cand_tensor.dtype, device=device)
                    cand_tensor = torch.cat([cand_tensor, padding], dim=1)
                cand_emb = model(cand_tensor).cpu().numpy().flatten()
                candidate_embs.append(cand_emb)

            # 코사인 유사도 계산
            candidate_embs = np.array(candidate_embs)
            similarities = F.cosine_similarity(
                torch.from_numpy(anchor_emb).unsqueeze(0),
                torch.from_numpy(candidate_embs)
            ).numpy()

            # Top-K 추출
            top_k_indices = np.argsort(similarities)[::-1][:max_k]

            # 각 K에 대해 Recall 계산
            for k in k_values:
                top_k_idx = top_k_indices[:k]
                # positive는 인덱스 0, 1, 2
                found = sum(1 for idx in top_k_idx if idx < 3)
                recall = found / 3.0  # 3개의 positive 중 몇 개를 찾았는가
                recalls[k].append(recall)

    # 평균 계산
    avg_recalls = {k: np.mean(recalls[k]) for k in k_values}

    print("\n  - Recall@K 결과:")
    for k, recall in avg_recalls.items():
        print(f"    Recall@{k}: {recall:.4f}")

    return avg_recalls

def evaluate_mrr(model, test_data, device, num_samples=500, max_len=100):
    """
    목적: Mean Reciprocal Rank (MRR) 평가

    입력:
        model (PatternEncoder): 학습된 리트리버 모델
        test_data (np.ndarray): 테스트 OHLC 데이터
        device (torch.device): 계산 장치
        num_samples (int): 평가에 사용할 샘플 수
        max_len (int): 최대 패턴 길이

    출력:
        float: MRR 값 (0~1, 높을수록 좋음)

    설명:
        MRR = Average(1 / rank of first relevant item)
        가장 유사한 패턴이 몇 번째에 나오는지 평가
    """
    print(f"\n[Evaluation] MRR 평가 시작 ({num_samples} 샘플)")
    model.eval()

    reciprocal_ranks = []

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="  - MRR 계산"):
            # 랜덤 anchor 패턴 선택
            anchor_len = np.random.randint(3, min(30, max_len))
            anchor_start = np.random.randint(0, len(test_data) - anchor_len)
            anchor_raw = test_data[anchor_start : anchor_start + anchor_len]
            anchor = normalize_window(anchor_raw)

            # Ground truth positive (가장 유사해야 할 패턴)
            true_positive = anchor + np.random.uniform(-0.1, 0.1, anchor.shape)

            # 랜덤 negative 샘플들 (99개)
            candidates = [true_positive]
            for _ in range(99):
                neg_start = np.random.randint(0, len(test_data) - anchor_len)
                while abs(neg_start - anchor_start) < 50:
                    neg_start = np.random.randint(0, len(test_data) - anchor_len)
                neg_raw = test_data[neg_start : neg_start + anchor_len]
                neg = normalize_window(neg_raw)
                candidates.append(neg)

            # Anchor 임베딩
            anchor_tensor = torch.from_numpy(anchor.astype(np.float32)).unsqueeze(0).to(device)
            if anchor_tensor.shape[1] < max_len:
                padding = torch.zeros(1, max_len - anchor_tensor.shape[1], 4, dtype=anchor_tensor.dtype, device=device)
                anchor_tensor = torch.cat([anchor_tensor, padding], dim=1)
            anchor_emb = model(anchor_tensor).cpu().numpy().flatten()

            # 모든 후보 임베딩
            candidate_embs = []
            for cand in candidates:
                cand_tensor = torch.from_numpy(cand.astype(np.float32)).unsqueeze(0).to(device)
                if cand_tensor.shape[1] < max_len:
                    padding = torch.zeros(1, max_len - cand_tensor.shape[1], 4, dtype=cand_tensor.dtype, device=device)
                    cand_tensor = torch.cat([cand_tensor, padding], dim=1)
                cand_emb = model(cand_tensor).cpu().numpy().flatten()
                candidate_embs.append(cand_emb)

            # 유사도 계산 및 순위 매기기
            candidate_embs = np.array(candidate_embs)
            similarities = F.cosine_similarity(
                torch.from_numpy(anchor_emb).unsqueeze(0),
                torch.from_numpy(candidate_embs)
            ).numpy()

            # true_positive의 순위 찾기 (인덱스 0)
            rank = np.where(np.argsort(similarities)[::-1] == 0)[0][0] + 1  # 1-based rank
            reciprocal_ranks.append(1.0 / rank)

    mrr = np.mean(reciprocal_ranks)
    print(f"\n  - MRR: {mrr:.4f}")

    return mrr

# endregion

# region  3. 모델 학습 및 로드 모듈 

def train_or_load_model(train_ohlc_data, test_ohlc_data, emb_dim, model_path, force_train=False, max_len=100, logger=None):
    """
    목적: PatternEncoder 모델을 학습하거나 기존 모델을 로드

    입력:
        train_ohlc_data (np.ndarray): 학습용 OHLC 데이터
        test_ohlc_data (np.ndarray): 테스트용 OHLC 데이터
        emb_dim (int): 임베딩 차원 크기
        model_path (str): 모델 저장/로드 경로
        force_train (bool): True면 강제로 재학습, False면 기존 모델 로드 시도
        max_len (int): 학습에 사용할 최대 패턴 길이
        logger (TrainingLogger): 학습 로거 객체

    출력:
        model (PatternEncoder): 학습된 또는 로드된 모델 (GPU 또는 CPU에 로드됨)

    학습 과정:
        1. TripletDataset으로 데이터 준비
        2. TripletMarginLoss로 유사 패턴은 가깝게, 비유사 패턴은 멀게 학습
        3. 각 에포크마다 학습 및 테스트 손실을 계산하여 출력
    """
    print(f"[Function: train_or_load_model] - 모델 학습 또는 로드 시작. Path: {model_path}")
    if logger:
        logger.log("[Function: train_or_load_model] - Retriever 모델 학습 또는 로드 시작.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device 상세 정보
    device_info = {
        "torch 버전": torch.__version__,
        "CUDA 사용 가능": torch.cuda.is_available(),
        "현재 디바이스": str(device),
        "GPU 이름": torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        "GPU 메모리 총량 (GB)": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}" if torch.cuda.is_available() else 'N/A',
        "GPU 메모리 사용량 (GB)": f"{torch.cuda.memory_allocated(0) / (1024**3):.2f}" if torch.cuda.is_available() else 'N/A',
        "GPU 메모리 캐시 (GB)": f"{torch.cuda.memory_reserved(0) / (1024**3):.2f}" if torch.cuda.is_available() else 'N/A',
        "CPU 코어 수": os.cpu_count()
    }

    for key, value in device_info.items():
        print(f"    - {key}: {value}")
        if logger:
            logger.log(f"    - {key}: {value}")

    model = PatternEncoder(emb_dim=emb_dim, max_len=max_len).to(device)

    # 기존 모델이 존재하고 재학습을 강제하지 않으면 로드
    if os.path.exists(model_path) and not force_train:
        print(f"  - 저장된 모델 로드: '{model_path}'")
        if logger:
            logger.log(f"  - 저장된 모델 로드: '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # 모델을 평가 모드로 설정
        return model

    print("  - 새로운 모델 학습 시작.")
    if logger:
        logger.log_header("Retriever (PatternEncoder) 모델 학습")
        logger.log(f"  - 모델 경로: {model_path}")
        logger.log(f"  - Embedding Dimension: {emb_dim}")
        logger.log(f"  - Max Length: {max_len}")

    # Triplet Loss 학습을 위한 데이터셋 생성
    train_dataset = TripletDataset(train_ohlc_data, max_len=max_len, specific_lengths= SPECIFIC_PATTERN_LENGTHS)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                                  collate_fn=collate_fn_triplet, num_workers=0)

    test_dataset = TripletDataset(test_ohlc_data, max_len=max_len, specific_lengths= SPECIFIC_PATTERN_LENGTHS)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                                 collate_fn=collate_fn_triplet, num_workers=0)

    # 옵티마이저 및 손실 함수 설정
    learning_rate = 5e-4
    margin = 0.4
    batch_size = 256
    num_epochs = 30

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 낮은 학습률로 안정적 학습
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    if logger:
        logger.log(f"  - 학습 샘플 수: {len(train_dataset)}")
        logger.log(f"  - 테스트 샘플 수: {len(test_dataset)}")
        logger.log(f"  - Batch Size: {batch_size}")
        logger.log(f"  - Learning Rate: {learning_rate}")
        logger.log(f"  - Epochs: {num_epochs}")
        logger.log(f"  - Loss Function: TripletMarginLoss (margin={margin}, p=2)")
        logger.log(f"  - Specific Pattern Lengths: {SPECIFIC_PATTERN_LENGTHS}")
        logger.log_separator()

    print(f"  - {len(train_dataset)}개의 학습 샘플과 {len(test_dataset)}개의 테스트 샘플로 학습 시작 (Device: {device})")

    # 학습 루프
    for epoch in range(num_epochs):
        # --- 학습 단계 ---
        model.train()
        total_train_loss = 0

        for i, (anchor, positive, negative) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train")):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- 평가 단계 ---
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for i, (anchor, positive, negative) in enumerate(tqdm(test_dataloader, desc=f"Epoch {epoch+1} Eval ")):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)

                loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_dataloader)

        print(f"  - Epoch {epoch+1}/{num_epochs} 완료, Avg Train Loss: {avg_train_loss:.6f}, Avg Test Loss: {avg_test_loss:.6f}")
        if logger:
            logger.log_epoch(epoch+1, num_epochs, avg_train_loss, avg_test_loss)

        # 매 5 에포크마다 평가 메트릭 계산 (선택적)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            print(f"\n  - Epoch {epoch+1}: 평가 메트릭 계산 중...")
            recalls = evaluate_retriever_recall(model, test_ohlc_data, device, k_values=[10, 50, 100], num_samples=200, max_len=max_len)
            mrr = evaluate_mrr(model, test_ohlc_data, device, num_samples=200, max_len=max_len)

            if logger:
                logger.log(f"\n  - Epoch {epoch+1} 평가 메트릭:")
                for k, recall in recalls.items():
                    logger.log(f"    Recall@{k}: {recall:.4f}")
                logger.log(f"    MRR: {mrr:.4f}")
                logger.log_separator()

    # 학습 완료 후 최종 평가
    print(f"\n--- 학습 완료 후 최종 평가 ---")
    final_recalls = evaluate_retriever_recall(model, test_ohlc_data, device, k_values=[10, 50, 100], num_samples=500, max_len=max_len)
    final_mrr = evaluate_mrr(model, test_ohlc_data, device, num_samples=500, max_len=max_len)

    if logger:
        logger.log_header("최종 평가 결과")
        for k, recall in final_recalls.items():
            logger.log(f"  Recall@{k}: {recall:.4f}")
        logger.log(f"  MRR: {final_mrr:.4f}")
        logger.log_separator()

    # 학습 완료 후 모델 저장
    print(f"  - 학습 완료. 모델 저장: '{model_path}'")
    if logger:
        logger.log(f"  - 학습 완료. 모델 저장: '{model_path}'")

    torch.save(model.state_dict(), model_path)
    return model

# endregion

# region  4. 임베딩 계산 및 유사도 검색 모듈 

def precompute_and_save_embeddings(full_ohlc_data, model, emb_path, min_len=3, max_len=100, specific_lengths=None, use_int8=False):
    """
    목적: 전체 데이터의 모든 가능한 패턴에 대한 임베딩을 사전 계산하여 저장
          실시간 검색 시 빠른 속도를 위한 전처리 작업

    입력:
        full_ohlc_data (np.ndarray): 전체 OHLC 데이터
        model (PatternEncoder): 학습된 인코더 모델
        emb_path (str): 임베딩 저장 경로
        min_len, max_len (int): 패턴 길이 범위 (specific_lengths가 None일 때만 사용)
        specific_lengths (list): 특정 길이 리스트 (예: [3,4,5,6,7,8,9,10,11,12,13,20,30,50,60,100])
        use_int8 (bool): True면 int8 양자화 적용 (메모리 75% 절감)

    출력:
        embedding_data (dict): 임베딩 데이터 딕셔너리
            'embeddings': np.ndarray, shape=(총_패턴_수, emb_dim) - int8 or float32
            'indices': list, [(start_idx, length), ...] 각 임베딩에 대응하는 인덱스
            'quantized': bool, 양자화 여부
            'scale': np.ndarray (양자화된 경우), 역양자화용 스케일
            'min_vals': np.ndarray (양자화된 경우), 역양자화용 최소값

    처리 과정:
        1. VariableSeqDataset으로 모든 가능한 패턴 생성
        2. 배치 단위로 임베딩 계산 (GPU 메모리 효율성)
        3. (선택적) int8 양자화 적용
        4. pickle 형태로 저장하여 빠른 로드 지원
    """
    print(f"[Function: precompute_and_save_embeddings] - 임베딩 사전 계산 시작...")
    device = next(model.parameters()).device # 현재 디바이스 정보 가져오기
    model.eval()  # 평가 모드로 설정 (dropout 등 비활성화)

    # 사전 계산 시에는 모든 가능한 패턴을 생성해야 하므로 VariableSeqDataset 사용
    dataset = VariableSeqDataset(full_ohlc_data, min_len=min_len, max_len=max_len, specific_lengths=SPECIFIC_PATTERN_LENGTHS)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    all_embeddings = []
    
    # 그래디언트 계산 비활성화로 메모리 절약 및 속도 향상
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  - 전체 임베딩 계산 중"):
            inputs = batch.to(device)  # (batch_size, max_seq_len, 4)
            embeddings_batch = model(inputs)  # (batch_size, emb_dim)
            all_embeddings.append(embeddings_batch.cpu().numpy())
    
    # 모든 배치의 임베딩을 하나로 합침
    all_embeddings = np.vstack(all_embeddings)  # (총_패턴_수, emb_dim)

    print(f"  - 사전 계산된 임베딩 Shape: {all_embeddings.shape}")
    print(f"  - 첫 번째 임베딩 벡터 (샘플): {all_embeddings[0][:5]}...")

    # Int8 양자화 적용 여부
    if use_int8:
        print("  - Int8 양자화 적용 중...")
        original_size = all_embeddings.nbytes / (1024 * 1024)  # MB
        quantized_emb, scale, min_vals = quantize_embeddings(all_embeddings)
        quantized_size = quantized_emb.nbytes / (1024 * 1024)  # MB

        print(f"    - 원본 크기: {original_size:.2f} MB (float32)")
        print(f"    - 양자화 후: {quantized_size:.2f} MB (int8)")
        print(f"    - 절감률: {(1 - quantized_size/original_size)*100:.1f}%")

        # 양자화된 임베딩과 복원 정보 저장
        embedding_data = {
            'embeddings': quantized_emb,
            'indices': dataset.indices,
            'quantized': True,
            'scale': scale,
            'min_vals': min_vals
        }
    else:
        # 기존 float32 저장
        embedding_data = {
            'embeddings': all_embeddings,
            'indices': dataset.indices,
            'quantized': False
        }

    # pickle로 저장 (numpy 배열과 리스트를 효율적으로 저장)
    with open(emb_path, 'wb') as f:
        pickle.dump(embedding_data, f)

    file_size = os.path.getsize(emb_path) / (1024 * 1024)  # MB
    print(f"  - 총 {len(all_embeddings)}개의 임베딩 계산 및 저장 완료")
    print(f"  - 저장 경로: {emb_path}")
    print(f"  - 파일 크기: {file_size:.2f} MB")
    return embedding_data

def find_similar_patterns(query_emb, embedding_data, device, top_k=3, target_length=None, power=21, exclude_range=None):
    """
    목적: 쿼리 임베딩과 유사한 패턴들을 사전 계산된 임베딩에서 빠르게 검색 (GPU 가속)

    입력:
        query_emb (np.ndarray): 쿼리 패턴의 임베딩 벡터, shape=(emb_dim,)
        embedding_data (dict): 사전 계산된 임베딩 데이터 (양자화 포함 가능)
        device (torch.device): 계산을 수행할 장치 (cuda or cpu)
        top_k (int): 반환할 상위 유사 패턴 개수
        target_length (int, optional): 특정 길이의 패턴만 검색할 경우 지정
        exclude_range (tuple, optional): 제외할 인덱스 범위 (start_idx, end_idx)

    출력:
        list: 유사 패턴 정보 리스트
              [{'sim': float, 'idx': int, 'len': int}, ...]

    검색 과정 (GPU 가속):
        1. (양자화된 경우) 임베딩을 float32로 역양자화
        2. 모든 임베딩과 쿼리 임베딩을 GPU 텐서로 변환
        3. F.cosine_similarity를 사용하여 모든 유사도를 한 번의 연산으로 병렬 계산
        4. torch.topk를 사용하여 가장 유사한 후보군을 GPU에서 빠르게 추출
        5. CPU로 결과를 가져와 후처리 및 필터링
    """
    print(f"[Function: find_similar_patterns] - GPU 가속 검색 시작...")
    if target_length:
        print(f"  - 목표 패턴 길이: {target_length}")
    if exclude_range:
        print(f"  - 제외 범위: {exclude_range[0]} ~ {exclude_range[1]}")

    all_embeddings_np = embedding_data['embeddings']
    indices = embedding_data['indices']

    # Int8 양자화된 임베딩이면 역양자화
    if embedding_data.get('quantized', False):
        print("  - Int8 양자화된 임베딩 감지, 역양자화 중...")
        all_embeddings_np = dequantize_embeddings(
            all_embeddings_np,
            embedding_data['scale'],
            embedding_data['min_vals']
        )
        print("  - 역양자화 완료")

    # 1. 데이터를 GPU 텐서로 이동
    query_tensor = torch.from_numpy(query_emb).to(device)
    all_embeddings_tensor = torch.from_numpy(all_embeddings_np).to(device)

    # 2. 모든 코사인 유사도를 GPU에서 병렬로 한 번에 계산
    # (1, emb_dim)과 (N, emb_dim)을 비교하여 (N,) 크기의 유사도 텐서 생성
    print("  - GPU에서 모든 유사도 병렬 계산 중...")
    sims_tensor = F.cosine_similarity(query_tensor.unsqueeze(0), all_embeddings_tensor)
    print("  - 유사도 계산 완료.")

    # 3. GPU에서 Top-K 후보를 빠르게 찾기
    # 필터링(길이, 자기자신)을 위해 최종 top_k보다 훨씬 많은 후보를 미리 추출
    candidate_k = min(top_k, len(sims_tensor))
    top_sims, top_indices = torch.topk(sims_tensor, k=candidate_k)

    # 4. 후처리를 위해 결과를 CPU로 다시 가져옴
    top_sims_cpu = top_sims.cpu().numpy()
    top_indices_cpu = top_indices.cpu().numpy()

    # 5. 후처리 및 필터링
    similarities = []
    for i in range(len(top_sims_cpu)):
        original_idx = top_indices_cpu[i]
        sim_score = top_sims_cpu[i]

        # 자기 자신 필터링 (유사도 == 1)
        if sim_score == 1:
            continue

        # power 함수 적용
        power_val = round(10000 / power) if round(10000 / power) % 2 == 1 else round(10000 / power) + 1
        final_sim = np.sign(sim_score) * np.power(np.abs(sim_score), power_val)

        info = {
            'sim': final_sim, # 보정된 유사도
            # 'sim': sim_score,  # 원본 유사도
            'idx': indices[original_idx][0],
            'len': indices[original_idx][1]
        }

        # Input pattern 범위 제외 필터링
        if exclude_range is not None:
            pattern_start = info['idx']
            pattern_end = pattern_start + info['len'] - 1
            exclude_start, exclude_end = exclude_range

            # 패턴이 제외 범위와 겹치는지 확인
            if not (pattern_end < exclude_start or pattern_start > exclude_end):
                continue

        # 특정 길이 필터링 (target_length가 지정된 경우)
        if target_length is not None:
            if info['len'] == target_length:
                similarities.append(info)
        else:
            similarities.append(info)

    # 최종적으로 top_k 개수만큼만 반환
    final_results = similarities[:top_k]

    print(f"  - 최종 필터링 후 Top {len(final_results)}: {final_results[:3]}")
    return final_results

# endregion

# region  5. 시각화 모듈 

def plot_candlestick(ax, ohlc_data, x_offset=0, width=0.8, color_override=None, alpha=1.0):
    """
    목적: matplotlib 축에 캔들스틱 차트를 그리는 헬퍼 함수
    
    입력:
        ax (matplotlib.axes): 그릴 축 객체
        ohlc_data (np.ndarray): OHLC 데이터, shape=(시간, 4)
        x_offset (float): X축 오프셋 (여러 차트를 나란히 그릴 때 사용)
        width (float): 캔들 몸통 폭
        color_override (str): 색상 강제 지정 ('green', 'red' 등)
        alpha (float): 투명도 (0.0~1.0, 예측 구간 표시 시 0.5 등 사용)
    
    출력:
        None (ax 객체에 직접 그림을 추가)
    
    그리기 방식:
        - 상승 캔들: 녹색 (종가 >= 시가)
        - 하락 캔들: 빨간색 (종가 < 시가)
        - 고가-저가: 검은 선
        - 시가-종가: 색칠된 사각형
    """
    for i, (o, h, l, c) in enumerate(ohlc_data):
        x = i + x_offset
        
        # 색상 결정: 강제 지정 or 상승/하락 자동 결정
        color = color_override if color_override else ('green' if c >= o else 'red')
        
        # 고가-저가 선 그리기
        ax.plot([x, x], [l, h], color='black', linewidth=1, zorder=1, alpha=alpha)
        
        # 몸통(시가-종가) 사각형 그리기
        body_height = abs(o - c)
        if body_height == 0: 
            body_height = (h - l) * 0.05  # 도지 캔들의 경우 작은 높이 부여
        
        body_bottom = min(o, c)
        rect = Rectangle((x - width / 2, body_bottom), width, body_height, 
                        facecolor=color, zorder=2, alpha=alpha)
        ax.add_patch(rect)

def visualize_results(query_normalized, similar_patterns, train_ohlc_data, time_data, query_start_idx=None,
                     pair=None, timeframe=None, query_seq_length=None, top_k=None,
                     cache_file=None, embeddings_file=None, encoder_file=None, csv_file=None,
                     full_ohlc_data=None, query_emb=None, logs_file=None):
    """
    목적: 쿼리 패턴과 유사 패턴들을 시각화하여 이미지 파일로 저장
          각 유사 패턴에 대해 향후 예측 구간도 함께 표시

    입력:
        query_normalized (np.ndarray): 정규화된 쿼리 패턴, shape=(seq_len, 4)
        similar_patterns (list): find_similar_patterns 함수의 출력 결과
        full_ohlc_data (np.ndarray): 전체 원본 OHLC 데이터 (정규화 전)
        time_data (pd.Series): 시간 정보
        query_start_idx (int, optional): 쿼리 패턴의 시작 인덱스
    
    출력:
        None (이미지 파일들을 './visualize_results/' 폴더에 저장)
        생성되는 파일:
            - {timestamp}_input_pattern.png: 입력 패턴
            - {timestamp}_similar_{i}.png: 각 유사 패턴 + 예측
            - {timestamp}_all_patterns.png: 모든 패턴을 하나로 합친 이미지
    
    시각화 특징:
        1. 입력 패턴: 예측 구간 표시 (점선)
        2. 유사 패턴: 실제 데이터 + 반투명 예측 구간
        3. 각 패턴마다 유사도, 시간, 길이 정보 표시
        4. 정규화 적용으로 패턴 모양 비교 용이
    """
    print(f"[Function: visualize_results] - {len(similar_patterns) + 1}개의 차트 이미지로 저장 시작...")
    import os
    from datetime import timezone, timedelta
    
    # 저장 디렉토리 생성
    save_dir = os.path.join(os.getcwd(), f"./output/embeddings/visualize_results/v{MODEL_VERSION}")
    os.makedirs(save_dir, exist_ok=True)

    # 파일명용 타임스탬프 생성
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")

    # --- 전체 subplot을 위한 큰 figure 준비 ---
    num_plots = len(similar_patterns) + 1  # 쿼리 + 유사 패턴들
    big_fig, big_axs = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots), 
                                   sharex=False, sharey=False)
    if num_plots == 1:
        big_axs = [big_axs]  # 단일 subplot의 경우 리스트로 변환

    # --- Plot 1: 입력 패턴 (개별 이미지 + 전체 이미지) ---
    # 개별 이미지 생성
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_candlestick(ax, query_normalized, alpha=1.0)

    # Input Pattern 시작~종료 시간 정보 생성
    input_start_time_str = ""
    input_time_str = ""
    if query_start_idx is not None:
        query_len = len(query_normalized)
        query_end_idx = query_start_idx + query_len - 1
        input_start_time_str = time_data.iloc[query_start_idx].strftime('%Y-%m-%d %H:%M')
        input_end_time_str = time_data.iloc[query_end_idx].strftime('%Y-%m-%d %H:%M')
        input_time_str = f", Time: {input_start_time_str} ~ {input_end_time_str}"

    ax.set_title(f'Input Pattern (Length: {len(query_normalized)}{input_time_str})', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 예측 구간 표시 (패턴 길이의 1/3 정도)
    forecast_len_query = math.ceil(len(query_normalized) / 3)
    ax.axvline(x=len(query_normalized) - 0.5, color='blue', linestyle='--',
              linewidth=2, label=f'Blank Forecast Area ({forecast_len_query} bars)')
    ax.set_xlim(-1, len(query_normalized) + forecast_len_query)
    ax.legend()
    fig.tight_layout(rect=[0, 0.97, 1, 1])

    # 개별 이미지 저장
    input_img_path = os.path.join(save_dir, f"{timestamp}_input_pattern.png")
    fig.savefig(input_img_path)
    plt.close(fig)

    # 전체 이미지의 첫 번째 subplot에도 동일하게 그리기
    plot_candlestick(big_axs[0], query_normalized, alpha=1.0)
    big_axs[0].set_title(f'Input Pattern (Length: {len(query_normalized)}{input_time_str})', fontsize=16)
    big_axs[0].grid(True, linestyle='--', alpha=0.6)

    big_axs[0].axvline(x=len(query_normalized) - 0.5, color='blue', linestyle='--',
                      linewidth=2, label=f'Blank Forecast Area ({forecast_len_query} bars)')
    big_axs[0].set_xlim(-1, len(query_normalized) + forecast_len_query)
    big_axs[0].legend()
    
    print(f"  - 차트 1 (Input) 이미지 저장 완료: {input_img_path}")

    # --- Plot 2~N: 유사 패턴들 + 예측 구간 ---
    for i, result in enumerate(similar_patterns, 1):
        # 개별 이미지 생성
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # 패턴 정보 추출
        pat_start_idx = result['idx']    # 패턴 시작 인덱스
        pat_len = result['len']          # 패턴 길이
        sim = result['sim']              # 유사도
        forecast_len = math.ceil(pat_len / 3)      # 예측 구간 길이
        forecast_start_idx = pat_start_idx + pat_len
        forecast_end_idx = forecast_start_idx + forecast_len

        # 예측 구간이 데이터 범위를 벗어나는지 확인
        if forecast_end_idx > len(full_ohlc_data):
            print(f"  - 경고: Top {i} 패턴의 예측 구간이 데이터 끝을 벗어납니다. 예측 없이 표시합니다.")
            forecast_len = 0

        # 패턴 데이터 추출 및 정규화
        pattern_raw = full_ohlc_data[pat_start_idx : pat_start_idx + pat_len]
        if pattern_raw.shape[0] == 0:
            print(f"  - 경고: Top {i} 패턴 데이터가 비어있습니다. 건너뜁니다. (인덱스: {pat_start_idx})")
            continue
        base_for_norm = pattern_raw[0, 0]  # 정규화 기준점
        epsilon = 1e-8
        pattern_normalized = (pattern_raw - base_for_norm) * 100.0 / (base_for_norm + epsilon)
        
        # 패턴 부분 그리기 (불투명)
        plot_candlestick(ax, pattern_normalized, alpha=1.0)
        plot_candlestick(big_axs[i], pattern_normalized, alpha=1.0)

        # 예측 구간이 있는 경우 그리기 (반투명)
        if forecast_len > 0:
            forecast_raw = full_ohlc_data[forecast_start_idx : forecast_end_idx]
            if forecast_raw.shape[0] > 0:
                forecast_normalized = (forecast_raw - base_for_norm) * 100.0 / (base_for_norm + epsilon)
                
                # 예측 구간을 패턴 오른쪽에 이어서 그리기
                plot_candlestick(ax, forecast_normalized, x_offset=pat_len, alpha=0.5)
                plot_candlestick(big_axs[i], forecast_normalized, x_offset=pat_len, alpha=0.5)
            
            # 패턴과 예측 구간 경계선
            ax.axvline(x=pat_len - 0.5, color='blue', linestyle='--', 
                      linewidth=2, label=f'Forecast ({forecast_len} bars)')
            ax.legend()
            big_axs[i].axvline(x=pat_len - 0.5, color='blue', linestyle='--', 
                              linewidth=2, label=f'Forecast ({forecast_len} bars)')
            big_axs[i].legend()

        # 제목 및 레이아웃 설정
        pattern_start_time = time_data.iloc[pat_start_idx]
        pattern_end_time = time_data.iloc[pat_start_idx + pat_len - 1]
        pattern_start_time_str = pattern_start_time.strftime('%Y-%m-%d %H:%M')
        pattern_end_time_str = pattern_end_time.strftime('%Y-%m-%d %H:%M')
        title = f'Top {i} Similar Pattern (Sim: {sim:.3f}, Time: {pattern_start_time_str} ~ {pattern_end_time_str}, Length: {pat_len} + Forecast: {forecast_len})'
        
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(-1, pat_len + forecast_len)
        fig.tight_layout(rect=[0, 0.97, 1, 1])
        
        # 개별 이미지 저장
        sim_img_path = os.path.join(save_dir, f"{timestamp}_similar_{i}.png")
        fig.savefig(sim_img_path)
        plt.close(fig)
        
        # 전체 이미지 subplot 설정
        big_axs[i].set_title(title, fontsize=14)
        big_axs[i].grid(True, linestyle='--', alpha=0.6)
        big_axs[i].set_xlim(-1, pat_len + forecast_len)
        
        print(f"  - 차트 {i+1} (Similar + Forecast) 이미지 저장 완료: {sim_img_path}")

    # 전체 이미지 저장
    big_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    all_img_path = os.path.join(save_dir, f"{timestamp}_all_patterns.png")
    big_fig.savefig(all_img_path)
    plt.close(big_fig)
    
    print(f"  - 전체 차트 한 장 이미지 저장 완료: {all_img_path}")
    print(f"  - 모든 차트 이미지 저장 완료. 폴더: {save_dir}")

    # JSON 파일 생성
    import json

    # Input pattern 정보 생성
    input_pattern_dict = {
        "start_time": input_start_time_str if query_start_idx is not None else "",
        "end_time": input_end_time_str if query_start_idx is not None else "",
        "ohlc": []
    }

    # Input pattern의 원본 OHLC 데이터 추출
    # full_ohlc_data가 제공되면 사용, 아니면 train_ohlc_data 사용
    ohlc_data_source = full_ohlc_data if full_ohlc_data is not None else train_ohlc_data
    # ohlc_data_source = train_ohlc_data

    if query_start_idx is not None:
        query_len = len(query_normalized)
        query_raw_ohlc = ohlc_data_source[query_start_idx : query_start_idx + query_len]
        for ohlc_row in query_raw_ohlc:
            input_pattern_dict["ohlc"].append({
                "open": float(ohlc_row[0]),
                "high": float(ohlc_row[1]),
                "low": float(ohlc_row[2]),
                "close": float(ohlc_row[3])
            })

    # Results 정보 생성
    results_list = []
    for i, result in enumerate(similar_patterns, 1):
        pat_start_idx = result['idx']
        pat_len = result['len']
        sim = result['sim']
        forecast_len = pat_len // 3
        forecast_start_idx = pat_start_idx + pat_len
        forecast_end_idx = forecast_start_idx + forecast_len

        # 시작, 종료 시간
        pattern_start_time = time_data.iloc[pat_start_idx]
        pattern_end_time = time_data.iloc[pat_start_idx + pat_len - 1]
        pattern_start_time_str = pattern_start_time.strftime('%Y-%m-%d %H:%M')
        pattern_end_time_str = pattern_end_time.strftime('%Y-%m-%d %H:%M')

        # 유사 패턴 OHLC 데이터 (full_ohlc_data 사용)
        sim_ohlc_list = []
        pattern_raw = full_ohlc_data[pat_start_idx : pat_start_idx + pat_len]
        for ohlc_row in pattern_raw:
            sim_ohlc_list.append({
                "open": float(ohlc_row[0]),
                "high": float(ohlc_row[1]),
                "low": float(ohlc_row[2]),
                "close": float(ohlc_row[3])
            })

        # 예측 구간 OHLC 데이터 (next_ohlc)
        next_ohlc_list = []
        if forecast_end_idx <= len(full_ohlc_data):
            forecast_raw = full_ohlc_data[forecast_start_idx : forecast_end_idx]
            for ohlc_row in forecast_raw:
                next_ohlc_list.append({
                    "open": float(ohlc_row[0]),
                    "high": float(ohlc_row[1]),
                    "low": float(ohlc_row[2]),
                    "close": float(ohlc_row[3])
                })

        result_dict = {
            "rank": i,
            "similarity": float(sim),
            "start_time": pattern_start_time_str,
            "end_time": pattern_end_time_str,
            "sim_ohlc": sim_ohlc_list,
            "next_ohlc": next_ohlc_list
        }
        results_list.append(result_dict)

    # 전체 JSON 구조 생성
    json_data = {
        "information": {
            "pair": pair if pair is not None else "",
            "timeframe": timeframe if timeframe is not None else "",
            "query_seq_length": query_seq_length if query_seq_length is not None else len(query_normalized),
            "search_time": datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S'),
            "top_k": top_k if top_k is not None else len(similar_patterns)
        },
        "input_pattern": input_pattern_dict,
        "results": results_list,
        "files_used": {
            "cache_file": cache_file if cache_file is not None else "",
            "embeddings_file": embeddings_file if embeddings_file is not None else "",
            "encoder_file": encoder_file if encoder_file is not None else "",
            "csv_file": csv_file if csv_file is not None else "",
            "script": os.path.abspath(__file__),
            "logs_file": logs_file if csv_file is not None else ""
        },
        "query_embedding": query_emb.tolist() if query_emb is not None else []
    }

    # JSON 파일 저장
    json_path = os.path.join(save_dir, f"{timestamp}_info.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"  - JSON 파일 저장 완료: {json_path}")

# endregion

# region  6. 메인 실행부 

class MainExecution:
    """
    
    6. 메인 실행부
    
    - 캐시 관리
    - Search 모드
    - Build Cache 모드
    """
    pass

def save_cache_atomically(cache_path, cache_data):
    """
    목적: 캐시를 원자적으로 저장하여 중간에 중단되어도 파일이 손상되지 않도록 함
    """
    temp_path = cache_path + '.tmp'
    try:
        # 1. 임시 파일에 먼저 저장
        with open(temp_path, 'wb') as f:
            pickle.dump(cache_data, f)
        # 2. 성공적으로 저장되면 원본 파일과 교체 (원자적 연산)
        os.replace(temp_path, cache_path)
        return True
    except (IOError, pickle.PickleError) as e:
        print(f"  - [경고] 캐시 파일 저장 실패: {e}")
        # 임시 파일이 남아있으면 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


if __name__ == "__main__":
    """
    메인 실행 플로우:
    
    1. 설정 및 파라미터 정의
    2. 데이터 로드 및 학습/테스트 분리
    3. 모델 학습 또는 로드 (Retriever Only)
    4. (캐시 미스 시) 쿼리 패턴 생성 및 검색
    5. (캐시 미스 시) 결과 캐시에 저장
    6. 결과 시각화
    
    실행 모드:
    - 'search': 단일 쿼리를 실행하고 결과를 캐시에 저장/로드 (기본값)
    - 'build_cache': 지정된 범위의 모든 조합을 미리 계산하여 캐시를 구축
    """

    # 랜덤 시드 고정 (재현성 확보)
    import os
    import random

    SEED = 11410004

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ['PYTHONHASHSEED'] = f'{SEED}'
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    #  설정 파라미터
    pair = "BTC"
    timeframe = "4H"
    # csv_path = f"../amt_data/Nvidia/{pair}USD_{timeframe}_20151201_20251107.csv"
    csv_path = f"./input_data/{pair}USD_{timeframe}_20130101_20251103.csv"
    # csv_path = "./input_data/BTCUSD_4H_20130101_20251103.csv"
    query_time_str = "2025-11-11 13:01"    # 분석할 패턴의 끝 시간
    query_seq_length = 3                     # 분석할 패턴의 길이 (캔들 개수)
    emb_dim = 64                             # 임베딩 벡터 차원
    top_k = 10                                # 최종 검색할 유사 패턴 개수
    
    # --- 모델 학습 제어 플래그 ---
    force_train_retriever = False              # True: Retriever(Bi-Encoder) 강제 재학습, False: 기존 모델 로드 시도

    MIN_PATTERN_LEN = 3                         # 모델이 처리할 수 있는 최소 길이
    MAX_PATTERN_LEN = 100                       # 모델이 처리할 수 있는 최대 길이
    target_length = query_seq_length            # 특정 길이 패턴만 검색 (None: 모든 길이)

    # 임베딩 사전 계산에 사용할 특정 패턴 길이 리스트
    SPECIFIC_PATTERN_LENGTHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 30, 50, 60, 100]

    # Int8 Quantization 설정 (메모리 75% 절감, 약간의 정밀도 손실)
    USE_INT8_QUANTIZATION = True  # True: int8 (1 byte), False: float32 (4 bytes)

    #  실행 모드 및 캐시 설정 
    # 'search': 설정된 단일 쿼리로 검색을 수행하고, 결과가 없으면 캐시에 저장합니다.
    # 'build_cache': 아래 '캐시 사전 구축 모드' 섹션의 주석을 해제하여 사용하세요.
    #                정의된 범위에 대해 모든 경우의 수의 결과를 미리 계산하여 캐시에 저장합니다.
    EXECUTION_MODE = 'search'
    # 파일명 규칙: BTC_4H_{타입}_emb{차원}_v{버전}.{확장자}
    MODEL_VERSION = 8  # v7.1: Retriever-Only System (Reranker Removed)
    CACHE_PATH = f'./output/embeddings/v{MODEL_VERSION}/v{MODEL_VERSION}_{pair}_{timeframe}_cache_emb{emb_dim}.pkl'

    # --- 입력 검증 ---
    if query_seq_length < MIN_PATTERN_LEN:
        raise ValueError(f"쿼리 길이는 최소 {MIN_PATTERN_LEN} 이상이어야 합니다.")
    if query_seq_length > MAX_PATTERN_LEN:
        raise ValueError(f"쿼리 길이는 최대 {MAX_PATTERN_LEN} 이하여야 합니다.")

    # 모델 및 임베딩 저장 경로
    retriever_model_path = f'./output/embeddings/v{MODEL_VERSION}/v{MODEL_VERSION}_{pair}_{timeframe}_encoder_multi_emb{emb_dim}.pth'
    emb_path = f'./output/embeddings/v{MODEL_VERSION}/v{MODEL_VERSION}_{pair}_{timeframe}_embeddings_emb{emb_dim}.pkl'
    log_path = f'./output/embeddings/v{MODEL_VERSION}/v{MODEL_VERSION}_{pair}_{timeframe}_training_log_emb{emb_dim}.txt'

    # --- 학습 필요 여부 확인 ---
    need_retriever_training = force_train_retriever or not os.path.exists(retriever_model_path)
    need_training = need_retriever_training

    # --- 로거 초기화 (학습이 필요한 경우에만) ---
    if need_training:
        logger = TrainingLogger(log_path, mode='w')  # 새 학습이므로 덮어쓰기
        logger.log_header("학습 실행 시작")

        # 하이퍼파라미터 딕셔너리 생성
        hyperparameters = {
            "실행 시간": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "데이터 쌍 (Pair)": pair,
            "타임프레임 (Timeframe)": timeframe,
            "CSV 경로": csv_path,
            "쿼리 시간 (Query Time)": query_time_str,
            "쿼리 길이 (Query Seq Length)": query_seq_length,
            "임베딩 차원 (Embedding Dimension)": emb_dim,
            "Top K": top_k,
            "모델 버전 (Model Version)": MODEL_VERSION,
            "최소 패턴 길이 (MIN_PATTERN_LEN)": MIN_PATTERN_LEN,
            "최대 패턴 길이 (MAX_PATTERN_LEN)": MAX_PATTERN_LEN,
            "타겟 길이 (Target Length)": target_length,
            "특정 패턴 길이 (SPECIFIC_PATTERN_LENGTHS)": SPECIFIC_PATTERN_LENGTHS,
            "Int8 Quantization 사용": USE_INT8_QUANTIZATION,
            "실행 모드 (EXECUTION_MODE)": EXECUTION_MODE,
            "강제 Retriever 재학습 (force_train_retriever)": force_train_retriever,
            "랜덤 시드 (SEED)": SEED,
            "Retriever 모델 경로": retriever_model_path,
            "임베딩 경로": emb_path,
            "캐시 경로": CACHE_PATH,
        }

        logger.log_hyperparameters(hyperparameters)
    else:
        logger = None
        print("--- 학습이 필요하지 않음. 기존 모델 사용. 로그 파일 덮어쓰기 방지. ---")

    # --- 캐시 로드 ---
    print(f"--- 캐시 파일 로드 시도: {CACHE_PATH} ---")
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'rb') as f:
                pattern_cache = pickle.load(f)
            print(f"  - 캐시 로드 완료. 현재 {len(pattern_cache)}개의 쿼리 결과 저장됨.")
        else:
            pattern_cache = {}
            print("  - 캐시 파일이 없어 새로 생성합니다.")
    except (IOError, pickle.PickleError) as e:
        print(f"  - 캐시 파일 로드 중 오류 발생 ({e}). 새 캐시를 생성합니다.")
        pattern_cache = {}

    print("=" * 80)
    print(f"패턴 분석 시스템 시작 (Retriever Only Mode: {EXECUTION_MODE})")
    print("=" * 80)
    
    print("\n--- 1. 데이터 로드 및 분리 ---")
    full_ohlc_data, full_time_data = load_ohlc_data(csv_path)

    split_ratio = 0.9
    split_idx = int(len(full_ohlc_data) * split_ratio)
    train_ohlc_data, train_time_data = full_ohlc_data[:split_idx], full_time_data[:split_idx]
    test_ohlc_data, test_time_data = full_ohlc_data[split_idx:], full_time_data[split_idx:]
    print(f"  - 데이터 분리 완료: 학습 {len(train_ohlc_data)}개, 테스트 {len(test_ohlc_data)}개")

    print("\n--- 2. Retriever(Bi-Encoder) 모델 로드 ---")
    retriever_model = train_or_load_model(train_ohlc_data, test_ohlc_data, emb_dim, retriever_model_path, force_train=force_train_retriever, max_len=MAX_PATTERN_LEN, logger=logger)

    device = next(retriever_model.parameters()).device

    # --- 전체 임베딩은 한 번만 로드/계산 ---
    print("\n--- 전체 임베딩 로드 또는 사전 계산 (Retriever용) ---")
    if os.path.exists(emb_path) and not force_train_retriever:
        print(f"  - 저장된 임베딩 파일 로드: {emb_path}")
        with open(emb_path, 'rb') as f:
            embedding_data = pickle.load(f)
        # 양자화 정보 출력
        if embedding_data.get('quantized', False):
            print(f"  - 양자화된 임베딩 (int8) 로드 완료")
        else:
            print(f"  - 일반 임베딩 (float32) 로드 완료")
    else:
        embedding_data = precompute_and_save_embeddings(
            train_ohlc_data,
            retriever_model,
            emb_path,
            max_len=MAX_PATTERN_LEN,
            specific_lengths=SPECIFIC_PATTERN_LENGTHS,
            use_int8=USE_INT8_QUANTIZATION
        )


    if EXECUTION_MODE == 'search':
        print("\n--- [Search Mode] ---")
        query_key = (query_time_str, query_seq_length, target_length, top_k)
        
        if query_key in pattern_cache:
            print(f"\n--- 캐시 히트! 저장된 결과를 사용합니다. Key: {query_key} ---")
            cached_results = pattern_cache[query_key]
            print(f"  - 캐시된 패턴 개수: {len(cached_results)}개")
            
            # 시각화를 위해 쿼리 패턴은 따로 생성 (완성된 캔들만 사용)
            try:
                query_time = pd.to_datetime(query_time_str).tz_localize(None)  
                
                # 완성된 캔들만 필터링
                completed_candles = []
                for i, candle_time in enumerate(full_time_data):
                    if i < len(full_time_data) - 1:
                        next_candle_time = full_time_data.iloc[i + 1]
                    else:
                        next_candle_time = candle_time + pd.Timedelta(hours=4)
                    
                    if query_time >= next_candle_time:
                        completed_candles.append(i)
                
                if not completed_candles:
                    raise ValueError("완성된 캔들이 없습니다.")
                    
                end_idx = completed_candles[-1]
                start_idx = end_idx - query_seq_length + 1
                if start_idx < 0:
                    raise ValueError(f"패턴 길이({query_seq_length})에 비해 시작 시간이 너무 빠릅니다.")
                query_raw = full_ohlc_data[start_idx : end_idx + 1]
                query_normalized = normalize_window(query_raw)

                # 쿼리 임베딩 생성 (캐시 히트 케이스)
                retriever_model.eval()
                with torch.no_grad():
                    query_tensor_orig = torch.from_numpy(query_normalized.astype(np.float32)).unsqueeze(0).to(device)
                    current_len = query_tensor_orig.shape[1]

                    # 0패딩 적용
                    if current_len < MAX_PATTERN_LEN:
                        padding = torch.zeros(1, MAX_PATTERN_LEN - current_len, 4, dtype=query_tensor_orig.dtype, device=device)
                        query_tensor_resized = torch.cat([query_tensor_orig, padding], dim=1)
                    else:
                        query_tensor_resized = query_tensor_orig[:, :MAX_PATTERN_LEN, :]

                    query_emb = retriever_model(query_tensor_resized).cpu().numpy().flatten()
                    QUERY_EMB = query_emb
                print(f"  - 쿼리 임베딩 생성 완료 (캐시 히트). Shape: {QUERY_EMB.shape}")

                # 캐시된 결과에서 Input Pattern 범위 제외
                print(f"  - Input Pattern 범위 제외 (idx: {start_idx} ~ {end_idx})")
                excluded_results = []
                for result in cached_results:
                    pattern_start = result['idx']
                    pattern_end = pattern_start + result['len'] - 1

                    # 패턴이 input pattern 범위와 겹치는지 확인
                    if not (pattern_end < start_idx or pattern_start > end_idx):
                        print(f"    - 제외됨: idx={pattern_start}, len={result['len']} (겹침)")
                        continue
                    excluded_results.append(result)

                print(f"  - Exclusion 필터 후: {len(excluded_results)}개 패턴 남음")

                # Top_k개 선택
                final_similarities = excluded_results[:top_k]
                print(f"  - 최종 선택: {len(final_similarities)}개 패턴 (목표: {top_k}개)")

            except (IndexError, ValueError) as e:
                print(f"[오류] 캐시된 결과를 시각화하기 위한 쿼리 패턴 생성 실패: {e}")
                query_normalized = None # 시각화 실패 시 대비
                final_similarities = cached_results[:top_k]  # 오류 시 exclusion 없이 top_k개 사용
        else:
            print(f"\n--- 캐시 미스. 새로운 검색을 시작합니다. Key: {query_key} ---")
            
            print("\n--- 4. 쿼리 벡터 생성 및 임베딩 (Retriever용) ---")
            try:
                query_time = pd.to_datetime(query_time_str)
                
                # 완성된 캔들만 필터링
                completed_candles = []
                for i, candle_time in enumerate(full_time_data):
                    if i < len(full_time_data) - 1:
                        next_candle_time = full_time_data.iloc[i + 1]
                    else:
                        next_candle_time = candle_time + pd.Timedelta(hours=4)
                    
                    if query_time >= next_candle_time:
                        completed_candles.append(i)
                
                if not completed_candles:
                    raise ValueError("완성된 캔들이 없습니다.")
                    
                absolute_end_idx = completed_candles[-1]
                absolute_start_idx = absolute_end_idx - query_seq_length + 1
                if absolute_start_idx < 0:
                    raise ValueError(f"패턴 길이({query_seq_length})에 비해 시작 시간이 너무 빠릅니다.")
            except IndexError:
                raise ValueError(f"'{query_time_str}'에 해당하는 데이터를 찾을 수 없습니다.")

            query_raw = full_ohlc_data[absolute_start_idx : absolute_end_idx + 1]
            query_normalized = normalize_window(query_raw)
            
            # 시각화를 위해 원본 길이의 정규화된 쿼리 보존
            query_normalized_for_viz = normalize_window(query_raw)
            
            retriever_model.eval()
            with torch.no_grad():
                # 모델 입력을 위해 쿼리 텐서를 MAX_PATTERN_LEN으로 0패딩
                query_tensor_orig = torch.from_numpy(query_normalized_for_viz.astype(np.float32)).unsqueeze(0).to(device)
                current_len = query_tensor_orig.shape[1]

                # 0패딩 적용
                if current_len < MAX_PATTERN_LEN:
                    padding = torch.zeros(1, MAX_PATTERN_LEN - current_len, 4, dtype=query_tensor_orig.dtype, device=device)
                    query_tensor_resized = torch.cat([query_tensor_orig, padding], dim=1)
                else:
                    query_tensor_resized = query_tensor_orig[:, :MAX_PATTERN_LEN, :]

                query_emb = retriever_model(query_tensor_resized).cpu().numpy().flatten()
                QUERY_EMB = query_emb
            print(f"  - 쿼리 임베딩 생성 완료. Shape: {query_emb.shape}")

            print("\n--- 5. Retriever로 유사 패턴 검색 ---")
            # Input pattern 범위 제외
            exclude_range = (absolute_start_idx, absolute_end_idx)
            # 직접 top_k개 검색
            candidate_patterns = find_similar_patterns(query_emb, embedding_data, device, top_k=top_k, target_length=target_length, power=query_seq_length, exclude_range=exclude_range)

            print(f"\n--- Retriever 검색 결과 Top {len(candidate_patterns)} ---")
            for i, p in enumerate(candidate_patterns):
                print(f"    {i+1}. 인덱스: {p['idx']}, 길이: {p['len']}, 유사도(Retriever Score): {p['sim']:.4f}")

            # 인덱스가 비슷한 패턴들 필터링 (±2 범위 내에서 가장 높은 점수만 유지)
            print(f"\n--- 인덱스 유사 패턴 필터링 (±2 범위) ---")
            print(f"  - 필터링 전: {len(candidate_patterns)}개 패턴")
            filtered_results = filter_similar_indices(candidate_patterns, index_tolerance=2)
            print(f"  - 필터링 후: {len(filtered_results)}개 패턴")

            # 캐시에는 25개 저장 (input pattern 제외하지 않음 - 다른 쿼리에서 재사용 가능)
            cache_save_count = 25
            results_to_cache = filtered_results[:cache_save_count]
            print(f"  - 캐시 저장용: {len(results_to_cache)}개 패턴")

            # 현재 검색에서는 top_k개만 사용
            final_similarities = filtered_results[:top_k]
            print(f"  - 최종 선택: {len(final_similarities)}개 패턴 (목표: {top_k}개)")

            pattern_cache[query_key] = results_to_cache  # 25개를 캐시에 저장
            if save_cache_atomically(CACHE_PATH, pattern_cache):
                print(f"\n--- 검색 완료. 캐시 업데이트 및 저장 완료 (캐시에 {len(results_to_cache)}개 저장) ---")

        print(f"\n--- 최종 Top {min(top_k, len(final_similarities))} 유사 패턴 (From Cache or New Search) ---")
        for i, p in enumerate(final_similarities):
            print(f"    {i+1}. 인덱스: {p['idx']}, 길이: {p['len']}, 유사도(Retriever Score): {p['sim']:.4f}")
    
        print("\n--- 7. 결과 시각화 ---")
        # 시각화에는 원본 길이의 쿼리 사용
        if 'query_normalized_for_viz' not in locals():
            # 캐시 히트 시 시각화를 위한 쿼리 생성
            try:
                query_time = pd.to_datetime(query_time_str).tz_localize(None)
                completed_candles = [i for i, t in enumerate(full_time_data) if (i < len(full_time_data) - 1 and query_time >= full_time_data.iloc[i+1]) or (i == len(full_time_data) - 1 and query_time >= t + pd.Timedelta(hours=4))]
                if not completed_candles:
                    raise ValueError("완성된 캔들이 없습니다.")
                end_idx = completed_candles[-1]
                start_idx = end_idx - query_seq_length + 1
                if start_idx < 0:
                    raise ValueError(f"패턴 길이({query_seq_length})에 비해 시작 시간이 너무 빠릅니다.")
                query_raw = full_ohlc_data[start_idx : end_idx + 1]
                query_normalized_for_viz = normalize_window(query_raw)
            except (IndexError, ValueError) as e:
                print(f"[오류] 캐시된 결과를 시각화하기 위한 쿼리 패턴 생성 실패: {e}")
                query_normalized_for_viz = None

        if final_similarities and query_normalized_for_viz is not None:
            # query_start_idx 전달 (absolute_start_idx 또는 start_idx 사용)
            if 'absolute_start_idx' in locals():
                visualize_results(query_normalized_for_viz, final_similarities, train_ohlc_data, full_time_data,
                                query_start_idx=absolute_start_idx,
                                pair=pair, timeframe=timeframe, query_seq_length=query_seq_length, top_k=top_k,
                                cache_file=CACHE_PATH, embeddings_file=emb_path, encoder_file=retriever_model_path,
                                csv_file=csv_path, full_ohlc_data=full_ohlc_data,
                                query_emb=QUERY_EMB, logs_file=log_path)
            elif 'start_idx' in locals():
                visualize_results(query_normalized_for_viz, final_similarities, train_ohlc_data, full_time_data,
                                query_start_idx=start_idx,
                                pair=pair, timeframe=timeframe, query_seq_length=query_seq_length, top_k=top_k,
                                cache_file=CACHE_PATH, embeddings_file=emb_path, encoder_file=retriever_model_path,
                                csv_file=csv_path, full_ohlc_data=full_ohlc_data,
                                query_emb=QUERY_EMB, logs_file=log_path)
            else:
                visualize_results(query_normalized_for_viz, final_similarities, train_ohlc_data, full_time_data,
                                query_start_idx=None,
                                pair=pair, timeframe=timeframe, query_seq_length=query_seq_length, top_k=top_k,
                                cache_file=CACHE_PATH, embeddings_file=emb_path, encoder_file=retriever_model_path,
                                csv_file=csv_path, full_ohlc_data=full_ohlc_data,
                                query_emb=QUERY_EMB, logs_file=log_path)
        else:
            print("  - 유사 패턴을 찾지 못했거나 쿼리 패턴 생성에 실패하여 시각화를 건너뜁니다.")

    elif EXECUTION_MODE == 'build_cache':
        print("\n--- [Cache Build Mode] ---")
        print("경고: 이 모드는 매우 오랜 시간이 소요될 수 있습니다.")
        
        # =
        #   캐시 구축을 위한 범위 설정 (필요에 맞게 수정하여 사용)
        # =
        # 예시: 2024년 6월 1일부터 7월 15일까지 100개 데이터 포인트마다 쿼리
        start_date = "2023-06-01"
        end_date = "2024-07-15"
        time_indices_to_cache = full_time_data[(full_time_data >= start_date) & (full_time_data <= end_date)].index[::100]
        
        # 테스트할 쿼리 길이와 목표 길이 (특정 길이만 사용)
        query_lengths_to_cache = SPECIFIC_PATTERN_LENGTHS
        target_lengths_to_cache = SPECIFIC_PATTERN_LENGTHS
        
        total_jobs = len(time_indices_to_cache) * len(query_lengths_to_cache) * len(target_lengths_to_cache)
        print(f"  - 총 {total_jobs}개의 조합에 대해 캐시 구축을 시작합니다.")
        
        progress = 0
        for start_idx in tqdm(time_indices_to_cache, desc="Caching Progress"):
            current_time_str = full_time_data.iloc[start_idx].strftime('%Y-%m-%d %H:%M:%S')
            for q_len in query_lengths_to_cache:
                for t_len in target_lengths_to_cache:
                    progress += 1
                    print(f"\n--- Job {progress}/{total_jobs}: {current_time_str}, q_len={q_len}, t_len={t_len} ---")
                    
                    query_key = (current_time_str, q_len, t_len, top_k)
                    if query_key in pattern_cache:
                        print("  - 이미 캐시됨. 건너뜁니다.")
                        continue

                    try:
                        # 쿼리 생성 및 보간 (search 모드와 동일하게)
                        query_raw = full_ohlc_data[start_idx : start_idx + q_len]
                        if len(query_raw) < q_len: continue
                        
                        query_normalized_for_viz = normalize_window(query_raw)

                        # 임베딩 및 재정렬을 위한 0패딩
                        with torch.no_grad():
                            query_tensor_orig = torch.from_numpy(query_normalized_for_viz.astype(np.float32)).unsqueeze(0).to(device)
                            current_len = query_tensor_orig.shape[1]

                            # 0패딩 적용
                            if current_len < MAX_PATTERN_LEN:
                                padding = torch.zeros(1, MAX_PATTERN_LEN - current_len, 4, dtype=query_tensor_orig.dtype, device=device)
                                query_tensor_resized = torch.cat([query_tensor_orig, padding], dim=1)
                            else:
                                query_tensor_resized = query_tensor_orig[:, :MAX_PATTERN_LEN, :]

                            query_emb = retriever_model(query_tensor_resized).cpu().numpy().flatten()
                        
                        # Retriever로 검색
                        # Input pattern 범위 제외
                        exclude_range = (start_idx, start_idx + q_len - 1)
                        candidate_patterns = find_similar_patterns(query_emb, embedding_data, device, top_k=top_k, target_length=t_len, power=q_len, exclude_range=exclude_range)
                        
                        print(f"--- Retriever 검색 결과 Top 5 ---")
                        for i, p in enumerate(candidate_patterns[:5]):
                            print(f"    {i+1}. 인덱스: {p['idx']}, 길이: {p['len']}, 유사도(Retriever Score): {p['sim']:.4f}")
                        
                        # 인덱스 유사 패턴 필터링
                        filtered_results = filter_similar_indices(candidate_patterns, index_tolerance=2)
                        final_similarities = filtered_results[:top_k]
                        
                        # 캐시 저장
                        pattern_cache[query_key] = final_similarities

                    except Exception as e:
                        print(f"  - [오류] 캐시 생성 중 오류 발생: {e}")

            # 일정 주기마다 캐시 파일 저장
            if progress % 10 == 0:
                if save_cache_atomically(CACHE_PATH, pattern_cache):
                    print(f"  - 중간 캐시 저장 완료. (Total: {len(pattern_cache)} items)")

        # 최종 저장
        if save_cache_atomically(CACHE_PATH, pattern_cache):
            print(f"\n--- 캐시 구축 완료. 최종 캐시 크기: {len(pattern_cache)} ---")


    # --- 로거 종료 ---
    if logger is not None:
        logger.close()
        print(f"\n--- 학습 로그 저장 완료: {log_path} ---")

    print("\n" + "=" * 80)
    print("프로그램 실행 완료.")
    print("=" * 80)

# endregion