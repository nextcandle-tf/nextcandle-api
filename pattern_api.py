#!/usr/bin/env python3
"""
Pattern Detection API Server
웹 대시보드와 패턴 검출 모델을 연결하는 Flask API 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from dotenv import load_dotenv

# 환경변수 로드 (api 디렉토리의 .env 파일 명시적 로드)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta, timezone
import ccxt
import logging
import pytz
import pickle
import jwt
import sqlite3
import json
import yagmail
import random
import string
import math

# 권한 시스템 import
from database import db_manager, UserTier
from auth_middleware import require_auth, get_tier_limits
from scheduler import AutoAnalysisScheduler
from performance_cache import performance_cache, cached, task_manager

# 패턴 검출 스크립트 import
sys.path.append('/home/andy/candle-model/scripts')
from pattern_detecter_with_cache_v8_64emd_no_reranker import (
    load_ohlc_data, normalize_window, PatternEncoder, 
    find_similar_patterns, precompute_and_save_embeddings,
    train_or_load_model, save_cache_atomically, collate_fn
)


from flasgger import Swagger

app = Flask(__name__)
# Swagger Configuration
app.config['SWAGGER'] = {
    'title': 'Candle Pattern Finder API',
    'uiversion': 3,
    'description': 'API for cryptocurrency candle pattern analysis and similarity search',
    'version': '1.0.0',
    'specs_route': '/apidocs/'
}
swagger = Swagger(app)

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'temp-dev-key-CHANGE-IN-PRODUCTION')
# CORS(app)
CORS(app, supports_credentials=True, origins=[
    'https://pattern-finder.com',
    'http://localhost:3000',
    'http://127.0.0.1:3000'
]) # 허가된 도메인만 접근 허용

# Global variables for loaded model and data
model = None
# reranker_model = None  # Removed in v8
embedding_data = None
# reranker_cache = None # Removed in v8
full_ohlc_data = None
full_time_data = None
full_ohlc_data = None
full_time_data = None
binance_exchange = None
auto_scheduler = None

# 이메일 인증 코드 임시 저장소 (실제 운영환경에서는 Redis 등 사용 권장)
verification_codes = {}

# 이메일 발송 설정
def get_email_config():
    """환경변수에서 이메일 설정 가져오기"""
    return {
        'email': os.getenv('EMAIL_USER'),
        'password': os.getenv('EMAIL_PASSWORD'),
        'smtp_host': os.getenv('EMAIL_HOST', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('EMAIL_PORT', '587'))
    }

def generate_verification_code():
    """6자리 랜덤 인증 코드 생성"""
    return ''.join(random.choices(string.digits, k=6))

def send_verification_email(email, code):
    """인증 코드 이메일 발송"""
    try:
        config = get_email_config()
        if not config['email'] or not config['password']:
            print(f"개발 환경: {email}의 인증 코드는 {code}입니다.")
            return True  # 개발 환경에서는 성공으로 처리
        
        yag = yagmail.SMTP(config['email'], config['password'])
        
        subject = "[Pattern Finder] 이메일 인증 코드"
        body = f"""
        <h2>Pattern Finder 회원가입</h2>
        <p>안녕하세요,</p>
        <p>회원가입을 완료하기 위해 아래 인증 코드를 입력해주세요:</p>
        <div style="background-color: #f0f8ff; padding: 20px; margin: 20px 0; text-align: center; border-radius: 8px;">
            <h1 style="color: #2563eb; font-size: 32px; margin: 0; letter-spacing: 8px;">{code}</h1>
        </div>
        <p>이 코드는 10분간 유효합니다.</p>
        <p>감사합니다,<br>Pattern Finder 팀</p>
        """
        
        yag.send(to=email, subject=subject, contents=body)
        return True
    except Exception as e:
        print(f"이메일 발송 실패: {e}")
        return False

# Configuration
CONFIG = {
    'csv_path': '/home/andy/candle-model/input_data/New_bs+bn_BTCUSD_250720_update, 240.csv',
    'model_path': '/home/andy/candle-model/output/embeddings/v8/v8_BTC_4H_encoder_multi_emb64.pth',
    'emb_path': '/home/andy/candle-model/output/embeddings/v8/v8_BTC_4H_embeddings_emb64.pkl',
    'emb_dim': 64, # Updated to 64
    'max_pattern_len': 100,
    'min_pattern_len': 3,
    'candidate_count': 100,
    'binance_api_key': os.getenv('BINANCE_API_KEY', ''),
    'binance_api_secret': os.getenv('BINANCE_API_SECRET', '')
}

# 한국 시간대 설정 (UTC+9)
KST = pytz.timezone('Asia/Seoul')

def get_kst_candle_intervals():
    """KST 기준 4시간 캔들 시작 시간들 (1, 5, 9, 13, 17, 21시)"""
    return [1, 5, 9, 13, 17, 21]

def get_current_kst_time():
    """현재 KST 시간 반환"""
    return datetime.now(KST)

def find_last_completed_kst_interval(current_kst_time=None):
    """
    현재 KST 시간 기준으로 가장 최근에 완료된 4시간 캔들 구간의 끝 시간 찾기
    
    Args:
        current_kst_time: KST 시간 (없으면 현재 시간 사용)
    
    Returns:
        datetime: 가장 최근 완료된 구간의 끝 시간 (KST) - 패턴의 완료 시점
    """
    if current_kst_time is None:
        current_kst_time = get_current_kst_time()
    
    current_hour = current_kst_time.hour
    
    # 4시간 캔들 구간: 21-01시, 01-05시, 05-09시, 09-13시, 13-17시, 17-21시
    # 각 구간은 마지막 시간(01, 05, 09, 13, 17, 21)에 완료됨
    
    last_completed_hour = None
    target_date = current_kst_time.date()
    
    if current_hour >= 21:  # 21시 이후 → 17-21시 구간이 21시에 완료됨
        last_completed_hour = 21
    elif current_hour >= 17:  # 17시 이후 → 13-17시 구간이 17시에 완료됨
        last_completed_hour = 17
    elif current_hour >= 13:  # 13시 이후 → 09-13시 구간이 13시에 완료됨
        last_completed_hour = 13
    elif current_hour >= 9:   # 9시 이후 → 05-09시 구간이 09시에 완료됨
        last_completed_hour = 9
    elif current_hour >= 5:   # 5시 이후 → 01-05시 구간이 05시에 완료됨
        last_completed_hour = 5
    elif current_hour >= 1:   # 1시 이후 → 전날 21-01시 구간이 01시에 완료됨
        last_completed_hour = 1
    
    # 완료된 구간의 끝 시간 생성
    result = current_kst_time.replace(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=last_completed_hour,
        minute=0,
        second=0,
        microsecond=0
    )
    
    
    print(f"🕐 Current KST: {current_kst_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Last completed interval end: {result.strftime('%Y-%m-%d %H:%M:%S KST')}")
    
    return result

def find_last_completed_candle_time(current_kst_time=None):
    """
    4시간 캔들 기준으로 가장 최근에 완료된 캔들의 시간을 반환 (패턴의 마지막 캔들)
    Returns:
        datetime: 가장 최근 완료된 캔들 시간 (KST)
    """
    if current_kst_time is None:
        current_kst_time = get_current_kst_time()
    
    current_hour = current_kst_time.hour
    
    # 4시간 캔들 완료 시간: 01시, 05시, 09시, 13시, 17시, 21시
    last_completed_candle_hour = None
    target_date = current_kst_time.date()
    
    if current_hour >= 21:  # 21시 이후 → 21시 캔들 완료됨
        last_completed_candle_hour = 21
    elif current_hour >= 17:  # 17시 이후 → 17시 캔들 완료됨
        last_completed_candle_hour = 17
    elif current_hour >= 13:  # 13시 이후 → 13시 캔들 완료됨
        last_completed_candle_hour = 13
    elif current_hour >= 9:   # 9시 이후 → 09시 캔들 완료됨
        last_completed_candle_hour = 9
    elif current_hour >= 5:   # 5시 이후 → 05시 캔들 완료됨
        last_completed_candle_hour = 5
    elif current_hour >= 1:   # 1시 이후 → 01시 캔들 완료됨
        last_completed_candle_hour = 1
    else:  # 1시 이전 (0시대) → 전날 21시 캔들 완료됨
        last_completed_candle_hour = 21
        target_date = (current_kst_time - timedelta(days=1)).date()
    
    # 마지막 완료된 캔들 시간 생성
    result = current_kst_time.replace(
        year=target_date.year,
        month=target_date.month, 
        day=target_date.day,
        hour=last_completed_candle_hour, 
        minute=0, 
        second=0, 
        microsecond=0
    )
    
    print(f"🕐 Current KST: {current_kst_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Last completed candle: {result.strftime('%Y-%m-%d %H:%M:%S KST')}")
    
    return result

def find_last_completed_candle_start_time(current_kst_time=None):
    """가장 최근에 완료된 4시간 캔들의 '시작' 시각(KST)을 반환"""
    if current_kst_time is None:
        current_kst_time = get_current_kst_time()
    # 기존 유틸: 마지막 완료 '완료 시각'(01/05/09/13/17/21)을 반환
    last_completed_end = find_last_completed_kst_interval(current_kst_time)
    return last_completed_end - timedelta(hours=4)

def find_last_completed_candle_start_time_before_point(target_kst_time):
    """특정 시점 직전에 완료된 4시간 캔들의 '시작' 시각(KST)을 반환"""
    current_hour = target_kst_time.hour
    current_minute = target_kst_time.minute
    
    # 4시간 캔들 완료 시간: 01시, 05시, 09시, 13시, 17시, 21시
    # 경계값 로직 수정: 만약 입력이 정확히 17:00이라면, 17:00에 "끝난" 캔들을 찾아야 함 (시작 13:00)
    # 기존 로직은 17:00 입력 시 17:00에 "시작하는" 캔들을 가리킬 수 있었음
    
    last_completed_candle_end_hour = None
    target_date = target_kst_time.date()
    
    # 정확한 경계 시간(분=0)인지 확인
    is_exact_boundary = (current_minute == 0)
    
    if current_hour > 21 or (current_hour == 21 and is_exact_boundary):
        last_completed_candle_end_hour = 21
    elif current_hour > 17 or (current_hour == 17 and is_exact_boundary):
        last_completed_candle_end_hour = 17
    elif current_hour > 13 or (current_hour == 13 and is_exact_boundary):
        last_completed_candle_end_hour = 13
    elif current_hour > 9 or (current_hour == 9 and is_exact_boundary):
        last_completed_candle_end_hour = 9
    elif current_hour > 5 or (current_hour == 5 and is_exact_boundary):
        last_completed_candle_end_hour = 5
    elif current_hour > 1 or (current_hour == 1 and is_exact_boundary):
        last_completed_candle_end_hour = 1
    else:  
        # 01:00 미만 또는 (01:00이 아닌 00:XX 등) -> 전날 21:00 완료
        last_completed_candle_end_hour = 21  
        target_date = (target_kst_time - timedelta(days=1)).date()
    
    # 마지막 완료된 캔들의 완료 시간 (패턴의 마지막 시점)
    completed_time = target_kst_time.replace(
        year=target_date.year,
        month=target_date.month, 
        day=target_date.day,
        hour=last_completed_candle_end_hour, 
        minute=0, 
        second=0, 
        microsecond=0
    )
    
    # 시작 시간 = 완료 시간 - 4시간
    start_time = completed_time - timedelta(hours=4)
    
    print(f"🕐 Target KST: {target_kst_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Adjusted Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S KST')} (Ends at {completed_time.strftime('%H:%M')})")
    
    return start_time

def convert_kst_to_utc_timestamp(kst_datetime):
    """KST datetime을 UTC 타임스탬프(밀리초)로 변환"""
    utc_dt = kst_datetime.astimezone(pytz.UTC)
    return int(utc_dt.timestamp() * 1000)

def convert_utc_to_kst(utc_timestamp_ms):
    """UTC 타임스탬프(밀리초)를 KST datetime으로 변환"""
    utc_dt = datetime.fromtimestamp(utc_timestamp_ms / 1000, tz=pytz.UTC)
    return utc_dt.astimezone(KST)

def sync_binance_time():
    """Binance 서버 시간 동기화"""
    global binance_exchange
    try:
        if binance_exchange is None:
            print("❌ Binance exchange not initialized")
            return False
            
        # 기존 타임아웃 설정 백업
        original_timeout = getattr(binance_exchange, 'timeout', 10000)
        
        # 서버 시간 가져오기용 짧은 타임아웃 설정 (5초)
        binance_exchange.timeout = 5000
        
        # Binance 서버 시간 가져오기
        server_time = binance_exchange.fetch_time()
        local_time = binance_exchange.milliseconds()
        
        # 원래 타임아웃으로 복원
        binance_exchange.timeout = original_timeout
        
        # 시간 차이 계산 (밀리초)
        time_offset = server_time - local_time
        
        # 시간 차이를 0에 가깝게 보정 (반대 값 사용)
        safe_offset = -time_offset
        
        # 시간 차이를 거래소 객체에 설정
        binance_exchange.options['timeDifference'] = safe_offset
        
        print(f"🕐 Binance time sync: offset={time_offset}ms → safe_offset={safe_offset}ms (server={server_time}, local={local_time})")
        
        # 시간 차이에 따른 로그 레벨 조정
        if abs(time_offset) > 1000:
            print(f"⚠️  Large time difference detected: {time_offset}ms")
        elif abs(time_offset) < 50:
            print("✅ Time sync excellent: < 50ms difference")
            
        return True
    except Exception as e:
        print(f"❌ Time sync failed: {e}")
        # 타임아웃 복원 시도
        try:
            if binance_exchange:
                binance_exchange.timeout = original_timeout
        except:
            pass
        return False

def initialize_binance():
    """바이낸스 거래소 초기화 (Public API 사용)"""
    global binance_exchange
    try:
        # API 키 확인
        api_key = CONFIG['binance_api_key']
        api_secret = CONFIG['binance_api_secret']
        
        if api_key and api_secret:
            # API 키가 있는 경우 인증된 연결
            binance_exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'timeDifference': 0,
                }
            })
            print("✅ Binance exchange initialized with API credentials")
        else:
            # API 키가 없는 경우 Public API만 사용
            binance_exchange = ccxt.binance({
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'timeDifference': 0,
                }
            })
            print("✅ Binance exchange initialized with Public API (no credentials)")
        
        # 초기 시간 동기화
        if sync_binance_time():
            print("✅ Initial time synchronization completed")
        else:
            print("⚠️  Initial time synchronization failed")
            
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Binance exchange: {e}")
        return False


def _minmax(data, epsilon=1e-8):
    """MinMax 스케일링 함수 (trading 스크립트에서 가져옴)"""
    min_val = torch.min(data, dim=1, keepdim=True)[0]
    max_val = torch.max(data, dim=1, keepdim=True)[0]
    range_val = max_val - min_val
    range_val = torch.where(range_val < epsilon, torch.tensor(epsilon, dtype=range_val.dtype, device=range_val.device), range_val)
    normalized = (data - min_val) / range_val
    return normalized

def safe_fetch_ohlcv(exchange, symbol, timeframe, limit=None, since=None, max_retries=3):
    """안전한 OHLCV 가져오기 (시간 동기화 및 재시도 포함)"""
    for attempt in range(max_retries):
        try:
            # Binance API 호출 전 매번 시간 동기화
            if hasattr(exchange, 'id') and exchange.id == 'binance':
                if not sync_binance_time():
                    print(f"⚠️  Time sync failed on attempt {attempt + 1}")
                    
            # API 호출
            if since is not None:
                ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            else:
                ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            print(f"✅ Successfully fetched {len(ohlcvs)} candles on attempt {attempt + 1}")
            return ohlcvs
            
        except ccxt.NetworkError as e:
            error_msg = str(e)
            if "-1021" in error_msg or "Timestamp" in error_msg:
                print(f"🔄 Timestamp error on attempt {attempt + 1}, retrying with fresh sync...")
                if attempt < max_retries - 1:
                    # 강제 재동기화 시도
                    sync_binance_time()
                    import time
                    time.sleep(1)  # 1초 대기 후 재시도
                    continue
            raise e
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  API call failed on attempt {attempt + 1}: {e}")
                import time
                time.sleep(1)
                continue
            raise e
    
    raise Exception(f"Failed to fetch OHLCV data after {max_retries} attempts")

@cached(ttl=300)  # 5분 캐시
def fetch_and_preprocess_data(exchange, symbol, timeframe, sequence_length, scaling_method='original_minmax', epsilon=1e-8, is_auto_analysis=False):
    """OHLCV 데이터를 가져와 모델을 위해 전처리합니다. (KST 기준 정렬)"""
    try:
        # KST 기준으로 가장 최근 완료된 구간의 끝 시간 찾기
        last_completed_end_time = find_last_completed_kst_interval()
        
        if is_auto_analysis:
            # 자동분석은 한 캔들 전 데이터 사용
            auto_analysis_end_time = last_completed_end_time - timedelta(hours=4)
            end_utc_timestamp = convert_kst_to_utc_timestamp(auto_analysis_end_time)
            print(f"🤖 Auto-analysis: Using one candle earlier - {auto_analysis_end_time.strftime('%Y-%m-%d %H:%M:%S KST')}")
        else:
            # 일반 분석은 최신 완료 시점 사용
            end_utc_timestamp = convert_kst_to_utc_timestamp(last_completed_end_time)
        
        # 해당 시점을 마지막으로 하는 sequence_length개 캔들 가져오기 (완료된 캔들까지 포함)
        # 4시간 캔들이므로 4 * 60 * 60 * 1000ms = 14400000ms per candle
        start_utc_timestamp = end_utc_timestamp - ((sequence_length - 1) * 4 * 60 * 60 * 1000)
        
        if is_auto_analysis:
            print(f"📊 [DEBUG] Auto-analysis: Fetching {sequence_length} candles ending at KST {auto_analysis_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"📊 [DEBUG] Fetching {sequence_length} candles ending at KST {last_completed_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 [DEBUG] UTC timestamps: start={start_utc_timestamp}, end={end_utc_timestamp}")
        
        # 안전한 OHLCV 가져오기 (재시도 로직 포함)
        ohlcvs = safe_fetch_ohlcv(exchange, symbol, timeframe, 
                                 limit=sequence_length, 
                                 since=start_utc_timestamp)
        if len(ohlcvs) < sequence_length:
            logging.warning(f"데이터 부족: {len(ohlcvs)}개 가져옴, {sequence_length}개 필요")
            return None, None, None

        # 검증을 위해 가져온 모든 캔들 로깅 (KST 시간으로 표시)
        ohlc_strings = []
        for i, ohlc in enumerate(ohlcvs):
            kst_time = convert_utc_to_kst(ohlc[0])
            ts = kst_time.strftime('%Y-%m-%d %H:%M:%S')
            ohlc_strings.append(f"#{i+1} {ts} O:{ohlc[1]} H:{ohlc[2]} L:{ohlc[3]} C:{ohlc[4]} V:{ohlc[5]}")
        print(f"가져온 캔들 (총 {len(ohlcvs)}개, KST 시간):\n" + "\n".join(ohlc_strings))
        print(f"입력용 캔들: 모든 {sequence_length}개 (#{1}-#{sequence_length}) - 마지막 완료된 캔들까지 포함")

        # 입력으로 사용할 모든 `sequence_length`개의 캔들 (마지막 완료된 캔들까지 포함)
        input_candles = np.array([ohlc[1:5] for ohlc in ohlcvs], dtype=np.float32)  # OHLC 데이터만 사용

        # 스케일링 및 모델 입력을 위한 형태 변경: (1, seq_len, features)
        input_data = torch.FloatTensor(input_candles).unsqueeze(0)

        # 스케일링 적용
        if scaling_method == 'original_minmax':
            input_data_scaled = _minmax(input_data.clone(), epsilon)  # 원본 수정을 피하기 위해 clone 사용
        else:
            input_data_scaled = input_data  # 스케일링 없음 또는 알 수 없는 방법

        return input_data_scaled, ohlcvs, input_candles

    except ccxt.NetworkError as e:
        logging.error(f"OHLCV 데이터 가져오기 중 네트워크 오류: {e}")
    except ccxt.ExchangeError as e:
        logging.error(f"OHLCV 데이터 가져오기 중 거래소 오류: {e}")
    except Exception as e:
        logging.error(f"OHLCV 데이터 처리 중 오류: {e}")

    return None, None, None

def perform_pattern_search(query_normalized, query_length, top_k, target_length=None):
    """1단계 패턴 검색 (Zero Padding for v8)"""
    # Renamed from perform_2_stage_pattern_search to perform_pattern_search
    # Removes Reranker and uses simple padding
    import torch.nn.functional as F
    
    print(f"🔍 [DEBUG] perform_pattern_search (v8) called: query_length={query_length}, top_k={top_k}, target_length={target_length}")
    print(f"🔍 [DEBUG] Query shape: {query_normalized.shape}")
    
    try:
        # CUDA 우선 사용, 실패시 CPU 폴백
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
        except Exception as e:
            print(f"⚠️ CUDA error, falling back to CPU: {e}")
            device = torch.device('cpu')
            model.to(device)
        
        model.eval()
        with torch.no_grad():
            # 원본 쿼리를 텐서로 변환 (seq_len, 4) -> (1, seq_len, 4)
            query_tensor = torch.from_numpy(query_normalized.astype(np.float32)).unsqueeze(0).to(device)
            current_len = query_tensor.shape[1]
            max_len = CONFIG['max_pattern_len'] # 100

            # Zero Padding (v8 compatible)
            if current_len < max_len:
                padding = torch.zeros(1, max_len - current_len, 4, dtype=query_tensor.dtype, device=device)
                query_tensor_padded = torch.cat([query_tensor, padding], dim=1)
            else:
                query_tensor_padded = query_tensor[:, :max_len, :]
            
            print(f"🔍 [DEBUG] Query shape after padding: {query_tensor_padded.shape}")
            
            # Retriever 모델로 임베딩 생성
            query_emb = model(query_tensor_padded).cpu().numpy().flatten()
        
        # 유사 패턴 검색 (1단계만 수행)
        final_similarities = find_similar_patterns(
            query_emb, embedding_data, device,
            top_k=top_k, target_length=target_length, power=query_length
        )
        
        print(f"🔍 [DEBUG] Final results count: {len(final_similarities)}")
        print(f"🔍 [DEBUG] Top 5: {[(c['idx'], c['sim']) for c in final_similarities[:5]]}")
        
        return final_similarities
        
    except Exception as e:
        print(f"Error in pattern search: {e}")
        import traceback
        traceback.print_exc()
        return []

def perform_historical_pattern_search(query_time_str, query_normalized, query_length, top_k, target_length=None):
    """과거 시점 패턴 검색 (Zero Padding for v8)"""

    try:
        # 캐시 키 생성 (실시간 분석은 캐시 비활성화)
        # current_time = datetime.now().strftime('%Y-%m-%d %H:%M') # Not used for historical
        # Cache key modified to include top_k and target_length
        cache_key = (query_time_str, query_length, target_length, top_k)
        
        # 1. 캐시 확인 (Removed specific reranker cache logic, maybe re-enable later if needed)
        # For v8, we might rely on different caching or just fast execution.
        
        # 1단계: Retriever를 위해 쿼리를 Zero Padding (v8 compatible)
        # CUDA 우선 사용, 실패시 CPU 폴백
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
        except Exception as e:
            print(f"⚠️ CUDA error, falling back to CPU: {e}")
            device = torch.device('cpu')
            model.to(device)
        
        model.eval()
        with torch.no_grad():
            # 원본 쿼리를 텐서로 변환 (seq_len, 4) -> (1, seq_len, 4)
            query_tensor = torch.from_numpy(query_normalized.astype(np.float32)).unsqueeze(0).to(device)
            current_len = query_tensor.shape[1]
            max_len = CONFIG['max_pattern_len'] # 100

            # Zero Padding (v8 compatible)
            if current_len < max_len:
                padding = torch.zeros(1, max_len - current_len, 4, dtype=query_tensor.dtype, device=device)
                query_tensor_padded = torch.cat([query_tensor, padding], dim=1)
            else:
                query_tensor_padded = query_tensor[:, :max_len, :]
            
            print(f"🔍 [DEBUG] Query shape after padding: {query_tensor_padded.shape}")
            
            # Retriever 모델로 임베딩 생성
            query_emb = model(query_tensor_padded).cpu().numpy().flatten()
        
        # 유사 패턴 검색 (1단계만 수행)
        final_similarities = find_similar_patterns(
            query_emb, embedding_data, device,
            top_k=top_k, target_length=target_length, power=query_length
        )
        
        print(f"🔍 [DEBUG] Final results count: {len(final_similarities)}")
        print(f"🔍 [DEBUG] Top 5: {[(c['idx'], c['sim']) for c in final_similarities[:5]]}")
        
        return final_similarities
        
    except Exception as e:
        print(f"Error in historical pattern search: {e}")
        import traceback
        traceback.print_exc()
        return []

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 재귀적으로 변환"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def initialize_ai_system():
    """AI 시스템 초기화: 모델과 데이터 로드"""
    global model, embedding_data, full_ohlc_data, full_time_data, train_ohlc_data, train_time_data
    
    try:
        print("Initializing pattern detection system...")
        
        
        # 1. 데이터 로드
        full_ohlc_data, full_time_data = load_ohlc_data(CONFIG['csv_path'])
        
        # 2. 학습/테스트 분리
        split_idx = int(len(full_ohlc_data) * 0.9)
        train_ohlc_data = full_ohlc_data[:split_idx]
        train_time_data = full_time_data[:split_idx]
        test_ohlc_data = full_ohlc_data[split_idx:]
        test_time_data = full_time_data[split_idx:]
        
        # 3. Retriever 모델 로드
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PatternEncoder(emb_dim=CONFIG['emb_dim'], max_len=CONFIG['max_pattern_len']).to(device)
        
        if os.path.exists(CONFIG['model_path']):
            model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
            print(f"Retriever model loaded from {CONFIG['model_path']}")
        else:
            print("Warning: Retriever model file not found. Training required.")
            return False
        
        # 4. 임베딩 데이터 로드 (Quantized)
        if os.path.exists(CONFIG['emb_path']):
            with open(CONFIG['emb_path'], 'rb') as f:
                embedding_data = pickle.load(f)
            
            # v8에서는 임베딩이 딕셔너리 형태로 저장됨 {'embeddings': ..., 'meta': ...}
            if isinstance(embedding_data, dict) and 'embeddings' in embedding_data:
                # 임베딩이 양자화(int8) 되어 있는지 확인하고 로드
                # find_similar_patterns 함수 내부에서 처리하므로 그대로 둠
                print(f"Embeddings loaded from {CONFIG['emb_path']} (Keys: {embedding_data.keys()})")
            else:
                 print(f"Embeddings loaded from {CONFIG['emb_path']} (Legacy format)")

        else:
            print("Warning: Embeddings file not found. Precomputing required.")
            return False
        
        # 5. 바이낸스 거래소 초기화
        initialize_binance()
        
        # 5. 자동 분석 스케줄러 초기화
        global auto_scheduler
        def analysis_wrapper(**kwargs):
            """스케줄러용 분석 함수 래퍼"""
            try:
                print(f"🤖 [AUTO-ANALYSIS] Starting with params: {kwargs}")
                # live_analysis_api와 동일한 로직 사용
                input_data_scaled, ohlcvs, input_candles = fetch_and_preprocess_data(
                    binance_exchange, kwargs['symbol'], kwargs['timeframe'], kwargs['query_length'], is_auto_analysis=True
                )
                
                if input_data_scaled is None:
                    return None
                
                query_normalized = normalize_window(input_candles)
                print(f"🤖 [AUTO-ANALYSIS] Input candles shape: {input_candles.shape}")
                print(f"🤖 [AUTO-ANALYSIS] Input candles first 3: {input_candles[:3] if len(input_candles) >= 3 else input_candles}")
                print(f"🤖 [AUTO-ANALYSIS] Normalized shape: {query_normalized.shape}")
                print(f"🤖 [AUTO-ANALYSIS] Normalized first 3: {query_normalized[:3] if len(query_normalized) >= 3 else query_normalized}")
                
                similarities = perform_pattern_search(
                    query_normalized, kwargs['query_length'], kwargs['top_k'], kwargs.get('target_length')
                )
                
                print(f"🤖 [AUTO-ANALYSIS] Final similarities count: {len(similarities)}")
                for i, sim in enumerate(similarities[:3]):
                    print(f"  #{i+1}: idx={sim['idx']}, sim={sim['sim']:.6f}, len={sim['len']}")
                
                # 결과 포맷팅
                result_patterns = []
                for sim_info in similarities:
                    pattern_start_idx = sim_info['idx']
                    pattern_length = sim_info['len']
                    forecast_length = math.ceil(pattern_length / 3)
                    
                    pattern_data = convert_numpy_types(full_ohlc_data[pattern_start_idx:pattern_start_idx + pattern_length])
                    # 패턴의 마지막 캔들 완료 시점 (마지막 캔들의 시작 시점)
                    pattern_start_time = full_time_data.iloc[pattern_start_idx + pattern_length - 1]
                    
                    forecast_start_idx = pattern_start_idx + pattern_length
                    forecast_end_idx = forecast_start_idx + forecast_length
                    forecast_data = None
                    
                    if forecast_end_idx <= len(full_ohlc_data):
                        forecast_data = convert_numpy_types(full_ohlc_data[forecast_start_idx:forecast_end_idx])
                    
                    # CSV 시간이 4시간 간격이 아닐 수 있으므로 올바른 4시간 완료 시점으로 정규화
                    # 패턴 시작 시점에서 4시간을 더해 완료 시점 계산
                    pattern_completion_time = pattern_start_time + timedelta(hours=4)
                    formatted_pattern = pattern_to_frontend_format(
                        sim_info, pattern_data, pattern_completion_time, forecast_data
                    )
                    if formatted_pattern:
                        result_patterns.append(formatted_pattern)
                
                # 쿼리 패턴 정보 생성 (게스트 UI 표시용)
                query_candles = []
                for ohlc in input_candles:
                    query_candles.append({
                        'open': float(ohlc[0]),
                        'high': float(ohlc[1]),
                        'low': float(ohlc[2]),
                        'close': float(ohlc[3]),
                        'volume': 1000000  # 임시 볼륨
                    })
                
                # 자동분석은 한 캔들 전 시점 사용
                # 현재 완료된 캔들에서 4시간(1캔들) 전 시점으로 계산
                current_completed_time = find_last_completed_candle_time()
                auto_analysis_time = current_completed_time - timedelta(hours=4)
                query_pattern = {
                    'timestamp': auto_analysis_time.strftime('%Y.%m.%d %H:%M'),
                    'symbol': kwargs['symbol'],
                    'confidence': 95,
                    'candles': query_candles,
                    'source': 'auto_analysis'
                }

                return {
                    'query_pattern': query_pattern,
                    'similar_patterns': result_patterns,
                    'debug_info': {
                        'analysis_type': 'auto_analysis',
                        'candle_count': len(ohlcvs) if ohlcvs else 0,
                        'first_candle_time': convert_utc_to_kst(ohlcvs[0][0]).strftime('%Y-%m-%d %H:%M:%S KST') if ohlcvs and len(ohlcvs) > 0 else None,
                        'last_candle_time': convert_utc_to_kst(ohlcvs[-1][0]).strftime('%Y-%m-%d %H:%M:%S KST') if ohlcvs and len(ohlcvs) > 0 else None,
                        'input_candles_shape': input_candles.shape if input_candles is not None else None,
                        'normalized_data_sample': query_normalized[:2].tolist() if query_normalized is not None and len(query_normalized) >= 2 else None,
                        'similarities_raw': [{'idx': s['idx'], 'sim': float(s['sim']), 'len': s['len']} for s in similarities[:3]] if similarities else []
                    }
                }
                
            except Exception as e:
                print(f"Auto analysis wrapper error: {e}")
                return None
        
        auto_scheduler = AutoAnalysisScheduler(analysis_wrapper)
        auto_scheduler.start()
        
        print("System initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        return False

def get_current_pattern():
    """현재 시점의 패턴 데이터 생성 (최신 5개 캔들)"""
    try:
        # 최신 5개 캔들 사용
        current_raw = full_ohlc_data[-5:]
        current_normalized = normalize_window(current_raw)
        
        # 현재 시간 (데이터의 마지막 시간)
        current_time = full_time_data.iloc[-1]
        
        # 캔들 데이터를 프론트엔드 형식으로 변환
        candles = []
        for ohlc in current_raw:
            candles.append({
                'open': float(ohlc[0]),
                'high': float(ohlc[1]),
                'low': float(ohlc[2]),
                'close': float(ohlc[3]),
                'volume': 1000000  # 임시 볼륨 데이터
            })
        
        return {
            'timestamp': current_time.strftime("%Y.%m.%d %H:%M"),
            'symbol': 'BTC/USDT',
            'confidence': 92,  # 임시 신뢰도
            'candles': candles,
            'normalized': current_normalized.tolist()
        }
        
    except Exception as e:
        print(f"Error getting current pattern: {e}")
        return None

def pattern_to_frontend_format(pattern_info, pattern_data, time_info, forecast_data=None):
    """패턴 데이터를 프론트엔드 형식으로 변환"""
    try:
        # 한국시간 변환 (UTC+9)
        kst = pytz.timezone('Asia/Seoul')
        if hasattr(time_info, 'tz_localize'):
            # pandas datetime인 경우
            if time_info.tz is None:
                time_info_kst = time_info.tz_localize('UTC').tz_convert(kst)
            else:
                time_info_kst = time_info.tz_convert(kst)
        else:
            # 일반 datetime인 경우
            if time_info.tzinfo is None:
                time_info_kst = pytz.UTC.localize(time_info).astimezone(kst)
            else:
                time_info_kst = time_info.astimezone(kst)
        # 패턴 캔들 데이터
        pattern_candles = []
        for ohlc in pattern_data:
            pattern_candles.append({
                'open': float(ohlc[0]),
                'high': float(ohlc[1]),
                'low': float(ohlc[2]),
                'close': float(ohlc[3]),
                # 'volume': int(np.random.randint(500000, 2000000)),  # 임시 볼륨
                'volume': 1000000,  # 임시 볼륨 (랜덤 값 제거)
                'type': 'pattern'  # 패턴 구간 표시
            })
        
        # 미래 캔들 데이터 (예측 구간)
        forecast_candles = []
        if forecast_data is not None and len(forecast_data) > 0:
            for ohlc in forecast_data:
                forecast_candles.append({
                    'open': float(ohlc[0]),
                    'high': float(ohlc[1]),
                    'low': float(ohlc[2]),
                    'close': float(ohlc[3]),
                    # 'volume': int(np.random.randint(500000, 2000000)),  # 임시 볼륨
                    'volume': 1000000,  # 임시 볼륨 (랜덤 값 제거)
                    'type': 'forecast'  # 예측 구간 표시
                })
            
            # 실제 미래 가격 변화 계산 (패턴 마지막 종가 대비 예상 구간 최대 상승/하락)
            if len(forecast_data) > 0:
                pattern_close = pattern_data[-1][3]  # 패턴 마지막 종가
                
                # 예상 구간의 모든 캔들에서 최고가와 최저가 찾기
                forecast_highs = [candle[1] for candle in forecast_data]  # 모든 high 값
                forecast_lows = [candle[2] for candle in forecast_data]   # 모든 low 값
                
                max_high = max(forecast_highs)  # 예상 구간 최대 고가
                min_low = min(forecast_lows)    # 예상 구간 최소 저가
                
                # 패턴 마지막 종가 대비 최대 상승/하락률 계산
                max_rise = ((max_high - pattern_close) / pattern_close) * 100    # 최대 상승
                max_fall = ((min_low - pattern_close) / pattern_close) * 100     # 최대 하락 (음수)
                
                price_change_7d = round(max_rise, 1)   # 최대 상승률
                price_change_3d = round(max_fall, 1)   # 최대 하락률
            else:
                price_change_7d = 0.0  # 최대 상승
                price_change_3d = 0.0  # 최대 하락
        else:
            # # 미래 데이터가 없는 경우 임시 값
            # price_change_7d = float(np.random.uniform(-20, 20))
            # price_change_3d = float(np.random.uniform(-15, 15))
            # 미래 데이터가 없는 경우 임시 값 (랜덤 값 제거)
            price_change_7d = 0.0
            price_change_3d = 0.0
        
        # 전체 캔들 데이터 (패턴 + 예측)
        all_candles = pattern_candles + forecast_candles
        
        # pattern_detecter.py에서 이미 실제 similarity 공식이 적용되어 전달됨
        # power = round(200 / power) if round(100 / power) % 2 == 1 else round(100 / power) + 1
        # sim = np.sign(cosine_similarity) * np.power(np.abs(cosine_similarity), power)

        # ✅ 시간 정책: time_info를 마지막 캔들의 시작 시각으로 직접 사용
        hour = time_info_kst.hour
        
        # 4시간 캔들의 시작 시각들: 1, 5, 9, 13, 17, 21
        start_hours = {1, 5, 9, 13, 17, 21}
        
        if hour in start_hours:
            # time_info가 이미 시작 시각인 경우
            ts_start = time_info_kst
            ts_complete = time_info_kst + timedelta(hours=4)
        else:
            # time_info가 완료 시각이나 기타 시각인 경우 → 4시간 그리드에 맞춤
            # 가장 가까운 이전 시작 시각을 찾기
            anchors = [1, 5, 9, 13, 17, 21]
            same_day_anchors = [time_info_kst.replace(hour=h, minute=0, second=0, microsecond=0) for h in anchors]
            prev_day_21 = (time_info_kst - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
            
            # time_info 이전 또는 같은 시각의 시작점들만 고려
            candidates = [a for a in same_day_anchors if a <= time_info_kst] + [prev_day_21]
            ts_start = max(candidates) if candidates else time_info_kst.replace(hour=21, minute=0, second=0, microsecond=0) - timedelta(days=1)
            ts_complete = ts_start + timedelta(hours=4)

        # 시점 표시용으로 8시간 빼기
        display_time = ts_start - timedelta(hours=8)
        display_complete = ts_complete - timedelta(hours=8)
        
        return {
            'id': int(pattern_info['idx']),
            # ✅ 호환성: timestamp = 시작 시각 (8시간 뺀 값)
            'timestamp': display_time.strftime("%Y.%m.%d %H:%M"),
            # ✅ 명시적으로 둘 다 제공 (8시간 뺀 값)
            'timestamp_start': display_time.strftime("%Y.%m.%d %H:%M"),
            'timestamp_complete': display_complete.strftime("%Y.%m.%d %H:%M"),
            'similarity': float(pattern_info['sim']),
            'priceChange7d': price_change_7d,
            'priceChange3d': price_change_3d,
            'candles': all_candles,
            'pattern_length': len(pattern_candles),
            'forecast_length': len(forecast_candles),
            'has_forecast': len(forecast_candles) > 0
        }
        
    except Exception as e:
        print(f"Error converting pattern to frontend format: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """API 상태 확인
    ---
    tags:
      - System
    responses:
      200:
        description: API and system component status
        schema:
            type: object
            properties:
                status:
                    type: string
                    example: ok
                model_loaded:
                    type: boolean
                binance_connected:
                    type: boolean
    """
    next_analysis = auto_scheduler.get_next_analysis_time() if auto_scheduler else None
    last_analysis = auto_scheduler.get_last_analysis_time() if auto_scheduler else None
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'embeddings_loaded': embedding_data is not None,
        'cache_loaded': False, # Legacy cache removed
        'cache_size': 0,
        'data_loaded': full_ohlc_data is not None,
        'binance_connected': binance_exchange is not None,
        'auto_scheduler_running': auto_scheduler.is_running if auto_scheduler else False,
        'next_auto_analysis': next_analysis,
        'last_auto_analysis': last_analysis
    })

@app.route('/api/auto-analysis', methods=['GET'])
@require_auth(min_tier=UserTier.GUEST.value)
def get_auto_analysis_api():
    """자동 분석 결과 조회 (게스트 이상)
    ---
    tags:
      - Analysis
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: Bearer token
    responses:
      200:
        description: Cached auto-analysis results
        schema:
          type: object
          properties:
            results:
              type: object
              description: Analysis results including query and similar patterns
            analysis_time:
              type: string
              description: ISO timestamp of analysis
    """
    try:
        # 사용자 권한 확인
        user_tier = request.user_info['tier']
        limits = get_tier_limits(user_tier)
        
        # 게스트는 3:3, top_k=3 조합 조회
        if user_tier == 'guest':
            query_length = 3
            target_length = 3
            top_k = 3
        else:
            # 멤버/프리미엄은 기본 3:3 조합으로 조회 (더 많은 결과)
            query_length = 3
            target_length = 3  
            top_k = limits['max_top_k']
        
        # 🎯 간단하게: 가장 최근 자동분석 캐시 결과 가져오기
        print(f"🔍 Looking for most recent auto analysis cache (q{query_length}:t{target_length}, top{top_k})")
        
        # 데이터베이스에서 가장 최신 캐시 가져오기 (top_k는 >=로 비교하므로 캐시에 더 많은 결과가 있어도 OK)
        results, analysis_time = db_manager.get_latest_auto_analysis(
            symbol="BTC/USDT", 
            timeframe="4h", 
            query_length=query_length, 
            target_length=target_length, 
            top_k=top_k  # 캐시에 top_k >= 이 값인 결과 찾기
        )
        
        if results and analysis_time:
            print(f"✅ Found cached analysis from: {analysis_time}")
            result = (json.dumps(results), analysis_time)
        else:
            print("❌ No matching cache found")
            result = None
        
        if result:
            results = json.loads(result[0])
            analysis_time = result[1]
            print(f"✅ Found most recent cached analysis from {analysis_time}")
            
            # 🎯 게스트/회원용으로 결과 제한
            if results.get('similar_patterns') and len(results['similar_patterns']) > top_k:
                results['similar_patterns'] = results['similar_patterns'][:top_k]
                print(f"    Limited results to top {top_k} for {user_tier}")
            
            # ✅ 캐시된 쿼리 패턴의 시간 정보는 이미 올바르게 설정되어 있으므로 그대로 사용
            # pattern_to_frontend_format()에서 이미 8시간 빼기가 적용되어 있음
        else:
            print(f"❌ No cached analysis found, trying live analysis...")
            # 캐시에 없으면 실시간 분석 실행
            try:
                # auto analysis용 한 캔들 전 데이터로 실시간 분석
                input_data_scaled, ohlcvs, input_candles = fetch_and_preprocess_data(
                    binance_exchange, "BTC/USDT", "4h", query_length, is_auto_analysis=True
                )
                
                if input_data_scaled is not None:
                    query_normalized = normalize_window(input_candles)
                    similarities = perform_pattern_search(
                        query_normalized, query_length, top_k, target_length
                    )
                    
                    # 결과 포맷팅 (auto analysis와 동일한 방식)
                    result_patterns = []
                    for sim_info in similarities:
                        pattern_start_idx = sim_info['idx']
                        pattern_length = sim_info['len']
                        forecast_length = math.ceil(pattern_length / 3)
                        
                        pattern_data = convert_numpy_types(full_ohlc_data[pattern_start_idx:pattern_start_idx + pattern_length])
                        # 패턴의 마지막 캔들 완료 시점 (CSV에서 직접 가져오기)
                        pattern_completion_time = full_time_data.iloc[pattern_start_idx + pattern_length - 1]
                        
                        forecast_start_idx = pattern_start_idx + pattern_length
                        forecast_end_idx = forecast_start_idx + forecast_length
                        forecast_data = None
                        
                        if forecast_end_idx <= len(full_ohlc_data):
                            forecast_data = convert_numpy_types(full_ohlc_data[forecast_start_idx:forecast_end_idx])
                        
                        formatted_pattern = pattern_to_frontend_format(
                            sim_info, pattern_data, pattern_completion_time, forecast_data
                        )
                        if formatted_pattern:
                            result_patterns.append(formatted_pattern)
                    
                    # 쿼리 패턴 정보 생성
                    query_candles = []
                    for ohlc in input_candles:
                        query_candles.append({
                            'open': float(ohlc[0]),
                            'high': float(ohlc[1]),
                            'low': float(ohlc[2]),
                            'close': float(ohlc[3]),
                            'volume': 1000000
                        })
                    
                    # ✅ 올바른 시간 기점 사용 (쿼리 패턴의 마지막 캔들 완료 시간)
                    # 자동 분석은 한 캔들 전 데이터를 사용하므로, 해당 시점의 완료 시간을 계산
                    auto_analysis_time = get_current_kst_time() - timedelta(hours=4)
                    pattern_completion_time = find_last_completed_candle_time(auto_analysis_time)
                    
                    # 8시간 빼기 적용 (display_time)
                    display_time = pattern_completion_time - timedelta(hours=8)
                    pattern_start_time = pattern_completion_time - timedelta(hours=4*query_length)
                    display_start_time = pattern_start_time - timedelta(hours=8)
                    
                    query_pattern = {
                        'timestamp': display_time.strftime('%Y.%m.%d %H:%M'),
                        'timestamp_start': display_start_time.strftime('%Y.%m.%d %H:%M'),
                        'timestamp_complete': display_time.strftime('%Y.%m.%d %H:%M'),
                        'symbol': "BTC/USDT",
                        'confidence': 95,
                        'candles': query_candles,
                        'source': 'live_analysis'
                    }
                    
                    results = {
                        'query_pattern': query_pattern,
                        'similar_patterns': result_patterns
                    }
                    analysis_time = datetime.now().isoformat()
                    
                    print(f"✅ Live analysis completed for {query_pattern['timestamp']}")
                else:
                    print("❌ Failed to fetch live data")
            except Exception as e:
                print(f"❌ Live analysis error: {e}")
        
        if not results:
            return jsonify({'error': 'No analysis results available'}), 404
        
        # 결과 개수 제한 (추가 안전장치)
        if 'similar_patterns' in results:
            max_results = limits['max_top_k']
            results['similar_patterns'] = results['similar_patterns'][:max_results]
        
        return jsonify({
            'analysis_time': analysis_time,
            'results': results,
            'user_tier': user_tier,
            'limits': limits,
            'query_config': {
                'query_length': query_length,
                'target_length': target_length,
                'top_k': top_k
            }
        })
        
    except Exception as e:
        print(f"Error in auto analysis API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/current-pattern', methods=['GET'])
def get_current_pattern_api():
    """현재 패턴 정보 반환"""
    if model is None or embedding_data is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    current_pattern = get_current_pattern()
    if current_pattern is None:
        return jsonify({'error': 'Failed to get current pattern'}), 500
    
    return jsonify(current_pattern)

@app.route('/api/latest-time', methods=['GET'])
def get_latest_time_api():
    """가장 최근에 완료된 캔들 시간 반환
    ---
    tags:
      - System
    description: Returns the latest completed candle time in KST
    responses:
      200:
        description: Latest candle timestamp information
        schema:
            type: object
            properties:
                latest_time:
                    type: string
                    example: "2024-12-12T17:00"
                kst_time:
                    type: string
                timestamp:
                    type: string
    """
    try:
        # 가장 최근에 완료된 캔들 시간 (KST) 가져오기
        latest_completed_time = find_last_completed_candle_time()
        
        # datetime-local 형식으로 변환 (YYYY-MM-DDTHH:MM)
        formatted_time = latest_completed_time.strftime('%Y-%m-%dT%H:%M')
        
        return jsonify({
            'latest_time': formatted_time,
            'kst_time': latest_completed_time.strftime('%Y-%m-%d %H:%M:%S KST'),
            'timestamp': latest_completed_time.isoformat()
        })
    except Exception as e:
        print(f"Error getting latest time: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/live-analysis', methods=['POST'])
@require_auth(min_tier=UserTier.GUEST.value)
def live_analysis_api():
    """실시간 바이낸스 데이터로 패턴 분석 (게스트 이상)"""
    if model is None or embedding_data is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    if binance_exchange is None:
        return jsonify({'error': 'Binance exchange not initialized'}), 500
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC/USDT')
        timeframe = data.get('timeframe', '4h')
        query_length = data.get('query_length', 5)
        target_length = data.get('target_length')  # None이면 모든 길이
        top_k = data.get('top_k', 3)
        custom_candles = data.get('custom_candles')  # 커스텀 캔들 데이터
        
        print(f"👤 [MANUAL-ANALYSIS] Live analysis parameters:")
        print(f"  Symbol: {symbol}")
        print(f"  Timeframe: {timeframe}")
        print(f"  Query length: {query_length}")
        print(f"  Target length: {target_length}")
        print(f"  Top K: {top_k}")
        print(f"  Custom candles: {'Yes' if custom_candles else 'No'}")
        
        # 🎯 커스텀 캔들이 아닌 경우에만 캐시 확인
        if not custom_candles:
            print(f"🔍 [CACHE-CHECK] Checking cache for live analysis...")
            cache_results, cache_time = db_manager.get_latest_auto_analysis(
                symbol=symbol, 
                timeframe=timeframe, 
                query_length=query_length, 
                target_length=target_length if target_length else 3,  # None일 때 기본값
                top_k=top_k
            )
            
            if cache_results and cache_results.get('similar_patterns'):
                print(f"✅ [CACHE-HIT] Found cached results from {cache_time}")
                print(f"📊 [CACHE-HIT] {len(cache_results['similar_patterns'])} patterns in cache")
                
                # 캐시된 결과 반환
                return jsonify({
                    'success': True,
                    'query_pattern': cache_results.get('query_pattern'),
                    'similar_patterns': cache_results.get('similar_patterns'),
                    'live_data_info': cache_results.get('live_data_info'),
                    'source': 'cache',
                    'cache_time': cache_time
                })
            else:
                print(f"❌ [CACHE-MISS] No cache found, proceeding with live analysis...")
        
        if custom_candles:
            print(f"🎯 [CUSTOM-ANALYSIS] Starting custom candle analysis")
            print(f"  Custom candles data: {len(custom_candles)} candles")
            
            # 커스텀 캔들 데이터 처리
            # import numpy as np
            
            # 커스텀 캔들을 numpy 배열로 변환
            input_candles = np.array(custom_candles, dtype=np.float32)
            print(f"🎯 [CUSTOM-ANALYSIS] Input candles shape: {input_candles.shape}")
            print(f"🎯 [CUSTOM-ANALYSIS] Input candles: {input_candles}")
            
            # 정규화 (패턴 검출용)
            query_normalized = normalize_window(input_candles)
            print(f"🎯 [CUSTOM-ANALYSIS] Normalized shape: {query_normalized.shape}")
            print(f"🎯 [CUSTOM-ANALYSIS] Normalized first 3: {query_normalized[:3] if len(query_normalized) >= 3 else query_normalized}")
            
            # 분석을 위한 가짜 ohlcvs 데이터 생성 (차트 표시용)
            ohlcvs = []
            for i, candle in enumerate(input_candles):
                ohlcvs.append({
                    'time': int((pd.Timestamp.now() - pd.Timedelta(hours=(len(input_candles) - i))).timestamp()),
                    'open': float(candle[0]),
                    'high': float(candle[1]),
                    'low': float(candle[2]),
                    'close': float(candle[3]),
                    'volume': 1000.0  # 임시 볼륨
                })
        else:
            print(f"👤 [MANUAL-ANALYSIS] Starting live analysis for {symbol} on {timeframe} timeframe")
            
            # 1. 바이낸스에서 실시간 데이터 가져오기
            input_data_scaled, ohlcvs, input_candles = fetch_and_preprocess_data(
                binance_exchange, symbol, timeframe, query_length
            )
            
            if input_data_scaled is None:
                return jsonify({'error': 'Failed to fetch live data from Binance'}), 500
            
            # 2. 정규화 (패턴 검출용)
            query_normalized = normalize_window(input_candles)
            print(f"👤 [MANUAL-ANALYSIS] Input candles shape: {input_candles.shape}")
            print(f"👤 [MANUAL-ANALYSIS] Input candles first 3: {input_candles[:3] if len(input_candles) >= 3 else input_candles}")
            print(f"👤 [MANUAL-ANALYSIS] Normalized shape: {query_normalized.shape}")
            print(f"👤 [MANUAL-ANALYSIS] Normalized first 3: {query_normalized[:3] if len(query_normalized) >= 3 else query_normalized}")
        
        # 공통: 패턴 검색 (Reranker 제거됨)
        similarities = perform_pattern_search(
            query_normalized, query_length, top_k, target_length
        )
        
        print(f"👤 [MANUAL-ANALYSIS] Final similarities count: {len(similarities)}")
        for i, sim in enumerate(similarities[:3]):
            print(f"  #{i+1}: idx={sim['idx']}, sim={sim['sim']:.6f}, len={sim['len']}")
        
        # numpy 타입을 Python 타입으로 변환
        similarities = convert_numpy_types(similarities)
        
        print(f"Converted similarities: {similarities}")
        
        # 5. 프론트엔드 형식으로 변환
        result_patterns = []
        for sim_info in similarities:
            try:
                pattern_start_idx = sim_info['idx']
                pattern_length = sim_info['len']
                forecast_length = math.ceil(pattern_length / 3)  # 패턴 길이의 1/3만큼 미래 예측
                
                if pattern_start_idx + pattern_length > len(full_time_data):
                    continue

                # 패턴 데이터 (numpy 배열을 Python list로 변환)
                pattern_data = full_ohlc_data[pattern_start_idx:pattern_start_idx + pattern_length]
                pattern_data = convert_numpy_types(pattern_data)
                # 패턴의 마지막 캔들 완료 시점 (CSV에서 직접 가져오기)
                pattern_completion_time = full_time_data.iloc[pattern_start_idx + pattern_length - 1]
                
                # 미래 데이터 (예측 구간)
                forecast_start_idx = pattern_start_idx + pattern_length
                forecast_end_idx = forecast_start_idx + forecast_length
                forecast_data = None
                
                if forecast_end_idx <= len(full_ohlc_data):
                    forecast_data = full_ohlc_data[forecast_start_idx:forecast_end_idx]
                    forecast_data = convert_numpy_types(forecast_data)
                
                formatted_pattern = pattern_to_frontend_format(
                    sim_info, pattern_data, pattern_completion_time, forecast_data
                )
                if formatted_pattern:
                    result_patterns.append(formatted_pattern)
            except Exception as e:
                print(f"⚠️ Error processing similarity result {sim_info}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 6. 현재 패턴 정보 생성
        candles = []
        for ohlc in input_candles:
            candles.append({
                'open': float(ohlc[0]),
                'high': float(ohlc[1]),
                'low': float(ohlc[2]),
                'close': float(ohlc[3]),
                'volume': 1000000  # 임시 볼륨
            })
        
        current_time = datetime.now().strftime("%Y.%m.%d %H:%M")
        # confidence = int(85 + np.random.randint(-10, 15))  # 임시 신뢰도
        confidence = 85  # 임시 신뢰도 (랜덤 값 제거)
        
        live_pattern = {
            'timestamp': current_time,
            'symbol': symbol,
            'confidence': confidence,
            'candles': candles,
            'source': 'live_binance'
        }
        
        result = {
            'query_pattern': live_pattern,
            'similar_patterns': result_patterns,
            'live_data_info': {
                'symbol': symbol,
                'timeframe': timeframe,
                'query_length': query_length,
                'fetched_candles': len(ohlcvs),
                'used_candles': len(input_candles),
                'latest_candle_time': datetime.fromtimestamp(ohlcvs[-1][0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_candles_time_range': f"{datetime.fromtimestamp(ohlcvs[0][0] / 1000).strftime('%Y-%m-%d %H:%M')} ~ {datetime.fromtimestamp(ohlcvs[query_length-1][0] / 1000).strftime('%Y-%m-%d %H:%M')}",
                'data_source': 'Binance API'
            },
            'debug_info': {
                'analysis_type': 'manual_analysis',
                'candle_count': len(ohlcvs),
                'first_candle_time': convert_utc_to_kst(ohlcvs[0][0]).strftime('%Y-%m-%d %H:%M:%S KST'),
                'last_candle_time': convert_utc_to_kst(ohlcvs[-1][0]).strftime('%Y-%m-%d %H:%M:%S KST'),
                'input_candles_shape': input_candles.shape,
                'normalized_data_sample': query_normalized[:2].tolist() if len(query_normalized) >= 2 else query_normalized.tolist()
            }
        }
        
        # 최종 안전장치: 모든 numpy 타입 변환
        result = convert_numpy_types(result)
        
        # 🎯 커스텀 캔들이 아니고 캐시 미스였다면 결과를 캐시에 저장
        if not custom_candles and 'cache_time' not in result:
            try:
                # from datetime import datetime
                import pytz
                KST = pytz.timezone('Asia/Seoul')
                current_time_kst = datetime.now(KST)
                
                print(f"💾 [CACHE-SAVE] Saving analysis results to cache...")
                db_manager.cache_auto_analysis(
                    analysis_time=current_time_kst,
                    symbol=symbol,
                    timeframe=timeframe,
                    query_length=query_length,
                    target_length=target_length if target_length else 3,
                    top_k=top_k,
                    results=result
                )
                print(f"✅ [CACHE-SAVE] Results cached successfully")
                result['source'] = 'live_analysis_cached'
            except Exception as cache_error:
                print(f"⚠️ [CACHE-SAVE] Failed to cache results: {cache_error}")
                result['source'] = 'live_analysis'
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in live analysis: {e}")
        return jsonify({'error': str(e)}), 500

@cached(ttl=180)  # 3분 캐시
def _find_similar_patterns_cached(query_data_str, query_length, top_k, target_length):
    """캐시된 유사 패턴 검색 헬퍼"""
    try:
        query_normalized = np.fromstring(query_data_str, sep=',').reshape(-1, 4)
        
        # 모델 추론
        device = next(model.parameters()).device
        query_tensor = torch.tensor(query_normalized, dtype=torch.float32)
        query_tensor_resized = query_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_emb = model(query_tensor_resized).cpu().numpy().flatten()
        
        candidate_patterns = find_similar_patterns(
            query_emb, embedding_data, device,
            top_k=CONFIG['candidate_count'], target_length=target_length, power=query_length
        )
        
        # v8: Re-ranking Removed
        # No reranking needed for similar patterns cache helper
        
        return {
            'status': 'success',
            'patterns': [pattern_to_frontend_format(
                pattern_info={
                    'start_idx': p['idx'],
                    'similarity': p['sim'],
                    'target_length': target_length
                },
                pattern_data=convert_numpy_types(full_ohlc_data[p['idx']:p['idx']+p['len']]),
                time_info=full_time_data.iloc[p['idx']+p['len']-1],
                forecast_data=convert_numpy_types(full_ohlc_data[p['idx']+p['len']:p['idx']+p['len']+math.ceil(p['len']/3)]) if p['idx']+p['len']+math.ceil(p['len']/3) <= len(full_ohlc_data) else None
            ) for p in candidate_patterns[:top_k]]
        }
        
        return {
            'status': 'success',
            'patterns': [pattern_to_frontend_format(
                pattern_info={
                    'start_idx': p[0],
                    'similarity': p[1],
                    'target_length': target_length
                },
                pattern_data=p[2],
                time_info=p[3],
                forecast_data=p[4] if len(p) > 4 else None
            ) for p in final_patterns]
        }
    except Exception as e:
        raise e

@app.route('/api/similar-patterns', methods=['POST'])
def find_similar_patterns_api():
    """유사 패턴 검색"""
    if model is None or embedding_data is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        
        # 파라미터 추출
        query_time = data.get('query_time')
        query_length = data.get('query_length', 5)
        top_k = data.get('top_k', 3)
        target_length = data.get('target_length')
        
        if not query_time:
            # 현재 패턴 사용
            current_pattern = get_current_pattern()
            if current_pattern is None:
                return jsonify({'error': 'Failed to get current pattern'}), 500
            
            query_normalized = np.array(current_pattern['normalized'])
        else:
            # 특정 시간의 패턴 사용
            query_start_idx = full_time_data[full_time_data >= pd.to_datetime(query_time)].index[0]
            query_raw = full_ohlc_data[query_start_idx:query_start_idx + query_length]
            query_normalized = normalize_window(query_raw)
        
        # 캐시된 유사 패턴 검색 사용
        query_data_str = ','.join(map(str, query_normalized.flatten()))
        result = _find_similar_patterns_cached(query_data_str, query_length, top_k, target_length)
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify({'error': 'Pattern search failed'}), 500
        
        # numpy 타입을 Python 타입으로 변환
        similarities = convert_numpy_types(similarities)
        
        print(f"Converted similarities in similar patterns: {similarities}")
        
        # 프론트엔드 형식으로 변환
        result_patterns = []
        for sim_info in similarities:
            pattern_start_idx = sim_info['idx']
            pattern_length = sim_info['len']
            forecast_length = math.ceil(pattern_length / 3)  # 패턴 길이의 1/3만큼 미래 예측
            
            # 패턴 데이터 (numpy 배열을 Python list로 변환)
            pattern_data = full_ohlc_data[pattern_start_idx:pattern_start_idx + pattern_length]
            pattern_data = convert_numpy_types(pattern_data)
            # 패턴의 마지막 캔들 완료 시점 (CSV에서 직접 가져오기)
            pattern_completion_time = full_time_data.iloc[pattern_start_idx + pattern_length - 1]
            
            # 미래 데이터 (예측 구간)
            forecast_start_idx = pattern_start_idx + pattern_length
            forecast_end_idx = forecast_start_idx + forecast_length
            forecast_data = None
            
            if forecast_end_idx <= len(full_ohlc_data):
                forecast_data = full_ohlc_data[forecast_start_idx:forecast_end_idx]
                forecast_data = convert_numpy_types(forecast_data)
            
            formatted_pattern = pattern_to_frontend_format(
                sim_info, pattern_data, pattern_completion_time, forecast_data
            )
            if formatted_pattern:
                result_patterns.append(formatted_pattern)
        
        result = {
            'query_pattern': current_pattern if not query_time else {
                'timestamp': full_time_data.iloc[query_start_idx + query_length - 1].strftime("%Y.%m.%d %H:%M"),  # 마지막 캔들의 완료 시점
                'length': query_length,
                'normalized': query_normalized.tolist()
            },
            'similar_patterns': result_patterns
        }
        
        # 최종 안전장치: 모든 numpy 타입 변환
        result = convert_numpy_types(result)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in similar patterns search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical-analysis', methods=['POST'])
@require_auth(min_tier=UserTier.MEMBER.value)
def historical_analysis_api():
    """특정 과거 시점을 기준으로 한 패턴 분석 (회원 이상)
    ---
    tags:
      - Analysis
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
      - name: body
        in: body
        required: true
        schema:
            type: object
            required:
              - historical_point
            properties:
                historical_point:
                    type: string
                    description: Target timestamp (YYYY-MM-DD HH:MM)
                query_length:
                    type: integer
                    default: 5
                top_k:
                    type: integer
                    default: 3
    responses:
      200:
        description: Historical analysis results
        schema:
            type: object
            properties:
                query_pattern:
                    type: object
                similar_patterns:
                    type: array
                    items:
                        type: object
                historical_data_info:
                    type: object
    """
    if model is None or embedding_data is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        data = request.get_json()
        historical_point = data.get('historical_point')
        query_length = data.get('query_length', 5)
        target_length = data.get('target_length')
        top_k = data.get('top_k', 3)
        
        if not historical_point:
            return jsonify({'error': 'historical_point is required'}), 400
        
        print(f"\n=== Historical Analysis Request ===")
        print(f"  Historical point: {historical_point}")
        print(f"  Query length: {query_length}")
        print(f"  Target length: {target_length}")
        print(f"  Top K: {top_k}")
        
        
        # 1) 입력값을 KST로 파싱
        try:
            target_time_naive = pd.to_datetime(historical_point)
            target_time_kst = KST.localize(target_time_naive) if target_time_naive.tz is None else target_time_naive.astimezone(KST)
            print(f"[HIST] Parsed target time (KST): {target_time_kst}")

            # 2) ✅ 입력 시점 '직전'에 완료된 캔들의 '시작' 시각으로 보정
            last_completed_start_kst = find_last_completed_candle_start_time_before_point(target_time_kst)
            print(f"[HIST] Adjusted to last completed candle START before target (KST): {last_completed_start_kst.strftime('%Y-%m-%d %H:%M:%S')}")

            # 3) 내부 처리용 naive datetime 변환
            target_time = last_completed_start_kst.replace(tzinfo=None)

        except Exception as e:
            print(f"[HIST] ERROR: Failed to parse datetime: {e}")
            return jsonify({'error': 'Invalid datetime format. Use YYYY-MM-DD HH:MM'}), 400
        
        # 2. 캐시 확인 먼저 수행 (바이낸스 API 호출 방지)
        # Cache logic removed in v8

        
        # 3. CSV 데이터 범위 확인
        csv_start_time = full_time_data.iloc[0]
        csv_end_time = full_time_data.iloc[-1]
        print(f"  CSV data range: {csv_start_time} ~ {csv_end_time}")
        
        # 4. 요청 시점이 CSV 범위 내에 있는지 확인
        use_binance_api = False
        if target_time < csv_start_time or target_time > csv_end_time:
            print(f"  WARNING: Target time {target_time} is outside CSV range!")
            print(f"  CSV range: {csv_start_time} ~ {csv_end_time}")
            print(f"  Will use Binance API to fetch historical data...")
            use_binance_api = True
        
        if use_binance_api:
            # Binance API를 사용해서 해당 시점 데이터 가져오기
            print(f"  Fetching data from Binance API...")
            if binance_exchange is None:
                return jsonify({'error': 'Binance API not initialized'}), 500
            
            try:
                # ✅ KST 시점을 UTC 타임스탬프로 변환 (시작 시각 기준)
                target_utc_timestamp = convert_kst_to_utc_timestamp(last_completed_start_kst)
                print(f"[HIST] Target UTC timestamp (START): {target_utc_timestamp}")
                
                # query_length + 10개 정도 여유분을 두고 가져오기 (4h 기준)
                fetch_limit = query_length + 10
                print(f"[HIST] Fetching {fetch_limit} candles ending at START KST {last_completed_start_kst.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 해당 시점을 마지막으로 하는 캔들들 가져오기
                start_utc_timestamp = target_utc_timestamp - (fetch_limit * 4 * 60 * 60 * 1000)
                
                # 안전한 OHLCV 가져오기 (재시도 로직 포함)
                ohlcvs = safe_fetch_ohlcv(
                    binance_exchange, 
                    'BTC/USDT', 
                    '4h', 
                    limit=fetch_limit,
                    since=start_utc_timestamp
                )
                print(f"  Fetched {len(ohlcvs)} candles from Binance")
                
                if len(ohlcvs) < query_length:
                    return jsonify({'error': f'Not enough Binance data. Got {len(ohlcvs)}, need {query_length}'}), 400
                
                # target_time 이전의 캔들만 필터링
                filtered_ohlcvs = []
                for ohlcv in ohlcvs:
                    candle_time = pd.to_datetime(ohlcv[0], unit='ms')
                    if candle_time <= target_time:
                        filtered_ohlcvs.append(ohlcv)
                
                print(f"  Filtered to {len(filtered_ohlcvs)} candles before target time")
                
                if len(filtered_ohlcvs) < query_length:
                    return jsonify({'error': f'Not enough Binance data before target time. Got {len(filtered_ohlcvs)}, need {query_length}'}), 400
                
                # 마지막 query_length개 캔들 사용
                query_ohlcvs = filtered_ohlcvs[-query_length:]
                query_raw = np.array([[ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4]] for ohlcv in query_ohlcvs], dtype=np.float32)
                query_time = pd.to_datetime(query_ohlcvs[-1][0], unit='ms')
                
                print(f"  Using Binance data - Query time range: {pd.to_datetime(query_ohlcvs[0][0], unit='ms')} ~ {query_time}")
                print(f"  Query data shape: {query_raw.shape}")
                
                data_source = "Binance API"
                available_times = filtered_ohlcvs  # For consistency in return data
                query_start_global_idx = 0  # Not used for Binance data
                query_end_global_idx = query_length - 1  # Not used for Binance data
                
            except Exception as e:
                print(f"  ERROR: Failed to fetch Binance data: {e}")
                return jsonify({'error': f'Failed to fetch Binance data: {str(e)}'}), 500
        
        else:
            # CSV 데이터 사용
            print(f"  Using CSV data...")
            available_times = full_time_data[full_time_data <= target_time]
            print(f"  Available data points before target time: {len(available_times)}")
            print(f"  Latest available time: {available_times.iloc[-1] if len(available_times) > 0 else 'None'}")
            
            if len(available_times) < query_length:
                return jsonify({'error': f'Not enough historical data. Need at least {query_length} candles before {historical_point}'}), 400
            
            # 가장 마지막 시점을 기준으로 query_length만큼 역순으로 가져오기
            end_idx = len(available_times) - 1
            start_idx = end_idx - query_length + 1
            
            print(f"  Index calculation: start_idx={start_idx}, end_idx={end_idx}")
            
            if start_idx < 0:
                print(f"  ERROR: start_idx < 0. Available: {len(available_times)}, Required: {query_length}")
                return jsonify({'error': f'Not enough data. Available: {len(available_times)}, Required: {query_length}'}), 400
            
            # 쿼리 패턴 데이터 추출
            query_start_global_idx = available_times.index[start_idx]
            query_end_global_idx = available_times.index[end_idx]
            
            print(f"  Global indices: start={query_start_global_idx}, end={query_end_global_idx}")
            
            query_raw = full_ohlc_data[query_start_global_idx:query_end_global_idx + 1]
            query_time = full_time_data.iloc[query_end_global_idx]  # 마지막 시점을 표시
            
            print(f"  Query data time range: {full_time_data.iloc[query_start_global_idx]} ~ {full_time_data.iloc[query_end_global_idx]}")
            print(f"  Query data shape: {query_raw.shape}")
            
            data_source = "Historical CSV Data"
        
        # 공통 로그 출력
        print(f"  Query raw data (first 3 candles):")
        for i in range(min(3, len(query_raw))):
            print(f"    Candle #{i+1}: O={query_raw[i,0]:.2f} H={query_raw[i,1]:.2f} L={query_raw[i,2]:.2f} C={query_raw[i,3]:.2f}")
        
        print(f"  Actual query timestamp: {query_time}")
        print(f"  Data source: {data_source}")
        
        # 7. 정규화
        query_normalized = normalize_window(query_raw)
        print(f"  Query normalized data (first 3 candles):")
        for i in range(min(3, len(query_normalized))):
            print(f"    Norm #{i+1}: O={query_normalized[i,0]:.4f} H={query_normalized[i,1]:.4f} L={query_normalized[i,2]:.4f} C={query_normalized[i,3]:.4f}")
        
        # 8. 2단계 패턴 검색 (캐시 포함)
        print(f"  Starting 2-stage pattern search with top_k={top_k}, target_length={target_length}")
        similarities = perform_historical_pattern_search(
            historical_point, query_normalized, query_length, top_k, target_length
        )
        
        # numpy 타입을 Python 타입으로 변환
        similarities = convert_numpy_types(similarities)
        
        print(f"  Found {len(similarities)} similar patterns:")
        for i, sim in enumerate(similarities):
            print(f"    Pattern #{i+1}: idx={sim['idx']}, similarity={sim['sim']:.4f}, length={sim['len']}")
            pattern_start_timestamp = full_time_data.iloc[sim['idx']]
            print(f"      Timestamp: {pattern_start_timestamp}")
        
        # 7. 프론트엔드 형식으로 변환
        result_patterns = []
        print(f"  Converting {len(similarities)} patterns to frontend format...")
        for i, sim_info in enumerate(similarities):
            print(f"    Processing pattern #{i+1}: idx={sim_info['idx']}, similarity={sim_info['sim']:.4f}")
            pattern_start_idx = sim_info['idx']
            pattern_length = sim_info['len']
            forecast_length = math.ceil(pattern_length / 3)
            
            # 검색된 패턴이 현재 쿼리 시점과 겹치지 않는지 확인 (CSV 데이터만)
            if not use_binance_api:
                pattern_end_idx = pattern_start_idx + pattern_length
                if pattern_end_idx > query_start_global_idx:
                    print(f"      Skipping overlapping pattern: idx={pattern_start_idx}, end_idx={pattern_end_idx} > query_start={query_start_global_idx}")
                    continue  # 겹치는 패턴은 제외
            
            # 패턴 데이터 (numpy 배열을 Python list로 변환)
            pattern_data = full_ohlc_data[pattern_start_idx:pattern_start_idx + pattern_length]
            pattern_data = convert_numpy_types(pattern_data)
            # 패턴의 마지막 캔들 완료 시점 (CSV에서 직접 가져오기)
            pattern_completion_time = full_time_data.iloc[pattern_start_idx + pattern_length - 1]
            
            # 미래 데이터 (예측 구간)
            forecast_start_idx = pattern_start_idx + pattern_length
            forecast_end_idx = forecast_start_idx + forecast_length
            forecast_data = None
            
            if forecast_end_idx <= len(full_ohlc_data):
                forecast_data = full_ohlc_data[forecast_start_idx:forecast_end_idx]
                forecast_data = convert_numpy_types(forecast_data)
            
            formatted_pattern = pattern_to_frontend_format(
                sim_info, pattern_data, pattern_completion_time, forecast_data
            )
            if formatted_pattern:
                result_patterns.append(formatted_pattern)
                print(f"      ✓ Pattern #{i+1} successfully added to results")
            else:
                print(f"      ✗ Pattern #{i+1} failed to format, skipping")
        
        print(f"  Final result: {len(result_patterns)} patterns ready for frontend")
        
        # 8. 현재 쿼리 패턴 정보 생성
        candles = []
        for ohlc in query_raw:
            candles.append({
                'open': float(ohlc[0]),
                'high': float(ohlc[1]),
                'low': float(ohlc[2]),
                'close': float(ohlc[3]),
                'volume': 1000000  # 임시 볼륨
            })
        
        # 쿼리 패턴 시간도 한국시간으로 변환
        kst = pytz.timezone('Asia/Seoul')
        if hasattr(query_time, 'tz_localize'):
            if query_time.tz is None:
                query_time_kst = query_time.tz_localize('UTC').tz_convert(kst)
            else:
                query_time_kst = query_time.tz_convert(kst)
        else:
            if query_time.tzinfo is None:
                query_time_kst = pytz.UTC.localize(query_time).astimezone(kst)
            else:
                query_time_kst = query_time.astimezone(kst)
                
        if hasattr(target_time, 'tz_localize'):
            if target_time.tz is None:
                target_time_kst = target_time.tz_localize('UTC').tz_convert(kst)
            else:
                target_time_kst = target_time.tz_convert(kst)
        else:
            if target_time.tzinfo is None:
                target_time_kst = pytz.UTC.localize(target_time).astimezone(kst)
            else:
                target_time_kst = target_time.astimezone(kst)

        query_pattern = {
            'timestamp': query_time_kst.strftime("%Y.%m.%d %H:%M"),
            'symbol': f'BTC/USDT (Historical - 요청: {target_time_kst.strftime("%Y.%m.%d %H:%M")})',
            'confidence': 95,  # 과거 데이터는 확실함
            'candles': candles,
            'source': 'historical_data'
        }
        
        result = {
            'query_pattern': query_pattern,
            'similar_patterns': result_patterns,
            'historical_data_info': {
                'requested_point': historical_point,
                'data_source': data_source,
                'actual_query_start': query_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(query_time, 'strftime') else str(query_time),
                'actual_query_end': query_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(query_time, 'strftime') else str(query_time),
                'query_length': query_length,
                'available_data_points': len(available_times) if not use_binance_api else len(query_raw),
                'found_patterns': len(result_patterns)
            }
        }
        
        # 최종 안전장치: 모든 numpy 타입 변환
        result = convert_numpy_types(result)
        
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in historical analysis: {e}")
        return jsonify({'error': str(e)}), 500

# 인증 관련 API들
@app.route('/api/auth/send-verification', methods=['POST'])
def send_verification_code():
    """이메일 인증 코드 발송"""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email required'}), 400
            
        # 이메일 형식 검증
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # 기존 사용자 확인
        existing_user = db_manager.get_user_by_email(email)
        if existing_user:
            return jsonify({'error': 'Email already exists'}), 409
        
        # 재발송 제한 확인 (30초 쿨다운) 및 만료된 코드 정리
        if email in verification_codes:
            stored_data = verification_codes[email]
            
            # 만료된 코드 정리
            if datetime.now() > stored_data.get('expires', datetime.min):
                del verification_codes[email]
            else:
                # 30초 쿨다운 확인
                last_sent = stored_data.get('last_sent', datetime.min)
                if datetime.now() - last_sent < timedelta(seconds=30):
                    return jsonify({'error': 'Please wait 30 seconds before requesting another code'}), 429
        
        # 인증 코드 생성 및 저장
        code = generate_verification_code()
        verification_codes[email] = {
            'code': code,
            'expires': datetime.now() + timedelta(minutes=10),
            'last_sent': datetime.now()
        }
        
        # 이메일 발송
        if send_verification_email(email, code):
            return jsonify({
                'status': 'success',
                'message': 'Verification code sent successfully'
            })
        else:
            return jsonify({'error': 'Failed to send verification code'}), 500
            
    except Exception as e:
        print(f"Error sending verification code: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/verify-code', methods=['POST'])
def verify_code():
    """인증 코드 검증"""
    try:
        data = request.get_json()
        email = data.get('email')
        code = data.get('code')
        
        if not email or not code:
            return jsonify({'error': 'Email and code required'}), 400
        
        # 저장된 인증 코드 확인
        if email not in verification_codes:
            return jsonify({'error': 'No verification code found'}), 404
        
        stored_data = verification_codes[email]
        
        # 만료 시간 확인
        if datetime.now() > stored_data['expires']:
            del verification_codes[email]
            return jsonify({'error': 'Verification code expired'}), 410
        
        # 코드 일치 확인
        if stored_data['code'] != code:
            return jsonify({'error': 'Invalid verification code'}), 400
        
        return jsonify({
            'status': 'success',
            'message': 'Code verified successfully'
        })
        
    except Exception as e:
        print(f"Error verifying code: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/register', methods=['POST'])
def register_api():
    """회원 가입 (인증 코드 검증 후)"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        verification_code = data.get('verification_code')
        
        if not email or not password or not verification_code:
            return jsonify({'error': 'Email, password and verification code required'}), 400
        
        # 인증 코드 재검증
        if email not in verification_codes:
            return jsonify({'error': 'No verification code found. Please request a new code.'}), 404
        
        stored_data = verification_codes[email]
        
        # 만료 시간 확인
        if datetime.now() > stored_data['expires']:
            del verification_codes[email]
            return jsonify({'error': 'Verification code expired. Please request a new code.'}), 410
        
        # 코드 일치 확인
        if stored_data['code'] != verification_code:
            return jsonify({'error': 'Invalid verification code'}), 400
        
        # 인증 성공 후 사용자 등록
        user_uuid = db_manager.register_user(email, password)
        
        # 인증 코드 정리
        del verification_codes[email]
        if user_uuid:
            # JWT 토큰 생성
            payload = {
                'user_uuid': user_uuid,
                'tier': UserTier.MEMBER.value,
                'exp': datetime.now(timezone.utc) + timedelta(days=1)
            }
            token = jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')
            
            return jsonify({
                'status': 'success',
                'message': 'User registered successfully',
                'token': token
            })
        else:
            return jsonify({'error': 'Email already exists'}), 409
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def login_api():
    """
    User Login
    ---
    tags:
      - Authentication
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - email
            - password
          properties:
            email:
              type: string
              example: user@example.com
            password:
              type: string
              example: password123
    responses:
      200:
        description: Login successful
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            token:
              type: string
              description: JWT Token
            user_uuid:
              type: string
            tier:
              type: string
            email:
              type: string
            limits:
              type: object
      401:
        description: Invalid credentials
      500:
        description: Internal server error
    """
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        user_uuid, tier = db_manager.login_user(email, password)
        if user_uuid:
            # JWT 토큰 생성
            payload = {
                'user_uuid': user_uuid,
                'tier': tier,
                'exp': datetime.now(timezone.utc) + timedelta(days=1)
            }
            token = jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')
            
            # 사용자 정보도 함께 조회
            user_info = db_manager.get_user_info(user_uuid)
            
            return jsonify({
                'status': 'success',
                'message': 'Login successful',
                'token': token,
                'user_uuid': user_uuid,
                'email': email,
                'tier': tier,
                'limits': get_tier_limits(tier)
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/upgrade', methods=['POST'])
@require_auth(min_tier=UserTier.MEMBER.value)
def upgrade_premium_api():
    """프리미엄 업그레이드"""
    try:
        user_uuid = request.user_info['uuid']
        
        # 실제로는 결제 시스템과 연동
        success = db_manager.upgrade_to_premium(user_uuid, days=30)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Upgraded to premium successfully',
                'new_limits': get_tier_limits(UserTier.PREMIUM.value)
            })
        else:
            return jsonify({'error': 'Upgrade failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/info', methods=['GET'])
@require_auth(min_tier=UserTier.GUEST.value)
def user_info_api():
    """사용자 정보 조회"""
    try:
        user_info = request.user_info
        user_uuid = user_info['uuid']
        
        # 일일 사용량 조회
        daily_usage = {
            'live_analysis': db_manager.get_daily_usage_count(user_uuid, '/api/live-analysis'),
            'historical_analysis': db_manager.get_daily_usage_count(user_uuid, '/api/historical-analysis'),
            'total': db_manager.get_daily_usage_count(user_uuid)
        }
        
        return jsonify({
            'user_info': user_info,
            'daily_usage': daily_usage,
            'limits': get_tier_limits(user_info['tier'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
@require_auth(min_tier=UserTier.PREMIUM.value)
def train_model_api():
    """모델 재학습 API (프리미엄 전용)"""
    try:
        global model, embedding_data
        
        print("Starting model training...")
        test_ohlc_data = full_ohlc_data[int(len(full_ohlc_data) * 0.9):]
        
        model = train_or_load_model(
            full_ohlc_data, test_ohlc_data, 
            CONFIG['emb_dim'], CONFIG['model_path'], 
            force_train=True, max_len=CONFIG['max_pattern_len']
        )
        
        print("Recomputing embeddings...")
        embedding_data = precompute_and_save_embeddings(
            full_ohlc_data, model, CONFIG['emb_path']
        )
        
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ======================
# Admin API Endpoints
# ======================

@app.route('/api/admin/users', methods=['GET'])
@require_auth(min_tier='admin')
def admin_list_users():
    """관리자: 모든 사용자 목록 조회"""
    try:
        users = db_manager.get_all_users()
        return jsonify({
            'status': 'success',
            'users': users,
            'total_count': len(users)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/user/<user_uuid>/info', methods=['GET'])
@require_auth(min_tier='admin')
def admin_get_user_info(user_uuid):
    """관리자: 특정 사용자 정보 조회"""
    try:
        user_info = db_manager.get_user_info(user_uuid)
        if not user_info:
            return jsonify({'error': 'User not found'}), 404
        
        # 사용량 통계 추가
        usage_stats = db_manager.get_user_usage_stats(user_uuid)
        user_info['usage_stats'] = usage_stats
        
        return jsonify({
            'status': 'success',
            'user': user_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/user/<user_uuid>/password', methods=['PUT'])
@require_auth(min_tier='admin')
def admin_change_password(user_uuid):
    """관리자: 사용자 비밀번호 변경"""
    try:
        data = request.get_json()
        new_password = data.get('new_password')
        
        if not new_password:
            return jsonify({'error': 'New password is required'}), 400
        
        if len(new_password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        success = db_manager.change_user_password(user_uuid, new_password)
        if not success:
            return jsonify({'error': 'Failed to change password or user not found'}), 404
        
        return jsonify({
            'status': 'success',
            'message': 'Password changed successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/user/<user_uuid>/tier', methods=['PUT'])
@require_auth(min_tier='admin')
def admin_change_tier(user_uuid):
    """관리자: 사용자 등급 변경"""
    try:
        data = request.get_json()
        new_tier = data.get('tier')
        premium_days = data.get('premium_days', 30)  # 프리미엄인 경우 기본 30일
        
        if new_tier not in [UserTier.GUEST.value, UserTier.MEMBER.value, UserTier.PREMIUM.value]:
            return jsonify({'error': 'Invalid tier. Must be guest, member, or premium'}), 400
        
        # 프리미엄으로 업그레이드하는 경우 만료일 설정
        premium_until = None
        if new_tier == UserTier.PREMIUM.value:
            premium_until = datetime.now() + timedelta(days=premium_days)
        
        success = db_manager.change_user_tier(user_uuid, new_tier, premium_until)
        if not success:
            return jsonify({'error': 'Failed to change tier or user not found'}), 404
        
        return jsonify({
            'status': 'success',
            'message': f'User tier changed to {new_tier}',
            'tier': new_tier,
            'premium_until': premium_until.isoformat() if premium_until else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/user/<user_uuid>/active', methods=['PUT'])
@require_auth(min_tier='admin')
def admin_toggle_user_active(user_uuid):
    """관리자: 사용자 활성화/비활성화"""
    try:
        data = request.get_json()
        is_active = data.get('is_active', True)
        
        success = db_manager.set_user_active_status(user_uuid, is_active)
        if not success:
            return jsonify({'error': 'Failed to update user status or user not found'}), 404
        
        return jsonify({
            'status': 'success',
            'message': f'User {"activated" if is_active else "deactivated"} successfully',
            'is_active': is_active
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
@require_auth(min_tier='admin')
def admin_get_stats():
    """관리자: 전체 시스템 통계"""
    try:
        stats = db_manager.get_system_stats()
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/user/<user_uuid>', methods=['DELETE'])
@require_auth(min_tier='admin')
def admin_delete_user(user_uuid):
    """관리자: 사용자 삭제"""
    try:
        # 자신을 삭제하는 것을 방지
        current_user_uuid = request.user_info['uuid']
        if user_uuid == current_user_uuid:
            return jsonify({'error': 'Cannot delete yourself'}), 400
        
        # 사용자 존재 확인
        user_info = db_manager.get_user_info(user_uuid)
        if not user_info:
            return jsonify({'error': 'User not found'}), 404
        
        # 다른 관리자 삭제 방지 (선택사항)
        if user_info['tier'] == 'admin':
            return jsonify({'error': 'Cannot delete admin users'}), 400
        
        # 데이터베이스에서 사용자 및 관련 데이터 삭제
        success = db_manager.delete_user(user_uuid)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'User {user_info.get("email", user_uuid)} deleted successfully'
            })
        else:
            return jsonify({'error': 'Failed to delete user'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/force-auto-analysis', methods=['POST'])
@require_auth(min_tier='admin')
def admin_force_auto_analysis():
    """관리자: 자동 분석 강제 실행"""
    try:
        if auto_scheduler:
            auto_scheduler.force_run_now()
            return jsonify({
                'status': 'success',
                'message': 'Auto analysis manually triggered'
            })
        else:
            return jsonify({'error': 'Auto scheduler not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/system/cleanup', methods=['POST'])
@require_auth(min_tier='admin')
def admin_system_cleanup():
    """관리자: 시스템 정리 (오래된 게스트 세션, 로그 등)"""
    try:
        # 30일 이전 게스트 세션 정리
        cleanup_count = db_manager.cleanup_old_guest_sessions(days=30)
        
        return jsonify({
            'status': 'success',
            'message': f'Cleaned up {cleanup_count} old guest sessions',
            'cleanup_count': cleanup_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/bulk-action', methods=['POST'])
@require_auth(min_tier='admin')
def admin_bulk_user_action():
    """관리자: 사용자 일괄 작업"""
    try:
        data = request.get_json()
        action = data.get('action')
        user_uuids = data.get('user_uuids', [])
        
        if not action or not user_uuids:
            return jsonify({'error': 'Action and user_uuids are required'}), 400
        
        current_user_uuid = request.user_info['uuid']
        results = []
        
        for user_uuid in user_uuids:
            # 자신에게는 작업하지 않음
            if user_uuid == current_user_uuid:
                results.append({'uuid': user_uuid, 'status': 'skipped', 'reason': 'Cannot modify self'})
                continue
            
            try:
                if action == 'activate':
                    db_manager.set_user_active(user_uuid, True)
                    results.append({'uuid': user_uuid, 'status': 'success', 'action': 'activated'})
                elif action == 'deactivate':
                    db_manager.set_user_active(user_uuid, False)
                    results.append({'uuid': user_uuid, 'status': 'success', 'action': 'deactivated'})
                elif action == 'delete':
                    # 관리자는 삭제하지 않음
                    user_info = db_manager.get_user_info(user_uuid)
                    if user_info and user_info['tier'] == 'admin':
                        results.append({'uuid': user_uuid, 'status': 'skipped', 'reason': 'Cannot delete admin'})
                    else:
                        db_manager.delete_user(user_uuid)
                        results.append({'uuid': user_uuid, 'status': 'success', 'action': 'deleted'})
                else:
                    results.append({'uuid': user_uuid, 'status': 'error', 'reason': 'Unknown action'})
            except Exception as e:
                results.append({'uuid': user_uuid, 'status': 'error', 'reason': str(e)})
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def initialize_system():
    """시스템 전체 초기화"""
    try:
        print("🔧 Initializing system components...")
        
        # 1. 바이낸스 초기화
        if not initialize_binance():
            print("❌ Failed to initialize Binance")
            return False
            
        # 2. 데이터베이스 초기화 (이미 __init__에서 호출됨)
        print("✅ Database initialized")
        
        # 3. AI 모델 및 임베딩 데이터 초기화
        print("🤖 Initializing AI model and embeddings...")
        if not initialize_ai_system():
            print("❌ Failed to initialize AI system")
            return False
        print("✅ AI system initialized successfully")
        
        # 4. 자동 분석 스케줄러 초기화 (일단 None으로 패스, 나중에 수정 가능)
        global auto_scheduler
        auto_scheduler = None  # AutoAnalysisScheduler(pattern_analysis_func) 
        print("✅ Auto analysis scheduler skipped (can be enabled later)")
        
        print("✅ All system components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False


# Gunicorn 호환성을 위해 앱 시작 시 자동 초기화
print("Starting Pattern Detection API Server...")
if not initialize_system():
    print("Failed to initialize system. Please check the configuration.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Server starting on http://localhost:{port}")
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=False)