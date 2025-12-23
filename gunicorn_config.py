# Gunicorn 설정 파일
# Pattern Finder API 운영 환경 설정

import multiprocessing
import os

# 서버 소켓
bind = "0.0.0.0:5000"
backlog = 2048

# 워커 프로세스
# 워커 프로세스
workers = 1  # GPU 메모리 부족 방지를 위해 1개로 제한 (모델 용량이 큼)
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 5

# 요청 제한
max_requests = 1000
max_requests_jitter = 100

# 로깅
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 성능 최적화
preload_app = False  # PyTorch CUDA와 호환성을 위해 False로 설정 (fork 문제 해결)
worker_tmp_dir = "/dev/shm"  # RAM 디스크 사용 (있는 경우)

# 프로세스 이름
proc_name = "pattern_finder_api"

# 보안
user = None  # 실행 사용자 (필요시 설정)
group = None  # 실행 그룹 (필요시 설정)