#!/usr/bin/env python3
"""
Authentication & Authorization Middleware
사용자 인증 및 권한 검증 미들웨어
"""

from functools import wraps
from flask import request, jsonify, current_app
from database import db_manager, UserTier
from datetime import datetime
import jwt

class PermissionDeniedError(Exception):
    pass

class UsageLimitExceededError(Exception):
    pass

def get_user_from_request():
    """요청에서 사용자 정보 추출"""
    user_uuid = None
    
    # Authorization Bearer 토큰에서 JWT 추출 (우선)
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, current_app.config.get('JWT_SECRET_KEY', 'temp-dev-key-CHANGE-IN-PRODUCTION'), algorithms=['HS256'])
            user_uuid = payload['user_uuid']
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
            # 토큰이 유효하지 않은 경우, 게스트로 처리
            print(f"JWT validation failed: {e}")
    
    # 레거시 인증 우회 제거 - 보안상 위험

    # UUID가 있으면 사용자 정보 조회
    if user_uuid:
        user_info = db_manager.get_user_info(user_uuid)
        if user_info and user_info['is_active']:
            return user_info

    # UUID가 없거나 유효하지 않으면 게스트로 처리
    ip_address = request.remote_addr
    guest_uuid = db_manager.create_guest_session(ip_address)
    print(f"📝 Guest session created: {guest_uuid}")
    return db_manager.get_user_info(guest_uuid)

def check_tier_permission(required_tier, user_tier):
    """권한 레벨 확인"""
    # Admin은 모든 권한 허용
    if user_tier == UserTier.ADMIN.value:
        return True
        
    tier_levels = {
        UserTier.GUEST.value: 0,
        UserTier.MEMBER.value: 1,
        UserTier.PREMIUM.value: 2,
        UserTier.ADMIN.value: 3
    }
    
    return tier_levels.get(user_tier, 0) >= tier_levels.get(required_tier, 0)

def validate_search_params(user_tier, query_length, target_length, top_k):
    """검색 파라미터 권한 검증"""
    # Admin은 모든 제한 없음
    if user_tier == UserTier.ADMIN.value:
        return
        
    if user_tier == UserTier.GUEST.value:
        # 게스트: 3캔들-3캔들, top3만
        if query_length != 3 or (target_length and target_length != 3) or top_k > 3:
            raise PermissionDeniedError("Guest users can only use 3-candle patterns with top 3 results")
    
    elif user_tier == UserTier.MEMBER.value:
        # 회원: 3~100캔들, top10까지 (프리미엄 기능을 일반 회원으로 이전)
        if query_length < 3 or query_length > 100 or (target_length and (target_length < 3 or target_length > 100)) or top_k > 10:
            raise PermissionDeniedError("Member users can use 3-100 candle patterns with top 10 results")
    
    elif user_tier == UserTier.PREMIUM.value:
        # 유료회원: 3~100캔들, top10까지
        if query_length < 3 or query_length > 100 or (target_length and (target_length < 3 or target_length > 100)) or top_k > 10:
            raise PermissionDeniedError("Premium users can use 3-100 candle patterns with top 10 results")

def check_daily_usage_limit(user_uuid, user_tier, endpoint):
    """일일 사용량 제한 확인"""
    daily_count = db_manager.get_daily_usage_count(user_uuid, endpoint)
    
    if user_tier == UserTier.GUEST.value:
        # 게스트: live-analysis는 허용, historical-analysis는 제한
        if endpoint in ['/api/historical-analysis']:
            raise PermissionDeniedError("Guest users can only perform live analysis")
    
    elif user_tier == UserTier.MEMBER.value:
        # 회원: 과거 검색 일일 1000건 (프리미엄 기능 이전)
        if daily_count >= 1000:
            raise UsageLimitExceededError("Daily usage limit exceeded (1000 requests)")
    
    elif user_tier == UserTier.PREMIUM.value:
        # 유료회원: 일일 1000건
        if daily_count >= 1000:
            raise UsageLimitExceededError("Daily usage limit exceeded (1000 requests)")
    
    elif user_tier == UserTier.ADMIN.value:
        # 관리자: 제한 없음
        pass

def require_auth(min_tier=UserTier.GUEST.value):
    """인증 및 권한 검증 데코레이터"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # 1. 사용자 정보 추출
                user_info = get_user_from_request()
                user_uuid = user_info['uuid']
                user_tier = user_info['tier']
                
                # 2. 권한 레벨 확인
                if not check_tier_permission(min_tier, user_tier):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                # 3. 일일 사용량 확인
                endpoint = request.endpoint or request.path
                check_daily_usage_limit(user_uuid, user_tier, endpoint)
                
                # 4. 검색 파라미터 검증 (해당 API만)
                if endpoint in ['/api/live-analysis', '/api/historical-analysis']:
                    data = request.get_json() or {}
                    query_length = data.get('query_length', 3)
                    target_length = data.get('target_length')
                    top_k = data.get('top_k', 3)
                    
                    validate_search_params(user_tier, query_length, target_length, top_k)
                
                # 5. 사용량 로깅
                db_manager.log_usage(user_uuid, endpoint, data if 'data' in locals() else None, request.remote_addr)
                
                # 6. 사용자 정보를 request 객체에 추가
                request.user_info = user_info
                
                return f(*args, **kwargs)
                
            except PermissionDeniedError as e:
                return jsonify({'error': str(e), 'code': 'PERMISSION_DENIED'}), 403
            except UsageLimitExceededError as e:
                return jsonify({'error': str(e), 'code': 'USAGE_LIMIT_EXCEEDED'}), 429
            except Exception as e:
                print(f"Auth middleware error: {e}")
                return jsonify({'error': 'Authentication failed'}), 401
        
        return decorated_function
    return decorator

def get_tier_limits(tier):
    """권한별 제한 정보 반환"""
    limits = {
        UserTier.GUEST.value: {
            'query_lengths': [3],
            'target_lengths': [3],
            'max_top_k': 3,
            'daily_searches': 0,  # 자동 분석만
            'historical_search': False,
            'features': ['auto_analysis_view']
        },
        UserTier.MEMBER.value: {
            'query_lengths': list(range(3, 101)),  # 3-100 캔들 (프리미엄 기능 이전)
            'target_lengths': list(range(3, 101)), 
            'max_top_k': 10,  # Top 10까지
            'daily_searches': 1000,  # 일일 1000건
            'historical_search': True,
            'daily_historical': 1000,
            'features': ['live_analysis', 'historical_analysis', 'auto_analysis_view', 'custom_analysis']  # 커스텀 분석 추가
        },
        UserTier.PREMIUM.value: {
            'query_lengths': list(range(3, 101)),
            'target_lengths': list(range(3, 101)),
            'max_top_k': 10,
            'daily_searches': 1000,
            'historical_search': True,
            'daily_historical': 1000,
            'features': ['live_analysis', 'historical_analysis', 'auto_analysis_view', 'api_access']
        },
        UserTier.ADMIN.value: {
            'query_lengths': list(range(3, 101)),  # 동일한 범위지만 실제로는 무제한
            'target_lengths': list(range(3, 101)),  # 동일한 범위지만 실제로는 무제한
            'max_top_k': 999999,  # 무제한
            'daily_searches': 999999,  # 무제한
            'historical_search': True,
            'daily_historical': 999999,  # 무제한
            'features': ['live_analysis', 'historical_analysis', 'auto_analysis_view', 'api_access', 'admin_panel']
        }
    }
    
    return limits.get(tier, limits[UserTier.GUEST.value])