#!/usr/bin/env python3
"""
User Database Management
사용자 정보 및 권한 관리 데이터베이스
"""

import sqlite3
import hashlib
import uuid
from datetime import datetime, timedelta
from enum import Enum
import json
import os

class UserTier(Enum):
    GUEST = "guest"
    MEMBER = "member" 
    PREMIUM = "premium"
    ADMIN = "admin"

class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # 절대 경로로 데이터베이스 파일 경로 설정
            default_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
            db_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path}")
            self.db_path = db_url.replace("sqlite:///", "")
        else:
            self.db_path = db_path
        
        # 데이터베이스 디렉토리가 존재하는지 확인하고 생성
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        self.init_database()
    
    def init_database(self):
        """데이터베이스 테이블 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT,
                tier TEXT NOT NULL DEFAULT 'guest',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                premium_until TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # 사용량 추적 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_uuid TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                request_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_uuid) REFERENCES users (uuid)
            )
        ''')
        
        # 자동 분석 결과 캐시 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auto_analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_time TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                query_length INTEGER NOT NULL,
                target_length INTEGER,
                top_k INTEGER NOT NULL,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 기존 테이블에 컬럼 추가 (존재하지 않는 경우)
        try:
            cursor.execute('ALTER TABLE auto_analysis_cache ADD COLUMN target_length INTEGER')
        except:
            pass
        try:
            cursor.execute('ALTER TABLE auto_analysis_cache ADD COLUMN top_k INTEGER NOT NULL DEFAULT 3')
        except:
            pass
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def create_guest_session(self, ip_address=None):
        """게스트 세션 생성"""
        guest_uuid = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (uuid, tier, last_login)
            VALUES (?, ?, ?)
        ''', (guest_uuid, UserTier.GUEST.value, datetime.now()))
        
        conn.commit()
        conn.close()
        
        print(f"📝 Guest session created: {guest_uuid}")
        return guest_uuid
    
    def register_user(self, email, password):
        """회원 가입"""
        import bcrypt
        user_uuid = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (uuid, email, password_hash, tier)
                VALUES (?, ?, ?, ?)
            ''', (user_uuid, email, password_hash, UserTier.MEMBER.value))
            
            conn.commit()
            print(f"✅ User registered: {email}")
            return user_uuid
            
        except sqlite3.IntegrityError:
            print(f"❌ Email already exists: {email}")
            return None
        finally:
            conn.close()
    
    def login_user(self, email, password):
        """로그인"""
        import bcrypt
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 이메일로 사용자 조회
        cursor.execute('''
            SELECT uuid, tier, premium_until, password_hash FROM users 
            WHERE email = ? AND is_active = 1
        ''', (email,))
        
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return None, None
        
        user_uuid, tier, premium_until, stored_hash = result
        
        # bcrypt로 비밀번호 확인
        if not bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            conn.close()
            print(f"❌ Login failed: {email} (wrong password)")
            return None, None
        
        # 프리미엄 만료 확인
        if tier == UserTier.PREMIUM.value and premium_until:
            if datetime.fromisoformat(premium_until) < datetime.now():
                # 프리미엄 만료 -> 일반 회원으로 강등
                tier = UserTier.MEMBER.value
                cursor.execute('UPDATE users SET tier = ? WHERE uuid = ?', (tier, user_uuid))
        
        # 마지막 로그인 시간 업데이트
        cursor.execute('UPDATE users SET last_login = ? WHERE uuid = ?', 
                     (datetime.now(), user_uuid))
        conn.commit()
        conn.close()
        
        print(f"✅ User logged in: {email} ({tier})")
        return user_uuid, tier
    
    def get_user_by_email(self, email):
        """이메일로 사용자 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT uuid, email, tier, created_at, last_login, premium_until, is_active
            FROM users WHERE email = ? AND is_active = 1
        ''', (email,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'uuid': result[0],
                'email': result[1],
                'tier': result[2],
                'created_at': result[3],
                'last_login': result[4],
                'premium_until': result[5],
                'is_active': result[6]
            }
        return None
    
    def get_user_info(self, user_uuid):
        """사용자 정보 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT uuid, email, tier, created_at, last_login, premium_until, is_active
            FROM users WHERE uuid = ? AND is_active = 1
        ''', (user_uuid,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'uuid': result[0],
                'email': result[1],
                'tier': result[2],
                'created_at': result[3],
                'last_login': result[4],
                'premium_until': result[5],
                'is_active': result[6]
            }
        return None
    
    def upgrade_to_premium(self, user_uuid, days=30):
        """프리미엄 업그레이드"""
        premium_until = datetime.now() + timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET tier = ?, premium_until = ? 
            WHERE uuid = ? AND is_active = 1
        ''', (UserTier.PREMIUM.value, premium_until, user_uuid))
        
        conn.commit()
        conn.close()
        
        print(f"✅ User upgraded to premium: {user_uuid} until {premium_until}")
        return True
    
    def log_usage(self, user_uuid, endpoint, request_data=None, ip_address=None):
        """사용량 로깅"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO usage_log (user_uuid, endpoint, request_data, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (user_uuid, endpoint, json.dumps(request_data) if request_data else None, ip_address))
        
        conn.commit()
        conn.close()
    
    def get_daily_usage_count(self, user_uuid, endpoint=None):
        """일일 사용량 조회"""
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if endpoint:
            cursor.execute('''
                SELECT COUNT(*) FROM usage_log 
                WHERE user_uuid = ? AND endpoint = ?
                AND timestamp >= ? AND timestamp < ?
            ''', (user_uuid, endpoint, today, tomorrow))
        else:
            cursor.execute('''
                SELECT COUNT(*) FROM usage_log 
                WHERE user_uuid = ? 
                AND timestamp >= ? AND timestamp < ?
            ''', (user_uuid, today, tomorrow))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 0
    
    def cache_auto_analysis(self, analysis_time, symbol, timeframe, query_length, results, target_length=None, top_k=10):
        """자동 분석 결과 캐시"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO auto_analysis_cache 
            (analysis_time, symbol, timeframe, query_length, target_length, top_k, results)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (analysis_time, symbol, timeframe, query_length, target_length, top_k, json.dumps(results)))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Auto analysis cached: {analysis_time}")
    
    def get_latest_auto_analysis(self, symbol="BTC/USDT", timeframe="4h", query_length=3, target_length=None, top_k=None):
        """최신 자동 분석 결과 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 조건에 맞는 캐시 검색
        where_conditions = ["symbol = ?", "timeframe = ?", "query_length = ?"]
        params = [symbol, timeframe, query_length]
        
        if target_length is not None:
            where_conditions.append("target_length = ?")
            params.append(target_length)
        
        if top_k is not None:
            where_conditions.append("top_k >= ?")  # top_k 이상의 결과가 있으면 사용 가능 (하위 호환성)
            params.append(top_k)
        
        where_clause = " AND ".join(where_conditions)
        
        cursor.execute(f'''
            SELECT results, analysis_time, target_length, top_k FROM auto_analysis_cache
            WHERE {where_clause}
            ORDER BY analysis_time DESC LIMIT 1
        ''', params)
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0]), result[1]
        return None, None

    # ======================
    # Admin Methods
    # ======================
    
    def get_all_users(self):
        """모든 사용자 목록 조회 (관리자용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT uuid, email, tier, created_at, last_login, premium_until, is_active
            FROM users ORDER BY created_at DESC
        ''')
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'uuid': row[0],
                'email': row[1],
                'tier': row[2],
                'created_at': row[3],
                'last_login': row[4],
                'premium_until': row[5],
                'is_active': bool(row[6])
            })
        
        conn.close()
        return users
    
    def get_user_usage_stats(self, user_uuid):
        """사용자 사용량 통계 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 오늘 사용량
        today = datetime.now().date()
        cursor.execute('''
            SELECT COUNT(*) FROM usage_log 
            WHERE user_uuid = ? AND DATE(timestamp) = ?
        ''', (user_uuid, today))
        daily_count = cursor.fetchone()[0]
        
        # 전체 사용량
        cursor.execute('''
            SELECT COUNT(*) FROM usage_log 
            WHERE user_uuid = ?
        ''', (user_uuid,))
        total_count = cursor.fetchone()[0]
        
        # 엔드포인트별 사용량
        cursor.execute('''
            SELECT endpoint, COUNT(*) as count FROM usage_log 
            WHERE user_uuid = ? 
            GROUP BY endpoint 
            ORDER BY count DESC
        ''', (user_uuid,))
        endpoint_stats = [{'endpoint': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 최근 7일 사용량
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count 
            FROM usage_log 
            WHERE user_uuid = ? AND timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''', (user_uuid,))
        weekly_stats = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'daily_usage': daily_count,
            'total_usage': total_count,
            'endpoint_stats': endpoint_stats,
            'weekly_stats': weekly_stats
        }
    
    def change_user_password(self, user_uuid, new_password):
        """사용자 비밀번호 변경 (관리자용)"""
        import bcrypt
        
        # 비밀번호 해시화
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET password_hash = ? WHERE uuid = ?
        ''', (hashed_password, user_uuid))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def change_user_tier(self, user_uuid, new_tier, premium_until=None):
        """사용자 등급 변경 (관리자용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET tier = ?, premium_until = ? 
            WHERE uuid = ?
        ''', (new_tier, premium_until, user_uuid))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def set_user_active_status(self, user_uuid, is_active):
        """사용자 활성화/비활성화 상태 변경 (관리자용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET is_active = ? WHERE uuid = ?
        ''', (1 if is_active else 0, user_uuid))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def get_system_stats(self):
        """전체 시스템 통계 조회 (관리자용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 통계
        cursor.execute('SELECT tier, COUNT(*) FROM users GROUP BY tier')
        user_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        active_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # 오늘 사용량
        today = datetime.now().date()
        cursor.execute('SELECT COUNT(*) FROM usage_log WHERE DATE(timestamp) = ?', (today,))
        daily_requests = cursor.fetchone()[0]
        
        # 전체 사용량
        cursor.execute('SELECT COUNT(*) FROM usage_log')
        total_requests = cursor.fetchone()[0]
        
        # 최근 가입자
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE DATE(created_at) >= date('now', '-7 days')
        ''')
        weekly_signups = cursor.fetchone()[0]
        
        # 프리미엄 만료 예정
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE tier = 'premium' AND premium_until <= datetime('now', '+7 days')
        ''')
        expiring_premium = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'users': {
                'total': total_users,
                'active': active_users,
                'by_tier': user_stats,
                'weekly_signups': weekly_signups,
                'expiring_premium': expiring_premium
            },
            'usage': {
                'daily_requests': daily_requests,
                'total_requests': total_requests
            }
        }
    
    def delete_user(self, user_uuid):
        """사용자 및 관련 데이터 완전 삭제 (관리자용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 먼저 사용자 존재 확인
            cursor.execute('SELECT uuid FROM users WHERE uuid = ?', (user_uuid,))
            if not cursor.fetchone():
                return False
            
            # 관련 데이터 먼저 삭제 (외래키 제약 조건 때문)
            cursor.execute('DELETE FROM usage_log WHERE user_uuid = ?', (user_uuid,))
            cursor.execute('DELETE FROM auto_analysis_cache WHERE user_uuid = ?', (user_uuid,))
            
            # 사용자 삭제
            cursor.execute('DELETE FROM users WHERE uuid = ?', (user_uuid,))
            
            success = cursor.rowcount > 0
            conn.commit()
            return success
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def set_user_active(self, user_uuid, is_active):
        """사용자 활성화 상태 변경 (별칭 메서드)"""
        return self.set_user_active_status(user_uuid, is_active)
    
    def cleanup_old_guest_sessions(self, days=30):
        """오래된 게스트 세션 정리"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # N일 이전의 게스트 사용자 및 관련 데이터 삭제
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            # 먼저 삭제할 게스트 UUID들 조회
            cursor.execute('''
                SELECT uuid FROM users 
                WHERE tier = 'guest' 
                AND email IS NULL 
                AND created_at < ?
            ''', (cutoff_date,))
            
            guest_uuids = [row[0] for row in cursor.fetchall()]
            
            if not guest_uuids:
                return 0
            
            # 관련 데이터 삭제
            placeholders = ','.join(['?' for _ in guest_uuids])
            cursor.execute(f'DELETE FROM usage_log WHERE user_uuid IN ({placeholders})', guest_uuids)
            cursor.execute(f'DELETE FROM auto_analysis_cache WHERE user_uuid IN ({placeholders})', guest_uuids)
            
            # 게스트 사용자 삭제
            cursor.execute(f'DELETE FROM users WHERE uuid IN ({placeholders})', guest_uuids)
            
            deleted_count = len(guest_uuids)
            conn.commit()
            return deleted_count
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

# 전역 데이터베이스 인스턴스
db_manager = DatabaseManager()