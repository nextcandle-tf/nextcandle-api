#!/usr/bin/env python3
"""
데이터베이스 관리 CLI
— 사용자 조회/추가/삭제/등급변경, 사용량 통계 조회 등을 위한 스크립트입니다.

개선 사항
- 등급별 사용자 조회(list-tier) 기능 추가
- 전체 사용법과 출력 문구를 한글로 정비
"""

import sqlite3
import sys
import os
from datetime import datetime
import uuid
import hashlib

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import DatabaseManager, UserTier

class DBController:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.db_path = self.db_manager.db_path
    
    def list_users(self):
        """모든 사용자 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, uuid, email, tier, created_at, last_login, premium_until, is_active 
            FROM users ORDER BY created_at DESC
        ''')
        users = cursor.fetchall()
        conn.close()
        
        print("=" * 110)
        print(f"{'ID':<4} | {'UUID':<12} | {'Email':<25} | {'Tier':<8} | {'Created':<19} | {'Last Login':<19} | {'Premium Until':<19} | {'Active':<6}")
        print("=" * 110)
        
        for user in users:
            uuid_short = user[1][:8] + "..." if user[1] else "None"
            email = user[2] if user[2] else "None"
            created = user[4][:19] if user[4] else "None"
            last_login = user[5][:19] if user[5] else "None"
            premium_until = user[6][:19] if user[6] else "None"
            is_active = "Yes" if user[7] else "No"
            
            print(f"{user[0]:<4} | {uuid_short:<12} | {email:<25} | {user[3]:<8} | {created:<19} | {last_login:<19} | {premium_until:<19} | {is_active:<6}")
        
        print(f"\nTotal users: {len(users)}")

    def list_users_by_tier(self, tier: str):
        """특정 등급 사용자만 조회

        Args:
            tier (str): guest | member | premium | admin
        """
        # 유효성 검사 (대소문자 허용)
        valid_tiers = [t.value for t in UserTier]
        tier_norm = tier.lower()
        if tier_norm not in valid_tiers:
            print(f"유효하지 않은 등급입니다. 사용 가능한 등급: {', '.join(valid_tiers)}")
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, uuid, email, tier, created_at, last_login, premium_until, is_active 
            FROM users WHERE tier = ? ORDER BY created_at DESC
        ''', (tier_norm,))
        users = cursor.fetchall()
        conn.close()

        print(f"등급 '{tier_norm}' 사용자 목록")
        print("=" * 110)
        print(f"{'ID':<4} | {'UUID':<12} | {'Email':<25} | {'Tier':<8} | {'Created':<19} | {'Last Login':<19} | {'Premium Until':<19} | {'Active':<6}")
        print("=" * 110)

        for user in users:
            uuid_short = user[1][:8] + "..." if user[1] else "None"
            email = user[2] if user[2] else "None"
            created = user[4][:19] if user[4] else "None"
            last_login = user[5][:19] if user[5] else "None"
            premium_until = user[6][:19] if user[6] else "None"
            is_active = "Yes" if user[7] else "No"

            print(f"{user[0]:<4} | {uuid_short:<12} | {email:<25} | {user[3]:<8} | {created:<19} | {last_login:<19} | {premium_until:<19} | {is_active:<6}")

        print(f"\n총 {len(users)}명")
        return True
    
    def get_user_by_email(self, email):
        """이메일로 사용자 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, uuid, email, tier, created_at, last_login, premium_until, is_active 
            FROM users WHERE email = ?
        ''', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            print("사용자 정보:")
            print(f"ID: {user[0]}")
            print(f"UUID: {user[1]}")
            print(f"Email: {user[2]}")
            print(f"Tier: {user[3]}")
            print(f"Created: {user[4]}")
            print(f"Last Login: {user[5]}")
            print(f"Premium Until: {user[6]}")
            print(f"Active: {'Yes' if user[7] else 'No'}")
        else:
            print(f"해당 이메일 '{email}' 사용자를 찾을 수 없습니다.")
        
        return user
    
    def get_user_by_uuid(self, user_uuid):
        """UUID로 사용자 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, uuid, email, tier, created_at, last_login, premium_until, is_active 
            FROM users WHERE uuid = ?
        ''', (user_uuid,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            print("사용자 정보:")
            print(f"ID: {user[0]}")
            print(f"UUID: {user[1]}")
            print(f"Email: {user[2]}")
            print(f"Tier: {user[3]}")
            print(f"Created: {user[4]}")
            print(f"Last Login: {user[5]}")
            print(f"Premium Until: {user[6]}")
            print(f"Active: {'Yes' if user[7] else 'No'}")
        else:
            print(f"해당 UUID '{user_uuid}' 사용자를 찾을 수 없습니다.")
        
        return user
    
    def change_user_tier(self, email, new_tier, days=30):
        """사용자 등급 변경"""
        # Validate tier
        valid_tiers = [tier.value for tier in UserTier]
        new_tier = new_tier.lower()
        if new_tier not in valid_tiers:
            print(f"유효하지 않은 등급입니다. 사용 가능한 등급: {', '.join(valid_tiers)}")
            return False

        # 사용자 조회(현재 등급/UUID 확인)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT uuid, tier FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            print(f"해당 이메일 '{email}' 사용자를 찾을 수 없습니다.")
            return False

        user_uuid, old_tier = row[0], row[1]

        # 프리미엄은 만료일을 지정된 기간으로 부여
        if new_tier == UserTier.PREMIUM.value:
            self.db_manager.upgrade_to_premium(user_uuid, days=days)
            print(f"사용자 '{email}' 등급을 '{old_tier}' → '{new_tier}' ({days}일)로 변경했습니다.")
        else:
            self.db_manager.change_user_tier(user_uuid, new_tier)
            print(f"사용자 '{email}' 등급을 '{old_tier}' → '{new_tier}' 로 변경했습니다.")
        
        return True
    
    def add_user(self, email, password=None, tier="guest"):
        """사용자 추가"""
        # Validate tier
        valid_tiers = [tier.value for tier in UserTier]
        tier = tier.lower()
        if tier not in valid_tiers:
            print(f"유효하지 않은 등급입니다. 사용 가능한 등급: {', '.join(valid_tiers)}")
            return False
        
        # register_user는 비밀번호가 필요하며 기본 등급은 MEMBER로 생성됩니다.
        if not password:
            print("비밀번호가 필요합니다. 예: python3 db_control.py add test@example.com mypass member")
            return False

        try:
            user_uuid = self.db_manager.register_user(email, password)
            if not user_uuid:
                print(f"사용자 '{email}' 추가 실패(이미 존재할 수 있음)")
                return False

            # 요청 등급이 MEMBER가 아니면 후처리로 등급 변경
            if tier == UserTier.PREMIUM.value:
                # 프리미엄은 만료일 자동 부여(기본 30일)
                self.db_manager.upgrade_to_premium(user_uuid, days=30)
                final_tier = tier
            elif tier == UserTier.MEMBER.value:
                final_tier = tier  # 그대로
            else:
                # guest/admin 등은 직접 등급 변경 API 사용
                self.db_manager.change_user_tier(user_uuid, tier)
                final_tier = tier

            print("사용자 추가 완료:")
            print(f"  Email: {email}")
            print(f"  UUID: {user_uuid}")
            print(f"  Tier: {final_tier}")
            return True
        except Exception as e:
            print(f"사용자 추가 중 오류: {e}")
            return False
    
    def delete_user(self, email):
        """사용자 삭제"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 존재 확인
        cursor.execute('SELECT id, email FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if not user:
            print(f"해당 이메일 '{email}' 사용자를 찾을 수 없습니다.")
            conn.close()
            return False
        
        # 삭제 전 UUID 조회
        cursor.execute('SELECT uuid FROM users WHERE email = ?', (email,))
        user_data = cursor.fetchone()
        
        # 연관 레코드 우선 삭제
        if user_data:
            cursor.execute('DELETE FROM usage_log WHERE user_uuid = ?', (user_data[0],))
        
        # 사용자 삭제
        cursor.execute('DELETE FROM users WHERE email = ?', (email,))
        
        conn.commit()
        conn.close()
        
        print(f"사용자 '{email}' (ID: {user[0]}) 삭제 완료")
        return True
    
    def reset_usage_logs(self, email=None):
        """사용량 로그 삭제 (특정 사용자 또는 전체)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if email:
            # 사용자 UUID 조회
            cursor.execute('SELECT uuid FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            if user:
                cursor.execute('DELETE FROM usage_log WHERE user_uuid = ?', (user[0],))
                print(f"사용자 '{email}'의 사용량 로그를 삭제했습니다.")
            else:
                print(f"해당 이메일 '{email}' 사용자를 찾을 수 없습니다.")
        else:
            cursor.execute('DELETE FROM usage_log')
            print("모든 사용량 로그를 삭제했습니다.")
        
        conn.commit()
        conn.close()
        return True
    
    def change_password(self, email, new_password):
        """사용자 비밀번호 변경"""
        # 사용자 UUID 조회
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT uuid FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            print(f"해당 이메일 '{email}' 사용자를 찾을 수 없습니다.")
            return False
        
        user_uuid = user[0]
        
        try:
            # DatabaseManager의 change_user_password 메소드 사용
            success = self.db_manager.change_user_password(user_uuid, new_password)
            
            if success:
                print(f"사용자 '{email}'의 비밀번호를 성공적으로 변경했습니다.")
                return True
            else:
                print(f"비밀번호 변경에 실패했습니다.")
                return False
                
        except Exception as e:
            print(f"비밀번호 변경 중 오류 발생: {e}")
            return False

    def show_usage_stats(self):
        """사용량 통계 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User count by tier
        cursor.execute('SELECT tier, COUNT(*) FROM users GROUP BY tier ORDER BY tier')
        tier_stats = cursor.fetchall()
        
        # Usage log stats
        cursor.execute('SELECT COUNT(*) FROM usage_log')
        total_requests = cursor.fetchone()[0]
        
        cursor.execute('SELECT endpoint, COUNT(*) FROM usage_log GROUP BY endpoint ORDER BY COUNT(*) DESC')
        endpoint_stats = cursor.fetchall()
        
        # Active users
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        active_users = cursor.fetchone()[0]
        
        conn.close()
        
        print("=== 사용량 통계 ===")
        print("등급별 사용자 수:")
        for tier, count in tier_stats:
            print(f"  {tier}: {count}")
        
        print(f"\n사용자 집계:")
        print(f"  전체 사용자: {sum(count for _, count in tier_stats)}")
        print(f"  활성 사용자: {active_users}")
        
        print(f"\nAPI 사용량:")
        print(f"  총 요청 수: {total_requests}")
        print("  상위 엔드포인트:")
        for endpoint, count in endpoint_stats[:5]:
            print(f"    {endpoint}: {count}")

def main():
    controller = DBController()
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python3 db_control.py list                               # 모든 사용자 조회")
        print("  python3 db_control.py list-tier <tier>                   # 특정 등급 사용자만 조회")
        print("  python3 db_control.py get <email>                        # 이메일로 사용자 조회")
        print("  python3 db_control.py get-uuid <uuid>                    # UUID로 사용자 조회")
        print("  python3 db_control.py tier <email> <new_tier> [days]     # 사용자 등급 변경 (premium일 경우 기간 지정 가능)")
        print("  python3 db_control.py add <email> [password] [tier]      # 사용자 추가")
        print("  python3 db_control.py delete <email>                     # 사용자 삭제")
        print("  python3 db_control.py password <email> <new_password>    # 사용자 비밀번호 변경")
        print("  python3 db_control.py reset-logs [email]                 # 사용량 로그 초기화(특정/전체)")
        print("  python3 db_control.py stats                              # 사용량 통계 조회")
        print("")
        print("사용 가능한 등급: guest, member, premium, admin")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        controller.list_users()
    
    elif command == "list-tier":
        if len(sys.argv) < 3:
            print("사용법: python3 db_control.py list-tier <tier>")
            print("예: python3 db_control.py list-tier premium")
            return
        controller.list_users_by_tier(sys.argv[2])
    
    elif command == "get":
        if len(sys.argv) < 3:
            print("사용법: python3 db_control.py get <email>")
            return
        controller.get_user_by_email(sys.argv[2])
    
    elif command == "get-uuid":
        if len(sys.argv) < 3:
            print("사용법: python3 db_control.py get-uuid <uuid>")
            return
        controller.get_user_by_uuid(sys.argv[2])
    
    elif command == "tier":
        if len(sys.argv) < 4:
            print("사용법: python3 db_control.py tier <email> <new_tier> [days]")
            print("예: python3 db_control.py tier user@example.com premium 60")
            return
        email = sys.argv[2]
        new_tier = sys.argv[3]
        days = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        controller.change_user_tier(email, new_tier, days)
    
    elif command == "add":
        if len(sys.argv) < 3:
            print("사용법: python3 db_control.py add <email> [password] [tier]")
            return
        email = sys.argv[2]
        password = sys.argv[3] if len(sys.argv) > 3 else None
        tier = sys.argv[4] if len(sys.argv) > 4 else "guest"
        controller.add_user(email, password, tier)
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("사용법: python3 db_control.py delete <email>")
            return
        email = sys.argv[2]
        confirm = input(f"사용자 '{email}'를 정말 삭제하시겠습니까? (y/N): ")
        if confirm.lower() == 'y':
            controller.delete_user(email)
        else:
            print("삭제를 취소했습니다.")
    
    elif command == "password":
        if len(sys.argv) < 4:
            print("사용법: python3 db_control.py password <email> <new_password>")
            print("예: python3 db_control.py password user@example.com newpassword123")
            return
        email = sys.argv[2]
        new_password = sys.argv[3]
        controller.change_password(email, new_password)
    
    elif command == "reset-logs":
        if len(sys.argv) > 2:
            controller.reset_usage_logs(sys.argv[2])
        else:
            confirm = input("모든 사용자의 사용량 로그를 삭제하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                controller.reset_usage_logs()
            else:
                print("삭제를 취소했습니다.")
    
    elif command == "stats":
        controller.show_usage_stats()
    
    else:
        print(f"알 수 없는 명령어입니다: {command}")

if __name__ == "__main__":
    main()