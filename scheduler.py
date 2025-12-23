#!/usr/bin/env python3
"""
Auto Analysis Scheduler
4시간 간격 자동 분석 스케줄러 (KST 기준 1시, 5시, 9시, 13시, 17시, 21시)
"""

import schedule
import time
import threading
from datetime import datetime, timezone, timedelta
import pytz
from database import db_manager
import logging

# 한국 시간대
KST = pytz.timezone('Asia/Seoul')

class AutoAnalysisScheduler:
    def __init__(self, pattern_analysis_func):
        """
        Args:
            pattern_analysis_func: 패턴 분석을 수행하는 함수
                signature: func(symbol, timeframe, query_length) -> results
        """
        self.pattern_analysis_func = pattern_analysis_func
        self.is_running = False
        self.scheduler_thread = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_schedule(self):
        """스케줄 설정 - KST 기준 4시간 간격"""
        # KST 1시, 5시, 9시, 13시, 17시, 21시에 실행
        schedule.every().day.at("01:00").do(self.run_auto_analysis).tag('auto_analysis')
        schedule.every().day.at("05:00").do(self.run_auto_analysis).tag('auto_analysis')
        schedule.every().day.at("09:00").do(self.run_auto_analysis).tag('auto_analysis')
        schedule.every().day.at("13:00").do(self.run_auto_analysis).tag('auto_analysis')
        schedule.every().day.at("17:00").do(self.run_auto_analysis).tag('auto_analysis')
        schedule.every().day.at("21:00").do(self.run_auto_analysis).tag('auto_analysis')
        
        # Binance 시간 동기화 - 10분마다 실행
        schedule.every(10).minutes.do(self.sync_binance_time).tag('time_sync')
        
        self.logger.info("📅 Auto analysis schedule setup complete")
        self.logger.info("🕐 Analysis times (KST): 01:00, 05:00, 09:00, 13:00, 17:00, 21:00")
        self.logger.info("⏰ Binance time sync: every 10 minutes")
    
    def run_auto_analysis(self):
        """자동 분석 실행 - 기본 패턴들만 캐시 업데이트"""
        try:
            current_time_kst = datetime.now(KST)
            self.logger.info(f"🚀 Starting basic pattern auto analysis at {current_time_kst.strftime('%Y-%m-%d %H:%M:%S KST')}")
            
            symbol = "BTC/USDT"
            timeframe = "4h"
            top_k = 10  # 최대값으로 설정
            
            # 🎯 기본 패턴들만 미리 캐시 (자주 사용되는 조합)
            self.logger.info("🥇 Basic pattern analysis - Most commonly used combinations")
            basic_combinations = [
                (3, 3),  # 게스트/회원 기본
                (5, 5),  # 회원 기본
                (3, 5),  # 혼합 패턴 1
                (5, 3),  # 혼합 패턴 2
            ]
            
            successful_basic = 0
            for query_length, target_length in basic_combinations:
                self.logger.info(f"  📊 Analyzing {query_length}:{target_length} pattern...")
                success = self._run_single_analysis(symbol, timeframe, current_time_kst, query_length, target_length, top_k)
                if success:
                    successful_basic += 1
                    self.logger.info(f"  ✅ {query_length}:{target_length} pattern cached successfully")
                else:
                    self.logger.warning(f"  ❌ {query_length}:{target_length} pattern analysis failed")
            
            self.logger.info(f"🎉 Basic pattern analysis completed: {successful_basic}/{len(basic_combinations)} successful")
            self.logger.info(f"📈 Success rate: {(successful_basic/len(basic_combinations))*100:.1f}%")
            self.logger.info("💡 Other patterns will be computed on-demand when requested by users")
                
        except Exception as e:
            self.logger.error(f"❌ Auto analysis error: {e}")
    
    def _run_single_analysis(self, symbol, timeframe, analysis_time, query_length, target_length, top_k):
        """단일 설정에 대한 분석 실행"""
        try:
            # 패턴 분석 실행
            results = self.pattern_analysis_func(
                symbol=symbol,
                timeframe=timeframe, 
                query_length=query_length,
                target_length=target_length,
                top_k=top_k
            )
            
            if results and results.get('similar_patterns'):
                # 분석 결과를 캐시에 저장
                db_manager.cache_auto_analysis(
                    analysis_time=analysis_time,
                    symbol=symbol,
                    timeframe=timeframe,
                    query_length=query_length,
                    target_length=target_length,
                    top_k=top_k,
                    results=results
                )
                
                pattern_count = len(results.get('similar_patterns', []))
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"    ❌ Analysis error for {query_length}:{target_length} - {e}")
            return False
    
    def sync_binance_time(self):
        """Binance 시간 동기화"""
        try:
            # 이 함수는 pattern_api.py에서 구현된 sync_binance_time을 호출
            from pattern_api import sync_binance_time
            sync_binance_time()
            self.logger.info("⏰ Binance time synchronized successfully")
        except Exception as e:
            self.logger.error(f"❌ Binance time sync error: {e}")
    
    def start(self):
        """스케줄러 시작"""
        if self.is_running:
            self.logger.warning("⚠️ Scheduler is already running")
            return
        
        self.setup_schedule()
        self.is_running = True
        
        def run_scheduler():
            self.logger.info("🎯 Auto analysis scheduler started")
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
            self.logger.info("⏹️ Auto analysis scheduler stopped")
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        # 정시에만 실행되도록 즉시 실행 제거
        self.logger.info("⏰ Scheduler will run at scheduled times: 01:00, 05:00, 09:00, 13:00, 17:00, 21:00 KST")
    
    def stop(self):
        """스케줄러 중지"""
        self.is_running = False
        schedule.clear('auto_analysis')
        schedule.clear('time_sync')
        self.logger.info("🛑 Auto analysis scheduler stopped")
    
    def get_next_analysis_time(self):
        """다음 분석 예정 시간 반환"""
        if not self.is_running:
            return None
        
        next_run = schedule.next_run()
        if next_run:
            # UTC to KST 변환
            kst_time = next_run.astimezone(KST)
            return kst_time.strftime('%Y-%m-%d %H:%M:%S KST')
        return None
    
    def get_last_analysis_time(self):
        """마지막 분석 시간 반환"""
        # 가장 최근 분석 시간을 위해 아무 필터 없이 조회
        results, analysis_time = db_manager.get_latest_auto_analysis(
            symbol="BTC/USDT", timeframe="4h", query_length=3, target_length=3, top_k=3
        )
        if analysis_time:
            # ISO string을 datetime으로 변환 후 KST로 표시
            dt = datetime.fromisoformat(analysis_time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=KST)
            return dt.strftime('%Y-%m-%d %H:%M:%S KST')
        return None
    
    def force_run_now(self):
        """수동으로 즉시 분석 실행"""
        self.logger.info("🔥 Force running auto analysis...")
        threading.Thread(target=self.run_auto_analysis, daemon=True).start()

# 전역 스케줄러 인스턴스 (pattern_api.py에서 초기화)
auto_scheduler = None