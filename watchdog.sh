#!/bin/bash
# 봇 감시 스크립트 - 3시간 이상 로그 변화 없으면 자동 재시작

LOG_FILE="/Users/cng/bot.log"
BOT_SCRIPT="recruiter_bot.py"
MAX_IDLE=3600   # 1시간 (초)
CHECK_INTERVAL=1800  # 30분마다 체크

echo "$(date): watchdog 시작"

while true; do
    sleep $CHECK_INTERVAL

    # 봇 프로세스 확인
    if ! pgrep -f "$BOT_SCRIPT" > /dev/null; then
        echo "$(date): 봇 프로세스 없음 — 재시작"
        source ~/.bot_env
        rm -f /Users/cng/chrome-bot-profile/Singleton*
        cd /Users/cng && OPENAI_API_KEY="$OPENAI_API_KEY" PYTHONIOENCODING=utf-8 nohup python3 -u "$BOT_SCRIPT" >> bot.log 2>&1 &
        continue
    fi

    # 로그 마지막 수정 시간 확인
    LAST_MOD=$(stat -f %m "$LOG_FILE" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    IDLE=$((NOW - LAST_MOD))
    echo "$(date): 체크 — 유휴 ${IDLE}초 / 기준 ${MAX_IDLE}초"

    if [ $IDLE -gt $MAX_IDLE ]; then
        echo "$(date): 로그 ${IDLE}초 동안 변화 없음 — 봇 재시작"
        pkill -f "$BOT_SCRIPT"
        pkill -f chrome
        sleep 5
        rm -f /Users/cng/chrome-bot-profile/Singleton*
        source ~/.bot_env
        cd /Users/cng && OPENAI_API_KEY="$OPENAI_API_KEY" PYTHONIOENCODING=utf-8 nohup python3 -u "$BOT_SCRIPT" >> bot.log 2>&1 &
    fi
done
