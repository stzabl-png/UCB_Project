#!/bin/bash
# 实时进度监控 — HaWoR EgoDex
# 用法: bash watch_progress.sh
OUT_DIR=~/Project/Affordance2Grasp/data_hub/ProcessedData/egocentric/egodex
LOG=~/Project/Affordance2Grasp/output/hawor_egodex.log
PID_FILE=~/Project/Affordance2Grasp/output/hawor_egodex.pid
TOTAL=3051

START_TIME=$(date +%s)
PREV_DONE=0
PREV_TIME=$START_TIME

while true; do
    clear
    NOW=$(date +%s)

    # 核心计数：mano.npz 文件数 = 实际完成数
    DONE=$(find "$OUT_DIR" -name "mano.npz" 2>/dev/null | wc -l)
    REMAIN=$(( TOTAL - DONE ))
    PCT=$(( DONE * 100 / TOTAL ))
    FILLED=$(( PCT * 44 / 100 ))

    BAR=""
    for i in $(seq 1 $FILLED 2>/dev/null); do BAR="${BAR}█"; done
    for i in $(seq 1 $(( 44 - FILLED )) 2>/dev/null); do BAR="${BAR}░"; done

    # 速度 & ETA（基于滑动窗口：本次 vs 上次采样差值）
    WINDOW=$(( NOW - PREV_TIME ))
    DELTA=$(( DONE - PREV_DONE ))
    if [ $DELTA -gt 0 ] && [ $WINDOW -gt 0 ]; then
        SEC_PER=$(( WINDOW / DELTA ))
        ETA_SEC=$(( REMAIN * SEC_PER ))
        ETA_H=$(( ETA_SEC / 3600 ))
        ETA_M=$(( (ETA_SEC % 3600) / 60 ))
        RATE_H=$(( DELTA * 3600 / WINDOW ))
        ETA_STR="${ETA_H}h ${ETA_M}m"
        RATE_STR="${RATE_H} seq/h"
    elif [ $DONE -gt 0 ]; then
        ELAPSED=$(( NOW - START_TIME ))
        if [ $ELAPSED -lt 1 ]; then ELAPSED=1; fi
        SEC_PER=$(( ELAPSED / DONE ))
        ETA_SEC=$(( REMAIN * SEC_PER ))
        ETA_H=$(( ETA_SEC / 3600 ))
        ETA_M=$(( (ETA_SEC % 3600) / 60 ))
        RATE_H=$(( DONE * 3600 / ELAPSED ))
        ETA_STR="${ETA_H}h ${ETA_M}m"
        RATE_STR="${RATE_H} seq/h (均值)"
    else
        ETA_STR="计算中..."
        RATE_STR="-"
    fi
    PREV_DONE=$DONE
    PREV_TIME=$NOW

    # ── 进程状态检测（三层）────────────────────────────────────
    # 1) 用 pgrep 直接搜索
    HAWOR_PID=$(pgrep -f "batch_hawor" | head -1)
    if [ -n "$HAWOR_PID" ]; then
        STATUS="🟢 运行中 (PID $HAWOR_PID)"
    else
        # 2) 读取 PID 文件，确认进程是否存活
        SAVED_PID=""
        [ -f "$PID_FILE" ] && SAVED_PID=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$SAVED_PID" ] && kill -0 "$SAVED_PID" 2>/dev/null; then
            STATUS="🟢 运行中 (PID $SAVED_PID)"
        else
            # 3) 检查日志最后修改时间（容忍 10 分钟内刚启动未写入）
            LOG_AGE=9999
            if [ -f "$LOG" ]; then
                LOG_MTIME=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
                LOG_AGE=$(( NOW - LOG_MTIME ))
            fi
            if [ $LOG_AGE -lt 600 ]; then
                STATUS="🟡 日志活跃 (${LOG_AGE}s前更新) — 进程可能刚启动"
            else
                LOG_AGE_MIN=$(( LOG_AGE / 60 ))
                STATUS="🔴 进程已停止！(日志 ${LOG_AGE_MIN} 分钟未更新)"
            fi
        fi
    fi

    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║          HaWoR EgoDex — 实时进度监控                 ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    printf "  进度  [%s] %d%%\n" "$BAR" "$PCT"
    echo ""
    printf "  📁 mano.npz 已完成: %d / %d\n" "$DONE" "$TOTAL"
    printf "  ⏳ 剩余: %d 条\n" "$REMAIN"
    printf "  ⚡ 速度: %s\n" "$RATE_STR"
    printf "  🏁 预计完成: %s\n" "$ETA_STR"
    echo ""
    echo "  状态: $STATUS"
    echo ""
    echo "───────────────────────────────────────────────────────"
    echo "  日志最新3行:"
    # 提取 tqdm 进度条最新状态（去掉回车符）
    tail -5 "$LOG" 2>/dev/null | tr '\r' '\n' | grep "HaWoR\|seq/it\|it/s" | tail -3 | sed 's/^/  /'
    echo ""
    printf "  刷新: 每30秒   更新时间: %s   Ctrl+C 退出\n" "$(date '+%H:%M:%S')"

    sleep 30
done
