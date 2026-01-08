#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
import sys
from datetime import datetime


def read_meminfo():
    meminfo = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                rest = parts[1].strip()
                value = rest.split()[0]
                try:
                    meminfo[key] = int(value)  # kB
                except ValueError:
                    pass
    except FileNotFoundError:
        print("未找到 /proc/meminfo，本脚本需在Linux上运行。", file=sys.stderr)
        sys.exit(1)
    return meminfo


def get_system_memory():
    info = read_meminfo()
    mem_total_kb = info.get("MemTotal", 0)
    mem_free_kb = info.get("MemFree", 0)
    buffers_kb = info.get("Buffers", 0)
    cached_kb = info.get("Cached", 0)
    sreclaimable_kb = info.get("SReclaimable", 0)
    shmem_kb = info.get("Shmem", 0)


    mem_available_kb = info.get("MemAvailable")
    if mem_available_kb is None:
        # 估算：free + buffers + cache(含可回收slab) - shmem
        mem_available_kb = mem_free_kb + buffers_kb + (cached_kb + sreclaimable_kb - shmem_kb)
        if mem_available_kb < 0:
            mem_available_kb = 0

    mem_used_kb = max(mem_total_kb - mem_available_kb, 0)

    total_mb = mem_total_kb / 1024.0
    used_mb = mem_used_kb / 1024.0
    percent = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0

    return {
        "total_mb": total_mb,
        "used_mb": used_mb,
        "percent": percent,
    }


def get_process_memory_kb(pid):
    status_path = f"/proc/{pid}/status"
    try:
        with open(status_path, "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # VmRSS: <value> kB
                    if len(parts) >= 2:
                        try:
                            return int(parts[1])
                        except ValueError:
                            return None
        # 兜底：尝试statm（单位为pages）
        with open(f"/proc/{pid}/statm", "r") as f:
            content = f.read().strip().split()
            if len(content) >= 2:
                resident_pages = int(content[1])
                page_size_kb = os.sysconf("SC_PAGE_SIZE") // 1024
                return resident_pages * page_size_kb
    except FileNotFoundError:
        return None
    except PermissionError:
        return None
    except Exception:
        return None


def format_row(ts, sys_used_mb, sys_total_mb, sys_percent, proc_rss_mb):
    time_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    if proc_rss_mb is None:
        return f"{time_str} | System Used: {sys_used_mb:.1f}MB / {sys_total_mb:.1f}MB ({sys_percent:.1f}%)"
    else:
        return (
            f"{time_str} | System Used: {sys_used_mb:.1f}MB / {sys_total_mb:.1f}MB ({sys_percent:.1f}%)"
            f" | Proc RSS: {proc_rss_mb:.1f}MB"
        )


def ensure_writable(path):
    if path is None:
        return
    parent = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    # 测试写权限
    try:
        with open(path, "a") as _:
            pass
    except Exception as e:
        print(f"无法写入日志文件: {path}，错误: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="实时监控内存消耗（系统与可选进程）")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔秒，默认1s")
    parser.add_argument("--pid", type=int, default=None, help="要监控的进程PID，可选")
    parser.add_argument("--log", type=str, default=None, help="将结果追加写入CSV文件路径，可选")
    parser.add_argument("--header", action="store_true", help="写CSV时先写表头（若文件新建建议加）")
    args = parser.parse_args()

    if args.interval <= 0:
        print("interval 必须 > 0", file=sys.stderr)
        sys.exit(1)

    if args.pid is not None and not os.path.exists(f"/proc/{args.pid}"):
        print(f"PID {args.pid} 不存在或不可访问。", file=sys.stderr)
        sys.exit(1)

    if args.log:
        ensure_writable(args.log)
        if args.header and (not os.path.exists(args.log) or os.path.getsize(args.log) == 0):
            with open(args.log, "a", encoding="utf-8") as f:
                if args.pid is None:
                    f.write("timestamp,system_used_mb,system_total_mb,system_percent\n")
                else:
                    f.write("timestamp,system_used_mb,system_total_mb,system_percent,proc_rss_mb,pid\n")

    print("按 Ctrl-C 结束监控。")
    try:
        while True:
            ts = datetime.now()

            sys_mem = get_system_memory()
            sys_used_mb = sys_mem["used_mb"]
            sys_total_mb = sys_mem["total_mb"]
            sys_percent = sys_mem["percent"]

            proc_rss_mb = None
            if args.pid is not None:
                rss_kb = get_process_memory_kb(args.pid)
                if rss_kb is not None:
                    proc_rss_mb = rss_kb / 1024.0

            # 控制台输出
            line = format_row(ts, sys_used_mb, sys_total_mb, sys_percent, proc_rss_mb)
            print(line)

            # 写CSV
            if args.log:
                with open(args.log, "a", encoding="utf-8") as f:
                    if proc_rss_mb is None:
                        f.write(
                            f"{ts.isoformat(timespec='seconds')},{sys_used_mb:.3f},{sys_total_mb:.3f},{sys_percent:.3f}\n"
                        )
                    else:
                        f.write(
                            f"{ts.isoformat(timespec='seconds')},{sys_used_mb:.3f},{sys_total_mb:.3f},{sys_percent:.3f},{proc_rss_mb:.3f},{args.pid}\n"
                        )

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()


