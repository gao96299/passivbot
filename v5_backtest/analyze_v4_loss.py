"""V4 回放亏损归因分析"""
import csv, json, sqlite3
from collections import defaultdict
from datetime import datetime, timezone

D = r"E:\code\workspace\passivbot\v5_backtest\results\v4_2026-09-01_2026-10-01"

# ---------- 1. 成交分类统计 ----------
fills = list(csv.DictReader(open(D + r"\fills.csv", encoding="utf-8")))

def cat(r):
    reason = r["reason"]
    if "BREAK" in reason: return "破局单"
    if "止盈" in reason or "防守" in reason: return "平仓单"
    return "开仓单"

by_cat = defaultdict(lambda: {"n": 0, "fee": 0.0})
for r in fills:
    c = cat(r)
    by_cat[c]["n"] += 1
    by_cat[c]["fee"] += float(r["fee"])

print("=== 1. 成交分类 ===")
for c, v in by_cat.items():
    print(f"  {c}: {v['n']} 笔, 手续费 {v['fee']:.2f} USDT")

# ---------- 2. 按天统计权益变化 ----------
eq = list(csv.DictReader(open(D + r"\equity.csv", encoding="utf-8")))
daily = defaultdict(lambda: {"start": None, "end": None})
for r in eq:
    day = r["time"][:10]
    if daily[day]["start"] is None:
        daily[day]["start"] = float(r["equity"])
    daily[day]["end"] = float(r["equity"])

print("\n=== 2. 逐日权益变化 (亏损日标红) ===")
worst_days = []
for day in sorted(daily):
    s, e = daily[day]["start"], daily[day]["end"]
    chg = e - s
    worst_days.append((day, chg, s, e))
    mark = " <<<" if chg < -5 else ""
    print(f"  {day}: {s:10.2f} -> {e:10.2f}  ({chg:+8.2f}){mark}")

worst_days.sort(key=lambda x: x[1])
print("\n=== 3. 最惨的 5 天 ===")
for day, chg, s, e in worst_days[:5]:
    print(f"  {day}: {chg:+.2f} USDT ({s:.2f} -> {e:.2f})")

# ---------- 4. 破局单分析 ----------
breaks = [r for r in fills if "BREAK" in r["reason"]]
print(f"\n=== 4. 破局单明细 ({len(breaks)} 笔) ===")
reduce_breaks = [r for r in breaks if r["reduceOnly"] == "True"]
open_breaks = [r for r in breaks if r["reduceOnly"] == "False"]
print(f"  破局平仓(割肉)单: {len(reduce_breaks)} 笔")
print(f"  破局移仓(开仓)单: {len(open_breaks)} 笔")
mkt = [r for r in breaks if r["taker"] == "True"]
print(f"  其中市价强制单: {len(mkt)} 笔")

# ---------- 5. 期末持仓浮亏 ----------
last = eq[-1]
print(f"\n=== 5. 期末状态 ===")
print(f"  期末 BTC 价: {last['btc_price']}")
print(f"  期末空头: {last['short_pos']} 张")
print(f"  已实现余额: {last['balance']}")
print(f"  总权益: {last['equity']}")
print(f"  期末浮动盈亏: {float(last['equity']) - float(last['balance']):+.2f} USDT")

# ---------- 6. 从 DB 看合并盈亏 ----------
conn = sqlite3.connect(D + r"\v4_replay.db")
cur = conn.cursor()
merges = cur.execute("SELECT merge_group_label, profit, net_profit, total_fee FROM merged_history").fetchall()
total_profit = sum(m[1] for m in merges)
total_net = sum(m[2] for m in merges)
total_fee_m = sum(m[3] for m in merges)
loss_merges = [m for m in merges if m[2] < 0]
print(f"\n=== 6. 消消乐合并 ({len(merges)} 次) ===")
print(f"  毛利合计: {total_profit:+.2f}")
print(f"  手续费合计: {total_fee_m:.2f}")
print(f"  净利合计: {total_net:+.2f}")
print(f"  亏损合并: {len(loss_merges)} 次, 合计 {sum(m[2] for m in loss_merges):+.2f}")
print(f"  盈利合并: {len(merges)-len(loss_merges)} 次, 合计 {sum(m[2] for m in merges if m[2]>=0):+.2f}")

# 亏损合并的标签分布
loss_labels = defaultdict(lambda: [0, 0.0])
for m in loss_merges:
    key = m[0].split("-")[0] if m[0] else "?"
    loss_labels[key][0] += 1
    loss_labels[key][1] += m[2]
print("  亏损合并来源:")
for k, (n, amt) in sorted(loss_labels.items(), key=lambda x: x[1][1]):
    print(f"    {k}: {n} 次, {amt:+.2f}")

# ---------- 7. 资金流水重建 ----------
print(f"\n=== 7. 资金流水总账 ===")
total_fee_fills = sum(float(r["fee"]) for r in fills)
print(f"  初始资金:        10000.00")
print(f"  消消乐合并净利:  {total_net:+.2f}  (毛利 {total_profit:+.2f} - 合并手续费 {total_fee_m:.2f})")
print(f"  全部成交手续费:  {-total_fee_fills:.2f}")
print(f"  期末浮动盈亏:    {float(last['equity']) - float(last['balance']):+.2f}")
print(f"  期末权益:        {last['equity']}")
conn.close()
