"""
V5 策略回放回测器
==================
用 Passivbot 的 HLCV 历史数据缓存，驱动 F:\\oneProject\\src\\v5 的原版策略代码
(MarketSenseUnit + OrderGenerator + RiskManager + OrderTracker) 做逐分钟回放。

保真度说明:
- 策略决策 100% 使用你的实盘代码 (import src.v5.*)，无任何翻译/重写
- time.time() 被替换为模拟时钟，LOCKDOWN 迟滞环/冷却期按模拟时间正确推进
- OrderTracker 使用独立临时 SQLite，消消乐/债务账本/公积金逻辑原样运行

模拟简化 (与实盘的差异，必须知晓):
1. 成交模型: 限价单在 1m K 线 high/low 穿越挂单价时按挂单价成交 (maker)
2. post_only 语义: 若 K 线开盘价已穿越挂单价(会立即成交)，按 OKX 规则撤单而非成交
3. 决策频率: 实盘按 tick 节流(>=1s)，回放为每根 1m K 线决策一次
4. 盘口代理: bid/ask = close ∓ half_spread (默认 1bp)
5. 保证金占用率近似: (总持仓面值/杠杆) / 权益

用法:
  python v5_backtest\\v5_replay.py --start 2026-09-01 --end 2026-10-01
  # 覆盖 V5 参数 (等价于改 .env):
  python v5_backtest\\v5_replay.py --start 2026-09-01 --end 2026-10-01 -e GRID_SPACING_PERCENT=0.002 -e HEDGE_TRIGGER_RATIO=0.5
"""

import argparse
import gzip
import json
import os
import sys
import time as _time_module
from datetime import datetime, timezone

import numpy as np

PB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5_PROJECT_ROOT = os.environ.get("V5_PROJECT_ROOT", r"F:\oneProject")
HLCV_ROOT = os.path.join(PB_ROOT, "caches", "hlcvs_data")


# ----------------------------------------------------------------------
# 模拟时钟: 在导入 V5 之前打补丁，V5 内部所有 time.time() 都走模拟时间
# ----------------------------------------------------------------------
class SimClock:
    t = 0.0


_time_module.time = lambda: SimClock.t


def parse_args():
    ap = argparse.ArgumentParser(description="V5 replay backtester")
    ap.add_argument("--start", required=True, help="回测起始日期 YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="回测结束日期 YYYY-MM-DD (UTC, 不含)")
    ap.add_argument("--data", default=None, help="HLCV 数据集目录 (默认自动在 caches/hlcvs_data 里找覆盖区间)")
    ap.add_argument("--balance", type=float, default=10000.0, help="初始资金 USDT")
    ap.add_argument("--maker-fee", type=float, default=0.0002, help="maker 手续费率 (OKX 默认 0.02%)")
    ap.add_argument("--half-spread", type=float, default=0.0001, help="盘口半价差 (1bp)")
    ap.add_argument("--out", default=None, help="输出目录 (默认 v5_backtest/results/<start>_<end>)")
    ap.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖 V5 配置 (在导入 settings 前注入环境变量)，可多次")
    return ap.parse_args()


def inject_env(overrides):
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"-e 参数格式错误: {item} (应为 KEY=VALUE)")
        k, v = item.split("=", 1)
        os.environ[k.strip()] = v.strip()
        print(f"[env] 覆盖 V5 配置: {k.strip()} = {v.strip()}")


def find_dataset(start_ms, end_ms, explicit=None):
    """在 caches/hlcvs_data 里找覆盖 [start, end] 的数据集"""
    candidates = [explicit] if explicit else [
        os.path.join(HLCV_ROOT, d) for d in os.listdir(HLCV_ROOT)
        if not d.startswith(".") and os.path.isdir(os.path.join(HLCV_ROOT, d))
    ]
    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        tf = os.path.join(c, "timestamps.npy.gz")
        if not os.path.exists(tf):
            continue
        ts = np.load(gzip.open(tf))
        if ts[0] <= start_ms and ts[-1] >= end_ms:
            return c, ts
    raise SystemExit("找不到覆盖该区间的 HLCV 数据集。先跑一次 passivbot backtest 生成缓存，或用 --data 指定目录。")


def load_dataset(ds_dir):
    ts = np.load(gzip.open(os.path.join(ds_dir, "timestamps.npy.gz")))
    hlcvs = np.load(gzip.open(os.path.join(ds_dir, "hlcvs.npy.gz")))
    return ts, hlcvs[:, 0, :]  # (n, [high, low, close, volume])


class CandleAggregator15m:
    """把 1m K 线聚合成 15m K 线喂给 MarketSenseUnit"""

    def __init__(self, on_candle):
        self.on_candle = on_candle
        self.cur = None

    def push(self, ts_ms, high, low, close):
        bucket = (ts_ms // 900_000) * 900_000
        if self.cur is None or self.cur["timestamp"] != bucket:
            if self.cur is not None:
                self.on_candle(self.cur)
            self.cur = {"timestamp": bucket, "open": close, "high": high, "low": low, "close": close}
        else:
            self.cur["high"] = max(self.cur["high"], high)
            self.cur["low"] = min(self.cur["low"], low)
            self.cur["close"] = close

    def flush(self):
        if self.cur is not None:
            self.on_candle(self.cur)
            self.cur = None


class V5Replay:
    def __init__(self, args, settings):
        self.args = args
        self.settings = settings
        self.C = settings.CONTRACT_VALUE
        self.lev = float(settings.LEVERAGE)

        # --- 导入并实例化 V5 原版组件 ---
        from src.v5.engine.state_machine import V5StateMachine
        from src.v5.tracking.order_tracker import OrderTracker
        import logging
        for name in ("MarketSenseUnit", "OrderGenerator", "RiskManager", "StateMachine", "OrderTracker"):
            logging.getLogger(name).setLevel(logging.CRITICAL)

        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.db_path = os.path.join(out_dir, "v5_replay.db")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        self.sm = V5StateMachine(symbol=settings.SYMBOL)
        self.tracker = OrderTracker(db_path=self.db_path)

        # --- 账户与持仓状态 (双向持仓模式) ---
        self.balance = args.balance
        self.long_pos = 0.0
        self.short_pos = 0.0
        self.long_avg = 0.0
        self.short_avg = 0.0
        self.active_orders = {}
        self.fills = []
        self.equity_rows = []
        self.state_rows = []
        self.cancelled_post_only = 0
        self.fee_paid = 0.0
        self.realized_pnl = 0.0
        self.peak_equity = args.balance
        self.max_dd = 0.0

    # ------------------------------------------------------------------
    def upnl(self, price):
        lp = (price - self.long_avg) * self.long_pos * self.C if self.long_pos > 0 else 0.0
        sp = (self.short_avg - price) * self.short_pos * self.C if self.short_pos > 0 else 0.0
        return lp + sp

    def equity(self, price):
        return self.balance + self.upnl(price)

    def margin_usage(self, price):
        eq = self.equity(price)
        if eq <= 0:
            return 1.0
        imr = (self.long_pos + self.short_pos) * self.C * price / self.lev
        return max(0.0, min(1.0, imr / eq))

    # ------------------------------------------------------------------
    def _apply_fill(self, o, fill_price, ts_ms):
        """成交后的持仓/资金/账本更新"""
        qty = float(o["qty"])
        side = o["side"]
        is_reduce = o.get("reduceOnly", False)
        fee = qty * self.C * fill_price * self.args.maker_fee
        self.balance -= fee
        self.fee_paid += fee

        if side == "buy":
            if is_reduce:
                closed = min(qty, self.short_pos)
                realized = (self.short_avg - fill_price) * closed * self.C
                self.balance += realized
                self.realized_pnl += realized
                self.short_pos = max(0.0, self.short_pos - qty)
                if self.short_pos <= 1e-9:
                    self.short_avg = 0.0
            else:
                new_pos = self.long_pos + qty
                self.long_avg = (self.long_avg * self.long_pos + fill_price * qty) / new_pos if new_pos > 0 else 0.0
                self.long_pos = new_pos
        else:
            if is_reduce:
                closed = min(qty, self.long_pos)
                realized = (fill_price - self.long_avg) * closed * self.C
                self.balance += realized
                self.realized_pnl += realized
                self.long_pos = max(0.0, self.long_pos - qty)
                if self.long_pos <= 1e-9:
                    self.long_avg = 0.0
            else:
                new_pos = self.short_pos + qty
                self.short_avg = (self.short_avg * self.short_pos + fill_price * qty) / new_pos if new_pos > 0 else 0.0
                self.short_pos = new_pos

        cid = o["orderLinkId"]
        self.active_orders.pop(cid, None)
        ev = {**o, "clOrdId": cid, "status": "filled", "fill_price": fill_price, "fee": -fee}
        self.tracker.upsert_order(ev)
        self.fills.append({
            "ts": ts_ms, "side": side, "price": fill_price, "qty": qty, "fee": fee,
            "reduceOnly": is_reduce, "is_hedge": o.get("is_hedge", False),
            "reason": o.get("reason", ""), "groupId": o.get("groupId", ""),
            "balance": self.balance,
        })
        # 消消乐撮合 (与实盘一致: 成交即触发)
        self.tracker.check_and_execute_netting()

    def process_fills(self, bar_high, bar_low, bar_open, ts_ms):
        """逐单检查成交; post_only 开盘穿越则按 OKX 规则撤单"""
        for cid, o in list(self.active_orders.items()):
            price = float(o["price"])
            if o["side"] == "buy":
                if bar_open <= price:
                    # 开盘即穿越 -> post_only 会被交易所拒单/撤销
                    self.active_orders.pop(cid, None)
                    self.tracker.upsert_order({**o, "clOrdId": cid, "status": "canceled"})
                    self.cancelled_post_only += 1
                elif bar_low <= price:
                    self._apply_fill(o, price, ts_ms)
            else:
                if bar_open >= price:
                    self.active_orders.pop(cid, None)
                    self.tracker.upsert_order({**o, "clOrdId": cid, "status": "canceled"})
                    self.cancelled_post_only += 1
                elif bar_high >= price:
                    self._apply_fill(o, price, ts_ms)

    # ------------------------------------------------------------------
    def drive_decision(self, price, ask, bid, ts_ms):
        """复现 main._drive_grid_decision 的过滤管线 + state_machine.tick"""
        S = self.settings
        now = SimClock.t

        # 1. 偏离清理 (>1% 且存活 >5s 的非对冲非减仓单)
        dev_thr = price * S.DEVIATION_THRESHOLD_PERCENT
        for cid, o in list(self.active_orders.items()):
            if not o.get("is_hedge") and not o.get("reduceOnly"):
                if now - float(o.get("place_time", 0.0)) > 5.0:
                    if abs(float(o["price"]) - price) > dev_thr:
                        self.active_orders.pop(cid, None)
                        self.tracker.upsert_order({**o, "clOrdId": cid, "status": "canceled"})

        # 2. 注入公积金 + 状态机决策
        self.sm.order_generator.unstuck_budget = self.tracker.get_unstuck_balance()
        net = self.long_pos - self.short_pos
        safe_orders, ctx = self.sm.tick(
            current_price=price,
            net_exposure=net,
            margin_usage=self.margin_usage(price),
            active_orders=list(self.active_orders.values()),
            unmerged_orders=self.tracker.unmerged_orders,
            current_ask=ask,
            current_bid=bid,
            actual_long_pos=self.long_pos,
            actual_short_pos=self.short_pos,
        )
        self.state_rows.append((ts_ms, ctx.system_state, ctx.trend, net, self.long_pos, self.short_pos))
        if not safe_orders:
            return

        # 3. main 里的过滤管线: 组排他 / 对冲单飞 / 侧别硬顶 / 价差去重
        active_open_buys = [o for o in self.active_orders.values() if o["side"] == "buy" and not o.get("reduceOnly") and not o.get("is_hedge")]
        active_open_sells = [o for o in self.active_orders.values() if o["side"] == "sell" and not o.get("reduceOnly") and not o.get("is_hedge")]
        active_hedges = [o for o in self.active_orders.values() if o.get("is_hedge")]
        gap_tol = price * (S.BASE_GRID_SPACING_PERCENT * 0.4)

        for new_o in safe_orders:
            side, p = new_o["side"], float(new_o["price"])
            is_reduce, is_hedge = new_o.get("reduceOnly", False), new_o.get("is_hedge", False)
            gid = new_o.get("groupId", "")

            if any(a.get("groupId") == gid and a["side"] == side and a.get("reduceOnly", False) == is_reduce
                   for a in self.active_orders.values()):
                continue

            if is_hedge:
                in_flight = sum(float(h["qty"]) for h in active_hedges if h["side"] == side)
                if in_flight >= 1.0:
                    continue
            elif is_reduce:
                if any(a.get("reduceOnly") and a["side"] == side and
                       ((gid and a.get("groupId") == gid) or abs(float(a["price"]) - p) < gap_tol)
                       for a in self.active_orders.values()):
                    continue
            elif side == "buy":
                if len(active_open_buys) >= S.MAX_ACTIVE_ORDERS_PER_SIDE:
                    continue
                if any(abs(float(b["price"]) - p) < gap_tol for b in active_open_buys):
                    continue
                active_open_buys.append(new_o)
            else:
                if len(active_open_sells) >= S.MAX_ACTIVE_ORDERS_PER_SIDE:
                    continue
                if any(abs(float(s["price"]) - p) < gap_tol for s in active_open_sells):
                    continue
                active_open_sells.append(new_o)

            cid = new_o["orderLinkId"][:32]
            placed = {**new_o, "orderLinkId": cid, "place_time": now, "status": "open"}
            self.active_orders[cid] = placed
            self.tracker.upsert_order({**placed, "clOrdId": cid})

    # ------------------------------------------------------------------
    def run(self, ts_all, bars, start_ms, end_ms):
        agg = CandleAggregator15m(self.sm.market_sense.update_candle)
        warmup = 0
        n = len(ts_all)
        for i in range(n):
            ts = int(ts_all[i])
            high, low, close, _vol = bars[i]
            SimClock.t = ts / 1000.0

            if ts < start_ms:
                agg.push(ts, high, low, close)  # 预热 15m 感知
                warmup += 1
                continue
            if ts >= end_ms:
                break

            # 1) 先撮合已有挂单
            self.process_fills(high, low, bars[i - 1][2] if i > 0 else close, ts)
            # 2) 更新 15m 感知
            agg.push(ts, high, low, close)
            # 3) 决策 (盘口用上一根收盘构造, 避免用当根未来数据)
            prev_close = bars[i - 1][2] if i > 0 else close
            bid = prev_close * (1 - self.args.half_spread)
            ask = prev_close * (1 + self.args.half_spread)
            self.drive_decision(prev_close, ask, bid, ts)

            # 4) 记录权益
            eq = self.equity(close)
            self.peak_equity = max(self.peak_equity, eq)
            dd = (self.peak_equity - eq) / self.peak_equity if self.peak_equity > 0 else 0.0
            self.max_dd = max(self.max_dd, dd)
            self.equity_rows.append((ts, self.balance, eq, close, self.long_pos, self.short_pos))

        agg.flush()
        print(f"[replay] 预热 {warmup} 根 1m K 线, 回测 {len(self.equity_rows)} 根")

    # ------------------------------------------------------------------
    def report(self):
        import csv
        eq = np.array(self.equity_rows, dtype=float)
        out = self.out_dir

        with open(os.path.join(out, "fills.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "side", "price", "qty", "fee", "reduceOnly", "is_hedge", "reason", "groupId", "balance_after"])
            for r in self.fills:
                w.writerow([datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            r["side"], f"{r['price']:.2f}", r["qty"], f"{r['fee']:.6f}",
                            r["reduceOnly"], r["is_hedge"], r["reason"], r["groupId"], f"{r['balance']:.2f}"])

        with open(os.path.join(out, "equity.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "balance", "equity", "btc_price", "long_pos", "short_pos"])
            for row in self.equity_rows:
                w.writerow([datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            f"{row[1]:.2f}", f"{row[2]:.2f}", f"{row[3]:.2f}", row[4], row[5]])

        # 从 V5 账本读取消消乐/债务/公积金结果
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            merged = cur.execute("SELECT COUNT(*), COALESCE(SUM(net_profit),0), COALESCE(SUM(profit),0), COALESCE(SUM(total_fee),0) FROM merged_history").fetchone()
        except Exception:
            merged = (0, 0.0, 0.0, 0.0)
        try:
            debts = cur.execute("SELECT COUNT(*), COALESCE(SUM(loss_amount),0), COALESCE(SUM(recovered_amount),0) FROM deadlock_breaker_records").fetchone()
        except Exception:
            debts = (0, 0.0, 0.0)
        try:
            pool = cur.execute("SELECT balance, total_deposited, total_spent FROM unstuck_pool WHERE id=1").fetchone() or (0.0, 0.0, 0.0)
        except Exception:
            pool = (0.0, 0.0, 0.0)
        conn.close()

        final_eq = eq[-1, 2] if len(eq) else self.args.balance
        start_eq = self.args.balance
        gain = final_eq - start_eq
        days = (eq[-1, 0] - eq[0, 0]) / 86_400_000 if len(eq) > 1 else 1

        from collections import Counter
        reason_cnt = Counter(f["reason"].split("-")[0] if f["reason"] else "?" for f in self.fills)

        summary = {
            "初始资金": start_eq,
            "期末权益": round(final_eq, 2),
            "净盈亏": round(gain, 2),
            "收益率%": round(gain / start_eq * 100, 4),
            "年化%": round(((final_eq / start_eq) ** (365 / max(days, 1)) - 1) * 100, 2) if final_eq > 0 else None,
            "最大回撤%": round(self.max_dd * 100, 4),
            "成交笔数": len(self.fills),
            "post_only被撤(开盘穿越)": self.cancelled_post_only,
            "手续费合计": round(self.fee_paid, 2),
            "消消乐配对数": merged[0],
            "消消乐净利合计": round(merged[1], 4),
            "债务建账笔数": debts[0],
            "债务总额": round(debts[1], 4),
            "已追偿": round(debts[2], 4),
            "公积金余额": round(pool[0], 4),
            "期末多头(张)": self.long_pos,
            "期末空头(张)": self.short_pos,
            "成交原因分布": dict(reason_cnt),
        }
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 画权益曲线
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            t = [datetime.fromtimestamp(x / 1000, tz=timezone.utc) for x in eq[:, 0]]
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1, 1]})
            axes[0].plot(t, eq[:, 2], lw=1.0, color="tab:blue", label="equity")
            axes[0].plot(t, eq[:, 1], lw=0.8, color="tab:green", alpha=0.7, label="realized balance")
            axes[0].axhline(start_eq, color="gray", ls="--", lw=0.6)
            axes[0].set_title(f"V5 replay | net {gain:+.2f} USDT ({gain / start_eq * 100:+.3f}%) | maxDD {self.max_dd * 100:.3f}%")
            axes[0].legend(); axes[0].grid(alpha=0.3)
            axes[1].plot(t, eq[:, 4], lw=0.8, color="tab:green", label="long pos")
            axes[1].plot(t, -eq[:, 5], lw=0.8, color="tab:red", label="-short pos")
            axes[1].set_ylabel("position (contracts)"); axes[1].legend(); axes[1].grid(alpha=0.3)
            axes[2].plot(t, eq[:, 3], lw=0.8, color="black")
            axes[2].set_ylabel("BTC price"); axes[2].grid(alpha=0.3)
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            fig.tight_layout()
            fig.savefig(os.path.join(out, "equity.png"), dpi=110)
            plt.close(fig)
        except Exception as e:
            print(f"[plot] 画图失败: {e}")

        print("\n" + "=" * 60)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("=" * 60)
        print(f"输出目录: {out}")


def main():
    args = parse_args()
    inject_env(args.env)

    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    ds, _ts = find_dataset(start_ms, end_ms, args.data)
    print(f"[data] 使用数据集: {ds}")
    ts_all, bars = load_dataset(ds)

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                f"{args.start}_{args.end}")

    # 导入 V5 (必须在环境变量注入之后)
    sys.path.insert(0, V5_PROJECT_ROOT)
    from src.v5.config import settings as v5_settings
    print(f"[v5] 已加载 V5 配置: SYMBOL={v5_settings.SYMBOL} LEVERAGE={v5_settings.LEVERAGE} "
          f"BASE_QTY={v5_settings.BASE_ORDER_QTY} GRID={v5_settings.BASE_GRID_SPACING_PERCENT} "
          f"MAX_NET_EXP={v5_settings.MAX_NET_EXPOSURE} HEDGE_TRIGGER={v5_settings.HEDGE_TRIGGER_RATIO}")

    replay = V5Replay(args, v5_settings)
    replay.run(ts_all, bars, start_ms, end_ms)
    replay.report()


if __name__ == "__main__":
    main()
