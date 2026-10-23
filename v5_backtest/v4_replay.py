"""
V4 策略回放回测器
==================
与 v5_replay.py 同架构: 用 Passivbot HLCV 缓存驱动 F:\\oneProject\\src\\v4 的原版代码。

V4 与 V5 的结构差异 (回放器已适配):
- 无 MarketSenseUnit / 15m K线感知 (V4 不看趋势, 纯敞口状态机)
- 状态机四态: NORMAL / SKEW / DEFENSE / ALERT
- 破局 (deadlock breaker): 30 分钟无成交 -> 近端平仓 + 远端移仓开仓
- 账本 API: record_new_order / update_order_status / execute_merge / get_unmerged_orders

模拟简化 (与 v5_replay 相同):
1. 限价单在 1m K 线 high/low 穿越挂单价时按挂单价成交 (maker)
2. post_only 开盘穿越 -> 撤单
3. 每根 1m K 线决策一次
4. bid/ask = 上一根收盘 ∓ half_spread
5. 破局单的二档盘口 (ask2/bid2) 用 ∓2×half_spread 近似
6. 破局重试>=3次的市价降级单: 按当根收盘价成交

用法:
  python v5_backtest\\v4_replay.py --start 2026-09-01 --end 2026-10-01
  python v5_backtest\\v4_replay.py --start 2026-09-01 --end 2026-10-01 -e GRID_SPACING_PERCENT=0.002
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
V4_PROJECT_ROOT = os.environ.get("V4_PROJECT_ROOT", r"F:\oneProject")
HLCV_ROOT = os.path.join(PB_ROOT, "caches", "hlcvs_data")


class SimClock:
    t = 0.0


_time_module.time = lambda: SimClock.t


def parse_args():
    ap = argparse.ArgumentParser(description="V4 replay backtester")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--balance", type=float, default=10000.0)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0005)
    ap.add_argument("--half-spread", type=float, default=0.0001)
    ap.add_argument("--out", default=None)
    ap.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE")
    return ap.parse_args()


def inject_env(overrides):
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"-e 参数格式错误: {item}")
        k, v = item.split("=", 1)
        os.environ[k.strip()] = v.strip()
        print(f"[env] 覆盖 V4 配置: {k.strip()} = {v.strip()}")


def find_dataset(start_ms, end_ms, explicit=None):
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
            return c
    raise SystemExit("找不到覆盖该区间的 HLCV 数据集。先跑 passivbot backtest 生成缓存。")


def load_dataset(ds_dir):
    ts = np.load(gzip.open(os.path.join(ds_dir, "timestamps.npy.gz")))
    hlcvs = np.load(gzip.open(os.path.join(ds_dir, "hlcvs.npy.gz")))
    return ts, hlcvs[:, 0, :]


class V4Replay:
    def __init__(self, args, settings):
        self.args = args
        self.S = settings
        self.C = settings.CONTRACT_VALUE
        self.lev = float(settings.LEVERAGE)

        from src.v4.engine.state_machine import V4StateMachine
        from src.v4.tracking.order_tracker import OrderTracker
        import logging
        for name in ("OrderGenerator", "RiskManager", "OrderTracker"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        try:
            from src.utils.log_config import v4_logger
            v4_logger.setLevel(logging.CRITICAL)
        except Exception:
            pass

        os.makedirs(args.out, exist_ok=True)
        self.out_dir = args.out
        self.db_path = os.path.join(args.out, "v4_replay.db")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        self.sm = V4StateMachine(symbol=settings.SYMBOL)
        self.tracker = OrderTracker(db_path=self.db_path)

        # 账户状态 (与实盘一致: 底仓对策略隐藏)
        self.balance = args.balance
        self.long_pos = 0.0      # 策略有效持仓 (不含底仓)
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

        # 破局状态 (复现 main 的成员)
        self.last_new_order_time = SimClock.t
        self.deadlock_breaker_active = False
        self.breaker_retry_count = 0
        self.current_breaker_close_id = ""
        self.current_breaker_open_id = ""
        self._price_history = []
        self.breaker_events = 0

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
    def _apply_fill(self, o, fill_price, ts_ms, taker=False):
        qty = float(o["qty"])
        side = o["side"]
        is_reduce = o.get("reduceOnly", False)
        rate = self.args.taker_fee if taker else self.args.maker_fee
        fee = qty * self.C * fill_price * rate
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
        # V4 账本路径: record_new_order 已在挂单时调用, 成交走 update_order_status
        self.tracker.update_order_status(cid, "filled", fill_price, qty, -fee)
        # 成交即重置破局计时器 (与实盘一致)
        self.last_new_order_time = SimClock.t
        self.fills.append({
            "ts": ts_ms, "side": side, "price": fill_price, "qty": qty, "fee": fee,
            "reduceOnly": is_reduce, "taker": taker,
            "reason": o.get("reason", ""), "groupId": o.get("groupId", ""),
            "balance": self.balance,
        })
        # 消消乐: V4 在平仓单 (v4r*) 成交时找同组开仓单合并
        if "V4R" in cid.upper():
            db_order = self.tracker.get_order_by_clOrdId(cid)
            if db_order:
                group_label = db_order.get("group_label", "")
                group_id = group_label.split("-")[-1] if group_label else ""
                os_side = db_order["side"]
                opp = "sell" if os_side == "buy" else "buy"
                targets = [x for x in self.tracker.get_unmerged_orders()
                           if x["side"] == opp and x["status"] == "filled"
                           and group_id and group_id in x.get("group_label", "")]
                if targets:
                    if os_side == "buy":
                        self.tracker.execute_merge([cid], [targets[0]["clOrdId"]], f"V4-PROFIT-{group_id}")
                    else:
                        self.tracker.execute_merge([targets[0]["clOrdId"]], [cid], f"V4-PROFIT-{group_id}")

    def process_fills(self, bar_high, bar_low, bar_open, bar_close, ts_ms):
        for cid, o in list(self.active_orders.items()):
            price = float(o["price"])
            is_market = o.get("orderType") == "market"
            if is_market:
                # 市价降级单: 按当根收盘价成交 (taker)
                self._apply_fill(o, bar_close, ts_ms, taker=True)
                continue
            if o["side"] == "buy":
                if bar_open <= price:
                    self.active_orders.pop(cid, None)
                    self.tracker.update_order_status(cid, "canceled")
                    self.cancelled_post_only += 1
                elif bar_low <= price:
                    self._apply_fill(o, price, ts_ms)
            else:
                if bar_open >= price:
                    self.active_orders.pop(cid, None)
                    self.tracker.update_order_status(cid, "canceled")
                    self.cancelled_post_only += 1
                elif bar_high >= price:
                    self._apply_fill(o, price, ts_ms)

    # ------------------------------------------------------------------
    def _place_orders(self, orders, ts_ms):
        """V4 发单路径: 组排他过滤 -> record_new_order -> active_orders"""
        now = SimClock.t
        for o in orders:
            gid = o.get("groupId", "")
            side = o["side"]
            is_reduce = o.get("reduceOnly", False)
            dup = any(a.get("groupId") == gid and a["side"] == side
                      and a.get("reduceOnly", False) == is_reduce
                      for a in self.active_orders.values())
            if dup:
                continue
            cid = o["orderLinkId"]
            self.tracker.record_new_order(cid, side, o["qty"], o["price"], group_label=f"V4-G-{gid}")
            placed = {**o, "place_time": now, "status": "open"}
            self.active_orders[cid] = placed

    def check_and_break_deadlock(self, ask, bid, ts_ms):
        """忠实复现 main._check_and_break_deadlock 的核心状态机"""
        S = self.S
        net = self.net_exposure()

        # 1. 破局活跃中的重试与成交检查
        if self.deadlock_breaker_active:
            close_active = self.current_breaker_close_id in self.active_orders
            if not close_active:
                db_order = self.tracker.get_order_by_clOrdId(self.current_breaker_close_id)
                is_filled = db_order and db_order.get("status") in ("filled", "merged")
                if is_filled:
                    self.deadlock_breaker_active = False
                    self.breaker_retry_count = 0
                    self.tracker.set_system_flag("deadlock_breaker_active", "0")
                    self.current_breaker_open_id = ""
                else:
                    if self.breaker_retry_count < 5:
                        self.breaker_retry_count += 1
                        if self.current_breaker_open_id in self.active_orders:
                            self.active_orders.pop(self.current_breaker_open_id, None)
                            self.tracker.update_order_status(self.current_breaker_open_id, "canceled")
                        self.current_breaker_open_id = ""
                    else:
                        if self.current_breaker_open_id in self.active_orders:
                            self.active_orders.pop(self.current_breaker_open_id, None)
                            self.tracker.update_order_status(self.current_breaker_open_id, "canceled")
                        self.current_breaker_open_id = ""
                        self.deadlock_breaker_active = False
                        self.breaker_retry_count = 0
                        self.tracker.set_system_flag("deadlock_breaker_active", "0")
                        self.last_new_order_time = SimClock.t
                        return
            else:
                return

        # 2. 超时检测
        if not self.deadlock_breaker_active:
            if SimClock.t - self.last_new_order_time < self.deadlock_timeout:
                return
            self.breaker_retry_count = 0

        if ask <= 0 or bid <= 0:
            return
        mid = (ask + bid) / 2.0

        # 3. 单边动量保护 (3 分钟前价格对比)
        if self._price_history:
            old_prices = [p for t, p in self._price_history if SimClock.t - t >= 180]
            if old_prices:
                old_p = old_prices[0]
                chg = (mid - old_p) / old_p if old_p > 0 else 0.0
                if net < 0 and chg > 0.005:
                    self.last_new_order_time = SimClock.t - self.deadlock_timeout + 60
                    return
                elif net > 0 and chg < -0.005:
                    self.last_new_order_time = SimClock.t - self.deadlock_timeout + 60
                    return

        # 4. 构造破局单
        gap = mid * S.BASE_GRID_SPACING_PERCENT
        buy_price = round(mid - gap, S.PRICE_PRECISION)
        sell_price = round(mid + gap, S.PRICE_PRECISION)
        tick = 10 ** -S.PRICE_PRECISION
        ask2 = round(ask + 2 * tick * 10 if S.PRICE_PRECISION <= 2 else ask * (1 + 2 * self.args.half_spread), S.PRICE_PRECISION)
        bid2 = round(bid - 2 * tick * 10 if S.PRICE_PRECISION <= 2 else bid * (1 - 2 * self.args.half_spread), S.PRICE_PRECISION)
        if ask2 <= ask:
            ask2 = round(ask * (1 + 2 * self.args.half_spread), S.PRICE_PRECISION)
        if bid2 >= bid:
            bid2 = round(bid * (1 - 2 * self.args.half_spread), S.PRICE_PRECISION)

        import uuid as _uuid
        ts = str(int(SimClock.t * 1000))
        break_gid = str(_uuid.uuid4()).replace("-", "")[:4]

        unmerged = self.tracker.get_unmerged_orders()
        open_buys = [o for o in unmerged if o["side"] == "buy" and o["status"] == "filled"]
        open_sells = [o for o in unmerged if o["side"] == "sell" and o["status"] == "filled"]

        breaker_orders = []
        close_id = ""
        open_id = ""
        use_market = self.deadlock_breaker_active and self.breaker_retry_count >= 3

        if net > 0 and open_buys:
            target_buy = min(open_buys, key=lambda x: abs(x["filled_price"] - mid))
            tgid = target_buy["group_label"].split("-")[-1] if target_buy.get("group_label") else break_gid
            sell_link_id = f"v4rsk{tgid}{ts}b"
            if use_market:
                breaker_orders.append({"symbol": S.SYMBOL, "side": "sell", "orderType": "market",
                                       "qty": target_buy["qty"], "price": 0, "orderLinkId": sell_link_id,
                                       "reduceOnly": True, "groupId": tgid,
                                       "reason": f"V4-BREAK-REDUCE-LONG-MKT-{tgid}"})
            else:
                breaker_orders.append({"symbol": S.SYMBOL, "side": "sell", "orderType": "post_only",
                                       "qty": target_buy["qty"], "price": ask2, "orderLinkId": sell_link_id,
                                       "reduceOnly": True, "groupId": tgid,
                                       "reason": f"V4-BREAK-REDUCE-LONG-{tgid}"})
            close_id = sell_link_id
            buy_link_id = f"v4obk{break_gid}{ts}a"
            breaker_orders.append({"symbol": S.SYMBOL, "side": "buy", "orderType": "post_only",
                                   "qty": target_buy["qty"], "price": buy_price, "orderLinkId": buy_link_id,
                                   "reduceOnly": False, "groupId": break_gid,
                                   "reason": f"V4-BREAK-OPEN-{break_gid}"})
            open_id = buy_link_id
        elif net < 0 and open_sells:
            target_sell = min(open_sells, key=lambda x: abs(x["filled_price"] - mid))
            tgid = target_sell["group_label"].split("-")[-1] if target_sell.get("group_label") else break_gid
            buy_link_id = f"v4rbk{tgid}{ts}a"
            if use_market:
                breaker_orders.append({"symbol": S.SYMBOL, "side": "buy", "orderType": "market",
                                       "qty": target_sell["qty"], "price": 0, "orderLinkId": buy_link_id,
                                       "reduceOnly": True, "groupId": tgid,
                                       "reason": f"V4-BREAK-REDUCE-SHORT-MKT-{tgid}"})
            else:
                breaker_orders.append({"symbol": S.SYMBOL, "side": "buy", "orderType": "post_only",
                                       "qty": target_sell["qty"], "price": bid2, "orderLinkId": buy_link_id,
                                       "reduceOnly": True, "groupId": tgid,
                                       "reason": f"V4-BREAK-REDUCE-SHORT-{tgid}"})
            close_id = buy_link_id
            sell_link_id = f"v4obk{break_gid}{ts}b"
            breaker_orders.append({"symbol": S.SYMBOL, "side": "sell", "orderType": "post_only",
                                   "qty": target_sell["qty"], "price": sell_price, "orderLinkId": sell_link_id,
                                   "reduceOnly": False, "groupId": break_gid,
                                   "reason": f"V4-BREAK-OPEN-{break_gid}"})
            open_id = sell_link_id
        elif net == 0 and (open_buys or open_sells):
            max_buy_loss = max((abs(o["filled_price"] - mid) for o in open_buys), default=0)
            max_sell_loss = max((abs(o["filled_price"] - mid) for o in open_sells), default=0)
            if open_buys and max_buy_loss >= max_sell_loss:
                tb = max(open_buys, key=lambda x: abs(x["filled_price"] - mid))
                tgid = tb["group_label"].split("-")[-1] if tb.get("group_label") else break_gid
                sl = f"v4rsk{tgid}{ts}b"
                breaker_orders.append({"symbol": S.SYMBOL, "side": "sell", "orderType": "post_only",
                                       "qty": tb["qty"], "price": ask2, "orderLinkId": sl,
                                       "reduceOnly": True, "groupId": tgid,
                                       "reason": f"V4-BREAK-NETLOCKED-LONG-{tgid}"})
                close_id = sl
            elif open_sells:
                tsell = max(open_sells, key=lambda x: abs(x["filled_price"] - mid))
                tgid = tsell["group_label"].split("-")[-1] if tsell.get("group_label") else break_gid
                bl = f"v4rbk{tgid}{ts}a"
                breaker_orders.append({"symbol": S.SYMBOL, "side": "buy", "orderType": "post_only",
                                       "qty": tsell["qty"], "price": bid2, "orderLinkId": bl,
                                       "reduceOnly": True, "groupId": tgid,
                                       "reason": f"V4-BREAK-NETLOCKED-SHORT-{tgid}"})
                close_id = bl

        if not breaker_orders:
            self.last_new_order_time = SimClock.t
            return

        # 5. [Atomic Quota Swap] 撤销同向旧平仓单释放配额 (复现 main 步骤5)
        target_reduce_side = "buy" if net < 0 else "sell"
        for clid, o in list(self.active_orders.items()):
            if o.get("reduceOnly") and o.get("side") == target_reduce_side and clid != close_id:
                self.active_orders.pop(clid, None)
                self.tracker.update_order_status(clid, "canceled")

        # 6. 风控
        safe = self.sm.risk_manager.process_orders(
            current_net_pos=net,
            active_orders=list(self.active_orders.values()),
            new_orders=breaker_orders,
            current_price=mid,
            grid_spacing=S.BASE_GRID_SPACING_PERCENT,
            filled_buy_count=len(open_buys),
            filled_sell_count=len(open_sells),
            actual_long_pos=self.long_pos + S.BASE_LONG_QTY,
            actual_short_pos=self.short_pos + S.BASE_SHORT_QTY,
        )
        if not safe:
            return

        self.breaker_events += 1
        self.current_breaker_close_id = close_id
        self.current_breaker_open_id = open_id
        self._place_orders(safe, ts_ms)
        self.deadlock_breaker_active = True
        self.last_new_order_time = SimClock.t
        self.tracker.set_system_flag("deadlock_breaker_active", "1")

    # ------------------------------------------------------------------
    def net_exposure(self):
        return self.long_pos - self.short_pos

    def drive_decision(self, price, ask, bid, ts_ms):
        """复现 main._execute_strategy_cycle"""
        S = self.S
        now = SimClock.t
        mid = (ask + bid) / 2.0

        # 价格历史 (动量保护用)
        self._price_history.append((now, mid))
        self._price_history = [(t, p) for t, p in self._price_history if now - t <= 300.0]

        # 偏离清理 (>1% 且存活 >2s, 豁免破局单; 开仓单 >30 分钟未成交也清理)
        if S.ENABLE_DEVIATION_CLEANUP:
            dev_thr = mid * S.DEVIATION_THRESHOLD_PERCENT
            for cid, o in list(self.active_orders.items()):
                if o.get("status") == "canceling":
                    continue
                if any(p in cid for p in ("v4obk", "v4rbk", "v4rsk")):
                    continue
                if now - float(o.get("place_time", 0)) > 2.0:
                    far = abs(float(o["price"]) - mid) > dev_thr
                    stale_open = (not o.get("reduceOnly")) and (now - float(o.get("place_time", 0)) > 1800.0)
                    if far or stale_open:
                        self.active_orders.pop(cid, None)
                        self.tracker.update_order_status(cid, "canceled")

        # 状态机决策
        unmerged = self.tracker.get_unmerged_orders()
        safe_orders, action_flag = self.sm.tick(
            current_price=mid,
            net_exposure=self.net_exposure(),
            margin_usage=self.margin_usage(price),
            active_orders=list(self.active_orders.values()),
            long_pos=self.long_pos,
            short_pos=self.short_pos,
            unmerged_orders=unmerged,
            current_ask=ask,
            current_bid=bid,
            actual_long_pos=self.long_pos + S.BASE_LONG_QTY,
            actual_short_pos=self.short_pos + S.BASE_SHORT_QTY,
        )
        self.state_rows.append((ts_ms, self.sm.current_state, self.net_exposure(), self.long_pos, self.short_pos))

        if action_flag == "EXECUTE_NETTING":
            # ALERT 态: 最远端一多一空直接账本合并 (V4 原逻辑即记账模式)
            buys = [o for o in unmerged if o["side"] == "buy" and o["status"] == "filled"]
            sells = [o for o in unmerged if o["side"] == "sell" and o["status"] == "filled"]
            if buys and sells:
                buys.sort(key=lambda x: abs(x["filled_price"] - mid), reverse=True)
                sells.sort(key=lambda x: abs(x["filled_price"] - mid), reverse=True)
                self.tracker.execute_merge([buys[0]["clOrdId"]], [sells[0]["clOrdId"]], "V4-EMERGENCY-NETTING")
            return

        self._place_orders(safe_orders, ts_ms)

    # ------------------------------------------------------------------
    def run(self, ts_all, bars, start_ms, end_ms):
        warmup = 0
        n = len(ts_all)
        for i in range(n):
            ts = int(ts_all[i])
            high, low, close, _vol = bars[i]
            SimClock.t = ts / 1000.0

            if ts < start_ms:
                warmup += 1
                continue
            if ts >= end_ms:
                break

            prev_close = bars[i - 1][2] if i > 0 else close
            bid = prev_close * (1 - self.args.half_spread)
            ask = prev_close * (1 + self.args.half_spread)

            # 1) 撮合
            self.process_fills(high, low, bars[i - 1][2] if i > 0 else close, close, ts)
            # 2) 常规网格决策
            self.drive_decision(prev_close, ask, bid, ts)
            # 3) 破局检查
            self.check_and_break_deadlock(ask, bid, ts)

            eq = self.equity(close)
            self.peak_equity = max(self.peak_equity, eq)
            dd = (self.peak_equity - eq) / self.peak_equity if self.peak_equity > 0 else 0.0
            self.max_dd = max(self.max_dd, dd)
            self.equity_rows.append((ts, self.balance, eq, close, self.long_pos, self.short_pos))

        print(f"[replay] 跳过预热 {warmup} 根, 回测 {len(self.equity_rows)} 根 1m K 线")

    # ------------------------------------------------------------------
    def report(self):
        import csv
        eq = np.array(self.equity_rows, dtype=float)
        out = self.out_dir

        with open(os.path.join(out, "fills.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "side", "price", "qty", "fee", "reduceOnly", "taker", "reason", "groupId", "balance_after"])
            for r in self.fills:
                w.writerow([datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            r["side"], f"{r['price']:.2f}", r["qty"], f"{r['fee']:.6f}",
                            r["reduceOnly"], r["taker"], r["reason"], r["groupId"], f"{r['balance']:.2f}"])

        with open(os.path.join(out, "equity.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "balance", "equity", "btc_price", "long_pos", "short_pos"])
            for row in self.equity_rows:
                w.writerow([datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            f"{row[1]:.2f}", f"{row[2]:.2f}", f"{row[3]:.2f}", row[4], row[5]])

        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            merged = cur.execute("SELECT COUNT(*), COALESCE(SUM(profit),0) FROM merge_history").fetchone()
        except Exception:
            try:
                tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                merged = (0, 0.0)
                for t in tables:
                    if "merge" in t.lower():
                        merged = cur.execute(f"SELECT COUNT(*), 0 FROM {t}").fetchone()
                        break
            except Exception:
                merged = (0, 0.0)
        conn.close()

        final_eq = eq[-1, 2] if len(eq) else self.args.balance
        gain = final_eq - self.args.balance
        days = (eq[-1, 0] - eq[0, 0]) / 86_400_000 if len(eq) > 1 else 1

        from collections import Counter
        reason_cnt = Counter()
        for f in self.fills:
            r = f["reason"]
            key = r.split("-")[0] if r else "?"
            if "BREAK" in r:
                key = "BREAK破局"
            elif "止盈" in r:
                key = "止盈平仓"
            elif "加仓" in r or "对称" in r:
                key = "网格开仓"
            elif "防守" in r:
                key = "紧急防守"
            reason_cnt[key] += 1

        state_cnt = Counter(s[1] for s in self.state_rows)

        summary = {
            "初始资金": self.args.balance,
            "期末权益": round(final_eq, 2),
            "净盈亏": round(gain, 2),
            "收益率%": round(gain / self.args.balance * 100, 4),
            "年化%": round(((final_eq / self.args.balance) ** (365 / max(days, 1)) - 1) * 100, 2) if final_eq > 0 else None,
            "最大回撤%": round(self.max_dd * 100, 4),
            "成交笔数": len(self.fills),
            "post_only被撤(开盘穿越)": self.cancelled_post_only,
            "手续费合计": round(self.fee_paid, 2),
            "破局触发次数": self.breaker_events,
            "账本合并次数": merged[0],
            "期末多头(张)": self.long_pos,
            "期末空头(张)": self.short_pos,
            "状态分布": dict(state_cnt),
            "成交原因分布": dict(reason_cnt),
        }
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

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
            axes[0].axhline(self.args.balance, color="gray", ls="--", lw=0.6)
            axes[0].set_title(f"V4 replay | net {gain:+.2f} USDT ({gain / self.args.balance * 100:+.3f}%) | maxDD {self.max_dd * 100:.3f}%")
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

    ds = find_dataset(start_ms, end_ms, args.data)
    print(f"[data] 使用数据集: {ds}")
    ts_all, bars = load_dataset(ds)

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                f"v4_{args.start}_{args.end}")

    sys.path.insert(0, V4_PROJECT_ROOT)
    from src.v4.config import settings as v4_settings
    print(f"[v4] 已加载 V4 配置: SYMBOL={v4_settings.SYMBOL} LEVERAGE={v4_settings.LEVERAGE} "
          f"BASE_QTY={v4_settings.BASE_ORDER_QTY} GRID={v4_settings.BASE_GRID_SPACING_PERCENT} "
          f"MAX_NET_EXP={v4_settings.MAX_NET_EXPOSURE} MAX_POS_SIDE={v4_settings.MAX_POSITION_PER_SIDE} "
          f"BASE_LONG_QTY={v4_settings.BASE_LONG_QTY}")

    replay = V4Replay(args, v4_settings)
    replay.deadlock_timeout = v4_settings.DEADLOCK_TIMEOUT_SECONDS
    replay.run(ts_all, bars, start_ms, end_ms)
    replay.report()


if __name__ == "__main__":
    main()
