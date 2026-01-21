#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KDJ超卖与短期趋势共振策略 (KDJ Oversold & Short-term Trend Resonance Strategy)
用户自定义策略回测脚本
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import os

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_stock_data(symbol, start_date, end_date):
    """
    使用AkShare获取股票数据并转换为标准格式
    """
    try:
        # 格式化日期为 YYYYMMDD
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # 获取前复权数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        
        if df.empty:
            return pd.DataFrame()
            
        # 重命名列
        df = df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume'
        })
        
        # 设置索引
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        return df
    except Exception as e:
        print(f"获取数据失败 {symbol}: {e}")
        return pd.DataFrame()

def calculate_kdj(df, n=9, m1=3, m2=3):
    """
    计算标准KDJ指标
    """
    df = df.copy()
    # 计算RSV
    low_min = df['Low'].rolling(window=n, min_periods=n).min()
    high_max = df['High'].rolling(window=n, min_periods=n).max()
    
    # 避免除以零
    denominator = high_max - low_min
    denominator = denominator.replace(0, np.nan) # Handle division by zero
    
    rsv = (df['Close'] - low_min) / denominator * 100
    rsv = rsv.fillna(50) # 初始值填充
    
    # 转换为numpy数组以提高循环效率
    rsv_values = rsv.values.flatten().astype(float)
    
    # 递归计算K, D
    k_values = []
    d_values = []
    
    k_prev = 50
    d_prev = 50
    
    for r in rsv_values:
        if np.isnan(r):
            k = k_prev
            d = d_prev
        else:
            k = (m1 - 1) / m1 * k_prev + 1 / m1 * r
            d = (m2 - 1) / m2 * d_prev + 1 / m2 * k
        
        k_values.append(k)
        d_values.append(d)
        k_prev = k
        d_prev = d
        
    df['K'] = k_values
    df['D'] = d_values
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

def calculate_indicators(df):
    """
    计算策略所需的所有指标
    """
    # 1. KDJ
    df = calculate_kdj(df)
    
    # 2. MA20
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 3. MA20斜率 (今日MA20 - 昨日MA20)
    df['MA20_Slope'] = df['MA20'].diff()
    
    return df

def run_strategy(df, initial_capital=100000):
    """
    执行策略逻辑
    """
    # 确保索引是时间序列
    df = df.copy()
    
    # 组合状态
    cash = initial_capital
    shares = 0
    position_pct = 0.0 # 0, 0.5, 1.0
    
    # 策略状态
    max_j_since_full = -999 # 满仓后记录J值最高点
    in_exit_phase = False # 是否处于退出阶段（已减仓）
    
    # 记录
    trades = []
    portfolio_value = []
    
    # 从第20天开始遍历（等待MA20计算完成）
    start_idx = 20
    
    for i in range(len(df)):
        current_date = df.index[i]
        price = df['Close'].iloc[i]
        
        # 记录组合价值（操作前）
        # 注意：这里记录的是当天的收盘价值，包含当天的价格波动
        # 但交易是在当天收盘价进行的（回测假设），所以操作后的状态反映了当天的结果
        
        if i < start_idx:
            portfolio_value.append(initial_capital)
            continue
            
        kdj_j = df['J'].iloc[i]
        ma20_slope = df['MA20_Slope'].iloc[i]
        
        action = None
        trade_info = None
        
        # --- 策略逻辑 ---
        
        # A. 买入规则 (建立底仓 50%)
        # 条件：空仓 且 J < 0 且 MA20_Slope > -0.02
        if position_pct == 0.0:
            if kdj_j < 0 and ma20_slope > -0.02:
                # 买入50%资金
                target_value = (cash + shares * price) * 0.5
                shares_to_buy = int(target_value / (price * 100)) * 100 # 按手取整
                
                if shares_to_buy > 0 and cash >= shares_to_buy * price:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares += shares_to_buy
                    position_pct = 0.5
                    in_exit_phase = False
                    action = "BUY_BASE"
                    trade_info = {
                        'Date': current_date, 'Action': '买入底仓', 'Price': price, 
                        'Shares': shares_to_buy, 'Amount': cost, 'J': kdj_j, 'Slope': ma20_slope
                    }
        
        # B. 加仓规则 (加至满仓 100%)
        # 条件：持有底仓 且 J > 40 且 未处于退出阶段
        elif position_pct == 0.5 and not in_exit_phase:
            if kdj_j > 40:
                # 用剩余资金买入
                shares_to_buy = int(cash / (price * 100)) * 100
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares += shares_to_buy
                    position_pct = 1.0
                    action = "ADD_FULL"
                    
                    # 初始化止盈状态
                    max_j_since_full = kdj_j
                    
                    trade_info = {
                        'Date': current_date, 'Action': '加仓至满', 'Price': price, 
                        'Shares': shares_to_buy, 'Amount': cost, 'J': kdj_j, 'Slope': ma20_slope
                    }
        
        # C. 卖出/止盈规则
        # 条件：持有仓位 (满仓 或 减仓后)
        elif position_pct > 0.0:
            # 只有当达到过满仓状态(J>40触发)后，才启用基于Max_J的止盈
            # 如果一直在底仓状态(0.5)，策略描述没有明确说何时卖出，但通常"加仓规则"隐含了趋势确认。
            # 如果J值一直没超过40怎么卖？
            # 策略原文："仅在持有仓位且 J 值曾冲高过 40 以后触发"。
            # 这意味着如果买入后J值从未超过40，可能一直持有直到亏损？
            # 这是一个策略漏洞，但我们严格按描述执行。
            # 不过，如果position_pct == 1.0，说明肯定触发过 J > 40。
            # 如果position_pct == 0.5 且 in_exit_phase == True，说明也触发过。
            # 唯独 position_pct == 0.5 且 in_exit_phase == False (刚买入底仓)，此时不触发止盈逻辑，直到J>40加仓。
            
            if position_pct == 1.0 or in_exit_phase:
                # 更新 Max_J
                if kdj_j > max_j_since_full:
                    max_j_since_full = kdj_j
                
                # 1. 减仓信号: J < Max_J * 0.8 (仅限满仓时)
                if position_pct == 1.0:
                    if kdj_j < max_j_since_full * 0.8:
                        # 卖出50%当前持仓
                        shares_to_sell = int((shares / 2) / 100) * 100
                        if shares_to_sell > 0:
                            revenue = shares_to_sell * price
                            cash += revenue
                            shares -= shares_to_sell
                            position_pct = 0.5
                            in_exit_phase = True
                            action = "SELL_HALF"
                            trade_info = {
                                'Date': current_date, 'Action': '减仓止盈', 'Price': price, 
                                'Shares': shares_to_sell, 'Amount': revenue, 'J': kdj_j, 'Max_J': max_j_since_full
                            }
                
                # 2. 清仓信号: J < Max_J * 0.6
                # 阈值检查
                if kdj_j < max_j_since_full * 0.6:
                    if shares > 0:
                        revenue = shares * price
                        cash += revenue
                        shares_sold = shares
                        shares = 0
                        position_pct = 0.0
                        in_exit_phase = False
                        action = "SELL_ALL"
                        trade_info = {
                            'Date': current_date, 'Action': '清仓止盈', 'Price': price, 
                            'Shares': shares_sold, 'Amount': revenue, 'J': kdj_j, 'Max_J': max_j_since_full
                        }
                        # 重置状态
                        max_j_since_full = -999

        if trade_info:
            trades.append(trade_info)
            
        # 更新组合价值
        current_equity = cash + shares * price
        portfolio_value.append(current_equity)
        
    # 最终强制清仓计算价值
    final_equity = cash + shares * df['Close'].iloc[-1]
    # 如果最后还有持仓，加一笔虚拟卖出记录以便计算收益（可选，这里只更新价值）
    
    df['组合价值'] = portfolio_value
    
    # --- 统计指标 ---
    
    # 1. 总收益率
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    # 2. 最大回撤
    portfolio_series = pd.Series(portfolio_value)
    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # 3. 交易统计 (胜率、平均收益、平均持仓天数)
    # 需要将买入和卖出配对。由于策略有加仓和分批卖出，配对比较复杂。
    # 简化计算：以"一轮完整交易"（从空仓到空仓，或到回测结束）为一个周期
    # 或者计算每一笔卖出的收益率（相对于平均持仓成本）
    
    cycle_profits = []
    cycle_returns = []
    cycle_durations = []
    
    current_cycle_pnl = 0
    current_cycle_cost = 0
    current_cycle_start_date = None
    
    # 重构交易流以计算每轮盈亏
    # 这是一个近似计算，因为分批买卖导致成本变化
    # 更精确的方法：FIFO 或 平均成本法
    # 这里使用累计投入和累计回收来计算一轮（直到清仓）的收益
    
    temp_cash_flow = 0
    temp_start_date = None
    is_holding = False
    
    for trade in trades:
        action = trade['Action']
        amount = trade['Amount']
        date = trade['Date']
        
        if '买入' in action or '加仓' in action:
            if not is_holding:
                temp_start_date = date
                is_holding = True
            temp_cash_flow -= amount # 支出
            
        elif '卖出' in action or '减仓' in action or '清仓' in action:
            temp_cash_flow += amount # 收入
            
            if '清仓' in action or (shares == 0 and action == 'SELL_ALL'): # 实际上 shares 在这里无法直接访问，需依赖 action
                # 结束一轮
                profit = temp_cash_flow
                cycle_profits.append(profit)
                
                # 估算投入成本（近似为最大占用资金，即负现金流的绝对值）
                # 这里简单用利润/绝对支出来算收益率不太准，因为分批投入。
                # 收益率 = 利润 / 总投入成本 (Sum of buy amounts)
                # 需重新遍历trades来获取该轮的总买入额
                
                is_holding = False
                if temp_start_date:
                    duration = (date - temp_start_date).days
                    cycle_durations.append(duration)
                temp_cash_flow = 0
                temp_start_date = None

    # 计算胜率等
    # 需要更严谨的循环来匹配每一轮
    round_trips = []
    current_round = {'buys': [], 'sells': []}
    
    for trade in trades:
        if '买入' in trade['Action'] or '加仓' in trade['Action']:
            if not current_round['buys'] and not current_round['sells']:
                current_round['start_date'] = trade['Date']
            current_round['buys'].append(trade['Amount'])
        elif '卖出' in trade['Action'] or '减仓' in trade['Action'] or '清仓' in trade['Action']:
            current_round['sells'].append(trade['Amount'])
            if '清仓' in trade['Action']:
                current_round['end_date'] = trade['Date']
                round_trips.append(current_round)
                current_round = {'buys': [], 'sells': []}
    
    # 统计每轮结果
    round_stats = []
    for r in round_trips:
        total_buy = sum(r['buys'])
        total_sell = sum(r['sells'])
        profit = total_sell - total_buy
        ret_pct = (profit / total_buy) * 100 if total_buy > 0 else 0
        duration = (r['end_date'] - r['start_date']).days
        round_stats.append({
            'profit': profit,
            'return': ret_pct,
            'duration': duration
        })
    
    if round_stats:
        avg_return = np.mean([r['return'] for r in round_stats])
        avg_profit = np.mean([r['profit'] for r in round_stats])
        avg_duration = np.mean([r['duration'] for r in round_stats])
        win_rate = len([r for r in round_stats if r['profit'] > 0]) / len(round_stats) * 100
        win_count = len([r for r in round_stats if r['profit'] > 0])
        total_rounds = len(round_stats)
    else:
        avg_return = 0
        avg_profit = 0
        avg_duration = 0
        win_rate = 0
        win_count = 0
        total_rounds = 0

    return {
        '总收益率': total_return,
        '最终价值': final_equity,
        '初始资金': initial_capital,
        '最大回撤': max_drawdown,
        '交易记录': trades,
        '数据': df,
        '平均每次交易收益率': avg_return,
        '平均单笔盈利': avg_profit,
        '平均持仓天数': avg_duration,
        '胜率': win_rate,
        '胜场': win_count,
        '总场次': total_rounds
    }

def plot_results(data, stock_name, strategy_result):
    """
    绘制回测结果
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # 1. 股价与买卖点
    ax1.plot(data.index, data['Close'], label='收盘价', color='blue', alpha=0.6)
    ax1.plot(data.index, data['MA20'], label='MA20', color='orange', linestyle='--', alpha=0.8)
    
    trades = strategy_result['交易记录']
    for trade in trades:
        date = trade['Date']
        price = trade['Price']
        action = trade['Action']
        
        if '买入' in action:
            ax1.scatter(date, price, color='red', marker='^', s=100, zorder=5, label='买入' if '买入' not in [l.get_label() for l in ax1.get_lines()] else "")
        elif '加仓' in action:
             ax1.scatter(date, price, color='magenta', marker='^', s=100, zorder=5, label='加仓' if '加仓' not in [l.get_label() for l in ax1.get_lines()] else "")
        elif '减仓' in action:
            ax1.scatter(date, price, color='orange', marker='v', s=100, zorder=5, label='减仓' if '减仓' not in [l.get_label() for l in ax1.get_lines()] else "")
        elif '清仓' in action:
            ax1.scatter(date, price, color='green', marker='v', s=100, zorder=5, label='清仓' if '清仓' not in [l.get_label() for l in ax1.get_lines()] else "")
            
    ax1.set_title(f'{stock_name} KDJ+MA20趋势共振策略回测')
    ax1.set_ylabel('价格')
    ax1.legend()
    
    # 2. KDJ指标
    ax2.plot(data.index, data['K'], label='K', alpha=0.5)
    ax2.plot(data.index, data['D'], label='D', alpha=0.5)
    ax2.plot(data.index, data['J'], label='J', color='purple', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(40, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(100, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylabel('KDJ')
    ax2.legend(loc='upper right')
    
    # 3. 组合价值
    ax3.plot(data.index, data['组合价值'], label='策略组合价值', color='red')
    ax3.plot(data.index, [strategy_result['初始资金']] * len(data), label='初始资金', color='gray', linestyle='--')
    ax3.set_ylabel('资产价值')
    ax3.legend()
    
    # 格式化日期
    plt.tight_layout()
    filename = f'images/kdj_trend_{stock_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    return filename

def print_trades(trades, stock_name):
    print(f"\n{stock_name} 详细交易记录:")
    print("=" * 100)
    print(f"{'日期':<12} {'操作':<10} {'价格':<8} {'股数':<8} {'金额':<10} {'J值':<8} {'MA20斜率':<10} {'Max_J':<8}")
    print("-" * 100)
    
    for t in trades:
        date_str = t['Date'].strftime('%Y-%m-%d')
        max_j = f"{t.get('Max_J', 0):.2f}" if 'Max_J' in t else "-"
        slope = f"{t.get('Slope', 0):.4f}" if 'Slope' in t else "-"
        print(f"{date_str:<12} {t['Action']:<10} {t['Price']:<8.2f} {t['Shares']:<8} {t['Amount']:<10.0f} {t['J']:<8.2f} {slope:<10} {max_j:<8}")

def main():
    stock_name = "中钨高新"
    stock_code = "000657"
    years = 1
    initial_capital = 100000
    
    print("KDJ超卖与短期趋势共振策略回测")
    print("=" * 60)
    print("策略逻辑：")
    print("1. 买入: J<0 且 MA20斜率>-0.02 -> 建仓50%")
    print("2. 加仓: 持仓且 J>40 -> 加仓至100%")
    print("3. 减仓: J < Max_J*0.8 -> 卖出50%")
    print("4. 清仓: J < Max_J*0.6 -> 卖出全部")
    print("=" * 60)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365 + 60) # 多取60天用于MA20计算
    
    print(f"\n正在获取 {stock_name} ({stock_code}) 数据...")
    df = get_stock_data(stock_code, start_date, end_date)
    
    if df.empty:
        print("数据获取失败！")
        return
        
    print("计算指标...")
    df = calculate_indicators(df)
    
    print("执行策略...")
    result = run_strategy(df, initial_capital)
    
    print(f"\n{stock_name} 回测结果 (过去{years}年):")
    print(f"总收益率: {result['总收益率']:.2f}%")
    print(f"最终价值: {result['最终价值']:.2f} 元")
    print(f"最大回撤: {result['最大回撤']:.2f}%")
    print(f"平均每次交易收益率: {result['平均每次交易收益率']:.2f}%")
    print(f"平均持仓天数: {result['平均持仓天数']:.1f} 天")
    print(f"胜率: {result['胜率']:.2f}% ({result['胜场']}/{result['总场次']})")
    
    print_trades(result['交易记录'], stock_name)
    
    filename = plot_results(result['数据'], stock_name, result)
    print(f"\n图表已保存: {filename}")

if __name__ == "__main__":
    main()