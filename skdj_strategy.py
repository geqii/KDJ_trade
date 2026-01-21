#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKDJ策略回测脚本
K值<30买入，K值>80卖出
使用AkShare获取数据
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

def calculate_skdj(data, n=9, m=3):
    """
    计算SKDJ指标
    SKDJ是KDJ指标的慢速版本，更加平滑
    """
    # 计算RSV值
    low_min = data['Low'].rolling(window=n, min_periods=n).min()
    high_max = data['High'].rolling(window=n, min_periods=n).max()
    rsv = (data['Close'] - low_min) / (high_max - low_min) * 100
    
    # 计算K值（RSV的M日移动平均）
    k = rsv.rolling(window=m, min_periods=m).mean()
    
    # 计算D值（K值的M日移动平均）
    d = k.rolling(window=m, min_periods=m).mean()
    
    return k, d

def run_skdj_strategy(data, initial_capital=100000, k_buy=30, k_sell=80):
    """
    运行SKDJ策略
    """
    # 计算SKDJ指标
    data['K'], data['D'] = calculate_skdj(data)
    
    # 初始化策略变量
    position = 0  # 当前持仓状态：0=空仓，1=持仓
    cash = initial_capital
    shares = 0
    trades = []  # 记录交易
    portfolio_value = []  # 记录组合价值
    
    # 遍历数据
    for i in range(len(data)):
        if pd.isna(data['K'].iloc[i]):
            portfolio_value.append(cash)
            continue
            
        current_price = data['Close'].iloc[i]
        current_date = data.index[i]
        k_value = data['K'].iloc[i]
        
        # 买入信号：K值<30且当前空仓
        if k_value < k_buy and position == 0:
            # 计算可买入的股数（100股为单位）
            shares_to_buy = int(cash / (current_price * 100)) * 100
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                cash -= cost
                shares = shares_to_buy
                position = 1
                trades.append({
                    '日期': current_date,
                    '操作': '买入',
                    '价格': current_price,
                    '股数': shares_to_buy,
                    '金额': cost,
                    'K值': k_value
                })
        
        # 卖出信号：K值>80且当前持仓
        elif k_value > k_sell and position == 1:
            revenue = shares * current_price
            cash += revenue
            trades.append({
                '日期': current_date,
                '操作': '卖出',
                '价格': current_price,
                '股数': shares,
                '金额': revenue,
                'K值': k_value
            })
            shares = 0
            position = 0
        
        # 计算当前组合价值
        current_value = cash + shares * current_price
        portfolio_value.append(current_value)
    
    # 最终清仓
    if position == 1:
        final_revenue = shares * data['Close'].iloc[-1]
        cash += final_revenue
        trades.append({
            '日期': data.index[-1],
            '操作': '卖出(清仓)',
            '价格': data['Close'].iloc[-1],
            '股数': shares,
            '金额': final_revenue,
            'K值': data['K'].iloc[-1]
        })
        final_value = cash
    else:
        final_value = cash
    
    # 计算收益率
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # 添加组合价值到数据框
    data['组合价值'] = portfolio_value
    
    # --- 计算回撤 ---
    portfolio_series = pd.Series(portfolio_value)
    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min() * 100  # 转为百分比
    avg_drawdown = drawdown.mean() * 100 # 转为百分比
    
    # --- 计算持仓天数 ---
    holding_days_list = []
    buy_date = None
    for trade in trades:
        if trade['操作'] == '买入':
            buy_date = trade['日期']
        elif '卖出' in trade['操作'] and buy_date is not None:
            days = (trade['日期'] - buy_date).days
            holding_days_list.append(days)
            buy_date = None
            
    avg_holding_days = np.mean(holding_days_list) if holding_days_list else 0
    
    return {
        '总收益率': total_return,
        '最终价值': final_value,
        '初始资金': initial_capital,
        '交易记录': trades,
        '数据': data,
        '最大回撤': max_drawdown,
        '平均回撤': avg_drawdown,
        '平均持仓天数': avg_holding_days
    }

def plot_results(data, stock_name, strategy_result):
    """
    绘制回测结果
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # 绘制股价和K值
    ax1.plot(data.index, data['Close'], label='收盘价', color='blue', alpha=0.7)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(data.index, data['K'], label='SKDJ-K值', color='red', alpha=0.8)
    ax1_twin.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='买入线(K=30)')
    ax1_twin.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='卖出线(K=80)')
    
    # 标记买卖点
    trades = strategy_result['交易记录']
    for trade in trades:
        if trade['操作'] == '买入':
            ax1.scatter(trade['日期'], trade['价格'], color='green', marker='^', s=100, zorder=5)
        elif '卖出' in trade['操作']:
            ax1.scatter(trade['日期'], trade['价格'], color='red', marker='v', s=100, zorder=5)
    
    ax1.set_ylabel('股价 (元)')
    ax1_twin.set_ylabel('K值')
    ax1.set_title(f'{stock_name} SKDJ策略回测结果')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # 绘制组合价值
    ax2.plot(data.index, data['组合价值'], label='策略组合价值', color='purple', linewidth=2)
    ax2.plot(data.index, [strategy_result['初始资金']] * len(data), label='初始资金', color='gray', linestyle='--', alpha=0.7)
    ax2.set_ylabel('组合价值 (元)')
    ax2.set_title('策略组合价值变化')
    ax2.legend()
    
    # 绘制K值和买卖信号
    ax3.plot(data.index, data['K'], label='K值', color='red', alpha=0.8)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='买入线(K=30)')
    ax3.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='卖出线(K=80)')
    ax3.fill_between(data.index, 0, 30, alpha=0.1, color='green', label='买入区域')
    ax3.fill_between(data.index, 80, 100, alpha=0.1, color='red', label='卖出区域')
    ax3.set_ylabel('K值')
    ax3.set_xlabel('日期')
    ax3.set_title('SKDJ指标和交易信号')
    ax3.legend()
    
    # 设置日期格式
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f'images/skdj_strategy_{stock_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return filename

def print_trades(trades, stock_name):
    """
    打印交易记录
    """
    print(f"\n{stock_name} 交易记录:")
    print("=" * 80)
    print(f"{'日期':<12} {'操作':<8} {'价格(元)':<10} {'股数':<8} {'金额(元)':<12} {'K值':<8}")
    print("-" * 80)
    
    for trade in trades:
        print(f"{trade['日期'].strftime('%Y-%m-%d'):<12} {trade['操作']:<8} {trade['价格']:<10.2f} {trade['股数']:<8} {trade['金额']:<12.2f} {trade['K值']:<8.2f}")

def main():
    """
    主函数
    """
    initial_capital = 100000
    
    # 股票配置：名称 -> (代码, 回测年数)
    stocks_config = {
        '中钨高新': ('000657', 1)
    }
    
    results = {}
    
    print("=" * 60)
    print("SKDJ策略规则说明")
    print("=" * 60)
    print("1. 指标计算: SKDJ (慢速KDJ)")
    print("   - N (周期) = 9")
    print("   - M (平滑) = 3")
    print("   - K值 = RSV的M日移动平均")
    print("   - D值 = K值的M日移动平均")
    print(f"2. 买入条件: K值 < 30 且 当前空仓")
    print(f"3. 卖出条件: K值 > 80 且 当前持仓")
    print("4. 仓位管理: 全仓买入 (按100股取整)")
    print("=" * 60)
    print("\n开始回测...")
    
    end_date = datetime.now()
    
    for stock_name, (stock_code, years) in stocks_config.items():
        print(f"\n回测 {stock_name} ({stock_code}) - 过去 {years} 年...")
        
        start_date = end_date - timedelta(days=years*365)
        
        # 获取股票数据
        data = get_stock_data(stock_code, start_date, end_date)
        
        if data.empty:
            print(f"警告: {stock_name} 数据获取失败")
            continue
        
        # 运行策略
        result = run_skdj_strategy(data, initial_capital)
        result['years'] = years
        results[stock_name] = result
        
        # 打印结果
        print(f"\n{stock_name} 回测结果:")
        print(f"总收益率: {result['总收益率']:.2f}%")
        print(f"最终价值: {result['最终价值']:.2f} 元")
        print(f"初始资金: {result['初始资金']:.2f} 元")
        print(f"最大回撤: {result['最大回撤']:.2f}%")
        print(f"平均回撤: {result['平均回撤']:.2f}%")
        print(f"平均持仓天数: {result['平均持仓天数']:.1f} 天")
        print(f"交易次数: {len(result['交易记录'])} 次")
        
        # 打印交易记录
        print_trades(result['交易记录'], stock_name)
        
        # 绘制图表
        filename = plot_results(result['数据'], stock_name, result)
        print(f"图表已保存: {filename}")
            
    
    # 对比分析
    print("\n" + "=" * 60)
    print("SKDJ策略回测对比分析")
    print("=" * 60)
    
    for stock_name, result in results.items():
        years = result.get('years', 1)
        print(f"\n{stock_name} ({years}年):")
        print(f"  总收益率: {result['总收益率']:.2f}%")
        if years > 0:
            print(f"  年化收益率: {(result['总收益率']/years):.2f}%")
        print(f"  最大回撤: {result['最大回撤']:.2f}%")
        print(f"  平均持仓天数: {result['平均持仓天数']:.1f} 天")
        
        if len(result['交易记录']) > 0:
            # 计算实际盈利情况
            realized_profits = []
            buy_trade = None
            for trade in result['交易记录']:
                if trade['操作'] == '买入':
                    buy_trade = trade
                elif '卖出' in trade['操作'] and buy_trade:
                    profit = (trade['价格'] - buy_trade['价格']) * trade['股数']
                    realized_profits.append(profit)
                    buy_trade = None
            
            if realized_profits:
                avg_profit = np.mean(realized_profits)
                win_count = len([p for p in realized_profits if p > 0])
                win_rate = win_count / len(realized_profits) * 100
                print(f"  平均单笔盈利: {avg_profit:.2f} 元")
                
                # 计算平均交易收益率
                trade_returns = []
                buy_trade = None
                for trade in result['交易记录']:
                    if trade['操作'] == '买入':
                        buy_trade = trade
                    elif '卖出' in trade['操作'] and buy_trade:
                        # 收益率 = (卖出价格 - 买入价格) / 买入价格
                        ret = (trade['价格'] - buy_trade['价格']) / buy_trade['价格'] * 100
                        trade_returns.append(ret)
                        buy_trade = None
                
                if trade_returns:
                    avg_return = np.mean(trade_returns)
                    print(f"  平均每次交易收益率: {avg_return:.2f}%")
                
                print(f"  胜率: {win_rate:.2f}% ({win_count}/{len(realized_profits)})")

if __name__ == "__main__":
    main()