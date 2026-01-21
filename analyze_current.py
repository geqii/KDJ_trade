#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析当前股票状态
获取最新行情，计算SKDJ指标，判断是否适合买入
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def get_realtime_data(symbol):
    """
    获取实时/最新行情数据
    """
    try:
        # 获取历史数据（包含最新一天）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60) # 获取足够的数据计算指标
        
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # 获取前复权数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        
        if df.empty:
            print(f"获取数据失败 {symbol}")
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

def analyze_stock(stock_name, stock_code):
    """
    分析单只股票
    """
    print(f"\n正在分析 {stock_name} ({stock_code})...")
    
    df = get_realtime_data(stock_code)
    if df.empty:
        print("无法获取数据")
        return

    # 计算指标
    k, d = calculate_skdj(df)
    df['K'] = k
    df['D'] = d
    
    # 获取最新数据
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_price = latest['Close']
    current_k = latest['K']
    current_d = latest['D']
    prev_k = prev['K']
    
    date_str = latest.name.strftime('%Y-%m-%d')
    
    print(f"数据日期: {date_str}")
    print(f"当前价格: {current_price:.2f}")
    print(f"当前K值: {current_k:.2f}")
    print(f"当前D值: {current_d:.2f}")
    print(f"前一日K值: {prev_k:.2f}")
    
    # 判断信号
    print("\n【SKDJ策略分析 (买入:K<30, 卖出:K>80)】")
    
    if current_k < 30:
        print(f"🟢 信号: **买入** (K值 {current_k:.2f} < 30)")
        print("建议: 当前处于超卖区域，符合买入条件。")
    elif current_k > 80:
        print(f"🔴 信号: **卖出** (K值 {current_k:.2f} > 80)")
        print("建议: 当前处于超买区域，符合卖出条件。")
    else:
        print(f"⚪ 信号: **观望** (30 <= K值 {current_k:.2f} <= 80)")
        if current_k > prev_k:
             print("趋势: K值上升中")
        else:
             print("趋势: K值下降中")

def main():
    analyze_stock("四方股份", "601126")

if __name__ == "__main__":
    main()