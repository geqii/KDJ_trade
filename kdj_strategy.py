import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set font for matplotlib to support Chinese characters if available
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def calculate_kdj(df, n=9, m1=3, m2=3):
    """
    Calculate KDJ indicators.
    """
    df = df.copy()
    # Calculate RSV
    low_min = df['Low'].rolling(window=n).min()
    high_max = df['High'].rolling(window=n).max()
    
    # Avoid division by zero
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50) # Fill NaN with 50 initially
    
    # Ensure it's a float series (flatten to 1D array to avoid index issues during iteration)
    rsv_values = rsv.values.flatten().astype(float)
    
    # Calculate K, D, J using recursive method
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
    Calculate all necessary indicators for the strategy.
    """
    # 1. KDJ
    df = calculate_kdj(df)
    
    # 2. MA20
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 3. MA20 Slope: (Today MA20 - Yesterday MA20)
    df['MA20_Slope'] = df['MA20'].diff()
    
    return df

def run_strategy(df, initial_capital=100000):
    """
    Execute the KDJ Oversold & Short-term Trend Resonance Strategy.
    """
    df = df.copy().reset_index()
    
    # Portfolio State
    cash = initial_capital
    shares = 0
    position_pct = 0.0 # 0, 0.5, 1.0
    
    # Strategy State
    max_j_since_full = -999 # To track Max_J after full position
    has_peaked_40 = False # To track if J > 40 condition met for selling
    in_exit_phase = False # To prevent re-adding after partial sell
    
    # Logging
    trades = []
    equity_curve = []
    
    # We iterate from index 20 (to ensure MA20 exists)
    for i in range(20, len(df)):
        today = df.iloc[i]
        date = today['Date']
        price = today['Close']
        kdj_j = today['J']
        ma20_slope = today['MA20_Slope']
        
        # --- Update Equity ---
        current_equity = cash + shares * price
        equity_curve.append({'Date': date, 'Equity': current_equity})
        
        action = None
        
        # --- Logic ---
        
        # A. Buy Condition (Build Base Position 50%)
        # Condition: Empty position AND J < 0 AND MA20_Slope > -0.02
        if position_pct == 0.0:
            if kdj_j < 0 and ma20_slope > -0.02:
                # Calculate shares to buy (50% of capital)
                target_value = current_equity * 0.5
                shares_to_buy = int(target_value / price)
                
                # A-share rule: Buy in lots of 100
                shares_to_buy = (shares_to_buy // 100) * 100
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares += shares_to_buy
                    position_pct = 0.5
                    in_exit_phase = False # Reset exit phase
                    action = "建仓 (50%)"
                    trades.append({
                        '日期': date, '操作': '建仓 (50%)', '价格': price, 
                        '股数': shares_to_buy, 'J值': kdj_j, 'MA20斜率': ma20_slope
                    })
        
        # B. Add Position Condition (Add to 100%)
        # Condition: Holding base position (0.5) AND J > 40 AND Not in exit phase
        elif position_pct == 0.5 and not in_exit_phase:
            if kdj_j > 40:
                # Calculate shares to buy (Remaining capital to reach 100% roughly)
                shares_to_buy = int(cash / price)
                
                # A-share rule: Buy in lots of 100
                shares_to_buy = (shares_to_buy // 100) * 100
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares += shares_to_buy
                    position_pct = 1.0
                    action = "加仓 (100%)"
                    
                    # Initialize Sell State Tracking
                    max_j_since_full = kdj_j # Start recording Max_J
                    has_peaked_40 = True # Confirmed J > 40 (since that triggered this)
                    
                    trades.append({
                        '日期': date, '操作': '加仓 (100%)', '价格': price, 
                        '股数': shares_to_buy, 'J值': kdj_j, 'MA20斜率': ma20_slope
                    })
            
            # Note: What if J drops < 0 again while at 0.5? 
            # Strategy doesn't specify stop loss for 0.5 position or re-entry.
            # We strictly follow the "Add Condition".
            
        # C. Sell/Take Profit Condition
        # Condition: Holding position AND J was > 40 (implied by getting to 1.0)
        # Note: Logic says "Only trigger when holding position and J has peaked over 40".
        # This applies to our 1.0 position state which entered at J > 40.
        elif position_pct > 0.5 or (position_pct == 0.5 and in_exit_phase): 
             # Update Max_J
            if kdj_j > max_j_since_full:
                max_j_since_full = kdj_j
            
            # 1. Reduce Position Signal: J < Max_J * 0.8
            # Only if we are at 100% (haven't sold first half yet)
            if position_pct == 1.0:
                if kdj_j < max_j_since_full * 0.8:
                    # Sell 50% of current holding
                    shares_to_sell = int(shares / 2)
                    
                    # A-share rule: Sell in lots of 100 is preferred but not strictly required (odd lots allowed in sell)
                    # But for consistency with strategy "50%", let's try to keep it clean.
                    # However, strictly speaking, you can sell 1 share in A-share.
                    # But let's round to nearest 100 for "strategy logic" or just keep integer.
                    # User asked "A股没有散股，最少一手是100股". Usually means for BUY.
                    # For SELL, you can sell odd lots. But let's stick to 100 multiples for clean testing if possible,
                    # or just integer is fine. Let's assume we want to keep holding in 100s.
                    shares_to_sell = (shares_to_sell // 100) * 100
                    
                    if shares_to_sell > 0:
                        revenue = shares_to_sell * price
                        cash += revenue
                        shares -= shares_to_sell
                        position_pct = 0.5 # Back to 0.5
                        in_exit_phase = True # Enter exit phase
                        action = "减仓 (50%)"
                        trades.append({
                            '日期': date, '操作': '减仓 (50%)', '价格': price, 
                            '股数': shares_to_sell, 'J值': kdj_j, 'Max_J': max_j_since_full
                        })

            # 2. Clear Position Signal: J < Max_J * 0.6
            # Can happen from 1.0 directly or from 0.5 (after first sell)
            # Threshold check
            if kdj_j < max_j_since_full * 0.6:
                if shares > 0:
                    revenue = shares * price
                    cash += revenue
                    shares_sold = shares
                    shares = 0
                    position_pct = 0.0
                    in_exit_phase = False # Reset
                    action = "清仓 (卖出)"
                    trades.append({
                        '日期': date, '操作': '清仓 (卖出)', '价格': price, 
                        '股数': shares_sold, 'J值': kdj_j, 'Max_J': max_j_since_full
                    })
                    # Reset State
                    max_j_since_full = -999
                    has_peaked_40 = False

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)

def calculate_performance_breakdown(equity_curve):
    """
    Calculate monthly and quarterly performance breakdown.
    """
    if equity_curve.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    df = equity_curve.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # --- Monthly Breakdown ---
    # Resample to month end, taking the last equity value
    monthly = df.resample('M').last()
    
    # Calculate monthly PnL
    # First month PnL = End Equity - Initial Capital (if first entry) or Previous Month End
    # But simpler: shift equity by 1 to get start equity
    monthly['Start_Equity'] = monthly['Equity'].shift(1)
    
    # Fill first start equity properly
    # If the backtest starts mid-month, the first month start equity should be initial capital
    # We can infer it from the diff
    
    # Simpler approach using diff() on the resampled series
    monthly['Profit'] = monthly['Equity'].diff()
    
    # Fix first month profit
    first_month_idx = monthly.index[0]
    monthly.loc[first_month_idx, 'Profit'] = monthly.loc[first_month_idx, 'Equity'] - 100000
    
    # Recalculate Start_Equity for clarity
    monthly['Start_Equity'] = monthly['Equity'] - monthly['Profit']
    monthly['Return_Pct'] = (monthly['Profit'] / monthly['Start_Equity']) * 100
    
    # Rename columns to Chinese
    monthly.columns = ['期初权益', '期末权益', '收益额', '收益率(%)']
    
    # --- Quarterly Breakdown ---
    quarterly = df.resample('Q').last()
    quarterly['Profit'] = quarterly['Equity'].diff()
    quarterly.loc[quarterly.index[0], 'Profit'] = quarterly.loc[quarterly.index[0], 'Equity'] - 100000
    quarterly['Start_Equity'] = quarterly['Equity'] - quarterly['Profit']
    quarterly['Return_Pct'] = (quarterly['Profit'] / quarterly['Start_Equity']) * 100
    
    # Rename columns to Chinese
    quarterly.columns = ['期初权益', '期末权益', '收益额', '收益率(%)']
    
    # Return with Chinese columns
    return monthly[['期初权益', '期末权益', '收益额', '收益率(%)']], quarterly[['期初权益', '期末权益', '收益额', '收益率(%)']]

def plot_results(df, trades_df, ticker_name=""):
    """
    Visualize the strategy results.
    """
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Price and MA20 with Buy/Sell markers
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['Close'], label='收盘价', alpha=0.6)
    ax1.plot(df.index, df['MA20'], label='MA20均线', color='orange', linestyle='--')
    
    # Plot Buy/Sell markers
    if not trades_df.empty:
        buys = trades_df[trades_df['操作'].str.contains('建仓|加仓')]
        sells = trades_df[trades_df['操作'].str.contains('减仓|清仓')]
        
        # Map dates to index for plotting
        # We need to ensure we can find the index corresponding to the date
        df_reset = df.reset_index()
        
        for _, trade in buys.iterrows():
            if trade['日期'] in df_reset['Date'].values:
                idx = df_reset[df_reset['Date'] == trade['日期']].index[0]
                marker = '^' 
                color = 'r' if '建仓' in trade['操作'] else 'm' 
                ax1.scatter(idx, trade['价格'], marker='^', color=color, s=100, zorder=5, label=trade['操作'])

        for _, trade in sells.iterrows():
             if trade['日期'] in df_reset['Date'].values:
                idx = df_reset[df_reset['Date'] == trade['日期']].index[0]
                color = 'g' if '减仓' in trade['操作'] else 'k' 
                ax1.scatter(idx, trade['价格'], marker='v', color=color, s=100, zorder=5, label=trade['操作'])

    # Deduplicate legend labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax1.set_title(f'{ticker_name} 价格与交易记录')
    
    # Subplot 2: KDJ
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(df.index, df['K'], label='K')
    ax2.plot(df.index, df['D'], label='D')
    ax2.plot(df.index, df['J'], label='J', color='purple')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(40, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('KDJ Indicator')
    ax2.legend()
    
    # Subplot 3: MA20 Slope
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(df.index, df['MA20_Slope'], label='MA20 Slope', color='brown')
    ax3.axhline(-0.02, color='red', linestyle='--', label='Threshold -0.02')
    ax3.set_title('MA20 Slope')
    ax3.legend()
    
    # Set x-axis limit to actual data range
    if not df.empty:
        plt.xlim(df.index.min(), df.index.max())
    
    plt.tight_layout()
    filename = f'strategy_result_{ticker_name}.png'
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    # Define targets and period
    # Calculate date range for "past year"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Targets: Name -> Ticker
    targets = {
        '四方精创': '300468.SZ'
    }
    
    print(f"Backtesting Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
    
    for name, ticker in targets.items():
        print(f"{'='*30}")
        print(f"Processing {name} ({ticker})...")
        print(f"{'='*30}")
        
        # 1. Fetch Data
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error fetching data for {name}: {e}")
            continue
        
        if df.empty:
            print(f"No data fetched for {name}. Check symbol or connection.")
            continue
            
        # Flatten columns if MultiIndex (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 2. Calculate Indicators
        print("Calculating indicators...")
        df = calculate_indicators(df)
        
        # 3. Run Strategy
        print("Running strategy...")
        trades, equity = run_strategy(df)
        
        # 4. Output Results
        print(f"\n--- 交易记录: {name} ---")
        if not trades.empty:
            print(trades[['日期', '操作', '价格', '股数', 'J值', 'MA20斜率']].to_string())
            
            final_equity = equity.iloc[-1]['Equity']
            print(f"\n初始资金: 100000")
            print(f"最终权益: {final_equity:.2f}")
            print(f"总收益率: {((final_equity - 100000) / 100000) * 100:.2f}%")
            
            # Calculate Monthly/Quarterly Breakdown
            monthly_perf, quarterly_perf = calculate_performance_breakdown(equity)
            
            print(f"\n--- 月度收益表现: {name} ---")
            pd.set_option('display.float_format', '{:.2f}'.format)
            print(monthly_perf.to_string())
            
            print(f"\n--- 季度收益表现: {name} ---")
            print(quarterly_perf.to_string())
            
            # 5. Plot
            plot_results(df, trades, ticker_name=name)
        else:
            print("没有触发交易。")
        
        print("\n")
