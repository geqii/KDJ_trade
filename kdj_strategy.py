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
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares += shares_to_buy
                    position_pct = 0.5
                    in_exit_phase = False # Reset exit phase
                    action = "BUY_BASE"
                    trades.append({
                        'Date': date, 'Action': 'BUY_BASE', 'Price': price, 
                        'Shares': shares_to_buy, 'J': kdj_j, 'Slope': ma20_slope
                    })
        
        # B. Add Position Condition (Add to 100%)
        # Condition: Holding base position (0.5) AND J > 40 AND Not in exit phase
        elif position_pct == 0.5 and not in_exit_phase:
            if kdj_j > 40:
                # Calculate shares to buy (Remaining capital to reach 100% roughly)
                # Actually simpler: use remaining cash
                shares_to_buy = int(cash / price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares += shares_to_buy
                    position_pct = 1.0
                    action = "ADD_FULL"
                    
                    # Initialize Sell State Tracking
                    max_j_since_full = kdj_j # Start recording Max_J
                    has_peaked_40 = True # Confirmed J > 40 (since that triggered this)
                    
                    trades.append({
                        'Date': date, 'Action': 'ADD_FULL', 'Price': price, 
                        'Shares': shares_to_buy, 'J': kdj_j, 'Slope': ma20_slope
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
                    revenue = shares_to_sell * price
                    cash += revenue
                    shares -= shares_to_sell
                    position_pct = 0.5 # Back to 0.5
                    in_exit_phase = True # Enter exit phase
                    action = "SELL_HALF"
                    trades.append({
                        'Date': date, 'Action': 'SELL_HALF', 'Price': price, 
                        'Shares': shares_to_sell, 'J': kdj_j, 'Max_J': max_j_since_full
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
                    action = "SELL_ALL"
                    trades.append({
                        'Date': date, 'Action': 'SELL_ALL', 'Price': price, 
                        'Shares': shares_sold, 'J': kdj_j, 'Max_J': max_j_since_full
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
    # For the first month, we need to find the equity at the very beginning of the backtest
    # Or assume the first record in equity_curve is the start.
    # Actually, equity_curve starts from index 20 of daily data.
    # Let's use the first available equity in the curve as the start base for the first month?
    # Better: Use the initial capital (100000) for the very first period? 
    # Or just use the equity curve diff.
    
    # Let's refine:
    # We want exact profit per month.
    # Profit = Equity_End - Equity_Start
    # Equity_Start of Month M is Equity_End of Month M-1.
    
    # Fill the first NaN Start_Equity with the first equity value from the daily curve 
    # (or better, the value before the first month end). 
    # A robust way:
    # 1. Get month ends
    # 2. Add the very first start date equity to the series
    
    # Simpler approach using diff() on the resampled series
    monthly['Profit'] = monthly['Equity'].diff()
    
    # Fix first month profit: 
    # It currently is NaN. We need (First Month End Equity - Equity at start of backtest)
    initial_equity = df.iloc[0]['Equity'] 
    # Actually, 'initial_equity' in the dataframe is the equity at day 20. 
    # If the first month end is later than day 20, the diff is correct if we insert the start.
    
    # Let's just fill the first NaN with (First Month Equity - Initial Capital)
    # Assuming start capital is 100000. But if we run multiple stocks, we need to pass it or infer.
    # Let's infer from the first row of daily data if possible, or just use 100000 default.
    # The 'run_strategy' uses 100000.
    
    first_month_idx = monthly.index[0]
    # If the first daily record is in the same month as first_month_idx
    monthly.loc[first_month_idx, 'Profit'] = monthly.loc[first_month_idx, 'Equity'] - 100000
    
    # Recalculate Start_Equity for clarity
    monthly['Start_Equity'] = monthly['Equity'] - monthly['Profit']
    monthly['Return_Pct'] = (monthly['Profit'] / monthly['Start_Equity']) * 100
    
    # --- Quarterly Breakdown ---
    quarterly = df.resample('Q').last()
    quarterly['Profit'] = quarterly['Equity'].diff()
    quarterly.loc[quarterly.index[0], 'Profit'] = quarterly.loc[quarterly.index[0], 'Equity'] - 100000
    quarterly['Start_Equity'] = quarterly['Equity'] - quarterly['Profit']
    quarterly['Return_Pct'] = (quarterly['Profit'] / quarterly['Start_Equity']) * 100
    
    return monthly[['Start_Equity', 'Equity', 'Profit', 'Return_Pct']], quarterly[['Start_Equity', 'Equity', 'Profit', 'Return_Pct']]

def plot_results(df, trades_df, ticker_name=""):
    """
    Visualize the strategy results.
    """
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Price and MA20 with Buy/Sell markers
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close Price', alpha=0.6)
    ax1.plot(df.index, df['MA20'], label='MA20', color='orange', linestyle='--')
    
    # Plot Buy/Sell markers
    if not trades_df.empty:
        buys = trades_df[trades_df['Action'].str.contains('BUY|ADD')]
        sells = trades_df[trades_df['Action'].str.contains('SELL')]
        
        # Map dates to index for plotting
        # We need to ensure we can find the index corresponding to the date
        df_reset = df.reset_index()
        
        for _, trade in buys.iterrows():
            idx = df_reset[df_reset['Date'] == trade['Date']].index[0]
            marker = '^' if trade['Action'] == 'BUY_BASE' else 'v' # Use different marker logic? No, Up arrow for buy
            color = 'r' if trade['Action'] == 'BUY_BASE' else 'm' # Red for base, Magenta for add
            ax1.scatter(idx, trade['Price'], marker='^', color=color, s=100, zorder=5, label=trade['Action'])

        for _, trade in sells.iterrows():
            idx = df_reset[df_reset['Date'] == trade['Date']].index[0]
            color = 'g' if trade['Action'] == 'SELL_HALF' else 'k' # Green for half, Black for clear
            ax1.scatter(idx, trade['Price'], marker='v', color=color, s=100, zorder=5, label=trade['Action'])

    # Deduplicate legend labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax1.set_title(f'{ticker_name} Price & Trades')
    
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
        '四方股份': '601126.SS'
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
        print(f"\n--- Trade Log: {name} ---")
        if not trades.empty:
            print(trades[['Date', 'Action', 'Price', 'J', 'Slope', 'Shares']].to_string())
            
            final_equity = equity.iloc[-1]['Equity']
            print(f"\nInitial Capital: 100000")
            print(f"Final Equity: {final_equity:.2f}")
            print(f"Total Return: {((final_equity - 100000) / 100000) * 100:.2f}%")
            
            # Calculate Monthly/Quarterly Breakdown
            monthly_perf, quarterly_perf = calculate_performance_breakdown(equity)
            
            print(f"\n--- Monthly Performance: {name} ---")
            pd.set_option('display.float_format', '{:.2f}'.format)
            print(monthly_perf.to_string())
            
            print(f"\n--- Quarterly Performance: {name} ---")
            print(quarterly_perf.to_string())
            
            # 5. Plot
            plot_results(df, trades, ticker_name=name)
        else:
            print("No trades triggered.")
        
        print("\n")
