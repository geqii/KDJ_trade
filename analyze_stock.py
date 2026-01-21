import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def calculate_indicators_for_analysis(df, today_price):
    """
    Append today's hypothetical data and calculate indicators.
    """
    # Create a new row for today
    last_date = df.index[-1]
    today_date = last_date + timedelta(days=1)
    if today_date.weekday() >= 5: # Skip weekend if logic needed, but simple append is fine for estimate
        today_date += timedelta(days=2)
        
    # We only need Close, High, Low for KDJ and Close for MA20
    # Assuming today's High/Low roughly around Close or previous volatility
    # To be conservative for J < 0 check:
    # If price dropped 4%, Low is likely at least 36.7. 
    # Let's assume High was open or yesterday close, Low is close (worst case for bulls)
    # Actually KDJ uses Low(9). If today is low, it affects RSV significantly.
    
    new_row = pd.DataFrame({
        'Open': [today_price * 1.04], # Approx
        'High': [today_price * 1.04],
        'Low': [today_price], # Close at low
        'Close': [today_price],
        'Volume': [0]
    }, index=[today_date])
    
    # Concatenate
    df_extended = pd.concat([df, new_row])
    
    # Calculate MA20
    df_extended['MA20'] = df_extended['Close'].rolling(window=20).mean()
    df_extended['MA20_Slope'] = df_extended['MA20'].diff()
    
    # Calculate KDJ
    n=9; m1=3; m2=3
    low_min = df_extended['Low'].rolling(window=n).min()
    high_max = df_extended['High'].rolling(window=n).max()
    rsv = (df_extended['Close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    
    k_values = []
    d_values = []
    k = 50; d = 50
    
    # We need to recalculate KDJ for the whole series or at least enough window
    # Simple loop for all
    for r in rsv:
        k = (m1 - 1) / m1 * k + 1 / m1 * r
        d = (m2 - 1) / m2 * d + 1 / m2 * k
        k_values.append(k)
        d_values.append(d)
        
    df_extended['K'] = k_values
    df_extended['D'] = d_values
    df_extended['J'] = 3 * df_extended['K'] - 2 * df_extended['D']
    
    return df_extended.iloc[-1]

if __name__ == "__main__":
    ticker = "300468.SZ" # 四方精创
    assumed_price = 36.7
    
    print(f"Fetching recent data for {ticker}...")
    # Get enough data for MA20
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        print("Failed to fetch data.")
    else:
        print(f"Last actual close: {df.iloc[-1]['Close']:.2f} on {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Assumed Today Close: {assumed_price}")
        
        result = calculate_indicators_for_analysis(df, assumed_price)
        
        print("\n--- Analysis Result ---")
        print(f"MA20 Slope: {result['MA20_Slope']:.4f}")
        print(f"J Value:    {result['J']:.4f}")
        
        print("\n--- Strategy Check ---")
        is_j_oversold = result['J'] < 0
        is_trend_up = result['MA20_Slope'] > -0.02
        
        print(f"1. J < 0 (Oversold): {'YES' if is_j_oversold else 'NO'} ({result['J']:.2f})")
        print(f"2. MA20 Slope > -0.02 (Trend): {'YES' if is_trend_up else 'NO'} ({result['MA20_Slope']:.4f})")
        
        if is_j_oversold and is_trend_up:
            print("\n>>> CONCLUSION: FIT for BUY (建仓) <<<")
        else:
            print("\n>>> CONCLUSION: WAIT (不满足条件) <<<")
            if not is_j_oversold:
                print("  - Reason: Not oversold enough (J > 0)")
            if not is_trend_up:
                print("  - Reason: Trend is broken (MA20 Slope too negative)")
