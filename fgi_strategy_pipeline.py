import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import random
from datetime import datetime

# File upload helpers (Colab compatible)
def upload_file():
    from google.colab import files
    print("Please upload Excel or CSV file:")
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file was uploaded.")
    name = list(uploaded.keys())[0]
    print(f"Uploaded file: {name}")
    return name, uploaded

# Read uploaded data
def read_uploaded_file(uploaded):
    name = list(uploaded.keys())[0]
    content = uploaded[name]
    if name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(content))
    elif name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(io.BytesIO(content), engine='openpyxl')
    else:
        raise ValueError("Unsupported file format. Only CSV, XLS, XLSX are supported.")

# Backtesting FGI strategy
def backtest_fgi_strategy(df, buy_threshold, sell_threshold,
                          initial_capital=10_000_000, commission_rate=0.0025):
    cash = initial_capital
    units = 0
    current_state = 'CASH'
    portfolio = []
    dates = []
    trade_count = 0
    fgi_low_seen = False
    fgi_high_seen = False
    pending_action = ('Buy', 0)

    for i in range(len(df)):
        date_i = pd.to_datetime(df['date'].iloc[i])
        fgi_i = df['fgi'].iloc[i]
        price_i = df['price'].iloc[i]
        has_next = (i < len(df) - 1)
        # Execute yesterday's decision
        if pending_action and pending_action[1] == i - 1:
            act, _ = pending_action
            if act == 'Buy' and price_i > 0:
                buy_amt = cash / (1 + commission_rate)
                units = buy_amt / price_i
                cash = 0
                current_state = 'STOCK'
                trade_count += 1
                fgi_low_seen = False
                fgi_high_seen = False
            elif act == 'Sell' and units > 0:
                sell_amt = units * price_i
                cash = sell_amt * (1 - commission_rate)
                units = 0
                current_state = 'CASH'
                trade_count += 1
                fgi_low_seen = False
                fgi_high_seen = False
            pending_action = None
        # Generate signal for next day
        if has_next and pending_action is None and not pd.isna(fgi_i):
            if current_state == 'CASH':
                if fgi_i < buy_threshold:
                    fgi_low_seen = True
                if fgi_low_seen and fgi_i > buy_threshold:
                    pending_action = ('Buy', i)
            else:
                if fgi_i > sell_threshold:
                    fgi_high_seen = True
                if fgi_high_seen and fgi_i < sell_threshold:
                    pending_action = ('Sell', i)
        # Record portfolio value
        if current_state == 'CASH':
            pv = cash
        else:
            pv = units * price_i if price_i > 0 else np.nan
        portfolio.append(pv)
        dates.append(date_i)
    series = pd.Series(portfolio, index=dates).dropna()
    if series.empty:
        return {'Strategy': f'B<{buy_threshold}>S<{sell_threshold}>', 'ROI':0, 'CAGR':0,
                'MDD':0, 'Calmar':0, 'Trades':trade_count, 'Final Value': initial_capital}
    final_val = series.iloc[-1]
    roi = final_val / initial_capital - 1
    days = (series.index[-1] - series.index[0]).days
    years = days / 365.25 if days>0 else 1/365
    cagr = (final_val / initial_capital) ** (1/years) - 1
    mdd = ((series - series.cummax()) / series.cummax()).min()
    calmar = cagr / abs(mdd) if mdd!=0 else np.nan
    return {'Strategy': f'B<{buy_threshold}>S<{sell_threshold}>', 'ROI':roi,
            'CAGR':cagr, 'MDD':mdd, 'Calmar':calmar,
            'Trades':trade_count, 'Final Value': final_val}

# Multi-objective fitness calculation
def calculate_multi_fitness(individual, df, initial_capital,
                            commission_rate, weights=None):
    buy_th, sell_th = individual
    if weights is None:
        weights = {'roi':0.5,'mdd':0.2,'calmar':0.15,'trades':0.05,'volatility':0.1}
    # Normalize weights
    total = sum(weights.values())
    weights = {k:v/total for k,v in weights.items()}
    res = backtest_fgi_strategy(df, buy_th, sell_th,
                                initial_capital, commission_rate)
    # volatility
    port = pd.Series(res.get('portfolio_values', []))
    ret = port.pct_change().dropna()
    vol = ret.std() if len(ret)>1 else 0
    # trades score
    tr = res['Trades']
    if tr<3: ts=0.3
    elif tr<=20: ts=1
    else: ts=0.5
    comps = {'roi':res['ROI'], 'mdd':-abs(res['MDD']),
             'calmar':min(res['Calmar'],10)/10, 'trades':ts,
             'volatility':-min(vol,0.5)/0.5}
    comp_score = sum(weights[m]*comps[m] for m in comps)
    return {'individual':individual, 'metrics':res,
            'fitness_components':comps, 'composite':comp_score}

# GA progress plot
def plot_ga_progress(best_fitness_history, avg_fitness_history,
                     buy_thresh_history, sell_thresh_history,
                     generation_count=None, save_path='ga_progress.png'):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Fitness
    axs[0,0].plot(best_fitness_history, label='Best Fitness')
    axs[0,0].plot(avg_fitness_history, label='Average Fitness')
    axs[0,0].set_title('GA Fitness Progress')
    axs[0,0].legend()
    # Buy threshold evolution
    axs[0,1].plot(buy_thresh_history)
    axs[0,1].set_title('Buy Threshold over Generations')
    axs[0,1].set_xlabel('Generation')
    # Sell threshold evolution
    axs[1,0].plot(sell_thresh_history)
    axs[1,0].set_title('Sell Threshold over Generations')
    axs[1,0].set_xlabel('Generation')
    # Hide unused subplot
    axs[1,1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# Example main pipeline
def main():
    name, uploaded = upload_file()
    df = read_uploaded_file(uploaded)
    # Ensure date format
    df['date'] = pd.to_datetime(df['date'])
    # Run a sample backtest
    result = backtest_fgi_strategy(df, buy_threshold=16, sell_threshold=13)
    print("Sample Strategy Result:", result)
    # Example GA structures would go here...

if __name__ == "__main__":
    main()
