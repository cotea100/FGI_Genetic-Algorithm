import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# Backtesting function (simplified for Streamlit)
def backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital=10_000_000, commission_rate=0.0025):
    cash = initial_capital
    units = 0
    current_state = 'CASH'
    portfolio = []
    dates = []
    fgi_low_seen_since_sell = False
    fgi_high_seen_since_buy = False
    pending_action = ('Buy', 0)

    for i in range(len(df)):
        date_i = df.loc[i, 'date']
        fgi_i = df.loc[i, 'fgi']
        price_i = df.loc[i, 'price']
        has_next = (i < len(df) - 1)

        # Execute pending action
        if pending_action and pending_action[1] == i - 1:
            action, _ = pending_action
            if action == 'Buy':
                buy_amount = cash / (1 + commission_rate)
                units = buy_amount / price_i
                cash = 0
                current_state = 'STOCK'
                fgi_low_seen_since_sell = False
                fgi_high_seen_since_buy = False
            elif action == 'Sell':
                sell_amount = units * price_i
                cash = sell_amount * (1 - commission_rate)
                units = 0
                current_state = 'CASH'
                fgi_low_seen_since_sell = False
                fgi_high_seen_since_buy = False
            pending_action = None

        # Decide next action
        if has_next and pending_action is None:
            if current_state == 'CASH':
                if fgi_i < buy_threshold:
                    fgi_low_seen_since_sell = True
                if fgi_low_seen_since_sell and fgi_i > buy_threshold:
                    pending_action = ('Buy', i)
            else:
                if fgi_i > sell_threshold:
                    fgi_high_seen_since_buy = True
                if fgi_high_seen_since_buy and fgi_i < sell_threshold:
                    pending_action = ('Sell', i)

        # Record portfolio value
        portfolio.append(cash if current_state == 'CASH' else units * price_i)
        dates.append(date_i)

    final = portfolio[-1]
    roi = (final / initial_capital) - 1
    return final, roi, dates, portfolio

# Streamlit UI
st.title("FGI Strategy Genetic Optimization")

uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=['csv','xls','xlsx'])
if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    # Column selection
    cols = df.columns.tolist()
    date_col = st.selectbox("Select date column", cols)
    price_col = st.selectbox("Select price column", cols)
    fgi_col = st.selectbox("Select FGI column", cols)

    # Subset and rename
    df = df[[date_col, price_col, fgi_col]].copy()
    df.columns = ['date', 'price', 'fgi']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df[['price', 'fgi']] = df[['price', 'fgi']].interpolate()

    st.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Inputs
    initial_capital = st.number_input("Initial capital (KRW)", value=10000000)
    commission_rate = st.number_input("Commission rate", value=0.0025, format="%.4f")

    fgi_min, fgi_max = int(df['fgi'].min()), int(df['fgi'].max())
    buy_threshold = st.slider("Buy threshold", fgi_min, fgi_max, fgi_min + 5)
    sell_threshold = st.slider("Sell threshold", fgi_min, fgi_max, fgi_min + 13)

    if st.button("Run Backtest"):
        final, roi, dates, portfolio = backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital, commission_rate)
        st.write(f"Final value: {final:,.0f} KRW")
        st.write(f"ROI: {roi:.2%}")

        # Plot
        fig, ax = plt.subplots()
        ax.plot(dates, portfolio)
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True)
        st.pyplot(fig)

        # PDF generation in memory
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "FGI Strategy Backtest Report", 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, f"Buy threshold: {buy_threshold}", 0, 1)
        pdf.cell(0, 8, f"Sell threshold: {sell_threshold}", 0, 1)
        pdf.cell(0, 8, f"Initial capital: {initial_capital}", 0, 1)
        pdf.cell(0, 8, f"Final value: {final:,.0f}", 0, 1)
        pdf.cell(0, 8, f"ROI: {roi:.2%}", 0, 1)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        report_name = f"report_{now}.pdf"
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=report_name,
            mime="application/pdf"
        )