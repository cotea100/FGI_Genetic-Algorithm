import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# Simplified backtesting function
def backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital=10_000_000, commission_rate=0.0025):
    cash = initial_capital
    units = 0
    current_state = 'CASH'
    portfolio = []
    dates = []
    fgi_low_seen = False
    fgi_high_seen = False
    pending = ('Buy', 0)

    for i in range(len(df)):
        date_i = df['date'].iloc[i]
        fgi_i = df['fgi'].iloc[i]
        price_i = df['price'].iloc[i]
        has_next = i < len(df) - 1

        # Execute pending action
        if pending and pending[1] == i-1:
            action = pending[0]
            if action == 'Buy' and price_i > 0:
                buy_amt = cash / (1 + commission_rate)
                units = buy_amt / price_i
                cash = 0
                current_state = 'STOCK'
                fgi_low_seen = False
                fgi_high_seen = False
            elif action == 'Sell' and units > 0:
                sell_amt = units * price_i
                cash = sell_amt * (1 - commission_rate)
                units = 0
                current_state = 'CASH'
                fgi_low_seen = False
                fgi_high_seen = False
            pending = None

        # Decision logic
        if has_next and not pending:
            if current_state == 'CASH':
                if fgi_i < buy_threshold:
                    fgi_low_seen = True
                if fgi_low_seen and fgi_i > buy_threshold:
                    pending = ('Buy', i)
            else:
                if fgi_i > sell_threshold:
                    fgi_high_seen = True
                if fgi_high_seen and fgi_i < sell_threshold:
                    pending = ('Sell', i)

        # Record portfolio
        value = cash if current_state == 'CASH' else units * price_i
        portfolio.append(value)
        dates.append(date_i)

    final = portfolio[-1]
    roi = (final / initial_capital) - 1
    return final, roi, dates, portfolio

# Streamlit UI
st.title("FGI Strategy App - Updated")

# GA integration stub
use_ga = st.sidebar.checkbox("Use GA Optimization", value=False)
if use_ga:
    st.sidebar.subheader("GA Parameters")
    pop_size = st.sidebar.number_input("Population Size", 10, 200, 50, step=10)
    gens = st.sidebar.number_input("Generations", 10, 200, 80, step=10)
    mut_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.25, 0.01)
    parent_count = st.sidebar.number_input("Parents Count", 2, pop_size, int(pop_size*0.25))
    run_label = "Run GA Backtest"
else:
    run_label = "Run Single Backtest"

uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=['csv','xls','xlsx'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    cols = df.columns.tolist()
    date_col = st.selectbox("Select date column", cols)
    price_col = st.selectbox("Select price column", cols)
    fgi_col = st.selectbox("Select FGI column", cols)

    df = df[[date_col, price_col, fgi_col]].copy()
    df.columns = ['date', 'price', 'fgi']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df[['price','fgi']] = df[['price','fgi']].interpolate()

    st.write("Date range:", df['date'].min(), "to", df['date'].max())

    initial_capital = st.number_input("Initial capital (KRW)", value=10000000)
    commission_rate = st.number_input("Commission rate", value=0.0025, format="%.4f")

    if not use_ga:
        buy_threshold = st.slider("Buy threshold", int(df['fgi'].min()), int(df['fgi'].max()), 5)
        sell_threshold = st.slider("Sell threshold", int(df['fgi'].min()), int(df['fgi'].max()), 13)

    if st.button(run_label):
        if use_ga:
            # GA optimization call (user implements run_genetic_algorithm)
            st.info("GA optimization is not implemented in this stub.")
        else:
            final, roi, dates, portfolio = backtest_fgi_strategy(
                df, buy_threshold, sell_threshold, initial_capital, commission_rate
            )
            final_value = float(final)
            roi_value = float(roi)

            st.write(f"Final value: {final_value:,.0f} KRW")
            st.write(f"ROI: {roi_value:.2%}")

            fig, ax = plt.subplots()
            ax.plot(dates, portfolio)
            ax.set_title("Portfolio Value Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.grid(True)
            st.pyplot(fig)

            # PDF in-memory
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial",'B',16)
            pdf.cell(0,10,"FGI Strategy Backtest Report",0,1,'C')
            pdf.ln(10)
            pdf.set_font("Arial",'',12)
            pdf.cell(0,8,f"Buy threshold: {buy_threshold}",0,1)
            pdf.cell(0,8,f"Sell threshold: {sell_threshold}",0,1)
            pdf.cell(0,8,f"Initial capital: {initial_capital}",0,1)
            pdf.cell(0,8,f"Final value: {final_value:,.0f}",0,1)
            pdf.cell(0,8,f"ROI: {roi_value:.2%}",0,1)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"report_{now}.pdf",
                mime="application/pdf"
            )