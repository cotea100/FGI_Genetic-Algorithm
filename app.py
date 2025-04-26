import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# Genetic algorithm and backtesting functions (simplified)
def backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital=10_000_000, commission_rate=0.0025):
    cash = initial_capital
    units = 0
    current_state = 'CASH'
    portfolio = []
    dates = []
    trade_count = 0
    fgi_low_seen_since_sell = False
    fgi_high_seen_since_buy = False
    pending_action = ('Buy', 0)

    for i in range(len(df)):
        date_i = df['date'].iloc[i]
        fgi_i = df['fgi'].iloc[i]
        price_i = df['price'].iloc[i]
        has_next = (i < len(df) - 1)

        # Execute pending action
        if pending_action and pending_action[1] == i-1:
            action, _ = pending_action
            if action == 'Buy':
                buy_amount = cash / (1 + commission_rate)
                units = buy_amount / price_i
                cash = 0
                current_state = 'STOCK'
                trade_count += 1
                fgi_low_seen_since_sell = False
                fgi_high_seen_since_buy = False
            if action == 'Sell':
                sell_amount = units * price_i
                cash = sell_amount * (1 - commission_rate)
                units = 0
                current_state = 'CASH'
                trade_count += 1
                fgi_low_seen_since_sell = False
                fgi_high_seen_since_buy = False
            pending_action = None

        # Decide next
        if has_next and not pending_action:
            if current_state == 'CASH':
                if fgi_i < buy_threshold: fgi_low_seen_since_sell = True
                if fgi_low_seen_since_sell and fgi_i > buy_threshold:
                    pending_action = ('Buy', i)
            else:
                if fgi_i > sell_threshold: fgi_high_seen_since_buy = True
                if fgi_high_seen_since_buy and fgi_i < sell_threshold:
                    pending_action = ('Sell', i)

        # Record portfolio
        if current_state == 'CASH':
            portfolio.append(cash)
        else:
            portfolio.append(units * price_i)
        dates.append(date_i)

    final = portfolio[-1]
    roi = (final / initial_capital) - 1
    return final, roi, dates, portfolio

# Streamlit UI
st.title("FGI Strategy Genetic Optimization")

uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=['csv','xls','xlsx'])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, engine='openpyxl')
    st.write("Columns:", df.columns.tolist())
    date_col = st.selectbox("Select date column", df.columns)
    price_col = st.selectbox("Select price column", df.columns)
    fgi_col = st.selectbox("Select FGI column", df.columns)

    df = df[[date_col, price_col, fgi_col]].rename(columns={date_col:'date', price_col:'price', fgi_col:'fgi'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df[['price','fgi']] = df[['price','fgi']].interpolate()

    st.write("Date range:", df['date'].min(), "to", df['date'].max())

    initial_capital = st.number_input("Initial capital (KRW)", value=10000000)
    commission_rate = st.number_input("Commission rate", value=0.0025, format="%.4f")

    buy_threshold = st.slider("Buy threshold", int(df['fgi'].min()), int(df['fgi'].max()), 5)
    sell_threshold = st.slider("Sell threshold", int(df['fgi'].min()), int(df['fgi'].max()), 13)

    if st.button("Run Backtest"):
        final, roi, dates, portfolio = backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital, commission_rate)
        st.write(f"Final value: {final:,.0f} KRW")
        st.write(f"ROI: {roi:.2%}")

        # Plot portfolio over time
        fig, ax = plt.subplots()
        ax.plot(dates, portfolio)
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True)
        st.pyplot(fig)

        # Generate PDF report
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
        pdf_file = f"report_{now}.pdf"
        pdf_output_path = f"/mnt/data/{pdf_file}"
        pdf.output(pdf_output_path)

        with open(pdf_output_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=pdf_file,
                mime="application/pdf"
            )