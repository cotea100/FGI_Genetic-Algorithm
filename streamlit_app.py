import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
from datetime import datetime

st.set_page_config(page_title='FGI Strategy Backtester', layout='wide')

# -- Helper functions --------------------------------------------------------
def read_data(uploaded_file):
    """Read CSV or Excel into DataFrame with date, price, fgi columns."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None
    # Ensure required columns
    for col in ['date','price','fgi']:
        if col not in df.columns:
            st.error(f"필수 컬럼 '{col}' 없음")
            return None
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def backtest_fgi(df, buy_th, sell_th, init_cap=10_000_000, commission=0.0025):
    cash = init_cap
    units = 0
    state = 'CASH'
    port_vals = []
    dates = []
    trades = 0
    fgi_low_seen = False
    fgi_high_seen = False
    pending = ('Buy', 0)
    for i in range(len(df)):
        date_i = df['date'].iloc[i]
        fgi_i = df['fgi'].iloc[i]
        price_i = df['price'].iloc[i]
        has_next = i < len(df)-1
        # Execute pending
        if pending and pending[1] == i-1:
            act = pending[0]
            if act=='Buy' and price_i>0:
                amt = cash/(1+commission)
                units = amt/price_i
                cash=0
                state='STOCK'
                trades+=1
                fgi_low_seen=False
                fgi_high_seen=False
            elif act=='Sell' and units>0:
                amt=units*price_i
                cash=amt*(1-commission)
                units=0
                state='CASH'
                trades+=1
                fgi_low_seen=False
                fgi_high_seen=False
            pending=None
        # Signal
        if has_next and pending is None and not np.isnan(fgi_i):
            if state=='CASH':
                if fgi_i<buy_th: fgi_low_seen=True
                if fgi_low_seen and fgi_i>buy_th:
                    pending=('Buy',i)
            else:
                if fgi_i>sell_th: fgi_high_seen=True
                if fgi_high_seen and fgi_i<sell_th:
                    pending=('Sell',i)
        # Portfolio val
        pv = cash if state=='CASH' else units*price_i
        port_vals.append(pv)
        dates.append(date_i)
    series = pd.Series(port_vals, index=dates).dropna()
    if series.empty:
        return None
    final = series.iloc[-1]
    roi = final/init_cap-1
    days=(series.index[-1]-series.index[0]).days
    yrs = days/365.25 if days>0 else 1/365
    cagr=(final/init_cap)**(1/yrs)-1
    mdd=(series-series.cummax())/series.cummax()
    mdd_val=mdd.min()
    return {
        'portfolio':series,
        'ROI':roi, 'CAGR':cagr,
        'MDD':mdd_val, 'Trades':trades
    }

# -- UI ----------------------------------------------------------------------
st.title('FGI 기반 매매 전략 백테스터')

# Sidebar inputs
st.sidebar.header('데이터 입력')
file = st.sidebar.file_uploader('CSV 또는 Excel 파일 업로드', type=['csv','xls','xlsx'])

st.sidebar.header('전략 파라미터')
buy_th = st.sidebar.slider('매수 기준 FGI <', min_value=0, max_value=100, value=16)
sell_th = st.sidebar.slider('매도 기준 FGI >', min_value=0, max_value=100, value=13)
init_cap = st.sidebar.number_input('초기 자본(원)', min_value=1000, value=10_000_000, step=1000)
commission = st.sidebar.number_input('수수료율', min_value=0.0, max_value=0.05, value=0.0025, step=0.0001)

if file:
    df = read_data(file)
    if df is not None:
        st.subheader('데이터 미리보기')
        st.dataframe(df.head())

        if st.button('백테스트 실행'):
            with st.spinner('전략을 실행 중입니다...'):
                res = backtest_fgi(df, buy_th, sell_th, init_cap, commission)
            if res:
                st.subheader('전략 성과')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('ROI', f"{res['ROI']*100:.2f}%")
                col2.metric('CAGR', f"{res['CAGR']*100:.2f}%")
                col3.metric('MDD', f"{res['MDD']*100:.2f}%")
                col4.metric('거래 횟수', res['Trades'])

                # 누적 수익률 차트
                st.subheader('누적 수익률 차트')
                fig, ax = plt.subplots(figsize=(8,4))
                (res['portfolio']/init_cap).plot(ax=ax)
                ax.set_ylabel('누적 수익률')
                ax.grid(True)
                st.pyplot(fig)

                # 드로우다운
                st.subheader('Drawdown')
                draw = (res['portfolio'] - res['portfolio'].cummax())/res['portfolio'].cummax()
                fig2, ax2 = plt.subplots(figsize=(8,4))
                draw.plot(ax=ax2)
                ax2.set_ylabel('Drawdown')
                ax2.grid(True)
                st.pyplot(fig2)

                # PDF 다운로드
                pdf_buf = io.BytesIO()
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', size=12)
                pdf.cell(0, 10, 'FGI Backtest Report', ln=True)
                pdf.ln(5)
                pdf.cell(0, 8, f"매수: FGI<{buy_th}, 매도: FGI>{sell_th}", ln=True)
                pdf.cell(0, 8, f"ROI: {res['ROI']*100:.2f}%", ln=True)
                pdf.cell(0, 8, f"CAGR: {res['CAGR']*100:.2f}%", ln=True)
                pdf.cell(0, 8, f"MDD: {res['MDD']*100:.2f}%", ln=True)
                pdf.cell(0, 8, f"Trades: {res['Trades']}", ln=True)
                pdf.output(pdf_buf)
                pdf_buf.seek(0)
                st.download_button('PDF 리포트 다운로드', data=pdf_buf,
                                    file_name='fgi_backtest_report.pdf',
                                    mime='application/pdf')
else:
    st.info('왼쪽 패널에서 파일을 업로드하세요.')
