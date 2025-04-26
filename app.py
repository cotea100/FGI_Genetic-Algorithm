
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
import time
import random
from datetime import datetime

# --- Genetic Algorithm functions from original code ---

def backtest_fgi_strategy(df, buy_threshold, sell_threshold, initial_capital=10_000_000, commission_rate=0.0025):
    cash = initial_capital
    units = 0
    state = 'CASH'
    portfolio = []
    dates = []
    fgi_low = False
    fgi_high = False
    pending = ('Buy', 0)

    for i in range(len(df)):
        date_i = df['date'].iloc[i]
        fgi_i = df['fgi'].iloc[i]
        price_i = df['price'].iloc[i]
        has_next = i < len(df) - 1

        if pending and pending[1] == i-1:
            if pending[0] == 'Buy':
                buy_amt = cash / (1 + commission_rate)
                units = buy_amt / price_i
                cash = 0
                state = 'STOCK'
                fgi_low = fgi_high = False
            else:
                cash = units * price_i * (1 - commission_rate)
                units = 0
                state = 'CASH'
                fgi_low = fgi_high = False
            pending = None

        if has_next and not pending:
            if state == 'CASH':
                if fgi_i < buy_threshold: fgi_low = True
                if fgi_low and fgi_i > buy_threshold:
                    pending = ('Buy', i)
            else:
                if fgi_i > sell_threshold: fgi_high = True
                if fgi_high and fgi_i < sell_threshold:
                    pending = ('Sell', i)

        value = cash if state=='CASH' else units * price_i
        portfolio.append(value)
        dates.append(date_i)

    final = portfolio[-1]
    roi = final / initial_capital - 1
    return final, roi, dates, portfolio

def selection_multi_objective(pop, fit_results, num):
    elite = [tuple(max(fit_results, key=lambda x: x['composite'])['individual'])]
    for m in ['roi','mdd','calmar','trades','volatility']:
        if len(elite)>=num: break
        best = max(fit_results, key=lambda x: x['fitness_components'][m] if m in x['fitness_components'] else -999)
        if tuple(best['individual']) not in elite:
            elite.append(tuple(best['individual']))
    while len(elite)<num and pop:
        ind = random.choice(pop)
        if ind not in elite: elite.append(ind)
    return elite

def calculate_multi_fitness(ind, df, init_cap, comm, weights):
    buy, sell = ind
    final, roi, _, _ = backtest_fgi_strategy(df, buy, sell, init_cap, comm)
    # dummy mdd, calmar, trades, volatility
    mdd=-0.1; calmar=roi/0.1; trades=10; volatility=0.05
    comps = {'roi':roi,'mdd':-abs(mdd),'calmar':min(calmar,10)/10,'trades':0.8,'volatility':-volatility/0.5}
    comp = sum(comps[k]*weights[k] for k in weights)
    return {'individual':ind,'composite':comp,'fitness_components':comps,'metrics':{'roi':roi,'mdd':mdd,'calmar':calmar,'trades':trades,'volatility':volatility}}

def create_initial_population(size, brange, srange):
    return [(random.randint(*brange), random.randint(*srange)) for _ in range(size)]

def run_genetic_algorithm(df, init_cap, comm, pop_size, gens, mut_rate, parent_count, weights):
    pop = create_initial_population(pop_size, (5,25),(5,25))
    best_hist=[]; avg_hist=[]; buy_hist=[]; sell_hist=[]
    for g in range(gens):
        fits = [calculate_multi_fitness(ind, df, init_cap, comm, weights) for ind in pop]
        comps = [f['composite'] for f in fits]
        avg_hist.append(np.mean(comps)); best_hist.append(max(comps))
        best_idx=np.argmax(comps)
        best=pop[best_idx]
        buy_hist.append(best[0]); sell_hist.append(best[1])
        parents = selection_multi_objective(pop, fits, parent_count)
        off = []
        while len(off)<pop_size-len(parents):
            p1,p2=random.sample(parents,2)
            off.append((p1[0] if random.random()<0.5 else p2[0], p1[1] if random.random()<0.5 else p2[1]))
        # mutation
        pop = parents + [(max(5,min(x[0]+random.randint(-3,3),25)), max(5,min(x[1]+random.randint(-3,3),25))) for x in off]
    return best, best_idx, best_hist, avg_hist, buy_hist, sell_hist

# --- Streamlit UI ---

st.title("FGI Strategy Genetic Optimization")

# Sidebar GA params
use_ga = st.sidebar.checkbox("Enable GA Optimization", value=False)
if use_ga:
    pop_size = st.sidebar.number_input("Population Size", 10,200,50,10)
    gens = st.sidebar.number_input("Generations", 10,200,80,10)
    mut_rate = st.sidebar.slider("Mutation Rate", 0.0,1.0,0.25,0.01)
    parent_count = st.sidebar.number_input("Parents Count",1,pop_size,int(pop_size*0.2))
    weights = {'roi':0.5,'mdd':0.2,'calmar':0.15,'trades':0.05,'volatility':0.1}

uploaded = st.file_uploader("Upload CSV/XLSX", type=['csv','xls','xlsx'])
if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded, engine='openpyxl')
    cols=list(df.columns)
    date_col=st.selectbox("Date column",cols); price_col=st.selectbox("Price column",cols); fgi_col=st.selectbox("FGI column",cols)
    df=df[[date_col,price_col,fgi_col]].copy(); df.columns=['date','price','fgi']
    df['date']=pd.to_datetime(df['date']); df=df.sort_values('date').reset_index(drop=True)
    df[['price','fgi']]=df[['price','fgi']].interpolate()
    init_cap=st.number_input("Initial capital",10000000); comm=st.number_input("Commission rate",0.0025,format="%.4f")
    if use_ga:
        if st.button("Run GA"):
            with st.spinner("Optimizing..."):
                best, idx, bh, ah, buy_h, sell_h = run_genetic_algorithm(df, init_cap, comm, pop_size, gens, mut_rate, parent_count, weights)
            st.write(f"Best thresholds: Buy>{best[0]}, Sell<{best[1]}")
            # Plot progress
            fig1,ax1=plt.subplots(); ax1.plot(bh,label='Best'); ax1.plot(ah,label='Avg'); ax1.legend(); ax1.set_title("GA Fitness")
            st.pyplot(fig1)
            # Backtest best
            final, roi, dates, port = backtest_fgi_strategy(df, best[0], best[1], init_cap, comm)
            st.write(f"Final: {final:,.0f} KRW ROI: {roi:.2%}")
            fig2,ax2=plt.subplots(); ax2.plot(dates,port); ax2.set_title("Portfolio"); st.pyplot(fig2)
            # PDF generation
            pdf=FPDF(); pdf.add_page()
            pdf.set_font("Arial",'B',14); pdf.cell(0,10,"GA Optimization Report",0,1,'C')
            pdf.set_font("Arial","",12)
            pdf.cell(0,8,f"Buy>{best[0]}, Sell<{best[1]}",0,1)
            # embed figures
            img1=BytesIO(); fig1.savefig(img1,format='PNG'); img1.seek(0)
            pdf.image(img1, x=10, y=50, w=190)
            img2=BytesIO(); fig2.savefig(img2,format='PNG'); img2.seek(0)
            pdf.add_page(); pdf.image(img2, x=10, y=20, w=190)
            pdf_bytes=pdf.output(dest='S').encode('latin-1')
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"GA_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    else:
        buy=st.slider("Buy threshold",int(df['fgi'].min()),int(df['fgi'].max()),5)
        sell=st.slider("Sell threshold",int(df['fgi'].min()),int(df['fgi'].max()),13)
        if st.button("Run Backtest"):
            final, roi, dates, port = backtest_fgi_strategy(df, buy, sell, init_cap, comm)
            st.write(f"Final: {final:,.0f} KRW ROI: {roi:.2%}")
            fig,ax=plt.subplots(); ax.plot(dates,port); ax.set_title("Portfolio"); st.pyplot(fig)
            pdf=FPDF(); pdf.add_page()
            pdf.set_font("Arial",'B',14); pdf.cell(0,10,"Backtest Report",0,1,'C')
            pdf.set_font("Arial","",12)
            pdf.cell(0,8,f"Buy>{buy}, Sell<{sell}",0,1)
            img=BytesIO(); fig.savefig(img,format='PNG'); img.seek(0)
            pdf.image(img, x=10, y=50, w=190)
            pdf_bytes=pdf.output(dest='S').encode('latin-1')
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
