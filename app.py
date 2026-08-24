import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# --- 1. 頁面設定 ---
st.set_page_config(page_title="終極股市看板", layout="wide", page_icon="📈")
st.title("📈 終極股市看板 (全指標分析 + 策略回測)")

# 初始化 Session State
if 'backtest_result' not in st.session_state:
    st.session_state.backtest_result = None
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None

# --- 2. 側邊欄設定 ---
st.sidebar.header("查詢設定")
market_type = st.sidebar.radio("1️⃣ 請選擇市場", ["🇹🇼 台股 (Taiwan)", "🇺🇸 美股 (US)"], horizontal=True)

tw_stocks = {
    "2330 台積電": "2330.TW", 
    "🔍 自行輸入代號": "custom", 
    "2317 鴻海": "2317.TW", 
    "2454 聯發科": "2454.TW",
    "2603 長榮": "2603.TW", 
    "2382 廣達": "2382.TW", 
    "3231 緯創": "3231.TW",
    "2327 國巨": "2327.TW",
    "0050 元大台灣50": "0050.TW", 
    "0056 元大高股息": "0056.TW", 
    "2408 南亞科": "2408.TW", 
    "2344 華邦電": "2344.TW"
}

us_stocks = {
    "NVDA (NVIDIA)": "NVDA",
    "🔍 自行輸入代號": "custom",
    "AAPL (Apple)": "AAPL",
    "TSLA (Tesla)": "TSLA",
    "MSFT (Microsoft)": "MSFT", 
    "AMD (AMD)": "AMD", 
    "QQQ (Nasdaq 100)": "QQQ", 
    "SPY (S&P 500)": "SPY", 
    "SOXX (Semiconductor)": "SOXX", 
    "TQQQ (3x Long QQQ)": "TQQQ"
}

current_dict = tw_stocks if "台股" in market_type else us_stocks
options_list = list(current_dict.keys())
default_option = "2330 台積電" if "台股" in market_type else "NVDA (NVIDIA)"
default_index = options_list.index(default_option) if default_option in options_list else 0

selected_label = st.sidebar.selectbox("2️⃣ 搜尋或選擇股票", options=options_list, index=default_index)

if current_dict[selected_label] == "custom":
    raw_input = st.sidebar.text_input("請輸入代號 (例如: 2330, 8299.TWO 或 NVDA)")
    if raw_input:
        cleaned = raw_input.strip()
        if "台股" in market_type:
            if cleaned.isdigit():
                ticker_input = f"{cleaned}.TW"
            else:
                ticker_input = cleaned.upper()
        else:
            ticker_input = cleaned.upper()
    else:
        ticker_input = None
else:
    ticker_input = current_dict[selected_label]

if ticker_input != st.session_state.last_ticker:
    st.session_state.backtest_result = None
    st.session_state.last_ticker = ticker_input

period = st.sidebar.selectbox("3️⃣ 資料時間範圍", ("3mo", "6mo", "1y", "2y", "5y", "10y", "20y", "max"), index=2)

# --- 3. 指標計算 ---
def calculate_indicators(df):
    df = df.copy()
    # 均線與布林通道
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['std']
    df['BB_Lower'] = df['MA20'] - 2 * df['std']

    # KD 指標
    min_9 = df['Low'].rolling(window=9).min()
    max_9 = df['High'].rolling(window=9).max()
    denom = (max_9 - min_9).replace(0, np.nan)
    df['RSV'] = ((df['Close'] - min_9) / denom * 100).fillna(50)
    
    k_vals, d_vals = [50.0], [50.0]
    for rsv in df['RSV'].iloc[1:]:
        k = (2/3) * k_vals[-1] + (1/3) * rsv
        d = (2/3) * d_vals[-1] + (1/3) * k
        k_vals.append(k)
        d_vals.append(d)
    df['K'] = k_vals
    df['D'] = d_vals

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEA']

    # RSI
    def get_rsi(series, p):
        delta = series.diff()
        u = delta.clip(lower=0)
        d = -1 * delta.clip(upper=0)
        ema_u = u.ewm(com=p - 1, adjust=False).mean()
        ema_d = d.ewm(com=p - 1, adjust=False).mean()
        rs = ema_u / ema_d.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    df['RSI6'] = get_rsi(df['Close'], 6)
    df['RSI12'] = get_rsi(df['Close'], 12)
    df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100

    # DMI / ADX
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    alpha = 1 / 14
    df['TR14'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DM14'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['+DI'] = (df['+DM14'] / df['TR14'].replace(0, np.nan) * 100).fillna(0)
    df['-DI'] = (df['-DM14'] / df['TR14'].replace(0, np.nan) * 100).fillna(0)
    
    di_sum = (df['+DI'] + df['-DI']).replace(0, np.nan)
    df['DX'] = (abs(df['+DI'] - df['-DI']) / di_sum * 100).fillna(0)
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    return df

# --- 4. 智能訊號分析 ---
def analyze_signals(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    score = 0 

    # 均線趨勢
    if last['Close'] > last['MA20'] and last['Close'] > last['MA60']:
        signals.append(("均線趨勢", "多頭排列 (站上月/季線)", "偏多", "red"))
        score += 2
    elif last['Close'] < last['MA20'] and last['Close'] < last['MA60']:
        signals.append(("均線趨勢", "空頭排列 (跌破月/季線)", "偏空", "green"))
        score -= 2
    else:
        signals.append(("均線趨勢", "均線糾結震盪", "中立", "gray"))

    # 成交量
    if last['Volume'] > 1.5 * last['Vol_MA5']:
        signals.append(("成交量能", "爆量 (>5日均量1.5倍)", "人氣匯集", "red"))
        score += 0.5
    elif last['Volume'] < 0.6 * last['Vol_MA5']:
        signals.append(("成交量能", "量縮 (<5日均量0.6倍)", "觀望", "gray"))
    else:
        signals.append(("成交量能", "量能溫和", "正常", "gray"))

    # 布林通道
    if last['Close'] > last['BB_Upper']:
        signals.append(("布林通道", "突破上軌", "強勢/超買", "red"))
        score += 0.5
    elif last['Close'] < last['BB_Lower']:
        signals.append(("布林通道", "跌破下軌", "弱勢/超賣", "green"))
        score -= 0.5
    else:
        signals.append(("布林通道", "通道內運行", "正常", "gray"))

    # KD
    if last['K'] > last['D'] and prev['K'] <= prev['D']:
        signals.append(("KD指標", f"黃金交叉 (K={last['K']:.1f})", "買進", "red"))
        score += 1.5
    elif last['K'] < last['D'] and prev['K'] >= prev['D']:
        signals.append(("KD指標", f"死亡交叉 (K={last['K']:.1f})", "賣出", "green"))
        score -= 1.5
    elif last['K'] > 80:
        signals.append(("KD指標", f"高檔鈍化 (K={last['K']:.1f})", "強勢/警戒", "orange"))
    elif last['K'] < 20:
        signals.append(("KD指標", f"低檔鈍化 (K={last['K']:.1f})", "弱勢/反彈", "blue"))
    else:
        signals.append(("KD指標", f"區間整理 (K={last['K']:.1f})", "中立", "gray"))

    # MACD
    if last['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0:
        signals.append(("MACD", "柱狀體翻紅", "轉強", "red"))
        score += 1
    elif last['MACD_Hist'] < 0 and prev['MACD_Hist'] >= 0:
        signals.append(("MACD", "柱狀體翻綠", "轉弱", "green"))
        score -= 1
    else:
        signals.append(("MACD", "動能持平", "中立", "gray"))

    # RSI
    if last['RSI6'] > 80:
        signals.append(("RSI", f"短線過熱 ({last['RSI6']:.1f})", "拉回風險", "green"))
        score -= 1
    elif last['RSI6'] < 20:
        signals.append(("RSI", f"短線超賣 ({last['RSI6']:.1f})", "反彈機會", "red"))
        score += 1
    else:
        signals.append(("RSI", f"數值中性 ({last['RSI6']:.1f})", "正常", "gray"))

    # 乖離率
    if last['BIAS20'] > 10:
        signals.append(("乖離率", f"正乖離偏高 ({last['BIAS20']:.1f}%)", "過熱拉回", "green"))
        score -= 1
    elif last['BIAS20'] < -10:
        signals.append(("乖離率", f"負乖離偏大 ({last['BIAS20']:.1f}%)", "跌深反彈", "red"))
        score += 1
    else:
        signals.append(("乖離率", f"正常 ({last['BIAS20']:.1f}%)", "中立", "gray"))

    # DMI/ADX
    if last['ADX'] > 25:
        trend = "多方" if last['+DI'] > last['-DI'] else "空方"
        color = "red" if trend == "多方" else "green"
        signals.append(("DMI/ADX", f"趨勢明確 ({trend} ADX={last['ADX']:.1f})", "延續", color))
        score += 1 if trend == "多方" else -1
    else:
        signals.append(("DMI/ADX", f"無明顯趨勢 (ADX={last['ADX']:.1f})", "盤整", "gray"))

    final_suggestion = "⏳ 觀望 / 中立"
    final_color = "gray"
    if score >= 3.5:
        final_suggestion = "🚀 強力買進"
        final_color = "red"
    elif score >= 1.5:
        final_suggestion = "📈 偏多操作"
        final_color = "red"
    elif score <= -3.5:
        final_suggestion = "📉 強力賣出"
        final_color = "green"
    elif score <= -1.5:
        final_suggestion = "💸 偏空/減碼"
        final_color = "green"

    return signals, final_suggestion, final_color

# --- 5. 回測引擎 ---
def run_backtest(df, strategy, param1, param2, initial_cash=1000000, market="TW"):
    cash = float(initial_cash)
    position = 0
    trade_log = []
    equity_curve = []
    
    bt_df = df.copy()
    bt_df['Raw_Signal'] = 0

    if strategy == "雙均線策略 (MA Crossover)":
        short_ma = bt_df['Close'].rolling(window=int(param1)).mean()
        long_ma = bt_df['Close'].rolling(window=int(param2)).mean()
        valid = short_ma.notna() & long_ma.notna()
        bt_df.loc[valid & (short_ma > long_ma), 'Raw_Signal'] = 1
        bt_df.loc[valid & (short_ma <= long_ma), 'Raw_Signal'] = 0

    elif strategy == "RSI 逆勢策略 (RSI Reversal)":
        rsi_data = bt_df['RSI6']
        holding = 0
        signals = []
        for r in rsi_data:
            if pd.isna(r):
                signals.append(0)
                continue
            if r < param1:
                holding = 1
            elif r > param2:
                holding = 0
            signals.append(holding)
        bt_df['Raw_Signal'] = signals

    # 次日執行 (Shift 1 防止未來函數)
    bt_df['Target_Position'] = bt_df['Raw_Signal'].shift(1).fillna(0)
    bt_df['Trade_Action'] = bt_df['Target_Position'].diff().fillna(0)

    fee_rate = 0.001425
    tax_rate = 0.003 if "TW" in market else 0.0

    for i in range(len(bt_df)):
        exec_price = bt_df['Open'].iloc[i] if not pd.isna(bt_df['Open'].iloc[i]) else bt_df['Close'].iloc[i]
        close_price = bt_df['Close'].iloc[i]
        date = bt_df.index[i]
        action = bt_df['Trade_Action'].iloc[i]

        # 買進
        if action == 1 and position == 0:
            max_shares = int(cash / (exec_price * (1 + fee_rate)))
            if max_shares > 0:
                cost = max_shares * exec_price
                fee = cost * fee_rate
                cash -= (cost + fee)
                position = max_shares
                trade_log.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Type': 'Buy',
                    'Price': round(exec_price, 2),
                    'Shares': max_shares,
                    'Fee': round(fee, 2),
                    'Tax': 0.0,
                    'Cash_Balance': round(cash, 2)
                })

        # 賣出
        elif action == -1 and position > 0:
            revenue = position * exec_price
            fee = revenue * fee_rate
            tax = revenue * tax_rate
            cash += (revenue - fee - tax)
            trade_log.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Type': 'Sell',
                'Price': round(exec_price, 2),
                'Shares': position,
                'Fee': round(fee, 2),
                'Tax': round(tax, 2),
                'Cash_Balance': round(cash, 2)
            })
            position = 0

        # 當日結算淨值
        current_equity = cash + (position * close_price)
        equity_curve.append(current_equity)

    bt_df['Equity'] = equity_curve
    final_val = equity_curve[-1]
    total_ret = (final_val - initial_cash) / initial_cash * 100
    trades_df = pd.DataFrame(trade_log)
    return bt_df, trades_df, total_ret, final_val

# --- 6. 資料獲取與防錯快取 ---
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker, period_choice):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period_choice)
        
        # 備援機制
        if df is None or df.empty:
            df = yf.download(ticker, period=period_choice, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if df is None or df.empty or len(df) < 5:
            return None, None

        # 獨立抓取 info，即使失敗也不中斷
        info = {}
        try:
            raw_info = stock.info
            if isinstance(raw_info, dict):
                info = raw_info
        except Exception:
            pass

        if not info:
            info = {
                'longName': ticker,
                'currency': 'TWD' if '.TW' in ticker or '.TWO' in ticker else 'USD'
            }

        df = calculate_indicators(df)
        return df, info
    except Exception as e:
        print(f"Fetch Error: {e}")
        return None, None

# --- 7. 主程式 ---
if ticker_input:
    with st.spinner(f"正在全速運算 {ticker_input} 所有數據..."):
        data, info = get_stock_data(ticker_input, period)

    if data is not None:
        tab1, tab2 = st.tabs(["📊 全方位市場儀表板", "🧪 策略回測實驗室"])

        # TAB 1: 儀表板
        with tab1:
            signal_list, suggestion, sugg_color = analyze_signals(data)
            col1, col2 = st.columns([3, 1])
            with col1:
                stock_name = info.get('longName', ticker_input) if info else ticker_input
                currency = info.get('currency', 'TWD') if info else 'TWD'
                current_price = data['Close'].iloc[-1]
                change = current_price - data['Close'].iloc[-2]
                pct_change = (change / data['Close'].iloc[-2]) * 100
                color_text = "red" if change >= 0 else "green"
                st.markdown(f"## {stock_name} ({ticker_input})")
                st.markdown(f"<h2 style='color:{color_text}; margin-top:-15px;'>{current_price:.2f} {currency} ({change:+.2f} / {pct_change:+.2f}%)</h2>", unsafe_allow_html=True)
            with col2:
                st.markdown("### 綜合建議")
                st.markdown(f"<h3 style='color:{sugg_color}; border: 2px solid {sugg_color}; padding: 8px; text-align: center; border-radius: 8px;'>{suggestion}</h3>", unsafe_allow_html=True)

            with st.expander("🤖 查看【8 大指標全方位智能診斷】", expanded=True):
                cols = st.columns(4)
                for i, (indicator, meaning, action, color) in enumerate(signal_list):
                    with cols[i % 4]:
                        st.markdown(f"**{indicator}**")
                        st.caption(meaning)
                        hex_color = {"red": "#FF4B4B", "green": "#21C354", "orange": "#FFA500", "blue": "#1E90FF"}.get(color, "#808080")
                        st.markdown(f"<span style='color:{hex_color}; font-weight:bold'>● {action}</span>", unsafe_allow_html=True)
                        st.write("---")

            st.subheader("技術分析圖表 (7層詳細版)")
            fig = make_subplots(
                rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.015,
                row_heights=[0.38, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12]
            )
            date_strings = data.index.strftime('%Y-%m-%d')

            # 1. K線 + 布林 + 均線
            fig.add_trace(go.Candlestick(x=date_strings, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="K線", increasing_line_color='red', decreasing_line_color='green'), row=1, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['BB_Upper'], mode='lines', name="BB上軌", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['BB_Lower'], mode='lines', name="BB下軌", line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.08)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['MA20'], mode='lines', name="MA20", line=dict(color='#1E90FF', width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['MA60'], mode='lines', name="MA60", line=dict(color='#9370DB', width=1.2)), row=1, col=1)
            
            # 2. 成交量
            vol_colors = ['red' if c >= o else 'green' for c, o in zip(data['Close'], data['Open'])]
            fig.add_trace(go.Bar(x=date_strings, y=data['Volume'], name="成交量", marker_color=vol_colors), row=2, col=1)
            
            # 3. KD
            fig.add_trace(go.Scatter(x=date_strings, y=data['K'], name="K", line=dict(color='orange', width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['D'], name="D", line=dict(color='blue', width=1)), row=3, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)
            
            # 4. MACD
            hist_colors = ['red' if v >= 0 else 'green' for v in data['MACD_Hist']]
            fig.add_trace(go.Bar(x=date_strings, y=data['MACD_Hist'], name="MACD柱狀", marker_color=hist_colors), row=4, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['DIF'], name="DIF", line=dict(color='orange', width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['DEA'], name="DEA", line=dict(color='blue', width=1)), row=4, col=1)
            
            # 5. RSI
            fig.add_trace(go.Scatter(x=date_strings, y=data['RSI6'], name="RSI6", line=dict(color='magenta', width=1.2)), row=5, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=5, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=5, col=1)
            
            # 6. 乖離率
            fig.add_trace(go.Scatter(x=date_strings, y=data['BIAS20'], name="BIAS20", line=dict(color='teal', width=1.2)), row=6, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=6, col=1)
            
            # 7. DMI / ADX
            fig.add_trace(go.Scatter(x=date_strings, y=data['+DI'], name="+DI", line=dict(color='red', width=1)), row=7, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['-DI'], name="-DI", line=dict(color='green', width=1)), row=7, col=1)
            fig.add_trace(go.Scatter(x=date_strings, y=data['ADX'], name="ADX", line=dict(color='black', width=1.5)), row=7, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="gray", row=7, col=1)

            fig.update_layout(height=1300, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20))
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True)

        # TAB 2: 回測
        with tab2:
            st.subheader("🛠️ 策略回測設定")
            with st.form("backtest_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    strategy_type = st.selectbox("選擇策略", ["雙均線策略 (MA Crossover)", "RSI 逆勢策略 (RSI Reversal)"])
                    initial_capital = st.number_input("初始資金 (NTD/USD)", value=1000000, step=100000)
                with c2:
                    if strategy_type == "雙均線策略 (MA Crossover)":
                        p1 = st.number_input("短期均線 (MA Short)", value=5, min_value=1)
                        p2 = st.number_input("長期均線 (MA Long)", value=20, min_value=1)
                    else:
                        p1 = st.number_input("RSI 買進閾值 (低於買進)", value=30, min_value=1, max_value=50)
                        p2 = st.number_input("RSI 賣出閾值 (高於賣出)", value=70, min_value=50, max_value=99)
                with c3:
                    st.write("")
                    st.write("")
                    run_btn = st.form_submit_button("🚀 開始回測", type="primary")

            if run_btn:
                market_tag = "TW" if "台股" in market_type else "US"
                bt_data, trades, ret, final_val = run_backtest(data, strategy_type, p1, p2, initial_capital, market=market_tag)
                st.session_state.backtest_result = (bt_data, trades, ret, final_val, initial_capital)

            if st.session_state.backtest_result is not None:
                bt_data, trades, ret, final_val, init_cap = st.session_state.backtest_result
                
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("初始資金", f"${init_cap:,.0f}")
                m2.metric("最終資產", f"${int(final_val):,}")
                m3.metric("總報酬率", f"{ret:.2f}%", delta=f"{ret:.2f}%")
                m4.metric("總交易次數", f"{len(trades)} 筆")

                if len(trades) == 0:
                    st.warning("⚠️ 區間內無觸發任何交易訊號，請放寬策略參數或拉長歷史資料範圍。")
                else:
                    st.subheader("📈 資金曲線與交易點位")
                    bt_fig = make_subplots(specs=[[{"secondary_y": True}]])
                    bt_date_strings = bt_data.index.strftime('%Y-%m-%d')
                    
                    bt_fig.add_trace(go.Candlestick(
                        x=bt_date_strings, open=bt_data['Open'], high=bt_data['High'], 
                        low=bt_data['Low'], close=bt_data['Close'], name="K線", opacity=0.4
                    ), secondary_y=False)

                    buy_points = trades[trades['Type'] == 'Buy']
                    sell_points = trades[trades['Type'] == 'Sell']

                    if not buy_points.empty:
                        bt_fig.add_trace(go.Scatter(
                            x=buy_points['Date'], y=buy_points['Price'], mode='markers', name='買進點',
                            marker=dict(symbol='triangle-up', size=13, color='red')
                        ), secondary_y=False)
                    if not sell_points.empty:
                        bt_fig.add_trace(go.Scatter(
                            x=sell_points['Date'], y=sell_points['Price'], mode='markers', name='賣出點',
                            marker=dict(symbol='triangle-down', size=13, color='green')
                        ), secondary_y=False)

                    bt_fig.add_trace(go.Scatter(
                        x=bt_date_strings, y=bt_data['Equity'], mode='lines', name='資產淨值',
                        line=dict(color='gold', width=2.5)
                    ), secondary_y=True)

                    bt_fig.update_layout(height=600, hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20))
                    bt_fig.update_xaxes(type="category")
                    bt_fig.update_yaxes(title_text="股價", secondary_y=False)
                    bt_fig.update_yaxes(title_text="總資產 (Equity)", secondary_y=True)
                    st.plotly_chart(bt_fig, use_container_width=True)

                    with st.expander("📄 查看詳細交易紀錄清單"):
                        st.dataframe(trades, use_container_width=True)
    else:
        st.error(f"無法取得代號【{ticker_input}】之交易數據，請確認代號正確或網路連線狀態。")
