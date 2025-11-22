import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# --- 1. 頁面設定 ---
st.set_page_config(page_title="終極股市看板", layout="wide")
st.title("📈 終極股市看板 (全指標分析 + 策略回測)")

# --- 初始化 Session State ---
if 'backtest_result' not in st.session_state:
    st.session_state.backtest_result = None
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None

# --- 2. 側邊欄輸入 ---
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
    "MSFT (Microsoft)": "MSFT", "AMD (AMD)": "AMD", "QQQ (Nasdaq 100)": "QQQ", 
    "SPY (S&P 500)": "SPY", "SOXX (Semiconductor)": "SOXX", "TQQQ (3x Long QQQ)": "TQQQ"
}

current_dict = tw_stocks if "台股" in market_type else us_stocks
options_list = list(current_dict.keys())

# 設定預設選項
if "台股" in market_type:
    default_option = "2330 台積電"
else:
    default_option = "NVDA (NVIDIA)"
try:
    default_index = options_list.index(default_option)
except ValueError:
    default_index = 0

selected_label = st.sidebar.selectbox("2️⃣ 搜尋或選擇股票", options=options_list, index=default_index)

if current_dict[selected_label] == "custom":
    raw_input = st.sidebar.text_input("請輸入代號 (如 2330 或 NVDA)")
    if raw_input:
        if "台股" in market_type:
            ticker_input = f"{raw_input}.TW" if raw_input.isdigit() and ".TW" not in raw_input.upper() else raw_input.upper()
        else:
            ticker_input = raw_input.upper()
    else:
        ticker_input = None
else:
    ticker_input = current_dict[selected_label]

# 若切換股票，清除舊的回測結果
if ticker_input != st.session_state.last_ticker:
    st.session_state.backtest_result = None
    st.session_state.last_ticker = ticker_input

# 預設時間拉長到 5y，避免 MA200 算不出來
period = st.sidebar.selectbox("3️⃣ 資料時間範圍", ("3mo", "6mo", "1y", "2y", "5y", "10y","20y", "max"), index=0)

# --- 3. 指標計算函數 ---
def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['std']
    df['BB_Lower'] = df['MA20'] - 2 * df['std']
    min_9 = df['Low'].rolling(window=9).min()
    max_9 = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - min_9) / (max_9 - min_9) * 100
    df['RSV'] = df['RSV'].fillna(50)
    k_list, d_list = [], []
    k, d = 50, 50
    for rsv in df['RSV']:
        k = (2/3) * k + (1/3) * rsv
        d = (2/3) * d + (1/3) * k
        k_list.append(k)
        d_list.append(d)
    df['K'] = k_list
    df['D'] = d_list
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEA']
    def get_rsi(series, period):
        delta = series.diff()
        u = delta.clip(lower=0)
        d = -1 * delta.clip(upper=0)
        ema_u = u.ewm(com=period-1, adjust=False).mean()
        ema_d = d.ewm(com=period-1, adjust=False).mean()
        rs = ema_u / ema_d
        return 100 - (100 / (1 + rs))
    df['RSI6'] = get_rsi(df['Close'], 6)
    df['RSI12'] = get_rsi(df['Close'], 12)
    df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['High_Diff'] = df['High'].diff()
    df['Low_Diff'] = df['Low'].diff()
    df['+DM'] = np.where((df['High_Diff'] > df['Low_Diff'].abs()) & (df['High_Diff'] > 0), df['High_Diff'], 0)
    df['-DM'] = np.where((df['Low_Diff'].abs() > df['High_Diff']) & (df['Low_Diff'] < 0), df['Low_Diff'].abs(), 0)
    alpha = 1/14
    df['TR14'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DM14'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['+DI'] = (df['+DM14'] / df['TR14']) * 100
    df['-DI'] = (df['-DM14'] / df['TR14']) * 100
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    return df

# --- 4. 智能訊號分析 ---
def analyze_signals(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    score = 0 
    if last['Close'] > last['MA20'] and last['Close'] > last['MA60']:
        signals.append(("均線趨勢", "多頭排列", "偏多", "red")); score += 2
    elif last['Close'] < last['MA20'] and last['Close'] < last['MA60']:
        signals.append(("均線趨勢", "空頭排列", "偏空", "green")); score -= 2
    else:
        signals.append(("均線趨勢", "糾結震盪", "中立", "gray"))
    if last['Volume'] > 1.5 * last['Vol_MA5']:
        signals.append(("成交量能", "爆量 (>1.5倍)", "人氣匯集", "red")); score += 0.5
    elif last['Volume'] < 0.6 * last['Vol_MA5']:
        signals.append(("成交量能", "量縮 (<0.6倍)", "觀望", "gray"))
    else:
        signals.append(("成交量能", "量能溫和", "正常", "gray"))
    if last['Close'] > last['BB_Upper']:
        signals.append(("布林通道", "突破上軌", "強勢/超買", "red")); score += 0.5
    elif last['Close'] < last['BB_Lower']:
        signals.append(("布林通道", "跌破下軌", "弱勢/超賣", "green")); score -= 0.5
    else:
        signals.append(("布林通道", "通道內", "正常", "gray"))
    if last['K'] > last['D'] and prev['K'] <= prev['D']:
        signals.append(("KD指標", "黃金交叉", "買進", "red")); score += 1.5
    elif last['K'] < last['D'] and prev['K'] >= prev['D']:
        signals.append(("KD指標", "死亡交叉", "賣出", "green")); score -= 1.5
    elif last['K'] > 80:
        signals.append(("KD指標", "高檔鈍化", "強勢/警戒", "orange"))
    elif last['K'] < 20:
        signals.append(("KD指標", "低檔鈍化", "弱勢/反彈", "blue"))
    else:
        signals.append(("KD指標", "中性", "中立", "gray"))
    if last['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0:
        signals.append(("MACD", "翻紅", "轉強", "red")); score += 1
    elif last['MACD_Hist'] < 0 and prev['MACD_Hist'] >= 0:
        signals.append(("MACD", "翻綠", "轉弱", "green")); score -= 1
    else:
        signals.append(("MACD", "震盪", "中立", "gray"))
    if last['RSI6'] > 80:
        signals.append(("RSI", "短線過熱 >80", "拉回風險", "green")); score -= 1
    elif last['RSI6'] < 20:
        signals.append(("RSI", "短線超賣 <20", "反彈機會", "red")); score += 1
    else:
        signals.append(("RSI", f"數值 {last['RSI6']:.1f}", "正常", "gray"))
    if last['BIAS20'] > 10:
        signals.append(("乖離率", "正乖離 >10%", "修正風險", "green")); score -= 1
    elif last['BIAS20'] < -10:
        signals.append(("乖離率", "負乖離 <-10%", "反彈機會", "red")); score += 1
    else:
        signals.append(("乖離率", "正常", "中立", "gray"))
    if last['ADX'] > 25:
        trend = "多方" if last['+DI'] > last['-DI'] else "空方"
        color = "red" if trend == "多方" else "green"
        signals.append(("DMI", f"趨勢明確 ({trend})", "延續", color)); score += 1 if trend == "多方" else -1
    else:
        signals.append(("DMI", "ADX<25", "盤整", "gray"))
    
    final_suggestion = "⏳ 觀望 / 中立"; final_color = "gray"
    if score >= 4: final_suggestion = "🚀 強力買進"; final_color = "red"
    elif score >= 1.5: final_suggestion = "📈 偏多操作"; final_color = "red"
    elif score <= -4: final_suggestion = "📉 強力賣出"; final_color = "green"
    elif score <= -1.5: final_suggestion = "💸 偏空/減碼"; final_color = "green"
    return signals, final_suggestion, final_color

# --- 5. 回測功能 (修正邏輯：預扣手續費 + 處理NaN) ---
def run_backtest(df, strategy, param1, param2, initial_cash=10000000):
    cash = initial_cash
    position = 0
    trade_log = []
    equity_curve = []
    
    bt_df = df.copy()
    bt_df['Signal'] = 0 # 初始化
    
    # 策略邏輯
    if strategy == "雙均線策略 (MA Crossover)":
        short_ma = bt_df['Close'].rolling(window=int(param1)).mean()
        long_ma = bt_df['Close'].rolling(window=int(param2)).mean()
        # 解決：填補 MA 計算初期的 NaN，避免 Signal 判斷錯誤
        short_ma = short_ma.fillna(0)
        long_ma = long_ma.fillna(0)
        bt_df.loc[short_ma > long_ma, 'Signal'] = 1
        
    elif strategy == "RSI 逆勢策略 (RSI Reversal)":
        holding = False
        signals = []
        # 解決：檢查 RSI 是否存在，並填補 NaN
        rsi_data = bt_df['RSI6'].fillna(50) 
        for r in rsi_data:
            if r < param1: holding = True
            elif r > param2: holding = False
            signals.append(1 if holding else 0)
        bt_df['Signal'] = signals

    # 計算倉位變化 (1: 買進, -1: 賣出)
    bt_df['Position_Change'] = bt_df['Signal'].diff().fillna(0)
    
    # 交易費率
    fee_rate = 0.001425 # 手續費
    tax_rate = 0.003    # 交易稅

    for i in range(len(bt_df)):
        price = bt_df['Close'].iloc[i]
        date = bt_df.index[i]
        change = bt_df['Position_Change'].iloc[i]
        
        # 買進條件
        if change == 1 and position == 0:
            # 修正：計算最大可買股數 (預扣手續費)
            # 公式：Cash >= Shares * Price * (1 + fee_rate)
            max_shares = int(cash / (price * (1 + fee_rate)))
            
            if max_shares > 0:
                cost = max_shares * price
                fee = cost * fee_rate
                cash -= (cost + fee)
                position = max_shares
                trade_log.append({'Date': date, 'Type': 'Buy', 'Price': price, 'Shares': max_shares, 'Balance': int(cash)})
        
        # 賣出條件
        elif change == -1 and position > 0:
            revenue = position * price
            fee = revenue * fee_rate
            tax = revenue * tax_rate
            cash += (revenue - fee - tax)
            trade_log.append({'Date': date, 'Type': 'Sell', 'Price': price, 'Shares': position, 'Balance': int(cash)})
            position = 0
            
        total_value = cash + (position * price)
        equity_curve.append(total_value)

    bt_df['Equity'] = equity_curve
    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100
    trades_df = pd.DataFrame(trade_log)
    return bt_df, trades_df, total_return, final_value

# --- 6. 資料獲取 ---
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        info = stock.info
        if df.empty: return None, None
        df = calculate_indicators(df)
        return df, info
    except:
        return None, None

# --- 7. 主程式邏輯 ---
if ticker_input:
    with st.spinner(f"正在全速運算 {ticker_input} 所有數據..."):
        data, info = get_stock_data(ticker_input, period)

    if data is not None:
        tab1, tab2 = st.tabs(["📊 全方位市場儀表板", "🧪 策略回測實驗室"])

        # TAB 1: 看盤
        with tab1:
            signal_list, suggestion, sugg_color = analyze_signals(data)
            col1, col2 = st.columns([3, 1])
            with col1:
                stock_name = info.get('longName', ticker_input)
                currency = info.get('currency', 'TWD')
                current_price = data['Close'].iloc[-1]
                change = current_price - data['Close'].iloc[-2]
                pct_change = (change / data['Close'].iloc[-2]) * 100
                color_text = "red" if change >= 0 else "green"
                st.markdown(f"## {stock_name} ({ticker_input})")
                st.markdown(f"<h2 style='color:{color_text}'>{current_price:.2f} {currency} ({change:+.2f} / {pct_change:+.2f}%)</h2>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"### 綜合建議")
                st.markdown(f"<h3 style='color:{sugg_color}; border: 2px solid {sugg_color}; padding: 5px; text-align: center; border-radius: 10px;'>{suggestion}</h3>", unsafe_allow_html=True)

            with st.expander("🤖 查看【8 大指標全方位智能診斷】", expanded=True):
                cols = st.columns(4) 
                for i, (indicator, meaning, action, color) in enumerate(signal_list):
                    with cols[i % 4]:
                        st.markdown(f"**{indicator}**"); st.caption(meaning)
                        if color == "red": st.markdown(f"<span style='color:red; font-weight:bold'>🔴 {action}</span>", unsafe_allow_html=True)
                        elif color == "green": st.markdown(f"<span style='color:green; font-weight:bold'>🟢 {action}</span>", unsafe_allow_html=True)
                        elif color == "orange": st.markdown(f"<span style='color:orange; font-weight:bold'>🟠 {action}</span>", unsafe_allow_html=True)
                        elif color == "blue": st.markdown(f"<span style='color:blue; font-weight:bold'>🔵 {action}</span>", unsafe_allow_html=True)
                        else: st.markdown(f"<span style='color:gray'>⚪ {action}</span>", unsafe_allow_html=True)
                        st.write("---")

            st.subheader("技術分析圖表 (7層詳細版)")
            fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]])
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="K線", increasing_line_color='red', decreasing_line_color='green'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name="BB上", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name="BB下", line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name="MA20", line=dict(color='blue', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MA60'], mode='lines', name="MA60", line=dict(color='purple', width=1)), row=1, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="量", marker_color=['red' if c >= o else 'green' for c, o in zip(data['Close'], data['Open'])]), row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['K'], name="K", line=dict(color='orange', width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['D'], name="D", line=dict(color='blue', width=1)), row=3, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1); fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name="MACD", marker_color=['red' if v >= 0 else 'green' for v in data['MACD_Hist']]), row=4, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['DIF'], name="DIF", line=dict(color='orange', width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['DEA'], name="DEA", line=dict(color='blue', width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI6'], name="RSI6", line=dict(color='magenta', width=1.5)), row=5, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=5, col=1); fig.add_hline(y=20, line_dash="dash", line_color="green", row=5, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BIAS20'], name="BIAS", line=dict(color='teal', width=1.5)), row=6, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=6, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['+DI'], name="+DI", line=dict(color='red', width=1)), row=7, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['-DI'], name="-DI", line=dict(color='green', width=1)), row=7, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], name="ADX", line=dict(color='black', width=1.5)), row=7, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="gray", row=7, col=1)
            fig.update_layout(height=1400, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20))
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, width="stretch")

        # TAB 2: 回測
        with tab2:
            st.subheader("🛠️ 設定回測參數 建議錢設多一點(沒零股)")
            with st.form("backtest_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    strategy_type = st.selectbox("選擇策略", ["雙均線策略 (MA Crossover)", "RSI 逆勢策略 (RSI Reversal)"])
                    initial_capital = st.number_input("初始資金", value=1000000, step=100000)
                with c2:
                    if strategy_type == "雙均線策略 (MA Crossover)":
                        p1 = st.number_input("短期均線 (MA Short)", value=5, min_value=1)
                        p2 = st.number_input("長期均線 (MA Long)", value=20, min_value=1)
                    else:
                        p1 = st.number_input("RSI 買進閾值 (低於此值買)", value=30)
                        p2 = st.number_input("RSI 賣出閾值 (高於此值賣)", value=70)
                with c3:
                    st.write("") 
                    st.write("") 
                    run_btn = st.form_submit_button("🚀 開始回測", type="primary")

            if run_btn:
                bt_data, trades, ret, final_val = run_backtest(data, strategy_type, p1, p2, initial_capital)
                st.session_state.backtest_result = (bt_data, trades, ret, final_val)

            if st.session_state.backtest_result is not None:
                bt_data, trades, ret, final_val = st.session_state.backtest_result
                
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                ret_color = "normal" if ret >=0 else "inverse"
                m1.metric("初始資金", f"${initial_capital:,}")
                m2.metric("最終資產", f"${int(final_val):,}")
                m3.metric("總報酬率", f"{ret:.2f}%", delta_color=ret_color)
                m4.metric("總交易次數", f"{len(trades)} 次")

                if len(trades) == 0:
                    st.warning("⚠️ 交易次數為 0，請檢查：1.策略參數是否太嚴格(沒有訊號) 2.資料時間範圍是否太短(均線算不出來)")
                
                st.subheader("📈 資金曲線與交易點位")
                bt_fig = make_subplots(specs=[[{"secondary_y": True}]])
                bt_fig.add_trace(go.Candlestick(x=bt_data.index, open=bt_data['Open'], high=bt_data['High'], low=bt_data['Low'], close=bt_data['Close'], name="股價", opacity=0.5), secondary_y=False)

                if not trades.empty:
                    buy_points = trades[trades['Type'] == 'Buy']
                    sell_points = trades[trades['Type'] == 'Sell']
                    bt_fig.add_trace(go.Scatter(x=buy_points['Date'], y=buy_points['Price'], mode='markers', name='買進點', marker=dict(symbol='triangle-up', size=12, color='red')), secondary_y=False)
                    bt_fig.add_trace(go.Scatter(x=sell_points['Date'], y=sell_points['Price'], mode='markers', name='賣出點', marker=dict(symbol='triangle-down', size=12, color='green')), secondary_y=False)

                bt_fig.add_trace(go.Scatter(x=bt_data.index, y=bt_data['Equity'], mode='lines', name='資產淨值', line=dict(color='gold', width=2)), secondary_y=True)
                bt_fig.update_layout(height=600, hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20))
                bt_fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                bt_fig.update_yaxes(title_text="股價", secondary_y=False)
                bt_fig.update_yaxes(title_text="總資產", secondary_y=True)
                st.plotly_chart(bt_fig, width="stretch")

                with st.expander("查看詳細交易紀錄"):
                    st.dataframe(trades)
    else:

        st.error(f"找不到代號：{ticker_input}，請確認輸入正確。")
