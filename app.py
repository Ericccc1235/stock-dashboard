import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# --- 1. 頁面設定 ---
st.set_page_config(page_title="旗艦級股市看板", layout="wide")
st.title("📈 旗艦級股市技術分析看板 (智能搜尋版)")

# --- 2. 側邊欄輸入 (搜尋邏輯大升級) ---
st.sidebar.header("查詢設定")

# 1. 選擇市場區域
market_type = st.sidebar.radio("1️⃣ 請選擇市場", ["🇹🇼 台股 (Taiwan)", "🇺🇸 美股 (US)"], horizontal=True)

# 定義預設的熱門清單 (格式: "顯示名稱": "真實代號")
tw_stocks = {
    "🔍 自行輸入代號": "custom",
    "2330 台積電": "2330.TW",
    "2317 鴻海": "2317.TW",
    "2454 聯發科": "2454.TW",
    "2303 聯電": "2303.TW",
    "2603 長榮": "2603.TW",
    "2609 陽明": "2609.TW",
    "2615 萬海": "2615.TW",
    "2382 廣達": "2382.TW",
    "3231 緯創": "3231.TW",
    "6669 緯穎": "6669.TW",
    "2357 華碩": "2357.TW",
    "2376 技嘉": "2376.TW",
    "2327 國巨": "2327.TW",
    "0050 元大台灣50": "0050.TW",
    "0056 元大高股息": "0056.TW",
    "00878 國泰永續高股息": "00878.TW",
    "00929 復華台灣科技優息": "00929.TW",
    "00919 群益台灣精選高息": "00919.TW",
    "00940 元大台灣價值高息": "00940.TW"
}

us_stocks = {
    "🔍 自行輸入代號": "custom",
    "NVDA (NVIDIA 輝達)": "NVDA",
    "AAPL (Apple 蘋果)": "AAPL",
    "TSLA (Tesla 特斯拉)": "TSLA",
    "MSFT (Microsoft 微軟)": "MSFT",
    "GOOG (Google 谷歌)": "GOOG",
    "AMZN (Amazon 亞馬遜)": "AMZN",
    "AMD (Advanced Micro Devices)": "AMD",
    "META (Meta/Facebook)": "META",
    "NFLX (Netflix 網飛)": "NFLX",
    "INTC (Intel 英特爾)": "INTC",
    "TSM (台積電ADR)": "TSM",
    "COIN (Coinbase)": "COIN",
    "QQQ (那斯達克100 ETF)": "QQQ",
    "SPY (標普500 ETF)": "SPY",
    "SOXX (半導體 ETF)": "SOXX",
    "TQQQ (三倍做多那斯達克)": "TQQQ"
}

# 根據選擇載入清單
current_list = tw_stocks if "台股" in market_type else us_stocks

# 2. 搜尋或選擇股票
selected_label = st.sidebar.selectbox("2️⃣ 搜尋或選擇股票 (可打字搜尋)", options=list(current_list.keys()))

# 3. 處理代號邏輯
if current_list[selected_label] == "custom":
    # 如果選「自行輸入」
    raw_input = st.sidebar.text_input("請輸入代號 (例如 2330 或 NVDA)")
    
    if raw_input:
        # 自動處理台股後綴
        if "台股" in market_type:
            # 如果使用者只輸入數字 (如 2330)，自動補上 .TW
            if raw_input.isdigit(): 
                ticker_input = f"{raw_input}.TW"
            # 如果使用者已經打 .TW 或 .TWO，就照舊
            elif ".TW" in raw_input.upper():
                ticker_input = raw_input.upper()
            # 處理上櫃股票 (這裡簡單假設如果是 4 位數且沒後綴，預設 .TW，若查不到可能需使用者手動打 .TWO)
            else:
                ticker_input = f"{raw_input}.TW"
        else:
            # 美股直接轉大寫
            ticker_input = raw_input.upper()
    else:
        ticker_input = None
else:
    # 如果選清單內的
    ticker_input = current_list[selected_label]

period = st.sidebar.selectbox("3️⃣ 時間範圍", ("6mo", "1y", "2y", "5y", "max"), index=1)

# --- 3. 技術指標計算函數 ---
def calculate_indicators(df):
    # 1. MA
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()

    # 2. BBands
    df['std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + 2 * df['std']
    df['BB_Lower'] = df['MA20'] - 2 * df['std']

    # 3. KD
    min_9 = df['Low'].rolling(window=9).min()
    max_9 = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - min_9) / (max_9 - min_9) * 100
    df['RSV'] = df['RSV'].fillna(50)
    k_list, d_list = [], []
    k_curr, d_curr = 50, 50
    for rsv in df['RSV']:
        k_curr = (2/3) * k_curr + (1/3) * rsv
        d_curr = (2/3) * d_curr + (1/3) * k_curr
        k_list.append(k_curr)
        d_list.append(d_curr)
    df['K'] = k_list
    df['D'] = d_list

    # 4. MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEA']

    # 5. RSI
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

    # 6. BIAS
    df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100

    # 7. DMI
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

# --- 4. 智能訊號判讀 ---
def analyze_signals(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    score = 0 

    # 1. MA
    if last['Close'] > last['MA20'] and last['Close'] > last['MA60']:
        signals.append(("均線趨勢", "多頭排列", "偏多", "red"))
        score += 2
    elif last['Close'] < last['MA20'] and last['Close'] < last['MA60']:
        signals.append(("均線趨勢", "空頭排列", "偏空", "green"))
        score -= 2
    else:
        signals.append(("均線趨勢", "糾結震盪", "中立", "gray"))

    # 2. Volume
    if last['Volume'] > 1.5 * last['Vol_MA5']:
        signals.append(("成交量能", "爆量 (>5日均量1.5倍)", "人氣匯集", "red"))
        score += 0.5
    elif last['Volume'] < 0.6 * last['Vol_MA5']:
        signals.append(("成交量能", "量縮 (<5日均量0.6倍)", "觀望", "gray"))
    else:
        signals.append(("成交量能", "量能溫和", "正常", "gray"))

    # 3. BBands
    if last['Close'] > last['BB_Upper']:
        signals.append(("布林通道", "突破上軌", "強勢/超買", "red"))
        score += 0.5
    elif last['Close'] < last['BB_Lower']:
        signals.append(("布林通道", "跌破下軌", "弱勢/超賣", "green"))
        score -= 0.5
    else:
        signals.append(("布林通道", "通道內", "正常", "gray"))

    # 4. KD
    if last['K'] > last['D'] and prev['K'] <= prev['D']:
        signals.append(("KD指標", "黃金交叉", "買進", "red"))
        score += 1.5
    elif last['K'] < last['D'] and prev['K'] >= prev['D']:
        signals.append(("KD指標", "死亡交叉", "賣出", "green"))
        score -= 1.5
    elif last['K'] > 80:
        signals.append(("KD指標", "高檔鈍化", "強勢/警戒", "orange"))
    elif last['K'] < 20:
        signals.append(("KD指標", "低檔鈍化", "弱勢/反彈", "blue"))
    else:
        signals.append(("KD指標", "中性", "中立", "gray"))

    # 5. MACD
    if last['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0:
        signals.append(("MACD", "翻紅", "轉強", "red"))
        score += 1
    elif last['MACD_Hist'] < 0 and prev['MACD_Hist'] >= 0:
        signals.append(("MACD", "翻綠", "轉弱", "green"))
        score -= 1
    elif last['MACD_Hist'] > 0 and last['MACD_Hist'] > prev['MACD_Hist']:
        signals.append(("MACD", "動能增強", "續強", "red"))
    else:
        signals.append(("MACD", "震盪", "中立", "gray"))

    # 6. RSI
    if last['RSI6'] > 80:
        signals.append(("RSI", "短線過熱", "拉回風險", "green"))
        score -= 1
    elif last['RSI6'] < 20:
        signals.append(("RSI", "短線超賣", "反彈機會", "red"))
        score += 1
    else:
        signals.append(("RSI", "正常", "中立", "gray"))

    # 7. BIAS
    if last['BIAS20'] > 10:
        signals.append(("乖離率", "正乖離大", "修正風險", "green"))
        score -= 1
    elif last['BIAS20'] < -10:
        signals.append(("乖離率", "負乖離大", "反彈機會", "red"))
        score += 1
    else:
        signals.append(("乖離率", "正常", "中立", "gray"))

    # 8. DMI
    if last['ADX'] > 25:
        trend = "多方" if last['+DI'] > last['-DI'] else "空方"
        color = "red" if trend == "多方" else "green"
        signals.append(("DMI", f"趨勢明確 ({trend})", "趨勢延續", color))
        score += 1 if trend == "多方" else -1
    else:
        signals.append(("DMI", "ADX<25", "盤整", "gray"))

    # 總結
    final_suggestion = "⏳ 觀望 / 中立"
    final_color = "gray"
    if score >= 4:
        final_suggestion = "🚀 強力買進"
        final_color = "red"
    elif score >= 1.5:
        final_suggestion = "📈 偏多操作"
        final_color = "red"
    elif score <= -4:
        final_suggestion = "📉 強力賣出"
        final_color = "green"
    elif score <= -1.5:
        final_suggestion = "💸 偏空/減碼"
        final_color = "green"

    return signals, final_suggestion, final_color

# --- 5. 獲取數據 ---
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

# --- 6. 主程式 ---
if ticker_input:
    # 顯示載入中動畫
    with st.spinner(f"正在下載 {ticker_input} 數據中..."):
        data, info = get_stock_data(ticker_input, period)
    
    if data is not None and not data.empty:
        signal_list, suggestion, sugg_color = analyze_signals(data)

        # 顯示頭部
        col1, col2 = st.columns([3, 1])
        with col1:
            stock_name = info.get('longName', ticker_input)
            currency = info.get('currency', 'TWD')
            current_price = data['Close'].iloc[-1]
            change = current_price - data['Close'].iloc[-2]
            pct_change = (change / data['Close'].iloc[-2]) * 100
            color_text = "red" if change >= 0 else "green"
            
            # 如果是美股，顯示 USD，台股顯示 TWD
            st.markdown(f"## {stock_name} ({ticker_input})")
            st.markdown(f"<h2 style='color:{color_text}'>{current_price:.2f} {currency} ({change:+.2f} / {pct_change:+.2f}%)</h2>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"### 綜合建議")
            st.markdown(f"<h3 style='color:{sugg_color}; border: 2px solid {sugg_color}; padding: 5px; text-align: center; border-radius: 10px;'>{suggestion}</h3>", unsafe_allow_html=True)

        # 智能分析
        with st.expander("🤖 查看【8 大指標全方位智能診斷】", expanded=True):
            cols = st.columns(4) 
            for i, (indicator, meaning, action, color) in enumerate(signal_list):
                with cols[i % 4]:
                    st.markdown(f"**{indicator}**")
                    st.caption(meaning)
                    if color == "red": st.markdown(f"<span style='color:red; font-weight:bold'>🔴 {action}</span>", unsafe_allow_html=True)
                    elif color == "green": st.markdown(f"<span style='color:green; font-weight:bold'>🟢 {action}</span>", unsafe_allow_html=True)
                    elif color == "orange": st.markdown(f"<span style='color:orange; font-weight:bold'>🟠 {action}</span>", unsafe_allow_html=True)
                    elif color == "blue": st.markdown(f"<span style='color:blue; font-weight:bold'>🔵 {action}</span>", unsafe_allow_html=True)
                    else: st.markdown(f"<span style='color:gray'>⚪ {action}</span>", unsafe_allow_html=True)
                    st.write("---")

        # 繪圖區域
        st.subheader("技術分析圖表")
        fig = make_subplots(
            rows=7, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.01,
            row_heights=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], 
                   [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        )

        # 1. Main
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="K線", increasing_line_color='red', decreasing_line_color='green'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name="BB上", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name="BB下", line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name="MA20", line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MA60'], mode='lines', name="MA60", line=dict(color='purple', width=1)), row=1, col=1)

        # 2. Volume
        vol_colors = ['red' if c >= o else 'green' for c, o in zip(data['Close'], data['Open'])]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="量", marker_color=vol_colors), row=2, col=1)

        # 3. KD
        fig.add_trace(go.Scatter(x=data.index, y=data['K'], mode='lines', name="K", line=dict(color='orange', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['D'], mode='lines', name="D", line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

        # 4. MACD
        macd_colors = ['red' if v >= 0 else 'green' for v in data['MACD_Hist']]
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name="MACD", marker_color=macd_colors), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['DIF'], mode='lines', name="DIF", line=dict(color='orange', width=1)), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['DEA'], mode='lines', name="DEA", line=dict(color='blue', width=1)), row=4, col=1)

        # 5. RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI6'], mode='lines', name="RSI6", line=dict(color='magenta', width=1.5)), row=5, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=5, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=5, col=1)

        # 6. BIAS
        fig.add_trace(go.Scatter(x=data.index, y=data['BIAS20'], mode='lines', name="BIAS20", line=dict(color='teal', width=1.5)), row=6, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=6, col=1)

        # 7. DMI
        fig.add_trace(go.Scatter(x=data.index, y=data['+DI'], mode='lines', name="+DI", line=dict(color='red', width=1)), row=7, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['-DI'], mode='lines', name="-DI", line=dict(color='green', width=1)), row=7, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name="ADX", line=dict(color='black', width=1.5)), row=7, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=7, col=1)

        fig.update_layout(height=1400, xaxis_rangeslider_visible=False, title_text=f"{ticker_input} 技術圖表", hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        
        axes_labels = {1: "股價", 2: "量", 3: "KD", 4: "MACD", 5: "RSI", 6: "BIAS", 7: "DMI"}
        for i, label in axes_labels.items():
            fig.update_yaxes(title_text=label, row=i, col=1)

        st.plotly_chart(fig, width="stretch")

        with st.expander("查看詳細歷史數據"):
            st.dataframe(data.sort_index(ascending=False))
    else:
        st.error(f"找不到代號：{ticker_input}，請確認輸入是否正確。")