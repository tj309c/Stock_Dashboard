import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
import io
import numpy as np  # For Monte Carlo
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai 
import json # For parsing AI responses
import pytz # For timezone-aware date comparison
from datetime import datetime, timedelta # For Earnings Calendar
from alpha_vantage.fundamentaldata import FundamentalData # For Earnings
import csv # For parsing AV response

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Stock Valuation Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- GOOGLE AI CONFIG ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Error configuring Google AI. Is your API key in .streamlit/secrets.toml? Error: {e}")

# --- DATA FETCHING & CACHING ---

@st.cache_data(ttl=3600)
def get_stock_price_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(period="5y") # yfinance returns OHLCV data
    return data

@st.cache_data(ttl=86400)
def get_company_info(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    return stock.info

@st.cache_data(ttl=86400)
def get_income_statement(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    return stock.financials

@st.cache_data(ttl=86400)
def get_balance_sheet(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    return stock.balance_sheet

@st.cache_data(ttl=86400)
def get_cash_flow(ticker_symbol):
    """Fetches the annual cash flow statement."""
    stock = yf.Ticker(ticker_symbol)
    return stock.cash_flow

@st.cache_data(ttl=1800) # Cache news for 30 minutes
def get_stock_news(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    return stock.news

# --- EARNINGS CALENDAR FUNCTION ---
@st.cache_data(ttl=3600) # Cache earnings for 1 hour
def get_earnings_calendar(api_key):
    """Fetches the earnings calendar from Alpha Vantage."""
    try:
        fd = FundamentalData(key=api_key, output_format='pandas')
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=1month&apikey={api_key}"
        df = pd.read_csv(url)
        df['reportDate'] = pd.to_datetime(df['reportDate'])
        
        today = pd.to_datetime('today').normalize()
        thirty_days_from_now = today + pd.Timedelta(days=30)
        
        df = df[(df['reportDate'] >= today) & (df['reportDate'] <= thirty_days_from_now)]
        df = df[['symbol', 'name', 'reportDate', 'estimate']]
        df.columns = ['Ticker', 'Company Name', 'Report Date', 'Analyst EPS']
        
        return df.sort_values(by="Report Date")
        
    except Exception as e:
        st.error(f"Failed to retrieve earnings calendar from Alpha Vantage: {e}")
        st.info("Please ensure your `alpha_vantage` API key is in .streamlit/secrets.toml")
        return pd.DataFrame() # Return empty dataframe on failure


# --- AI-POWERED DCF DEFAULTS ---
@st.cache_data(ttl=86400) # Cache AI suggestions for 1 day
def get_ai_dcf_assumptions(ticker, company_name, sector, industry):
    st.write(f"Querying AI for conservative DCF assumptions for {ticker}...")
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        prompt = f"""
        You are a senior financial analyst. For the stock {ticker} ({company_name}), which is in the {sector} sector and {industry} industry, provide conservative default assumptions for a 5-year DCF model.
        
        I need:
        1.  'fcf_growth': A 5-year free cash flow growth rate (%).
        2.  'wacc': The Weighted Average Cost of Capital (WACC) as a discount rate (%).
        3.  'perpetual_growth': A conservative perpetual (terminal) growth rate (%).
        
        Also provide standard deviations for a Monte Carlo simulation on these three, representing reasonable volatility for this type of company:
        4.  'growth_std_dev': Standard deviation for FCF growth (%).
        5.  'wacc_std_dev': Standard deviation for WACC (%).
        6.  'perpetual_std_dev': Standard deviation for perpetual growth (%).

        Return ONLY a single, minified JSON object with these 6 keys. All values must be floats.
        Example: {{"fcf_growth": 8.5, "wacc": 9.2, "perpetual_growth": 2.5, "growth_std_dev": 1.5, "wacc_std_dev": 1.0, "perpetual_std_dev": 0.5}}
        """
        
        response = model.generate_content(prompt)
        json_str = response.text.strip().lstrip("```json").lstrip("```").rstrip("```")
        ai_defaults = json.loads(json_str)
        return ai_defaults
        
    except Exception as e:
        st.error(f"Error getting AI assumptions: {e}")
        return {
            "fcf_growth": 5.0, "wacc": 10.0, "perpetual_growth": 2.0,
            "growth_std_dev": 1.5, "wacc_std_dev": 1.0, "perpetual_std_dev": 0.5
        }

# --- PROPHET FUNCTION ---
@st.cache_data(ttl=86400)
def get_prophet_forecast(price_data, weeks_to_forecast):
    prophet_df = price_data.reset_index()
    prophet_df = prophet_df[['Date', 'Close']]
    prophet_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=weeks_to_forecast * 7)
    forecast = model.predict(future)
    return model, forecast

# --- API STATUS CHECK ---
@st.cache_data(ttl=600) 
def get_yfinance_api_status():
    try:
        test_ticker = yf.Ticker("AAPL") 
        if test_ticker.info:
            return (True, "yfinance API is responsive.")
        else:
            return (False, "yfinance API is responding but returned no data for test ticker.")
    except Exception as e:
        return (False, f"yfinance API is unreachable. Error: {e}")

# --- DCF HELPER FUNCTION (MODIFIED) ---
def calculate_dcf(fcf, cash, total_debt, shares_outstanding, growth, wacc, perpetual):
    """
    Calculates DCF and returns the final value AND a dictionary of intermediate steps for debugging.
    """
    try:
        fcf_forecast = []
        for i in range(1, 6):
            fcf_forecast.append(fcf * (1 + growth)**i)
        
        if (wacc - perpetual) <= 0: 
            return 0, {} # Return 0 and empty dict if WACC <= perpetual rate

        # B. Calculate Terminal Value
        terminal_value = (fcf_forecast[-1] * (1 + perpetual)) / (wacc - perpetual)
        
        # C. Discount FCF and Terminal Value
        discounted_fcf = []
        for i, fcf_val in enumerate(fcf_forecast):
            discounted_fcf.append(fcf_val / (1 + wacc)**(i + 1))
        
        discounted_terminal_value = terminal_value / (1 + wacc)**5
        
        # A. Calculate Enterprise Value
        enterprise_value = sum(discounted_fcf) + discounted_terminal_value
        
        # B. Bridge to Equity Value
        equity_value = enterprise_value + cash - total_debt
        
        # C. Calculate Value Per Share
        if shares_outstanding > 0:
            value_per_share = equity_value / shares_outstanding
        else:
            value_per_share = 0
            
        intermediates = {
            "fcf_forecast": fcf_forecast,
            "terminal_value": terminal_value,
            "discounted_fcf": discounted_fcf,
            "discounted_tv": discounted_terminal_value,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value
        }
            
        return value_per_share, intermediates
        
    except (TypeError, OverflowError):
        return 0, {} # Return 0 and empty dict on any math error

# --- DCF DEBUGGER TEXT FUNCTION (MODIFIED for 3-Year Avg) ---
def generate_dcf_debug_text(ticker, inputs, intermediates, result):
    text = f"--- DCF Debug Report for {ticker} ---\n"
    text += f"--- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n"
    
    text += "--- 1. INPUT VARIABLES (From yfinance & Sidebar) ---\n"
    text += "A. FCF (3-Year Average): ${inputs['fcf']:,.0f}\n"
    text += "   (Source: 3-Year Avg OCF + 3-Year Avg CapEx)\n\n"
    text += f"   - 3-Year Avg OCF: ${inputs['ocf']:,.0f}\n"
    text += "     (Values: " + ", ".join([f"${x:,.0f}" for x in inputs['ocf_series']]) + ")\n"
    text += f"   - 3-Year Avg CapEx: ${inputs['capex']:,.0f}\n"
    text += "     (Values: " + ", ".join([f"${x:,.0f}" for x in inputs['capex_series']]) + ")\n\n"
    
    text += f"B. Total Cash: ${inputs['cash']:,.0f}\n"
    text += f"C. Total Debt: ${inputs['total_debt']:,.0f}\n"
    text += f"D. Shares Outstanding: {inputs['shares']:,}\n"
    text += f"E. Growth Rate (g): {inputs['g'] * 100:.2f}%\n"
    text += f"F. Discount Rate (WACC): {inputs['wacc'] * 100:.2f}%\n"
    text += f"G. Perpetual Rate (p): {inputs['p'] * 100:.2f}%\n\n"
    
    if inputs['fcf'] < 0:
        text += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        text += "CRITICAL WARNING: The '3-Year Average FCF' is negative.\n"
        text += "This forces all projected FCFs and the Terminal Value to be negative, resulting in a negative Enterprise Value and a negative Intrinsic Value.\n"
        text += "This is common for companies with high capital expenditures. The model's math is correct, but this FCF input is not suitable for a simple DCF.\n"
        text += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"

    text += "--- 2. INTERMEDIATE CALCULATIONS (The Math) ---\n"
    text += "A. Project Future FCF (FCF * (1+g)^year):\n"
    for i, fcf_val in enumerate(intermediates.get('fcf_forecast', [])):
        text += f"  - Year {i+1}: ${fcf_val:,.0f}\n"
    
    text += "\nB. Calculate Terminal Value (TV):\n"
    text += "   Formula: (FCF_Year_5 * (1+p)) / (WACC - p)\n"
    text += f"   TV = (${intermediates.get('fcf_forecast', [0])[-1]:,.0f} * (1 + {inputs['p']})) / ({inputs['wacc']} - {inputs['p']})\n"
    text += f"   TV = ${intermediates.get('terminal_value', 0):,.0f}\n"

    text += "\nC. Discount FCF and Terminal Value:\n"
    text += "   Formula: Value / (1 + WACC)^year\n"
    for i, dfcf in enumerate(intermediates.get('discounted_fcf', [])):
        text += f"  - Discounted FCF Year {i+1}: ${dfcf:,.0f}\n"
    text += f"  - Discounted TV (Year 5): ${intermediates.get('discounted_tv', 0):,.0f}\n"

    text += "\n--- 3. FINAL VALUATION (The Bridge) ---\n"
    text += "A. Calculate Enterprise Value (EV):\n"
    text += "   Formula: Sum of Discounted FCFs + Discounted TV\n"
    text += f"   EV = ${sum(intermediates.get('discounted_fcf', [0])):,.0f} + ${intermediates.get('discounted_tv', 0):,.0f}\n"
    text += f"   EV = ${intermediates.get('enterprise_value', 0):,.0f}\n"

    text += "\nB. Bridge to Equity Value (Your Requested Step):\n"
    text += "   Formula: Enterprise Value + Cash - Debt\n"
    text += f"   Equity Value = ${intermediates.get('enterprise_value', 0):,.0f} + ${inputs['cash']:,.0f} - ${inputs['total_debt']:,.0f}\n"
    text += f"   Equity Value = ${intermediates.get('equity_value', 0):,.0f}\n"

    text += "\nC. Calculate Intrinsic Value Per Share:\n"
    text += "   Formula: Equity Value / Shares Outstanding\n"
    text += f"   Value per Share = ${intermediates.get('equity_value', 0):,.0f} / {inputs['shares']:,}\n"
    text += f"   FINAL INTRINSIC VALUE = ${result:,.2f}\n"
    
    return text

# --- TECHNICAL SIGNAL HELPER FUNCTION ---
def get_technical_signals(data):
    """Analyzes the latest data point to generate bullish/bearish signals."""
    bullish_signals = []
    bearish_signals = []
    
    # Get the latest data point
    latest = data.iloc[-1]
    
    # 1. RSI
    if latest['RSI_14'] < 30:
        bullish_signals.append("RSI is Oversold (< 30)")
    if latest['RSI_14'] > 70:
        bearish_signals.append("RSI is Overbought (> 70)")
        
    # 2. MACD (Histogram)
    if latest['MACDh_12_26_9'] > 0 and data.iloc[-2]['MACDh_12_26_9'] < 0:
        bullish_signals.append("MACD Crosses Above Signal (Bullish Crossover)")
    elif latest['MACDh_12_26_9'] < 0 and data.iloc[-2]['MACDh_12_26_9'] > 0:
        bearish_signals.append("MACD Crosses Below Signal (Bearish Crossover)")
        
    # 3. SMA Trend
    if latest['Close'] > latest['SMA50']:
        bullish_signals.append("Price > 50-Day Avg (Short-term Bullish)")
    if latest['Close'] < latest['SMA50']:
        bearish_signals.append("Price < 50-Day Avg (Short-term Bearish)")
    if latest['SMA50'] > latest['SMA200'] and data.iloc[-2]['SMA50'] < data.iloc[-2]['SMA200']:
         bullish_signals.append("Golden Cross (SMA 50 crosses SMA 200)")
    if latest['SMA50'] < latest['SMA200'] and data.iloc[-2]['SMA50'] > data.iloc[-2]['SMA200']:
         bearish_signals.append("Death Cross (SMA 50 crosses below SMA 200)")

    # 4. Bollinger Bands
    if latest['Close'] < latest['BBL_20_2.0_2.0']:
        bullish_signals.append("Price below Lower Bollinger Band (Oversold)")
    if latest['Close'] > latest['BBU_20_2.0_2.0']:
        bearish_signals.append("Price above Upper Bollinger Band (Overbought)")
        
    # 5. ADX Trend Strength
    if latest['ADX_14'] > 25:
        if latest['DMP_14'] > latest['DMN_14']:
            bullish_signals.append(f"Strong Uptrend (ADX: {latest['ADX_14']:.0f})")
        else:
            bearish_signals.append(f"Strong Downtrend (ADX: {latest['ADX_14']:.0f})")
            
    return bullish_signals, bearish_signals

# --- FINAL RANKING FUNCTION ---
def get_final_ranking(info, intrinsic_price, market_price, tech_signals):
    bullish_count, bearish_count = tech_signals
    
    # 1. Valuation Score (1-5)
    valuation_score = 3
    if market_price > 0:
        upside = (intrinsic_price / market_price) - 1
        if upside > 0.25: valuation_score = 1 # Strong Buy
        elif upside > 0.10: valuation_score = 2 # Buy
        elif upside < -0.25: valuation_score = 5 # Strong Sell
        elif upside < -0.10: valuation_score = 4 # Sell
        else: valuation_score = 3 # Hold
    
    # 2. Technical Score (1-5)
    technical_score = 3
    net_signals = bullish_count - bearish_count
    if net_signals >= 2: technical_score = 1
    elif net_signals == 1: technical_score = 2
    elif net_signals <= -2: technical_score = 5
    elif net_signals == -1: technical_score = 4
    else: technical_score = 3

    # 3. Fundamental Score (1-5)
    fundamental_score = 3
    try:
        roe = info.get('returnOnEquity', 0)
        de_ratio = info.get('debtToEquity', 100) / 100.0 # Standardize
        fund_points = 0
        if roe > 0.15: fund_points += 1
        if roe < 0.05: fund_points -= 1
        if de_ratio < 1.0: fund_points += 1
        if de_ratio > 2.0: fund_points -= 1
        if fund_points == 2: fundamental_score = 1
        elif fund_points == 1: fundamental_score = 2
        elif fund_points == -1: fundamental_score = 4
        elif fund_points == -2: fundamental_score = 5
        else: fundamental_score = 3
    except:
        fundamental_score = 3 # Default to Hold if data is missing

    # Final Score (Average)
    final_score = (valuation_score + technical_score + fundamental_score) / 3
    
    # Score to Rating
    if final_score <= 1.66: rating = "1 - Strong Buy"
    elif final_score <= 2.33: rating = "2 - Buy"
    elif final_score <= 3.66: rating = "3 - Hold"
    elif final_score <= 4.33: rating = "4 - Sell"
    else: rating = "5 - Strong Sell"
        
    return rating, valuation_score, technical_score, fundamental_score

# --- STREAMLIT APP LAYOUT ---

st.title("📈 Stock Valuation Dashboard")

# --- GLOBAL API STATUS ---
api_ok, api_message = get_yfinance_api_status()
if api_ok:
    st.success(f"**API Status:** {api_message}")
else:
    st.error(f"**API Status:** {api_message}")
    st.warning("Data on this dashboard may be incomplete or missing. Please do not trust the final numbers until this is resolved.")

st.write("A unified model blending fundamentals and technicals (using yfinance).")


# --- SIDEBAR (Controls) ---
st.sidebar.header("Controls")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()


# --- MAIN PAGE DATA LOAD (MOVED UP) ---
# --- Load all data *before* drawing the rest of the sidebar ---
if ticker_symbol:
    
    with st.spinner(f"Loading all data for {ticker_symbol}..."):
        price_data_cached = get_stock_price_data(ticker_symbol)
        info = get_company_info(ticker_symbol)
        income_data = get_income_statement(ticker_symbol)
        balance_sheet_data = get_balance_sheet(ticker_symbol)
        cash_flow_data = get_cash_flow(ticker_symbol)
        news_data = get_stock_news(ticker_symbol) 

        data_errors = []
        if price_data_cached.empty: data_errors.append("`Price Data (History)` is missing.")
        if not info: data_errors.append("`Company Info` is missing.")
        if income_data.empty: data_errors.append("`Income Statement` is missing.")
        if balance_sheet_data.empty: data_errors.append("`Balance Sheet` is missing.")
        if cash_flow_data.empty: data_errors.append("`Cash Flow Statement` is missing.")

    # --- Static Defaults (hardcoded) ---
    STATIC_GROWTH = 5.0
    STATIC_WACC = 10.0
    STATIC_PERPETUAL = 2.0
    STATIC_GROWTH_STD = 1.5
    STATIC_WACC_STD = 1.0
    STATIC_PERPETUAL_STD = 0.5

    # --- Initialize session state keys if they don't exist ---
    if 'fcf_growth' not in st.session_state: st.session_state.fcf_growth = STATIC_GROWTH
    if 'wacc' not in st.session_state: st.session_state.wacc = STATIC_WACC
    if 'perpetual_growth' not in st.session_state: st.session_state.perpetual_growth = STATIC_PERPETUAL
    if 'growth_std_dev' not in st.session_state: st.session_state.growth_std_dev = STATIC_GROWTH_STD
    if 'wacc_std_dev' not in st.session_state: st.session_state.wacc_std_dev = STATIC_WACC_STD
    if 'perpetual_std_dev' not in st.session_state: st.session_state.perpetual_std_dev = STATIC_PERPETUAL_STD

    # --- (Rest of Sidebar) ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Price & Technicals", 
            "Fundamental Deep Dive", 
            "Raw Financials", 
            "Intrinsic Value (DCF)",
            "📅 Earnings Calendar", # <-- NEW TAB
            "Future Forecast (Prophet)",
            "🤖 AI Stock Assistant"
        ],
        label_visibility="collapsed" 
    )

    # --- FIX: Button logic MUST come before widgets are instantiated ---
    st.sidebar.header("DCF Model Assumptions")
    ai_defaults_button = st.sidebar.button("Set to AI Suggestions")
    reset_dcf_button = st.sidebar.button("Reset DCF Defaults (5/10/2)")

    if ai_defaults_button:
        with st.spinner("Asking AI for default assumptions..."):
            ai_defaults = get_ai_dcf_assumptions(
                ticker_symbol, 
                info.get('longName', ''), 
                info.get('sector', ''), 
                info.get('industry', '')
            )
        # Update session state *before* sliders are drawn
        st.session_state.fcf_growth = ai_defaults['fcf_growth']
        st.session_state.wacc = ai_defaults['wacc']
        st.session_state.perpetual_growth = ai_defaults['perpetual_growth']
        st.session_state.growth_std_dev = ai_defaults['growth_std_dev']
        st.session_state.wacc_std_dev = ai_defaults['wacc_std_dev']
        st.session_state.perpetual_std_dev = ai_defaults['perpetual_std_dev']
        st.rerun() # Rerun to make sure the new values are used

    if reset_dcf_button:
        st.session_state.fcf_growth = STATIC_GROWTH
        st.session_state.wacc = STATIC_WACC
        st.session_state.perpetual_growth = STATIC_PERPETUAL
        st.rerun()

    # --- Draw DCF Sliders (now read from updated session_state) ---
    st.sidebar.slider("FCF Growth Rate (Years 1-5) (%)", 1.0, 20.0, key='fcf_growth')
    st.sidebar.slider("Discount Rate (WACC) (%)", 5.0, 20.0, key='wacc')
    st.sidebar.slider("Perpetual Growth Rate (%)", 0.0, 5.0, key='perpetual_growth')

    # --- FIX: Add new button logic for Monte Carlo ---
    st.sidebar.header("Monte Carlo Assumptions")
    reset_mc_button = st.sidebar.button("Reset Monte Carlo Defaults") # <-- NEW BUTTON

    if reset_mc_button:
        st.session_state.growth_std_dev = STATIC_GROWTH_STD
        st.session_state.wacc_std_dev = STATIC_WACC_STD
        st.session_state.perpetual_std_dev = STATIC_PERPETUAL_STD
        st.rerun()

    # --- Draw Monte Carlo Sliders ---
    st.sidebar.number_input("Number of Simulations", 1000, 20000, 10000, key='simulations')
    st.sidebar.slider("Growth Rate Std. Dev. (%)", 0.1, 5.0, key='growth_std_dev')
    st.sidebar.slider("WACC Std. Dev. (%)", 0.1, 5.0, key='wacc_std_dev')
    st.sidebar.slider("Perpetual Std. Dev. (%)", 0.1, 1.0, key='perpetual_std_dev')


    # --- MAIN PAGE LOGIC ---
    st.header(f"{ticker_symbol} Analysis: {info.get('longName', '')}")

    if data_errors:
        st.error(f"**Data Error:** Failed to retrieve all required data for {ticker_symbol}. Models will be disabled.")
        with st.expander("Click to see error details"):
            for err in data_errors:
                st.warning(err)
            st.info("This is often a temporary `yfinance` issue or a problem with the specific ticker.")
    else:
        st.success(f"**Data Status:** Successfully loaded all financial data for {ticker_symbol}.")
        
        # --- PRE-CALCULATIONS FOR RANKING & DCF ---
        fcf, cash, total_debt, shares_outstanding, market_price = 0, 0, 0, 0, 0
        op_cash_flow_avg, cap_ex_avg = 0, 0
        op_cash_flow_series, cap_ex_series = pd.Series([0]), pd.Series([0])
        intrinsic_price_per_share = 0
        dcf_intermediates = {} # Store debug data
        
        fcf_growth_rate = st.session_state.fcf_growth / 100.0
        discount_rate = st.session_state.wacc / 100.0
        perpetual_growth_rate = st.session_state.perpetual_growth / 100.0
        
        growth_std = st.session_state.growth_std_dev / 100.0
        wacc_std = st.session_state.wacc_std_dev / 100.0
        perpetual_std = st.session_state.perpetual_std_dev / 100.0
        
        try:
            shares_outstanding = info.get('sharesOutstanding', 0)
            market_price = price_data_cached['Close'].iloc[-1]
            total_debt = info.get('totalDebt', 0)
            
            # --- FIX: Use 3-YEAR AVERAGE FCF with FALLBACKS ---
            if 'Total Cash From Operating Activities' in cash_flow_data.index:
                op_cash_flow_series = cash_flow_data.loc['Total Cash From Operating Activities']
            elif 'Operating Cash Flow' in cash_flow_data.index:
                op_cash_flow_series = cash_flow_data.loc['Operating Cash Flow']
            else:
                op_cash_flow_series = pd.Series([0]) # Fallback

            if 'Capital Expenditure' in cash_flow_data.index:
                cap_ex_series = cash_flow_data.loc['Capital Expenditure']
            else:
                cap_ex_series = pd.Series([0]) # Fallback
            
            # Get the first 3 available years, or fewer if not available
            op_cash_flow_avg = op_cash_flow_series.iloc[:3].mean()
            cap_ex_avg = cap_ex_series.iloc[:3].mean()
            
            fcf = op_cash_flow_avg + cap_ex_avg # This is now the 3-year average FCF
            # --- END FIX ---

            cash = balance_sheet_data.loc['Total Cash, Cash Equivalents And Short Term Investments'].iloc[0] if 'Total Cash, Cash Equivalents And Short Term Investments' in balance_sheet_data.index else 0

            if total_debt == 0 and 'Total Debt' in balance_sheet_data.index:
                 total_debt = balance_sheet_data.loc['Total Debt'].iloc[0]
            if cash == 0 and 'Cash And Cash Equivalents' in balance_sheet_data.index:
                cash = balance_sheet_data.loc['Cash And Cash Equivalents'].iloc[0]

            
            # --- MODIFIED: Get both price and debug data ---
            intrinsic_price_per_share, dcf_intermediates = calculate_dcf(fcf, cash, total_debt, shares_outstanding, fcf_growth_rate, discount_rate, perpetual_growth_rate)
        
        except Exception as e:
            st.warning(f"Could not calculate DCF value due to missing financial data: {e}")

        
        # --- Calculate Technicals (used in multiple tabs) ---
        bbands = ta.bbands(price_data_cached['Close'], length=20, std=2.0)
        rsi = ta.rsi(price_data_cached['Close'], length=14)
        macd = ta.macd(price_data_cached['Close'], fast=12, slow=26, signal=9)
        sma50 = ta.sma(price_data_cached['Close'], length=50)
        sma200 = ta.sma(price_data_cached['Close'], length=200)
        adx_data = ta.adx(price_data_cached['High'], price_data_cached['Low'], price_data_cached['Close'], length=14)
        obv_data = ta.obv(price_data_cached['Close'], price_data_cached['Volume'])
        
        price_data = price_data_cached.join([bbands, rsi, macd, sma50, sma200, adx_data, obv_data])
        price_data.rename(columns={'SMA_50': 'SMA50', 'SMA_200': 'SMA200', 'OBV': 'OBV'}, inplace=True, errors='ignore')
        price_data.dropna(inplace=True)
        
        bullish_signals, bearish_signals = [], []
        if not price_data.empty:
            bullish_signals, bearish_signals = get_technical_signals(price_data)


        # --- NAVIGATION LOGIC ---
        
        if page == "Price & Technicals":
            st.subheader("Technical Analysis")
            
            # --- REQ #1: FINAL RANKING ---
            if not price_data.empty and intrinsic_price_per_share > 0:
                final_rating, v_score, t_score, f_score = get_final_ranking(
                    info, 
                    intrinsic_price_per_share, 
                    market_price, 
                    (len(bullish_signals), len(bearish_signals))
                )
                
                st.markdown(f"## Final Recommendation: **{final_rating}**")
                
                with st.expander("See Scoring Breakdown & Assumptions"):
                    st.write(f"""
                    This rating is an average of three scores (1=Strong Buy, 5=Strong Sell), weighted equally.
                    -   **Valuation Score (1-5):** `{v_score}`
                        - *Assumption:* Based on DCF upside. >25% upside is a '1', >10% is a '2', etc.
                    -   **Technical Score (1-5):** `{t_score}`
                        - *Assumption:* Based on the net signal count below. 2+ net bullish signals is a '1', 2+ net bearish is a '5', etc.
                    -   **Fundamental Score (1-5):** `{f_score}`
                        - *Assumption:* Based on ROE (>15%) and Debt/Equity (<1.0).
                    """)
                st.divider()

            # --- Technical Signal Summary ---
            st.subheader("Technical Signal Summary")
            st.write("Based on the most recent data point:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bullish Signals", len(bullish_signals))
                with st.expander("See Bullish Signals"):
                    if bullish_signals:
                        for signal in bullish_signals:
                            st.success(f"• {signal}")
                    else:
                        st.info("No bullish signals detected.")
            with col2:
                st.metric("Bearish Signals", len(bearish_signals))
                with st.expander("See Bearish Signals"):
                    if bearish_signals:
                        for signal in bearish_signals:
                            st.error(f"• {signal}")
                    else:
                        st.info("No bearish signals detected.")
            st.divider()

            # --- Plot Charts ---
            st.subheader("Price with Moving Averages & Bollinger Bands")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['BBU_20_2.0_2.0'], fill=None, mode='lines', line_color='grey', name='Upper Band'))
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['BBL_20_2.0_2.0'], fill='tonexty', mode='lines', line_color='grey', name='Lower Band'))
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', line_color='blue', name='Close Price'))
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA50'], mode='lines', line_color='orange', name='SMA 50'))
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA200'], mode='lines', line_color='red', name='SMA 200'))
            fig.update_layout(title=f"{ticker_symbol} Price & Indicators", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Relative Strength Index (RSI)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=price_data.index, y=price_data['RSI_14'], mode='lines', line_color='purple', name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi, use_container_width=True)

            st.subheader("Moving Average Convergence Divergence (MACD)")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=price_data.index, y=price_data['MACD_12_26_9'], mode='lines', line_color='blue', name='MACD'))
            fig_macd.add_trace(go.Scatter(x=price_data.index, y=price_data['MACDs_12_26_9'], mode='lines', line_color='orange', name='Signal'))
            fig_macd.add_trace(go.Bar(x=price_data.index, y=price_data['MACDh_12_26_9'], name='Histogram', marker_color='grey'))
            st.plotly_chart(fig_macd, use_container_width=True)
            
            st.subheader("Average Directional Index (ADX)")
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(x=price_data.index, y=price_data['ADX_14'], mode='lines', line_color='blue', name='ADX'))
            fig_adx.add_trace(go.Scatter(x=price_data.index, y=price_data['DMP_14'], mode='lines', line_color='green', name='DI+ (Bullish)'))
            fig_adx.add_trace(go.Scatter(x=price_data.index, y=price_data['DMN_14'], mode='lines', line_color='red', name='DI- (Bearish)'))
            fig_adx.add_hline(y=25, line_dash="dash", line_color="grey")
            st.info("When ADX (blue) is above 25, it signals a strong trend (green > red = uptrend, red > green = downtrend).")
            st.plotly_chart(fig_adx, use_container_width=True)
            
            st.subheader("On-Balance Volume (OBV)")
            fig_obv = go.Figure()
            fig_obv.add_trace(go.Scatter(x=price_data.index, y=price_data['OBV'], mode='lines', line_color='purple', name='OBV'))
            st.plotly_chart(fig_obv, use_container_width=True)


        elif page == "Fundamental Deep Dive":
            st.subheader(f"{info.get('longName', ticker_symbol)} Fundamentals")
            
            if info:
                # --- REQ #5: Social Sentiment Placeholder ---
                st.subheader("Social Sentiment (Placeholder)")
                st.info("Note: Real-time social sentiment requires complex, often paid, APIs (e.g., Stocktwits, Alpha Vantage). This is a visual placeholder to show functionality.")
                placeholder_sentiment = np.random.randint(20, 80)
                st.progress(placeholder_sentiment, text=f"Sentiment Score: {placeholder_sentiment}/100")
                
                # Dummy data for the bar chart
                sentiment_sources = pd.DataFrame({
                    'source': ['Reddit', 'X (Twitter)', 'Google Trends', 'News Articles'],
                    'score': [np.random.randint(20, 80), np.random.randint(20, 80), np.random.randint(20, 80), np.random.randint(20, 80)]
                })
                st.bar_chart(sentiment_sources, x='source', y='score')

                # --- Expanded Metrics ---
                st.subheader("Key Valuation Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                col2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A')
                col3.metric("Forward P/E", f"{info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else 'N/A')
                col4.metric("Price-to-Sales (TTM)", f"{info.get('priceToSalesTrailing12Months', 'N/A'):.2f}" if info.get('priceToSalesTrailing12Months') else 'N/A')
                col1.metric("Price-to-Book", f"{info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else 'N/A')
                col2.metric("Enterprise Value", f"${info.get('enterpriseValue', 0):,}")
                col3.metric("EV-to-Revenue", f"{info.get('enterpriseToRevenue', 'N/A'):.2f}" if info.get('enterpriseToRevenue') else 'N/A')
                col4.metric("EV-to-EBITDA", f"{info.get('enterpriseToEbitda', 'N/A'):.2f}" if info.get('enterpriseToEbitda') else 'N/A')

                st.subheader("Profitability & Margins")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Profit Margin", f"{info.get('profitMargins', 0) * 100:.2f}%")
                col2.metric("Operating Margin", f"{info.get('operatingMargins', 0) * 100:.2f}%")
                col3.metric("Return on Assets (ROA)", f"{info.get('returnOnAssets', 0) * 100:.2f}%")
                col4.metric("Return on Equity (ROE)", f"{info.get('returnOnEquity', 0) * 100:.2f}%")
                
                st.subheader("Dividends & Splits")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%")
                col2.metric("Dividend Rate", f"{info.get('dividendRate', 'N/A')}" if info.get('dividendRate') else 'N/A')
                col3.metric("Payout Ratio", f"{info.get('payoutRatio', 0) * 100:.2f}%")
                last_split_date = pd.to_datetime(info.get('lastSplitDate'), unit='s').date() if info.get('lastSplitDate') else 'N/A'
                col4.metric("Last Split", f"{info.get('lastSplitFactor', 'N/A')} on {last_split_date}")

                st.subheader("Balance Sheet & Cash Flow")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Cash", f"${info.get('totalCash', 0):,}")
                col2.metric("Total Debt", f"${info.get('totalDebt', 0):,}")
                col3.metric("Debt-to-Equity", f"{info.get('debtToEquity', 'N/A'):.2f}" if info.get('debtToEquity') else 'N/A')
                col4.metric("Operating Cash Flow", f"${info.get('operatingCashflow', 0):,}")

                st.subheader("Company Summary")
                st.write(info.get('longBusinessSummary', 'No summary available.'))

                # --- REQ #6: RECENT NEWS (FIXED with 7-DAY FILTER) ---
                st.subheader("Recent News (Past 7 Days)")
                
                # Define the 7-day-ago timestamp (in UTC)
                seven_days_ago = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)

                with st.expander("Debug Raw News Data"):
                    st.write("This shows the raw data received from the `yfinance` API.")
                    st.json(news_data)

                if news_data:
                    valid_articles_found = False
                    for item in news_data:
                        # --- FIX: Use correct nested JSON structure ---
                        content = item.get('content', {})
                        
                        title = content.get('title')
                        publish_time_str = content.get('pubDate')
                        
                        # --- FIX: Check for title and publish time ---
                        if title and publish_time_str:
                            
                            # --- FIX: Parse and filter by date ---
                            try:
                                # Convert pubDate string to timezone-aware datetime
                                article_date = pd.to_datetime(publish_time_str).tz_convert(pytz.UTC)
                                if article_date < seven_days_ago:
                                    continue # Skip this article, it's too old
                            except Exception:
                                continue # Skip if date is unparseable

                            valid_articles_found = True
                            
                            # --- FIX: Get correct link and publisher ---
                            link = content.get('canonicalUrl', {}).get('url', '#')
                            if link == '#': # Fallback if canonicalUrl is missing
                                 link = content.get('clickThroughUrl', {}).get('url', '#')
                                 
                            publisher = content.get('provider', {}).get('displayName', 'Unknown Publisher')
                            
                            time_str = article_date.strftime('%Y-%m-%d %H:%M')
                            
                            st.markdown(f"**[{title}]({link})**")
                            st.write(f"*{publisher} - {time_str}*")
                            st.divider()
                    
                    if not valid_articles_found:
                        st.warning("No recent news (within the past 7 days) with a valid title was found.")
                else:
                    st.write("No news found for this ticker.")
            else:
                st.warning("No company info available.")


        # --- REQ #4: Raw Financials (Formatted) ---
        elif page == "Raw Financials":
            st.subheader("Raw Annual Income Statement")
            try:
                numeric_cols_income = income_data.select_dtypes(include=np.number).columns
                formatters_income = {col: '{:,.0f}' for col in numeric_cols_income}
                st.dataframe(income_data.style.format(formatters_income, na_rep="-"), use_container_width=True)
            except Exception as e:
                st.error(f"Could not format Income Statement: {e}")

            st.subheader("Raw Annual Balance Sheet")
            try:
                numeric_cols_balance = balance_sheet_data.select_dtypes(include=np.number).columns
                formatters_balance = {col: '{:,.0f}' for col in numeric_cols_balance}
                st.dataframe(balance_sheet_data.style.format(formatters_balance, na_rep="-"), use_container_width=True)
            except Exception as e:
                st.error(f"Could not format Balance Sheet: {e}")

            st.subheader("Raw Annual Cash Flow Statement")
            try:
                numeric_cols_cash = cash_flow_data.select_dtypes(include=np.number).columns
                formatters_cash = {col: '{:,.0f}' for col in numeric_cols_cash}
                st.dataframe(cash_flow_data.style.format(formatters_cash, na_rep="-"), use_container_width=True)
            except Exception as e:
                st.error(f"Could not format Cash Flow Statement: {e}")


        elif page == "Intrinsic Value (DCF)":
            st.header("Discounted Cash Flow (DCF) Model")
            
            # --- REQ #3: DCF VISUALS ---
            if intrinsic_price_per_share > 0 and market_price > 0:
                upside_percent = (intrinsic_price_per_share / market_price) - 1
                if upside_percent > 0.20:
                    st.markdown(f"<h1 style='text-align: center; color: green;'>STRONG BUY</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'><img src='[https://i.giphy.com/media/lPIPQicnYJiVMfU6Tq/giphy.gif](https://i.giphy.com/media/lPIPQicnYJiVMfU6Tq/giphy.gif)' width='300'></p>", unsafe_allow_html=True)
                elif upside_percent > 0.10:
                    st.markdown(f"<h1 style='text-align: center; color: lightgreen;'>Buy</h1>", unsafe_allow_html=True)
                elif upside_percent < -0.20:
                    st.markdown(f"<h1 style='text-align: center; color: red;'>STRONG SELL</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'><img src='[https://i.giphy.com/media/UfX4XeBMXWmNoGvBVK/giphy.gif](https://i.giphy.com/media/UfX4XeBMXWmNoGvBVK/giphy.gif)' width='300'></p>", unsafe_allow_html=True)
                elif upside_percent < -0.10:
                    st.markdown(f"<h1 style='text-align: center; color: orange;'>Sell</h1>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h1 style='text-align: center; color: grey;'>Hold</h1>", unsafe_allow_html=True)
            
            st.write("This model calculates the **Equity Value** per share by taking the company's projected cash flow (Enterprise Value), adding cash, and subtracting debt.")

            # --- NEW: Add warning for negative FCF ---
            if fcf < 0:
                st.warning(f"**Warning:** The 3-Year Average Free Cash Flow (FCF) for {ticker_symbol} is **${fcf:,.0f}**. A negative FCF will result in a negative valuation. This simple DCF model is not suitable for this company.")

            if discount_rate <= perpetual_growth_rate:
                st.error("Error: Discount Rate must be greater than Perpetual Growth Rate.")
            else:
                # --- 4. DISPLAY STATIC DCF RESULTS ---
                st.subheader("Valuation Results (Static Model)")
                col1, col2 = st.columns(2)
                col1.metric("Estimated Intrinsic Value", f"${intrinsic_price_per_share:,.2f}")
                col2.metric("Current Market Price", f"${market_price:,.2f}")
                
                if intrinsic_price_per_share > (market_price * 1.1): 
                    st.success(f"Based on our model, {ticker_symbol} appears **undervalued**.")
                elif intrinsic_price_per_share < (market_price * 0.9): 
                    st.warning(f"Based on our model, {ticker_symbol} appears **overvalued**.")
                else: 
                    st.info(f"Based on our model, {ticker_symbol} appears **fairly valued**.")

                st.subheader("Static Model Inputs")
                col1, col2, col3 = st.columns(3)
                # --- MODIFIED: Show 3-Year Avg FCF ---
                col1.metric("3-Year Avg. FCF", f"${fcf/1_000_000_000:,.2f}B")
                col2.metric("Plus: Total Cash", f"${cash/1_000_000_000:,.2f}B")
                col3.metric("Minus: Total Debt", f"${total_debt/1_000_000_000:,.2f}B")

                # --- NEW: DEBUGGER DOWNLOAD BUTTON ---
                st.subheader("Model Debugger")
                st.write("Export a .txt file explaining all the variables and math used in this calculation.")
                
                # Create the inputs dict for the debug function
                debug_inputs = {
                    "fcf": fcf, "cash": cash, "total_debt": total_debt, "shares": shares_outstanding,
                    "g": fcf_growth_rate, "wacc": discount_rate, "p": perpetual_growth_rate,
                    "ocf": op_cash_flow_avg, "capex": cap_ex_avg,
                    "ocf_series": op_cash_flow_series.iloc[:3],
                    "capex_series": cap_ex_series.iloc[:3]
                }
                
                debug_string = generate_dcf_debug_text(
                    ticker_symbol, 
                    debug_inputs, 
                    dcf_intermediates, 
                    intrinsic_price_per_share
                )
                
                st.download_button(
                    label="Export DCF Debugger (.txt)",
                    data=debug_string,
                    file_name=f"{ticker_symbol}_DCF_Debug_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

                # --- 5. MONTE CARLO SIMULATION ---
                st.subheader("Monte Carlo Simulation")
                if st.button(f"Run {st.session_state.simulations:,} Simulations"):
                    with st.spinner("Running Monte Carlo simulation..."):
                        
                        rand_growth = np.random.normal(fcf_growth_rate, growth_std, st.session_state.simulations)
                        rand_wacc = np.random.normal(discount_rate, wacc_std, st.session_state.simulations)
                        rand_perpetual = np.random.normal(perpetual_growth_rate, perpetual_std, st.session_state.simulations)
                        
                        results = []
                        for i in range(st.session_state.simulations):
                            if rand_wacc[i] > rand_perpetual[i] and rand_wacc[i] > 0 and rand_perpetual[i] >= 0:
                                sim_price, _ = calculate_dcf(fcf, cash, total_debt, shares_outstanding, rand_growth[i], rand_wacc[i], rand_perpetual[i])
                                if sim_price > 0 and sim_price < (market_price * 10): # Filter outliers
                                    results.append(sim_price)
                        
                        if not results:
                            st.error("Simulation failed. Check assumptions (e.g., WACC must be > Perpetual Growth).")
                        else:
                            results_series = pd.Series(results)
                            
                            fig = go.Figure(data=[go.Histogram(x=results_series, nbinsx=100, name="Price Distribution")])
                            
                            p5 = results_series.quantile(0.05)
                            p95 = results_series.quantile(0.95)
                            mean = results_series.mean()
                            
                            fig.add_vline(x=p5, line_dash="dash", line_color="green", name="5th Percentile")
                            fig.add_vline(x=p95, line_dash="dash", line_color="red", name="95th Percentile")
                            fig.add_vline(x=mean, line_color="black", name="Mean")
                            
                            fig.update_layout(title="Distribution of Intrinsic Values", xaxis_title="Intrinsic Value per Share ($)", yaxis_title="Frequency")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Simulation Statistics")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Mean (Average) Value", f"${mean:,.2f}")
                            col2.metric("5th Percentile", f"${p5:,.2f}")
                            col3.metric("95th Percentile", f"${p95:,.2f}")
                            st.info(f"There is a 90% probability that the intrinsic value is between ${p5:,.2f} and ${p95:,.2f}.")

        # --- NEW: EARNINGS CALENDAR PAGE ---
        elif page == "📅 Earnings Calendar":
            st.header("📅 Upcoming Earnings Calendar (Next 30 Days)")
            
            st.error("""
            **API LIMITATION:** Your request for the "Top 5 companies by market cap" is not feasible with the free Alpha Vantage API.
            
            * **Reason:** The Earnings Calendar API does *not* provide Market Cap. To get it, we would need a *separate API call* for *every* company, which would exceed the 25-call/day free limit instantly.
            * **Solution:** This tab shows the *full* earnings calendar for the next 30 days, as returned by the API (1 API call).
            """)
            
            try:
                av_key = st.secrets["alpha_vantage"]["API_KEY"]
                with st.spinner("Fetching earnings calendar from Alpha Vantage..."):
                    earnings_df = get_earnings_calendar(av_key)
                
                if not earnings_df.empty:
                    # Group by date and display
                    for report_date, group in earnings_df.groupby('Report Date'):
                        st.subheader(report_date.strftime('%Y-%m-%d'))
                        st.dataframe(group[['Ticker', 'Company Name', 'Analyst EPS']], use_container_width=True)
                else:
                    st.warning("No upcoming earnings found in the next 30 days.")
                    
            except Exception as e:
                st.error("Failed to load Alpha Vantage API key.")
                st.info("Please add `[alpha_vantage] \n API_KEY = 'YOUR_KEY'` to your `.streamlit/secrets.toml` file.")
                st.exception(e)


        elif page == "Future Forecast (Prophet)":
            st.header("Future Price Forecast (using Prophet)")
            st.write("This model uses time-series forecasting to predict future price movements, including a probabilistic range.")
            
            weeks_to_forecast = st.slider("Weeks to Forecast", 4, 104, 52) 
            
            if st.button("Run Price Forecast"):
                with st.spinner("Running forecast... This may take a minute."):
                    try:
                        model, forecast = get_prophet_forecast(price_data_cached, weeks_to_forecast)
                        st.subheader(f"{weeks_to_forecast}-Week Price Forecast")
                        fig = plot_plotly(model, forecast)
                        fig.update_layout(title=f"{ticker_symbol} Forecast", xaxis_title="Date", yaxis_title="Price")
                        st.write("The shaded blue area represents the probabilistic range (uncertainty interval) of the forecast.")
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("Forecast Data")
                        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(weeks_to_forecast * 7))
                    except Exception as e:
                        st.error(f"An error occurred during forecasting: {e}")
                        st.exception(e)
            else:
                st.info("Click 'Run Price Forecast' to generate the model and chart.")
        
        elif page == "🤖 AI Stock Assistant":
            st.header("🤖 AI Stock Assistant")
            st.write(f"Ask me anything about {ticker_symbol}, the dashboard, or general stock market concepts.")

            try:
                if "gemini_chat" not in st.session_state or st.session_state.current_ticker != ticker_symbol:
                    
                    context = f"""
                    You are an AI financial assistant for a stock dashboard.
                    The user is currently analyzing the following stock:
                    - Ticker: {ticker_symbol}
                    - Company Name: {info.get('longName')}
                    - Current Market Price: ${market_price:,.2f}
                    - P/E Ratio: {info.get('trailingPE', 'N/A')}
                    - EPS: {info.get('trailingEps', 'N/A')}
                    
                    Here is the data from the dashboard's DCF (Discounted Cash Flow) model:
                    - Static Intrinsic Value: ${intrinsic_price_per_share:,.2f}
                    - Model Verdict: {'Undervalued' if intrinsic_price_per_share > market_price * 1.1 else 'Overvalued' if intrinsic_price_per_share < market_price * 0.9 else 'Fairly Valued'}
                    - 3-Year Avg. FCF: ${fcf:,.0f}
                    - Total Cash: ${cash:,.0f}
                    - Total Debt: ${total_debt:,.0f}
                    - Shares Outstanding: {shares_outstanding:,}
                    
                    Today's date is {pd.Timestamp.now().strftime('%Y-%m-%d')}.
                    
                    Based on this context, answer the user's questions. If the question is general (e.g., "What is a P/E ratio?"), answer it without needing context.
                    """
                    
                    model = genai.GenerativeModel("models/gemini-2.5-pro") 
                    
                    st.session_state.gemini_chat = model.start_chat(history=[
                        {'role': 'user', 'parts': [context]},
                        {'role': 'model', 'parts': [f"Hello! I'm ready to answer your questions about {ticker_symbol} or anything else. What's on your mind?"]}
                    ])
                    st.session_state.current_ticker = ticker_symbol 

                for message in st.session_state.gemini_chat.history:
                    if message.parts[0].text.startswith("You are an AI financial assistant"):
                        continue
                    with st.chat_message(message.role):
                        st.markdown(message.parts[0].text)

                if prompt := st.chat_input(f"Ask about {ticker_symbol}..."):
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    try:
                        response = st.session_state.gemini_chat.send_message(prompt)
                        with st.chat_message("model"):
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error communicating with AI: {e}")

            except Exception as e:
                st.error(f"Failed to initialize AI context. The DCF model may have failed to run. Error: {e}")
                st.exception(e)