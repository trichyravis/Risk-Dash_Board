
"""
Portfolio Risk Dashboard - Real Nifty 50 Data
The Mountain Path - World of Finance

Prof. V. Ravichandran
28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence

Features:
- Real-time data from Yahoo Finance
- Actual Nifty 50 stocks
- Live correlations and volatilities
- VaR/ES based on real market behavior
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================
COLORS = {
    'dark_blue': '#003366',
    'medium_blue': '#004d80',
    'light_blue': '#ADD8E6',
    'accent_gold': '#FFD700',
    'bg_dark': '#0a1628',
    'card_bg': '#112240',
    'text_primary': '#e6f1ff',
    'text_secondary': '#8892b0',
    'text_dark': '#1a1a2e',
    'success': '#28a745',
    'danger': '#dc3545',
}

BRANDING = {
    'name': 'The Mountain Path - World of Finance',
    'instructor': 'Prof. V. Ravichandran',
    'credentials': '28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence',
    'icon': '🏔️',
    'linkedin': 'https://www.linkedin.com/in/trichyravis',
    'github': 'https://github.com/trichyravis',
}

# ============================================================================
# NIFTY 50 STOCKS & MIXED PORTFOLIO OPTIONS
# ============================================================================
NIFTY50_STOCKS = {
    # Large Cap Blue Chips
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'ICICIBANK.NS': 'ICICI Bank',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ITC.NS': 'ITC Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro',
    'AXISBANK.NS': 'Axis Bank',
    'ASIANPAINT.NS': 'Asian Paints',
    'MARUTI.NS': 'Maruti Suzuki',
    'TITAN.NS': 'Titan Company',
}

US_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com',
    'NVDA': 'NVIDIA Corp.',
    'JPM': 'JP Morgan Chase',
    'V': 'Visa Inc.',
}

COMMODITIES = {
    'GC=F': 'Gold Futures',
    'SI=F': 'Silver Futures',
}

INDICES = {
    '^NSEI': 'Nifty 50 Index',
    '^GSPC': 'S&P 500 Index',
}

# Preset portfolios
PRESET_PORTFOLIOS = {
    'Nifty 50 Blue Chips': {
        'RELIANCE.NS': 0.15,
        'TCS.NS': 0.12,
        'HDFCBANK.NS': 0.12,
        'INFY.NS': 0.10,
        'ICICIBANK.NS': 0.10,
        'BHARTIARTL.NS': 0.10,
        'ITC.NS': 0.08,
        'KOTAKBANK.NS': 0.08,
        'LT.NS': 0.08,
        'MARUTI.NS': 0.07,
    },
    'Global Tech Focused': {
        'TCS.NS': 0.15,
        'INFY.NS': 0.15,
        'AAPL': 0.15,
        'MSFT': 0.15,
        'GOOGL': 0.12,
        'NVDA': 0.12,
        'AMZN': 0.08,
        '^NSEI': 0.08,
    },
    'Balanced Global': {
        'RELIANCE.NS': 0.10,
        'HDFCBANK.NS': 0.10,
        'AAPL': 0.10,
        'MSFT': 0.10,
        'JPM': 0.08,
        'GC=F': 0.10,
        '^NSEI': 0.12,
        '^GSPC': 0.10,
        'BHARTIARTL.NS': 0.10,
        'TCS.NS': 0.10,
    },
}

# ============================================================================
# STYLING FUNCTION
# ============================================================================
def apply_styles():
    """Apply all custom CSS styling"""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

        .stApp {{
            background: linear-gradient(135deg, #1a2332 0%, #243447 50%, #2a3f5f 100%);
        }}
        
        .main {{
            color: {COLORS['text_primary']} !important;
        }}
        
        .main *, .main p, .main span, .main div, .main li, .main label {{
            color: {COLORS['text_primary']} !important;
        }}
        
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {{
            color: {COLORS['accent_gold']} !important;
            font-family: 'Playfair Display', serif;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['bg_dark']} 0%, {COLORS['dark_blue']} 100%);
            border-right: 1px solid rgba(255,215,0,0.2);
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {COLORS['text_primary']} !important;
        }}

        section[data-testid="stSidebar"] input {{
            color: {COLORS['text_dark']} !important;
            background-color: #ffffff !important;
        }}

        .header-container {{
            background: linear-gradient(135deg, {COLORS['dark_blue']}, {COLORS['medium_blue']});
            border: 2px solid {COLORS['accent_gold']};
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }}
        
        .header-container h1 {{
            font-family: 'Playfair Display', serif;
            color: {COLORS['accent_gold']};
            margin: 0;
            font-size: 2rem;
        }}
        
        .header-container p {{
            color: {COLORS['text_primary']};
            font-family: 'Source Sans Pro', sans-serif;
            margin: 0.3rem 0 0;
            font-size: 0.9rem;
        }}

        .metric-card {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(255,215,0,0.3);
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
            margin-bottom: 0.8rem;
        }}
        
        .metric-card .label {{
            color: {COLORS['text_secondary']};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-card .value {{
            color: {COLORS['accent_gold']};
            font-size: 1.6rem;
            font-weight: 700;
            font-family: 'Playfair Display', serif;
            margin-top: 0.3rem;
        }}

        .info-box {{
            background: rgba(0,51,102,0.5);
            border: 1px solid {COLORS['accent_gold']};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            color: {COLORS['text_primary']};
            margin: 0.8rem 0;
        }}

        .section-title {{
            font-family: 'Playfair Display', serif;
            color: {COLORS['accent_gold']};
            font-size: 1.3rem;
            border-bottom: 2px solid rgba(255,215,0,0.3);
            padding-bottom: 0.5rem;
            margin: 1.5rem 0 1rem;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(255,215,0,0.3);
            border-radius: 8px;
            color: {COLORS['text_primary']};
            padding: 0.5rem 1rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: {COLORS['dark_blue']};
            border: 2px solid {COLORS['accent_gold']};
            color: {COLORS['accent_gold']};
        }}

        .stAlert {{
            background-color: rgba(255, 255, 255, 0.95) !important;
        }}
        
        .stAlert p, .stAlert span, .stAlert div {{
            color: {COLORS['text_dark']} !important;
        }}

        /* Expander styling for Asset Details */
        .streamlit-expanderHeader {{
            background: {COLORS['card_bg']} !important;
            border: 1px solid rgba(255,215,0,0.3) !important;
            border-radius: 8px !important;
            color: {COLORS['text_primary']} !important;
        }}
        
        .streamlit-expanderHeader p {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600 !important;
        }}
        
        .streamlit-expanderContent {{
            background: rgba(17, 34, 64, 0.5) !important;
            border: 1px solid rgba(255,215,0,0.2) !important;
            border-top: none !important;
        }}

        footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# COMPONENT FUNCTIONS
# ============================================================================
def header_container(title, subtitle=None, description=None):
    subtitle_html = f'<p style="font-size:1rem; color:{COLORS["accent_gold"]}; font-weight:600; margin:0.5rem 0;">{subtitle}</p>' if subtitle else ""
    description_html = f'<p style="font-size:0.85rem; color:{COLORS["text_primary"]}; margin:0.3rem 0;">{description}</p>' if description else ""
    
    st.markdown(f"""
    <div class="header-container">
        <h1>{BRANDING['icon']} {title}</h1>
        {subtitle_html}
        {description_html}
        <p>{BRANDING['name']}</p>
        <p style="font-size:0.8rem; color:{COLORS['text_secondary']};">
            {BRANDING['instructor']} | {BRANDING['credentials']}
        </p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, help_text=None):
    help_html = f' title="{help_text}"' if help_text else ''
    st.markdown(f"""
    <div class="metric-card"{help_html}>
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def info_box(content, title=None):
    title_html = f"<h4 style='color:{COLORS['accent_gold']}; margin-top:0;'>{title}</h4>" if title else ""
    st.markdown(f"""
    <div class="info-box">
        {title_html}
        {content}
    </div>
    """, unsafe_allow_html=True)


def warning_box(message):
    st.markdown(f"""
    <div class="info-box" style="border-color:{COLORS['danger']};">
        <span style="color:{COLORS['danger']};">⚠</span> {message}
    </div>
    """, unsafe_allow_html=True)


def footer():
    st.divider()
    st.markdown(f"""
    <div style="text-align:center; padding:1.5rem;">
        <p style="color:{COLORS['accent_gold']}; font-family:'Playfair Display', serif; 
                  font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;">
            {BRANDING['icon']} {BRANDING['name']}
        </p>
        <p style="color:{COLORS['text_secondary']}; font-size:0.85rem; margin:0.3rem 0;">
            {BRANDING['instructor']} | {BRANDING['credentials']}
        </p>
        <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid rgba(255,215,0,0.3);">
            <p style="color:{COLORS['text_primary']}; font-size:0.9rem; margin:0.5rem 0;">
                <a href="{BRANDING['linkedin']}" target="_blank" 
                   style="color:{COLORS['accent_gold']}; text-decoration:none; margin:0 1rem;">
                    🔗 LinkedIn
                </a>
                <a href="{BRANDING['github']}" target="_blank" 
                   style="color:{COLORS['accent_gold']}; text-decoration:none; margin:0 1rem;">
                    💻 GitHub
                </a>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_market_data(tickers, start_date, end_date):
    """Fetch real market data from Yahoo Finance"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error("No data received from Yahoo Finance")
            return None, None
        
        # Handle single ticker vs multiple tickers
        if len(tickers) == 1:
            # Single ticker - data structure is different
            if 'Close' in data.columns:
                prices = data['Close'].to_frame()
                prices.columns = tickers
            else:
                st.error("'Close' column not found in data")
                return None, None
        else:
            # Multiple tickers
            if 'Close' in data.columns:
                prices = data['Close']
            elif isinstance(data.columns, pd.MultiIndex):
                # Sometimes data comes with MultiIndex columns
                prices = data['Close']
            else:
                st.error("Unable to extract 'Close' prices from data")
                return None, None
        
        # Drop any tickers with insufficient data
        prices = prices.dropna(axis=1, how='all')
        
        if prices.empty:
            st.error("No valid price data after cleaning")
            return None, None
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Additional validation
        if returns.empty:
            st.error("Unable to calculate returns from price data")
            return None, None
        
        return prices, returns
    
    except Exception as e:
        st.warning(f"Primary fetch failed: {str(e)}")
        st.info("🔄 Trying alternative method (individual ticker fetch)...")
        return fetch_market_data_alternative(tickers, start_date, end_date)


@st.cache_data(ttl=3600)
def fetch_market_data_alternative(tickers, start_date, end_date):
    """Alternative: Fetch tickers one by one"""
    all_prices = []
    valid_tickers = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty and 'Close' in data.columns:
                prices_series = data['Close']
                prices_series.name = ticker
                all_prices.append(prices_series)
                valid_tickers.append(ticker)
        except:
            continue
    
    if not all_prices:
        return None, None
    
    prices_df = pd.concat(all_prices, axis=1)
    returns_df = prices_df.pct_change().dropna()
    
    return prices_df, returns_df


def get_asset_name(ticker):
    """Get human-readable name for ticker"""
    all_assets = {**NIFTY50_STOCKS, **US_STOCKS, **COMMODITIES, **INDICES}
    return all_assets.get(ticker, ticker)


# ============================================================================
# RISK CALCULATION FUNCTIONS
# ============================================================================
def calculate_var_es(returns, confidence=0.95, method='historical'):
    """Calculate VaR and Expected Shortfall"""
    if method == 'historical':
        var = np.percentile(returns, (1 - confidence) * 100)
        es = returns[returns <= var].mean()
    elif method == 'parametric':
        mu = returns.mean()
        sigma = returns.std()
        var = stats.norm.ppf(1 - confidence, mu, sigma)
        es = mu - sigma * stats.norm.pdf(stats.norm.ppf(1 - confidence)) / (1 - confidence)
    else:  # monte carlo
        simulated = np.random.normal(returns.mean(), returns.std(), 10000)
        var = np.percentile(simulated, (1 - confidence) * 100)
        es = simulated[simulated <= var].mean()
    
    return var, es


def calculate_portfolio_metrics(returns_df, weights):
    """Calculate portfolio-level metrics"""
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    metrics = {
        'mean_return': portfolio_returns.mean() * 252,
        'volatility': portfolio_returns.std() * np.sqrt(252),
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
        'max_drawdown': (portfolio_returns.cumsum().cummax() - portfolio_returns.cumsum()).max(),
        'skewness': stats.skew(portfolio_returns),
        'kurtosis': stats.kurtosis(portfolio_returns),
    }
    
    return portfolio_returns, metrics


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Portfolio Risk Dashboard | Mountain Path",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_styles()

# ============================================================================
# HEADER
# ============================================================================
header_container(
    title="Portfolio Risk Dashboard",
    subtitle="Real-Time Market Data Analysis",
    description="Live Nifty 50 Stocks | Global Equities | VaR & ES | Stress Testing"
)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.markdown(f"""
<div style="text-align:center; padding:1.2rem; background:rgba(255,215,0,0.08);
     border-radius:10px; margin-bottom:1.5rem; border:2px solid {COLORS['accent_gold']};">
    <h3 style="color:{COLORS['accent_gold']}; margin:0;">{BRANDING['icon']} RISK ANALYTICS</h3>
    <p style="color:{COLORS['text_secondary']}; font-size:0.75rem; margin:5px 0 0;">
        Real Market Data</p>
</div>
""", unsafe_allow_html=True)

# Portfolio Selection
st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>📊 Portfolio Selection</p>", unsafe_allow_html=True)

portfolio_mode = st.sidebar.radio(
    "Portfolio Type",
    ["Preset Portfolio", "Custom Selection"],
    label_visibility="collapsed"
)

if portfolio_mode == "Preset Portfolio":
    selected_preset = st.sidebar.selectbox(
        "Choose Preset",
        list(PRESET_PORTFOLIOS.keys())
    )
    portfolio_dict = PRESET_PORTFOLIOS[selected_preset].copy()
    
    # Option to modify preset weights
    modify_weights = st.sidebar.checkbox("Customize Weights", value=False)
    
    if modify_weights:
        st.sidebar.markdown("**Adjust Weights (must sum to 100%):**")
        new_weights = {}
        
        for ticker in portfolio_dict.keys():
            weight_pct = st.sidebar.number_input(
                f"{get_asset_name(ticker)}",
                min_value=0.0,
                max_value=100.0,
                value=portfolio_dict[ticker] * 100,
                step=1.0,
                key=f"weight_{ticker}"
            )
            new_weights[ticker] = weight_pct / 100
        
        # Check if weights sum to 100%
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.sidebar.warning(f"⚠️ Weights sum to {total_weight*100:.1f}%. Must equal 100%!")
            # Auto-normalize
            if st.sidebar.button("Auto-Normalize Weights"):
                new_weights = {k: v/total_weight for k, v in new_weights.items()}
                st.sidebar.success("✅ Weights normalized!")
        
        portfolio_dict = new_weights
        
else:
    # Custom selection
    st.sidebar.markdown("**Select Assets:**")
    
    all_assets = {
        **NIFTY50_STOCKS,
        **US_STOCKS,
        **COMMODITIES,
        **INDICES
    }
    
    selected_tickers = st.sidebar.multiselect(
        "Choose assets (max 15)",
        options=list(all_assets.keys()),
        default=list(PRESET_PORTFOLIOS['Nifty 50 Blue Chips'].keys())[:5],
        format_func=lambda x: f"{get_asset_name(x)} ({x})",
        max_selections=15
    )
    
    if not selected_tickers:
        st.sidebar.warning("Please select at least one asset")
        st.stop()
    
    # Weight allocation method
    weight_method = st.sidebar.radio(
        "Weight Allocation Method",
        ["Equal Weight", "Custom Weights", "Market Cap Weighted"],
        help="Choose how to allocate portfolio weights"
    )
    
    if weight_method == "Equal Weight":
        equal_weight = 1.0 / len(selected_tickers)
        portfolio_dict = {ticker: equal_weight for ticker in selected_tickers}
        
    elif weight_method == "Custom Weights":
        st.sidebar.markdown("**Set Custom Weights (must sum to 100%):**")
        portfolio_dict = {}
        
        for ticker in selected_tickers:
            weight_pct = st.sidebar.number_input(
                f"{get_asset_name(ticker)}",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / len(selected_tickers),
                step=1.0,
                key=f"custom_weight_{ticker}"
            )
            portfolio_dict[ticker] = weight_pct / 100
        
        # Validate weights
        total_weight = sum(portfolio_dict.values())
        if abs(total_weight - 1.0) > 0.01:
            st.sidebar.warning(f"⚠️ Weights sum to {total_weight*100:.1f}%. Must equal 100%!")
            if st.sidebar.button("Auto-Normalize"):
                portfolio_dict = {k: v/total_weight for k, v in portfolio_dict.items()}
                st.sidebar.success("✅ Normalized!")
        else:
            st.sidebar.success(f"✅ Weights sum to {total_weight*100:.0f}%")
            
    else:  # Market Cap Weighted
        st.sidebar.info("💡 Market cap weights are approximated for demo purposes")
        # Simplified market cap weights (you can enhance this with real market cap data)
        market_cap_weights = {
            # Indian stocks - approximate market cap weights
            'RELIANCE.NS': 0.15, 'TCS.NS': 0.12, 'HDFCBANK.NS': 0.12,
            'INFY.NS': 0.08, 'ICICIBANK.NS': 0.08, 'BHARTIARTL.NS': 0.07,
            'ITC.NS': 0.06, 'KOTAKBANK.NS': 0.06,
            # US stocks
            'AAPL': 0.15, 'MSFT': 0.14, 'GOOGL': 0.10, 'AMZN': 0.09,
            'NVDA': 0.08, 'JPM': 0.05, 'V': 0.04,
            # Commodities & Indices
            'GC=F': 0.03, '^NSEI': 0.08, '^GSPC': 0.08,
        }
        
        # Get weights for selected tickers
        raw_weights = {t: market_cap_weights.get(t, 1.0) for t in selected_tickers}
        total = sum(raw_weights.values())
        portfolio_dict = {t: w/total for t, w in raw_weights.items()}

# Portfolio Value
st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>💰 Portfolio Settings</p>", unsafe_allow_html=True)
portfolio_value = st.sidebar.number_input(
    "Portfolio Value (₹)", 
    min_value=100000, 
    max_value=1000000000, 
    value=10000000, 
    step=100000,
    format="%d"
)

# Risk Parameters
st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>⚙️ Risk Calculation</p>", unsafe_allow_html=True)
confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
time_horizon = st.sidebar.selectbox("Time Horizon (days)", [1, 5, 10, 21], index=0)
var_method = st.sidebar.selectbox("VaR Method", ["historical", "parametric", "monte_carlo"])

# Data Period
st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>📅 Data Period</p>", unsafe_allow_html=True)
lookback_period = st.sidebar.selectbox(
    "Lookback Period",
    ["3 Months", "6 Months", "1 Year", "2 Years", "3 Years"],
    index=2
)

period_map = {
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "2 Years": 504,
    "3 Years": 756
}
lookback_days = period_map[lookback_period]

# ============================================================================
# FETCH REAL MARKET DATA
# ============================================================================
tickers = list(portfolio_dict.keys())
weights = np.array(list(portfolio_dict.values()))

# Calculate dates
end_date = datetime.now()
start_date = end_date - timedelta(days=int(lookback_days * 1.5))  # Extra buffer

with st.spinner('📡 Fetching real-time market data from Yahoo Finance...'):
    prices_df, returns_df = fetch_market_data(tickers, start_date, end_date)

if prices_df is None or returns_df is None or returns_df.empty:
    st.error("❌ Unable to fetch market data. Please try different assets or time period.")
    st.stop()

# Filter to only include tickers that have data
valid_tickers = returns_df.columns.tolist()
if len(valid_tickers) < len(tickers):
    missing = set(tickers) - set(valid_tickers)
    st.warning(f"⚠️ Could not fetch data for: {', '.join(missing)}")
    
    # Update weights for valid tickers only
    valid_weights = []
    for ticker in valid_tickers:
        valid_weights.append(portfolio_dict[ticker])
    weights = np.array(valid_weights)
    weights = weights / weights.sum()  # Renormalize

# Data info
data_start = returns_df.index[0].strftime('%Y-%m-%d')
data_end = returns_df.index[-1].strftime('%Y-%m-%d')
n_days = len(returns_df)

st.success(f"✅ Loaded {n_days} days of real market data ({data_start} to {data_end})")

# Calculate portfolio metrics
portfolio_returns, metrics = calculate_portfolio_metrics(returns_df, weights)
var_pct, es_pct = calculate_var_es(portfolio_returns, confidence_level, var_method)
var_amount = abs(var_pct) * portfolio_value
es_amount = abs(es_pct) * portfolio_value

# Display current portfolio allocation
with st.expander("📊 Current Portfolio Allocation", expanded=False):
    st.markdown("### Portfolio Composition")
    
    allocation_data = []
    for ticker in valid_tickers:
        weight = weights[list(valid_tickers).index(ticker)]
        allocation_data.append({
            'Asset': get_asset_name(ticker),
            'Ticker': ticker,
            'Weight (%)': f'{weight*100:.2f}%',
            'Value (₹)': f'₹{weight * portfolio_value:,.0f}'
        })
    
    allocation_df = pd.DataFrame(allocation_data)
    st.dataframe(allocation_df, use_container_width=True, hide_index=True)
    
    # Validation
    total_weight_pct = sum(weights) * 100
    if abs(total_weight_pct - 100) < 0.1:
        st.success(f"✅ Portfolio weights sum to {total_weight_pct:.1f}%")
    else:
        st.warning(f"⚠️ Portfolio weights sum to {total_weight_pct:.1f}%")

# ============================================================================
# MAIN DASHBOARD - TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Risk Overview",
    "📊 Portfolio Analytics",
    "🔥 Stress Testing",
    "🔗 Correlation Analysis",
    "📚 Asset Details"
])

# ========== TAB 1: RISK OVERVIEW ==========
with tab1:
    section_title("🎯 Key Risk Metrics")
    st.caption(f"Portfolio: ₹{portfolio_value:,.0f} | Confidence: {confidence_level*100:.0f}% | Method: {var_method.title()} | Data: {data_start} to {data_end}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Value at Risk", f"₹{var_amount:,.0f}", f"{abs(var_pct)*100:.2f}% loss")
    with col2:
        metric_card("Expected Shortfall", f"₹{es_amount:,.0f}", f"{abs(es_pct)*100:.2f}% loss")
    with col3:
        metric_card("Portfolio Volatility", f"{metrics['volatility']*100:.2f}%", "Annualized")
    with col4:
        metric_card("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", "Risk-adjusted")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Annual Return", f"{metrics['mean_return']*100:.2f}%", "Expected")
    with col2:
        metric_card("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%", "Worst decline")
    with col3:
        metric_card("Skewness", f"{metrics['skewness']:.3f}", "Asymmetry")
    
    st.markdown("---")
    section_title("📊 Risk Interpretation")
    
    col1, col2 = st.columns(2)
    with col1:
        info_box(f"""
            <strong>Value at Risk - {confidence_level*100:.0f}% Confidence</strong>
            <p style="margin-top:0.5rem;">
            Based on <strong>real market data</strong>, with {confidence_level*100:.0f}% confidence, 
            your portfolio losses will not exceed 
            <strong style="color:{COLORS['accent_gold']};">₹{var_amount:,.0f}</strong> 
            over the next {time_horizon} day(s).
            </p>
            <p style="margin-top:0.5rem; font-size:0.85rem;">
            This is calculated from actual historical volatility and correlations 
            in the {lookback_period.lower()} period.
            </p>
        """, title="VaR - Real Market Data")
    
    with col2:
        info_box(f"""
            <strong>Expected Shortfall (Tail Risk)</strong>
            <p style="margin-top:0.5rem;">
            If losses exceed VaR, the average loss based on historical data is 
            <strong style="color:{COLORS['danger']};">₹{es_amount:,.0f}</strong>.
            </p>
            <p style="margin-top:0.5rem; font-size:0.85rem;">
            ES captures the severity of tail events observed in actual market movements 
            and is the preferred risk measure under Basel III.
            </p>
        """, title="ES - Tail Risk Measure")
    
    st.markdown("---")
    section_title("📈 Portfolio Return Distribution (Real Data)")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1 = axes[0]
    ax1.hist(portfolio_returns * 100, bins=50, density=True, color=COLORS['light_blue'], alpha=0.7, edgecolor='white')
    ax1.axvline(var_pct * 100, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'VaR ({confidence_level*100:.0f}%)')
    ax1.axvline(es_pct * 100, color='darkred', linestyle=':', linewidth=2, label=f'ES ({confidence_level*100:.0f}%)')
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Actual Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    stats.probplot(portfolio_returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Test)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    section_title("📊 Historical Performance")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    axes[0].plot(cumulative_returns.index, cumulative_returns * 100, color=COLORS['accent_gold'], linewidth=2)
    axes[0].set_ylabel('Cumulative Return (%)')
    axes[0].set_title(f'Portfolio Cumulative Returns ({data_start} to {data_end})')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    axes[1].plot(portfolio_returns.index, portfolio_returns * 100, color=COLORS['medium_blue'], alpha=0.7, linewidth=0.8)
    axes[1].axhline(var_pct * 100, color=COLORS['danger'], linestyle='--', linewidth=1.5, label=f'VaR', alpha=0.7)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Daily Return (%)')
    axes[1].set_title('Daily Returns with VaR Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========== TAB 2: PORTFOLIO ANALYTICS ==========
with tab2:
    section_title("📊 Individual Asset Performance (Real Data)")
    
    asset_metrics = []
    for i, ticker in enumerate(valid_tickers):
        asset_returns = returns_df[ticker]
        asset_name = get_asset_name(ticker)
        
        asset_metrics.append({
            'Asset': asset_name,
            'Ticker': ticker,
            'Weight': f'{weights[i]*100:.1f}%',
            'Ann. Return': f'{asset_returns.mean()*252*100:.2f}%',
            'Ann. Volatility': f'{asset_returns.std()*np.sqrt(252)*100:.2f}%',
            'Sharpe': f'{(asset_returns.mean()*252)/(asset_returns.std()*np.sqrt(252)):.2f}' if asset_returns.std() > 0 else 'N/A',
            'VaR (95%)': f'{np.percentile(asset_returns, 5)*100:.2f}%',
            'Current Price': f'₹{prices_df[ticker].iloc[-1]:.2f}' if ticker.endswith('.NS') else f'${prices_df[ticker].iloc[-1]:.2f}',
        })
    
    metrics_df = pd.DataFrame(asset_metrics)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    section_title("📈 Asset Price Performance (Normalized)")
    
    # Normalize prices to 100
    normalized_prices = (prices_df / prices_df.iloc[0]) * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in valid_tickers:
        ax.plot(normalized_prices.index, normalized_prices[ticker], 
                label=f'{get_asset_name(ticker)}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price (Base=100)')
    ax.set_title(f'Asset Performance Comparison ({data_start} to {data_end})')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    section_title("🥧 Portfolio Composition")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = [get_asset_name(t) for t in valid_tickers]
        colors_list = [COLORS['dark_blue'], COLORS['medium_blue'], COLORS['light_blue'], 
                      COLORS['accent_gold'], COLORS['text_secondary']]
        colors_extended = colors_list * (len(valid_tickers) // len(colors_list) + 1)
        ax.pie(weights, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=colors_extended[:len(valid_tickers)])
        ax.set_title('Portfolio Allocation')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Asset type breakdown
        asset_types = {'Indian Stocks': 0, 'US Stocks': 0, 'Commodities': 0, 'Indices': 0}
        for i, ticker in enumerate(valid_tickers):
            if ticker in NIFTY50_STOCKS:
                asset_types['Indian Stocks'] += weights[i]
            elif ticker in US_STOCKS:
                asset_types['US Stocks'] += weights[i]
            elif ticker in COMMODITIES:
                asset_types['Commodities'] += weights[i]
            elif ticker in INDICES:
                asset_types['Indices'] += weights[i]
        
        asset_types = {k: v for k, v in asset_types.items() if v > 0}
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(asset_types.values(), labels=asset_types.keys(), autopct='%1.1f%%', 
               startangle=90, colors=[COLORS['accent_gold'], COLORS['medium_blue'], 
                                     COLORS['light_blue'], COLORS['dark_blue']])
        ax.set_title('Asset Class Distribution')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ========== TAB 3: STRESS TESTING ==========
with tab3:
    section_title("🔥 Stress Test Scenarios")
    
    st.markdown(f"""
    <div class="info-box">
        <h4 style='color:{COLORS['accent_gold']}; margin-top:0;'>Stress Testing Framework</h4>
        <p style='margin-top:0.5rem;'>
        <strong>Scenario Analysis on Real Portfolio</strong><br/>
        These scenarios simulate extreme market conditions applied to your actual portfolio holdings.
        Impacts are calculated based on real volatilities and correlations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    scenarios = {
        'Market Crash (-20%)': -0.20,
        'Moderate Correction (-10%)': -0.10,
        'Volatility Spike (2x)': -metrics['volatility'] * 1.5,
        'Asian Crisis Scenario': -0.15,
        'Global Recession': -0.25,
        'Black Swan Event (-30%)': -0.30,
    }
    
    stress_results = []
    for name, shock in scenarios.items():
        shocked_value = portfolio_value * (1 + shock)
        impact = shocked_value - portfolio_value
        stress_results.append({
            'Scenario': name,
            'Shock': f'{shock*100:.1f}%',
            'Portfolio Value': f'₹{shocked_value:,.0f}',
            'P&L Impact': f'₹{impact:,.0f}',
            'vs VaR': f'{abs(impact/var_amount):.1f}x'
        })
    
    st.dataframe(pd.DataFrame(stress_results), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    section_title("📊 Stress Impact Visualization")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    impacts = [float(s['P&L Impact'].replace('₹', '').replace(',', '')) for s in stress_results]
    colors_bars = [COLORS['danger'] if x < 0 else COLORS['success'] for x in impacts]
    ax.barh([s['Scenario'] for s in stress_results], impacts, color=colors_bars)
    ax.axvline(-var_amount, color=COLORS['accent_gold'], linestyle='--', linewidth=2, label='VaR')
    ax.axvline(-es_amount, color='orange', linestyle=':', linewidth=2, label='ES')
    ax.set_xlabel('P&L Impact (₹)')
    ax.set_title('Stress Test Scenario Impacts vs VaR/ES')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1e6:.1f}M'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========== TAB 4: CORRELATION ==========
with tab4:
    section_title("🔗 Asset Correlation Matrix (Real Data)")
    
    st.markdown(f"""
    <div class="info-box">
        <h4 style='color:{COLORS['accent_gold']}; margin-top:0;'>Correlation Analysis</h4>
        <p style='margin-top:0.5rem;'>
        <strong>Historical Correlations from Market Data</strong><br/>
        These correlations are calculated from actual historical returns over the selected period.
        Understanding correlations helps optimize diversification.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    correlation_matrix = returns_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    labels = [get_asset_name(t) for t in valid_tickers]
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=labels, yticklabels=labels,
                center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title(f'Asset Correlation Matrix ({lookback_period})')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    section_title("📊 Correlation Statistics")
    
    corr_values = []
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            corr_values.append(correlation_matrix.iloc[i, j])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Average Correlation", f"{np.mean(corr_values):.3f}", "Portfolio diversification")
    with col2:
        metric_card("Max Correlation", f"{np.max(corr_values):.3f}", "Highest co-movement")
    with col3:
        metric_card("Min Correlation", f"{np.min(corr_values):.3f}", "Lowest co-movement")

# ========== TAB 5: ASSET DETAILS ==========
with tab5:
    section_title("📚 Portfolio Holdings Details")
    
    # Summary info with better visibility
    st.markdown(f"""
    <div class="info-box">
        <p><strong>Current Portfolio:</strong> {len(valid_tickers)} assets</p>
        <p><strong>Data Period:</strong> {data_start} to {data_end} ({n_days} trading days)</p>
        <p><strong>Total Portfolio Value:</strong> ₹{portfolio_value:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    for i, ticker in enumerate(valid_tickers):
        asset_name = get_asset_name(ticker)
        asset_weight = weights[i]
        asset_value = portfolio_value * asset_weight
        asset_returns = returns_df[ticker]
        
        with st.expander(f"📊 {asset_name} ({ticker}) - {asset_weight*100:.1f}% allocation", expanded=False):
            # Use metric cards for better visibility
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Allocation</div>
                    <div class="value">₹{asset_value:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Weight</div>
                    <div class="value">{asset_weight*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Annual Return</div>
                    <div class="value">{asset_returns.mean()*252*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Annual Volatility</div>
                    <div class="value">{asset_returns.std()*np.sqrt(252)*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                sharpe = (asset_returns.mean()*252)/(asset_returns.std()*np.sqrt(252)) if asset_returns.std() > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Sharpe Ratio</div>
                    <div class="value">{sharpe:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">VaR (95%)</div>
                    <div class="value">{np.percentile(asset_returns, 5)*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Mini chart
            st.markdown(f"<p style='color:{COLORS['text_primary']}; margin-top:1rem; font-weight:600;'>Price History</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(prices_df.index, prices_df[ticker], color=COLORS['accent_gold'], linewidth=1.5)
            ax.set_title(f'{asset_name} Price History', color=COLORS['text_primary'])
            ax.set_xlabel('Date', color=COLORS['text_secondary'])
            ax.set_ylabel('Price', color=COLORS['text_secondary'])
            ax.tick_params(colors=COLORS['text_secondary'])
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#0f1824')
            fig.patch.set_facecolor('#0f1824')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

footer()
