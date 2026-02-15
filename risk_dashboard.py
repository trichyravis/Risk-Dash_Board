
"""
Portfolio Risk Dashboard - Standalone Version
The Mountain Path - World of Finance

Prof. V. Ravichandran
28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence

Single-file version with all components embedded for easy deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def generate_portfolio_data(n_assets=5, n_days=252):
    """Generate synthetic portfolio data"""
    np.random.seed(42)
    
    assets = [f'Asset {i+1}' for i in range(n_assets)]
    correlation = np.random.uniform(0.3, 0.7, size=(n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    
    mean_returns = np.random.uniform(-0.0005, 0.002, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)
    
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    returns_df = pd.DataFrame(returns, columns=assets, index=dates)
    prices_df = (1 + returns_df).cumprod() * 100
    
    return returns_df, prices_df, correlation, assets


def calculate_portfolio_metrics(returns_df, weights):
    """Calculate portfolio-level metrics"""
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    metrics = {
        'mean_return': portfolio_returns.mean() * 252,
        'volatility': portfolio_returns.std() * np.sqrt(252),
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
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
    subtitle="Real-Time Risk Monitoring & Stress Testing",
    description="VaR & ES | Scenario Analysis | Portfolio Analytics | Correlation Heatmaps"
)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.markdown(f"""
<div style="text-align:center; padding:1.2rem; background:rgba(255,215,0,0.08);
     border-radius:10px; margin-bottom:1.5rem; border:2px solid {COLORS['accent_gold']};">
    <h3 style="color:{COLORS['accent_gold']}; margin:0;">{BRANDING['icon']} RISK ANALYTICS</h3>
    <p style="color:{COLORS['text_secondary']}; font-size:0.75rem; margin:5px 0 0;">
        Configure your analysis</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>📊 Portfolio Settings</p>", unsafe_allow_html=True)
n_assets = st.sidebar.slider("Number of Assets", 3, 10, 5)
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=100000, max_value=100000000, 
                                          value=10000000, step=100000, format="%d")

st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>⚙️ Risk Calculation</p>", unsafe_allow_html=True)
confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
time_horizon = st.sidebar.selectbox("Time Horizon (days)", [1, 5, 10, 21], index=0)
var_method = st.sidebar.selectbox("VaR Method", ["historical", "parametric", "monte_carlo"])

st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>📅 Data Period</p>", unsafe_allow_html=True)
lookback_days = st.sidebar.slider("Lookback Period (days)", 63, 756, 252)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_portfolio_data(n_assets, n_days):
    return generate_portfolio_data(n_assets, n_days)

with st.spinner('Generating portfolio data...'):
    returns_df, prices_df, correlation_matrix, asset_names = load_portfolio_data(n_assets, lookback_days)

weights = np.ones(n_assets) / n_assets
portfolio_returns, metrics = calculate_portfolio_metrics(returns_df, weights)
var_pct, es_pct = calculate_var_es(portfolio_returns, confidence_level, var_method)
var_dollar = var_pct * portfolio_value
es_dollar = es_pct * portfolio_value

# ============================================================================
# MAIN DASHBOARD - TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Risk Overview",
    "📊 Portfolio Analytics",
    "🔥 Stress Testing",
    "🔗 Correlation Analysis",
    "📚 Risk Methodology"
])

# ========== TAB 1: RISK OVERVIEW ==========
with tab1:
    section_title("🎯 Key Risk Metrics")
    st.caption(f"Portfolio: ${portfolio_value:,.0f} | Confidence: {confidence_level*100:.0f}% | Method: {var_method.title()}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Value at Risk", f"${abs(var_dollar):,.0f}", f"{abs(var_pct)*100:.2f}% of portfolio")
    with col2:
        metric_card("Expected Shortfall", f"${abs(es_dollar):,.0f}", f"{abs(es_pct)*100:.2f}% of portfolio")
    with col3:
        metric_card("Volatility", f"{metrics['volatility']*100:.2f}%", "Annualized")
    with col4:
        metric_card("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", "Risk-adjusted")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
    with col2:
        metric_card("Skewness", f"{metrics['skewness']:.3f}")
    with col3:
        metric_card("Kurtosis", f"{metrics['kurtosis']:.3f}")
    
    st.markdown("---")
    section_title("📊 Risk Interpretation")
    
    col1, col2 = st.columns(2)
    with col1:
        info_box(f"""
            <strong>Value at Risk - {confidence_level*100:.0f}% Confidence</strong>
            <p style="margin-top:0.5rem;">
            With {confidence_level*100:.0f}% confidence, portfolio losses will not exceed 
            <strong style="color:{COLORS['accent_gold']};">${abs(var_dollar):,.0f}</strong> 
            over {time_horizon} day(s).
            </p>
        """, title="VaR Explanation")
    
    with col2:
        info_box(f"""
            <strong>Expected Shortfall</strong>
            <p style="margin-top:0.5rem;">
            If losses exceed VaR, average loss is 
            <strong style="color:{COLORS['danger']};">${abs(es_dollar):,.0f}</strong>.
            </p>
        """, title="ES Explanation")
    
    st.markdown("---")
    section_title("📈 Portfolio Return Distribution")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1 = axes[0]
    ax1.hist(portfolio_returns * 100, bins=50, density=True, color=COLORS['light_blue'], alpha=0.7, edgecolor='white')
    ax1.axvline(var_pct * 100, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'VaR ({confidence_level*100:.0f}%)')
    ax1.axvline(es_pct * 100, color='darkred', linestyle=':', linewidth=2, label=f'ES ({confidence_level*100:.0f}%)')
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    stats.probplot(portfolio_returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    section_title("📊 Performance Over Time")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    axes[0].plot(cumulative_returns.index, cumulative_returns * 100, color=COLORS['accent_gold'], linewidth=2)
    axes[0].set_ylabel('Cumulative Return (%)')
    axes[0].set_title('Cumulative Returns')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    axes[1].plot(portfolio_returns.index, portfolio_returns * 100, color=COLORS['medium_blue'], alpha=0.7, linewidth=0.8)
    axes[1].axhline(var_pct * 100, color=COLORS['danger'], linestyle='--', linewidth=1.5, label=f'VaR', alpha=0.7)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Daily Return (%)')
    axes[1].set_title('Daily Returns')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========== TAB 2: PORTFOLIO ANALYTICS ==========
with tab2:
    section_title("📊 Individual Asset Performance")
    
    asset_metrics = []
    for i, asset in enumerate(asset_names):
        asset_returns = returns_df[asset]
        asset_metrics.append({
            'Asset': asset,
            'Weight': f'{weights[i]*100:.1f}%',
            'Ann. Return': f'{asset_returns.mean()*252*100:.2f}%',
            'Ann. Volatility': f'{asset_returns.std()*np.sqrt(252)*100:.2f}%',
            'Sharpe': f'{(asset_returns.mean()*252)/(asset_returns.std()*np.sqrt(252)):.2f}',
            'VaR (95%)': f'{np.percentile(asset_returns, 5)*100:.2f}%',
        })
    
    st.dataframe(pd.DataFrame(asset_metrics), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    section_title("📈 Asset Prices")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for asset in asset_names:
        ax.plot(prices_df.index, prices_df[asset], label=asset, linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (Base=100)')
    ax.set_title('Asset Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========== TAB 3: STRESS TESTING ==========
with tab3:
    section_title("🔥 Stress Test Scenarios")
    
    scenarios = {
        'Market Crash (-20%)': -0.20,
        'Correction (-10%)': -0.10,
        'Vol Spike': -0.15,
        'Rate Shock': -0.05,
        'Credit Spread': -0.08,
        'Black Swan (-30%)': -0.30,
    }
    
    stress_results = []
    for name, shock in scenarios.items():
        shocked_value = portfolio_value * (1 + shock)
        impact = shocked_value - portfolio_value
        stress_results.append({
            'Scenario': name,
            'Shock': f'{shock*100:.1f}%',
            'Value': f'${shocked_value:,.0f}',
            'P&L': f'${impact:,.0f}',
            'vs VaR': f'{abs(impact/var_dollar):.1f}x'
        })
    
    st.dataframe(pd.DataFrame(stress_results), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(12, 6))
    impacts = [float(s['P&L'].replace('$', '').replace(',', '')) for s in stress_results]
    colors_bars = [COLORS['danger'] if x < 0 else COLORS['success'] for x in impacts]
    ax.barh([s['Scenario'] for s in stress_results], impacts, color=colors_bars)
    ax.axvline(var_dollar, color=COLORS['accent_gold'], linestyle='--', linewidth=2, label='VaR')
    ax.set_xlabel('P&L Impact ($)')
    ax.set_title('Stress Test Impacts')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========== TAB 4: CORRELATION ==========
with tab4:
    section_title("🔗 Correlation Matrix")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=asset_names, yticklabels=asset_names,
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Asset Correlations')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ========== TAB 5: METHODOLOGY ==========
with tab5:
    section_title("📚 VaR Methodology")
    
    info_box("""
        <strong>Three VaR Methods:</strong>
        <ul>
            <li><strong>Historical:</strong> Uses actual past returns (non-parametric)</li>
            <li><strong>Parametric:</strong> Assumes normal distribution</li>
            <li><strong>Monte Carlo:</strong> Simulates 10,000 scenarios</li>
        </ul>
    """)
    
    st.markdown("---")
    st.subheader("Expected Shortfall (ES)")
    info_box("""
        <p>ES measures average loss beyond VaR threshold.</p>
        <p style="margin-top:0.5rem;"><strong>Formula:</strong> ES = E[Loss | Loss > VaR]</p>
        <p style="margin-top:0.5rem;">ES is preferred by Basel III as it's a coherent risk measure.</p>
    """)

footer()
