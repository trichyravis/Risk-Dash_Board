"""
Portfolio Risk Dashboard
The Mountain Path - World of Finance

Prof. V. Ravichandran
28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence

Comprehensive risk management dashboard featuring:
- Value at Risk (VaR) & Expected Shortfall (ES)
- Stress Testing & Scenario Analysis
- Portfolio Analytics & Correlation Analysis
- Real-time Risk Monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats

from config import COLORS, BRANDING, get_page_config
from styles import apply_styles
from components import (
    header_container, sidebar_header, section_title, sidebar_section,
    metric_card, metric_card_advanced, info_box, formula_box,
    warning_box, footer, three_metric_row
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(**get_page_config(
    title="Portfolio Risk Dashboard",
    icon="📊"
))
apply_styles()

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
        # ES for normal distribution
        es = mu - sigma * stats.norm.pdf(stats.norm.ppf(1 - confidence)) / (1 - confidence)
    else:  # monte carlo
        simulated = np.random.normal(returns.mean(), returns.std(), 10000)
        var = np.percentile(simulated, (1 - confidence) * 100)
        es = simulated[simulated <= var].mean()
    
    return var, es


def generate_portfolio_data(n_assets=5, n_days=252):
    """Generate synthetic portfolio data"""
    np.random.seed(42)
    
    # Asset names
    assets = [f'Asset {i+1}' for i in range(n_assets)]
    
    # Generate correlated returns
    correlation = np.random.uniform(0.3, 0.7, size=(n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    
    # Generate returns with correlation
    mean_returns = np.random.uniform(-0.0005, 0.002, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)
    
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    # Create DataFrame
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    returns_df = pd.DataFrame(returns, columns=assets, index=dates)
    
    # Calculate prices from returns
    prices_df = (1 + returns_df).cumprod() * 100
    
    return returns_df, prices_df, correlation, assets


def stress_test_scenarios():
    """Define stress test scenarios"""
    scenarios = {
        'Market Crash (-20%)': -0.20,
        'Moderate Correction (-10%)': -0.10,
        'Volatility Spike (+50% vol)': 'vol_spike',
        'Interest Rate Shock (+200 bps)': 'rate_shock',
        'Credit Spread Widening': 'credit_shock',
        'Black Swan Event (-30%)': -0.30,
    }
    return scenarios


def calculate_portfolio_metrics(returns_df, weights):
    """Calculate portfolio-level metrics"""
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    metrics = {
        'mean_return': portfolio_returns.mean() * 252,  # Annualized
        'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
        'max_drawdown': (portfolio_returns.cumsum().cummax() - portfolio_returns.cumsum()).max(),
        'skewness': stats.skew(portfolio_returns),
        'kurtosis': stats.kurtosis(portfolio_returns),
    }
    
    return portfolio_returns, metrics


# ============================================================================
# GENERATE DATA
# ============================================================================

@st.cache_data
def load_portfolio_data(n_assets, n_days):
    """Load and cache portfolio data"""
    return generate_portfolio_data(n_assets, n_days)


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
sidebar_header("RISK PARAMETERS", "Configure your analysis")

sidebar_section("📊 Portfolio Settings")
n_assets = st.sidebar.slider("Number of Assets", 3, 10, 5)
portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)", 
    min_value=100000, 
    max_value=100000000, 
    value=10000000, 
    step=100000,
    format="%d"
)

sidebar_section("⚙️ Risk Calculation")
confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
time_horizon = st.sidebar.selectbox("Time Horizon", [1, 5, 10, 21], index=0)
var_method = st.sidebar.selectbox(
    "VaR Method",
    ["historical", "parametric", "monte_carlo"]
)

sidebar_section("📅 Data Period")
lookback_days = st.sidebar.slider("Lookback Period (days)", 63, 756, 252)

# ============================================================================
# LOAD DATA
# ============================================================================
with st.spinner('Generating portfolio data...'):
    returns_df, prices_df, correlation_matrix, asset_names = load_portfolio_data(n_assets, lookback_days)

# Equal weights for simplicity (can be customized)
weights = np.ones(n_assets) / n_assets

# Calculate portfolio metrics
portfolio_returns, metrics = calculate_portfolio_metrics(returns_df, weights)

# Calculate VaR and ES
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
    
    st.caption(f"Portfolio Value: ${portfolio_value:,.0f} | Confidence: {confidence_level*100:.0f}% | Method: {var_method.title()}")
    
    # Main risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card(
            "Value at Risk (VaR)",
            f"${abs(var_dollar):,.0f}",
            f"{abs(var_pct)*100:.2f}% of portfolio"
        )
    
    with col2:
        metric_card(
            "Expected Shortfall",
            f"${abs(es_dollar):,.0f}",
            f"{abs(es_pct)*100:.2f}% of portfolio"
        )
    
    with col3:
        metric_card(
            "Portfolio Volatility",
            f"{metrics['volatility']*100:.2f}%",
            "Annualized"
        )
    
    with col4:
        metric_card(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            "Risk-adjusted return"
        )
    
    # Additional metrics
    st.markdown("---")
    three_metric_row([
        ("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%", "Peak to trough decline"),
        ("Skewness", f"{metrics['skewness']:.3f}", "Return distribution asymmetry"),
        ("Kurtosis", f"{metrics['kurtosis']:.3f}", "Tail thickness")
    ])
    
    # VaR interpretation
    st.markdown("---")
    section_title("📊 Risk Interpretation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        info_box(f"""
            <strong>Value at Risk (VaR) - {confidence_level*100:.0f}% Confidence</strong>
            <p style="margin-top:0.5rem;">
            With {confidence_level*100:.0f}% confidence, your portfolio losses will not exceed 
            <strong style="color:{COLORS['accent_gold']};">${abs(var_dollar):,.0f}</strong> 
            over the next {time_horizon} day(s).
            </p>
            <p style="margin-top:0.5rem; font-size:0.85rem; color:{COLORS['text_secondary']};">
            This means there is a {(1-confidence_level)*100:.1f}% chance that losses could exceed this amount.
            </p>
        """, title="VaR Explanation")
    
    with col2:
        info_box(f"""
            <strong>Expected Shortfall (ES / CVaR)</strong>
            <p style="margin-top:0.5rem;">
            If losses exceed VaR, the average loss is expected to be 
            <strong style="color:{COLORS['danger']};">${abs(es_dollar):,.0f}</strong>.
            </p>
            <p style="margin-top:0.5rem; font-size:0.85rem; color:{COLORS['text_secondary']};">
            ES provides insight into tail risk beyond the VaR threshold and is a coherent risk measure 
            preferred by Basel III regulations.
            </p>
        """, title="ES Explanation")
    
    # Portfolio return distribution
    st.markdown("---")
    section_title("📈 Portfolio Return Distribution")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram with VaR/ES
    ax1 = axes[0]
    ax1.hist(portfolio_returns * 100, bins=50, density=True, 
             color=COLORS['light_blue'], alpha=0.7, edgecolor='white')
    ax1.axvline(var_pct * 100, color=COLORS['danger'], linestyle='--', 
                linewidth=2, label=f'VaR ({confidence_level*100:.0f}%)')
    ax1.axvline(es_pct * 100, color='darkred', linestyle=':', 
                linewidth=2, label=f'ES ({confidence_level*100:.0f}%)')
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Portfolio Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax2 = axes[1]
    stats.probplot(portfolio_returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Time series
    st.markdown("---")
    section_title("📊 Portfolio Performance Over Time")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Cumulative returns
    ax1 = axes[0]
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    ax1.plot(cumulative_returns.index, cumulative_returns * 100, 
             color=COLORS['accent_gold'], linewidth=2)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title('Portfolio Cumulative Returns')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    # Daily returns
    ax2 = axes[1]
    ax2.plot(portfolio_returns.index, portfolio_returns * 100, 
             color=COLORS['medium_blue'], alpha=0.7, linewidth=0.8)
    ax2.axhline(var_pct * 100, color=COLORS['danger'], linestyle='--', 
                linewidth=1.5, label=f'VaR ({confidence_level*100:.0f}%)', alpha=0.7)
    ax2.axhline(es_pct * 100, color='darkred', linestyle=':', 
                linewidth=1.5, label=f'ES ({confidence_level*100:.0f}%)', alpha=0.7)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Return (%)')
    ax2.set_title('Portfolio Daily Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ========== TAB 2: PORTFOLIO ANALYTICS ==========
with tab2:
    section_title("📊 Individual Asset Performance")
    
    # Asset metrics table
    asset_metrics = []
    for i, asset in enumerate(asset_names):
        asset_returns = returns_df[asset]
        asset_metrics.append({
            'Asset': asset,
            'Weight': f'{weights[i]*100:.1f}%',
            'Ann. Return': f'{asset_returns.mean()*252*100:.2f}%',
            'Ann. Volatility': f'{asset_returns.std()*np.sqrt(252)*100:.2f}%',
            'Sharpe Ratio': f'{(asset_returns.mean()*252)/(asset_returns.std()*np.sqrt(252)):.2f}',
            'VaR (95%)': f'{np.percentile(asset_returns, 5)*100:.2f}%',
        })
    
    metrics_df = pd.DataFrame(asset_metrics)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Asset price performance
    st.markdown("---")
    section_title("📈 Asset Price Performance")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for asset in asset_names:
        ax.plot(prices_df.index, prices_df[asset], label=asset, linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (Normalized to 100)')
    ax.set_title('Individual Asset Performance')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Portfolio composition
    st.markdown("---")
    section_title("🥧 Portfolio Composition")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        colors_list = [COLORS['dark_blue'], COLORS['medium_blue'], COLORS['light_blue'], 
                      COLORS['accent_gold'], COLORS['text_secondary']]
        colors_extended = colors_list * (len(asset_names) // len(colors_list) + 1)
        ax.pie(weights, labels=asset_names, autopct='%1.1f%%', startangle=90,
               colors=colors_extended[:len(asset_names)])
        ax.set_title('Portfolio Weights')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Contribution to portfolio risk
        portfolio_var = portfolio_returns.var()
        marginal_var = []
        for i in range(n_assets):
            weight_copy = weights.copy()
            weight_copy[i] += 0.01
            weight_copy = weight_copy / weight_copy.sum()
            perturbed_returns = (returns_df * weight_copy).sum(axis=1)
            marginal_var.append((perturbed_returns.var() - portfolio_var) / 0.01)
        
        contribution_to_risk = np.array(marginal_var) * weights
        contribution_pct = contribution_to_risk / contribution_to_risk.sum() * 100
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(asset_names, contribution_pct, color=COLORS['accent_gold'])
        ax.set_xlabel('Contribution to Portfolio Risk (%)')
        ax.set_title('Risk Contribution by Asset')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ========== TAB 3: STRESS TESTING ==========
with tab3:
    section_title("🔥 Stress Test Scenarios")
    
    info_box("""
        <strong>Purpose:</strong> Stress testing evaluates portfolio resilience under extreme market conditions.
        These scenarios represent plausible but severe events that could impact your portfolio.
    """)
    
    scenarios = stress_test_scenarios()
    
    # Calculate stress impacts
    stress_results = []
    for scenario_name, shock in scenarios.items():
        if isinstance(shock, float):
            # Simple return shock
            shocked_return = shock
            shocked_value = portfolio_value * (1 + shocked_return)
            impact = shocked_value - portfolio_value
        else:
            # Other scenarios - simplified
            if shock == 'vol_spike':
                shocked_return = -metrics['volatility'] * 1.5
            elif shock == 'rate_shock':
                shocked_return = -0.05  # Approximate duration impact
            else:  # credit_shock
                shocked_return = -0.08
            
            shocked_value = portfolio_value * (1 + shocked_return)
            impact = shocked_value - portfolio_value
        
        stress_results.append({
            'Scenario': scenario_name,
            'Shock': f'{shocked_return*100:.1f}%',
            'Portfolio Value': f'${shocked_value:,.0f}',
            'P&L Impact': f'${impact:,.0f}',
            'Impact vs VaR': f'{abs(impact/var_dollar):.1f}x VaR'
        })
    
    stress_df = pd.DataFrame(stress_results)
    st.dataframe(stress_df, use_container_width=True, hide_index=True)
    
    # Stress test visualization
    st.markdown("---")
    section_title("📊 Stress Test Impact Visualization")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenario_names = [s['Scenario'] for s in stress_results]
    impacts = [float(s['P&L Impact'].replace('$', '').replace(',', '')) for s in stress_results]
    colors_bars = [COLORS['danger'] if x < 0 else COLORS['success'] for x in impacts]
    
    bars = ax.barh(scenario_names, impacts, color=colors_bars)
    ax.axvline(var_dollar, color=COLORS['accent_gold'], linestyle='--', 
               linewidth=2, label=f'VaR ({confidence_level*100:.0f}%)')
    ax.axvline(es_dollar, color='orange', linestyle=':', 
               linewidth=2, label=f'ES ({confidence_level*100:.0f}%)')
    ax.set_xlabel('P&L Impact ($)')
    ax.set_title('Stress Test Scenario Impacts')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Format x-axis as currency
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Worst case analysis
    st.markdown("---")
    section_title("⚠️ Risk Assessment")
    
    worst_scenario = stress_results[np.argmin([float(s['P&L Impact'].replace('$', '').replace(',', '')) 
                                                for s in stress_results])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        warning_box(f"""
            <strong>Worst Case Scenario: {worst_scenario['Scenario']}</strong>
            <p style="margin-top:0.5rem;">
            Portfolio Value: {worst_scenario['Portfolio Value']}<br>
            Loss: {worst_scenario['P&L Impact']}<br>
            Impact: {worst_scenario['Impact vs VaR']}
            </p>
        """)
    
    with col2:
        info_box(f"""
            <strong>Risk Mitigation Recommendations:</strong>
            <ul>
                <li>Consider hedging strategies for tail risk</li>
                <li>Diversify across asset classes and geographies</li>
                <li>Maintain adequate capital buffers</li>
                <li>Regular stress testing and scenario updates</li>
                <li>Monitor early warning indicators</li>
            </ul>
        """, title="Recommendations")


# ========== TAB 4: CORRELATION ANALYSIS ==========
with tab4:
    section_title("🔗 Asset Correlation Matrix")
    
    info_box("""
        <strong>Correlation measures how assets move together:</strong>
        <ul>
            <li><strong>+1.0:</strong> Perfect positive correlation (move together)</li>
            <li><strong>0.0:</strong> No correlation (independent movements)</li>
            <li><strong>-1.0:</strong> Perfect negative correlation (move oppositely)</li>
        </ul>
        <p style="margin-top:0.5rem;">
        Lower correlations generally provide better diversification benefits.
        </p>
    """)
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use custom colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap, 
                xticklabels=asset_names, yticklabels=asset_names,
                center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title('Asset Correlation Heatmap', fontsize=14, pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Correlation statistics
    st.markdown("---")
    section_title("📊 Correlation Statistics")
    
    # Flatten correlation matrix (excluding diagonal)
    corr_values = []
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            corr_values.append(correlation_matrix[i][j])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_card("Average Correlation", f"{np.mean(corr_values):.3f}")
    with col2:
        metric_card("Max Correlation", f"{np.max(corr_values):.3f}")
    with col3:
        metric_card("Min Correlation", f"{np.min(corr_values):.3f}")
    
    # Rolling correlation (for first two assets)
    st.markdown("---")
    section_title("📈 Rolling Correlation Over Time")
    
    if n_assets >= 2:
        rolling_window = 21  # 21-day rolling window
        rolling_corr = returns_df[asset_names[0]].rolling(rolling_window).corr(returns_df[asset_names[1]])
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rolling_corr.index, rolling_corr, color=COLORS['accent_gold'], linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Rolling {rolling_window}-Day Correlation: {asset_names[0]} vs {asset_names[1]}')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ========== TAB 5: METHODOLOGY ==========
with tab5:
    section_title("📚 Risk Measurement Methodology")
    
    # VaR Methods
    st.subheader("Value at Risk (VaR) Calculation Methods")
    
    method_tab1, method_tab2, method_tab3 = st.tabs([
        "Historical Simulation",
        "Parametric (Variance-Covariance)",
        "Monte Carlo Simulation"
    ])
    
    with method_tab1:
        info_box("""
            <strong>Historical Simulation Method</strong>
            <p>Uses actual historical returns to estimate future risk.</p>
            <p style="margin-top:0.5rem;"><strong>Steps:</strong></p>
            <ol>
                <li>Collect historical return data</li>
                <li>Sort returns from worst to best</li>
                <li>Find the return at the (1-α) percentile</li>
            </ol>
            <p style="margin-top:0.5rem;"><strong>Advantages:</strong></p>
            <ul>
                <li>No distributional assumptions</li>
                <li>Captures actual tail behavior</li>
                <li>Easy to understand and implement</li>
            </ul>
            <p style="margin-top:0.5rem;"><strong>Limitations:</strong></p>
            <ul>
                <li>Limited by historical data</li>
                <li>May not capture new risks</li>
                <li>Requires sufficient data points</li>
            </ul>
        """)
        
        formula_box("""
VaR_α = Percentile(Returns, (1-α) × 100)

Example for 95% confidence:
VaR_0.95 = 5th percentile of return distribution
        """)
    
    with method_tab2:
        info_box("""
            <strong>Parametric (Variance-Covariance) Method</strong>
            <p>Assumes returns follow a normal distribution.</p>
            <p style="margin-top:0.5rem;"><strong>Steps:</strong></p>
            <ol>
                <li>Calculate mean (μ) and standard deviation (σ)</li>
                <li>Use z-score for desired confidence level</li>
                <li>VaR = μ + z_α × σ</li>
            </ol>
            <p style="margin-top:0.5rem;"><strong>Advantages:</strong></p>
            <ul>
                <li>Fast and computationally efficient</li>
                <li>Works well for normally distributed returns</li>
                <li>Easy to scale across portfolios</li>
            </ul>
            <p style="margin-top:0.5rem;"><strong>Limitations:</strong></p>
            <ul>
                <li>Assumes normal distribution (may underestimate tail risk)</li>
                <li>Doesn't capture fat tails or skewness</li>
                <li>Less accurate for non-linear instruments</li>
            </ul>
        """)
        
        formula_box("""
VaR_α = μ + z_α × σ

where:
μ = mean return
σ = standard deviation of returns
z_α = z-score at confidence level α

For 95% confidence: z_0.95 = -1.645
For 99% confidence: z_0.99 = -2.326
        """)
    
    with method_tab3:
        info_box("""
            <strong>Monte Carlo Simulation</strong>
            <p>Generates thousands of random scenarios to estimate risk.</p>
            <p style="margin-top:0.5rem;"><strong>Steps:</strong></p>
            <ol>
                <li>Estimate return distribution parameters</li>
                <li>Generate 10,000+ random return scenarios</li>
                <li>Calculate portfolio value for each scenario</li>
                <li>Find VaR at desired percentile</li>
            </ol>
            <p style="margin-top:0.5rem;"><strong>Advantages:</strong></p>
            <ul>
                <li>Can model complex portfolios</li>
                <li>Flexible distribution assumptions</li>
                <li>Captures path-dependent effects</li>
                <li>Can incorporate correlations</li>
            </ul>
            <p style="margin-top:0.5rem;"><strong>Limitations:</strong></p>
            <ul>
                <li>Computationally intensive</li>
                <li>Results vary with random seed</li>
                <li>Requires many assumptions</li>
            </ul>
        """)
        
        formula_box("""
Monte Carlo VaR Process:

1. Generate N scenarios: R_i ~ N(μ, σ)  for i=1 to 10,000
2. Calculate portfolio values: V_i = V_0 × (1 + R_i)
3. VaR_α = Percentile({V_i}, (1-α) × 100)
        """)
    
    # Expected Shortfall
    st.markdown("---")
    st.subheader("Expected Shortfall (ES / CVaR)")
    
    info_box("""
        <strong>Expected Shortfall: Average Loss Beyond VaR</strong>
        <p style="margin-top:0.5rem;">
        ES measures the expected loss given that the loss exceeds VaR. 
        Unlike VaR, ES is a <strong>coherent risk measure</strong> that satisfies:
        </p>
        <ul>
            <li><strong>Monotonicity:</strong> Riskier positions have higher ES</li>
            <li><strong>Sub-additivity:</strong> ES(A+B) ≤ ES(A) + ES(B)</li>
            <li><strong>Positive homogeneity:</strong> ES(λX) = λ × ES(X)</li>
            <li><strong>Translation invariance:</strong> Adding cash reduces ES</li>
        </ul>
        <p style="margin-top:0.5rem;">
        <strong>Basel III</strong> recommends ES over VaR for market risk capital requirements.
        </p>
    """)
    
    formula_box("""
Expected Shortfall:

ES_α = E[Loss | Loss > VaR_α]

For normal distribution:
ES_α = μ + σ × φ(z_α) / (1-α)

where φ is the standard normal PDF

Example: For 95% VaR, ES represents the average of the 
worst 5% of outcomes
    """)
    
    # Stress Testing
    st.markdown("---")
    st.subheader("Stress Testing Framework")
    
    info_box("""
        <strong>Types of Stress Tests:</strong>
        <ol style="margin-top:0.5rem;">
            <li><strong>Sensitivity Analysis:</strong> Single factor moves (e.g., -20% equity shock)</li>
            <li><strong>Scenario Analysis:</strong> Multiple correlated factor moves</li>
            <li><strong>Historical Scenarios:</strong> Replay past crisis events</li>
            <li><strong>Reverse Stress Tests:</strong> Find scenarios that cause failure</li>
        </ol>
        <p style="margin-top:0.5rem;"><strong>Regulatory Requirements:</strong></p>
        <ul>
            <li>Basel III: Regular stress testing required</li>
            <li>CCAR/DFAST: U.S. bank stress testing</li>
            <li>EBA: European Banking Authority stress tests</li>
        </ul>
    """)

# ============================================================================
# FOOTER
# ============================================================================
footer()
