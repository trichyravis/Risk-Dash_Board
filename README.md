# Portfolio Risk Dashboard
**The Mountain Path - World of Finance**

Prof. V. Ravichandran  
*28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence*

---

## 🎯 Overview

A comprehensive, interactive risk management dashboard for portfolio analysis featuring:
- **Value at Risk (VaR)** - Historical, Parametric, and Monte Carlo methods
- **Expected Shortfall (ES)** - Coherent tail risk measurement
- **Stress Testing** - Multiple crisis scenarios
- **Portfolio Analytics** - Performance metrics and attribution
- **Correlation Analysis** - Asset relationship visualization

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run risk_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Features

### 1. Risk Overview Tab
- **Key Risk Metrics**: VaR, ES, Volatility, Sharpe Ratio
- **Distribution Analysis**: Return histograms and Q-Q plots
- **Time Series**: Cumulative returns and daily performance
- **Risk Interpretation**: Clear explanations of metrics

### 2. Portfolio Analytics Tab
- **Individual Asset Metrics**: Returns, volatility, Sharpe ratios
- **Price Performance**: Multi-asset time series charts
- **Portfolio Composition**: Pie chart of weights
- **Risk Contribution**: Marginal VaR analysis

### 3. Stress Testing Tab
- **Pre-defined Scenarios**:
  - Market Crash (-20%)
  - Moderate Correction (-10%)
  - Volatility Spike (+50%)
  - Interest Rate Shock (+200 bps)
  - Credit Spread Widening
  - Black Swan Event (-30%)
- **Impact Visualization**: P&L impacts vs VaR
- **Worst-Case Analysis**: Risk mitigation recommendations

### 4. Correlation Analysis Tab
- **Correlation Heatmap**: Visual correlation matrix
- **Statistics**: Average, max, min correlations
- **Rolling Correlation**: Time-varying relationships

### 5. Risk Methodology Tab
- **VaR Methods**: Detailed explanations of all three methods
- **Expected Shortfall**: Theory and formulas
- **Stress Testing Framework**: Regulatory context

## ⚙️ Configuration Options

### Sidebar Controls

**Portfolio Settings:**
- Number of Assets: 3-10
- Portfolio Value: $100K - $100M

**Risk Calculation:**
- Confidence Level: 90% - 99%
- Time Horizon: 1, 5, 10, or 21 days
- VaR Method: Historical, Parametric, or Monte Carlo

**Data Period:**
- Lookback Period: 63-756 days

## 📈 Risk Metrics Explained

### Value at Risk (VaR)
With X% confidence, portfolio losses will not exceed VaR over the specified time horizon.

**Methods:**
1. **Historical**: Uses actual past returns
2. **Parametric**: Assumes normal distribution
3. **Monte Carlo**: Simulates thousands of scenarios

### Expected Shortfall (ES)
Average loss given that the loss exceeds VaR. A coherent risk measure preferred by Basel III.

**Formula:** ES = E[Loss | Loss > VaR]

### Sharpe Ratio
Risk-adjusted return metric: (Return - Risk-free rate) / Volatility

### Maximum Drawdown
Largest peak-to-trough decline in portfolio value.

## 🎓 Educational Use Cases

### For Students
- Learn different VaR calculation methods
- Understand tail risk and extreme events
- Practice portfolio risk analysis
- Explore correlation effects on diversification

### For Professionals
- Daily risk monitoring
- Regulatory compliance (Basel III)
- Scenario analysis and stress testing
- Risk reporting and communication

### For Researchers
- Compare VaR methodologies
- Analyze correlation dynamics
- Study stress test frameworks
- Validate risk models

## 📊 Technical Details

### Data Generation
- Uses synthetic data with realistic correlations
- Correlated multivariate normal returns
- Adjustable number of assets and time periods

### Risk Calculations
- **Historical VaR**: Empirical percentile
- **Parametric VaR**: Normal distribution assumption
- **Monte Carlo VaR**: 10,000 random scenarios
- **ES**: Conditional expectation beyond VaR

### Stress Testing
- Multiple pre-defined scenarios
- Impact measured in both percentage and dollar terms
- Comparison with VaR and ES thresholds

## 🎨 Design

Built using **The Mountain Path Design Template**:
- Professional branding and styling
- Consistent color scheme (Dark Blue, Gold accents)
- Reusable components for rapid development
- Mobile-responsive layout

## 📚 References

### Academic
- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk*
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*

### Regulatory
- Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk*
- Federal Reserve CCAR/DFAST stress testing guidelines

### Industry Standards
- RiskMetrics™ methodology (J.P. Morgan)
- GARP FRM curriculum

## 🔧 Customization

### Adding Real Data
Replace the `generate_portfolio_data()` function with:

```python
import yfinance as yf

def load_real_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    prices = data['Adj Close']
    returns = prices.pct_change().dropna()
    return returns, prices
```

### Custom Stress Scenarios
Edit the `stress_test_scenarios()` function:

```python
scenarios = {
    'Your Scenario': -0.15,  # -15% shock
    'Custom Event': 'custom_logic',
}
```

### Portfolio Weights
Modify the equal-weight assumption:

```python
# Custom weights
weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
```

## 📞 Support

**Instructor:** Prof. V. Ravichandran  
**Email:** [Your email]  
**LinkedIn:** [linkedin.com/in/trichyravis](https://www.linkedin.com/in/trichyravis)  
**GitHub:** [github.com/trichyravis](https://github.com/trichyravis)

## 📄 License

Educational use for The Mountain Path - World of Finance.  
© Prof. V. Ravichandran

---

**Version:** 1.0  
**Last Updated:** February 2026  
**Built with:** The Mountain Path Streamlit Design Template
