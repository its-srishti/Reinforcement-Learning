# Reinforcement-Learning

*"I have not failed. I've just found 10,000 ways that won't work"* - Thomas A. Edison

An advanced quantitative finance project that applies deep reinforcement learning to optimize retirement spending strategies, combining traditional financial modeling with adaptive AI-driven portfolio management to minimize the risk of retirees outliving their savings.

## Project Overview

This project challenges the traditional "4% rule" by developing dynamic retirement strategies that adapt to changing market conditions. Using nearly 100 years of historical market data (1926-2024) and Deep Q-Networks (DQN), the system learns optimal asset allocation and withdrawal policies that outperform static strategies.

### Key Innovation

**Traditional Approach**: Fixed 4% withdrawal rate with static 60/40 portfolio allocation
**Our Approach**: AI-driven dynamic allocation that adapts to market conditions, wealth levels, and remaining time horizon

## Core Features

**Historical Data Analysis**:
- SBBI (Stocks, Bonds, Bills, Inflation) dataset from 1926-2024
- 817 overlapping 30-year retirement scenarios
- Comprehensive backtesting across multiple market cycles

**Advanced Modeling**:
- Custom Gymnasium environment for retirement simulation
- Deep Q-Network implementation using Stable Baselines3
- Dynamic asset allocation based on current market conditions

**Risk Assessment**:
- Probability of shortfall analysis
- Expected shortfall duration quantification
- Tail risk analysis (95th percentile scenarios)

**Strategy Comparison**:
- Buy-and-Hold vs Annual Rebalancing
- Static rules vs Reinforcement Learning
- Comprehensive performance metrics

## Installation & Setup

### Requirements

```bash
pip install pandas numpy matplotlib seaborn gymnasium stable-baselines3 torch tqdm
```

### Data Dependencies

The project requires:
- **SBBI Data**: Historical returns for stocks, bonds, and inflation
- **Database Access**: Custom financial database connections
- **Custom Libraries**: 
  - `finds.database.SQL`: Database interface
  - `finds.structured.BusDay`: Business day calculations
  - `finds.utils.Store`: Data persistence utilities

### Environment Setup

Create a `secret.py` file with your database credentials:
```python
credentials = {
    'sql': {
        'server': 'your_server',
        'database': 'your_database',
        # additional connection parameters
    }
}

paths = {
    'data': '/path/to/data',
    'scratch': '/path/to/output'
}
```

## Methodology

### Historical Simulation Framework

**Episode Generation**: 
- 817 unique 30-year retirement periods from historical data
- Rolling window approach capturing different market regimes
- Monthly rebalancing with annual withdrawal adjustments

**Risk Metrics**:
- **Probability of Shortfall**: Percentage of scenarios where funds are depleted
- **Expected Shortfall Period**: Average years without funds in worst-case scenarios
- **Tail Risk Analysis**: Focus on 95th percentile outcomes

### Reinforcement Learning Implementation

**State Space**:
- Current wealth and asset allocation
- Recent market returns (stocks, bonds, inflation)
- Years remaining until end of retirement
- Current spending requirement

**Action Space**:
- Discrete allocation decisions (0% to 100% equity in 5% increments)
- Dynamic rebalancing based on market conditions

**Reward Function**:
```python
# Severe penalty for fund depletion
reward = -(T² × 100) if funds_depleted

# Reward proportional to wealth coverage ratio
reward = √(wealth / (spending × years_remaining)) if funds_available
```

**Training Process**:
- 500,000 timesteps per spending policy (3.0% to 5.0%)
- Deep Q-Network with experience replay
- Epsilon-greedy exploration strategy

## Key Results

### Traditional Strategy Performance

**4% Rule with 50/50 Allocation**:
- Probability of shortfall: 7.47%
- Historical data spans: 1926-2024 (817 scenarios)

**Buy-and-Hold vs Annual Rebalancing**:
- Buy-and-hold outperformed: 37.4% of scenarios
- Annual rebalancing outperformed: 29.7% of scenarios
- Performance varies significantly by market regime

### Reinforcement Learning Outcomes

**Dramatic Risk Reduction**:
- 3.0%-3.3% spending: 0% probability of shortfall
- 4.0% spending: 3% probability of shortfall (vs 7.47% traditional)
- 5.0% spending: 19% probability of shortfall (vs much higher for static strategies)

**Adaptive Allocation Patterns**:
- Early retirement: Higher equity allocation (60-80%)
- Later years: Gradual shift to more conservative allocations
- Dynamic response to market conditions and remaining wealth

### Performance Comparison Summary

| Strategy | 4% Withdrawal | Probability of Shortfall |
|----------|---------------|-------------------------|
| Traditional 4% Rule | 4.0% | 7.47% |
| Deep RL | 4.0% | 3.00% |
| **Improvement** | **Same** | **60% reduction** |

## Technical Architecture

### Environment Design

```python
class CustomEnv(gym.Env):
    def __init__(self, model: BaseModel, episodes: Episodes):
        # State: [years_remaining, market_returns, wealth_allocation]
        # Action: Discrete equity allocation (0-100% in 5% steps)
        # Reward: Wealth sustainability + depletion penalty
```

### Model Training

```python
# Deep Q-Network training
clf = DQN('MlpPolicy', env, verbose=0)
clf.learn(total_timesteps=500000)
```

### Strategy Evaluation

```python
def evaluate(env, model, episodes):
    # Returns: shortfall probability, expected shortfall, allocations
    # Tested across all historical 30-year periods
```

## Usage Examples

### Basic Strategy Comparison

```python
# Compare buy-and-hold vs rebalancing
base_model = BaseModel(T=30, W=[50, 50, 4.0])  # 50/50, 4% spending
fixed_model = FixedModel(T=30, W=[50, 50, 4.0])  # Annual rebalancing

# Evaluate across historical scenarios
episodes = Episodes(data=sbbi_data, T=30)
```

### Reinforcement Learning Training

```python
# Train RL model for specific spending policy
env = CustomEnv(model=BaseModel(T=30, W=[50, 50, 4.0]), episodes=episodes)
clf = DQN('MlpPolicy', env)
clf.learn(total_timesteps=500000)
```

### Risk Analysis

```python
# Analyze shortfall probabilities across allocation/spending combinations
allocs = range(0, 105, 5)  # 0% to 100% equity
rules = np.arange(3.0, 5.1, 0.1)  # 3% to 5% spending

# Generate probability heatmaps and contour plots
```

## Visualization Outputs

**Risk Heatmaps**: Probability of shortfall across allocation/spending combinations
**Contour Plots**: Iso-risk curves for portfolio planning
**Time Series**: Adaptive allocation patterns over retirement horizon
**Performance Metrics**: Comparative analysis across strategies

## Research Applications

**Financial Planning**:
- Personalized retirement withdrawal strategies
- Dynamic asset allocation recommendations
- Risk tolerance-based portfolio management

**Academic Research**:
- Behavioral finance and adaptive strategies
- Machine learning applications in portfolio theory
- Longevity risk modeling and mitigation

**Industry Applications**:
- Robo-advisor enhancement with RL
- Dynamic glide path optimization
- Risk management for retirement products

## Key Insights

**Market Timing Value**: RL strategies demonstrate the value of adaptive allocation during market downturns and recoveries

**Longevity Risk Mitigation**: Dynamic strategies significantly reduce the probability of outliving savings

**Historical Robustness**: Performance improvement consistent across different market regimes (1926-2024)

**Practical Implementation**: Discrete action space makes strategies implementable in real portfolio management

## Data Sources & Validation

**Historical Data**: SBBI dataset provides nearly 100 years of monthly returns
**Robustness Testing**: 817 overlapping 30-year scenarios ensure comprehensive evaluation
**Out-of-Sample Validation**: Rolling window approach prevents look-ahead bias

## Future Enhancements

**Model Extensions**:
- Continuous action spaces for finer allocation control
- Multi-asset class portfolios (REITs, international, commodities)
- Stochastic mortality modeling

**Advanced RL Algorithms**:
- Actor-Critic methods (A2C, PPO)
- Distributional reinforcement learning
- Multi-agent systems for couples

**Real-World Integration**:
- Transaction cost modeling
- Tax-aware optimization
- Social Security integration

## License

This project is available under the MIT License.

## References

**Academic Sources**:
- Benz, C., Ptak, J., & Rekenthaler, J. (2022). "The State of Retirement Income"
- Traditional 4% rule literature and safe withdrawal rate research

**Technical Libraries**:
- Stable Baselines3: Deep reinforcement learning implementations
- Gymnasium: RL environment framework
- SBBI Data: Morningstar/Ibbotson historical market data

**Market Data**:
- Stocks, Bonds, Bills, and Inflation (SBBI) dataset
- Federal Reserve Economic Data (FRED)

---

*Combining financial theory, historical data, and artificial intelligence for smarter retirement planning*
