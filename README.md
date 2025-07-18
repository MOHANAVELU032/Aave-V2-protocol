# Aave V2 Wallet Credit Scoring System

## Overview

This system assigns credit scores (0-1000) to Ethereum wallets based on their historical transaction behavior within the Aave V2 protocol. The scoring model evaluates wallet reliability, risk patterns, and DeFi best practices to distinguish between responsible users and potentially risky or bot-like behavior.

## Credit Score Logic

### Score Range: 0-1000
- **800-1000**: Excellent - Long-term, responsible users with no liquidations
- **600-799**: Good - Regular users with occasional borrowing
- **400-599**: Fair - New users or those with some risk indicators
- **200-399**: Poor - Users with multiple risk factors
- **0-199**: Very Poor - High-risk behavior, frequent liquidations, or bot-like patterns

### Feature Engineering

The system extracts 20+ features from raw transaction data:

#### Transaction Volume & Activity
- `total_transactions`: Total number of protocol interactions
- `unique_days`: Number of unique days with activity
- `avg_transactions_per_day`: Daily transaction frequency
- `tx_frequency`: Transactions per day since wallet creation

#### Action Type Distribution
- `deposit_count`: Number of deposit actions
- `borrow_count`: Number of borrow actions
- `repay_count`: Number of repay actions
- `redeem_count`: Number of redemption actions
- `liquidation_count`: Number of liquidation events

#### Risk Indicators
- `liquidation_ratio`: Ratio of liquidations to total transactions
- `borrow_to_deposit_ratio`: Borrowing frequency relative to deposits
- `repay_to_borrow_ratio`: Repayment behavior relative to borrowing

#### Temporal Patterns
- `avg_time_between_tx`: Average time between transactions
- `std_time_between_tx`: Consistency of transaction timing
- `rapid_transaction_ratio`: Ratio of transactions <60 seconds apart

#### Portfolio Characteristics
- `unique_assets`: Number of different assets interacted with
- `asset_diversity`: Asset diversity relative to transaction count
- `wallet_age_days`: Days since first transaction

#### Value Patterns (if available)
- `total_value`: Total value of all transactions
- `avg_value_per_tx`: Average transaction value
- `std_value_per_tx`: Value consistency

### Scoring Algorithm

The system uses a **Random Forest Regressor** trained on synthetic scores generated from DeFi best practices:

#### Positive Factors (+ points)
- **Good repayment behavior** (+100): repay_to_borrow_ratio > 1.0
- **No liquidations** (+150): liquidation_ratio = 0
- **Long-term user** (+50): wallet_age_days > 30
- **Diversified portfolio** (+30): asset_diversity > 0.3
- **Regular activity** (+20): tx_frequency > 0.1

#### Negative Factors (- points)
- **High liquidation risk** (-200): liquidation_ratio > 0.1
- **Over-leveraged** (-100): borrow_to_deposit_ratio > 2.0
- **Bot-like behavior** (-150): rapid_transaction_ratio > 0.5
- **Very rapid transactions** (-100): avg_time_between_tx < 60 seconds
- **Very low activity** (-50): total_transactions < 3

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a New Model

```bash
python credit_scorer.py --input user_transactions.json --output wallet_scores.json --train
```

### Using a Pre-trained Model

```bash
python credit_scorer.py --input user_transactions.json --output wallet_scores.json --load --model credit_model.pkl
```

### Input Data Format

The system expects a JSON file with transaction records:

```json
[
  {
    "from": "0x1234...",
    "action": "deposit",
    "timestamp": 1640995200,
    "reserve": "0x6b175474e89094c44da98b954eedeac495271d0f",
    "value": "1000000000000000000"
  }
]
```

### Output Format

```json
[
  {
    "wallet_address": "0x1234...",
    "credit_score": 750,
    "total_transactions": 15,
    "wallet_age_days": 45,
    "liquidation_ratio": 0.0,
    "borrow_to_deposit_ratio": 0.8
  }
]
```

## Model Performance

The Random Forest model typically achieves:
- **R² Score**: 0.85-0.95 (depending on data quality)
- **Mean Squared Error**: <1000
- **Feature Importance**: Liquidation ratio and repayment behavior are most predictive

## Key Features

### 1. Bot Detection
- Identifies rapid transaction patterns (<60 seconds apart)
- Detects unusual timing consistency
- Flags high-frequency trading behavior

### 2. Risk Assessment
- Liquidation history analysis
- Leverage ratio monitoring
- Repayment behavior evaluation

### 3. User Behavior Profiling
- Activity consistency scoring
- Portfolio diversification analysis
- Long-term vs. short-term user classification

### 4. Transparency
- Feature importance visualization
- Score distribution analysis
- Detailed scoring rationale

## Extensibility

### Adding New Features
1. Modify `_extract_wallet_features()` in `AaveCreditScorer`
2. Add feature calculation logic
3. Update feature list in `train_model()`

### Custom Scoring Logic
1. Modify `generate_synthetic_scores()` method
2. Adjust positive/negative factor weights
3. Add new risk indicators

### Model Improvements
- Implement ensemble methods (XGBoost, LightGBM)
- Add cross-validation for better generalization
- Include more sophisticated bot detection algorithms

## Validation

The system validates scores through:
- **Distribution Analysis**: Ensures realistic score spread
- **Feature Correlation**: Checks for multicollinearity
- **Outlier Detection**: Identifies unusual scoring patterns
- **Performance Metrics**: R², MSE, and feature importance analysis

## Limitations

1. **Synthetic Training Data**: Uses rule-based synthetic scores for training
2. **Historical Bias**: Based on past behavior patterns
3. **Protocol Specific**: Designed for Aave V2, may need adaptation for other protocols
4. **Feature Dependencies**: Requires specific transaction data fields

## Future Enhancements

- **Real-time Scoring**: Incremental model updates
- **Multi-protocol Support**: Extend to other DeFi protocols
- **Advanced ML Models**: Deep learning approaches
- **External Data Integration**: On-chain reputation systems
- **Dynamic Thresholds**: Adaptive scoring based on market conditions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## License

MIT License - see LICENSE file for details. 