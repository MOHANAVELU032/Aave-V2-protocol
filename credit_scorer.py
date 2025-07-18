#!/usr/bin/env python3
"""
Aave V2 Wallet Credit Scoring System

This script processes raw transaction data from Aave V2 protocol and assigns
credit scores (0-1000) to wallets based on their transaction behavior patterns.
Higher scores indicate reliable and responsible usage; lower scores reflect
risky, bot-like, or exploitative behavior.
"""

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import joblib
import os

warnings.filterwarnings('ignore')

class AaveCreditScorer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load transaction data from JSON file"""
        print(f"Loading data from {file_path}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} transactions")
        return df
    
    def engineer_features(self, df):
        """Engineer features from raw transaction data"""
        print("Engineering features...")
        
        # Handle the specific data format from user-wallet-transactions.json
        # Map the fields to our expected format
        if 'userWallet' in df.columns:
            df['from'] = df['userWallet']
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['date'] = df['timestamp'].dt.date
        
        # Extract value from actionData if available
        if 'actionData' in df.columns:
            df['value'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) if isinstance(x, dict) else 0)
            df['asset_symbol'] = df['actionData'].apply(lambda x: x.get('assetSymbol', 'unknown') if isinstance(x, dict) else 'unknown')
            df['asset_price_usd'] = df['actionData'].apply(lambda x: float(x.get('assetPriceUSD', 0)) if isinstance(x, dict) else 0)
        
        # Group by wallet address
        wallet_features = []
        
        for wallet, group in tqdm(df.groupby('from'), desc="Processing wallets"):
            features = self._extract_wallet_features(group, wallet)
            wallet_features.append(features)
        
        features_df = pd.DataFrame(wallet_features)
        print(f"Engineered features for {len(features_df)} wallets")
        return features_df
    
    def _extract_wallet_features(self, transactions, wallet):
        """Extract features for a single wallet"""
        features = {'wallet_address': wallet}
        
        # Basic transaction counts
        features['total_transactions'] = len(transactions)
        features['unique_days'] = transactions['date'].nunique()
        features['avg_transactions_per_day'] = features['total_transactions'] / max(features['unique_days'], 1)
        
        # Action type distribution
        action_counts = transactions['action'].value_counts()
        features['deposit_count'] = action_counts.get('deposit', 0)
        features['borrow_count'] = action_counts.get('borrow', 0)
        features['repay_count'] = action_counts.get('repay', 0)
        features['redeem_count'] = action_counts.get('redeemunderlying', 0)
        features['liquidation_count'] = action_counts.get('liquidationcall', 0)
        
        # Risk indicators
        features['liquidation_ratio'] = features['liquidation_count'] / max(features['total_transactions'], 1)
        features['borrow_to_deposit_ratio'] = features['borrow_count'] / max(features['deposit_count'], 1)
        features['repay_to_borrow_ratio'] = features['repay_count'] / max(features['borrow_count'], 1)
        
        # Temporal patterns
        transactions_sorted = transactions.sort_values('timestamp')
        if len(transactions_sorted) > 1:
            time_diffs = transactions_sorted['timestamp'].diff().dt.total_seconds()
            features['avg_time_between_tx'] = time_diffs.mean()
            features['std_time_between_tx'] = time_diffs.std()
            features['min_time_between_tx'] = time_diffs.min()
            features['max_time_between_tx'] = time_diffs.max()
            
            # Bot-like behavior indicators
            features['rapid_transactions'] = (time_diffs < 60).sum()
            features['rapid_transaction_ratio'] = features['rapid_transactions'] / max(features['total_transactions'], 1)
        else:
            features['avg_time_between_tx'] = 0
            features['std_time_between_tx'] = 0
            features['min_time_between_tx'] = 0
            features['max_time_between_tx'] = 0
            features['rapid_transactions'] = 0
            features['rapid_transaction_ratio'] = 0
        
        # Value-based features (if available)
        if 'value' in transactions.columns:
            features['total_value'] = transactions['value'].sum()
            features['avg_value_per_tx'] = transactions['value'].mean()
            features['std_value_per_tx'] = transactions['value'].std()
        else:
            features['total_value'] = 0
            features['avg_value_per_tx'] = 0
            features['std_value_per_tx'] = 0
        
        # Asset diversity (if available)
        if 'asset_symbol' in transactions.columns:
            features['unique_assets'] = transactions['asset_symbol'].nunique()
            features['asset_diversity'] = features['unique_assets'] / max(features['total_transactions'], 1)
        else:
            features['unique_assets'] = 1
            features['asset_diversity'] = 1.0
        
        # Time-based features
        features['first_tx_date'] = transactions['timestamp'].min()
        features['last_tx_date'] = transactions['timestamp'].max()
        features['wallet_age_days'] = (features['last_tx_date'] - features['first_tx_date']).days
        
        # Activity consistency
        features['tx_frequency'] = features['total_transactions'] / max(features['wallet_age_days'], 1)
        
        # Network and protocol info
        if 'network' in transactions.columns:
            features['network'] = transactions['network'].iloc[0]
        if 'protocol' in transactions.columns:
            features['protocol'] = transactions['protocol'].iloc[0]
        
        return features
    
    def generate_synthetic_scores(self, features_df):
        """Generate synthetic credit scores based on engineered features"""
        print("Generating synthetic credit scores...")
        
        # Create a scoring algorithm based on DeFi best practices
        scores = []
        
        for _, row in features_df.iterrows():
            score = 500  # Base score
            
            # Positive factors
            if row['repay_to_borrow_ratio'] > 1.0:
                score += 100  # Good repayment behavior
            if row['liquidation_ratio'] == 0:
                score += 150  # No liquidations
            if row['wallet_age_days'] > 30:
                score += 50   # Long-term user
            if row['asset_diversity'] > 0.3:
                score += 30   # Diversified portfolio
            if row['tx_frequency'] > 0.1:
                score += 20   # Regular activity
            if row['total_transactions'] > 10:
                score += 30   # Active user
            
            # Negative factors
            if row['liquidation_ratio'] > 0.1:
                score -= 200  # High liquidation risk
            if row['borrow_to_deposit_ratio'] > 2.0:
                score -= 100  # Over-leveraged
            if row['rapid_transaction_ratio'] > 0.5:
                score -= 150  # Bot-like behavior
            if row['avg_time_between_tx'] < 60:
                score -= 100  # Very rapid transactions
            if row['total_transactions'] < 3:
                score -= 50   # Very low activity
            if row['std_time_between_tx'] < 10 and row['total_transactions'] > 5:
                score -= 80   # Too consistent timing (bot-like)
            
            # Ensure score is within bounds
            score = max(0, min(1000, score))
            scores.append(score)
        
        return scores
    
    def train_model(self, features_df, scores):
        """Train the credit scoring model"""
        print("Training credit scoring model...")
        
        # Prepare features (exclude non-numeric columns)
        feature_cols = [col for col in features_df.columns if col not in 
                       ['wallet_address', 'first_tx_date', 'last_tx_date', 'network', 'protocol']]
        
        X = features_df[feature_cols].fillna(0)
        y = np.array(scores)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.3f}")
        
        self.feature_names = feature_cols
        return X_test_scaled, y_test, y_pred
    
    def predict_scores(self, features_df):
        """Predict credit scores for wallets"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        feature_cols = [col for col in features_df.columns if col not in 
                       ['wallet_address', 'first_tx_date', 'last_tx_date', 'network', 'protocol']]
        
        X = features_df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        scores = self.model.predict(X_scaled)
        scores = np.clip(scores, 0, 1000)  # Ensure scores are within bounds
        
        return scores
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
    
    def plot_feature_importance(self, save_path=None):
        """Plot feature importance"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance for Credit Scoring')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), 
                  [self.feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_distribution(self, scores, save_path=None):
        """Plot distribution of credit scores"""
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Credit Scores')
        plt.xlabel('Credit Score')
        plt.ylabel('Number of Wallets')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the credit scoring pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aave V2 Wallet Credit Scoring')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--output', default='wallet_scores.json', help='Path to output JSON file')
    parser.add_argument('--model', default='credit_model.pkl', help='Path to save/load model')
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--load', action='store_true', help='Load existing model')
    
    args = parser.parse_args()
    
    # Initialize scorer
    scorer = AaveCreditScorer()
    
    if args.load and os.path.exists(args.model):
        # Load existing model
        scorer.load_model(args.model)
        
        # Load and process data
        df = scorer.load_data(args.input)
        features_df = scorer.engineer_features(df)
        
        # Predict scores
        scores = scorer.predict_scores(features_df)
        
    else:
        # Load and process data
        df = scorer.load_data(args.input)
        features_df = scorer.engineer_features(df)
        
        # Generate synthetic scores for training
        synthetic_scores = scorer.generate_synthetic_scores(features_df)
        
        # Train model
        X_test, y_test, y_pred = scorer.train_model(features_df, synthetic_scores)
        
        # Save model
        scorer.save_model(args.model)
        
        # Plot results
        scorer.plot_feature_importance('feature_importance.png')
        scorer.plot_score_distribution(synthetic_scores, 'score_distribution.png')
        
        scores = synthetic_scores
    
    # Create results
    results = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        results.append({
            'wallet_address': row['wallet_address'],
            'credit_score': int(scores[i]),
            'total_transactions': int(row['total_transactions']),
            'wallet_age_days': int(row['wallet_age_days']),
            'liquidation_ratio': float(row['liquidation_ratio']),
            'borrow_to_deposit_ratio': float(row['borrow_to_deposit_ratio']),
            'repay_to_borrow_ratio': float(row['repay_to_borrow_ratio']),
            'asset_diversity': float(row['asset_diversity']),
            'rapid_transaction_ratio': float(row['rapid_transaction_ratio'])
        })
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"Processed {len(results)} wallets")
    
    # Print summary statistics
    scores_array = np.array(scores)
    print(f"\nScore Statistics:")
    print(f"  Mean: {scores_array.mean():.1f}")
    print(f"  Median: {np.median(scores_array):.1f}")
    print(f"  Std: {scores_array.std():.1f}")
    print(f"  Min: {scores_array.min():.1f}")
    print(f"  Max: {scores_array.max():.1f}")
    
    # Print score distribution
    excellent = np.sum(scores_array >= 800)
    good = np.sum((scores_array >= 600) & (scores_array < 800))
    fair = np.sum((scores_array >= 400) & (scores_array < 600))
    poor = np.sum((scores_array >= 200) & (scores_array < 400))
    very_poor = np.sum(scores_array < 200)
    
    print(f"\nScore Distribution:")
    print(f"  Excellent (800-1000): {excellent} wallets ({excellent/len(scores_array)*100:.1f}%)")
    print(f"  Good (600-799): {good} wallets ({good/len(scores_array)*100:.1f}%)")
    print(f"  Fair (400-599): {fair} wallets ({fair/len(scores_array)*100:.1f}%)")
    print(f"  Poor (200-399): {poor} wallets ({poor/len(scores_array)*100:.1f}%)")
    print(f"  Very Poor (0-199): {very_poor} wallets ({very_poor/len(scores_array)*100:.1f}%)")

if __name__ == "__main__":
    main() 