import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class PriceElasticityAnalyzer:
    def __init__(self, target_column='quantity_sold'):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.target_column = target_column
        self.elasticity_coefficients = {}

    def prepare_features(self, data):
        data = data.copy()
        data['price_log'] = np.log(data['price'])
        data['price_squared'] = data['price'] ** 2
        data['price_lag_1'] = data['price'].shift(1)
        data['price_change'] = data['price'] - data['price_lag_1']
        data['price_change_pct'] = data['price_change'] / data['price_lag_1']
        if 'competitor_price' in data.columns:
            data['price_ratio'] = data['price'] / data['competitor_price']
            data['price_difference'] = data['price'] - data['competitor_price']
            data['competitive_advantage'] = (data['competitor_price'] - data['price']) / data['price']
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['day_of_week'] = data['date'].dt.dayofweek
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            data['is_holiday'] = self._identify_holidays(data['date'])
        data['price_x_seasonality'] = data['price'] * data.get('seasonality', 1)
        data['price_x_inventory'] = data['price'] * data.get('inventory_level', 1)
        data['price_segment'] = pd.cut(data['price'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        return data

    def _identify_holidays(self, dates):
        holidays = []
        for date in dates:
            if date.month == 12 and date.day >= 20:
                holidays.append(1)
            elif date.month == 1 and date.day <= 7:
                holidays.append(1)
            elif date.month == 11 and date.day >= 23 and date.day <= 29:
                holidays.append(1)
            else:
                holidays.append(0)
        return holidays

    def engineer_elasticity_features(self, data):
        data = data.copy()
        price_percentiles = data['price'].quantile([0.25, 0.5, 0.75])
        data['below_p25'] = (data['price'] <= price_percentiles[0.25]).astype(int)
        data['p25_to_p50'] = ((data['price'] > price_percentiles[0.25]) & (data['price'] <= price_percentiles[0.5])).astype(int)
        data['p50_to_p75'] = ((data['price'] > price_percentiles[0.5]) & (data['price'] <= price_percentiles[0.75])).astype(int)
        data['above_p75'] = (data['price'] > price_percentiles[0.75]).astype(int)
        data['price_ma_7'] = data['price'].rolling(window=7, min_periods=1).mean()
        data['quantity_ma_7'] = data[self.target_column].rolling(window=7, min_periods=1).mean()
        data['price_volatility'] = data['price'].rolling(window=7, min_periods=1).std()
        data['quantity_volatility'] = data[self.target_column].rolling(window=7, min_periods=1).std()
        return data

    def encode_categorical_features(self, data, categorical_columns=None):
        if categorical_columns is None:
            categorical_columns = ['price_segment', 'product_category', 'customer_segment']
        data_encoded = data.copy()
        for col in categorical_columns:
            if col in data_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data_encoded[col] = self.label_encoders[col].fit_transform(data_encoded[col].astype(str))
                else:
                    data_encoded[col] = self.label_encoders[col].transform(data_encoded[col].astype(str))
        return data_encoded

    def train(self, data, test_size=0.2, random_state=42):
        data_prep = self.prepare_features(data)
        data_prep = self.engineer_elasticity_features(data_prep)
        data_prep = self.encode_categorical_features(data_prep)
        data_prep = data_prep.dropna()
        exclude_cols = [self.target_column, 'date', 'product_id'] if 'date' in data_prep.columns else [self.target_column]
        feature_columns = [col for col in data_prep.columns if col not in exclude_cols]
        X = data_prep[feature_columns]
        y = data_prep[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        self.model_performance = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        print(f"Model Performance:")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        return self.model_performance

    def calculate_price_elasticity(self, data, price_column='price', base_price=None):
        if self.model is None:
            raise ValueError("Model must be trained first!")
        if base_price is None:
            base_price = data[price_column].median()
        price_changes = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2]
        elasticities = []
        for price_change in price_changes:
            data_modified = data.copy()
            new_price = base_price * (1 + price_change)
            data_modified[price_column] = new_price
            data_prep = self.prepare_features(data_modified)
            data_prep = self.engineer_elasticity_features(data_prep)
            data_prep = self.encode_categorical_features(data_prep)
            data_prep = data_prep.dropna()
            exclude_cols = [self.target_column, 'date', 'product_id'] if 'date' in data_prep.columns else [self.target_column]
            feature_columns = [col for col in data_prep.columns if col not in exclude_cols]
            X = data_prep[feature_columns]
            X_scaled = self.scaler.transform(X)
            predicted_demand = self.model.predict(X_scaled).mean()
            elasticities.append(predicted_demand)
        base_demand = elasticities[4]
        elasticity_data = []
        for i, (price_change, demand) in enumerate(zip(price_changes, elasticities)):
            if price_change != 0:
                pct_change_demand = (demand - base_demand) / base_demand
                pct_change_price = price_change
                elasticity = pct_change_demand / pct_change_price
                elasticity_data.append({
                    'price_change_pct': price_change * 100,
                    'demand_change_pct': pct_change_demand * 100,
                    'elasticity': elasticity,
                    'new_price': base_price * (1 + price_change),
                    'predicted_demand': demand
                })
        elasticity_df = pd.DataFrame(elasticity_data)
        overall_elasticity = elasticity_df['elasticity'].mean()
        self.elasticity_coefficients = {
            'overall_elasticity': overall_elasticity,
            'base_price': base_price,
            'base_demand': base_demand,
            'elasticity_curve': elasticity_df
        }
        print(f"Overall Price Elasticity: {overall_elasticity:.4f}")
        if overall_elasticity < -1:
            print("Demand is elastic (|elasticity| > 1)")
        else:
            print("Demand is inelastic (|elasticity| < 1)")
        return self.elasticity_coefficients

    def optimize_price(self, data, price_range=None, cost_per_unit=10):
        if self.model is None:
            raise ValueError("Model must be trained first!")
        if price_range is None:
            current_price = data['price'].median()
            price_range = np.arange(current_price * 0.5, current_price * 1.5, 1)
        profits = []
        for price in price_range:
            data_test = data.copy()
            data_test['price'] = price
            data_prep = self.prepare_features(data_test)
            data_prep = self.engineer_elasticity_features(data_prep)
            data_prep = self.encode_categorical_features(data_prep)
            data_prep = data_prep.dropna()
            exclude_cols = [self.target_column, 'date', 'product_id'] if 'date' in data_prep.columns else [self.target_column]
            feature_columns = [col for col in data_prep.columns if col not in exclude_cols]
            X = data_prep[feature_columns]
            X_scaled = self.scaler.transform(X)
            predicted_demand = self.model.predict(X_scaled).mean()
            profit = predicted_demand * (price - cost_per_unit)
            profits.append(profit)
        optimal_idx = np.argmax(profits)
        optimal_price = price_range[optimal_idx]
        optimal_profit = profits[optimal_idx]
        optimization_results = {
            'optimal_price': optimal_price,
            'optimal_profit': optimal_profit,
            'price_range': price_range,
            'profits': profits
        }
        print(f"Optimal Price: ${optimal_price:.2f}")
        print(f"Expected Profit: ${optimal_profit:.2f}")
        return optimization_results

    def get_feature_importance(self, top_n=10):
        if self.feature_importance is None:
            raise ValueError("Model must be trained first!")
        return self.feature_importance.head(top_n)

def generate_sample_pricing_data(days=365):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'product_id': 'PROD_001',
        'price': np.random.normal(50, 15, days),
        'competitor_price': np.random.normal(52, 12, days),
        'inventory_level': np.random.normal(1000, 200, days),
        'seasonality': np.sin(2 * np.pi * np.arange(days) / 365),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], days),
        'customer_segment': np.random.choice(['Premium', 'Regular', 'Budget'], days)
    })
    data['price'] = np.maximum(data['price'], 10)
    data['competitor_price'] = np.maximum(data['competitor_price'], 10)
    data['inventory_level'] = np.maximum(data['inventory_level'], 100)
    base_demand = 100
    price_elasticity = -1.5
    price_effect = price_elasticity * np.log(data['price'] / data['price'].mean()) * base_demand
    competitor_effect = 20 * np.log(data['competitor_price'] / data['price'])
    seasonal_effect = 30 * data['seasonality']
    inventory_effect = 0.01 * (data['inventory_level'] - 1000)
    category_effect = np.where(data['product_category'] == 'Electronics', 20, 
                              np.where(data['product_category'] == 'Clothing', 0, -10))
    data['quantity_sold'] = (base_demand + price_effect + competitor_effect + 
                           seasonal_effect + inventory_effect + category_effect +
                           np.random.normal(0, 10, days))
    data['quantity_sold'] = np.maximum(data['quantity_sold'], 1)
    return data

if __name__ == "__main__":
    sample_data = generate_sample_pricing_data(500)
    analyzer = PriceElasticityAnalyzer()
    print("Training XGBoost price elasticity model...")
    performance = analyzer.train(sample_data)
    print("\nCalculating price elasticity...")
    elasticity_results = analyzer.calculate_price_elasticity(sample_data)
    print("\nOptimizing price for maximum profit...")
    optimization_results = analyzer.optimize_price(sample_data, cost_per_unit=20)
    print("\nTop 10 Most Important Features:")
    print(analyzer.get_feature_importance(10))
