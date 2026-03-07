"""
preprocessing.py

This module handles:
- Data loading
- Data analysis
- Trend discovery
- Feature engineering
- Data normalization
- Data standardization

Author: Jiayang (joe) Pang
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    def __init__(self, file_path: str):
        """
        Initialize preprocessor with dataset path
        """
        self.file_path = file_path
        self.df = None
        self.df_encoded = None
        self.df_minmax = None
        self.df_standardized = None

    # ==============================
    # 1. Load Data
    # ==============================
    def load_data(self):
        """
        Load dataset and convert date column
        """
        self.df = pd.read_csv(self.file_path)
        self.df['order_date'] = pd.to_datetime(self.df['order_date'])
        if 'order_id' in self.df.columns:
            self.df = self.df.drop(columns=['order_id'])
        return self.df

    # ==============================
    # 2. Basic Analysis
    # ==============================
    def basic_analysis(self):
        """
        Perform basic data analysis
        """
        analysis = {
            "shape": self.df.shape,
            "missing_values": self.df.isnull().sum(),
            "summary_statistics": self.df.describe(include='all')
        }
        return analysis

    # ==============================
    # 3. Discover Trends
    # ==============================
    def discover_trends(self):
        """
        Generate trend statistics (no plotting here)
        """
        trends = {}

        # Monthly order count
        trends["monthly_orders"] = (
            self.df.groupby(self.df['order_date'].dt.month).size()
        )

        # Average delivery time per restaurant
        trends["avg_delivery_time_by_restaurant"] = (
            self.df.groupby('restaurant_type')['delivery_time_minutes'].mean()
        )

        # Average delivery time by distance
        trends["distance_vs_time"] = (
            self.df[['delivery_distance_km', 'delivery_time_minutes']].corr()
        )

        return trends

    # ==============================
    # 4. Feature Engineering
    # ==============================
    def feature_engineering(self):
        """
        Extract useful features and encode categorical variables
        """
        df = self.df.copy()

        # Extract date features
        df['order_month'] = df['order_date'].dt.month
        df['order_day'] = df['order_date'].dt.day
        df['order_weekday'] = df['order_date'].dt.weekday

        # Drop unnecessary columns
        df = df.drop(columns=['order_date'])

        # Drop missing values
        df = df.dropna()

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, drop_first=True)

        self.df_encoded = df_encoded
        return df_encoded

    # ==============================
    # 5. Correlation with Targets
    # ==============================
    def target_correlations(self):
        """
        Compute correlation of all features with:
        - delivery_time_minutes
        - delivery_partner_rating
        """
        if self.df_encoded is None:
            raise ValueError("Run feature_engineering() first.")

        results = {}

        for target in ['delivery_time_minutes', 'delivery_partner_rating']:
            if target in self.df_encoded.columns:
                corr = self.df_encoded.corr(numeric_only=True)[target].drop(target)
                corr = corr.sort_values(key=lambda x: x.abs(), ascending=False)
                results[target] = corr

        return results

    # ==============================
    # 5. Normalize Data (Min-Max)
    # ==============================
    def normalize_minmax(self):
        """
        Apply Min-Max normalization
        """
        scaler = MinMaxScaler()
        numeric_cols = self.df_encoded.select_dtypes(include=[np.number]).columns

        df_scaled = self.df_encoded.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

        self.df_minmax = df_scaled
        return df_scaled

    # ==============================
    # 6. Standardize Data (Z-score)
    # ==============================
    def normalize_standard(self):
        """
        Apply StandardScaler normalization
        """
        scaler = StandardScaler()
        numeric_cols = self.df_encoded.select_dtypes(include=[np.number]).columns

        df_scaled = self.df_encoded.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

        self.df_standardized = df_scaled
        return df_scaled


# ==========================================
# Example usage (for testing only)
# ==========================================
if __name__ == "__main__":

    file_path = "./data/dataset.csv"  

    processor = DataPreprocessor(file_path)

    # Load
    df = processor.load_data()

    # Basic analysis
    analysis = processor.basic_analysis()

    print(analysis)

    # Discover trends
    trends = processor.discover_trends()
    

    # Feature engineering
    df_encoded = processor.feature_engineering()
    
    # Target correlations
    corrs = processor.target_correlations()

    print("\n=== Correlation with delivery_time_minutes ===")
    print(corrs.get("delivery_time_minutes"))

    print("\n=== Correlation with delivery_partner_rating ===")
    print(corrs.get("delivery_partner_rating"))
    # Normalize
    df_minmax = processor.normalize_minmax()
    df_standardized = processor.normalize_standard()

    print("Preprocessing completed successfully.")