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

        self.df.columns = self.df.columns.str.strip()

        obj_cols = self.df.select_dtypes(include="object").columns
        for col in obj_cols:
            self.df[col] = self.df[col].astype(str).str.strip()

        self.df['Order_Date'] = pd.to_datetime(self.df['Order_Date'], format='%d-%m-%Y', errors='coerce')

        self.df["Time_taken(min)"] = (
            self.df["Time_taken(min)"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
        )
        self.df["Time_taken(min)"] = pd.to_numeric(
            self.df["Time_taken(min)"],
            errors="coerce"
        )

        numeric_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "multiple_deliveries",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if 'ID' in self.df.columns:
            self.df = self.df.drop(columns=['ID'])
        if "Delivery_person_ID" in self.df.columns:
            self.df = self.df.drop(columns=["Delivery_person_ID"])

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
            self.df.groupby(self.df['Order_Date'].dt.month).size()
        )

        # Average delivery time per restaurant
        trends["avg_delivery_time_by_city"] = (
            self.df.groupby("City")['Time_taken(min)'].mean()
        )

        # Average delivery time by distance
        trends["rating_vs_time"] = (
            self.df[["Delivery_person_Ratings", "Time_taken(min)"]].corr()
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
        df['order_month'] = df['Order_Date'].dt.month
        df['order_day'] = df['Order_Date'].dt.day
        df['order_weekday'] = df['Order_Date'].dt.weekday
        # Time features
        df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
        df["Time_Order_picked"] = pd.to_datetime(df["Time_Order_picked"], format="%H:%M:%S", errors="coerce")

        df["order_hour"] = df["Time_Orderd"].dt.hour
        df["picked_hour"] = df["Time_Order_picked"].dt.hour
        df["prep_time_minutes"] = (
            (df["Time_Order_picked"] - df["Time_Orderd"]).dt.total_seconds() / 60.0
        )
        df.loc[df["prep_time_minutes"] < 0, "prep_time_minutes"] += 24 * 60

        # Drop raw date/time columns
        df = df.drop(columns=["Order_Date", "Time_Orderd", "Time_Order_picked"], errors="ignore")


        # Drop missing values
        print("Before dropna:", len(df))
        df = df.dropna()
        print("After dropna:", len(df))

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
        - Time_taken(min)
        - Delivery_person_Ratings
        """
        if self.df_encoded is None:
            raise ValueError("Run feature_engineering() first.")

        results = {}

        for target in ['Time_taken(min)', 'Delivery_person_Ratings']:
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

    file_path = "./data/train.csv"  

    processor = DataPreprocessor(file_path)

    # Load
    df = processor.load_data()

    # Basic analysis
    analysis = processor.basic_analysis()

    print(analysis)

    # Discover trends
    trends = processor.discover_trends()
    print("\n=== Monthly Orders ===")
    print(trends["monthly_orders"])

    print("\n=== Average Delivery Time by City ===")
    print(trends["avg_delivery_time_by_city"])

    print("\n=== Ratings vs Time Correlation ===")
    print(trends["rating_vs_time"])    

    # Feature engineering
    df_encoded = processor.feature_engineering()

    # Target correlations
    corrs = processor.target_correlations()

    print("\n=== Correlation with Time_taken(min) ===")
    print(corrs.get("Time_taken(min)"))

    print("\n=== Correlation with Delivery_person_Ratings ===")
    print(corrs.get("Delivery_person_Ratings"))

    # Normalize
    df_minmax = processor.normalize_minmax()
    df_standardized = processor.normalize_standard()

    print("Preprocessing completed successfully.")