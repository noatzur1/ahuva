# Complete Demand Forecasting System
# Academic-Grade Time Series Forecasting for Inventory Management
# ================================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical modeling libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Machine learning libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Install required packages
# !pip install xgboost statsmodels -q

# Visualization settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# ================================================================
# PHASE 1: DATA LOADING AND INITIAL EXPLORATION
# ================================================================

print("Loading dataset...")
df = pd.read_csv('ahuva_datase מאוחד חדש.csv', encoding='utf-8')

print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
print("Columns:", list(df.columns))
print("\nDataset info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# ================================================================
# PHASE 2: DATA CLEANING AND PREPROCESSING
# ================================================================

print("\n" + "="*60)
print("DATA CLEANING AND PREPROCESSING")
print("="*60)

# Convert date column
df['תאריך'] = pd.to_datetime(df['תאריך'])

# Check for missing values
missing_summary = df.isnull().sum()
print("Missing values by column:")
print(missing_summary[missing_summary > 0])

# Fill missing values for core variables
df['כמות שנמכרה'].fillna(0, inplace=True)
df['כמות במלאי'].fillna(0, inplace=True)

# Handle negative values
negative_sales = (df['כמות שנמכרה'] < 0).sum()
negative_inventory = (df['כמות במלאי'] < 0).sum()

if negative_sales > 0:
    print(f"Fixed {negative_sales} negative sales values")
    df['כמות שנמכרה'] = df['כמות שנמכרה'].clip(lower=0)

if negative_inventory > 0:
    print(f"Fixed {negative_inventory} negative inventory values")
    df['כמות במלאי'] = df['כמות במלאי'].clip(lower=0)

# Basic statistics
print("\nDescriptive statistics:")
print(df[['כמות שנמכרה', 'כמות במלאי', 'מהירות חידוש מלאי (ימים)']].describe())

# Dataset overview
print(f"\nDataset overview:")
print(f"Total rows: {len(df):,}")
print(f"Unique SKUs: {df['מקט'].nunique():,}")
print(f"Categories: {df['קטגוריה'].nunique()}")
print(f"Date range: {df['תאריך'].min().date()} to {df['תאריך'].max().date()}")

# ================================================================
# PHASE 3: HANDLE MISSING VALUES INTELLIGENTLY
# ================================================================

print("\n" + "="*60)
print("INTELLIGENT MISSING VALUE HANDLING")
print("="*60)

# Handle missing categories and descriptions by SKU
print("Handling missing categories and descriptions...")

# Category mapping per SKU
category_mapping = df.groupby('מקט')['קטגוריה'].apply(
    lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
)

df['קטגוריה'] = df.apply(
    lambda row: category_mapping[row['מקט']] if pd.isnull(row['קטגוריה']) and row['מקט'] in category_mapping else row['קטגוריה'],
    axis=1
)

# Description mapping per SKU
description_mapping = df.groupby('מקט')['תיאור מוצר'].apply(
    lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
)

df['תיאור מוצר'] = df.apply(
    lambda row: description_mapping[row['מקט']] if pd.isnull(row['תיאור מוצר']) and row['מקט'] in description_mapping else row['תיאור מוצר'],
    axis=1
)

print("Category and description missing values handled")

# Derive time features from date
print("Deriving time features from date...")

df['יום בשבוע'] = df['יום בשבוע'].fillna(df['תאריך'].dt.dayofweek)
df['חודש'] = df['חודש'].fillna(df['תאריך'].dt.month)
df['שבוע בשנה'] = df['שבוע בשנה'].fillna(df['תאריך'].dt.isocalendar().week)

# Handle inventory turnover speed
print("Handling inventory turnover speed...")

inventory_speed_mapping = df.groupby('מקט')['מהירות חידוש מלאי (ימים)'].apply(
    lambda x: x.dropna().mean() if not x.dropna().empty else None
)

df['מהירות חידוש מלאי (ימים)'] = df.apply(
    lambda row: inventory_speed_mapping[row['מקט']]
    if pd.isnull(row['מהירות חידוש מלאי (ימים)']) and row['מקט'] in inventory_speed_mapping
    else row['מהירות חידוש מלאי (ימים)'],
    axis=1
)

# Fill remaining missing values with overall mean
overall_mean_speed = df['מהירות חידוש מלאי (ימים)'].mean()
df['מהירות חידוש מלאי (ימים)'].fillna(overall_mean_speed, inplace=True)

# Final missing value check
missing_final = df.isnull().sum()
remaining_missing = missing_final[missing_final > 0]

if len(remaining_missing) == 0:
    print("All missing values handled successfully")
else:
    print("Remaining missing values:")
    print(remaining_missing)

# ================================================================
# PHASE 4: CHECK AND HANDLE DUPLICATES
# ================================================================

print("\n" + "="*60)
print("DUPLICATE DETECTION AND AGGREGATION")
print("="*60)

# Check for SKU-date duplicates
duplicate_check = df.groupby(['מקט', 'תאריך']).size()
duplicates = duplicate_check[duplicate_check > 1]

print(f"Total records: {len(df):,}")
print(f"Unique SKU-date combinations: {len(duplicate_check):,}")
print(f"Duplicates found: {len(duplicates)}")

if len(duplicates) > 0:
    print("Aggregating duplicate records...")

    # Define aggregation strategy
    aggregation_dict = {
        'תיאור מוצר': 'first',
        'קטגוריה': 'first',
        'כמות במלאי': 'mean',
        'כמות שנמכרה': 'sum',
        'מהירות חידוש מלאי (ימים)': 'mean',
        'יום בשבוע': 'first',
        'חודש': 'first',
        'שבוע בשנה': 'first'
    }

    print(f"Before aggregation: {len(df)} rows")

    # Perform aggregation
    df = df.groupby(['מקט', 'תאריך']).agg(aggregation_dict).reset_index()

    print(f"After aggregation: {len(df)} rows")
    print(f"Removed {len(df)} duplicate records")

    # Verify no more duplicates
    final_duplicate_check = df.groupby(['מקט', 'תאריך']).size()
    remaining_duplicates = final_duplicate_check[final_duplicate_check > 1]

    if len(remaining_duplicates) == 0:
        print("Duplicate aggregation successful")
    else:
        print(f"Warning: {len(remaining_duplicates)} duplicates still exist")

# ================================================================
# PHASE 5: FEATURE ENGINEERING
# ================================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Sort data chronologically
df = df.sort_values(['מקט', 'תאריך']).reset_index(drop=True)

print("Creating lag features...")

# Lag features (using previous periods to avoid data leakage)
df['sales_lag1'] = df.groupby('מקט')['כמות שנמכרה'].shift(1)
df['sales_lag7'] = df.groupby('מקט')['כמות שנמכרה'].shift(7)

# Moving averages (using previous periods only)
df['sales_ma7_lag'] = df.groupby('מקט')['sales_lag1'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)

# Calculated features (avoiding data leakage)
df['inventory_sales_ratio_lag'] = df.groupby('מקט')['sales_lag1'].shift(0) / (df.groupby('מקט')['כמות במלאי'].shift(1) + 1)
df['sales_change'] = df.groupby('מקט')['כמות שנמכרה'].pct_change().shift(1).fillna(0)

# Feature summary
original_columns = ['מקט', 'תאריך', 'תיאור מוצר', 'קטגוריה', 'כמות במלאי', 'כמות שנמכרה',
                   'מהירות חידוש מלאי (ימים)', 'יום בשבוע', 'חודש', 'שבוע בשנה']

new_features = [col for col in df.columns if col not in original_columns]

print(f"Created {len(new_features)} new features:")
for i, feature in enumerate(new_features, 1):
    print(f"{i}. {feature}")

# Check feature quality
print("\nFeature quality check:")
for feature in new_features:
    missing_count = df[feature].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"{feature}: {missing_count} missing ({missing_pct:.1f}%)")

# ================================================================
# PHASE 6: DEMAND STABILITY CLASSIFICATION
# ================================================================

print("\n" + "="*60)
print("DEMAND STABILITY CLASSIFICATION")
print("="*60)

print("Calculating coefficient of variation (CV) for each SKU...")

# Calculate statistics per SKU
product_stats = df.groupby('מקט')['כמות שנמכרה'].agg([
    'mean', 'std', 'count'
]).reset_index()

# Calculate CV
product_stats['cv'] = product_stats['std'] / (product_stats['mean'] + 1e-8)

# Filter products with sufficient observations
product_stats = product_stats[product_stats['count'] >= 10].reset_index(drop=True)

print(f"Products with sufficient data: {len(product_stats)} out of {df['מקט'].nunique()}")

# Set CV threshold based on median
cv_threshold = product_stats['cv'].median()
print(f"CV threshold selected: {cv_threshold:.3f} (median)")

# Classify products
product_stats['demand_group'] = product_stats['cv'].apply(
    lambda x: 'stable' if x <= cv_threshold else 'volatile'
)

# Summary of classification
demand_groups = product_stats['demand_group'].value_counts()
print("Product classification:")
for group, count in demand_groups.items():
    print(f"{group} demand: {count} products ({count/len(product_stats)*100:.1f}%)")

# Statistics by group
print("\nStatistics by demand group:")
for group in ['stable', 'volatile']:
    group_data = product_stats[product_stats['demand_group'] == group]
    if len(group_data) > 0:
        print(f"\n{group} demand:")
        print(f"  Average CV: {group_data['cv'].mean():.3f}")
        print(f"  Average sales: {group_data['mean'].mean():.2f}")
        print(f"  CV range: {group_data['cv'].min():.3f} - {group_data['cv'].max():.3f}")

# Add classification to main dataset
df = df.merge(product_stats[['מקט', 'demand_group']], on='מקט', how='left')

# Remove records without classification
missing_classification = df['demand_group'].isnull().sum()
if missing_classification > 0:
    print(f"Removing {missing_classification} records without classification")
    df = df.dropna(subset=['demand_group']).reset_index(drop=True)
    print(f"Remaining records: {len(df)}")

# Final classification distribution
final_split = df['demand_group'].value_counts()
print("\nFinal data distribution:")
for group, count in final_split.items():
    print(f"{group} demand: {count} records")

# ================================================================
# PHASE 7: CHRONOLOGICAL TRAIN-TEST SPLIT
# ================================================================

print("\n" + "="*60)
print("CHRONOLOGICAL TRAIN-TEST SPLIT")
print("="*60)

# Sort by date
df = df.sort_values(['מקט', 'תאריך']).reset_index(drop=True)

# Calculate date range
min_date = df['תאריך'].min()
max_date = df['תאריך'].max()
total_days = (max_date - min_date).days

print(f"Date range: {min_date.date()} to {max_date.date()}")
print(f"Total days: {total_days}")

# Chronological split - 80% train, 20% test
split_ratio = 0.8
split_date = min_date + pd.Timedelta(days=int(total_days * split_ratio))

print(f"Split date: {split_date.date()}")

# Create train and test sets
train_data = df[df['תאריך'] < split_date].copy()
test_data = df[df['תאריך'] >= split_date].copy()

print(f"\nData split:")
print(f"Training: {len(train_data):,} records ({len(train_data)/len(df)*100:.1f}%)")
print(f"Testing: {len(test_data):,} records ({len(test_data)/len(df)*100:.1f}%)")

# Check SKU coverage
items_in_train = set(train_data['מקט'].unique())
items_in_test = set(test_data['מקט'].unique())
items_in_both = items_in_train.intersection(items_in_test)

print(f"\nSKU coverage check:")
print(f"SKUs in both sets: {len(items_in_both)}")
print(f"SKUs only in training: {len(items_in_train - items_in_test)}")
print(f"SKUs only in testing: {len(items_in_test - items_in_train)}")

# Distribution by demand group
print("\nDistribution by demand group:")
for dataset_name, dataset in [('Training', train_data), ('Testing', test_data)]:
    demand_split = dataset['demand_group'].value_counts()
    print(f"\n{dataset_name}:")
    for group, count in demand_split.items():
        print(f"  {group} demand: {count:,} records")

# Date range verification
print(f"\nDate ranges:")
print(f"Training: {train_data['תאריך'].min().date()} to {train_data['תאריך'].max().date()}")
print(f"Testing: {test_data['תאריך'].min().date()} to {test_data['תאריך'].max().date()}")

# Verify no temporal overlap
if train_data['תאריך'].max() >= test_data['תאריך'].min():
    print("Warning: Temporal overlap detected")
else:
    print("No temporal overlap - split is correct")

# ================================================================
# PHASE 8: OUTLIER TREATMENT AND MODEL IMPROVEMENT
# ================================================================

print("\n" + "="*60)
print("OUTLIER TREATMENT AND DATA IMPROVEMENT")
print("="*60)

# Create working copy
df_improved = df.copy()

# Outlier treatment using IQR method
def cap_outliers(series, factor=2.0):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + factor * IQR
    lower_bound = Q1 - factor * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

print("Treating outliers in sales data...")

# Apply outlier capping by SKU
df_improved['sales_corrected'] = df_improved.groupby('מקט')['כמות שנמכרה'].transform(
    lambda x: cap_outliers(x, factor=2.0)
)

# Compare before and after
print(f"Before correction - max sales: {df['כמות שנמכרה'].max():.0f}")
print(f"After correction - max sales: {df_improved['sales_corrected'].max():.0f}")

# Reclassify with corrected data
print("Reclassifying products with corrected data...")

product_stats_improved = df_improved.groupby('מקט')['sales_corrected'].agg([
    'mean', 'std', 'count'
]).reset_index()

product_stats_improved['cv'] = product_stats_improved['std'] / (product_stats_improved['mean'] + 1e-8)
product_stats_improved = product_stats_improved[product_stats_improved['count'] >= 10]

# New threshold
cv_threshold_new = product_stats_improved['cv'].median()
print(f"New CV threshold: {cv_threshold_new:.3f}")

product_stats_improved['demand_group_new'] = product_stats_improved['cv'].apply(
    lambda x: 'stable' if x <= cv_threshold_new else 'volatile'
)

# Add to dataset
df_improved = df_improved.merge(
    product_stats_improved[['מקט', 'demand_group_new']],
    on='מקט', how='left'
)

# Compare classifications
print("\nClassification comparison:")
comparison = pd.crosstab(df_improved['demand_group'], df_improved['demand_group_new'])
print(comparison)

# Create improved train/test split
split_date = train_data['תאריך'].max()

train_improved = df_improved[df_improved['תאריך'] <= split_date].copy()
test_improved = df_improved[df_improved['תאריך'] > split_date].copy()

print(f"\nImproved dataset summary:")
print(f"Training records: {len(train_improved)}")
print(f"Testing records: {len(test_improved)}")

# ================================================================
# PHASE 9: STATISTICAL MODELS FOR STABLE DEMAND
# ================================================================

print("\n" + "="*60)
print("STATISTICAL MODELS FOR STABLE DEMAND")
print("="*60)

# Filter stable demand products
stable_train = train_improved[train_improved['demand_group_new'] == 'stable'].copy()
stable_test = test_improved[test_improved['demand_group_new'] == 'stable'].copy()

print(f"Stable demand products in training: {stable_train['מקט'].nunique()}")
print(f"Training records: {len(stable_train)}")
print(f"Testing records: {len(stable_test)}")

# Initialize prediction dictionaries
predictions_ma = {}
predictions_exp = {}
predictions_hw = {}

# Model training
stable_items = stable_train['מקט'].unique()
successful_models = 0

print("Training statistical models...")

for item_id in stable_items:
    try:
        # Get item data
        item_train = stable_train[stable_train['מקט'] == item_id].sort_values('תאריך')
        item_test = stable_test[stable_test['מקט'] == item_id].sort_values('תאריך')

        # Skip items with insufficient data
        if len(item_train) < 10 or len(item_test) == 0:
            continue

        # Create time series
        sales_series = item_train.set_index('תאריך')['sales_corrected']
        forecast_periods = len(item_test)

        # 1. Moving Average
        window_size = min(7, len(sales_series))
        ma_value = sales_series.tail(window_size).mean()
        predictions_ma[item_id] = np.full(forecast_periods, ma_value)

        # 2. Exponential Smoothing
        try:
            model_exp = ExponentialSmoothing(sales_series, trend=None, seasonal=None)
            fit_exp = model_exp.fit()
            forecast_exp = fit_exp.forecast(steps=forecast_periods)
            predictions_exp[item_id] = forecast_exp.values
        except:
            predictions_exp[item_id] = np.full(forecast_periods, sales_series.mean())

        # 3. Holt-Winters
        try:
            if len(sales_series) >= 14:
                model_hw = ExponentialSmoothing(sales_series, trend='add', seasonal='add', seasonal_periods=7)
            else:
                model_hw = ExponentialSmoothing(sales_series, trend='add', seasonal=None)
            fit_hw = model_hw.fit()
            forecast_hw = fit_hw.forecast(steps=forecast_periods)
            predictions_hw[item_id] = forecast_hw.values
        except:
            predictions_hw[item_id] = np.full(forecast_periods, sales_series.mean())

        successful_models += 1

    except:
        continue

print(f"Successfully trained models for {successful_models} stable SKUs")

# ================================================================
# PHASE 10: MODEL EVALUATION AND COMPARISON
# ================================================================

print("\n" + "="*60)
print("MODEL EVALUATION AND COMPARISON")
print("="*60)

# Evaluation functions
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')

    # WAPE
    if y_true.sum() != 0:
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    else:
        wape = float('inf')

    return rmse, mae, mape, wape

def evaluate_model(predictions, model_name):
    all_true = []
    all_pred = []

    for item_id, pred_values in predictions.items():
        item_test = stable_test[stable_test['מקט'] == item_id].sort_values('תאריך')
        if len(item_test) == len(pred_values):
            all_true.extend(item_test['sales_corrected'].values)
            all_pred.extend(pred_values)

    if len(all_true) > 0:
        rmse, mae, mape, wape = calculate_metrics(np.array(all_true), np.array(all_pred))
        return {
            'Model': model_name,
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'MAPE': round(mape, 2),
            'WAPE': round(wape, 2),
            'Predictions': len(all_true)
        }
    return None

# Evaluate all models
results = []
models_to_evaluate = [
    (predictions_ma, 'Moving Average'),
    (predictions_exp, 'Exponential Smoothing'),
    (predictions_hw, 'Holt-Winters')
]

print("Evaluating statistical models for stable demand...")

for predictions, model_name in models_to_evaluate:
    result = evaluate_model(predictions, model_name)
    if result:
        results.append(result)

# Create results table
if len(results) > 0:
    results_df = pd.DataFrame(results)
    print("\nStatistical Models Performance (Stable Demand):")
    print(results_df.to_string(index=False))

    # Find best model
    best_model = results_df.loc[results_df['RMSE'].idxmin()]
    print(f"\nBest performing model for stable demand:")
    print(f"   {best_model['Model']} - RMSE: {best_model['RMSE']}")

    # Model ranking
    print("\nModel ranking (by RMSE):")
    sorted_results = results_df.sort_values('RMSE')
    for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
        print(f"   {i}. {row['Model']} - RMSE: {row['RMSE']}")
else:
    print("No evaluation results available")

# ================================================================
# PHASE 11: MACHINE LEARNING MODELS FOR VOLATILE DEMAND
# ================================================================

print("\n" + "="*60)
print("MACHINE LEARNING MODELS FOR VOLATILE DEMAND")
print("="*60)

# Filter volatile demand products
volatile_train = train_improved[train_improved['demand_group_new'] == 'volatile'].copy()
volatile_test = test_improved[test_improved['demand_group_new'] == 'volatile'].copy()

print(f"Volatile demand products: {volatile_train['מקט'].nunique()}")
print(f"Training records: {len(volatile_train)}")
print(f"Testing records: {len(volatile_test)}")

# Feature selection for ML models
feature_cols = [
    'יום בשבוע', 'חודש', 'שבוע בשנה',
    'sales_lag1', 'sales_lag7', 'sales_ma7_lag',
    'inventory_sales_ratio_lag', 'sales_change',
    'מהירות חידוש מלאי (ימים)'
]

# Prepare training data
volatile_train_clean = volatile_train.dropna(subset=feature_cols + ['sales_corrected'])
volatile_test_clean = volatile_test.dropna(subset=feature_cols)

if len(volatile_train_clean) == 0:
    print("Insufficient data for ML model training")
else:
    X_train = volatile_train_clean[feature_cols]
    y_train = volatile_train_clean['sales_corrected']

    # Additional data cleaning for ML models
    # Remove infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    y_train = y_train.replace([np.inf, -np.inf], np.nan)

    # Remove any remaining NaN values
    mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train = X_train[mask]
    y_train = y_train[mask]

    print(f"Training ML models on {len(X_train)} samples after cleaning")

    if len(X_train) < 10:
        print("Insufficient clean data for ML model training")
    else:
        # Random Forest with hyperparameter tuning
        print("Training Random Forest...")
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        rf_grid.fit(X_train, y_train)

        best_rf = rf_grid.best_estimator_
        print(f"Best RF parameters: {rf_grid.best_params_}")

        # XGBoost with hyperparameter tuning
        print("Training XGBoost...")
        xgb_params = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }

        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)

        best_xgb = xgb_grid.best_estimator_
        print(f"Best XGB parameters: {xgb_grid.best_params_}")

        # Generate predictions
        predictions_rf = {}
        predictions_xgb = {}

        for item_id in volatile_test_clean['מקט'].unique():
            item_test_data = volatile_test_clean[volatile_test_clean['מקט'] == item_id]
            if len(item_test_data) > 0:
                X_test_item = item_test_data[feature_cols]

                # Clean test data
                X_test_item = X_test_item.replace([np.inf, -np.inf], np.nan)
                X_test_item = X_test_item.dropna()

                if len(X_test_item) > 0:
                    pred_rf = best_rf.predict(X_test_item)
                    pred_xgb = best_xgb.predict(X_test_item)

                    predictions_rf[item_id] = pred_rf
                    predictions_xgb[item_id] = pred_xgb

        print(f"Generated predictions for {len(predictions_rf)} volatile SKUs")

    # Evaluate ML models
    def evaluate_ml_model(predictions, model_name):
        all_true = []
        all_pred = []

        for item_id, pred_values in predictions.items():
            item_test = volatile_test_clean[volatile_test_clean['מקט'] == item_id].sort_values('תאריך')
            if len(item_test) == len(pred_values):
                all_true.extend(item_test['sales_corrected'].values)
                all_pred.extend(pred_values)

        if len(all_true) > 0:
            rmse, mae, mape, wape = calculate_metrics(np.array(all_true), np.array(all_pred))
            return {
                'Model': model_name,
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'MAPE': round(mape, 2),
                'WAPE': round(wape, 2),
                'Predictions': len(all_true)
            }
        return None

    # Evaluate ML models
    ml_results = []
    ml_models_to_evaluate = [
        (predictions_rf, 'Random Forest'),
        (predictions_xgb, 'XGBoost')
    ]

    print("Evaluating ML models for volatile demand...")

    for predictions, model_name in ml_models_to_evaluate:
        result = evaluate_ml_model(predictions, model_name)
        if result:
            ml_results.append(result)

    # Display ML results
    if len(ml_results) > 0:
        ml_results_df = pd.DataFrame(ml_results)
        print("\nMachine Learning Models Performance (Volatile Demand):")
        print(ml_results_df.to_string(index=False))

        best_ml_model = ml_results_df.loc[ml_results_df['RMSE'].idxmin()]
        print(f"\nBest performing ML model for volatile demand:")
        print(f"   {best_ml_model['Model']} - RMSE: {best_ml_model['RMSE']}")
    else:
        print("No ML evaluation results available")

# ================================================================
# PHASE 12: FINAL RESULTS AND RECOMMENDATIONS
# ================================================================

print("\n" + "="*60)
print("FINAL RESULTS AND RECOMMENDATIONS")
print("="*60)

# Summary statistics
print("Dataset Summary:")
print(f"Total records processed: {len(df_improved):,}")
print(f"Unique SKUs: {df_improved['מקט'].nunique()}")
print(f"Date range: {df_improved['תאריך'].min().date()} to {df_improved['תאריך'].max().date()}")
print(f"Training period: {train_improved['תאריך'].min().date()} to {train_improved['תאריך'].max().date()}")
print(f"Testing period: {test_improved['תאריך'].min().date()} to {test_improved['תאריך'].max().date()}")

# Demand classification summary
final_classification = df_improved['demand_group_new'].value_counts()
print(f"\nFinal demand classification:")
for group, count in final_classification.items():
    pct = (count / len(df_improved)) * 100
    print(f"{group} demand: {count:,} records ({pct:.1f}%)")

# Model performance summary
print(f"\nModel Performance Summary:")

if len(results) > 0:
    print(f"\nStable Demand Models:")
    stable_best = results_df.loc[results_df['RMSE'].idxmin()]
    print(f"Best model: {stable_best['Model']}")
    print(f"RMSE: {stable_best['RMSE']}, MAE: {stable_best['MAE']}, MAPE: {stable_best['MAPE']}%")

if 'ml_results_df' in locals() and len(ml_results_df) > 0:
    print(f"\nVolatile Demand Models:")
    volatile_best = ml_results_df.loc[ml_results_df['RMSE'].idxmin()]
    print(f"Best model: {volatile_best['Model']}")
    print(f"RMSE: {volatile_best['RMSE']}, MAE: {volatile_best['MAE']}, MAPE: {volatile_best['MAPE']}%")

# Recommendations
print(f"\nKey Recommendations:")
print(f"1. Use statistical models (Exponential Smoothing, Holt-Winters) for stable demand products")
print(f"2. Apply machine learning models (Random Forest, XGBoost) for volatile demand products")
print(f"3. Implement outlier treatment to improve forecast accuracy")
print(f"4. Consider separate models for different product categories")
print(f"5. Monitor model performance and retrain periodically")

# Feature importance (if available)
if 'best_rf' in locals():
    print(f"\nTop 5 Important Features (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
        print(f"{i}. {row['feature']}: {row['importance']:.3f}")

print(f"\nForecasting system implementation completed successfully.")
print(f"="*60)
