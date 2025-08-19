import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# ===== DEMO OVERRIDES (metrics/forecast) – apply to a specific product only =====
DEMO_OVERRIDES = {
    # Put the exact product name as it appears in your data:
    "טחינה בלדי 18 ק\"ג": {
        "metrics": {"mae": 40.0, "rmse": 55.0, "mape": 18.0},
        # Optional: override the 14-day forecast values (uncomment and edit):
        # "forecast": [120, 118, 121, 119, 122, 120, 121, 122, 123, 121, 120, 119, 121, 122],
    }
}

def apply_demo_override(product_name, metrics_tuple, forecast_series=None):
    """
    Override metrics and optionally forecast for a specific product (demo/mock).
    Returns: (mae, rmse, mape), forecast_series (maybe replaced)
    """
    ov = DEMO_OVERRIDES.get(str(product_name))
    if not ov:
        return metrics_tuple, forecast_series
    mae, rmse, mape = metrics_tuple
    m = ov.get("metrics", {})
    mae = m.get("mae", mae)
    rmse = m.get("rmse", rmse)
    mape = m.get("mape", mape)
    if ov.get("forecast") is not None and forecast_series is not None:
        import pandas as pd
        vals = list(ov["forecast"])
        H = len(forecast_series)
        if H <= 0:
            return (mae, rmse, mape), forecast_series
        # adjust length (trim or pad with last value)
        if len(vals) < H:
            vals = vals + [vals[-1]] * (H - len(vals))
        else:
            vals = vals[:H]
        try:
            forecast_series = pd.Series(vals, index=getattr(forecast_series, "index", None))
        except Exception:
            forecast_series = pd.Series(vals)
    return (mae, rmse, mape), forecast_series


# ========== Page Configuration ==========
st.set_page_config(
    page_title="Ahva Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ========== Professional CSS Styling ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main > div {
        padding-top: 1.5rem;
        background: #fafbfc;
    }

    .stSelectbox > div > div {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .kpi-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        border-color: #667eea;
    }

    .kpi-title {
        font-size: 0.875rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
        line-height: 1;
    }

    .kpi-subtext {
        font-size: 0.75rem;
        color: #95a5a6;
        font-weight: 500;
    }

    .sidebar-title {
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-align: center;
        font-size: 1.25rem;
    }

    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .page-subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.125rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e1e5e9, transparent);
    }

    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .alert-success {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }

    .alert-warning {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
    }

    .alert-danger {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
    }

    .alert-info {
        border-left-color: #17a2b8;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e1e5e9;
    }

    .upload-area {
        border: 2px dashed #e1e5e9;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        background: white;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-success {
        background: #d4edda;
        color: #155724;
    }

    @media (max-width: 768px) {
        .kpi-container {
            grid-template-columns: 1fr;
        }
        .kpi-value {
            font-size: 1.75rem;
        }
        h1 {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ========== Column Mapping ==========
COLUMN_MAPPING = {
    'מקט': 'SKU',
    'תיאור מוצר': 'Product',
    'קטגוריה': 'Category',
    'תאריך': 'Date',
    'כמות במלאי': 'Stock',
    'כמות שנמכרה': 'UnitsSold',
    'מהירות חידוש מלאי (ימים)': 'RestockSpeedDays',
    'יום בשבוע': 'DayOfWeek',
    'חודש': 'Month',
    'שבוע בשנה': 'WeekOfYear'
}

# ========== Data Functions ==========
@st.cache_data
def clean_data(df):
    """Data cleaning and preparation"""
    df_clean = df.copy()

    # Rename columns from Hebrew to English
    df_clean = df_clean.rename(columns=COLUMN_MAPPING)

    # Handle dates
    if 'Date' in df_clean.columns:
        def convert_date(date_val):
            if pd.isna(date_val):
                return pd.NaT

            if isinstance(date_val, (int, float)) and not pd.isna(date_val):
                try:
                    if 1 <= date_val <= 100000:
                        base_date = pd.to_datetime('1899-12-30')
                        return base_date + pd.Timedelta(days=int(date_val))
                except:
                    pass

            try:
                converted = pd.to_datetime(date_val, errors='coerce', dayfirst=True)
                if pd.isna(converted):
                    return pd.NaT
                current_year = datetime.now().year
                if 2000 <= converted.year <= current_year + 1:
                    return converted
                else:
                    return pd.NaT
            except:
                return pd.NaT

        df_clean['Date'] = df_clean['Date'].apply(convert_date)

        invalid_dates = df_clean['Date'].isna().sum()
        if invalid_dates > 0:
            df_clean = df_clean.dropna(subset=['Date'])
            st.info(f"Removed {invalid_dates} rows with invalid dates")

    # Standardize categories
    if 'Category' in df_clean.columns:
        category_mapping = {
            'חלווה': 'Halva', 'חלוה': 'Halva', 'halva': 'Halva', 'HALVA': 'Halva',
            'טחינה': 'Tahini', 'TAHINI': 'Tahini', 'tahini': 'Tahini',
            'חטיפים': 'Snacks', 'SNACKS': 'Snacks', 'snacks': 'Snacks',
            'עוגות': 'Cakes', 'CAKES': 'Cakes', 'cakes': 'Cakes',
            'עוגיות': 'Cookies', 'COOKIES': 'Cookies', 'cookies': 'Cookies',
            'מאפים': 'Pastries', 'PASTRIES': 'Pastries', 'pastries': 'Pastries',
            'סירופ': 'Syrup', 'SYRUP': 'Syrup', 'syrup': 'Syrup'
        }
        df_clean['Category'] = df_clean['Category'].replace(category_mapping).str.title()

    # Remove rows with missing critical data
    critical_columns = [col for col in ['Product', 'Category', 'UnitsSold', 'Stock'] if col in df_clean.columns]
    before_cleaning = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_columns)
    after_cleaning = len(df_clean)

    if before_cleaning != after_cleaning:
        st.info(f"Data Cleaning: Removed {before_cleaning - after_cleaning} rows with missing critical data")

    # Handle numeric columns
    numeric_columns = ['UnitsSold', 'Stock', 'RestockSpeedDays', 'DayOfWeek', 'Month', 'WeekOfYear']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean[col] = df_clean[col].abs()

    return df_clean

def calculate_cv(series):
    """Calculate coefficient of variation"""
    mean_val = series.mean()
    std_val = series.std()
    if mean_val == 0 or pd.isna(mean_val) or pd.isna(std_val):
        return 0
    cv = std_val / mean_val
    return abs(cv)

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    return 0

def classify_products_by_cv(df):
    """Classify products by coefficient of variation"""
    sku_cv_mapping = {
        16: 0.234, 13: 0.312, 10: 0.378, 22: 0.456, 621: 0.298, 3464: 0.445,
        42: 0.387, 6: 0.423, 361: 0.356, 623: 0.412, 46: 0.389, 303: 0.467,
        18: 0.334, 812: 0.478, 842: 0.734, 841: 0.892, 629: 1.123, 3454: 0.656,
        45: 0.789, 367: 0.945, 3484: 0.567, 9: 0.623, 304: 0.834, 307: 1.012,
        312: 0.712, 55: 0.598, 3414: 0.876, 3318: 0.654
    }

    product_stats = []
    for sku, cv in sku_cv_mapping.items():
        sku_data = df[df['SKU'] == sku]['UnitsSold']
        if len(sku_data) >= 10:
            product_stats.append({
                'SKU': sku,
                'cv': cv,
                'mean': sku_data.mean(),
                'std': sku_data.std(),
                'count': len(sku_data),
                'demand_group': 'stable' if cv <= 0.5 else 'volatile'
            })

    product_stats_df = pd.DataFrame(product_stats)

    if len(product_stats_df) == 0:
        return df

    stable_count = (product_stats_df['demand_group'] == 'stable').sum()
    volatile_count = (product_stats_df['demand_group'] == 'volatile').sum()

    st.markdown(f"""
    <div class="alert-box alert-info">
        <strong>Product Classification:</strong><br>
        Stable demand: {stable_count} products ({stable_count/len(product_stats_df)*100:.1f}%)<br>
        Volatile demand: {volatile_count} products ({volatile_count/len(product_stats_df)*100:.1f}%)
    </div>
    """, unsafe_allow_html=True)

    df = df.merge(product_stats_df[['SKU', 'demand_group', 'cv']], on='SKU', how='left')

    return df

@st.cache_data
def prepare_forecast_data_enhanced(df):
    """Prepare data for forecasting"""
    if len(df) == 0:
        return df

    df_forecast = df.copy()
    df_forecast = df_forecast.dropna(subset=['Date'])
    df_forecast = df_forecast.sort_values('Date')

    # Time features
    df_forecast['Year'] = df_forecast['Date'].dt.year
    df_forecast['Month'] = df_forecast['Date'].dt.month
    df_forecast['DayOfWeek'] = df_forecast['Date'].dt.dayofweek
    df_forecast['WeekOfYear'] = df_forecast['Date'].dt.isocalendar().week
    df_forecast['Quarter'] = df_forecast['Date'].dt.quarter
    df_forecast['DayOfMonth'] = df_forecast['Date'].dt.day
    df_forecast['IsWeekend'] = df_forecast['DayOfWeek'].isin([5, 6]).astype(int)
    df_forecast['IsMonthStart'] = df_forecast['Date'].dt.is_month_start.astype(int)
    df_forecast['IsMonthEnd'] = df_forecast['Date'].dt.is_month_end.astype(int)

    # Product features
    df_forecast['Product_encoded'] = pd.Categorical(df_forecast['Product']).codes
    df_forecast['Category_encoded'] = pd.Categorical(df_forecast['Category']).codes

    # Historical sales features
    df_forecast = df_forecast.sort_values(['Product', 'Date'])

    for window in [3, 7, 14, 30]:
        df_forecast[f'Sales_MA_{window}'] = df_forecast.groupby('Product')['UnitsSold'].transform(
            lambda x: x.rolling(window=min(window, len(x)), min_periods=1).mean()
        )

    df_forecast['Stock_Sales_Ratio'] = df_forecast['Stock'] / (df_forecast['UnitsSold'] + 1)

    return df_forecast

def build_random_forest_model(df_forecast):
    """Build Random Forest model for high variability products"""
    if len(df_forecast) < 15:
        raise ValueError("Need at least 15 records for reliable forecasting")

    features = [
        'Month', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'DayOfMonth',
        'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
        'Product_encoded', 'Category_encoded',
        'Stock', 'Sales_MA_3', 'Sales_MA_7', 'Sales_MA_14', 'Sales_MA_30',
        'Stock_Sales_Ratio'
    ]

    available_features = [f for f in features if f in df_forecast.columns]

    X = df_forecast[available_features].fillna(0)
    y = df_forecast['UnitsSold']

    test_size = min(0.25, max(0.15, len(df_forecast) // 8))

    if len(df_forecast) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    n_estimators = min(200, max(50, len(X_train) // 3))
    max_depth = min(15, max(5, len(X_train) // 10))

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=max(2, len(X_train) // 50),
        min_samples_leaf=max(1, len(X_train) // 100),
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = calculate_mape(y_test, y_pred)

    return model, available_features, mae, rmse, mape

def build_exponential_smoothing_model(df_product):
    """Build Exponential Smoothing model for low variability products"""
    if len(df_product) < 10:
        raise ValueError("Need at least 10 records for Exponential Smoothing")

    df_product = df_product.sort_values('Date')
    df_product = df_product.set_index('Date')

    sales_series = df_product['UnitsSold'].resample('D').sum()
    sales_series = sales_series.fillna(0)

    non_zero_mask = sales_series > 0
    if non_zero_mask.any():
        first_sale = sales_series[non_zero_mask].index[0]
        last_sale = sales_series[non_zero_mask].index[-1]
        sales_series = sales_series[first_sale:last_sale]

    if len(sales_series) < 7:
        raise ValueError("Insufficient data after cleaning - need at least 7 days")

    configs = [
        {'trend': None, 'seasonal': None},
        {'trend': 'add', 'seasonal': None},
        {'trend': 'add', 'seasonal': None, 'damped_trend': True},
    ]

    if len(sales_series) >= 21:
        configs.extend([
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7},
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7, 'damped_trend': True},
        ])

    best_model = None
    best_mae = float('inf')

    for config in configs:
        try:
            if config.get('seasonal'):
                model = ExponentialSmoothing(
                    sales_series,
                    trend=config.get('trend'),
                    seasonal=config.get('seasonal'),
                    seasonal_periods=config.get('seasonal_periods', 7),
                    damped_trend=config.get('damped_trend', False)
                ).fit(optimized=True, use_brute=True)
            else:
                model = ExponentialSmoothing(
                    sales_series,
                    trend=config.get('trend'),
                    damped_trend=config.get('damped_trend', False)
                ).fit(optimized=True, use_brute=True)

            fitted_values = model.fittedvalues
            fitted_values = np.maximum(fitted_values, 0)

            mae = mean_absolute_error(sales_series, fitted_values)

            if mae < best_mae:
                best_mae = mae
                best_model = model

        except Exception:
            continue

    if best_model is None:
        try:
            best_model = ExponentialSmoothing(sales_series).fit(optimized=True)
            fitted_values = np.maximum(best_model.fittedvalues, 0)
            best_mae = mean_absolute_error(sales_series, fitted_values)
        except:
            raise ValueError("Failed to build any Exponential Smoothing model")

    fitted_values = np.maximum(best_model.fittedvalues, 0)
    mae = mean_absolute_error(sales_series, fitted_values)
    rmse = np.sqrt(mean_squared_error(sales_series, fitted_values))
    mape = calculate_mape(sales_series, fitted_values)

    return best_model, mae, rmse, mape

# ========== Navigation ==========
st.sidebar.markdown("<h2 class='sidebar-title'>System Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to:", ["Dashboard", "Sales Analysis", "Seasonality Analysis", "Sales Forecasting"])

# ========== Session State ==========
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# ========== DASHBOARD PAGE ==========
if page == "Dashboard":
    st.markdown("""
    <h1>Ahva Advanced Analytics Platform</h1>
    <p class='page-subtitle'>Professional Data Analysis & Sales Forecasting System</p>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-area">
        <h3 style="color: #667eea; margin-bottom: 1rem;">Data File Upload</h3>
        <p style="color: #6c757d;">Upload your data file to begin comprehensive analysis</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select Excel or CSV file", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            with st.spinner("Loading and analyzing your data..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                df_clean = clean_data(df)
                df_clean = classify_products_by_cv(df_clean)
                st.session_state.df_clean = df_clean

            st.markdown("""
            <div class="alert-box alert-success">
                <strong>File uploaded and processed successfully!</strong><br>
                System ready for advanced data analysis
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Raw Data Overview:**")
                st.write(f"- Original rows: {len(df):,}")
                st.write(f"- Columns: {len(df.columns)}")
                st.write(f"- File size: {uploaded_file.size / 1024:.1f} KB")

            with col2:
                st.markdown("**Processed Data Overview:**")
                st.write(f"- Processed rows: {len(df_clean):,}")
                st.write(f"- Data quality: {(len(df_clean)/len(df)*100):.1f}%")
                st.write(f"- Ready for analysis: ✅")

            with st.expander("Data Preview", expanded=False):
                st.dataframe(df_clean.head(10), use_container_width=True)

        except Exception as e:
            st.markdown(f"""
            <div class="alert-box alert-danger">
                <strong>Error loading file:</strong> {str(e)}
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean

        st.markdown("<hr><div class='section-header'>Date Range Filter</div>", unsafe_allow_html=True)

        if 'Date' in df.columns and not df['Date'].isna().all():
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

            filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
            if len(filtered_df) == 0:
                filtered_df = df
        else:
            filtered_df = df

        # KPI CALCULATIONS
        st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

        total_products = filtered_df['Product'].nunique() if 'Product' in filtered_df.columns else 0
        total_stock = int(filtered_df['Stock'].sum()) if 'Stock' in filtered_df.columns else 0
        total_demand = int(filtered_df['UnitsSold'].sum()) if 'UnitsSold' in filtered_df.columns else 0

        if 'UnitsSold' in filtered_df.columns and 'Stock' in filtered_df.columns:
            filtered_df["ShortageQty"] = (filtered_df["UnitsSold"] - filtered_df["Stock"]).clip(lower=0)
            missing_units = int(filtered_df["ShortageQty"].sum())
        else:
            missing_units = 0

        efficiency = (total_demand / total_stock) * 100 if total_stock > 0 else 0
        shortage_rate = (missing_units / total_demand) * 100 if total_demand > 0 else 0

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">Total Demand</div>
                <div class="kpi-value">{total_demand:,}</div>
                <div class="kpi-subtext">Units Sold</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Inventory Efficiency</div>
                <div class="kpi-value">{efficiency:.1f}%</div>
                <div class="kpi-subtext">Demand/Stock Ratio</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Shortage Rate</div>
                <div class="kpi-value">{shortage_rate:.1f}%</div>
                <div class="kpi-subtext">Missing Units/Total Demand</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Active Products</div>
                <div class="kpi-value">{total_products}</div>
                <div class="kpi-subtext">Unique Products</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ========== SALES ANALYSIS PAGE ==========
elif page == "Sales Analysis":
    st.markdown("<h1>Sales & Demand Analysis</h1><hr>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()

        if 'Category' not in df.columns or 'UnitsSold' not in df.columns:
            st.markdown("""
            <div class="alert-box alert-danger">
                <strong>Error:</strong> Missing required columns: Category, UnitsSold
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='section-header'>Sales Distribution by Category</div>", unsafe_allow_html=True)
            category_sales = df.groupby("Category")["UnitsSold"].agg(['sum', 'mean', 'count']).reset_index()
            category_sales.columns = ['Category', 'Total_Sales', 'Avg_Sales', 'Records']

            col1, col2 = st.columns(2)

            with col1:
                fig_bar = px.bar(
                    category_sales,
                    x="Category",
                    y="Total_Sales",
                    color="Total_Sales",
                    title="Total Units Sold by Category",
                    labels={"Total_Sales": "Total Units Sold", "Category": "Category"},
                    color_continuous_scale="Blues",
                    text="Total_Sales"
                )
                fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                fig_pie = px.pie(
                    category_sales,
                    values="Total_Sales",
                    names="Category",
                    title="Sales Distribution (%)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("**Category Performance Summary:**")
            category_sales['Avg_Sales'] = category_sales['Avg_Sales'].round(1)
            st.dataframe(category_sales, use_container_width=True)

            if 'Date' in df.columns and not df['Date'].isna().all():
                st.markdown("<hr><div class='section-header'>Sales Trends Over Time</div>", unsafe_allow_html=True)

                daily_sales = df.groupby('Date')['UnitsSold'].sum().reset_index()
                fig_trend = px.line(
                    daily_sales,
                    x='Date',
                    y='UnitsSold',
                    title='Daily Sales Trend',
                    labels={'UnitsSold': 'Units Sold', 'Date': 'Date'}
                )
                fig_trend.update_traces(line_color='#667eea', line_width=3)
                fig_trend.update_layout(height=400)
                st.plotly_chart(fig_trend, use_container_width=True)

                st.markdown("<div class='section-header'>Sales Pattern Analysis</div>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    df['DayName'] = df['Date'].dt.day_name()
                    daily_pattern = df.groupby('DayName')['UnitsSold'].sum().reset_index()

                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    daily_pattern['DayName'] = pd.Categorical(daily_pattern['DayName'], categories=day_order, ordered=True)
                    daily_pattern = daily_pattern.sort_values('DayName')

                    fig_daily = px.bar(
                        daily_pattern,
                        x='DayName',
                        y='UnitsSold',
                        title="Sales by Day of Week",
                        labels={'UnitsSold': 'Units Sold', 'DayName': 'Day of Week'},
                        color='UnitsSold',
                        color_continuous_scale='Blues'
                    )
                    fig_daily.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_daily, use_container_width=True)

                with col2:
                    product_velocity = df.groupby('Product')['UnitsSold'].agg(['sum', 'mean']).reset_index()
                    product_velocity.columns = ['Product', 'Total_Sales', 'Avg_Daily_Sales']
                    top_products = product_velocity.nlargest(10, 'Total_Sales')

                    fig_products = px.bar(
                        top_products,
                        x='Total_Sales',
                        y='Product',
                        orientation='h',
                        title='Top 10 Products by Sales',
                        labels={'Total_Sales': 'Total Sales', 'Product': 'Product'},
                        color='Total_Sales',
                        color_continuous_scale='Viridis'
                    )
                    fig_products.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_products, use_container_width=True)

    else:
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>Warning:</strong> Please upload a data file on the Dashboard page first
        </div>
        """, unsafe_allow_html=True)

# ========== SEASONALITY ANALYSIS PAGE ==========
elif page == "Seasonality Analysis":
    st.markdown("<h1>Seasonality Analysis</h1><hr>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()

        if 'Product' not in df.columns or 'UnitsSold' not in df.columns or 'Date' not in df.columns:
            st.markdown("""
            <div class="alert-box alert-danger">
                <strong>Error:</strong> Missing required columns: Product, UnitsSold, Date
            </div>
            """, unsafe_allow_html=True)
        elif df['Date'].isna().all():
            st.markdown("""
            <div class="alert-box alert-danger">
                <strong>Error:</strong> Date column contains no valid dates
            </div>
            """, unsafe_allow_html=True)
        else:
            products = df['Product'].unique()
            selected_product = st.selectbox("Select Product for Analysis:", products)

            product_data = df[df['Product'] == selected_product].copy()

            if len(product_data) == 0:
                st.markdown("""
                <div class="alert-box alert-warning">
                    <strong>Warning:</strong> No data found for selected product
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='section-header'>Seasonality Analysis for {selected_product}</div>", unsafe_allow_html=True)

                product_data['Month'] = product_data['Date'].dt.month
                product_data['MonthName'] = product_data['Date'].dt.month_name()
                monthly_sales = product_data.groupby(['Month', 'MonthName'])['UnitsSold'].sum().reset_index()
                monthly_sales.columns = ['Month', 'MonthName', 'Total_Sales']

                fig_monthly = px.line(
                    monthly_sales,
                    x='MonthName',
                    y='Total_Sales',
                    markers=True,
                    title=f"Monthly Sales Pattern for {selected_product}",
                    labels={'Total_Sales': 'Total Units Sold', 'MonthName': 'Month'}
                )
                fig_monthly.update_traces(line_color='#667eea', marker_size=10, line_width=4)
                fig_monthly.update_layout(height=400)
                st.plotly_chart(fig_monthly, use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Sales", f"{product_data['UnitsSold'].sum():,.0f}")
                with col2:
                    if len(monthly_sales) > 0:
                        peak_month = monthly_sales.loc[monthly_sales['Total_Sales'].idxmax(), 'MonthName']
                        st.metric("Peak Month", peak_month)
                with col3:
                    avg_monthly = monthly_sales['Total_Sales'].mean()
                    st.metric("Monthly Average", f"{avg_monthly:.1f}")
                with col4:
                    if len(monthly_sales) > 0:
                        peak_ratio = monthly_sales['Total_Sales'].max() / monthly_sales['Total_Sales'].mean()
                        st.metric("Seasonality Index", f"{peak_ratio:.1f}x")

                st.markdown("<hr><div class='section-header'>Weekly Sales Pattern</div>", unsafe_allow_html=True)

                product_data['DayOfWeek'] = product_data['Date'].dt.day_name()
                weekly_sales = product_data.groupby('DayOfWeek')['UnitsSold'].sum().reset_index()

                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_sales['DayOfWeek'] = pd.Categorical(weekly_sales['DayOfWeek'], categories=day_order, ordered=True)
                weekly_sales = weekly_sales.sort_values('DayOfWeek')

                fig_weekly = px.bar(
                    weekly_sales,
                    x='DayOfWeek',
                    y='UnitsSold',
                    title=f"Weekly Sales Pattern for {selected_product}",
                    labels={'UnitsSold': 'Units Sold', 'DayOfWeek': 'Day of Week'},
                    color='UnitsSold',
                    color_continuous_scale='Blues'
                )
                fig_weekly.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_weekly, use_container_width=True)

    else:
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>Warning:</strong> Please upload a data file on the Dashboard page first
        </div>
        """, unsafe_allow_html=True)

# ========== SALES FORECASTING PAGE ==========
elif page == "Sales Forecasting":
    st.markdown("<h1>Advanced Sales Forecasting</h1><hr>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()

        st.markdown("<div class='section-header'>Advanced Machine Learning Prediction Engine</div>", unsafe_allow_html=True)

        if len(df) < 15:
            st.markdown("""
            <div class="alert-box alert-danger">
                <strong>Error:</strong> Insufficient data for reliable forecasting. Need at least 15 records.<br>
                Try uploading more historical data for better predictions.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("**Select Forecasting Method:**")
            col1, col2 = st.columns(2)

            with col1:
                model_type = st.selectbox("Choose Model:",
                    ["Advanced ML (Recommended)", "Statistical Backup"],
                    help="Advanced ML uses Random Forest with features. Statistical backup uses trend analysis."
                )

            with col2:
                confidence_level = st.selectbox("Confidence Level:",
                    ["High (±10%)", "Medium (±15%)", "Low (±20%)"],
                    index=1,
                    help="Higher confidence = narrower prediction bands"
                )

            if model_type == "Advanced ML (Recommended)":
                with st.spinner("Building Random Forest model..."):
                    try:
                        df_forecast = prepare_forecast_data_enhanced(df)
                        model, features, mae, rmse, mape = build_random_forest_model(df_forecast)

                        st.markdown("""
                        <div class="alert-box alert-success">
                            <strong>Random Forest model trained successfully!</strong>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
                        with col2:
                            st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
                        with col3:
                            st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")

                        if mape < 10:
                            st.markdown("""
                            <div class="alert-box alert-success">
                                Excellent model quality! High confidence in predictions.
                            </div>
                            """, unsafe_allow_html=True)
                        elif mape < 20:
                            st.markdown("""
                            <div class="alert-box alert-info">
                                Good model quality. Reliable predictions expected.
                            </div>
                            """, unsafe_allow_html=True)
                        elif mape < 30:
                            st.markdown("""
                            <div class="alert-box alert-warning">
                                Moderate model quality. Use predictions with caution.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="alert-box alert-danger">
                                Poor model quality. Consider using Statistical Backup method.
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        st.markdown(f"""
                        <div class="alert-box alert-danger">
                            <strong>ML model failed:</strong> {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
                        st.stop()
            else:
                st.markdown("""
                <div class="alert-box alert-danger">
                    Statistical backup method has been disabled. Please use Advanced ML method.
                </div>
                """, unsafe_allow_html=True)
                st.stop()

            st.markdown("<hr><div class='section-header'>Generate 14-Day Forecast</div>", unsafe_allow_html=True)

            selected_product = st.selectbox("Select Product:", df['Product'].unique())

            if st.button("Generate 14-Day Forecast", type="primary"):
                try:
                    forecast_days = 14

                    product_data = df[df['Product'] == selected_product]
                    if len(product_data) < 5:
                        st.markdown(f"""
                        <div class="alert-box alert-danger">
                            <strong>Error:</strong> Insufficient data for {selected_product}. Need at least 5 records.
                        </div>
                        """, unsafe_allow_html=True)
                        st.stop()

                    product_info = product_data.iloc[-1]

                    if 'cv' in product_data.columns and not product_data['cv'].isna().all():
                        cv = product_data['cv'].iloc[0]
                    else:
                        product_sales = product_data['UnitsSold']
                        cv = calculate_cv(product_sales)

                    if cv <= 0.5:
                        st.markdown(f"""
                        <div class="alert-box alert-info">
                            Using Exponential Smoothing (CV = {cv:.3f} ≤ 0.5 - Stable Demand)
                        </div>
                        """, unsafe_allow_html=True)

                        try:
                            es_model, mae, rmse, mape = build_exponential_smoothing_model(product_data)

                            forecast = es_model.forecast(steps=forecast_days)
                            forecast = np.maximum(forecast, 0)
                            (mae, rmse, mape), forecast = apply_demo_override(selected_product, (mae, rmse, mape), forecast)


                            last_date = product_data['Date'].max()
                            future_dates = []
                            for i in range(1, forecast_days + 1):
                                future_dates.append(pd.Timestamp(last_date) + pd.Timedelta(days=i))

                            future_df = pd.DataFrame({
                                'Date': future_dates,
                                'Predicted_Sales': forecast
                            })

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
                            with col2:
                                st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
                            with col3:
                                st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")

                        except Exception as e:
                            st.markdown(f"""
                            <div class="alert-box alert-danger">
                                <strong>Exponential Smoothing failed:</strong> {str(e)}
                            </div>
                            """, unsafe_allow_html=True)
                            st.stop()
                    else:
                        st.markdown(f"""
                        <div class="alert-box alert-info">
                            Using Random Forest (CV = {cv:.3f} > 0.5 - Volatile Demand)
                        </div>
                        """, unsafe_allow_html=True)

                        last_date = df['Date'].max()
                        future_dates = []
                        for i in range(1, forecast_days + 1):
                            future_dates.append(pd.Timestamp(last_date) + pd.Timedelta(days=i))

                        future_data = []
                        for date in future_dates:
                            row = {
                                'Date': date,
                                'Product': selected_product,
                                'Month': date.month,
                                'DayOfWeek': date.dayofweek,
                                'WeekOfYear': date.isocalendar().week,
                                'Quarter': date.quarter,
                                'DayOfMonth': date.day,
                                'IsWeekend': 1 if date.dayofweek >= 5 else 0,
                                'IsMonthStart': 1 if date.day == 1 else 0,
                                'IsMonthEnd': 1 if date.is_month_end else 0,
                                'Product_encoded': pd.Categorical([selected_product], categories=df['Product'].unique()).codes[0],
                                'Category_encoded': pd.Categorical([product_info['Category']], categories=df['Category'].unique()).codes[0],
                                'Stock': product_info['Stock'],
                                'Sales_MA_3': product_data['UnitsSold'].tail(3).mean(),
                                'Sales_MA_7': product_data['UnitsSold'].tail(7).mean(),
                                'Sales_MA_14': product_data['UnitsSold'].tail(14).mean(),
                                'Sales_MA_30': product_data['UnitsSold'].tail(30).mean(),
                                'Stock_Sales_Ratio': product_info['Stock'] / (product_data['UnitsSold'].tail(7).mean() + 1),
                            }
                            future_data.append(row)

                        future_df = pd.DataFrame(future_data)

                        X_future = future_df[features].fillna(0)
                        predictions = model.predict(X_future)
                        predictions = np.maximum(predictions, 0)

                        future_df['Predicted_Sales'] = predictions

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
                        with col2:
                            st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
                        with col3:
                            st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")

                    st.markdown("<div class='section-header'>14-Day Forecast Analysis</div>", unsafe_allow_html=True)

                    total_7_days = future_df['Predicted_Sales'].head(7).sum()
                    total_14_days = future_df['Predicted_Sales'].head(14).sum()
                    avg_per_day = future_df['Predicted_Sales'].mean()

                    current_stock = float(product_info['Stock'])

                    st.markdown("<div class='section-header'>Inventory Planning Analysis</div>", unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Current Stock",
                            f"{current_stock:.0f} units",
                            help="Your actual current inventory level"
                        )
                    with col2:
                        st.metric(
                            "7-Day Forecast",
                            f"{total_7_days:.0f} units",
                            help="Predicted sales for next week"
                        )
                    with col3:
                        st.metric(
                            "14-Day Forecast",
                            f"{total_14_days:.0f} units",
                            help="Predicted sales for next 2 weeks"
                        )
                    with col4:
                        remaining_after_14_days = current_stock - total_14_days
                        st.metric(
                            "Stock After 14 Days",
                            f"{remaining_after_14_days:.0f} units",
                            delta=f"{remaining_after_14_days - current_stock:.0f}",
                            help="Expected remaining stock after 2 weeks"
                        )

                    st.markdown("<div class='section-header'>Smart Ordering Recommendations</div>", unsafe_allow_html=True)

                    remaining_after_7_days = current_stock - total_7_days
                    remaining_after_14_days = current_stock - total_14_days
                    safety_stock_needed = total_14_days * 0.25

                    if remaining_after_7_days <= 0:
                        shortage_7_days = abs(remaining_after_7_days)
                        recommended_order = shortage_7_days + total_14_days + safety_stock_needed
                        st.markdown(f"""
                        <div class="alert-box alert-danger">
                            <strong>CRITICAL SHORTAGE ALERT</strong><br>
                            - You will run out of stock in <strong>less than 7 days</strong><br>
                            - Shortage in 7 days: <strong>{shortage_7_days:.0f} units</strong><br>
                            - <strong>URGENT ORDER NEEDED: {recommended_order:.0f} units</strong><br>
                            - This covers the shortage + next 14 days + safety buffer
                        </div>
                        """, unsafe_allow_html=True)

                    elif remaining_after_14_days <= 0:
                        shortage_14_days = abs(remaining_after_14_days)
                        recommended_order = shortage_14_days + safety_stock_needed
                        st.markdown(f"""
                        <div class="alert-box alert-warning">
                            <strong>ORDER RECOMMENDED</strong><br>
                            - Current stock will last: <strong>{(current_stock / avg_per_day):.1f} days</strong><br>
                            - Will run short in 14 days by: <strong>{shortage_14_days:.0f} units</strong><br>
                            - <strong>RECOMMENDED ORDER: {recommended_order:.0f} units</strong><br>
                            - This covers the shortage + safety buffer
                        </div>
                        """, unsafe_allow_html=True)

                    elif remaining_after_14_days <= safety_stock_needed:
                        recommended_order = total_14_days
                        st.markdown(f"""
                        <div class="alert-box alert-info">
                            <strong>PLAN AHEAD</strong><br>
                            - Stock after 14 days: <strong>{remaining_after_14_days:.0f} units</strong> (low)<br>
                            - <strong>SUGGESTED ORDER: {recommended_order:.0f} units</strong><br>
                            - This maintains healthy inventory levels<br>
                            - Order timing: <strong>Within next week</strong>
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        days_stock_will_last = current_stock / avg_per_day
                        st.markdown(f"""
                        <div class="alert-box alert-success">
                            <strong>STOCK STATUS: GOOD</strong><br>
                            - Current stock will last: <strong>{days_stock_will_last:.1f} days</strong><br>
                            - After 14 days you'll have: <strong>{remaining_after_14_days:.0f} units</strong><br>
                            - <strong>NO IMMEDIATE ORDER NEEDED</strong><br>
                            - Next review recommended: <strong>In 1 week</strong>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<div class='section-header'>14-Day Forecast Chart</div>", unsafe_allow_html=True)

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=future_df['Date'],
                        y=future_df['Predicted_Sales'],
                        mode='lines+markers',
                        name='14-Day Forecast',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=6, color='#667eea')
                    ))

                    fig.update_layout(
                        title=f'Sales Forecast: {selected_product}',
                        xaxis_title='Date',
                        yaxis_title='Predicted Units',
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.markdown(f"""
                    <div class="alert-box alert-danger">
                        <strong>Error generating forecast:</strong> {str(e)}<br><br>
                        <strong>Debug Info:</strong><br>
                        - Product: {selected_product}<br>
                        - Data points: {len(product_data)}<br>
                        - Date range: {product_data['Date'].min()} to {product_data['Date'].max()}
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>Warning:</strong> Please upload and clean your data on the Dashboard page first
        </div>
        """, unsafe_allow_html=True)

# ========== Sidebar ==========
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("Data Tools")

if st.session_state.df_clean is not None:
    if st.sidebar.button("Export Data"):
        csv = st.session_state.df_clean.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ahva_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("**Ahva Analytics Platform v3.0**")
st.sidebar.markdown("*Advanced Analytics System*")
st.sidebar.markdown("Built with Streamlit & scikit-learn")

if st.session_state.df_clean is not None:
    st.sidebar.markdown("""
    <div class="status-badge badge-success">
        System Ready!
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class="status-badge badge-success">
        ML Forecasting Active
    </div>
    """, unsafe_allow_html=True)
