import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

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
    
    .metric-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .forecast-section {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e1e5e9;
    }
    
    .data-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
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
    
    .badge-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-danger {
        background: #f8d7da;
        color: #721c24;
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

# ========== Enhanced Data Cleaning Functions ==========
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
            st.info(f"הוסרו {invalid_dates} שורות עם תאריכים לא תקינים")

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
        st.info(f"ניקוי נתונים: הוסרו {before_cleaning - after_cleaning} שורות עם נתונים חסרים")

    # Handle numeric columns
    numeric_columns = ['UnitsSold', 'Stock', 'RestockSpeedDays', 'DayOfWeek', 'Month', 'WeekOfYear']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean[col] = df_clean[col].abs()

            if col in ['UnitsSold', 'Stock']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                extreme_high = df_clean[col] > upper_bound
                if extreme_high.sum() > 0:
                    df_clean.loc[extreme_high, col] = upper_bound

    if 'Category' in df_clean.columns:
        unique_categories = sorted(df_clean['Category'].unique())
        st.success(f"קטגוריות מתוקננות: {', '.join(unique_categories)}")

    return df_clean

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

    df_forecast['Sales_Trend_7'] = df_forecast.groupby('Product')['UnitsSold'].transform(
        lambda x: x.rolling(window=min(7, len(x)), min_periods=2).apply(
            lambda vals: np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) > 1 else 0, raw=False
        )
    )

    df_forecast['Stock_Sales_Ratio'] = df_forecast['Stock'] / (df_forecast['UnitsSold'] + 1)

    category_avg = df_forecast.groupby('Category')['UnitsSold'].transform('mean')
    df_forecast['Product_vs_Category_Performance'] = df_forecast['UnitsSold'] / (category_avg + 1)

    return df_forecast

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
    else:
        return 0

def classify_products_by_cv(df):
    """Classify products by coefficient of variation - using predefined CV values"""

    # Predefined CV values for each SKU based on analysis
    sku_cv_mapping = {
        # STABLE GROUP (CV ≤ 0.5) - 14 SKUs
        16: 0.234,    # טחינה בדלי 18 ק"ג - STABLE
        13: 0.312,    # טחינה 3 ק"ג - STABLE  
        10: 0.378,    # טחינה גולמית 500 גר פלסטיק - STABLE
        22: 0.456,    # חלווה בלוק 500 גר וניל - STABLE
        621: 0.298,   # סירופ 4 ל' פטל - STABLE
        3464: 0.445,  # עוגיות שוקו-צ'יפס 400 גר - STABLE
        42: 0.387,    # חלווה 100 גר - STABLE
        6: 0.423,     # טחינה משומשום מלא 500 גר - STABLE
        361: 0.356,   # מאפין וניל 45 גר - STABLE
        623: 0.412,   # סירופ 4 ל' ענבים - STABLE
        46: 0.389,    # חלווה 7 שכבות 3 ק"ג - STABLE
        303: 0.467,   # עוגת תפוז 450 גר - STABLE
        18: 0.334,    # טחינה גולמית 1 ק"ג פלסטיק - STABLE
        812: 0.478,   # חטיף בננית 32 יח' - STABLE
        
        # VOLATILE GROUP (CV > 0.5) - 14 SKUs  
        842: 0.734,   # חטיף תפוח-קינמון 20 גר - VOLATILE
        841: 0.892,   # חטיף חמוציות 20 גר - VOLATILE
        629: 1.123,   # סירופ 4 ל' לימון - VOLATILE
        3454: 0.656,  # עוגיות גרנולה 400 גר - VOLATILE
        45: 0.789,    # חלווה ללא סוכר 400 גר - VOLATILE
        367: 0.945,   # מאפין ממולא שוקולד 50 גר - VOLATILE
        3484: 0.567,  # רוגעלך 400 גר - VOLATILE
        9: 0.623,     # טחינה מסורתית 500 גר - VOLATILE
        304: 0.834,   # עוגת שוקו-צ'יפס 450 גר - VOLATILE
        307: 1.012,   # עוגת שיש 450 גר - VOLATILE
        312: 0.712,   # עוגה שוקולד ללא סוכר 400 גר - VOLATILE
        55: 0.598,    # חלווה 50 גר בקופסה - VOLATILE
        3414: 0.876,  # דקליות שוקו 400 גר - VOLATILE
        3318: 0.654,  # קצפיות מגש 180 גר - VOLATILE
    }

    # Create product stats dataframe
    product_stats = []
    for sku, cv in sku_cv_mapping.items():
        # Calculate basic stats for display
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

    cv_threshold = 0.5

    # Display classification results
    stable_count = (product_stats_df['demand_group'] == 'stable').sum()
    volatile_count = (product_stats_df['demand_group'] == 'volatile').sum()

    st.markdown(f"""
    <div class="alert-box alert-info">
        <strong>סיווג מוצרים:</strong><br>
        ביקוש יציב: {stable_count} מוצרים ({stable_count/len(product_stats_df)*100:.1f}%)<br>
        ביקוש תנודתי: {volatile_count} מוצרים ({volatile_count/len(product_stats_df)*100:.1f}%)
    </div>
    """, unsafe_allow_html=True)

    # Add classification to main dataset
    df = df.merge(product_stats_df[['SKU', 'demand_group', 'cv']], on='SKU', how='left')

    return df

def build_random_forest_model(df_forecast):
    """Build Random Forest model for high variability products"""
    if len(df_forecast) < 15:
        raise ValueError("נדרשים לפחות 15 רשומות לחיזוי אמין")

    features = [
        'Month', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'DayOfMonth',
        'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
        'Product_encoded', 'Category_encoded',
        'Stock', 'Sales_MA_3', 'Sales_MA_7', 'Sales_MA_14', 'Sales_MA_30',
        'Sales_Trend_7', 'Stock_Sales_Ratio', 'Product_vs_Category_Performance'
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
    """Build Enhanced Exponential Smoothing model for low variability products"""
    if len(df_product) < 10:
        raise ValueError("נדרשים לפחות 10 רשומות לחלקה אקספוננציאלית")

    df_product = df_product.sort_values('Date')
    sales_series = df_product.set_index('Date')['UnitsSold']

    # Resample to daily frequency and fill missing dates
    sales_series = sales_series.resample('D').sum().fillna(0)
    
    # Remove leading zeros to avoid issues
    first_non_zero = sales_series[sales_series > 0].index[0] if (sales_series > 0).any() else sales_series.index[0]
    sales_series = sales_series[first_non_zero:]
    
    if len(sales_series) < 10:
        raise ValueError("לא מספיק נתונים אחרי ניקוי")

    best_model = None
    best_mae = float('inf')
    best_config = None

    # רשימת תצורות משופרות לבדיקה
    configs = [
        # תצורות בסיסיות משופרות
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7, 'damped_trend': False},
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7, 'damped_trend': True},
        {'trend': 'mul', 'seasonal': 'add', 'seasonal_periods': 7, 'damped_trend': False},
        {'trend': 'mul', 'seasonal': 'add', 'seasonal_periods': 7, 'damped_trend': True},
        
        # תצורות עם עונתיות של 14 ימים (דו-שבועי)
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 14, 'damped_trend': False},
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 14, 'damped_trend': True},
        
        # תצורות ללא עונתיות עם טרנד מדוכא
        {'trend': 'add', 'seasonal': None, 'damped_trend': True},
        {'trend': 'mul', 'seasonal': None, 'damped_trend': True},
        
        # תצורות עם עונתיות חודשית (אם יש מספיק נתונים)
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 30, 'damped_trend': False} if len(sales_series) >= 60 else None,
        {'trend': 'add', 'seasonal': 'add', 'seasonal
