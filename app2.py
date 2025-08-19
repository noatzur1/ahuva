import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Ahva Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📦"
)

# CSS Styling
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .kpi-container {
        display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 20px; border-radius: 10px; text-align: center;
        flex: 1; min-width: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(0,0,0,0.2); }
    .kpi-blue { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .kpi-green { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); }
    .kpi-red { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .kpi-purple { background: linear-gradient(135deg, #8360c3 0%, #2ebf91 100%); }
    .kpi-orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .kpi-title { font-size: 14px; margin-bottom: 10px; opacity: 0.9; font-weight: 500; }
    .kpi-value { font-size: 28px; font-weight: bold; margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); }
    .kpi-subtext { font-size: 12px; opacity: 0.8; }
    .sidebar-title { color: #2e4057; margin-bottom: 20px; font-weight: bold; text-align: center; }
    h1, h2, h3 { color: #2e4057; font-weight: 600; }
    hr { margin: 1rem 0; border: none; height: 2px; background: linear-gradient(90deg, #667eea, #764ba2); }
    .forecast-highlight {
        background: #f8f9fa; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #28a745; margin: 1rem 0;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem; border-radius: 8px; margin: 1rem 0;
        color: #2d3436; font-weight: 500;
    }
    @media (max-width: 768px) {
        .kpi-container { flex-direction: column; }
        .kpi-card { min-width: 100%; }
        .kpi-value { font-size: 24px; }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Data Cleaning Functions
@st.cache_data
def clean_data(df):
    """Enhanced data cleaning and preprocessing"""
    df_clean = df.copy()
    
    # Column name mapping from Hebrew to English
    column_mapping = {
        'תאריך': 'Date',
        'מק"ט': 'SKU',
        'תיאור מוצר': 'Product',
        'קטגוריה': 'Category',
        'כמות במלאי': 'Stock',
        'כמות שנמכרה': 'UnitsSold',
        'מהירות חידוש מלאי (ימים)': 'InventoryTurnover',
        'יום בשבוע': 'DayOfWeek',
        'חודש': 'Month',
        'שבוע בשנה': 'WeekOfYear',
        'עלות ליחידה (₪)': 'CostPerUnit',
        'מחיר ליחידה (₪)': 'PricePerUnit',
        'משקל יחידה (גרם)': 'WeightPerUnit'
    }
    
    # Rename columns if they exist
    df_clean = df_clean.rename(columns=column_mapping)
    
    # Enhanced date handling
    if 'Date' in df_clean.columns:
        def convert_date(date_val):
            if pd.isna(date_val):
                return pd.NaT
            
            # Handle Excel serial numbers
            if isinstance(date_val, (int, float)) and not pd.isna(date_val):
                try:
                    if 1 <= date_val <= 100000:
                        base_date = pd.to_datetime('1899-12-30')
                        return base_date + pd.Timedelta(days=int(date_val))
                except:
                    pass
            
            # Regular conversion
            try:
                converted = pd.to_datetime(date_val, errors='coerce')
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
    
    # Category standardization
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
    
    # Handle missing critical columns
    critical_columns = [col for col in ['Product', 'Category', 'UnitsSold', 'Stock'] if col in df_clean.columns]
    before_cleaning = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_columns)
    after_cleaning = len(df_clean)
    
    if before_cleaning != after_cleaning:
        st.info(f"Data Cleaning: Removed {before_cleaning - after_cleaning} rows with missing critical data")
    
    # Clean numeric columns
    numeric_columns = ['UnitsSold', 'Stock', 'CostPerUnit', 'PricePerUnit', 'WeightPerUnit', 'InventoryTurnover']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                df_clean[col] = df_clean[col].abs()
                
            # Handle extreme outliers
            if col in ['UnitsSold', 'Stock']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 3 * IQR
                extreme_high = df_clean[col] > upper_bound
                if extreme_high.sum() > 0:
                    df_clean.loc[extreme_high, col] = upper_bound
    
    if 'Category' in df_clean.columns:
        unique_categories = sorted(df_clean['Category'].unique())
        st.success(f"Categories standardized: {', '.join(unique_categories)}")
    
    return df_clean

@st.cache_data
def prepare_forecast_data_enhanced(df):
    """Enhanced data preparation for forecasting"""
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
    
    # Price and weight features
    df_forecast['PricePerUnit'] = pd.to_numeric(df_forecast.get('PricePerUnit', 0), errors='coerce').fillna(0)
    df_forecast['WeightPerUnit'] = pd.to_numeric(df_forecast.get('WeightPerUnit', 0), errors='coerce').fillna(0)
    
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

def classify_products_by_cv(df):
    """Classify products by coefficient of variation"""
    print("Calculating coefficient of variation (CV) for each SKU...")
    
    # Calculate statistics per SKU
    product_stats = df.groupby('Product')['UnitsSold'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Calculate CV
    product_stats['cv'] = product_stats['std'] / (product_stats['mean'] + 1e-8)
    
    # Filter products with sufficient observations
    product_stats = product_stats[product_stats['count'] >= 10].reset_index(drop=True)
    
    print(f"Products with sufficient data: {len(product_stats)} out of {df['Product'].nunique()}")
    
    # CV threshold: 0.5
    cv_threshold = 0.5
    print(f"CV threshold: {cv_threshold}")
    
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
    
    return product_stats

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    if mask.sum() > 0:
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        return float('inf')

def build_exponential_smoothing_model(product_data):
    """Build Exponential Smoothing model for stable demand products"""
    sales_series = product_data.set_index('Date')['UnitsSold']
    
    try:
        model = ExponentialSmoothing(sales_series, trend=None, seasonal=None)
        fit = model.fit()
        return fit
    except:
        # Fallback to simple average
        return sales_series.mean()

def build_random_forest_model(df_forecast):
    """Build Random Forest model for volatile demand products"""
    if len(df_forecast) < 15:
        raise ValueError("Need at least 15 records for reliable forecasting")
    
    features = [
        'Month', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'DayOfMonth',
        'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
        'Product_encoded', 'Category_encoded', 
        'Stock', 'PricePerUnit', 'WeightPerUnit',
        'Sales_MA_3', 'Sales_MA_7', 'Sales_MA_14', 'Sales_MA_30',
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

# Navigation
st.sidebar.markdown("<h2 class='sidebar-title'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to:", ["HOME", "Analysis", "Seasonality", "Forecasting"])

# Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# HOME PAGE
if page == "HOME":
    st.markdown("""
    <h1 style='margin-bottom: 10px; text-align: center;'>📦 Ahva Inventory Dashboard</h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>Advanced Analytics & Sales Forecasting Platform</p>
    <hr>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"], help="Upload your Ahva sales data file")

    if uploaded_file is not None:
        try:
            with st.spinner("Loading and analyzing your data..."):
                df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                df_clean = clean_data(df)
                st.session_state.df_clean = df_clean
            
            st.success("File uploaded and processed successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Raw Data Overview:**")
                st.write(f"- Original rows: {len(df):,}")
                st.write(f"- Columns: {len(df.columns)}")
                st.write(f"- File size: {uploaded_file.size / 1024:.1f} KB")
                
            with col2:
                st.write("**Cleaned Data Overview:**")
                st.write(f"- Processed rows: {len(df_clean):,}")
                st.write(f"- Data quality: {(len(df_clean)/len(df)*100):.1f}%")
                st.write(f"- Ready for analysis: ✅")
            
            with st.expander("Preview Your Data", expanded=False):
                st.dataframe(df_clean.head(10))
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean

        st.markdown("---")
        st.subheader("Date Range Filter")
        
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
        st.markdown("---")
        st.subheader("Key Performance Indicators")
        
        total_products = filtered_df['Product'].nunique() if 'Product' in filtered_df.columns else 0
        total_stock = int(filtered_df['Stock'].sum()) if 'Stock' in filtered_df.columns else 0
        total_demand = int(filtered_df['UnitsSold'].sum()) if 'UnitsSold' in filtered_df.columns else 0
        
        if 'UnitsSold' in filtered_df.columns and 'Stock' in filtered_df.columns:
            shortages = (filtered_df['UnitsSold'] > filtered_df['Stock']).sum()
            filtered_df["ShortageQty"] = (filtered_df["UnitsSold"] - filtered_df["Stock"]).clip(lower=0)
            missing_units = int(filtered_df["ShortageQty"].sum())
        else:
            shortages = 0
            missing_units = 0

        efficiency = (total_demand / total_stock) * 100 if total_stock > 0 else 0
        shortage_rate = (missing_units / total_demand) * 100 if total_demand > 0 else 0

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card kpi-purple">
                <div class="kpi-title">Total Demand</div>
                <div class="kpi-value">{total_demand:,}</div>
                <div class="kpi-subtext">Units Sold</div>
            </div>
            <div class="kpi-card kpi-orange">
                <div class="kpi-title">Efficiency</div>
                <div class="kpi-value">{efficiency:.1f}%</div>
                <div class="kpi-subtext">Demand/Stock Ratio</div>
            </div>
            <div class="kpi-card kpi-red">
                <div class="kpi-title">Shortage Rate</div>
                <div class="kpi-value">{shortage_rate:.1f}%</div>
                <div class="kpi-subtext">Missing Units / Total Demand</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ANALYSIS PAGE
elif page == "Analysis":
    st.markdown("<h1>Sales & Demand Analysis</h1><hr>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()

        if 'Category' not in df.columns or 'UnitsSold' not in df.columns:
            st.error("Missing required columns: Category, UnitsSold")
        else:
            # Sales by Category
            st.subheader("Sales Distribution by Category")
            category_sales = df.groupby("Category")["UnitsSold"].agg(['sum', 'mean', 'count']).reset_index()
            category_sales.columns = ['Category', 'Total_Sales', 'Avg_Sales', 'Records']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bar = px.bar(
                    category_sales, 
                    x="Category", 
                    y="Total_Sales",
                    color="Total_Sales",
                    title="Total Units Sold per Category",
                    labels={"Total_Sales": "Total Units Sold"},
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
            
            st.write("**Category Performance Summary:**")
            category_sales['Avg_Sales'] = category_sales['Avg_Sales'].round(1)
            st.dataframe(category_sales, use_container_width=True)

            # Time-based analysis
            if 'Date' in df.columns and not df['Date'].isna().all():
                st.markdown("---")
                st.subheader("Sales Trends Over Time")
                
                daily_sales = df.groupby('Date')['UnitsSold'].sum().reset_index()
                fig_trend = px.line(
                    daily_sales, 
                    x='Date', 
                    y='UnitsSold',
                    title='Daily Sales Trend',
                    labels={'UnitsSold': 'Units Sold'}
                )
                fig_trend.update_traces(line_color='#1f77b4', line_width=3)
                fig_trend.update_layout(height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Sales Pattern Analysis")
                
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
                        color='Total_Sales',
                        color_continuous_scale='Viridis'
                    )
                    fig_products.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_products, use_container_width=True)

    else:
        st.warning("Please upload a file in the HOME page first.")

# SEASONALITY PAGE
elif page == "Seasonality":
    st.markdown("<h1>Seasonality Analysis</h1><hr>", unsafe_allow_html=True)

    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()
        
        if 'Product' not in df.columns or 'UnitsSold' not in df.columns or 'Date' not in df.columns:
            st.error("Missing required columns: Product, UnitsSold, Date")
        elif df['Date'].isna().all():
            st.error("Date column contains no valid dates")
        else:
            products = df['Product'].unique()
            selected_product = st.selectbox("Select Product for Analysis:", products)
            
            product_data = df[df['Product'] == selected_product].copy()
            
            if len(product_data) == 0:
                st.warning("No data found for selected product.")
            else:
                st.subheader(f"Seasonality Analysis for {selected_product}")
                
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
                fig_monthly.update_traces(line_color='#1e88e5', marker_size=10, line_width=4)
                fig_monthly.update_layout(height=400)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                col
