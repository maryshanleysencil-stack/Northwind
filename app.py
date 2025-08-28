import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# --- Configuration for Streamlit Page ---
st.set_page_config(
    page_title="North Wind Traders - E-commerce BI Dashboard",
    page_icon="📈",
    layout="wide",
)

# --- Function to load and prepare data ---
@st.cache_data
def load_data(file_path):
    """
    Loads the CSV data and performs necessary cleaning and type conversions.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert date columns to datetime objects
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df['ShippedDate'] = pd.to_datetime(df['ShippedDate'], errors='coerce')
        
        # Drop rows where 'OrderDate' is NaT
        df.dropna(subset=['OrderDate'], inplace=True)
        
        # Sort by OrderDate for time-series analysis
        df.sort_values(by='OrderDate', inplace=True)
        
        # Create additional derived columns for analysis
        df['Year'] = df['OrderDate'].dt.year
        df['Month'] = df['OrderDate'].dt.month
        df['YearMonth'] = df['OrderDate'].dt.to_period('M')
        df['Quarter'] = df['OrderDate'].dt.quarter
        
        # Calculate delivery performance metrics
        df['ActualDeliveryTime'] = (df['ShippedDate'] - df['OrderDate']).dt.days
        df['DeliveryOnTime'] = df['DeliveryOnTime'].fillna(0)
        
        # Create customer full name for better visualization
        df['CustomerFullName'] = df['CustomerFirstName'] + ' ' + df['CustomerLastName']
        df['EmployeeFullName'] = df['EmployeeFirstName'] + ' ' + df['EmployeeLastName']
        
        # Calculate RFM components
        current_date = df['OrderDate'].max()
        df['Recency'] = (current_date - df['OrderDate']).dt.days
        
        return df
        
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# --- RFM Analysis Function ---
@st.cache_data
def calculate_rfm(df):
    """Calculate RFM (Recency, Frequency, Monetary) analysis for customer segmentation"""
    current_date = df['OrderDate'].max()
    
    rfm = df.groupby('CustomerEmail').agg({
        'OrderDate': lambda x: (current_date - x.max()).days,  # Recency
        'OrderID': 'count',  # Frequency
        'OrderTotal': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerEmail', 'Recency', 'Frequency', 'Monetary']
    
    # Create RFM scores (1-5 scale)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    
    # Combine RFM scores
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Customer segmentation based on RFM
    def segment_customers(row):
        if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif row['RFM_Score'] in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
            return 'Potential Loyalists'
        elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
            return 'New Customers'
        elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
            return 'Cannot Lose Them'
        elif row['RFM_Score'] in ['155', '254', '245']:
            return 'At Risk'
        else:
            return 'Others'
    
    rfm['Segment'] = rfm.apply(segment_customers, axis=1)
    return rfm

# --- Churn Prediction Function ---
@st.cache_data
def predict_churn(df):
    """Simple churn prediction model"""
    try:
        # Create features for churn prediction
        current_date = df['OrderDate'].max()
        customer_features = df.groupby('CustomerEmail').agg({
            'OrderDate': lambda x: (current_date - x.max()).days,  # Days since last order
            'OrderID': 'count',  # Total orders
            'OrderTotal': ['sum', 'mean'],  # Total and average spend
            'TotalLifetimeSpend': 'first',
            'PurchaseCount': 'first'
        }).reset_index()
        
        customer_features.columns = ['CustomerEmail', 'DaysSinceLastOrder', 'TotalOrders', 
                                   'TotalSpend', 'AvgOrderValue', 'LifetimeSpend', 'PurchaseCount']
        
        # Define churn (customers who haven't ordered in 90+ days)
        customer_features['Churned'] = (customer_features['DaysSinceLastOrder'] > 90).astype(int)
        
        # Prepare features for ML
        feature_cols = ['TotalOrders', 'TotalSpend', 'AvgOrderValue', 'DaysSinceLastOrder']
        X = customer_features[feature_cols].fillna(0)
        y = customer_features['Churned']
        
        if len(X) > 10 and y.sum() > 0:  # Ensure we have enough data
            # Train simple model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_scaled, y)
            
            # Predict churn probability
            churn_proba = model.predict_proba(X_scaled)[:, 1]
            customer_features['ChurnProbability'] = churn_proba
            
            return customer_features
        else:
            customer_features['ChurnProbability'] = 0
            return customer_features
    except:
        return None

# Load the data
file_name = 'ecommerce_data_flat_optimized.csv'
data = load_data(file_name)

# --- Main Dashboard Title ---
st.title("🛍️ North Wind Traders - E-commerce Analytics Dashboard")
st.markdown("*Prepared by Mary Shanley Sencil*")
st.markdown("*Comprehensive Business Intelligence Solution*")
st.markdown("---")

if data is not None:
    # --- Sidebar Filters ---
    st.sidebar.header("📊 Filters")
    
    # Date range filter
    min_date = data['OrderDate'].min().date()
    max_date = data['OrderDate'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Region filter
    regions = ['All'] + sorted(data['CustomerRegion'].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region:", regions)
    
    # Status filter
    statuses = ['All'] + sorted(data['OrderStatusName'].unique().tolist())
    selected_status = st.sidebar.selectbox("Select Order Status:", statuses)
    
    # Filter data based on selections
    filtered_data = data[
        (data['OrderDate'].dt.date >= start_date) & 
        (data['OrderDate'].dt.date <= end_date)
    ]
    
    if selected_region != 'All':
        filtered_data = filtered_data[filtered_data['CustomerRegion'] == selected_region]
    
    if selected_status != 'All':
        filtered_data = filtered_data[filtered_data['OrderStatusName'] == selected_status]

    # --- KPI Section (Enhanced) ---
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_revenue = filtered_data['OrderTotal'].sum()
    total_orders = filtered_data['OrderID'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    total_customers = filtered_data['CustomerEmail'].nunique()
    avg_delivery_time = filtered_data['ActualDeliveryTime'].mean()
    
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    col4.metric("Total Customers", f"{total_customers:,}")
    col5.metric("Avg Delivery Time", f"{avg_delivery_time:.1f} days" if not pd.isna(avg_delivery_time) else "N/A")

    st.markdown("---")

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sales Performance & Revenue Growth", 
        "👥 Customer Analytics & Retention", 
        "📦 Order & Fulfillment Analysis",
        "💰 Financial Health & Profitability",
        "🔍 Advanced Analytics"
    ])

    # --- TAB 1: Sales Performance & Revenue Growth ---
    with tab1:
        st.header("📈 Sales Performance & Revenue Growth Dashboard")
        st.markdown("Monitor sales trends, identify top performers, and forecast future revenue.")

        # Revenue trends over time
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Monthly revenue trends
            monthly_revenue = filtered_data.groupby('YearMonth').agg({
                'OrderTotal': 'sum',
                'OrderID': 'count'
            }).reset_index()
            monthly_revenue['YearMonth_str'] = monthly_revenue['YearMonth'].astype(str)
            
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_trend.add_trace(
                go.Scatter(x=monthly_revenue['YearMonth_str'], y=monthly_revenue['OrderTotal'],
                          mode='lines+markers', name='Revenue', line=dict(color='#1f77b4')),
                secondary_y=False,
            )
            
            fig_trend.add_trace(
                go.Scatter(x=monthly_revenue['YearMonth_str'], y=monthly_revenue['OrderID'],
                          mode='lines+markers', name='Orders', line=dict(color='#ff7f0e')),
                secondary_y=True,
            )
            
            fig_trend.update_xaxes(title="Month")
            fig_trend.update_yaxes(title_text="Revenue ($)", secondary_y=False)
            fig_trend.update_yaxes(title_text="Number of Orders", secondary_y=True)
            fig_trend.update_layout(title="Monthly Revenue and Order Trends", height=400)
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Growth metrics
            if len(monthly_revenue) >= 2:
                current_month_revenue = monthly_revenue['OrderTotal'].iloc[-1]
                previous_month_revenue = monthly_revenue['OrderTotal'].iloc[-2]
                revenue_growth = ((current_month_revenue - previous_month_revenue) / previous_month_revenue * 100)
                
                st.metric(
                    "Revenue Growth (MoM)", 
                    f"{revenue_growth:+.1f}%",
                    delta=f"${current_month_revenue - previous_month_revenue:,.0f}"
                )
            
            # Top performing metrics
            top_customer = filtered_data.groupby('CustomerEmail')['OrderTotal'].sum().idxmax()
            top_customer_revenue = filtered_data.groupby('CustomerEmail')['OrderTotal'].sum().max()
            
            st.metric("Top Customer Revenue", f"${top_customer_revenue:,.2f}")
            st.caption(f"Customer: {top_customer.split('@')[0]}")

        # Performance analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by Region
            region_performance = filtered_data.groupby('CustomerRegion').agg({
                'OrderTotal': 'sum',
                'OrderID': 'count'
            }).reset_index().sort_values('OrderTotal', ascending=False)
            
            fig_region = px.bar(
                region_performance,
                x='CustomerRegion',
                y='OrderTotal',
                title='Sales Performance by Region',
                labels={'CustomerRegion': 'Region', 'OrderTotal': 'Total Revenue ($)'},
                text='OrderTotal',
                color='OrderTotal',
                color_continuous_scale='Blues'
            )
            fig_region.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_region.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_region, use_container_width=True)

        with col2:
            # Top Employees by Sales
            employee_performance = filtered_data.groupby('EmployeeFullName').agg({
                'OrderTotal': 'sum',
                'OrderID': 'count'
            }).reset_index().sort_values('OrderTotal', ascending=False).head(10)
            
            fig_employee = px.bar(
                employee_performance,
                x='EmployeeFullName',
                y='OrderTotal',
                title='Top 10 Employees by Sales',
                labels={'EmployeeFullName': 'Employee', 'OrderTotal': 'Total Sales ($)'},
                text='OrderTotal',
                color='OrderTotal',
                color_continuous_scale='Greens'
            )
            fig_employee.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_employee.update_layout(showlegend=False, height=400)
            fig_employee.update_xaxes(tickangle=45)
            st.plotly_chart(fig_employee, use_container_width=True)

        # Revenue Forecasting
        st.subheader("🔮 Revenue Forecasting")
        
        if len(monthly_revenue) >= 6:
            # Prepare data for forecasting
            monthly_revenue_sorted = monthly_revenue.sort_values('YearMonth')
            X = np.arange(len(monthly_revenue_sorted)).reshape(-1, 1)
            y = monthly_revenue_sorted['OrderTotal'].values
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast next 6 months
            future_X = np.arange(len(monthly_revenue_sorted), len(monthly_revenue_sorted) + 6).reshape(-1, 1)
            forecast = model.predict(future_X)
            
            # Create forecast visualization
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=list(range(len(y))),
                y=y,
                mode='lines+markers',
                name='Historical Revenue',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=list(range(len(y), len(y) + 6)),
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig_forecast.update_layout(
                title="6-Month Revenue Forecast",
                xaxis_title="Month Index",
                yaxis_title="Revenue ($)",
                height=400
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Forecasted 6-Month Total", f"${forecast.sum():,.2f}")
            with col2:
                st.metric("Average Monthly Forecast", f"${forecast.mean():,.2f}")
        else:
            st.warning("Need at least 6 months of data for reliable forecasting.")

    # --- TAB 2: Customer Analytics & Retention ---
    with tab2:
        st.header("👥 Customer Analytics & Retention Dashboard")
        st.markdown("Analyze customer behavior, segments, and predict churn risk.")

        # Calculate RFM analysis
        rfm_data = calculate_rfm(filtered_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Customer segments pie chart
            segment_counts = rfm_data['Segment'].value_counts()
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title='Customer Segmentation (RFM Analysis)',
                hole=0.4
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            # Customer lifetime value distribution
            fig_clv = px.histogram(
                filtered_data[['CustomerEmail', 'TotalLifetimeSpend']].drop_duplicates(),
                x='TotalLifetimeSpend',
                nbins=30,
                title='Customer Lifetime Value Distribution',
                labels={'TotalLifetimeSpend': 'Lifetime Spend ($)'}
            )
            st.plotly_chart(fig_clv, use_container_width=True)
        
        with col3:
            # Purchase frequency distribution
            fig_freq = px.histogram(
                filtered_data[['CustomerEmail', 'PurchaseCount']].drop_duplicates(),
                x='PurchaseCount',
                nbins=20,
                title='Purchase Frequency Distribution',
                labels={'PurchaseCount': 'Number of Orders'}
            )
            st.plotly_chart(fig_freq, use_container_width=True)

        # Churn Analysis
        st.subheader("🚨 Churn Risk Analysis")
        churn_data = predict_churn(filtered_data)
        
        if churn_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # High-risk customers
                high_risk = churn_data[churn_data['ChurnProbability'] > 0.7].sort_values('ChurnProbability', ascending=False)
                if len(high_risk) > 0:
                    st.subheader("🔴 High Churn Risk Customers")
                    for i, row in high_risk.head(10).iterrows():
                        st.write(f"• {row['CustomerEmail'].split('@')[0]} - Risk: {row['ChurnProbability']:.1%}")
                else:
                    st.success("No high-risk customers identified!")
            
            with col2:
                # Churn risk distribution
                fig_churn = px.histogram(
                    churn_data,
                    x='ChurnProbability',
                    nbins=20,
                    title='Churn Risk Distribution',
                    labels={'ChurnProbability': 'Churn Probability'}
                )
                st.plotly_chart(fig_churn, use_container_width=True)

        # Customer cohort analysis
        st.subheader("📊 Customer Retention Cohort Analysis")
        
        # Create cohort analysis
        def create_cohort_analysis(df):
            df['OrderMonth'] = df['OrderDate'].dt.to_period('M')
            df['CohortGroup'] = df.groupby('CustomerEmail')['OrderDate'].transform('min').dt.to_period('M')
            
            df_cohort = df.groupby(['CohortGroup', 'OrderMonth'])['CustomerEmail'].nunique().reset_index()
            df_cohort['PeriodNumber'] = (df_cohort['OrderMonth'] - df_cohort['CohortGroup']).apply(attrgetter('n'))
            
            cohort_table = df_cohort.pivot(index='CohortGroup', columns='PeriodNumber', values='CustomerEmail')
            cohort_sizes = df.groupby('CohortGroup')['CustomerEmail'].nunique()
            cohort_table = cohort_table.divide(cohort_sizes, axis=0)
            
            return cohort_table
        
        try:
            from operator import attrgetter
            cohort_table = create_cohort_analysis(filtered_data)
            
            if not cohort_table.empty:
                fig_cohort = px.imshow(
                    cohort_table.values,
                    labels=dict(x="Period Number", y="Cohort Group", color="Retention Rate"),
                    x=[f"Month {i}" for i in range(cohort_table.shape[1])],
                    y=[str(idx) for idx in cohort_table.index],
                    title="Customer Retention Cohort Analysis",
                    color_continuous_scale="RdYlBu_r"
                )
                st.plotly_chart(fig_cohort, use_container_width=True)
        except Exception as e:
            st.info("Cohort analysis requires more diverse temporal data.")

    # --- TAB 3: Order & Fulfillment Analysis ---
    with tab3:
        st.header("📦 Order & Fulfillment Analysis Dashboard")
        st.markdown("Monitor order processing, delivery performance, and operational efficiency.")

        col1, col2 = st.columns(2)
        
        with col1:
            # Order status distribution
            status_dist = filtered_data['OrderStatusName'].value_counts()
            fig_status = px.pie(
                values=status_dist.values,
                names=status_dist.index,
                title='Order Status Distribution',
                hole=0.4
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        with col2:
            # Delivery performance
            delivery_performance = filtered_data.groupby('DeliveryOnTime').size()
            delivery_labels = {0: 'Late/Pending', 1: 'On Time'}
            
            fig_delivery = px.pie(
                values=delivery_performance.values,
                names=[delivery_labels.get(idx, f'Status {idx}') for idx in delivery_performance.index],
                title='Delivery Performance',
                hole=0.4,
                color_discrete_map={'On Time': 'green', 'Late/Pending': 'red'}
            )
            st.plotly_chart(fig_delivery, use_container_width=True)

        # Shipping analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Average delivery time by region
            delivery_by_region = filtered_data.groupby('CustomerRegion')['ActualDeliveryTime'].mean().sort_values(ascending=True)
            
            fig_delivery_region = px.bar(
                x=delivery_by_region.index,
                y=delivery_by_region.values,
                title='Average Delivery Time by Region',
                labels={'x': 'Region', 'y': 'Average Days'},
                text=delivery_by_region.values
            )
            fig_delivery_region.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_delivery_region, use_container_width=True)
        
        with col2:
            # Shipper performance
            shipper_performance = filtered_data.groupby('ShipperCompany').agg({
                'DeliveryOnTime': 'mean',
                'OrderID': 'count'
            }).sort_values('DeliveryOnTime', ascending=False)
            
            fig_shipper = px.bar(
                x=shipper_performance.index,
                y=shipper_performance['DeliveryOnTime'] * 100,
                title='Shipper On-Time Delivery Performance',
                labels={'x': 'Shipper', 'y': 'On-Time Delivery Rate (%)'},
                text=shipper_performance['DeliveryOnTime'] * 100
            )
            fig_shipper.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_shipper.update_xaxes(tickangle=45)
            st.plotly_chart(fig_shipper, use_container_width=True)

        # Order size analysis
        st.subheader("📊 Order Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Products per order distribution
            fig_products = px.histogram(
                filtered_data,
                x='TotalProductsInOrder',
                nbins=20,
                title='Distribution of Products per Order',
                labels={'TotalProductsInOrder': 'Number of Products', 'count': 'Number of Orders'}
            )
            st.plotly_chart(fig_products, use_container_width=True)
        
        with col2:
            # Order value distribution
            fig_order_value = px.histogram(
                filtered_data,
                x='OrderTotal',
                nbins=30,
                title='Order Value Distribution',
                labels={'OrderTotal': 'Order Value ($)', 'count': 'Number of Orders'}
            )
            st.plotly_chart(fig_order_value, use_container_width=True)

    # --- TAB 4: Financial Health & Profitability ---
    with tab4:
        st.header("💰 Financial Health & Profitability Dashboard")
        st.markdown("Track revenue, analyze profitability trends, and monitor financial KPIs.")

        # Financial KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate key financial metrics
        total_revenue = filtered_data['OrderTotal'].sum()
        completed_orders = filtered_data[filtered_data['OrderStatusName'].isin(['Delivered', 'Shipped'])]
        completed_revenue = completed_orders['OrderTotal'].sum()
        cancelled_orders = filtered_data[filtered_data['OrderStatusName'] == 'Cancelled']
        cancelled_revenue = cancelled_orders['OrderTotal'].sum()
        
        col1.metric("Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("Completed Revenue", f"${completed_revenue:,.2f}")
        col3.metric("Cancelled Revenue", f"${cancelled_revenue:,.2f}")
        col4.metric("Revenue Completion Rate", f"{(completed_revenue/total_revenue*100):.1f}%")

        # Revenue trends and profitability
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly revenue vs cancelled revenue
            monthly_financial = filtered_data.groupby('YearMonth').agg({
                'OrderTotal': 'sum'
            }).reset_index()
            
            monthly_cancelled = cancelled_orders.groupby('YearMonth')['OrderTotal'].sum().reset_index()
            monthly_financial = monthly_financial.merge(monthly_cancelled, on='YearMonth', how='left', suffixes=('_total', '_cancelled'))
            monthly_financial['OrderTotal_cancelled'] = monthly_financial['OrderTotal_cancelled'].fillna(0)
            monthly_financial['Net_Revenue'] = monthly_financial['OrderTotal_total'] - monthly_financial['OrderTotal_cancelled']
            
            fig_financial = go.Figure()
            fig_financial.add_trace(go.Scatter(
                x=monthly_financial['YearMonth'].astype(str),
                y=monthly_financial['OrderTotal_total'],
                name='Total Revenue',
                line=dict(color='blue')
            ))
            fig_financial.add_trace(go.Scatter(
                x=monthly_financial['YearMonth'].astype(str),
                y=monthly_financial['Net_Revenue'],
                name='Net Revenue',
                line=dict(color='green')
            ))
            
            fig_financial.update_layout(
                title='Revenue vs Net Revenue Trends',
                xaxis_title='Month',
                yaxis_title='Revenue ($)'
            )
            st.plotly_chart(fig_financial, use_container_width=True)
        
        with col2:
            # Revenue by tax status
            tax_revenue = filtered_data.groupby('OrderTaxStatusName')['OrderTotal'].sum()
            fig_tax = px.pie(
                values=tax_revenue.values,
                names=tax_revenue.index,
                title='Revenue by Tax Status',
                hole=0.4
            )
            st.plotly_chart(fig_tax, use_container_width=True)

        # Regional profitability analysis
        st.subheader("🌍 Regional Financial Performance")
        
        regional_financial = filtered_data.groupby('CustomerRegion').agg({
            'OrderTotal': 'sum',
            'OrderID': 'count'
        }).reset_index()
        regional_financial['Revenue_per_Order'] = regional_financial['OrderTotal'] / regional_financial['OrderID']
        
        fig_regional_financial = px.scatter(
            regional_financial,
            x='OrderID',
            y='OrderTotal',
            size='Revenue_per_Order',
            color='CustomerRegion',
            title='Regional Performance: Revenue vs Order Volume',
            labels={
                'OrderID': 'Number of Orders',
                'OrderTotal': 'Total Revenue ($)',
                'Revenue_per_Order': 'Revenue per Order ($)'
            }
        )
        st.plotly_chart(fig_regional_financial, use_container_width=True)

    # --- TAB 5: Advanced Analytics ---
    with tab5:
        st.header("🔍 Advanced Analytics & Insights")
        st.markdown("Deep-dive analytics, correlation analysis, and business insights.")

        # Customer-Employee relationship analysis
        st.subheader("👥 Customer-Employee Performance Matrix")

        # Create employee full names
        filtered_data['EmployeeFullName'] = filtered_data['EmployeeFirstName'] + ' ' + filtered_data['EmployeeLastName']

        # Performance metrics by employee and customer region
        col1, col2 = st.columns(2)

        with col1:
            # Revenue by Employee and Region
            emp_region_revenue = filtered_data.groupby(['EmployeeFullName', 'CustomerRegion']).agg({
                'OrderTotal': 'sum',
                'OrderID': 'count',
                'DeliveryOnTime': 'mean',
                'FulfillmentProcessingTime': 'mean'
            }).reset_index()

            emp_region_revenue['DeliveryOnTime'] = emp_region_revenue['DeliveryOnTime'] * 100
            emp_region_revenue['AvgProcessingTime'] = emp_region_revenue['FulfillmentProcessingTime']

            # Pivot for matrix view
            revenue_matrix = emp_region_revenue.pivot(
                index='EmployeeFullName',
                columns='CustomerRegion',
                values='OrderTotal'
            ).fillna(0)

            st.write("**Revenue by Employee and Region ($)**")
            st.dataframe(revenue_matrix.style.format("${:,.0f}").background_gradient(cmap='Greens', axis=None))

        with col2:
            # Orders count by Employee and Region
            orders_matrix = emp_region_revenue.pivot(
                index='EmployeeFullName',
                columns='CustomerRegion',
                values='OrderID'
            ).fillna(0)

            st.write("**Order Count by Employee and Region**")
            st.dataframe(orders_matrix.style.format("{:.0f}").background_gradient(cmap='Blues', axis=None))

        # Employee Performance Summary
        st.subheader("📊 Employee Performance Summary")

        emp_performance = filtered_data.groupby('EmployeeFullName').agg({
            'OrderTotal': ['sum', 'mean'],
            'OrderID': 'count',
            'DeliveryOnTime': 'mean',
            'FulfillmentProcessingTime': 'mean',
            'CustomerEmail': 'nunique'
        }).round(2)

        emp_performance.columns = ['Total Revenue', 'Avg Order Value', 'Total Orders', 'On-Time Delivery %', 'Avg Processing Time', 'Unique Customers']
        emp_performance['On-Time Delivery %'] = emp_performance['On-Time Delivery %'] * 100
        emp_performance = emp_performance.sort_values('Total Revenue', ascending=False)

        st.dataframe(emp_performance.style.format({
            'Total Revenue': '${:,.0f}',
            'Avg Order Value': '${:,.0f}',
            'Total Orders': '{:.0f}',
            'On-Time Delivery %': '{:.1f}%',
            'Avg Processing Time': '{:.0f} hours',
            'Unique Customers': '{:.0f}'
        }).background_gradient(cmap='RdYlGn', axis=0))

        # Customer-Employee Relationship Analysis
        st.subheader("🔗 Customer-Employee Relationship Insights")

        # Get unique employees for pagination
        unique_employees = sorted(filtered_data['EmployeeFullName'].unique())

        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            items_per_page = st.selectbox(
                "Items per page:",
                options=[5, 10, 15, 20],
                index=1,
                key="emp_page_size"
            )

        with col2:
            total_pages = (len(unique_employees) + items_per_page - 1) // items_per_page
            if total_pages > 1:
                current_page = st.slider(
                    f"Page (1-{total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="emp_page_slider"
                )
            else:
                current_page = 1

        with col3:
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(unique_employees))
            st.write(f"Showing {start_idx + 1}-{end_idx} of {len(unique_employees)} employees")

        # Get employees for current page
        page_employees = unique_employees[start_idx:end_idx]

        col1, col2 = st.columns(2)

        with col1:
            # Top customers by employee (paginated)
            top_customers_by_emp = filtered_data.groupby(['EmployeeFullName', 'CustomerEmail']).agg({
                'OrderTotal': 'sum',
                'OrderID': 'count'
            }).reset_index()

            # Get top 3 customers for each employee
            top_customers = top_customers_by_emp.sort_values(['EmployeeFullName', 'OrderTotal'], ascending=[True, False])
            top_customers = top_customers.groupby('EmployeeFullName').head(3)

            st.write("**Top 3 Customers by Revenue for Each Employee**")

            # Filter to only show employees on current page
            page_top_customers = top_customers[top_customers['EmployeeFullName'].isin(page_employees)]

            if len(page_top_customers) > 0:
                for emp in page_employees:
                    emp_data = page_top_customers[page_top_customers['EmployeeFullName'] == emp]
                    if len(emp_data) > 0:
                        st.write(f"**{emp}:**")
                        for _, row in emp_data.iterrows():
                            customer_name = row['CustomerEmail'].split('@')[0]
                            st.write(f"  - {customer_name}: ${row['OrderTotal']:,.0f} ({row['OrderID']} orders)")
                        st.write("")
                    else:
                        st.write(f"**{emp}:** No customer data available")
                        st.write("")
            else:
                st.info("No customer data available for the selected employees.")

        with col2:
            # Customer lifetime value by employee (paginated)
            clv_by_employee = filtered_data.groupby(['EmployeeFullName', 'CustomerEmail'])['TotalLifetimeSpend'].mean().reset_index()
            clv_by_employee = clv_by_employee.sort_values(['EmployeeFullName', 'TotalLifetimeSpend'], ascending=[True, False])

            st.write("**Average Customer Lifetime Value by Employee**")

            # Filter to only show employees on current page
            page_clv_data = clv_by_employee[clv_by_employee['EmployeeFullName'].isin(page_employees)]

            if len(page_clv_data) > 0:
                for emp in page_employees:
                    emp_clv = page_clv_data[page_clv_data['EmployeeFullName'] == emp]
                    if len(emp_clv) > 0:
                        avg_clv = emp_clv['TotalLifetimeSpend'].mean()
                        st.metric(f"{emp}", f"${avg_clv:,.0f}", f"{len(emp_clv)} customers")
                    else:
                        st.metric(f"{emp}", "$0", "0 customers")
            else:
                st.info("No CLV data available for the selected employees.")

        # Summary statistics for current page
        if len(page_employees) > 0:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            # Calculate page statistics
            page_data = filtered_data[filtered_data['EmployeeFullName'].isin(page_employees)]

            with col1:
                page_revenue = page_data['OrderTotal'].sum()
                st.metric("Page Total Revenue", f"${page_revenue:,.0f}")

            with col2:
                page_orders = page_data['OrderID'].count()
                st.metric("Page Total Orders", f"{page_orders:,}")

            with col3:
                page_customers = page_data['CustomerEmail'].nunique()
                st.metric("Page Unique Customers", f"{page_customers:,}")

        # Regional Performance Heatmap
        st.subheader("🌍 Regional Performance Heatmap")

        # Calculate performance metrics by region and employee
        region_performance = filtered_data.groupby(['CustomerRegion', 'EmployeeFullName']).agg({
            'OrderTotal': 'sum',
            'DeliveryOnTime': 'mean',
            'FulfillmentProcessingTime': 'mean',
            'OrderID': 'count'
        }).reset_index()

        region_performance['DeliveryOnTime'] = region_performance['DeliveryOnTime'] * 100

        # Create heatmap for delivery performance
        delivery_heatmap = region_performance.pivot(
            index='CustomerRegion',
            columns='EmployeeFullName',
            values='DeliveryOnTime'
        ).fillna(0)

        st.write("**On-Time Delivery Rate by Region and Employee (%)**")
        st.dataframe(delivery_heatmap.style.format("{:.1f}%").background_gradient(cmap='RdYlGn', axis=None))

        # Processing time heatmap
        processing_heatmap = region_performance.pivot(
            index='CustomerRegion',
            columns='EmployeeFullName',
            values='FulfillmentProcessingTime'
        ).fillna(0)

        st.write("**Average Processing Time by Region and Employee (hours)**")
        st.dataframe(processing_heatmap.style.format("{:.0f}").background_gradient(cmap='RdYlGn_r', axis=None))

        # Key Insights
        st.subheader("💡 Key Insights & Recommendations")

        # Calculate insights
        total_employees = len(filtered_data['EmployeeFullName'].unique())
        total_regions = len(filtered_data['CustomerRegion'].unique())

        # Best performing employee
        best_employee = emp_performance.index[0]
        best_revenue = emp_performance.iloc[0]['Total Revenue']

        # Region with highest revenue
        region_revenue = filtered_data.groupby('CustomerRegion')['OrderTotal'].sum().sort_values(ascending=False)
        top_region = region_revenue.index[0]
        top_region_revenue = region_revenue.iloc[0]

        # Delivery performance insights
        avg_delivery_rate = filtered_data['DeliveryOnTime'].mean() * 100
        avg_processing_time = filtered_data['FulfillmentProcessingTime'].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **🏆 Top Performer:** {best_employee}
            - Revenue: ${best_revenue:,.0f}
            - Consider mentoring other team members

            **🌍 Top Region:** {top_region}
            - Revenue: ${top_region_revenue:,.0f}
            - Focus marketing efforts here
            """)

        with col2:
            st.info(f"""
            **🚚 Delivery Performance:**
            - On-time delivery: {avg_delivery_rate:.1f}%
            - Avg processing time: {avg_processing_time:.0f} hours

            **📊 Coverage:**
            - {total_employees} employees serving {total_regions} regions
            - Matrix shows performance distribution
            """)

        # Recommendations
        st.subheader("🎯 Recommendations")

        recommendations = []

        # Low delivery rate regions
        low_delivery_regions = region_performance[region_performance['DeliveryOnTime'] < 80]['CustomerRegion'].unique()
        if len(low_delivery_regions) > 0:
            recommendations.append(f"📦 **Improve delivery in:** {', '.join(low_delivery_regions)} - On-time delivery below 80%")

        # High processing time employees
        high_processing_employees = emp_performance[emp_performance['Avg Processing Time'] > emp_performance['Avg Processing Time'].mean()]['Avg Processing Time']
        if len(high_processing_employees) > 0:
            recommendations.append(f"⚡ **Optimize processing for:** {', '.join(high_processing_employees.index)} - Above average processing time")

        # Revenue concentration
        revenue_concentration = emp_performance['Total Revenue'].max() / emp_performance['Total Revenue'].sum()
        if revenue_concentration > 0.3:
            recommendations.append(f"🔄 **Balance workload:** One employee handles {revenue_concentration:.1%} of total revenue")

        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success("✅ All performance metrics are within acceptable ranges!")