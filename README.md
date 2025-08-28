# 🛍️ North Wind Traders - E-commerce Analytics Dashboard

A comprehensive business intelligence solution for North Wind Traders' e-commerce analytics built with Streamlit, providing deep insights into sales performance, customer behavior, and operational efficiency.

**Prepared by: Mary Shanley Sencil**

## 🏢 About North Wind Traders

North Wind Traders is a global e-commerce company specializing in a diverse range of products including:

- **Food & Beverages**: Specialty foods, beverages, and gourmet products
- **Confections**: Chocolates, candies, and sweet treats
- **Condiments & Seasonings**: Spices, sauces, and culinary essentials
- **Seafood & Produce**: Fresh and frozen seafood, fruits, and vegetables
- **Dairy Products**: Cheese, milk, and dairy specialties
- **Grains & Cereals**: Rice, pasta, and breakfast cereals

With operations spanning multiple international regions including North America, South America, Europe, Asia, and Australia, North Wind Traders serves both B2B and B2C customers through an efficient supply chain and logistics network.

This analytics dashboard provides comprehensive business intelligence to support North Wind Traders' strategic decision-making across sales, marketing, operations, and finance functions.

### 🎯 Key Features

#### 1. 📈 Sales Performance & Revenue Growth
- **Revenue Trends**: Monthly revenue analysis with forecasting capabilities
- **Regional Performance**: Sales breakdown by customer regions
- **Employee Performance**: Top performers and sales contribution analysis
- **Predictive Forecasting**: 12-week revenue forecasting using linear regression

#### 2. 👥 Customer Analytics & Retention
- **RFM Analysis**: Recency, Frequency, Monetary value segmentation
- **Customer Segmentation**: Automated customer clustering using machine learning
- **Churn Prediction**: ML-powered customer churn risk assessment
- **Cohort Analysis**: Customer retention and behavior patterns

#### 3. 📦 Order & Fulfillment Analysis
- **Order Status Tracking**: Real-time order status distribution
- **Delivery Performance**: On-time delivery rates and analysis
- **Shipper Performance**: Comparative analysis of shipping providers
- **Order Size Analysis**: Product distribution and order value patterns

#### 4. 💰 Financial Health & Profitability
- **Revenue Analysis**: Completed vs. cancelled revenue tracking
- **Tax Status Analysis**: Revenue breakdown by tax categories
- **Financial KPIs**: Key financial metrics and completion rates
- **Trend Analysis**: Financial performance over time

#### 5. 🔍 Advanced Analytics & Insights
- **Customer-Employee Performance Matrix**: Comprehensive relationship analysis
- **Regional Performance Heatmaps**: Visual performance comparisons
- **Correlation Analysis**: Business metric interdependencies
- **Automated Recommendations**: Data-driven business insights

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   cd your-project-directory
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   - Place your e-commerce data file as `ecommerce_data_flat_optimized.csv` in the same directory
   - Ensure your data follows the expected format (see Data Format section below)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the dashboard**
   - Open your browser and navigate to `http://localhost:8501`

## 📋 Data Format Requirements

Your CSV file should contain the following key columns:

### Required Columns
- `OrderID` - Unique order identifier
- `OrderDate` - Order date (datetime format)
- `OrderTotal` - Total order value
- `OrderStatusName` - Order status (Shipped, Delivered, Cancelled, etc.)
- `CustomerEmail` - Customer email address
- `CustomerRegion` - Customer geographic region
- `EmployeeFirstName`, `EmployeeLastName` - Employee information
- `ShipperCompany` - Shipping provider name
- `DeliveryOnTime` - Binary indicator (1=On-time, 0=Late)
- `TotalLifetimeSpend` - Customer lifetime value
- `PurchaseCount` - Number of purchases by customer

### Optional but Recommended Columns
- `FulfillmentProcessingTime` - Order processing time in hours
- `TotalProductsInOrder` - Number of products in order
- `OrderTaxStatusName` - Tax status of the order
- `YearMonth` - Pre-computed year-month for time series analysis

## 🎨 Features & Capabilities

### Interactive Filtering
- **Date Range Selection**: Filter data by custom date ranges
- **Regional Filtering**: Focus on specific customer regions
- **Order Status Filtering**: Analyze specific order statuses

### Advanced Analytics
- **Machine Learning Integration**: Customer segmentation and churn prediction
- **Predictive Modeling**: Revenue forecasting using linear regression
- **Performance Benchmarking**: Comparative analysis across regions and employees
- **Automated Insights**: AI-powered recommendations based on data patterns

### Visualization Types
- **Line Charts**: Time series analysis and trends
- **Bar Charts**: Comparative performance analysis
- **Pie Charts**: Distribution and composition analysis
- **Heatmaps**: Correlation and performance matrices
- **Histograms**: Distribution analysis
- **Scatter Plots**: Relationship analysis

## 🔧 Configuration

### Forecasting Parameters
- **Forecast Horizon**: Configurable prediction periods (default: 12 weeks)
- **Algorithm**: Linear regression with confidence intervals
- **Data Aggregation**: Weekly aggregation for stable forecasting

### Performance Settings
- **Pagination**: Configurable items per page (5, 10, 15, 20)
- **Caching**: Automatic data caching for improved performance
- **Memory Optimization**: Efficient data processing for large datasets

## 📊 Business Value

### For North Wind Traders' Sales Teams
- Identify top-performing regions and customers within our global market presence
- Track sales team performance and productivity across different product categories
- Forecast demand and plan inventory accordingly for our diverse product portfolio

### For North Wind Traders' Customer Success
- Predict customer churn risk across our international customer base
- Understand customer lifetime value for our B2B and B2C segments
- Personalize retention strategies for different customer demographics

### For North Wind Traders' Operations
- Monitor delivery performance and shipper efficiency across global regions
- Optimize order fulfillment processes for our complex supply chain
- Track operational KPIs and bottlenecks in our international operations

### For North Wind Traders' Finance
- Monitor revenue trends and profitability across different market segments
- Track financial health and completion rates for our diverse revenue streams
- Analyze tax and revenue distributions for international compliance

## 🛠️ Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Ensure CSV file is in the correct format
   - Check column names match expected format
   - Verify date columns are properly formatted

2. **Performance Issues**
   - Use pagination for large datasets
   - Consider data sampling for very large files
   - Ensure adequate system memory

3. **Visualization Errors**
   - Check for missing or null values in key columns
   - Ensure numeric columns contain valid numbers
   - Verify date formats are consistent

### Getting Help
- Check the browser console for detailed error messages
- Verify all dependencies are installed correctly
- Ensure Python version compatibility

## 🔄 Updates & Maintenance

### Data Refresh
- Replace the CSV file with updated data
- Restart the Streamlit application
- Clear browser cache if needed

### Feature Updates
- The dashboard automatically adapts to new data columns
- Additional metrics can be added by modifying the analysis functions
- Custom visualizations can be integrated using Plotly

## 📄 License

This project is provided as-is for educational and business use. Please ensure compliance with your organization's data usage policies.

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with sample data
5. Submit a pull request

## 📞 Support

For technical support or feature requests:
- Check the troubleshooting section above
- Review error messages in the Streamlit interface
- Ensure all dependencies are properly installed

---

**Built with ❤️ using Streamlit, Plotly, and Python**

*North Wind Traders - Transforming e-commerce data into strategic business insights with this comprehensive analytics dashboard.*

**Prepared by: Mary Shanley Sencil**</content>
<parameter name="filePath">d:\User\Downloads\DiaTrack\DFU_Healing_ML\DiaTrack_DFU_Detection_Models\DS_Class\nene\README.md
