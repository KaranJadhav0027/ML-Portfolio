# Customer Segmentation Case Study

## Overview
This case study demonstrates customer segmentation using unsupervised machine learning techniques. The project analyzes customer behavior patterns, demographics, and purchasing habits to create meaningful customer segments for targeted marketing and business strategy.

## Problem Statement
Group customers into distinct segments based on their characteristics and behavior patterns to enable personalized marketing strategies, improve customer retention, and optimize business operations.

## Dataset
- **Type**: Synthetic customer dataset with realistic characteristics
- **Features**: 
  - **Demographics**: Age, Gender, Education_Level, Marital_Status, Num_Children
  - **Financial**: Annual_Income, Credit_Score, Avg_Order_Value
  - **Behavioral**: Spending_Score, Total_Purchases, Days_Since_Last_Purchase, Years_Customer
  - **Preferences**: Product_Category_Preference, Channel_Preference
- **Size**: 1000 customers (configurable)
- **Target**: Unsupervised segmentation (no predefined labels)

## Features
- **Data Generation**: Realistic synthetic customer dataset
- **Multiple Clustering Algorithms**: 
  - K-Means clustering
  - Hierarchical clustering
  - Automatic best method selection
- **Comprehensive Analysis**: 
  - Customer segment profiles
  - Demographic analysis by segment
  - Behavioral pattern analysis
  - Business recommendations
- **Visualization**: 
  - Income vs Spending scatter plots
  - Age vs Spending analysis
  - Segment distribution charts
  - Segment characteristics heatmaps
  - Demographic distribution by segment
- **Model**: Multiple clustering algorithms with StandardScaler preprocessing
- **Evaluation**: Silhouette score, Calinski-Harabasz score
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithms**: K-Means and Hierarchical Clustering
- **Preprocessing**: StandardScaler for numerical features, LabelEncoder for categorical
- **Feature Engineering**: Comprehensive customer feature set
- **Validation**: Multiple cluster validity metrics
- **Visualization**: Multi-dimensional customer analysis

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python CustomerSegmentationVisual.py --customers 1000 --max_k 8
```

### Command Line Arguments
- `--customers`: Number of customers to generate (default: 1000)
- `--max_k`: Maximum k for optimization (default: 8)

## Output
The application generates:
- **Model Performance Metrics**: Silhouette score, Calinski-Harabasz score for each method
- **Visualizations**: Saved in `artifacts_customer_segmentation/plots/`
  - `optimal_clusters_analysis.png`: Cluster optimization analysis
  - `customer_segmentation_analysis.png`: Comprehensive segmentation analysis
- **Trained Model**: Saved in `artifacts_customer_segmentation/models/` with timestamp
- **Business Insights**: Detailed segment analysis and recommendations

## Model Performance
The Customer Segmentation model typically achieves:
- **Silhouette Score**: 0.4-0.7 (good cluster separation)
- **Calinski-Harabasz Score**: High values indicating well-separated segments
- **Best Method**: Usually K-Means for customer data

## Key Insights
1. **Customer Segments**: Distinct groups based on income, spending, and behavior
2. **Demographic Patterns**: Age, gender, and education distribution by segment
3. **Behavioral Analysis**: Purchasing patterns and preferences by segment
4. **Business Value**: Actionable insights for marketing and retention strategies

## Customer Segment Types

### 1. High-Value Customers
- **Characteristics**: High income, high spending
- **Strategy**: Retention & upselling
- **Actions**: VIP service, premium products, loyalty rewards

### 2. Budget-Conscious Customers
- **Characteristics**: Lower income, price-sensitive
- **Strategy**: Value-focused marketing
- **Actions**: Discounts, budget bundles, cost-saving tools

### 3. High-Income, Low-Spending
- **Characteristics**: High income, conservative spending
- **Strategy**: Engagement & conversion
- **Actions**: Luxury showcases, personalized recommendations

### 4. Low-Income, High-Spending
- **Characteristics**: Lower income, high spending
- **Strategy**: Support & retention
- **Actions**: Flexible payments, budget tools, loyalty benefits

## Dataset Features Description

### Demographics
- **Age**: Customer age (18-80)
- **Gender**: Male/Female
- **Education_Level**: High School, Bachelor, Master, PhD
- **Marital_Status**: Single, Married, Divorced
- **Num_Children**: Number of children (0+)

### Financial
- **Annual_Income**: Yearly income ($20,000-$150,000)
- **Credit_Score**: Credit score (300-850)
- **Avg_Order_Value**: Average order value ($)

### Behavioral
- **Spending_Score**: Spending behavior score (1-100)
- **Total_Purchases**: Total number of purchases
- **Days_Since_Last_Purchase**: Days since last purchase (1-365)
- **Years_Customer**: Years as customer (0-20)

### Preferences
- **Product_Category_Preference**: Electronics, Clothing, Home, Sports, Books
- **Channel_Preference**: Online, Store, Both

## File Structure
```
Customer_Segmentation_Case_Study/
├── CustomerSegmentationVisual.py
├── requirements.txt
├── README.md
└── artifacts_customer_segmentation/
    ├── models/
    │   ├── customer_segmentation_model_*.joblib
    │   ├── scaler_*.joblib
    │   └── label_encoders_*.joblib
    └── plots/
        ├── optimal_clusters_analysis.png
        └── customer_segmentation_analysis.png
```

## Business Applications
Customer segmentation is used for:
- **Marketing Strategy**: Targeted campaigns for each segment
- **Product Development**: Features based on segment needs
- **Pricing Strategy**: Segment-specific pricing models
- **Customer Service**: Tailored support approaches
- **Retention Programs**: Segment-specific retention strategies

## Marketing Strategies by Segment

### High-Value Customers
- Premium product recommendations
- Exclusive offers and early access
- VIP customer service
- Loyalty rewards program

### Budget-Conscious Customers
- Discount offers and promotions
- Budget-friendly product bundles
- Price comparison tools
- Cost-saving tips and guides

### High-Income, Low-Spending
- Luxury product showcases
- Personalized recommendations
- Lifestyle-based marketing
- Premium service offerings

### Low-Income, High-Spending
- Flexible payment options
- Budget management tools
- Financial wellness resources
- Loyalty program benefits

## Algorithm Comparison
- **K-Means**: Fast, works well with numerical data, good for large datasets
- **Hierarchical**: Provides cluster hierarchy, no need to specify k, good for small datasets
- **Best Choice**: K-Means typically performs better for customer segmentation

## Dependencies
- pandas >= 2.1.0
- numpy >= 1.25.0
- matplotlib >= 3.8.0
- seaborn >= 0.12.2
- scikit-learn >= 1.3.0
- joblib >= 1.3.2

## Author
**Swamit Amit jadhav**  
*Date: 18/09/2025*


