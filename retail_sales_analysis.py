import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit page configuration
st.set_page_config(page_title="Retail Sales Analysis", layout="wide")
st.sidebar.title("Navigation")
sections = ["Introduction", "EDA", "Modeling", "Conclusion"]
selected_section = st.sidebar.radio("Choose a Section", sections)
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Load data when file is uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    df = None

# Introduction Section
if selected_section == "Introduction":
    st.title("Introduction")
    st.markdown("""
        Welcome to the **Retail Sales Analysis** application! This app is designed to help businesses uncover insights
        from sales data, identify trends, and predict future performance using machine learning models.
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            """
            <div style="text-align: center; font-size: 18px; color: #333; font-style: italic;">
                "Retail sales data can reveal valuable insights to improve sales performance."
            </div>
            """, unsafe_allow_html=True)

        st.image(
            "C:/Users/alish/Desktop/ids_project.jpg", 
            use_container_width=True,
            caption="Visual Representation of Retail Sales",
        )

    with col2:
        st.markdown("""### About the Dataset
The dataset contains retail sales data. Key columns include:
- **Transaction ID**: Unique identifier for each transaction.
- **Date**: Date of the transaction.
- **Customer ID**: Unique identifier for customers.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Product Category**: Category of the purchased product.
- **Quantity**: Quantity of the product purchased.
- **Price per Unit**: Price per unit of the product.
- **Total Amount**: Total amount spent.
""")
    if df is not None:
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())
    else:
        st.warning("Please upload a dataset to get started.")

# Exploratory Data Analysis (EDA) Section
elif selected_section == "EDA":
    if df is not None:
        st.title("Exploratory Data Analysis (EDA)")

        # 1. Summary Statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())

        # 2. Unique Value Counts
        st.subheader("Unique Values Per Column")
        st.write(df.nunique())

        # 3. Data Types
        st.subheader("Data Types")
        st.write(df.dtypes)

        # 4. Missing Values Heatmap
        st.subheader("Missing Values Heatmap")
        missing_values = df.isnull().sum()
        st.write(f"Missing values per column:\n{missing_values}")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        ax.set_title('Missing Values Heatmap')
        st.pyplot(fig)

        # 5. Histogram for Age
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        df['Age'].hist(bins=10, color='skyblue', edgecolor='black', ax=ax)
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # 6. Scatter Plot: Quantity vs Total Amount
        st.subheader("Quantity vs Total Amount")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Quantity', y='Total Amount', data=df, ax=ax)
        ax.set_title('Quantity vs Total Amount')
        st.pyplot(fig)

        # 7. Sales by Product Category and Gender
        st.subheader("Sales by Product Category and Gender")
        fig, ax = plt.subplots()
        sns.barplot(x='Product Category', y='Total Amount', hue='Gender', data=df, ax=ax)
        ax.set_title('Sales by Product Category and Gender')
        st.pyplot(fig)

        # 8. Correlation Heatmap
        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)

        # 9. Total Sales by Month
        st.subheader("Monthly Sales Trend")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        monthly_sales = df.groupby('Month')['Total Amount'].sum()
        fig, ax = plt.subplots()
        monthly_sales.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Monthly Sales Trend')
        ax.set_xlabel('Month')
        ax.set_ylabel('Total Sales')
        st.pyplot(fig)

        # 10. Yearly Sales
        st.subheader("Yearly Sales Trend")
        df['Year'] = df['Date'].dt.year
        yearly_sales = df.groupby('Year')['Total Amount'].sum()
        fig, ax = plt.subplots()
        yearly_sales.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Yearly Sales Trend')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Sales')
        st.pyplot(fig)

        # 11. Product Category Distribution
        st.subheader("Product Category Distribution")
        fig, ax = plt.subplots()
        df['Product Category'].value_counts().plot(kind='bar', color='green', ax=ax)
        ax.set_title('Distribution of Product Categories')
        st.pyplot(fig)

        # 12. Gender Distribution
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots()
        df['Gender'].value_counts().plot(kind='bar', color='purple', ax=ax)
        ax.set_title('Gender Distribution')
        st.pyplot(fig)

        # 13. Total Amount by Gender
        st.subheader("Total Amount by Gender")
        fig, ax = plt.subplots()
        df.groupby('Gender')['Total Amount'].sum().plot(kind='bar', ax=ax)
        ax.set_title('Total Sales by Gender')
        ax.set_ylabel('Total Amount')
        st.pyplot(fig)

        # 14. Total Amount by Age Group
        st.subheader("Total Amount by Age Group")
        age_bins = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['<18', '18-30', '30-45', '45-60', '60+'])
        df['Age Group'] = age_bins
        fig, ax = plt.subplots()
        df.groupby('Age Group')['Total Amount'].sum().plot(kind='bar', ax=ax, color='orange')
        ax.set_title('Total Amount by Age Group')
        ax.set_ylabel('Total Amount')
        st.pyplot(fig)

        # 15. Pairplot for Numeric Features
        st.subheader("Pairplot of Numeric Features")
        fig = sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
        st.pyplot(fig)


elif selected_section == "Modeling":
    if df is not None:
        st.title("Model Training")
        
        # Data Preprocessing
        df = df.drop(columns=['Transaction ID', 'Customer ID', 'Date'], errors='ignore')
        
        # Encode categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        # Ensure numeric data
        df = df.select_dtypes(include=['float64', 'int64'])
        
        # Train-test split
        X = df.drop(columns=['Total Amount'], errors='ignore')
        y = df['Total Amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        st.subheader("Model Evaluation")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared (R²):** {r2:.2f}")
        
        # Plot: Actual vs. Predicted
        st.subheader("Actual vs. Predicted Plot")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color="red")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs. Predicted")
        st.pyplot(fig)

        # Plot: Residuals Distribution
        st.subheader("Residuals Distribution")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax, color="purple")
        ax.set_title("Residuals Distribution")
        ax.set_xlabel("Residuals")
        st.pyplot(fig)

# Conclusion Section
elif selected_section == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
        This project analyzed retail sales data, uncovering critical insights into customer behavior, sales trends, and product performance. Key findings include:
        - Significant trends in monthly and yearly sales.
        - Gender and age significantly influence sales.
        - A machine learning model was developed to predict sales with measurable accuracy.
        
        ### Project Highlights:
        - **EDA:** Detailed analysis with 15 tasks revealing meaningful insights.
        - **Modeling:** Implemented a linear regression model with performance metrics.
        - **Visualization:** Comprehensive interactive charts for user engagement.
    """)
    st.image("C:/Users/alish/Desktop/conclusion.jpg", caption="Retail Sales Analysis Summary", use_column_width=True)
