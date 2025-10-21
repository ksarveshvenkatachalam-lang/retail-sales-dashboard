"""
RETAIL SALES DASHBOARD
Interactive Streamlit Dashboard for 307K+ transactions
Built with Streamlit, Pandas, and Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Retail Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD DATA WITH ROBUST ERROR HANDLING
# ============================================================================
@st.cache_data
def load_data():
    """Load and preprocess the sales data with comprehensive error handling"""
    
    # Option 1: Try to load from local file (CSV or ZIP)
    # Check for ZIP files first (compressed versions)
    possible_files = [
        'sampledata.csv.zip',   # YOUR EXACT FILE! (no spaces, .csv.zip)
        'sample data.zip',      # With space
        'sampledata.zip',       # Without space
        'sample data.csv.zip',  # Alternative naming
        'sample data.csv',      # Fallback to CSV
        'sampledata.csv',
        'sample_data.csv',
        'sampledata.csv.csv'
    ]
    
    file_found = None
    is_zip = False
    for filename in possible_files:
        if os.path.exists(filename):
            file_found = filename
            is_zip = filename.endswith('.zip')
            break
    
    # Option 2: If no local file, try loading from URL
    # Uncomment and add your URL if hosting externally
    # CSV_URL = "https://your-file-hosting-url/sampledata.csv"
    CSV_URL = None
    
    if file_found is None and CSV_URL is None:
        st.error("❌ **CSV file not found!**")
        st.info("💡 Looking for one of these files: " + ", ".join(possible_files))
        st.info("📁 Current directory contents:")
        try:
            files = os.listdir('.')
            csv_files = [f for f in files if f.endswith('.csv')]
            st.write("**All files:**", files[:20])  # Show first 20 files
            st.write("**CSV files:**", csv_files if csv_files else "None found")
        except Exception as e:
            st.write(f"Error listing files: {e}")
        st.stop()
    
    # Determine source
    if CSV_URL and file_found is None:
        source = CSV_URL
        st.info(f"📥 Loading data from URL...")
    else:
        source = file_found
    
    try:
        # Handle ZIP files
        if is_zip:
            import zipfile
            st.info(f"📦 Extracting data from compressed file: {source}")
            
            with zipfile.ZipFile(source, 'r') as zip_ref:
                # Get the first CSV file in the zip
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    st.error("❌ No CSV file found in the ZIP!")
                    st.stop()
                
                csv_filename = csv_files[0]
                st.info(f"📄 Found: {csv_filename}")
                
                # Read CSV directly from ZIP
                with zip_ref.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file, encoding='utf-8')
        else:
            # First, let's check what's in the file
            file_size = os.path.getsize(source) if not CSV_URL else "Unknown"
            st.info(f"📄 File size: {file_size:,} bytes" if isinstance(file_size, int) else f"📄 Loading from URL")
            
            with open(source, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                second_line = f.readline()
                
            # Check if file is empty
            if not first_line.strip():
                st.error("❌ The CSV file is empty!")
                st.error(f"File size on disk: {file_size} bytes")
                st.info("💡 **This means the file wasn't uploaded properly to GitHub.**")
                st.info("**Solutions:**")
                st.markdown("""
                1. Compress the file to .zip and upload that instead
                2. Use GitHub's web interface to upload
                3. Host the CSV elsewhere and load from URL
                4. Use GitHub Releases for large files
                """)
                st.stop()
                
            # Show preview
            with st.expander("🔍 File preview - First 2 lines"):
                st.code(f"Line 1: {first_line[:200]}")
                st.code(f"Line 2: {second_line[:200]}")
            
            # Read the CSV with multiple fallback options
            try:
                # Try standard read first
                df = pd.read_csv(source, encoding='utf-8')
            except:
                try:
                    # Try with different encoding
                    df = pd.read_csv(source, encoding='latin-1')
                except:
                    try:
                        # Try with Python engine for more flexibility
                        df = pd.read_csv(source, encoding='utf-8', engine='python', sep=None)
                    except Exception as e:
                        st.error(f"❌ Cannot parse CSV file: {str(e)}")
                        st.info(f"First line of file: {first_line}")
                        st.stop()
        
        # Check if dataframe is empty
        if df.empty:
            st.error("❌ CSV file loaded but contains no data!")
            st.stop()
        
        # Display column info for debugging (you can comment this out later)
        st.success(f"✅ Successfully loaded {len(df):,} rows from: **{source}**")
        with st.expander("🔍 Click to see detected columns"):
            st.write("**Original Columns:**", list(df.columns))
        
        # Clean and standardize column names
        # Remove leading/trailing spaces and convert to uppercase
        df.columns = df.columns.str.strip().str.upper()
        
        # Replace spaces with underscores for easier handling
        df.columns = df.columns.str.replace(' ', '_')
        
        # Show cleaned column names
        with st.expander("🔍 Click to see cleaned columns"):
            st.write("**Cleaned Columns:**", list(df.columns))
        
        # Ensure required columns exist with proper naming
        required_mappings = {
            'YEAR': 'YEAR',
            'MONTH': 'MONTH',
            'SUPPLIER': 'SUPPLIER',
            'ITEM_CODE': 'ITEM_CODE',
            'ITEM_DESCRIPTION': 'ITEM_DESCRIPTION',
            'ITEM_TYPE': 'ITEM_TYPE',
            'RETAIL_SALES': 'RETAIL_SALES',
            'RETAIL_TRANSFERS': 'RETAIL_TRANSFERS',
            'WAREHOUSE_SALES': 'WAREHOUSE_SALES'
        }
        
        # Check which columns we have
        missing = []
        for required_col in required_mappings.keys():
            if required_col not in df.columns:
                missing.append(required_col)
        
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.info(f"📋 Available columns: {', '.join(df.columns)}")
            st.stop()
        
        # Convert numeric columns
        numeric_cols = ['RETAIL_SALES', 'RETAIL_TRANSFERS', 'WAREHOUSE_SALES', 'YEAR', 'MONTH']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create date features
        df['Date'] = pd.to_datetime(
            df['YEAR'].astype(int).astype(str) + '-' + 
            df['MONTH'].astype(int).astype(str) + '-01',
            errors='coerce'
        )
        df['Quarter'] = df['Date'].dt.quarter
        df['Month_Name'] = df['Date'].dt.month_name()
        df['Year_Month'] = df['Date'].dt.to_period('M').astype(str)
        
        # Remove any rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        return df
        
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {str(e)}")
        import traceback
        with st.expander("🐛 Click to see full error details"):
            st.code(traceback.format_exc())
        st.stop()

# Load the data
df = load_data()

# ============================================================================
# HEADER
# ============================================================================
st.title("📊 Retail Sales Analytics Dashboard")
st.markdown(f"### Interactive Analysis of {len(df):,} Transactions")
st.markdown("---")

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
st.sidebar.header("🎯 Filters")
st.sidebar.markdown("Select criteria to filter the data")

# Year filter
years = sorted(df['YEAR'].unique())
selected_years = st.sidebar.multiselect(
    "Select Year(s):",
    options=years,
    default=years,
    help="Choose one or more years to analyze"
)

# Month filter
months = sorted(df['MONTH'].unique())
selected_months = st.sidebar.multiselect(
    "Select Month(s):",
    options=months,
    default=months,
    help="Choose specific months to filter"
)

# Item Type filter
item_types = sorted(df['ITEM_TYPE'].dropna().unique())
selected_types = st.sidebar.multiselect(
    "Select Item Type(s):",
    options=item_types,
    default=item_types,
    help="Filter by product category"
)

# Apply filters
df_filtered = df[
    (df['YEAR'].isin(selected_years)) &
    (df['MONTH'].isin(selected_months)) &
    (df['ITEM_TYPE'].isin(selected_types))
]

# Filter summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(df_filtered):,} / {len(df):,} records")
if len(df) > 0:
    st.sidebar.markdown(f"**Filtered:** {((1 - len(df_filtered)/len(df)) * 100):.1f}% excluded")

# ============================================================================
# KEY METRICS
# ============================================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_sales = df_filtered['RETAIL_SALES'].sum()
    st.metric(
        label="💰 Total Sales",
        value=f"${total_sales:,.0f}",
        help="Sum of all retail sales"
    )

with col2:
    avg_sale = df_filtered['RETAIL_SALES'].mean() if len(df_filtered) > 0 else 0
    st.metric(
        label="📊 Avg Transaction",
        value=f"${avg_sale:.2f}",
        help="Average sale amount"
    )

with col3:
    total_transactions = len(df_filtered)
    st.metric(
        label="🛒 Transactions",
        value=f"{total_transactions:,}",
        help="Total number of transactions"
    )

with col4:
    warehouse_units = df_filtered['WAREHOUSE_SALES'].sum()
    st.metric(
        label="📦 Warehouse Units",
        value=f"{warehouse_units:,.0f}",
        help="Total warehouse sales volume"
    )

with col5:
    unique_products = df_filtered['ITEM_DESCRIPTION'].nunique()
    st.metric(
        label="🏷️ Unique Products",
        value=f"{unique_products:,}",
        help="Number of distinct products"
    )

st.markdown("---")

# ============================================================================
# ROW 1: SALES TRENDS
# ============================================================================
st.subheader("📈 Sales Trends Over Time")

col1, col2 = st.columns(2)

with col1:
    # Monthly trend
    monthly_sales = df_filtered.groupby('Year_Month')['RETAIL_SALES'].sum().reset_index()
    monthly_sales['Year_Month'] = pd.to_datetime(monthly_sales['Year_Month'])
    
    fig_trend = px.line(
        monthly_sales,
        x='Year_Month',
        y='RETAIL_SALES',
        title='Monthly Sales Trend',
        labels={'Year_Month': 'Month', 'RETAIL_SALES': 'Sales ($)'}
    )
    fig_trend.update_traces(line_color='#667eea', line_width=3)
    fig_trend.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    # Yearly comparison
    yearly_sales = df_filtered.groupby('YEAR')['RETAIL_SALES'].sum().reset_index()
    
    fig_yearly = px.bar(
        yearly_sales,
        x='YEAR',
        y='RETAIL_SALES',
        title='Annual Sales Comparison',
        labels={'YEAR': 'Year', 'RETAIL_SALES': 'Sales ($)'},
        color='RETAIL_SALES',
        color_continuous_scale='Viridis',
        text='RETAIL_SALES'
    )
    fig_yearly.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_yearly.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_yearly, use_container_width=True)

st.markdown("---")

# ============================================================================
# ROW 2: CATEGORY ANALYSIS
# ============================================================================
st.subheader("🏷️ Category Performance")

col1, col2 = st.columns(2)

with col1:
    # Sales by item type
    type_sales = df_filtered.groupby('ITEM_TYPE')['RETAIL_SALES'].sum().reset_index()
    type_sales = type_sales.sort_values('RETAIL_SALES', ascending=False)
    
    fig_types = px.bar(
        type_sales,
        x='ITEM_TYPE',
        y='RETAIL_SALES',
        title='Sales by Product Category',
        labels={'ITEM_TYPE': 'Category', 'RETAIL_SALES': 'Sales ($)'},
        color='RETAIL_SALES',
        color_continuous_scale='Blues',
        text='RETAIL_SALES'
    )
    fig_types.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_types.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_types, use_container_width=True)

with col2:
    # Pie chart for market share
    fig_pie = px.pie(
        type_sales,
        values='RETAIL_SALES',
        names='ITEM_TYPE',
        title='Market Share by Category',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.0f}<br>Share: %{percent}'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ============================================================================
# ROW 3: TOP PERFORMERS
# ============================================================================
st.subheader("🏆 Top Performers")

col1, col2 = st.columns(2)

with col1:
    # Top 10 products
    top_products = df_filtered.groupby('ITEM_DESCRIPTION')['RETAIL_SALES'].sum().nlargest(10).reset_index()
    
    fig_products = px.bar(
        top_products,
        y='ITEM_DESCRIPTION',
        x='RETAIL_SALES',
        orientation='h',
        title='Top 10 Products by Sales',
        labels={'ITEM_DESCRIPTION': 'Product', 'RETAIL_SALES': 'Sales ($)'},
        color='RETAIL_SALES',
        color_continuous_scale='Reds',
        text='RETAIL_SALES'
    )
    fig_products.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_products.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig_products, use_container_width=True)

with col2:
    # Top 10 suppliers
    top_suppliers = df_filtered.groupby('SUPPLIER')['RETAIL_SALES'].sum().nlargest(10).reset_index()
    
    fig_suppliers = px.bar(
        top_suppliers,
        y='SUPPLIER',
        x='RETAIL_SALES',
        orientation='h',
        title='Top 10 Suppliers by Sales',
        labels={'SUPPLIER': 'Supplier', 'RETAIL_SALES': 'Sales ($)'},
        color='RETAIL_SALES',
        color_continuous_scale='Greens',
        text='RETAIL_SALES'
    )
    fig_suppliers.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_suppliers.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig_suppliers, use_container_width=True)

st.markdown("---")

# ============================================================================
# ROW 4: DISTRIBUTION ANALYSIS
# ============================================================================
st.subheader("📊 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    # Sales distribution histogram
    fig_dist = px.histogram(
        df_filtered[df_filtered['RETAIL_SALES'] > 0],
        x='RETAIL_SALES',
        nbins=50,
        title='Sales Amount Distribution',
        labels={'RETAIL_SALES': 'Sale Amount ($)'},
        color_discrete_sequence=['#764ba2']
    )
    fig_dist.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    # Box plot by category
    top_3_types = df_filtered.groupby('ITEM_TYPE')['RETAIL_SALES'].sum().nlargest(3).index
    df_box = df_filtered[
        df_filtered['ITEM_TYPE'].isin(top_3_types) & 
        (df_filtered['RETAIL_SALES'] > 0)
    ]
    
    if len(df_box) > 0:
        fig_box = px.box(
            df_box,
            x='ITEM_TYPE',
            y='RETAIL_SALES',
            title='Sales Distribution by Top 3 Categories',
            labels={'ITEM_TYPE': 'Category', 'RETAIL_SALES': 'Sale Amount ($)'},
            color='ITEM_TYPE',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_box.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No data available for box plot")

st.markdown("---")

# ============================================================================
# ROW 5: HEATMAP
# ============================================================================
st.subheader("🔥 Monthly Sales Heatmap")

heatmap_data = df_filtered.pivot_table(
    values='RETAIL_SALES',
    index='MONTH',
    columns='YEAR',
    aggfunc='sum',
    fill_value=0
)

if not heatmap_data.empty:
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Month", color="Sales ($)"),
        title="Sales Heatmap: Month vs Year",
        color_continuous_scale='RdYlGn',
        aspect='auto',
        text_auto='.0f'
    )
    fig_heatmap.update_xaxes(side="bottom")
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.info("No data available for heatmap")

st.markdown("---")

# ============================================================================
# ROW 6: SCATTER PLOT
# ============================================================================
st.subheader("🔍 Warehouse vs Retail Analysis")

fig_scatter = px.scatter(
    df_filtered,
    x='WAREHOUSE_SALES',
    y='RETAIL_SALES',
    color='ITEM_TYPE',
    size='RETAIL_SALES',
    hover_data=['ITEM_DESCRIPTION'],
    title='Warehouse Sales vs Retail Sales',
    labels={'WAREHOUSE_SALES': 'Warehouse Units', 'RETAIL_SALES': 'Retail Sales ($)'},
    opacity=0.6,
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_scatter.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ============================================================================
# DATA TABLE
# ============================================================================
st.subheader("📋 Detailed Data View")

show_data = st.checkbox("Show Raw Data", value=False)

if show_data:
    display_cols = [
        'Date', 'YEAR', 'MONTH', 'SUPPLIER', 'ITEM_DESCRIPTION', 
        'ITEM_TYPE', 'RETAIL_SALES', 'WAREHOUSE_SALES', 'RETAIL_TRANSFERS'
    ]
    
    st.dataframe(
        df_filtered[display_cols].head(100),
        use_container_width=True
    )
    
    # Download button
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_sales_data.csv',
        mime='text/csv',
        help="Download the currently filtered data"
    )

# ============================================================================
# KEY INSIGHTS
# ============================================================================
st.markdown("---")
st.subheader("💡 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    if len(df_filtered) > 0:
        best_year = df_filtered.groupby('YEAR')['RETAIL_SALES'].sum().idxmax()
        best_year_sales = df_filtered.groupby('YEAR')['RETAIL_SALES'].sum().max()
        st.info(f"""
        **📅 Best Year**
        
        **{int(best_year)}** generated the highest sales
        
        Total: ${best_year_sales:,.0f}
        """)
    else:
        st.info("No data available")

with col2:
    if len(df_filtered) > 0:
        best_month = df_filtered.groupby('MONTH')['RETAIL_SALES'].sum().idxmax()
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        st.success(f"""
        **📆 Peak Month**
        
        **{month_names.get(int(best_month), best_month)}** shows highest sales
        
        Consistent peak period
        """)
    else:
        st.info("No data available")

with col3:
    if len(df_filtered) > 0:
        best_category = df_filtered.groupby('ITEM_TYPE')['RETAIL_SALES'].sum().idxmax()
        category_sales = df_filtered.groupby('ITEM_TYPE')['RETAIL_SALES'].sum().max()
        st.warning(f"""
        **🏷️ Top Category**
        
        **{best_category}** leads in revenue
        
        Total: ${category_sales:,.0f}
        """)
    else:
        st.info("No data available")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p style='font-size: 1.1em; margin-bottom: 10px;'>
            📊 <strong>Interactive Dashboard</strong> | Built with Streamlit & Plotly
        </p>
        <p style='font-size: 0.9em;'>
            Data updates automatically based on your filter selections
        </p>
    </div>
    """, unsafe_allow_html=True)
