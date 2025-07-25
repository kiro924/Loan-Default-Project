import plotly.express as px
from xgboost import XGBClassifier
import streamlit as st
import pandas as pd
import zipfile
import joblib
import os

st.set_page_config(
page_title="Loan Default Analysis Dashboard",
page_icon="📊",
layout="wide",
initial_sidebar_state="collapsed"
)  

pages=st.sidebar.selectbox('Select Page', ['Home Page' , "📊 Analysis Page", "🤖 ML Prediction"])
if pages=='Home Page':

    st.markdown("""
    <style>
        .title {
            background-color: #ffffff;
            color: #616f89;
            padding: 10px;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            border: 4px solid #000083;
            border-radius: 10px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #ffffff;
            border: 2px solid #000083;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 15px;
            transition: transform 0.2s ease-in-out;
        }
        .metric-card:hover {
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.3);
            transform: scale(1.02);
        }
        .metric-value {
            color: #000083;
            font-size: 26px;
            font-weight: 600;
            font-style: italic;
            text-shadow: 1px 1px 2px #000083;
            margin: 10px 0;
        }
        .metric-label {
            margin-bottom: 5px;
            font-size: 20px;
            font-weight: 500;
            color: #999999;
        }
        .expander-header {
            font-size: 24px !important;
            font-weight: bold !important;
            color: #000083 !important;
        }
    </style>
""", unsafe_allow_html=True)

    st.markdown('<div class="title">Loan Default Analysis Overview</div>', unsafe_allow_html=True)
    st.image("dataset-cover.jpg")

    def load_data():
        current_dir = os.path.dirname(__file__)
        zip_path = os.path.join(current_dir, 'cleaned data.zip')
        csv_name = 'cleaned data.csv'
    
        # Extract CSV from zip if not already extracted
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(csv_name, current_dir)
    
        file_path = os.path.join(current_dir, csv_name)
        return pd.read_csv(file_path)
    
    # Load the DataFrame
    df = load_data()
    
    with st.expander("📊 Loan Portfolio Performance KPIs"):
        total_loans = len(df)
        default_loans = df[df['Status'] == 1]
        default_rate = len(default_loans) / total_loans        
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Default Rate</div>
                    <div class="metric-value">{default_rate:.0%}</div></div>""", unsafe_allow_html=True)

    # Default rate by loan type
        loan_type_defaults = df.groupby('loan_type')['Status'].apply(lambda x: (x == 1).mean()).reset_index()
        loan_type_defaults.columns = ['Loan Type', 'Default Rate']
        loan_type_defaults['Default Rate'] = loan_type_defaults['Default Rate'].apply(lambda x: f"{x:.1%}")
        st.markdown("#### 🏷️ Default Rate by Loan Type")
        st.dataframe(loan_type_defaults, use_container_width=True)
    
        # By Loan Purpose
        loan_purpose_defaults = df.groupby('loan_purpose')['Status'].apply(lambda x: (x == 1).mean()).reset_index()
        st.markdown("#### 🏷️ Default Rate by Loan Type")
        st.dataframe(loan_purpose_defaults, use_container_width=True)
    
        region_defaults = df.groupby('Region')['Status'].apply(lambda x: (x == 1).mean()).reset_index()
        st.markdown("🌍 Default Rate by Region")
        st.dataframe(region_defaults, use_container_width=True)

        df['Credit_Bucket'] = pd.cut(df['Credit_Score'], bins=[300, 579, 669, 739, 799, 850],
                                      labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        credit_score_defaults = df.groupby('Credit_Bucket')['Status'].apply(lambda x: (x == 1).mean()).reset_index()
        st.markdown("💳 Default Rate by Credit Score Range")
        st.dataframe(credit_score_defaults, use_container_width=True)

    with st.expander("🧮 Risk-Based KPIs"):
        default_df = df[df['Status'] == 1]
        avg_ltv = default_df['LTV'].mean()
        avg_interest_rate = default_df['rate_of_interest'].mean()        
        avg_interest_spread = default_df['Interest_rate_spread'].mean()        
        avg_income = default_df['income'].mean()        
        avg_dti = default_df['dtir1'].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Avg. LTV", f"{avg_ltv:.1f}%")
        col2.metric("💸 Avg. Interest Rate", f"{avg_interest_rate:.2f}%")
        col3.metric("🧾 Interest Rate Spread", f"{avg_interest_spread:.2f}%")
        
        col4, col5 = st.columns(2)
        col4.metric("👨‍👩‍👧‍👦 Avg. Income", f"${avg_income:,.0f}")
        col5.metric("📉 Avg. DTI Ratio", f"{avg_dti:.2f}")

    with st.expander("🏷️  Credit Policy KPIs"):
         default_df = df[df['Status'] == 1]
         total_loans = len(df)
         credit_type_defaults = df.groupby('Credit_Worthiness')['Status'].apply(lambda x: (x == 1).mean()).reset_index()
         credit_type_defaults.columns = ['Credit Type', 'Default Rate']
         credit_type_defaults['Default Rate'] = credit_type_defaults['Default Rate'].apply(lambda x: f"{x:.1%}")
         st.markdown("#### 📌 Default Rate by Credit Type")
         st.dataframe(credit_type_defaults, use_container_width=True)

         approval_stage_defaults = df.groupby('approv_in_adv')['Status'].apply(lambda x: (x == 1).mean()).reset_index()
         approval_stage_defaults.columns = ['Approval Stage', 'Default Rate']
         approval_stage_defaults['Default Rate'] = approval_stage_defaults['Default Rate'].apply(lambda x: f"{x:.1%}")
         st.markdown("#### ✅ Default Rate by Loan Approval Stage")
         st.dataframe(approval_stage_defaults, use_container_width=True)

         structure_flags = ['Neg_ammortization', 'lump_sum_payment', 'interest_only']
         structure_results = []
         for col in structure_flags:
            temp = df.groupby(col)['Status'].apply(lambda x: (x == 1).mean()).reset_index()
            temp.columns = ['Flag Value', 'Default Rate']
            temp['Loan Feature'] = col
            structure_results.append(temp)
         structure_df = pd.concat(structure_results)
         structure_df['Default Rate'] = structure_df['Default Rate'].apply(lambda x: f"{x:.1%}")
         st.markdown("#### 🏗️ Default Rate by Loan Structure Features")
         st.dataframe(structure_df[['Loan Feature', 'Flag Value', 'Default Rate']], use_container_width=True)

    with st.expander("🏘️ 4. Operational KPIs"):
  
        df['Upfront_charges'] = pd.to_numeric(df['Upfront_charges'], errors='coerce')
        df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
        
        defaulted = df[df['Status'] == 1]
        paid = df[df['Status'] == 0]

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Avg Loan (Defaulted)", 
                    f"${defaulted['loan_amount'].mean():,.0f}", 
                    delta=f"${defaulted['loan_amount'].mean() - paid['loan_amount'].mean():,.0f} vs Paid")
    
        col2.metric("📋 Avg Upfront Charges (Defaulted)",
                    f"${defaulted['Upfront_charges'].mean():,.0f}", 
                    delta=f"${defaulted['Upfront_charges'].mean() - paid['Upfront_charges'].mean():,.0f} vs Paid")
        
        if {'application_date', 'approval_date'}.issubset(df.columns):
            df['application_date'] = pd.to_datetime(df['application_date'])
            df['approval_date'] = pd.to_datetime(df['approval_date'])
            df['approval_time_days'] = (df['approval_date'] - df['application_date']).dt.days
    
            default_time = defaulted['approval_time_days'].mean()
            paid_time = paid['approval_time_days'].mean()
    
            col3.metric("⏱ Avg Approval Time (Defaulted)",
                        f"{default_time:.1f} days",
                        delta=f"{default_time - paid_time:.1f} vs Paid")

elif pages=="📊 Analysis Page":
    def load_data():
        current_dir = os.path.dirname(__file__)
        zip_path = os.path.join(current_dir, 'cleaned data.zip')
        csv_name = 'cleaned data.csv'
    
        # Extract CSV from zip if not already extracted
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(csv_name, current_dir)
    
        file_path = os.path.join(current_dir, csv_name)
        return pd.read_csv(file_path)
    
    # Load the DataFrame
    df = load_data()
    st.title('📊 Exploratory Data Analysis - Loan Default')
    st.sidebar.header('🔍 Filter Options')
    loan_purpose_filter=st.sidebar.multiselect('loan_purpose', df['loan_purpose'].unique(),default=df['loan_purpose'].unique())
    loan_type_filter=st.sidebar.multiselect('loan_type', df['loan_type'].unique(), default=df['loan_type'].unique())
    region_filter=st.sidebar.multiselect('Region', df['Region'].unique(), default=df['Region'].unique())

    filtered_df=df[
    (df['loan_purpose'].isin(loan_purpose_filter))&
    (df['loan_type'].isin(loan_type_filter))&
    (df['Region'].isin(region_filter))
    ]

    st.subheader("📈 Univariate Analysis")
    select_col=st.selectbox("Select a column for univariate analysis:", filtered_df.columns)

    if pd.api.types.is_numeric_dtype(filtered_df[select_col]):
        col1,col2=st.columns(2)
        col1.plotly_chart(px.histogram(filtered_df, x=select_col, nbins=50, barmode='group',
                                       title=f'histogram distribution of {select_col}'.title()))
        col2.plotly_chart(px.box(filtered_df, x=select_col, title=f'Box Plot of {select_col}'.title()))
        with st.expander(f"📊 Detailed Statistics for {select_col.title()}"):
            st.write(filtered_df[select_col].describe())
            st.write("🔼 Highest 5 Values:", filtered_df[select_col].nlargest(5))
            st.write("🔽 Lowest 5 Values:", filtered_df[select_col].nsmallest(5))
            q1 = filtered_df[select_col].quantile(0.25)
            q3 = filtered_df[select_col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            outliers = filtered_df[(filtered_df[select_col] > upper_bound) | (filtered_df[select_col] < lower_bound)]
            st.write(f"🚨 Outliers count: {outliers.shape[0]}")

    else:
        col1,col2=st.columns(2)
        cat_df=filtered_df[select_col].value_counts().reset_index()
        cat_df.columns=[select_col, 'Count']
        col1.plotly_chart(px.bar(cat_df, x=select_col, y='Count', text_auto=True, title=f'Count of each {select_col}'.title(),
                                color_discrete_sequence=px.colors.qualitative.Bold, labels=True))
        col2.plotly_chart(px.pie(cat_df, names=select_col, values='Count', title=f'percentage of each {select_col}'.title(),
                                color_discrete_sequence=px.colors.qualitative.Bold))
        with st.expander(f"📊 Frequency Distribution for {select_col.title()}"):
            st.write("🔢 Absolute Counts:")
            st.write(filtered_df[select_col].value_counts())
        
            st.write("📊 Percentage Distribution (%):")
            st.write((filtered_df[select_col].value_counts(normalize=True) * 100).round(2))


    st.subheader("🔁 Bivariate Analysis")
    if pd.api.types.is_numeric_dtype(filtered_df[select_col]):
        col1,col2=st.columns(2)
        col1.plotly_chart(px.histogram(filtered_df, x=select_col, color='Status', nbins=50, barmode='group',
                                       title=f'distribution of {select_col} by status'.title()))
        col2.plotly_chart(px.box(filtered_df, x='Status', y=select_col, title=f'Box Plot of {select_col} by status'.title()))
        with st.expander(f"📈 Status Breakdown by {select_col.title()}"):
            st.dataframe(filtered_df.groupby('Status')[select_col].describe().style.format("{:.2f}"))


    else:
        col1,col2=st.columns(2)
        series = (filtered_df.groupby([select_col, 'Status']).size().groupby(level=0).apply(lambda x: x / x.sum() * 100))
        series.name = 'percentage'
        if isinstance(series.index, pd.MultiIndex):
            temp_names = [f'level_{i}' for i in range(series.index.nlevels)]
            series.index = series.index.set_names(temp_names)
            cat = series.reset_index()
            # Rename to proper names
            cat = cat.rename(columns={temp_names[0]: select_col, temp_names[1]: 'Status'})
        else:
            cat = series.reset_index().rename(columns={series.index.name: select_col})
        col1.plotly_chart(px.bar(cat,x=select_col,y='percentage',color='Status',barmode='group',text_auto=True,
                                title=f'Loan Status Distribution by {select_col.title()}',
                                labels={'percentage': 'Percentage (%)'},
                                color_discrete_sequence=px.colors.qualitative.Dark2
    ))
    
        col2.plotly_chart(px.pie(cat,names=select_col,values='percentage',
                           title=f'Loan Status Distribution by {select_col.title()}',
                           color_discrete_sequence=px.colors.qualitative.Dark2
    ))

        st.subheader("🔀 Multivariate Analysis")
        if pd.api.types.is_object_dtype(filtered_df[select_col]):
            col1,col2=st.columns(2)
            cat1=df.groupby([select_col,'Status'])[['income']].median().reset_index().sort_values(
                ascending=False, by='income')

            col1.plotly_chart(px.bar(cat1, x=select_col, y='income', color='Status', barmode='group',text_auto=True,
                           title=f'average income by {select_col} and status'.title()))
            
            col2.plotly_chart(px.pie(cat1, names=select_col, values='income', color='Status',
                           title=f'average income by {select_col} and status'.title()))
            
else:
    def load_data():
        current_dir = os.getcwd()
        zip_path = os.path.join(current_dir, 'cleaned data.zip')
        csv_name = 'cleaned data.csv'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(csv_name, current_dir)
        file_path = os.path.join(current_dir, csv_name)
        return pd.read_csv(file_path)

    df = load_data()
    df = df.drop(['loan_amount', 'Interest_rate_spread', 'open_credit', 'construction_type',
                  'Secured_by', 'total_units', 'co-applicant_credit_type', 'Security_Type'], axis=1)
    x = df.drop('Status', axis=1)
    y = df['Status']
    
    st.title('🤖 Loan Default Prediction Model')
    
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'pipeline_pre.pkl')
    pipeline_pre = joblib.load(file_path)
    pipeline = joblib.load('Loan Prediction ML model')
    inputs = joblib.load('inputs')
    
    with st.form("loan_application_form"):
        st.header("📋 Enter Applicant Information")
    
        st.markdown("### 👤 Applicant Information")
        col1, col2 = st.columns(2)
        with col1:
            loan_limit = st.selectbox('Loan Limit', x['loan_limit'].unique())
            Gender = st.selectbox('Gender', x['Gender'].unique())
            age = st.selectbox('Age', x['age'].unique())
            income = st.slider('Income',
                int(x['income'].min()),
                int(x['income'].max()),
                5000,
                step=2000
            )
        with col2:
            credit_type = st.selectbox('Credit Type', x['credit_type'].unique())
            Credit_Score = st.slider('Credit Score',
                int(x['Credit_Score'].min()),
                int(x['Credit_Score'].max()),
                500,
                step=50
            )
    
        st.markdown("### 💰 Loan Details")
        col1, col2 = st.columns(2)
        with col1:
            approv_in_adv = st.selectbox('Approved in Advance', x['approv_in_adv'].unique())
            loan_type = st.selectbox('Loan Type', x['loan_type'].unique())
            loan_purpose = st.selectbox('Loan Purpose', x['loan_purpose'].unique())
        with col2:
            Credit_Worthiness = st.selectbox('Credit Worthiness', x['Credit_Worthiness'].unique())
            business_or_commercial = st.selectbox('Business/Commercial', x['business_or_commercial'].unique())
            rate_of_interest = st.slider('Rate of Interest',
                float(x['rate_of_interest'].min()),
                float(x['rate_of_interest'].max()),
                0.1,
                step=0.5
            )
    
        st.markdown("### 💸 Loan Terms")
        col1, col2 = st.columns(2)
        with col1:
            Upfront_charges = st.slider('Upfront Charges',
                float(x['Upfront_charges'].min()),
                float(x['Upfront_charges'].max()),
                1000.0,
                step=100.0
            )
            term = st.slider('Term (months)',
                int(x['term'].min()),
                int(x['term'].max()),
                60,
                step=20
            )
        with col2:
            Neg_ammortization = st.selectbox('Negative Amortization', x['Neg_ammortization'].unique())
            interest_only = st.selectbox('Interest Only', x['interest_only'].unique())
            lump_sum_payment = st.selectbox('Lump Sum Payment', x['lump_sum_payment'].unique())
    
        st.markdown("### 🏠 Property & Security")
        col1, col2 = st.columns(2)
        with col1:
            property_value = st.slider('Property Value',
                float(x['property_value'].min()),
                float(x['property_value'].max()),
                8000.0,
                step=1000.0
            )
            occupancy_type = st.selectbox('Occupancy Type', x['occupancy_type'].unique())
        with col2:
            Region = st.selectbox('Region', x['Region'].unique())
    
        st.markdown("### 📈 Financial Metrics")
        col1, col2 = st.columns(2)
        with col1:
            LTV = st.slider('Loan-to-Value (LTV)',
                float(x['LTV'].min()),
                float(x['LTV'].max()),
                0.1,
                step=0.1
            )
        with col2:
            dtir1 = st.slider('Debt-to-Income Ratio (DTIR)',
                float(x['dtir1'].min()),
                float(x['dtir1'].max()),
                0.1,
                step=10.0
            )

        st.markdown("### 📝 Submission Info")
        submission_of_application = st.selectbox('Submission Type', x['submission_of_application'].unique())
    
        submitted = st.form_submit_button("📊 Predict Loan Default")

        if submitted:
            user_input = {
                'loan_limit': loan_limit,
                'Gender': Gender,
                'approv_in_adv': approv_in_adv,
                'loan_type': loan_type,
                'loan_purpose': loan_purpose,
                'Credit_Worthiness': Credit_Worthiness,
                'open_credit': 0,
                'business_or_commercial': business_or_commercial,
                'loan_amount': 0,
                'rate_of_interest': rate_of_interest,
                'Interest_rate_spread': 0,
                'Upfront_charges': Upfront_charges,
                'term': term,
                'Neg_ammortization': Neg_ammortization,
                'interest_only': interest_only,
                'lump_sum_payment': lump_sum_payment,
                'property_value': property_value,
                'construction_type': 'Not Available',
                'occupancy_type': occupancy_type,
                'Secured_by': 'Not Available',
                'total_units': 0,
                'income': income,
                'credit_type': credit_type,
                'Credit_Score': Credit_Score,
                'co-applicant_credit_type': 'Not Available',
                'age': age,
                'submission_of_application': submission_of_application,
                'LTV': LTV,
                'Region': Region,
                'Security_Type': 'Not Available',
                'dtir1': dtir1,
            }
    
            inputs_df = pd.DataFrame([user_input], columns=inputs)
            prediction = pipeline.predict(inputs_df)[0]
    
            if prediction == 0:
                st.success("✅ Loan Approved")
                st.balloons()
            else:
                st.error("❌ Loan Defaulted")
                st.snow()

            inputs_df['Prediction'] = prediction
    
            st.write("### 🔎 Applicant Input Summary")
            st.dataframe(inputs_df.drop(columns=["Prediction"]))
    
            with open("prediction_log.csv", "a") as f:
                inputs_df.to_csv(f, header=f.tell() == 0, index=False)

    
