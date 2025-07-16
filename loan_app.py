import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title='Loan Prediction App', layout='wide')

pd.set_option('display.max_columns',None)
def load_data():
    current_dir = os.path.dirname("D:\Final Project")
    file_path = os.path.join(current_dir, 'cleaned data.csv')
    return pd.read_csv(file_path)

df = load_data()

pages=st.sidebar.selectbox('Select Page', ["📊 Analysis Page", "🤖 ML Prediction"])
if pages=="📊 Analysis Page":
    st.title('📊 Exploratory Data Analysis - Loan Default')
    st.sidebar.header('🔍 Filter Options')
    loan_purpose_filter=st.sidebar.multiselect('loan_purpose', df['loan_purpose'].unique(),default=df['loan_purpose'].unique())
    region_filter=st.sidebar.multiselect('Region', df['Region'].unique(), default=df['Region'].unique())
    status_filter=st.sidebar.multiselect('Status', df['Status'].unique(), default=df['Status'].unique())

    filtered_df=df[
    (df['loan_purpose'].isin(loan_purpose_filter))&
    (df['Region'].isin(region_filter))&
    (df['Status'].isin(status_filter))
    ]

    st.subheader("📈 Univariate Analysis")
    select_col=st.selectbox("Select a column for univariate analysis:", filtered_df.columns)

    if pd.api.types.is_numeric_dtype(filtered_df[select_col]):
        st.plotly_chart(px.histogram(filtered_df, x=select_col, nbins=50))
        st.plotly_chart(px.box(filtered_df, x=select_col))
        st.write(filtered_df[select_col].describe())
        st.write("🔼 Highest 5 Values:", filtered_df[select_col].nlargest(5))
        st.write("🔽 Lowest 5 Values:", filtered_df[select_col].nsmallest(5))
        q1=filtered_df[select_col].quantile(.25)
        q3=filtered_df[select_col].quantile(.75)
        iqr=q3 - q1
        upper_bound=q3 + 1.5 * iqr
        lower_bound=q1 - 1.5 * iqr
        outliers = filtered_df[(filtered_df[select_col] > upper_bound) | (filtered_df[select_col] < lower_bound)]
        st.write(f"🚨 Outliers count: {outliers.shape[0]}")

    else:
        cat_df=filtered_df[select_col].value_counts().reset_index()
        cat_df.columns=[select_col, 'Count']
        st.plotly_chart(px.bar(cat_df, x=select_col, y='Count'))
        st.plotly_chart(px.pie(cat_df, names=select_col, values='Count'))
        st.write(filtered_df[select_col].value_counts())
        st.write((filtered_df[select_col].value_counts(normalize=True)*100).round(2))

    st.subheader("🔁 Bivariate Analysis")
    if pd.api.types.is_numeric_dtype(filtered_df[select_col]):
        st.plotly_chart(px.histogram(filtered_df, x=select_col, color='Status', nbins=50, barmode='overlay'))
        st.plotly_chart(px.box(filtered_df, x='Status', y=select_col))
        st.write(filtered_df.groupby(select_col)['Status'].describe())

    else:
        # cat=filtered_df[select_col].value_counts().unstack().mul(100)
        # cat_long=cat.reset_index().mlte(id_vars=select_col, var_name='Status', value_name='percentage')
        # series = (df.groupby([select_col, 'Status']).size().groupby(level=0).apply(lambda x: x / x.sum() * 100))
        # cat = series.rename('percentage').reset_index(drop=False, allow_duplicates=True)

        # st.plotly_chart(px.bar(cat, x=select_col, y='percentage', barmode='group',
        #                       title=f'loan status distribution by {select_col}'.title(),
        #                       color_discrete_sequence=px.colors.qualitative.Dark2))

        # st.plotly_chart(px.pie(cat, names=select_col, values='percentage',
        #                       title=f'loan status distribution by {select_col}'.title(),
        #                       color_discrete_sequence=px.colors.qualitative.Dark2))
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
        st.plotly_chart(px.bar(cat,x=select_col,y='percentage',color='Status',barmode='group',
                                title=f'Loan Status Distribution by {select_col.title()}',
                                labels={'percentage': 'Percentage (%)'},
                                color_discrete_sequence=px.colors.qualitative.Dark2
    ))
    
        st.plotly_chart(px.pie(cat,names=select_col,values='percentage',
                           title=f'Loan Status Distribution by {select_col.title()}',
                           color_discrete_sequence=px.colors.qualitative.Dark2
    ))

        st.subheader("🔀 Multivariate Analysis")
        if pd.api.types.is_object_dtype(filtered_df[select_col]):
            cat1=df.groupby([select_col,'Status'])[['income']].median().reset_index().sort_values(
                ascending=False, by='income')

            st.plotly_chart(px.bar(cat1, x=select_col, y='income', color='Status', barmode='group',
                           title=f'average income by {select_col} and status'.title()))
            
            st.plotly_chart(px.pie(cat1, names=select_col, values='income', color='Status',
                           title=f'average income by {select_col} and status'.title()))
            
else:
    os.chdir(r'D:\Final Project')
    pd.set_option('display.max_columns',None)
    df=pd.read_csv('cleaned data.csv')
    x=df.drop('Status', axis=1)
    y=df['Status']
    st.title('🤖 Loan Default Prediction Model')
    pipeline_pre=joblib.load('pipeline_pre')
    pipeline=joblib.load('Loan Prediction ML model')
    inputs=joblib.load('inputs')
    st.subheader("📋 Enter Applicant Informations")

    user_input={
        'loan_limit': st.selectbox('loan_limit', x['loan_limit'].unique()),
        'Gender': st.selectbox('Gender', x['Gender'].unique()),
        'approv_in_adv': st.selectbox('approv_in_adv', x['approv_in_adv'].unique()),
        'loan_type': st.selectbox('loan_type', x['loan_type'].unique()),
        'loan_purpose': st.selectbox('loan_purpose', x['loan_purpose'].unique()),
        'Credit_Worthiness': st.selectbox('Credit_Worthiness', x['Credit_Worthiness'].unique()),
        'open_credit': st.selectbox('open_credit', x['open_credit'].unique()),
        'business_or_commercial': st.selectbox('business_or_commercial', x['business_or_commercial'].unique()),
        'loan_amount': st.slider('loan_amount', min_value=int(x['loan_amount'].min()),
                                max_value=int(x['loan_amount'].max()), value=6000, step=300),
        'rate_of_interest': st.slider('rate_of_interest', min_value=float(x['rate_of_interest'].min()),
                                max_value=float(x['rate_of_interest'].max()), value=0.5, step=0.5),
        'Interest_rate_spread': st.slider('Interest_rate_spread', min_value=float(x['Interest_rate_spread'].min()),
                                max_value=float(x['Interest_rate_spread'].max()), value=0.5, step=0.5),
        'Upfront_charges': st.slider('Upfront_charges', min_value=int(x['Upfront_charges'].min()),
                                max_value=int(x['Upfront_charges'].max()), value=1000, step=500),
        'term': st.slider('term', min_value=int(x['term'].min()),
                                max_value=int(x['term'].max()), value=60, step=60),
        'Neg_ammortization': st.selectbox('Neg_ammortization', x['Neg_ammortization'].unique()),
        'interest_only': st.selectbox('interest only', x['interest_only'].unique()),
        'lump_sum_payment': st.selectbox('lump_sum_payment', x['lump_sum_payment'].unique()),
        'property_value': st.slider('property value', min_value=int(x['property_value'].min()),
                                max_value=int(x['property_value'].max()), value=50000, step=20000),
        'construction_type': st.selectbox('construction_type', x['construction_type'].unique()),
        'occupancy_type': st.selectbox('occupancy_type', x['occupancy_type'].unique()),
        'Secured_by': st.selectbox('Secured_by', x['Secured_by'].unique()),
        'total_units': st.selectbox('total_units', x['total_units'].unique()),
        'income': st.slider('income', min_value=int(x['income'].min()),
                                max_value=int(x['income'].max()), value=5000, step=2000),
        'credit_type': st.selectbox('credit_type', x['credit_type'].unique()),
        'Credit_Score': st.slider('Credit_Score', min_value=int(x['Credit_Score'].min()),
                                max_value=int(x['Credit_Score'].max()), value=500, step=50),
        'co-applicant_credit_type': st.selectbox('co-applicant_credit_type', x['co-applicant_credit_type'].unique()),
        'age': st.selectbox('age', x['age'].unique()),
        'submission_of_application': st.selectbox('submission_of_application', x['submission_of_application'].unique()),
        'LTV': st.slider('LTV', min_value=int(x['LTV'].min()),max_value=int(x['LTV'].max()), value=10, step=5),
        'Region': st.selectbox('Region', x['Region'].unique()),
        'Security_Type': st.selectbox('Security_Type', x['Security_Type'].unique()),
        'dtir1': st.slider('dtir1', min_value=int(x['dtir1'].min()),max_value=int(x['dtir1'].max()), value=10, step=10),
    }

    if st.button('Predict Loan Default'):
        inputs_df=pd.DataFrame([user_input], columns=inputs)
        # for col in inputs_df.columns:
        #     if col in df.columns:
        #         inputs_df[col] = inputs_df[col].astype(df[col].dtype)
        st.write("Data types of inputs_df:", inputs_df.dtypes)
        st.write("Sample input values:", inputs_df.head())
        prediction=pipeline.predict(inputs_df)[0]
        st.success("✅ Loan Defaulted" if prediction==0 else "❌ Loan Rejected")

    
