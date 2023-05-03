# import json, pathlib, os
# import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 1250px;
        }
    </style>
    """, unsafe_allow_html=True)

# streamlit run your_script.py
# for google site use insert https://davidgaut-app-ccb-app-streamlit-p4ulid.streamlit.app/?embed=true


dict_ = dict(zip(['MPO','ES'],['Monetary Policy Orientation','Economic Sentiment']))
df = pd.read_csv('./streamlit_data.csv',index_col=0,header=[0,1]).rename({'ABH':'MPO','LM':'ES'},axis=1)

col1,_ = st.columns([50,1])
col2, col3 = st.columns([35,35])
with col2:
    key = st.multiselect("Index",['MPO','ES'],default=['MPO'],format_func=lambda x: dict_[x])
with col3:
    country = st.multiselect('Country',df['MPO'].columns.tolist(),default=['Euro Area'],format_func=lambda x: x.title())
    if key != '' and country != '':
        if not isinstance(country,list):
            search_term = country
    
targets = [' - '.join((i,c)) for i in key for c in country]
cols    = [' - '.join((i,c)) for i,c in df.columns]

df = df.droplevel(0,axis=1)
df.columns = cols

fig = px.line(df.reset_index(),x='date',y=targets,
                 width=1400, height=500)
fig.update_layout(
    title="Central Bank Speech Sentiment",
    xaxis_title="",
    yaxis_title="",
    legend_title="",
    font=dict(
        # family="Courier New, monospace",
        size=16,
        # color="RebeccaPurple"
    ),
    autosize=False,
    width=1000,
    height=400,
    margin=dict(
        l=30,
        r=0,
        b=30,
        t=50,
        pad=4
    ),
    legend=dict(
    yanchor="bottom",
    orientation='h',
    y=-0.45,
    xanchor="left",
    # x=0.10
    ),    
    # paper_bgcolor="LightSteelBlue",
)

# fig.show()
#st.markdown('***') #separator

with col1:
    st.plotly_chart(fig)
    st.caption("""<p style="font-family: Open Sans">This graph shows the economic sentiment and monetary policy orientation for central bankers\' speeches. A high monetary policy orientation reflects hawkish speeches a lower monetary policy orientation reflects dovish speeches. The methods to compute the score are based on Loughran and McDonald (2011) and Apel, Blix, and Hull (2021).</p>""",unsafe_allow_html=True,)

