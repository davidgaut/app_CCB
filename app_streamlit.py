import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 1250px;
        }
    </style>
    """, unsafe_allow_html=True)

# streamlit run your_script.py
# for google site use 
# <iframe
#   src="https://appccb-rvwuxqtvz4tv3hoqgadhr2.streamlit.app/?embed=true"
#   height="1200"
#   style="width:100%;border:none;"
# ></iframe>
# for git (cd app)
# git commit -m'app' -a
# git push

dict_ = dict(zip(['MPO','ES'],['Monetary Policy Orientation','Economic Sentiment']))
df = pd.read_csv('./streamlit_data.csv',index_col=0,header=[0,1]).rename({'ABH':'MPO','LM':'ES',},axis=1)

# Streamlit Settings
col1, _          = st.columns([50,1])
col3, col2, col4 = st.columns([35,35,35])
with col2:
    key = st.multiselect("Index",['MPO','ES',],default=['MPO'],format_func=lambda x: dict_[x] if x in ['MPO','ES'] else x)
with col3:
    country = st.multiselect('Country',df['MPO'].columns.tolist(),default=['Euro Area'],format_func=lambda x: x.title())
    if key != '' and country != '':
        if not isinstance(country,list):
            search_term = country
with col4:
    instrument = st.multiselect('Instrument',['2Y','5Y','10Y','Overnight Rate'], default=[], format_func = lambda x: x.title() if x!=None else x)

targets      = [' - '.join((i,c)) for i in key for c in country]
cols         = [' - '.join((i,c)) for i,c in df.columns]
instruments  = [' - '.join((i,c.replace('Euro Area','Germany') if not i.startswith('Overnight') else c)) for i in instrument for c in country]

df = df.droplevel(0,axis=1)
df.columns = cols

with_instruments = len(instrument)>=1

# Plotly Fig
fig = px.line(df.reset_index(),x='date',y=targets,
                 width=1400, height=500)

subfig = make_subplots(specs=[[{"secondary_y": True}]])
fig  = px.line(df.reset_index(), x='date',y=targets)
if with_instruments:
    instruments = [i for i in instruments if not i.startswith('None')]
    fig2 = px.line(df.reset_index(), x='date',y=instruments, )
    fig2.update_yaxes(showgrid=True, gridwidth=0,)
    fig2.update_traces(yaxis="y2",)
    fig = subfig.add_traces(fig.data + fig2.data)

    fig2.update_xaxes(showgrid=False, gridwidth=0,  gridcolor='LightPink')
    # fig2.update_layout(yaxis_title="Yields / Rates",)

fig.update_layout(
    title="Central Bank Speech Sentiment",
    xaxis_title="",
    # yaxis_title="Indices",
    legend_title="",
    font=dict(
        # family="Courier New, monospace",
        size=16,
        # color="RebeccaPurple"
    ),
    autosize=False,
    width=1000,
    height=400,
    margin=dict(l=30,r=0,b=30,t=50,pad=4),
    legend=dict(
        yanchor="bottom",
        orientation='h',
        y=-0.45,
        xanchor="left",
        # x=0.10
    ),    
    # paper_bgcolor="LightSteelBlue",
)
fig.update_yaxes(showgrid=False, gridwidth=0, gridcolor='LightPink')
for ins in instruments:
    fig.update_traces(patch={"line": {"dash": 'dot'}}, selector={"legendgroup": ins}) 

fig.update_yaxes(title_text="Indices", secondary_y=False)
fig.update_yaxes(title_text="Yields / Rates", secondary_y=True)
fig.show()
#st.markdown('***') #separator

with col1:
    st.plotly_chart(fig)
    st.caption("""<p style="font-family: Open Sans">This graph shows economic sentiment and monetary policy orientation indices for central bankers\' speeches. A high monetary policy orientation reflects a hawkish stance, and a lower monetary policy orientation reflects a dovish stance. Indices can be plotted against bond yields and overnight interest rates.</p>""",unsafe_allow_html=True,)

