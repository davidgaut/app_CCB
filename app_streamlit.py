import json, pathlib, os
import pandas as pd
import numpy as np

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

dict_ = dict(zip(['MPO','ES'],['Monetary Policy Orientation','Economic Sentiment']))
# if False:
#     # Paths
#     with open('../options.json') as f:
#         options = json.load(f)

#     with open('../options_paths.json') as f:
#         options_paths = json.load(f)

#     save_path = pathlib.Path('../figures/')
#     os.makedirs(save_path, exist_ok=True)
#     path_train = options_paths['path_train']

#     # Load Dataset
#     all_docs = pd.read_parquet(path_train+'speech_features.parquet').set_index('reference')
#     all_docs = all_docs.replace(['no_info','',np.nan,None],['NO_INFO','NO_INFO','NO_INFO','NO_INFO']).rename({'abh':'MPO','lm':'ES'},axis=1)
#     all_docs['country'] = all_docs.country.str.title()

#     #
#     country_TS = all_docs[['date','country','MPO','ES']].groupby(['country','date']).mean().reset_index(level=0)
#     country_TS = country_TS.pivot(columns='country')
#     country_TS.sort_index(inplace=True)


#     # Plot with Streamlit
#     data = all_docs

#     df = country_TS
#     df = df.resample('1B').ffill().loc['1997':].rolling('364D').mean().loc['2000':]
#     # .ewm(halflife=180).mean()
#     df.to_csv('./streamlit_data.csv')

# else: 
df = pd.read_csv('./streamlit_data.csv',index_col=0,header=[0,1]).rename({'ABH':'MPO','LM':'ES'},axis=1)

#
#st.markdown("<h2 style='text-align: center; color: black;'>Central Bank Speech Sentiment</h2>", unsafe_allow_html=True)

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
# buffer, col2 = st.columns([1, 40])
# with col2:
#     if not df.empty:
#         st.dataframe(df[['author_short']])
#     else:
#         st.write('Did not find any person matching the criteria')

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


# import json, pathlib, os
# import pandas as pd
# import numpy as np

# import plotly.express as px
# import streamlit as st
# st.set_page_config(layout="wide")

# # width = st.number_input("Width", 0, 300, 228)
# st.markdown("""
#     <style>
#         .stMultiSelect [data-baseweb=select] span{
#             max-width: 1250px;
#         }
#     </style>
#     """, unsafe_allow_html=True)


# with open('../options.json') as f:
#     options = json.load(f)

# with open('../options_paths.json') as f:
#     options_paths = json.load(f)

# save_path = pathlib.Path('../figures/')
# os.makedirs(save_path, exist_ok=True)

# # Paths
# path_train = options_paths['path_train']

# # Load Dataset
# all_docs = pd.read_parquet(path_train+'speech_features.parquet').set_index('reference')
# all_docs = all_docs.replace(['no_info','',np.nan,None],['NO_INFO','NO_INFO','NO_INFO','NO_INFO']).rename({'abh':'ABH','lm':'LM'},axis=1)
# all_docs['country'] = all_docs.country.str.title()

# #
# country_TS = all_docs[['date','country','ABH','LM']].groupby(['country','date']).mean().reset_index(level=0)
# country_TS = country_TS.pivot(columns='country')
# country_TS.sort_index(inplace=True)


# dict_ = dict(zip(['ABH','LM'],['Monetary Policy Orientation','Economic Sentiment']))

# # Plot with Streamlit
# data = all_docs

# df = country_TS
# df = df.resample('1B').ffill().loc['1997':].rolling('364D').mean().loc['2000':]
# # .ewm(halflife=180).mean()

# #
# st.markdown("<h1 style='text-align: center; color: black;'>Central Bank Speeches</h1>", unsafe_allow_html=True)
# st.markdown('***')

# _, col1, _ = st.columns([10,30,10])
# _,col2, col3,_ = st.columns([10,15,15,10])
# with col2:
#     key = st.multiselect("Index",['ABH','LM'],default=['ABH'],format_func=lambda x: dict_[x])
# with col3:
#     country = st.multiselect('Country',data.country.unique(),default=['Euro Area'],format_func=lambda x: x.title())
#     if key != '' and country != '':
#         if not isinstance(country,list):
#             search_term = country
    
# targets = [' - '.join((i,c)) for i in key for c in country]
# cols    = [' - '.join((i,c)) for i,c in df.columns]

# df = df.droplevel(0,axis=1)
# df.columns = cols
# # buffer, col2 = st.columns([1, 40])
# # with col2:
# #     if not df.empty:
# #         st.dataframe(df[['author_short']])
# #     else:
# #         st.write('Did not find any person matching the criteria')


# fig = px.line(df.reset_index(),x='date',y=targets,
#                  width=1200, height=500)
# fig.update_layout(
#     title="",
#     xaxis_title="",
#     yaxis_title="",
#     legend_title="",
#     font=dict(
#         # family="Courier New, monospace",
#         size=16,
#         # color="RebeccaPurple"
#     ),
#     autosize=False,
#     width=1200,
#     height=500,
#     margin=dict(
#         l=50,
#         r=1,
#         b=100,
#         t=100,
#         pad=4
#     ),
#     legend=dict(
#     yanchor="top",
#     orientation='h',
#     y=-0.15,
#     xanchor="left",
#     x=0.10
#     ),    
#     # paper_bgcolor="LightSteelBlue",
# )

# # fig.show()

# st.markdown('***') #separator
# with col1:
#     # st.markdown("<h5 style='text-align: center; color: black;'>{:s}</h1>".format('Indices'), unsafe_allow_html=True)
#     # st.line_chart(df[targets],)
#     st.plotly_chart(fig)

