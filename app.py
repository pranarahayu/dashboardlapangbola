import sys
import streamlit as st
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
import urllib

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title='Lapangbola Statistical Dashboard', layout='wide')
st.title('Lapangbola Statistical Dashboard')
st.markdown('Created by: Prana - R&D Division Lapangbola.com')

sys.path.append("assignxg.py")
import assignxg
from assignxg import assign_xg
from assignxg import assign_psxg
from assignxg import data_team
from assignxg import data_player
from assignxg import get_list
from assignxg import get_detail
from assignxg import get_cs
from assignxg import milestone

@st.cache_data(ttl=600)
def load_data(sheets_url):
    xlsx_url = sheets_url.replace("/edit#gid=", "/export?format=xlsx&gid=")
    return pd.read_excel(xlsx_url)

shots_data = load_data(st.secrets["data_shots"])
fixt1 = load_data(st.secrets["fixture"])
fixt1['GW'] = fixt1['GW'].astype(int)
df1 = load_data(st.secrets["testaja"])
df2 = load_data(st.secrets["datapemain"])
histdata = load_data(st.secrets["hist"])
from datetime import date
df1['Date'] = pd.to_datetime(df1.Date)
df1['Month'] = df1['Date'].dt.strftime('%B')
df = pd.merge(df1, df2.drop(['Name'], axis=1), on='Player ID', how='left')
fulldata = get_detail(df)
mlist = get_list(df)

tab1, tab2, tab3 = st.tabs(['**Competitions**', '**Teams**', '**Players**'])

with tab1:
    tab1.subheader('Competition Statistics')
    st.markdown('Untuk melihat statistik tiap pertandingan dan statistik full liga selama satu musim penuh.')
    mstats, fstats = st.tabs(['Match Stats', 'Full Stats'])
    with mstats:
        a, b, c = st.tabs(['Attempts Map', 'Heat Map', 'Match Stats'])
    with fstats:
        table, teams, players = st.tabs(['Standing & Top Stats', 'Team Stats', 'Player Stats'])
        with table:
            col1, col2 = st.columns(2)
            with col1:
                team = st.selectbox('Select Team', pd.unique(histdata['Team']), key='99')
            with col2:
                season = st.selectbox('Select Season(s)',['All Season', 'This Season'], key='98')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Gas price", value=4, delta=-0.5)
        with teams:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                komp = st.selectbox('Select Competition', ['Liga 1', 'Liga 2'], key='3')
            with col2:
                temp_full = fulldata[fulldata['Kompetisi']==komp]
                month = st.multiselect('Select Month', pd.unique(temp_full['Month']), key='14')
            with col3:
                temp_full = temp_full[temp_full['Month'].isin(month)]
                venue = st.multiselect('Select Venue', pd.unique(temp_full['Home/Away']), key='5')
            with col4:
                temp_full = temp_full[temp_full['Home/Away'].isin(venue)]
                gw = st.multiselect('Select Gameweek', pd.unique(temp_full['Gameweek']), key='4')
            with col5:
                cat = st.selectbox('Select Category', ['Goal Threat', 'in Possession', 'out of Possession', 'Misc'], key='13')
            show_tim_data = data_team(fulldata, komp, month, gw, venue, cat)
            st.write(show_tim_data)

            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')
            csv = convert_df(show_tim_data)
    
            st.download_button(label='Download Data', data=csv, file_name='Stats_Tim_Full.csv', mime='text/csv',)
            
        with players:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                komp = st.selectbox('Select Competition', ['Liga 1', 'Liga 2'], key='6')
                temp_pull = fulldata[fulldata['Kompetisi']==komp]
                team = st.multiselect('Select Teams', pd.unique(temp_pull['Team']), key='10')
            with col2:
                temp_pull = temp_pull[temp_pull['Team'].isin(team)]
                pos = st.multiselect('Select Positions', pd.unique(temp_pull['Position']), key='7')
                temp_pull = temp_pull[temp_pull['Position'].isin(pos)]
                age = st.multiselect('Select Age Group', pd.unique(temp_pull['Age Group']), key='11')
            with col3:
                temp_pull = temp_pull[temp_pull['Age Group'].isin(age)]
                nat = st.multiselect('Select Nationality', pd.unique(temp_pull['Nat. Status']), key='8')
                temp_pull = temp_pull[temp_pull['Nat. Status'].isin(nat)]
                month = st.multiselect('Select Month', pd.unique(temp_pull['Month']), key='12')   
            with col4:
                temp_pull = temp_pull[temp_pull['Month'].isin(month)]
                venue = st.multiselect('Select Venue', pd.unique(temp_pull['Home/Away']), key='9')
                temp_pull = temp_pull[temp_pull['Home/Away'].isin(venue)]
                gw = st.multiselect('Select Gameweek', pd.unique(temp_pull['Gameweek']), key='17')
            with col5:
                mins = st.number_input('Input minimum mins. played', min_value=0,
                                       max_value=90*max(fulldata['Gameweek']), step=90, key=18)
                metrik = st.multiselect('Select Metrics', mlist, key='19')
            cat = st.selectbox('Select Category', ['Total', 'per 90'], key='16')
            show_player_data = data_player(fulldata, komp, team, pos, month, venue, gw, age, nat, metrik, mins, cat, df2)
            st.write(show_player_data)

            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')
            csv = convert_df(show_player_data)
    
            st.download_button(label='Download Data', data=csv, file_name='Stats_Pemain_Full.csv', mime='text/csv',)
            

with tab2:
    tab2.subheader('Team Statistics')
    
with tab3:
    tab3.subheader('Player Statistics')
