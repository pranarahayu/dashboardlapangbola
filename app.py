import sys
import io
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
from assignxg import get_pct
from assignxg import get_pssw
from assignxg import get_wdl
from assignxg import get_skuad
from assignxg import get_formasi
from assignxg import get_radar
from assignxg import get_simi
from assignxg import get_playerlist

sys.path.append("fungsiplot.py")
import fungsiplot
from fungsiplot import plot_skuad
from fungsiplot import plot_skuadbar
from fungsiplot import plot_form
from fungsiplot import beli_pizza
from fungsiplot import plot_compare

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
th = load_data(st.secrets["th"])
cf = load_data(st.secrets["cf"])
cd = load_data(st.secrets["cd"])

from datetime import date
df1['Date'] = pd.to_datetime(df1.Date)
df1['Month'] = df1['Date'].dt.strftime('%B')
df22 = get_detail(df2)
df = pd.merge(df1, df2.drop(['Name'], axis=1), on='Player ID', how='left')
fulldata = get_detail(df)
mlist = get_list(fulldata)
no_temp = df1[df1['Kompetisi']=='Liga 1']
histodata = milestone(histdata, no_temp)
csdata = get_cs(no_temp)
curdata = df1[['Team','Assist','Yellow Card','Red Card']]
curdata = curdata.groupby(['Team'], as_index=False).sum()
curdata2 = pd.merge(curdata, csdata, on='Team', how='left')

tab1, tab2, tab3 = st.tabs(['**Competitions**', '**Teams**', '**Players**'])

with tab1:
    tab1.subheader('Competition Statistics')
    st.markdown('Untuk melihat statistik tiap pertandingan dan statistik full liga selama satu musim penuh.')
    mstats, fstats = st.tabs(['Match Stats', 'Full Stats'])
    with mstats:
        a, b, c = st.tabs(['Attempts Map', 'Heat Map', 'Match Stats'])
    with fstats:
        table, teams, players = st.tabs(['Standing & Milestones', 'Team Stats', 'Player Stats'])
        with table:
            col1, col2 = st.columns(2)
            with col1:
                season = st.selectbox('Select Season(s)',['All Season', 'This Season'], key='98')
            with col2:
                if (season == 'All Season'):
                    team = st.selectbox('Select Team', pd.unique(histdata['Team']), key='99')
                else:
                    team = st.selectbox('Select Team', pd.unique(no_temp['Team']), key='97')
                all_teams = st.checkbox('Select All Teams', key='300')
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            if all_teams:
                if (season == 'All Season'):
                    with col1:
                        st.metric(label="Goals", value=histodata['Goal'].sum())
                    with col2:
                        st.metric(label="Assists", value=histodata['Assist'].sum())
                    with col3:
                        st.metric(label="Yellow Cards", value=histodata['Yellow Card'].sum())
                    with col4:
                        st.metric(label="Red Cards", value=histodata['Red Card'].sum())
                    with col5:
                        st.metric(label="Concededs", value=histodata['Conceded'].sum())
                    with col6:
                        st.metric(label="Clean Sheets", value=histodata['Clean Sheet'].sum())
                else:
                    with col1:
                        st.metric(label="Goals", value=int(curdata2['Goal'].sum()))
                    with col2:
                        st.metric(label="Assists", value=curdata2['Assist'].sum())
                    with col3:
                        st.metric(label="Yellow Cards", value=curdata2['Yellow Card'].sum())
                    with col4:
                        st.metric(label="Red Cards", value=curdata2['Red Card'].sum())
                    with col5:
                        st.metric(label="Concededs", value=int(curdata2['Conceded'].sum()))
                    with col6:
                        st.metric(label="Clean Sheets", value=int(curdata2['Clean Sheet'].sum()))
            else:
                if (season == 'All Season'):
                    with col1:
                        st.metric(label="Goals", value=list((histodata[histodata['Team']==team]['Goal']).reset_index(drop=True))[0])
                    with col2:
                        st.metric(label="Assists", value=list((histodata[histodata['Team']==team]['Assist']).reset_index(drop=True))[0])
                    with col3:
                        st.metric(label="Yellow Cards", value=list((histodata[histodata['Team']==team]['Yellow Card']).reset_index(drop=True))[0])
                    with col4:
                        st.metric(label="Red Cards", value=list((histodata[histodata['Team']==team]['Red Card']).reset_index(drop=True))[0])
                    with col5:
                        st.metric(label="Concededs", value=list((histodata[histodata['Team']==team]['Conceded']).reset_index(drop=True))[0])
                    with col6:
                        st.metric(label="Clean Sheets", value=list((histodata[histodata['Team']==team]['Clean Sheet']).reset_index(drop=True))[0])
                else:
                    with col1:
                        st.metric(label="Goals", value=int(list((curdata2[curdata2['Team']==team]['Goal']).reset_index(drop=True))[0]))
                    with col2:
                        st.metric(label="Assists", value=list((curdata2[curdata2['Team']==team]['Assist']).reset_index(drop=True))[0])
                    with col3:
                        st.metric(label="Yellow Cards", value=list((curdata2[curdata2['Team']==team]['Yellow Card']).reset_index(drop=True))[0])
                    with col4:
                        st.metric(label="Red Cards", value=list((curdata2[curdata2['Team']==team]['Red Card']).reset_index(drop=True))[0])
                    with col5:
                        st.metric(label="Concededs", value=int(list((curdata2[curdata2['Team']==team]['Conceded']).reset_index(drop=True))[0]))
                    with col6:
                        st.metric(label="Clean Sheets", value=int(list((curdata2[curdata2['Team']==team]['Clean Sheet']).reset_index(drop=True))[0]))
            st.markdown('''<style>
                [data-testid="stMetricLabel"] > div:nth-child(1) {justify-content: center;}
                [data-testid="stMetricValue"] > div:nth-child(1) {justify-content: center;}
                [data-testid="stMetricLabel"] > div:nth-child(1) {justify-content: center;}
                [data-testid="stMetricValue"] > div:nth-child(1) {justify-content: center;}
                [data-testid="stMetricLabel"] > div:nth-child(1) {justify-content: center;}
                [data-testid="stMetricValue"] > div:nth-child(1) {justify-content: center;}
                </style>''', unsafe_allow_html=True)
        with teams:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                komp = st.selectbox('Select Competition', ['Liga 1', 'Liga 2'], key='3')
            with col2:
                temp_full = fulldata[fulldata['Kompetisi']==komp]
                month = st.multiselect('Select Month', pd.unique(temp_full['Month']), key='14')
                all_mos = st.checkbox('Select All Months', key='301')
                if all_mos:
                    month = pd.unique(temp_full['Month'])
            with col3:
                temp_full = temp_full[temp_full['Month'].isin(month)]
                venue = st.multiselect('Select Venue', pd.unique(temp_full['Home/Away']), key='5')
            with col4:
                temp_full = temp_full[temp_full['Home/Away'].isin(venue)]
                gw = st.multiselect('Select Gameweek', pd.unique(temp_full['Gameweek']), key='4')
                all_gws = st.checkbox('Select All GWs', key='302')
                if all_gws:
                    gw = pd.unique(temp_full['Gameweek'])
            with col5:
                cat = st.selectbox('Select Category', ['Goal Threat', 'in Possession', 'out of Possession', 'Misc'], key='13')
            show_tim_data = data_team(fulldata, komp, month, gw, venue, cat)
            st.write(show_tim_data)

            @st.cache_data(ttl=600)
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
                all_teams = st.checkbox('Select All Teams', key='303')
                if all_teams:
                    team = pd.unique(temp_pull['Team'])
            with col2:
                temp_pull = temp_pull[temp_pull['Team'].isin(team)]
                pos = st.multiselect('Select Positions', pd.unique(temp_pull['Position']), key='7')
                all_poss = st.checkbox('Select All Positions', key='304')
                if all_poss:
                    pos = pd.unique(temp_pull['Position'])
                temp_pull = temp_pull[temp_pull['Position'].isin(pos)]
                age = st.multiselect('Select Age Group', pd.unique(temp_pull['Age Group']), key='11')
            with col3:
                temp_pull = temp_pull[temp_pull['Age Group'].isin(age)]
                nat = st.multiselect('Select Nationality', pd.unique(temp_pull['Nat. Status']), key='8')
                temp_pull = temp_pull[temp_pull['Nat. Status'].isin(nat)]
                month = st.multiselect('Select Month', pd.unique(temp_pull['Month']), key='12')
                all_mos = st.checkbox('Select All Months', key='305')
                if all_mos:
                    month = pd.unique(temp_pull['Month'])
            with col4:
                temp_pull = temp_pull[temp_pull['Month'].isin(month)]
                venue = st.multiselect('Select Venue', pd.unique(temp_pull['Home/Away']), key='9')
                temp_pull = temp_pull[temp_pull['Home/Away'].isin(venue)]
                gw = st.multiselect('Select Gameweek', pd.unique(temp_pull['Gameweek']), key='17')
                all_gws = st.checkbox('Select All GWs', key='306')
                if all_gws:
                    gw = pd.unique(temp_pull['Gameweek'])
            with col5:
                mins = st.number_input('Input minimum mins. played', min_value=0,
                                       max_value=3060, step=90, key=18)
                metrik = st.multiselect('Select Metrics', mlist, key='19')
            cat = st.selectbox('Select Category', ['Total', 'per 90'], key='16')
            show_player_data = data_player(fulldata, komp, team, pos, month, venue, gw, age, nat, metrik, mins, cat, df22)
            st.write(show_player_data)
            #st.write(df22)

            @st.cache_data(ttl=600)
            def convert_df(df):
                return df.to_csv().encode('utf-8')
            csv = convert_df(show_player_data)
    
            st.download_button(label='Download Data', data=csv, file_name='Stats_Pemain_Full.csv', mime='text/csv',)
            

with tab2:
    tab2.subheader('Teams')
    pro, plo = st.tabs(['Team Profile', 'Plot Statistics'])
    with pro:
        col1, col2, col3 = st.columns(3)
        with col1:
            komp = st.selectbox('Select Competition', ['Liga 1', 'Liga 2'], key='50')
        with col2:
            smt = fulldata[fulldata['Kompetisi']==komp]
            team = st.selectbox('Select Team', pd.unique(smt['Team']), key='51')
        with col3:
            smt = smt[smt['Team']==team]
            gw = st.multiselect('Select GWs', pd.unique(smt['Gameweek']), key='52')
            all_gws = st.checkbox('Select All GWs', key='307')
            if all_gws:
                gw = pd.unique(smt['Gameweek'])
        ds = get_pssw(fulldata, th, team, gw)
        ds = ds.replace('', pd.NA)
        ps = ['PS1','PS2','PS3','PS4','PS5']
        s = ['S1','S2','S3','S4','S5','S6','S7']
        w = ['W1','W2','W3','W4','W5','W6','W7']

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(team+'\'s Results')
            rslt = get_wdl(fulldata, team, gw)
            st.dataframe(rslt, hide_index=True)
        with col2:
            st.subheader(team+'\'s Squad List')
            skd = get_skuad(df1, df2, team, gw)
            st.dataframe(skd)

        st.subheader(team+'\'s Squad - % of Minutes Played')
        col1, col2 = st.columns(2)
        with col1:
            skbr = plot_skuadbar(df1, df2, team, gw)
            st.pyplot(skbr)
        with col2:
            sksc = plot_skuad(df1, df2, team, gw)
            st.pyplot(sksc)

        st.subheader(team+'\'s Characteristics')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('**Play Style**')
            for col in ds[ps]:
                if (ds[col].isnull().values.any() == False):
                    st.markdown(':large_yellow_square:'+' '+list(ds[col])[0])
        with col2:
            st.markdown('**Strengths**')
            for col in ds[s]:
                if (ds[col].isnull().values.any() == False):
                    st.markdown(':large_green_square:'+' '+list(ds[col])[0])
        with col3:
            st.markdown('**Weaknesses**')
            for col in ds[w]:
                if (ds[col].isnull().values.any() == False):
                    st.markdown(':large_red_square:'+' '+list(ds[col])[0])

        st.subheader(team+'\'s Starting Formation')
        gw2 = st.selectbox('Select GW', pd.unique(smt['Gameweek']), key='53')
        full_form = get_formasi(df1, cd)
        sf = plot_form(full_form, cf, team, gw2)

        st.pyplot(sf)

    
with tab3:
    tab3.subheader('Players')
    pro, pse = st.tabs(['Player Profile', 'Player Search'])
    with pro:
        col1, col2 = st.columns(2)
        with col1:
            mins = st.number_input('Input minimum mins. played', min_value=90, max_value=3060, step=90, key=96)
        with col2:
            komp = st.selectbox('Select Competition', ['Liga 1', 'Liga 2'], key='101')
        rank_p90 = get_pct(df1, df2, mins, komp)[0]
        rank_tot = get_pct(df1, df2, mins, komp)[1]
        rank_pct = get_pct(df1, df2, mins, komp)[2]
        col2, col3, col4 = st.columns(3)
        with col2:
            tempp = rank_p90[rank_p90['Kompetisi']==komp]
            klub = st.selectbox('Select Team', pd.unique(tempp['Team']), key='102')
        with col3:
            tempp = tempp[tempp['Team']==klub]
            pos = st.selectbox('Select Position', pd.unique(tempp['Position']), key='103')
        with col4:
            tempp = tempp[tempp['Position']==pos]
            ply = st.selectbox('Select Player', pd.unique(tempp['Name']), key='104')

        col5, col6 = st.columns(2)
        with col5:
            rdr = get_radar(rank_pct,rank_p90,rank_tot,pos,ply)
            rdr['Percentile'] = rdr['Percentile']/100
            st.subheader(ply+' Scouting Report')
            st.caption('vs '+pos+' in '+komp+' | Min. '+str(mins)+' mins played')
            st.data_editor(rdr, column_config={'Percentile':st.column_config.ProgressColumn('Percentile',width='medium',min_value=0,max_value=1)},hide_index=True)
        
        with col6:
            smr = get_simi(rank_p90,df2,ply,pos)
            st.subheader('Similar Players to '+ply)
            st.dataframe(smr.head(7), hide_index=True)
        col7, col8 = st.columns(2)
        with col7:
            piz = beli_pizza(komp, pos, klub, ply, rank_pct, mins)
            with open('pizza.jpg', 'rb') as img:
                fn = 'Perf.Radar_'+ply+'.jpg'
                btn = st.download_button(label="Download Report as a Radar!", data=img,
                                         file_name=fn, mime="image/jpg")
        with col8:
            #mirip = smr.head(7)
            ply2 = st.selectbox('Select Similar Player', pd.unique(smr.head(7)['Name']), key='105')
            cpre = plot_compare(ply, ply2, pos, rank_p90)
            st.pyplot(cpre)
    with pse:
        db_temp = get_detail(df2)
        db_temp2 = db_temp[['Name','Age Group','Nat. Status']]
        temple = pd.merge(rank_pct, db_temp2, on='Name', how='left')
        templist = rank_pct.drop(['Name','Position','Team','MoP','Kompetisi'], axis=1)
        metlist = list(templist)
        col1, col2, col3 = st.columns(3)
        with col1:
            pos = st.selectbox('Select Position', pd.unique(temple['Position']), key='87')
            komp = st.multiselect('Select Competition', ['Liga 1', 'Liga 2'], key='89')
        with col2:
            nats = st.multiselect('Select Nat. Status', ['Foreign', 'Local'], key='86')
            ages = st.multiselect('Select Age Groups', ['Senior', 'U23'], key='88')
        with col3:
            mins = st.number_input('Input minimum mins. played', min_value=90,
                                   max_value=3060, step=90, key=85)
            arr_met = st.multiselect('Select Metrics', metlist, key='84')
            
        playlist = get_playerlist(temple, komp, pos, mins, nats, ages, arr_met)
        st.dataframe(playlist.head(10))
