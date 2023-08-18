import sys
import streamlit as st
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
import urllib

from mplsoccer import Pitch, VerticalPitch, PyPizza, FontManager
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

github_url = 'https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Bold.ttf'
url = github_url + '?raw=true'
response = urllib.request.urlopen(url)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()
bold = fm.FontProperties(fname=f.name)

github_url = 'https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Regular.ttf'
url = github_url + '?raw=true'
response = urllib.request.urlopen(url)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()
reg = fm.FontProperties(fname=f.name)

path_eff = [path_effects.Stroke(linewidth=2, foreground='#ffffff'),
            path_effects.Normal()]

@st.cache_data(ttl=600)
def load_data(sheets_url):
    xlsx_url = sheets_url.replace("/edit#gid=", "/export?format=xlsx&gid=")
    return pd.read_excel(xlsx_url)

shots_data = load_data(st.secrets["data_shots"])
fixt1 = load_data(st.secrets["fixture"])
fixt1['GW'] = fixt1['GW'].astype(int)
df1 = load_data(st.secrets["testaja"])
df2 = load_data(st.secrets["datapemain"])
from datetime import date
df1['Date'] = pd.to_datetime(df1.Date)
df1['Month'] = df1['Date'].dt.strftime('%B')
df = pd.merge(df1, df2.drop(['Player ID'], axis=1), on='Name', how='left')
fulldata = get_detail(df)
mlist = get_list(df)

tab1, tab2, tab3 = st.tabs(['**Competitions**', '**Teams**', '**Players**'])

with tab1:
    tab1.subheader('Competition Statistics')
    st.markdown('Untuk melihat statistik tiap pertandingan dan statistik full liga selama satu musim penuh.')
    mstats, fstats = st.tabs(['Match Stats', 'Full Stats'])
    with mstats:
        col1, col2, col3 = st.columns(3)
        with col1:
            komp = st.selectbox('Select Competition', ['Liga 1', 'Liga 2', 'Piala Indonesia'], key='0')
        with col2:
            pekan = st.selectbox('Select Stage/Gameweek', pd.unique(fixt1['GW']), key='1')
        with col3:
            temp0 = fixt1[fixt1['GW']==pekan].reset_index(drop=True)
            match = st.selectbox('Select Match', pd.unique(temp0['Match']), key='2')
            
        temp_fixt = fixt1[fixt1['Match']==match].reset_index(drop=True)
        home_team = temp_fixt['Home'][0]
        away_team = temp_fixt['Away'][0]
        temp1 = shots_data[shots_data['GW']==pekan].reset_index(drop=True)
        plot_data = temp1[(temp1['Team']==home_team) | (temp1['Team']==away_team)].reset_index(drop=True)

        #PLOT 1
        dataxg = assign_xg(plot_data)
        datapsxg = assign_psxg(plot_data)

        goals_h = dataxg[dataxg['Team']==home_team][dataxg['Event']=='Goal']['Event'].count()
        xgtot_h = round(dataxg[dataxg['Team']==home_team]['xG'].sum(),2)
        shots_h = dataxg[dataxg['Team']==home_team]['Event'].count()
        xgps_h = round((xgtot_h/shots_h),2)

        goals_a = dataxg[dataxg['Team']==away_team][dataxg['Event']=='Goal']['Event'].count()
        xgtot_a = round(dataxg[dataxg['Team']==away_team]['xG'].sum(),2)
        shots_a = dataxg[dataxg['Team']==away_team]['Event'].count()
        xgps_a = round((xgtot_a/shots_a),2)

        hgkq = round((sum(datapsxg[datapsxg['Team']==home_team]['PSxG'])-float(goals_a)),1)
        if (hgkq>0):
          gkqh = '+'+str(hgkq)
        else:
          gkqh = hgkq

        agkq = round((sum(datapsxg[datapsxg['Team']==away_team]['PSxG'])-float(goals_h)),1)
        if (agkq>0):
          gkqa = '+'+str(agkq)
        else:
          gkqa = agkq

        fig, ax = plt.subplots(figsize=(20, 20), dpi=500)

        pitch = Pitch(pitch_type='wyscout', corner_arcs=True, line_alpha=.65,
                      pitch_color='#ffffff', line_color='#000000', goal_alpha=.65,
                      stripe_color='#fcf8f7', goal_type='box',
                      pad_bottom=5, linewidth=3.5, stripe=True)
        pitch.draw(ax=ax)
        fig.patch.set_facecolor('#ffffff')

        data_home = dataxg[dataxg['Team']==home_team]
        data_home['X'] = 100-data_home['X']
        data_home['Y'] = 100-data_home['Y']
        data_away = dataxg[dataxg['Team']==away_team]

        ax.scatter(data_home[data_home['Event']=='Goal']['X'], data_home[data_home['Event']=='Goal']['Y'],
                   s=data_home[data_home['Event']=='Goal']['xG']*3000,
                   c='#cbfd06', marker='o', ec='#000000', lw=3)
        ax.scatter(data_home[data_home['Event']!='Goal']['X'], data_home[data_home['Event']!='Goal']['Y'],
                   s=data_home[data_home['Event']!='Goal']['xG']*3000,
                   c='#a6a6a6', marker='o', ec='#000000', lw=3)

        ax.scatter(data_away[data_away['Event']=='Goal']['X'], data_away[data_away['Event']=='Goal']['Y'],
                   s=data_away[data_away['Event']=='Goal']['xG']*3000,
                   c='#cbfd06', marker='o', ec='#000000', lw=3)
        ax.scatter(data_away[data_away['Event']!='Goal']['X'], data_away[data_away['Event']!='Goal']['Y'],
                   s=data_away[data_away['Event']!='Goal']['xG']*3000,
                   c='#a6a6a6', marker='o', ec='#000000', lw=3)

        annot_texts = ['Goals', 'Expected Goals (xG)', 'Shots', 'xG/Shot', 'GK Quality (PSxG-xG)']
        annot_y = [25 + y*9 for y in range(0,5)]
        annot_stats_h = [goals_h, xgtot_h, shots_h, xgps_h, gkqh]
        annot_stats_a = [goals_a, xgtot_a, shots_a, xgps_a, gkqa]

        for y, s, h, a in zip(annot_y, annot_texts, annot_stats_h, annot_stats_a):
          ax.add_patch(FancyBboxPatch((37.5, y), 25, 5, fc='#000000', ec='#ffffff',
                                      alpha=0.15, boxstyle=patches.BoxStyle('Round', pad=1)))
          ax.add_patch(FancyBboxPatch((32, y), 3, 5, fc='#000000', ec='#ffffff',
                                      alpha=0.15, boxstyle=patches.BoxStyle('Round', pad=1)))
          ax.add_patch(FancyBboxPatch((65, y), 3, 5, fc='#000000', ec='#ffffff',
                                      alpha=0.15, boxstyle=patches.BoxStyle('Round', pad=1)))
          ax.annotate(text=s, size=22, xy=(50,y), xytext=(0,-18),
                      textcoords='offset points', color='#000000', ha='center', fontproperties=bold,
                      zorder=9, va='center', path_effects=path_eff)
          ax.annotate(text=h, size=22, xy=(33.5,y), xytext=(0,-18),
                      textcoords='offset points', color='#000000', ha='center', fontproperties=bold,
                      zorder=9, va='center', path_effects=path_eff)
          ax.annotate(text=a, size=22, xy=(66.5,y), xytext=(0,-18),
                      textcoords='offset points', color='#000000', ha='center', fontproperties=bold,
                      zorder=9, va='center', path_effects=path_eff)

        ax.scatter(32, 95, s=500, c='#a6a6a6', marker='o', ec='#000000', lw=3)
        ax.text(34, 95, 'Shots', ha='left', fontproperties=bold, color='#000000', size='18', va='center')
        ax.scatter(42, 95, s=500, c='#cbfd06', marker='o', ec='#000000', lw=3)
        ax.text(44, 95, 'Goals', ha='left', fontproperties=bold, color='#000000', size='18', va='center')

        ax.text(52.5, 91, '-xG value->', ha='left', fontproperties=bold, color='#000000', size='18', va='center')
        annot_x = [53 + x*4 for x in range(0,3)]
        annot_s = [500 + s*200 for s in range(0,3)]
        for x, s in zip(annot_x, annot_s):
          ax.scatter(x, 95, s=s, c='#a6a6a6', marker='o', ec='#000000', lw=3)
        st.pyplot(fig)
    
    with fstats:
        table, teams, players = st.tabs(['Standing & Top Stats', 'Team Stats', 'Player Stats'])
        with teams:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                komp = st.selectbox('Select Competition', ['Liga 1', 'Liga 2'], key='3')
            with col2:
                temp_full = fulldata[fulldata['Competition']==komp]
                month = st.multiselect('Select Month', pd.unique(temp_full['Month']), key='14')
            with col3:
                temp_full = fulldata[fulldata['Month'].isin(month)]
                venue = st.multiselect('Select Venue', pd.unique(temp_full['Home/Away']), key='5')
            with col4:
                temp_full = temp_full[temp_full['Home/Away'].isin(venue)]
                gw = st.multiselect('Select Gameweek', pd.unique(temp_full['Gameweek']), key='4')
            with col5:
                cat = st.selectbox('Select Category', ['Goal Threat', 'in Possession', 'out of Possession', 'Misc'], key='13')
            show_tim_data = data_team(fulldata, komp, month, gw, venue, cat)
            st.write(show_tim_data)
        with players:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                komp = st.selectbox('Select Competition', ['Liga 1', 'Liga 2'], key='6')
                temp_pull = fulldata[fulldata['Competition']==komp]
                team = st.selectbox('Select Teams', pd.unique(temp_pull['Team']), key='10')
            with col2:
                temp_pull = temp_pull[temp_pull['Team'].isin(team)]
                pos = st.multiselect('Select Positions', pd.unique(temp_pull['Positions']), key='7')
                temp_pull = temp_pull[temp_pull['Position'].isin(pos)]
                age = st.multiselect('Select Age Group', pd.unique(temp_pull['Age Group']), key='11')
            with col3:
                temp_pull = temp_pull[temp_pull['Age Group'].isin(age)]
                nat = st.selectbox('Select Nationality', pd.unique(temp_pull['Nat. Status']), key='8')
                temp_pull = temp_pull[temp_pull['Nat. Status'].isin(nat)]
                month = st.selectbox('Select Month', pd.unique(temp_pull['Month']), key='12')   
            with col4:
                temp_pull = temp_pull[temp_pull['Month'].isin(month)]
                venue = st.selectbox('Select Venue', pd.unique(temp_pull['Home/Away']), key='9')
                temp_pull = temp_pull[temp_pull['Home/Away'].isin(venue)]
                gw = st.multiselect('Select Gameweek', pd.unique(temp_pull['Gameweek']), key='13')
            with col5:
                mins = st.number_input('Input minimum mins. played', min_value=0,
                                       max_value=90*max(full_data['Gameweek']), step=90, key=14)
                metrik = st.multiselect('Select Metrics', mlist, key='15')
            cat = st.selectbox('Select Category', ['Total', 'per 90'], key='16')
            if (cat == 'per 90'):
                show_player_data = data_player(fulldata, komp, team, pos, month, venue, gw, age, nat, metrik, mins, cat=='p90')
                st.write(show_player_data)
            else:
                show_player_data = data_player(fulldata, komp, team, pos, month, venue, gw, age, nat, metrik, mins, cat=='total')
                st.write(show_player_data)
            

with tab2:
    tab2.subheader('Team Statistics')
    
with tab3:
    tab3.subheader('Player Statistics')
