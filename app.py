import streamlit as st
import pandas as pd
import numpy as np
import os
import glob

st.set_page_config(page_title='Lapangbola Statistical Dashboard')
st.title('Lapangbola Statistical Dashboard')
st.markdown('Created by: Prana - R&D Division Lapangbola.com')

fixt1 = pd.read_excel('/app/dashboardlapangbola/data/fixtureliga1_23.xlsx')
fixt1['GW'] = fixt1['GW'].astype(int)

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
    
    with fstats:
        fstats.subheader('Test 2')

with tab2:
    tab2.subheader('Team Statistics')
    
with tab3:
    tab3.subheader('Player Statistics')
