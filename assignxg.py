import sys
import io

sys.path.append("xgmodel.py")
import xgmodel
from xgmodel import calculate_xG
from xgmodel import xgfix

sys.path.append("psxgmodel.py")
import psxgmodel
from psxgmodel import calculate_PSxG
from psxgmodel import psxgfix

import os
import pandas as pd
import glob
from datetime import date
import numpy as np
import math

from sklearn import preprocessing
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

posdict = {'gk':{'position':'Goalkeeper',
                 'metrics':['Name','Long Goal Kick Ratio','Pass Accuracy','Cross Claim',
                            'Keeper - Sweeper','Saves','Save Ratio','Penalty Save']},
           'cb':{'position':'Center Back',
                 'metrics':['Name','Shots','Goals','Assist','Pass Accuracy',
                            'Tackle','Intercept','Recovery','Blocks','Aerial Won Ratio']},
           'fb':{'position':'Fullback',
                 'metrics':['Name','Shots','Goals','Create Chance','Assist','Pass Accuracy','Dribble',
                            'Cross','Tackle','Intercept','Recovery','Blocks','Aerial Won Ratio']},
           'cm':{'position':'Midfielder',
                 'metrics':['Name','Shots','Goals','Create Chance','Shot on Target Ratio','Assist',
                            'Pass Accuracy','Dribble','Tackle','Intercept','Recovery','Blocks']},
           'cam/w':{'position':'Attacking 10/Winger',
                    'metrics':['Name','Shots','Goals','Create Chance','Shot on Target Ratio',
                               'Conversion Ratio','Assist','Pass Accuracy','Dribble','Cross',
                               'Tackle','Intercept','Recovery']},
           'fw':{'position':'Forward',
                 'metrics':['Name','Shots','Goals','Create Chance','Shot on Target Ratio',
                            'Conversion Ratio','Assist','Pass Accuracy','Dribble','Tackle',
                            'Intercept','Recovery','Aerial Won Ratio']}}

def assign_xg(data):
  df_match = data.copy()

  #Filtering Data
  df_match = df_match[['Team','Act Name','Action', 'Min', 'Sub 1', 'Sub 2', 'Sub 3', 'Sub 4', 'GW', 'X', 'Y']]
  df_match = df_match[(df_match['Action']=='shoot on target') | (df_match['Action']=='shoot off target') | (df_match['Action']=='shoot blocked') | (df_match['Action']=='goal') | (df_match['Action']=='penalty goal') | (df_match['Action']=='penalty missed')]
  df_match = df_match.reset_index()
  df_match = df_match.sort_values(by=['index'], ascending=False)

  #Cleaning Data
  shots = df_match.copy()
  shots['Mins'] = shots['Min']
  shots['Mins'] = shots['Mins'].astype(float)
  shots = shots[shots['X'].notna()]

  data = shots[['Act Name', 'Team', 'Action', 'Mins', 'Sub 1', 'Sub 2', 'Sub 3', 'Sub 4', 'X', 'Y']]
  data.loc[(data['Action'].str.contains('penalty goal')), 'Sub 3'] = 'Penalty'
  data.loc[(data['Action'].str.contains('penalty goal')), 'Sub 1'] = 'Right Foot'
  data.loc[(data['Action'].str.contains('penalty missed')), 'Sub 3'] = 'Penalty'
  data.loc[(data['Action'].str.contains('penalty missed')), 'Sub 2'] = 'Right Foot'
  data.loc[(data['Action'].str.contains('penalty')), 'X'] = 90
  data.loc[(data['Action'].str.contains('penalty')), 'Y'] = 50
  data.loc[(data['Action'].str.contains('penalty missed')) & ((data['Sub 1'].str.contains('Saved')) | (data['Sub 1'].str.contains('Cleared'))), 'Action'] = 'shoot on target'

  data['Action'] = data['Action'].replace(['shoot on target','shoot off target','shoot blocked','goal','penalty goal','penalty missed'],
                                        ['Shot On','Shot Off','Shot Blocked','Goal','Goal','Shot Off'])
  dft = data.groupby('Action', as_index=False)
  temp = pd.DataFrame(columns = ['Player', 'Team','Event','Mins',
                                 'Shot Type','Situation','X','Y'])
  if (('Goal' in data['Action'].unique()) == True):
    df1 = dft.get_group('Goal')
    df1f = df1[['Act Name', 'Team', 'Action', 'Mins', 'Sub 1', 'Sub 3', 'X', 'Y']]
    df1f.rename(columns = {'Action':'Event', 'Sub 1':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
  else:
    df1f = temp.copy()

  if (('Shot On' in data['Action'].unique()) == True):
    df2 = dft.get_group('Shot On')
    df2f = df2[['Act Name','Team', 'Action', 'Mins', 'Sub 2', 'Sub 3', 'X', 'Y']]
    df2f.rename(columns = {'Action':'Event', 'Sub 2':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
  else:
    df2f = temp.copy()

  if (('Shot Off' in data['Action'].unique()) == True):
    df3 = dft.get_group('Shot Off')
    if (('Penalty' in df3['Sub 3'].unique()) == True):
      df3f = df3[['Act Name','Team', 'Action', 'Mins', 'Sub 2', 'Sub 3', 'X', 'Y']]
      df3f.rename(columns = {'Action':'Event', 'Sub 2':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
    else:
      df3f = df3[['Act Name','Team', 'Action', 'Mins', 'Sub 3', 'Sub 4', 'X', 'Y']]
      df3f.rename(columns = {'Action':'Event', 'Sub 3':'Shot Type', 'Sub 4':'Situation', 'Act Name':'Player'}, inplace = True)
  else:
    df3f = temp.copy()

  if (('Shot Blocked' in data['Action'].unique()) == True):
    df4 = dft.get_group('Shot Blocked')
    df4f = df4[['Act Name','Team', 'Action', 'Mins', 'Sub 2', 'Sub 3', 'X', 'Y']]
    df4f.rename(columns = {'Action':'Event', 'Sub 2':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
  else:
    df4f = temp.copy()

  sa = pd.concat([df1f, df2f, df3f, df4f])
  sa = sa.dropna()
  sa.loc[(sa['Situation'].str.contains('Open play')), 'Situation'] = 'Open Play'
  sa.loc[(sa['Situation'].str.contains('Freekick')), 'Situation'] = 'Set-Piece Free Kick'
  sa.loc[(sa['Shot Type'].str.contains('Header')), 'Shot Type'] = 'Head'

  df_co = sa.sort_values(by=['Mins'], ascending=False)
  df_co['x'] = 100-df_co['X']
  df_co['y'] = df_co['Y']
  df_co['c'] = abs(df_co['Y']-50)

  x=df_co['x']*1.05
  y=df_co['c']*0.68

  df_co['X2']=(100-df_co['X'])*1.05
  df_co['Y2']=df_co['Y']*0.68

  df_co['Distance'] = np.sqrt(x**2 + y**2)
  c=7.32
  a=np.sqrt((y-7.32/2)**2 + x**2)
  b=np.sqrt((y+7.32/2)**2 + x**2)
  k = (c**2-a**2-b**2)/(-2*a*b)
  gamma = np.arccos(k)
  if gamma.size<0:
    gamma = np.pi + gamma
  df_co['Angle Rad'] = gamma
  df_co['Angle Degrees'] = gamma*180/np.pi

  def wasitgoal(row):
    if row['Event'] == 'Goal':
      return 1
    else:
      return 0
  df_co['goal'] = df_co.apply(lambda row: wasitgoal(row), axis=1)

  df_co = df_co.sort_values(by=['Mins'])
  df_co = df_co.reset_index()
  shotdata = df_co[['Player','Team','Event','Mins','Shot Type','Situation','X2','Y2','Distance','Angle Rad','Angle Degrees','goal']]

  shots = shotdata.dropna()
  shots.rename(columns = {'Player':'player', 'Event':'event', 'Mins':'mins', 'Shot Type':'shottype',
                          'Situation':'situation','X2':'X','Y2':'Y', 'Distance':'distance',
                          'Angle Rad':'anglerad','Angle Degrees':'angledeg'}, inplace = True)

  body_part_list=[]
  for index, rows in shots.iterrows():
      if (rows['shottype']=='Right Foot') or (rows['shottype']=='Left Foot'):
          body_part_list.append('Foot')
      elif (rows['shottype']=='Head'):
          body_part_list.append('Head')
      else:
          body_part_list.append('Head')

  shots['body_part']=body_part_list

  shots=shots[shots.body_part != 'Other']
  shots=shots.sort_values(by=['mins'])
  shots.loc[(shots['situation'].str.contains('Set')), 'situation'] = 'Indirect'
  shots.loc[(shots['situation'].str.contains('Corner')), 'situation'] = 'Indirect'
  shots.loc[(shots['situation'].str.contains('Throw')), 'situation'] = 'Open Play'
  shots.loc[(shots['situation'].str.contains('Counter')), 'situation'] = 'Open Play'
  dfxg = shots[['distance', 'angledeg', 'body_part', 'situation', 'goal']]

  #Assign numerical value to body part and situation
  label_encoder = preprocessing.LabelEncoder()
  label_encoder.fit(dfxg["body_part"])
  dfxg['body_part_num'] = label_encoder.transform(dfxg["body_part"])

  label_encoder = preprocessing.LabelEncoder()
  label_encoder.fit(dfxg["situation"])
  dfxg['situation_num'] = label_encoder.transform(dfxg["situation"])

  #assigning xG
  xG=dfxg.apply(calculate_xG, axis=1)
  dfxg = dfxg.assign(xG=xG)
  dfxg['xG'] = dfxg.apply(lambda row: xgfix(row), axis=1)

  fixdata = df_co[['Player', 'Team', 'Event', 'Mins', 'X', 'Y']]
  fixdata['xG'] = dfxg['xG']

  return fixdata

def assign_psxg(data):
  df_match = data.copy()

  #Filtering Data
  df_match = df_match[['Team','Act Name','Action', 'Min', 'Sub 1', 'Sub 2', 'Sub 3', 'Sub 4', 'GW', 'X', 'Y']]
  df_match = df_match[(df_match['Action']=='shoot on target') | (df_match['Action']=='shoot off target') | (df_match['Action']=='shoot blocked') | (df_match['Action']=='goal') | (df_match['Action']=='penalty goal') | (df_match['Action']=='penalty missed')]
  df_match = df_match.reset_index()
  df_match = df_match.sort_values(by=['index'], ascending=False)

  #Cleaning Data
  shots = df_match.copy()
  shots['Mins'] = shots['Min']
  shots['Mins'] = shots['Mins'].astype(float)
  shots = shots[shots['X'].notna()]

  data = shots[['Act Name', 'Team', 'Action', 'Mins', 'Sub 1', 'Sub 2', 'Sub 3', 'Sub 4', 'X', 'Y']]
  data.loc[(data['Action'].str.contains('penalty goal')), 'Sub 3'] = 'Penalty'
  data.loc[(data['Action'].str.contains('penalty goal')), 'Sub 1'] = 'Right Foot'
  data.loc[(data['Action'].str.contains('penalty missed')), 'Sub 3'] = 'Penalty'
  data.loc[(data['Action'].str.contains('penalty missed')), 'Sub 2'] = 'Right Foot'
  data.loc[(data['Action'].str.contains('penalty')), 'X'] = 90
  data.loc[(data['Action'].str.contains('penalty')), 'Y'] = 50
  data.loc[(data['Action'].str.contains('penalty missed')) & ((data['Sub 1'].str.contains('Saved')) | (data['Sub 1'].str.contains('Cleared'))), 'Action'] = 'shoot on target'

  data['Action'] = data['Action'].replace(['shoot on target','shoot off target','shoot blocked','goal','penalty goal','penalty missed'],
                                          ['Shot On','Shot Off','Shot Blocked','Goal','Goal','Shot Off'])
  dft = data.groupby('Action', as_index=False)
  temp = pd.DataFrame(columns = ['Player', 'Team','Event','Mins',
                                 'Shot Type','Situation','X','Y'])
  if (('Goal' in data['Action'].unique()) == True):
    df1 = dft.get_group('Goal')
    df1f = df1[['Act Name', 'Team', 'Action', 'Mins', 'Sub 1', 'Sub 3', 'X', 'Y']]
    df1f.rename(columns = {'Action':'Event', 'Sub 1':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
  else:
    df1f = temp.copy()

  if (('Shot On' in data['Action'].unique()) == True):
    df2 = dft.get_group('Shot On')
    df2f = df2[['Act Name','Team', 'Action', 'Mins', 'Sub 2', 'Sub 3', 'X', 'Y']]
    df2f.rename(columns = {'Action':'Event', 'Sub 2':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
  else:
    df2f = temp.copy()

  sa = pd.concat([df1f, df2f])
  sa = sa.dropna()
  sa.loc[(sa['Situation'].str.contains('Open play')), 'Situation'] = 'Open Play'
  sa.loc[(sa['Situation'].str.contains('Freekick')), 'Situation'] = 'Set-Piece Free Kick'
  sa.loc[(sa['Shot Type'].str.contains('Header')), 'Shot Type'] = 'Head'

  df_co = sa.sort_values(by=['Mins'], ascending=False)
  df_co['x'] = 100-df_co['X']
  df_co['y'] = df_co['Y']
  df_co['c'] = abs(df_co['Y']-50)

  x=df_co['x']*1.05
  y=df_co['c']*0.68

  df_co['X2']=(100-df_co['X'])*1.05
  df_co['Y2']=df_co['Y']*0.68

  df_co['Distance'] = np.sqrt(x**2 + y**2)
  c=7.32
  a=np.sqrt((y-7.32/2)**2 + x**2)
  b=np.sqrt((y+7.32/2)**2 + x**2)
  k = (c**2-a**2-b**2)/(-2*a*b)
  gamma = np.arccos(k)
  if gamma.size<0:
    gamma = np.pi + gamma
  df_co['Angle Rad'] = gamma
  df_co['Angle Degrees'] = gamma*180/np.pi

  def wasitgoal(row):
    if row['Event'] == 'Goal':
      return 1
    else:
      return 0
  df_co['goal'] = df_co.apply(lambda row: wasitgoal(row), axis=1)

  df_co = df_co.sort_values(by=['Mins'])
  df_co = df_co.reset_index()
  shotdata = df_co[['Player','Team','Event','Mins','Shot Type','Situation','X2','Y2','Distance','Angle Rad','Angle Degrees','goal']]

  shots = shotdata.dropna()
  shots.rename(columns = {'Player':'player', 'Event':'event', 'Mins':'mins', 'Shot Type':'shottype',
                          'Situation':'situation','X2':'X','Y2':'Y', 'Distance':'distance',
                          'Angle Rad':'anglerad','Angle Degrees':'angledeg'}, inplace = True)

  body_part_list=[]
  for index, rows in shots.iterrows():
      if (rows['shottype']=='Right Foot') or (rows['shottype']=='Left Foot'):
          body_part_list.append('Foot')
      elif (rows['shottype']=='Head'):
          body_part_list.append('Head')
      else:
          body_part_list.append('Head')

  shots['body_part']=body_part_list

  shots=shots[shots.body_part != 'Other']
  shots=shots.sort_values(by=['mins'])
  shots.loc[(shots['situation'].str.contains('Set')), 'situation'] = 'Indirect'
  shots.loc[(shots['situation'].str.contains('Corner')), 'situation'] = 'Indirect'
  shots.loc[(shots['situation'].str.contains('Throw')), 'situation'] = 'Open Play'
  shots.loc[(shots['situation'].str.contains('Counter')), 'situation'] = 'Open Play'
  psxg = shots[['distance', 'angledeg', 'body_part', 'situation', 'goal']]

  #Assign numerical value to body part and situation
  label_encoder = preprocessing.LabelEncoder()
  label_encoder.fit(psxg["body_part"])
  psxg['body_part_num'] = label_encoder.transform(psxg["body_part"])

  label_encoder = preprocessing.LabelEncoder()
  label_encoder.fit(psxg["situation"])
  psxg['situation_num'] = label_encoder.transform(psxg["situation"])

  #assigning xG
  PSxG=psxg.apply(calculate_PSxG, axis=1)
  psxg = psxg.assign(PSxG=PSxG)
  psxg['PSxG'] = psxg.apply(lambda row: psxgfix(row), axis=1)

  fixdata = df_co[['Player', 'Team', 'Event', 'Mins', 'X', 'Y']]
  fixdata['PSxG'] = psxg['PSxG']

  return fixdata

def add_og(data):
  df = data.copy()
  dfog = df[['Match', 'Team', 'Opponent', 'Own Goal']]
  dfog = dfog.groupby(['Match', 'Team', 'Opponent'], as_index=False).sum()
  dfog['Team'] = dfog['Opponent']
  dfog['Goal - Own Goal'] = dfog['Own Goal']
  df_clean = dfog[['Team', 'Goal - Own Goal']]
  df_clean = df_clean.groupby('Team', as_index=False).sum()
  return df_clean

def add_conc(data):
  df = data.copy()
  conc = df[['Opponent','Goal','Penalty Goal','Shot on','Shot off','Shot Blocked']]
  conc['Goals Conceded'] = conc['Goal']+conc['Penalty Goal']
  conc['Shots Allowed'] = conc['Shot on']+conc['Shot off']+conc['Shot Blocked']
  conc = conc[['Opponent','Goals Conceded','Shots Allowed']].rename(columns={'Opponent':'Team'})
  df_clean1 = conc.groupby('Team', as_index=False).sum()

  dfog = df[['Team','Own Goal']]
  df_clean2 = dfog.groupby('Team', as_index=False).sum()

  df_clean = pd.merge(df_clean1, df_clean2, on='Team', how='left')
  df_clean['Goals Conceded'] = df_clean['Goals Conceded']+df_clean['Own Goal']
  df_clean = df_clean[['Team','Goals Conceded','Shots Allowed']]

  return df_clean

def data_team(data, komp, month, gw, venue, cat):
  df = data.copy()
  df_og = data.copy()
  gw_list = gw
  vn_list = venue
  mn_list = month

  from datetime import date
  df['Date'] = pd.to_datetime(df.Date)
  df['Month'] = df['Date'].dt.strftime('%B')

  df = df[df['Kompetisi']==komp]
  df = df[df['Home/Away'].isin(vn_list)]
  df = df[df['Gameweek'].isin(gw_list)]
  df = df[df['Month'].isin(mn_list)]

  dfog = add_og(df)
  dfconc = add_conc(df)

  df['Shots'] = df['Shot on']+df['Shot off']+df['Shot Blocked']
  df['Goals'] = df['Penalty Goal']+df['Goal']
  df['Penalties Awarded'] = df['Penalty Goal']+df['Penalty Missed']
  df['Shots - Inside Box'] = df['Shot on - Inside Box']+df['Shot off - Inside Box']+df['Shot Blocked - Inside Box']
  df['Shots - Outside Box'] = df['Shot on - Outside Box']+df['Shot off - Outside Box']+df['Shot Blocked - Outside Box']
  df['Goals - Inside Box'] = df['Penalty Goal']+df['Goal - Inside Box']
  df['Goals - Open Play'] = df['Goal - Open Play']+df['Goal - Counter Attack']
  df['Goals - Set Pieces'] = df['Goal - Set-Piece Free Kick']+df['Goal - Throw in']+df['Goal - Corner Kick']
  df['Total Pass'] = df['Pass']+df['Pass Fail']
  df['Chances Created'] = df['Key Pass']+df['Assist']
  df['Crosses'] = df['Cross']+df['Cross Fail']
  df['Dribbles'] = df['Dribble']+df['Dribble Fail']
  df['Tackles'] = df['Tackle']+df['Tackle Fail']
  df['Defensive Actions'] = df['Tackles']+df['Intercept']+df['Clearance']+df['Recovery']
  df['Saves'] = df['Save']+df['Penalty Save']
  df['Blocks'] = df['Block']+df['Block Cross']
  df['Aerial Duels'] = df['Aerial Won']+df['Aerial Lost']
  df['Errors'] = df['Error Goal - Error Led to Chance'] + df['Error Goal - Error Led to Goal']

  df = df[['Team', 'Shots', 'Shot on', 'Shot off', 'Shot Blocked',
           'Shots - Inside Box', 'Shots - Outside Box',
           'Goals', 'Penalties Awarded', 'Penalty Goal',
           'Goals - Inside Box', 'Goal - Outside Box', 'Goals - Open Play',
           'Goals - Set Pieces', 'Goal - Corner Kick', 'Total Pass', 'Pass',
           'Chances Created', 'Crosses', 'Cross', 'Dribbles', 'Dribble',
           'Tackles', 'Intercept', 'Clearance', 'Recovery', 'Defensive Actions',
           'Blocks', 'Aerial Duels', 'Aerial Won', 'Saves',
           'Errors', 'Error Goal - Error Led to Chance', 'Error Goal - Error Led to Goal',
           'Own Goal', 'Foul', 'Yellow Card', 'Red Card', 'Offside', 'Goal Kick - Goal Kick Launch',
           'Goal Kick - No Sub-action', 'Goal Kick', 'Pass - Long Ball']]
  df = df.groupby('Team', as_index=False).sum()
  df['Conversion Ratio'] = round(df['Goals']/df['Shots'],2)
  df['Shot on Target Ratio'] = round(df['Shot on']/df['Shots'],2)
  df['Successful Cross Ratio'] = round(df['Cross']/df['Crosses'],2)
  df['Pass per Shot'] = round(df['Total Pass']/df['Shots'],2)
  df['Pass Accuracy'] = round(df['Pass']/df['Total Pass'],2)
  df['Aerial Won Ratio'] = round(df['Aerial Won']/df['Aerial Duels'],2)
  df['Goal Kick Grounded Ratio'] = round(df['Goal Kick - No Sub-action']/df['Goal Kick'],2)
  df['Long Ball Ratio'] = round(df['Pass - Long Ball']/df['Pass'],2)

  df1 = pd.merge(df, dfog, on='Team', how='left')
  df2 = pd.merge(df1, dfconc, on='Team', how='left')
  df2['Goals'] = df2['Goals'] + df2['Goal - Own Goal']

  df2 = df2[['Team', 'Shots', 'Shot on', 'Shot off', 'Shot Blocked', 'Shot on Target Ratio',
             'Shots - Inside Box', 'Shots - Outside Box', 'Goals', 'Penalties Awarded', 'Penalty Goal',
             'Goals - Inside Box', 'Goal - Outside Box', 'Goals - Open Play', 'Goals - Set Pieces',
             'Goal - Corner Kick', 'Goal - Own Goal', 'Total Pass', 'Pass', 'Pass Accuracy',
             'Chances Created', 'Pass per Shot', 'Crosses', 'Cross', 'Successful Cross Ratio',
             'Dribbles', 'Dribble', 'Tackles', 'Intercept', 'Clearance', 'Recovery', 'Defensive Actions',
             'Blocks', 'Aerial Duels', 'Aerial Won', 'Aerial Won Ratio', 'Saves', 'Goals Conceded', 'Shots Allowed',
             'Errors', 'Error Goal - Error Led to Chance', 'Error Goal - Error Led to Goal',
             'Own Goal', 'Foul', 'Yellow Card', 'Red Card', 'Offside', 'Goal Kick Grounded Ratio', 'Long Ball Ratio',
             'Goal Kick - Goal Kick Launch', 'Goal Kick - No Sub-action']]

  gt = ['Team', 'Shots', 'Shot on', 'Shot off', 'Shot Blocked', 'Shot on Target Ratio', 'Shots - Inside Box',
        'Shots - Outside Box', 'Goals', 'Penalties Awarded', 'Penalty Goal', 'Goals - Inside Box', 'Goal - Outside Box',
        'Goals - Open Play', 'Goals - Set Pieces', 'Goal - Corner Kick', 'Goal - Own Goal']
  
  ip = ['Team', 'Total Pass', 'Pass', 'Pass Accuracy', 'Chances Created',
        'Pass per Shot', 'Crosses', 'Cross', 'Successful Cross Ratio', 'Dribbles', 'Dribble']
  
  op = ['Team', 'Tackles', 'Intercept', 'Clearance', 'Recovery', 'Defensive Actions',
        'Blocks', 'Aerial Duels', 'Aerial Won', 'Aerial Won Ratio', 'Saves',
        'Goals Conceded', 'Shots Allowed']
  
  misc = ['Team', 'Errors', 'Error Goal - Error Led to Chance', 'Error Goal - Error Led to Goal',
          'Own Goal', 'Foul', 'Yellow Card', 'Red Card', 'Offside', 'Goal Kick Grounded Ratio',
          'Goal Kick - Goal Kick Launch', 'Goal Kick - No Sub-action']
  
  if (cat == 'Goal Threat'):
    df2 = df2[gt]
  elif (cat == 'in Possession'):
    df2 = df2[ip]
  elif (cat == 'out of Possession'):
    df2 = df2[op]
  else:
    df2 = df2[misc]

  return df2

def get_list(data):
  df = data.copy()
  df['Shots'] = df['Shot on']+df['Shot off']+df['Shot Blocked']
  df['Goals'] = df['Penalty Goal']+df['Goal']
  df['Goals Cont.'] = df['Goals']+df['Assist']
  df['Penalties Given'] = df['Penalty Goal']+df['Penalty Missed']
  df['Shots - Inside Box'] = df['Shot on - Inside Box']+df['Shot off - Inside Box']+df['Shot Blocked - Inside Box']
  df['Shots - Outside Box'] = df['Shot on - Outside Box']+df['Shot off - Outside Box']+df['Shot Blocked - Outside Box']
  df['Goals - Inside Box'] = df['Penalty Goal']+df['Goal - Inside Box']
  df['Goals - Open Play'] = df['Goal - Open Play']+df['Goal - Counter Attack']
  df['Goals - Set Pieces'] = df['Goal - Set-Piece Free Kick']+df['Goal - Throw in']+df['Goal - Corner Kick']
  df['Total Pass'] = df['Pass']+df['Pass Fail']
  df['Chances Created'] = df['Key Pass']+df['Assist']
  df['Crosses'] = df['Cross']+df['Cross Fail']
  df['Dribbles'] = df['Dribble']+df['Dribble Fail']
  df['Tackles'] = df['Tackle']+df['Tackle Fail']
  df['Defensive Actions'] = df['Tackles']+df['Intercept']+df['Clearance']+df['Recovery']
  df['Saves'] = df['Save']+df['Penalty Save']
  df['Blocks'] = df['Block']+df['Block Cross']
  df['Aerial Duels'] = df['Aerial Won']+df['Aerial Lost']
  df['Errors'] = df['Error Goal - Error Led to Chance'] + df['Error Goal - Error Led to Goal']

  jatuh = ['No','Player ID','Team ID','Position (in match)','Gameweek','Opponent','Match','Home/Away','Venue',
           'Date','Result','Starter/Subs','Subs','Player Rating','Ball Possession','Pass Team','Kick In','Unnamed: 296','DoB',
           'Unnamed: 297','Unnamed: 298','Unnamed: 299','Unnamed: 300','Unnamed: 301','Unnamed: 302','Unnamed: 303',
           'Fantasy Assist','Fantasy Assist - Penalty','Fantasy Assist - Free kick','Fantasy Assist - Goal by rebound',
           'Fantasy Assist - Own goal by pass/cross','Fantasy Assist - Own goal by rebound','Unnamed: 310','Kompetisi','Nickname']

  df = df.drop(jatuh, axis=1)
  df['Conversion Ratio'] = round(df['Goals']/df['Shots'],2)
  df['Shot on Target Ratio'] = round(df['Shot on']/df['Shots'],2)
  df['Successful Cross Ratio'] = round(df['Cross']/df['Crosses'],2)
  df['Pass per Shot'] = round(df['Total Pass']/df['Shots'],2)
  df['Pass Accuracy'] = round(df['Pass']/df['Total Pass'],2)
  df['Aerial Won Ratio'] = round(df['Aerial Won']/df['Aerial Duels'],2)
  df['Goal Kick Grounded Ratio'] = round(df['Goal Kick - No Sub-action']/df['Goal Kick'],2)
  df['Long Ball Ratio'] = round(df['Pass - Long Ball']/df['Pass'],2)

  metrik = list(df)
  return metrik

def get_detail(data):
  db = data.copy()
  import datetime as dt
  from datetime import date

  today = date.today()
  db['Age'] = db['DoB'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

  #adding age group based on the player's age
  def age_group(row):
    if row['DoB'] < dt.datetime(2000,1,1):
      return 'Senior'
    else:
      return 'U23'
  db['Age Group'] = db.apply(lambda row: age_group(row), axis=1)

  #adding nationality status
  def status(row):
    if row['Nationality'] == 'Indonesia':
      return 'Local'
    elif row['Nationality'] != 'Indonesia':
      return 'Foreign'

  db['Nat. Status'] = db.apply(lambda row: status(row), axis=1)

  return db

def data_player(data, komp, team, pos, month, venue, gw, age, nat, metrik, mins, cat, data2):
  df = data.copy()
  db = data2.copy()
  gw_list = gw
  vn_list = venue
  mn_list = month
  ag_list = age
  nt_list = nat
  ps_list = pos
  mt_list = metrik
  tm_list = team

  df = df[df['Kompetisi']==komp]
  df = df[df['Team'].isin(tm_list)]
  df = df[df['Home/Away'].isin(vn_list)]
  df = df[df['Gameweek'].isin(gw_list)]
  df = df[df['Month'].isin(mn_list)]
  df = df[df['Age Group'].isin(ag_list)]
  df = df[df['Position'].isin(ps_list)]
  df = df[df['Nat. Status'].isin(nt_list)]

  df['Shots'] = df['Shot on']+df['Shot off']+df['Shot Blocked']
  df['Goals'] = df['Penalty Goal']+df['Goal']
  df['Goals Cont.'] = df['Goals']+df['Assist']
  df['Penalties Given'] = df['Penalty Goal']+df['Penalty Missed']
  df['Shots - Inside Box'] = df['Shot on - Inside Box']+df['Shot off - Inside Box']+df['Shot Blocked - Inside Box']
  df['Shots - Outside Box'] = df['Shot on - Outside Box']+df['Shot off - Outside Box']+df['Shot Blocked - Outside Box']
  df['Goals - Inside Box'] = df['Penalty Goal']+df['Goal - Inside Box']
  df['Goals - Open Play'] = df['Goal - Open Play']+df['Goal - Counter Attack']
  df['Goals - Set Pieces'] = df['Goal - Set-Piece Free Kick']+df['Goal - Throw in']+df['Goal - Corner Kick']
  df['Total Pass'] = df['Pass']+df['Pass Fail']
  df['Chances Created'] = df['Key Pass']+df['Assist']
  df['Crosses'] = df['Cross']+df['Cross Fail']
  df['Dribbles'] = df['Dribble']+df['Dribble Fail']
  df['Tackles'] = df['Tackle']+df['Tackle Fail']
  df['Defensive Actions'] = df['Tackles']+df['Intercept']+df['Clearance']+df['Recovery']
  df['Saves'] = df['Save']+df['Penalty Save']
  df['Blocks'] = df['Block']+df['Block Cross']
  df['Aerial Duels'] = df['Aerial Won']+df['Aerial Lost']
  df['Errors'] = df['Error Goal - Error Led to Chance'] + df['Error Goal - Error Led to Goal']

  jatuh = ['No','Team ID','Player ID','Position (in match)','Gameweek','Opponent','Match','Home/Away','Venue',
           'Date','Result','Starter/Subs','Subs','Player Rating','Ball Possession','Pass Team','Kick In','Unnamed: 296',
           'Unnamed: 297','Unnamed: 298','Unnamed: 299','Unnamed: 300','Unnamed: 301','Unnamed: 302','Unnamed: 303',
           'Fantasy Assist','Fantasy Assist - Penalty','Fantasy Assist - Free kick','Fantasy Assist - Goal by rebound',
           'Fantasy Assist - Own goal by pass/cross','Fantasy Assist - Own goal by rebound','Unnamed: 310','Kompetisi',
           'Month','Nickname','DoB','Position','Nationality','Nat. Status','Age Group']

  df = df.drop(jatuh, axis=1)
  df = df.groupby(['Name','Team'], as_index=False).sum()
  df['Conversion Ratio'] = round(df['Goals']/df['Shots'],2)
  df['Shot on Target Ratio'] = round(df['Shot on']/df['Shots'],2)
  df['Successful Cross Ratio'] = round(df['Cross']/df['Crosses'],2)
  df['Pass per Shot'] = round(df['Total Pass']/df['Shots'],2)
  df['Pass Accuracy'] = round(df['Pass']/df['Total Pass'],2)
  df['Aerial Won Ratio'] = round(df['Aerial Won']/df['Aerial Duels'],2)
  df['Goal Kick Grounded Ratio'] = round(df['Goal Kick - No Sub-action']/df['Goal Kick'],2)
  df['Long Ball Ratio'] = round(df['Pass - Long Ball']/df['Pass'],2)

  temp = db[['Name', 'Position', 'Nationality']]
  dfx = pd.merge(df, temp, on='Name', how='left')
  datafull = dfx[dfx['MoP'] >= mins].reset_index(drop=True)
  datafull = datafull[mt_list]

  def p90_Calculator(variable_value):
    p90_value = round((((variable_value/df['MoP']))*90),2)
    return p90_value
    
  temp2 = df.drop(['Name', 'Team'], axis=1)
  p90 = temp2.apply(p90_Calculator)
  p90['Name'] = df['Name']
  p90['Team'] = df['Team']
  p90['MoP'] = df['MoP']
  p90['Conversion Ratio'] = df['Conversion Ratio']
  p90['Shot on Target Ratio'] = df['Shot on Target Ratio']
  p90['Successful Cross Ratio'] = df['Successful Cross Ratio']
  p90['Pass per Shot'] = df['Pass per Shot']
  p90['Pass Accuracy'] = df['Pass Accuracy']
  p90['Aerial Won Ratio'] = df['Aerial Won Ratio']
  p90['Goal Kick Grounded Ratio'] = df['Goal Kick Grounded Ratio']
  p90['Long Ball Ratio'] = df['Long Ball Ratio']

  p902 = pd.merge(p90, temp, on='Name', how='left')
  data90 = p902[p902['MoP'] >= mins].reset_index(drop=True)
  data90 = data90[mt_list]

  if (cat=='per 90'):
    return data90
  elif (cat=='Total'):
    return datafull

def get_cs(data):
  df = data.copy()
  uk = df[['Match', 'Result']]
  uk = uk.groupby(['Match', 'Result'], as_index=False).nunique()

  uk['Home'] = uk['Match'].str.split(' -').str[0]
  uk['Away'] = uk['Match'].str.split('- ').str[1]
  uk['FTHG'] = uk['Result'].str.split(' -').str[0]
  uk['FTAG'] = uk['Result'].str.split('- ').str[1]
  uk['FTHG'] = uk['FTHG'].astype(int)
  uk['FTAG'] = uk['FTAG'].astype(int)

  hcs = []
  acs = []
  for i in range(len(uk)):
    if (uk['FTAG'][i] == 0) and (uk['FTHG'][i] == 0):
      hcsit = 1
      acsit = 1
    elif (uk['FTAG'][i] == 0) and (uk['FTHG'][i] != 0):
      hcsit = 1
      acsit = 0
    elif (uk['FTAG'][i] != 0) and (uk['FTHG'][i] == 0):
      hcsit = 0
      acsit = 1
    else:
      hcsit = 0
      acsit = 0
    hcs.append(hcsit)
    acs.append(acsit)
  uk['CSH'] = hcs
  uk['CSA'] = acs
  ukh = uk[['Home','FTHG','FTAG','CSH']]
  ukh = ukh.groupby('Home', as_index=False).sum().rename(columns={'Home':'Team','FTHG':'Goal',
                                                                  'FTAG':'Conceded','CSH':'Clean Sheet'})

  uka = uk[['Away','FTAG','FTHG','CSA']]
  uka = uka.groupby('Away', as_index=False).sum().rename(columns={'Away':'Team','FTAG':'Goal',
                                                                  'FTHG':'Conceded','CSA':'Clean Sheet'})

  csdata = pd.concat([ukh, uka]).groupby('Team').sum().reset_index()

  return csdata

def milestone(data, data2):
  hist = data.copy()
  df = data2.copy()
  
  csdata = get_cs(df)
  df = df[['Team','Assist','Yellow Card','Red Card']]
  df = df.groupby(['Team'], as_index=False).sum()
  df2 = pd.merge(df, csdata, on='Team', how='left')

  mstone = pd.concat([hist, df2]).groupby('Team').sum().reset_index()

  return mstone

def get_sum90(data, data2, min):
  df = data.copy()
  db = data2.copy()
  df['Shots'] = df['Shot on']+df['Shot off']+df['Shot Blocked']
  df['Goals'] = df['Penalty Goal']+df['Goal']
  df['Total Pass'] = df['Pass']+df['Pass Fail']
  df['Saves'] = df['Save']+df['Penalty Save']
  df['Blocks'] = df['Block']+df['Block Cross']
  df['Aerial Duels'] = df['Aerial Won']+df['Aerial Lost']
  df['Crosses'] = df['Cross']+df['Cross Fail']

  df_data = df[['Name','Team','Kompetisi','MoP','Goals','Shots','Shot on',
                'Create Chance','Assist','Pass','Total Pass',
                'Pass - Progressive Pass', 'Pass - Through Pass',
                'Cross','Dribble','Tackle','Intercept','Recovery','Blocks',
                'Aerial Duels','Aerial Won','Saves','Shots on Target Faced',
                'Penalty Save','Keeper - Sweeper','Cross Claim','Goal Kick',
                'Goal Kick - Goal Kick Launch', 'Crosses']]
  df_sum = df_data.groupby(['Name','Team','Kompetisi'], as_index=False).sum()
  df_sum['Conversion Ratio'] = round(df_sum['Goals']/df_sum['Shots'],2)
  df_sum['Shot on Target Ratio'] = round(df_sum['Shot on']/df_sum['Shots'],2)
  df_sum['Successful Cross Ratio'] = round(df_sum['Cross']/df_sum['Crosses'],2)
  df_sum['Pass per Shot'] = round(df_sum['Total Pass']/df_sum['Shots'],2)
  df_sum['Pass Accuracy'] = round(df_sum['Pass']/df_sum['Total Pass'],2)
  df_sum['Aerial Won Ratio'] = round(df_sum['Aerial Won']/df_sum['Aerial Duels'],2)
  df_sum['Save Ratio'] = round(df_sum['Saves']/df_sum['Shots on Target Faced'],2)
  df_sum['Long Goal Kick Ratio'] = round(df_sum['Goal Kick - Goal Kick Launch']/df_sum['Goal Kick'],2)
  df_sum.replace([np.inf, -np.inf], 0, inplace=True)
  df_sum.fillna(0, inplace=True)

  temp = df_sum.drop(['Name','Team','Kompetisi'], axis=1)

  def p90_Calculator(variable_value):
    p90_value = round((((variable_value/temp['MoP']))*90),2)
    return p90_value
  p90 = temp.apply(p90_Calculator)

  p90['Name'] = df_sum['Name']
  p90['Team'] = df_sum['Team']
  p90['Kompetisi'] = df_sum['Kompetisi']
  p90['MoP'] = df_sum['MoP']
  p90['Conversion Ratio'] = df_sum['Conversion Ratio']
  p90['Shot on Target Ratio'] = df_sum['Shot on Target Ratio']
  p90['Successful Cross Ratio'] = df_sum['Successful Cross Ratio']
  p90['Pass per Shot'] = df_sum['Pass per Shot']
  p90['Pass Accuracy'] = df_sum['Pass Accuracy']
  p90['Aerial Won Ratio'] = df_sum['Aerial Won Ratio']
  p90['Save Ratio'] = df_sum['Save Ratio']
  p90['Long Goal Kick Ratio'] = df_sum['Long Goal Kick Ratio']

  pos = db[['Name','Position']]
  data_full = pd.merge(pos, p90, on='Name', how='left')
  data_full = data_full.loc[(data_full['MoP']>=min)].reset_index(drop=True)

  return data_full, df_sum

def get_pct(data, komp):
  data_full = data.copy()
  data_full = data_full[data_full['Kompetisi']==komp]
  df4 = data_full.groupby('Position', as_index=False)
  midfielder = df4.get_group('Midfielder')
  goalkeeper = df4.get_group('Goalkeeper')
  forward = df4.get_group('Forward')
  att_10 = df4.get_group('Attacking 10')
  center_back = df4.get_group('Center Back')
  fullback = df4.get_group('Fullback')
  winger = df4.get_group('Winger')

  #calculating the average stats per position
  #winger
  temp = winger.copy()
  winger = winger.drop(['Name','Position','Team','Kompetisi'], axis=1)
  winger.loc['mean'] = round((winger.mean()),2)
  winger['Name'] = temp['Name']
  winger['Position'] = temp['Position']
  winger['Team'] = temp['Team']
  winger['Kompetisi'] = temp['Kompetisi']
  values1 = {"Name": 'Average W', "Position": 'Winger', "Team": 'League Average', "Kompetisi": komp}
  winger = winger.fillna(value=values1)

  #fb
  temp = fullback.copy()
  fullback = fullback.drop(['Name','Position','Team','Kompetisi'], axis=1)
  fullback.loc['mean'] = round((fullback.mean()),2)
  fullback['Name'] = temp['Name']
  fullback['Position'] = temp['Position']
  fullback['Team'] = temp['Team']
  fullback['Kompetisi'] = temp['Kompetisi']
  values2 = {"Name": 'Average FB', "Position": 'Fullback', "Team": 'League Average', "Kompetisi": komp}
  fullback = fullback.fillna(value=values2)

  #cb
  temp = center_back.copy()
  center_back = center_back.drop(['Name','Position','Team','Kompetisi'], axis=1)
  center_back.loc['mean'] = round((center_back.mean()),2)
  center_back['Name'] = temp['Name']
  center_back['Position'] = temp['Position']
  center_back['Team'] = temp['Team']
  center_back['Kompetisi'] = temp['Kompetisi']
  values3 = {"Name": 'Average CB', "Position": 'Center Back', "Team": 'League Average', "Kompetisi": komp}
  center_back = center_back.fillna(value=values3)

  #cam
  temp = att_10.copy()
  att_10 = att_10.drop(['Name','Position','Team','Kompetisi'], axis=1)
  att_10.loc['mean'] = round((att_10.mean()),2)
  att_10['Name'] = temp['Name']
  att_10['Position'] = temp['Position']
  att_10['Team'] = temp['Team']
  att_10['Kompetisi'] = temp['Kompetisi']
  values4 = {"Name": 'Average CAM', "Position": 'Attacking 10', "Team": 'League Average', "Kompetisi": komp}
  att_10 = att_10.fillna(value=values4)

  #forward
  temp = forward.copy()
  forward = forward.drop(['Name','Position','Team','Kompetisi'], axis=1)
  forward.loc['mean'] = round((forward.mean()),2)
  forward['Name'] = temp['Name']
  forward['Position'] = temp['Position']
  forward['Team'] = temp['Team']
  forward['Kompetisi'] = temp['Kompetisi']
  values5 = {"Name": 'Average FW', "Position": 'Forward', "Team": 'League Average', "Kompetisi": komp}
  forward = forward.fillna(value=values5)

  #gk
  temp = goalkeeper.copy()
  goalkeeper = goalkeeper.drop(['Name','Position','Team','Kompetisi'], axis=1)
  goalkeeper.loc['mean'] = round((goalkeeper.mean()),2)
  goalkeeper['Name'] = temp['Name']
  goalkeeper['Position'] = temp['Position']
  goalkeeper['Team'] = temp['Team']
  goalkeeper['Kompetisi'] = temp['Kompetisi']
  values6 = {"Name": 'Average GK', "Position": 'Goalkeeper', "Team": 'League Average', "Kompetisi": komp}
  goalkeeper = goalkeeper.fillna(value=values6)

  #cm
  temp = midfielder.copy()
  midfielder = midfielder.drop(['Name','Position','Team','Kompetisi'], axis=1)
  midfielder.loc['mean'] = round((midfielder.mean()),2)
  midfielder['Name'] = temp['Name']
  midfielder['Position'] = temp['Position']
  midfielder['Team'] = temp['Team']
  midfielder['Kompetisi'] = temp['Kompetisi']
  values7 = {"Name": 'Average CM', "Position": 'Midfielder', "Team": 'League Average', "Kompetisi": komp}
  midfielder = midfielder.fillna(value=values7)

  #percentile rank
  rank_cm = round(((midfielder.rank(pct=True))*100),2)
  rank_gk = round(((goalkeeper.rank(pct=True))*100),2)
  rank_fw = round(((forward.rank(pct=True))*100),2)
  rank_cam = round(((att_10.rank(pct=True))*100),2)
  rank_cb = round(((center_back.rank(pct=True))*100),2)
  rank_fb = round(((fullback.rank(pct=True))*100),2)
  rank_w = round(((winger.rank(pct=True))*100),2)

  #adding Name and Position back
  rank_cm['Name'] = midfielder['Name']
  rank_gk['Name'] = goalkeeper['Name']
  rank_fw['Name'] = forward['Name']
  rank_cam['Name'] = att_10['Name']
  rank_cb['Name'] = center_back['Name']
  rank_fb['Name'] = fullback['Name']
  rank_w['Name'] = winger['Name']

  rank_cm['Position'] = midfielder['Position']
  rank_gk['Position'] = goalkeeper['Position']
  rank_fw['Position'] = forward['Position']
  rank_cam['Position'] = att_10['Position']
  rank_cb['Position'] = center_back['Position']
  rank_fb['Position'] = fullback['Position']
  rank_w['Position'] = winger['Position']

  rank_cm['Team'] = midfielder['Team']
  rank_gk['Team'] = goalkeeper['Team']
  rank_fw['Team'] = forward['Team']
  rank_cam['Team'] = att_10['Team']
  rank_cb['Team'] = center_back['Team']
  rank_fb['Team'] = fullback['Team']
  rank_w['Team'] = winger['Team']

  rank_cm['Kompetisi'] = midfielder['Kompetisi']
  rank_gk['Kompetisi'] = goalkeeper['Kompetisi']
  rank_fw['Kompetisi'] = forward['Kompetisi']
  rank_cam['Kompetisi'] = att_10['Kompetisi']
  rank_cb['Kompetisi'] = center_back['Kompetisi']
  rank_fb['Kompetisi'] = fullback['Kompetisi']
  rank_w['Kompetisi'] = winger['Kompetisi']

  rank_cm['MoP'] = midfielder['MoP']
  rank_gk['MoP'] = goalkeeper['MoP']
  rank_fw['MoP'] = forward['MoP']
  rank_cam['MoP'] = att_10['MoP']
  rank_cb['MoP'] = center_back['MoP']
  rank_fb['MoP'] = fullback['MoP']
  rank_w['MoP'] = winger['MoP']

  rank_liga = pd.concat([rank_cm, rank_gk, rank_fw, rank_cam, rank_cb, rank_fb, rank_w]).reset_index(drop=True)

  return rank_liga

def get_pssw(data, data2, team, gw):
  df = data.copy()
  #df['Gameweek'] = df['Gameweek'].astype(str)
  th = data2.copy()
  gw_list = gw
  df = df[df['Team']==team]
  df = df[df['Gameweek'].isin(gw_list)]

  df_rel = df[['Gameweek','Match','Team','Opponent','Shot off','Shot on','Shot Blocked',
               'Shot off - Inside Box','Shot off - Outside Box','Shot on - Inside Box',
               'Shot on - Outside Box','Ball Possession','Foul','Tackle Fail','Tackle',
               'Pass','Pass Fail','Dribble Fail','Loose Ball','Touch','Goal','Offside',
               'Intercept','Recovery','Aerial Won','Aerial Lost','Pass - Long Ball',
               'Pass - Short Pass','Fouled','Cross','Cross Fail','Cross Blocked','Penalty Goal']]
  
  dfx = df_rel.drop(['Opponent'], axis=1)
  dfx['Shots'] = dfx['Shot on'] + dfx['Shot off'] + dfx['Shot Blocked']
  dfx['Goals'] = dfx['Goal'] + dfx['Penalty Goal']
  df1 = dfx.groupby(['Gameweek','Match','Team'], as_index=False).sum()

  df3 = df_rel.drop(['Match','Opponent'], axis=1)
  df3 = df3.groupby(['Gameweek','Team'], as_index=False).sum()

  df4 = df_rel.drop(['Match','Team'], axis=1)
  df4 = df4.groupby(['Gameweek','Opponent'], as_index=False).sum()

  df_x = dfx.groupby(['Gameweek','Match'], as_index=False)['Ball Possession'].max()
  df_y = dfx.groupby(['Gameweek','Match'], as_index=False)['Ball Possession'].min()
  df_z = pd.merge(df_x, df_y, how='left', on=['Gameweek','Match'])

  df2 = pd.DataFrame()
  df2['Gameweek'] = df1['Gameweek']
  df2['Match'] = df1['Match']
  df2['Team'] = df1['Team']

  df2['PS_Att - Shot-Inside'] = df1['Shot off - Inside Box']+df1['Shot on - Inside Box']
  df2['PS_Att - Shot-Outside'] = df1['Shot off - Outside Box']+df1['Shot on - Outside Box']
  df2['PS_Att - Shot-Freq'] = df1['Shot off']+df1['Shot on']+df1['Shot Blocked']
  df2['PS_Att - Cross-Freq'] = round(((df1['Cross']+df1['Cross Fail']+df1['Cross Blocked'])/df1['Touch']),2)
  df2['PS_Att - Long Pass/Short Pass'] = df1['Pass - Long Ball']/df1['Pass - Short Pass']
  df2['PS_Deff - Agg.'] = df1['Foul']+df1['Tackle']+df1['Tackle Fail']

  df2['SW_Att - Offside'] = df1['Offside']/df1['Touch']
  df2['SW_Att - Poss-Eff.'] = round(((df1['Pass']+df1['Pass Fail'])/df2['PS_Att - Shot-Freq']),2)
  df2['SW_Att - Poss-RTT.'] = round(((df1['Loose Ball']+df1['Pass Fail']+df1['Dribble Fail'])/df1['Touch']),2)
  #df2['SW_Att - Fin.'] = abs(df1['Goals'].mean()-df1['Goals'])+abs(df1['Shots'].mean()-df1['Shots'])
  df2['SW_Att - Fin.'] = round((df1['Goals']/df2['PS_Att - Shot-Freq']),2)
  df2['SW_Deff - Regain'] = round(((df1['Intercept']+df1['Tackle']+df1['Recovery'])/df4['Touch']),2)
  df2['SW_Deff - Aerial'] = round((df1['Aerial Won']/(df1['Aerial Won']+df1['Aerial Lost'])),2)
  df2['SW_Deff - Tackle'] = round((df1['Tackle']/(df1['Tackle']+df1['Tackle Fail'])),2)

  df2.fillna(0, inplace=True)

  dfx = df_rel.drop(['Team','Opponent'], axis=1)
  df1 = dfx.groupby(['Gameweek','Match'], as_index=False).sum()

  df3 = df_rel.drop(['Match','Opponent'], axis=1)
  df3 = df3.groupby(['Gameweek','Team'], as_index=False).sum()

  df4 = df_rel.drop(['Match','Team'], axis=1)
  df4 = df4.groupby(['Gameweek','Opponent'], as_index=False).sum()

  df_x = dfx.groupby(['Gameweek','Match'], as_index=False)['Ball Possession'].max()
  df_y = dfx.groupby(['Gameweek','Match'], as_index=False)['Ball Possession'].min()
  df_z = pd.merge(df_x, df_y, how='left', on=['Gameweek','Match'])

  df23 = pd.DataFrame()

  df23['Gameweek'] = df1['Gameweek']
  df23['Match'] = df1['Match']
  df23['PS_Poss - Poss-Dom.'] = df_z['Ball Possession_x']-df_z['Ball Possession_y']

  dfk = pd.concat([df23, df2], ignore_index=True)
  dfk.replace([np.inf, -np.inf], 0, inplace=True)
  dfk.fillna(0, inplace=True)

  dfk = dfk.drop(['Gameweek','Match'], axis=1).groupby(['Team'], as_index=False).sum()
  tmp = dfk.drop('Team', axis=1)
  tmp = tmp/len(gw_list)
  tmp['Team'] = dfk['Team']
  tmp = tmp[tmp['Team']==team].reset_index(drop=True)

  ts = tmp.copy()
  def pstyle(row):  
    if row['PS_Att - Shot-Inside'] > th.iloc[0]['upper'] and row['PS_Att - Shot-Outside'] < th.iloc[1]['under']:
        return 'Work ball into box'
    elif row['PS_Att - Shot-Outside'] > th.iloc[1]['upper'] and row['PS_Att - Shot-Inside'] < th.iloc[0]['under']:
        return 'Shots on sight'
    else:
        return ''
  ts['PS1'] = ts.apply(lambda row: pstyle(row), axis=1)

  def pstyle(row):  
    if row['PS_Att - Shot-Freq'] > th.iloc[2]['upper']:
        return 'Shot more often'
    elif row['PS_Att - Shot-Freq'] < th.iloc[2]['under']:
        return 'Shot less often'
    else:
        return ''
  ts['PS2'] = ts.apply(lambda row: pstyle(row), axis=1)

  def pstyle(row):  
    if row['PS_Att - Cross-Freq'] > th.iloc[3]['upper']:
        return 'Cross more often'
    elif row['PS_Att - Cross-Freq'] < th.iloc[3]['under']:
        return 'Cross less often'
    else:
        return ''
  ts['PS3'] = ts.apply(lambda row: pstyle(row), axis=1)

  def pstyle(row):  
    if row['PS_Att - Long Pass/Short Pass'] > th.iloc[4]['upper']:
        return 'Direct passing'
    elif row['PS_Att - Long Pass/Short Pass'] < th.iloc[4]['under']:
        return 'Shorter passing'
    else:
        return 'Mixed passing'
  ts['PS4'] = ts.apply(lambda row: pstyle(row), axis=1)

  def pstyle(row):  
    if row['PS_Poss - Poss-Dom.'] > th.iloc[5]['upper']:
        return 'Possession football'
    elif row['PS_Poss - Poss-Dom.'] < th.iloc[5]['under']:
        return 'Counter-attack'
    else:
        return ''
  ts['PS5'] = ts.apply(lambda row: pstyle(row), axis=1)

  def pstyle(row):  
    if row['PS_Deff - Agg.'] > th.iloc[6]['upper']:
        return 'Get stuck in'
    elif row['PS_Deff - Agg.'] < th.iloc[6]['under']:
        return 'Stay on feet'
    else:
        return ''
  ts['PS5'] = ts.apply(lambda row: pstyle(row), axis=1)

  def sw(row):  
    if row['SW_Att - Offside'] < th.iloc[7]['under']:
        return 'Beating offside trap'
    else:
        return ''
  ts['S1'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Att - Offside'] > th.iloc[7]['upper']:
        return 'Beating offside trap'
    else:
        return ''
  ts['W1'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Att - Poss-Eff.'] > th.iloc[8]['upper']:
        return 'Effective in possession'
    else:
        return ''
  ts['S2'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Att - Poss-Eff.'] < th.iloc[8]['under']:
        return 'Ineffective in possession'
    else:
        return ''
  ts['W2'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Att - Poss-RTT.'] > th.iloc[9]['upper']:
        return 'Retaining possession'
    else:
        return ''
  ts['S3'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Att - Poss-RTT.'] < th.iloc[9]['under']:
        return 'Retaining possession'
    else:
        return ''
  ts['W3'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Att - Fin.'] > th.iloc[10]['upper']:
        return 'Finishing scoring chances'
    else:
        return ''
  ts['S4'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Att - Fin.'] < th.iloc[10]['under']:
        return 'Finishing scoring chances'
    else:
        return ''
  ts['W4'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Deff - Regain'] > th.iloc[11]['upper']:
        return 'Regaining possesion'
    else:
        return ''
  ts['S5'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Deff - Regain'] < th.iloc[11]['under']:
        return 'Regaining possesion'
    else:
        return ''
  ts['W5'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Deff - Aerial'] > th.iloc[12]['upper']:
        return 'Aerial duels'
    else:
        return ''
  ts['S6'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Deff - Aerial'] < th.iloc[12]['under']:
        return 'Aerial duels'
    else:
        return ''
  ts['W6'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Deff - Tackle'] > th.iloc[13]['upper']:
        return 'Tackling effectiveness'
    else:
        return ''
  ts['S7'] = ts.apply(lambda row: sw(row), axis=1)

  def sw(row):  
    if row['SW_Deff - Tackle'] < th.iloc[13]['under']:
        return 'Tackling effectiveness'
    else:
        return ''
  ts['W7'] = ts.apply(lambda row: sw(row), axis=1)

  desc = ts[['Team','PS1','PS2','PS3','PS4','PS5', 
             'S1','S2','S3','S4','S5','S6','S7',
             'W1','W2','W3','W4','W5','W6','W7']]

  return desc

def get_wdl(data, team, gws):
  df = data.copy()
  gw_list = gws
  df = df[df['Team']==team]
  df = df[df['Gameweek'].isin(gw_list)]

  uk = df[['Match', 'Result', 'Date', 'Gameweek']]
  uk = uk.groupby(['Match', 'Result', 'Date', 'Gameweek'], as_index=False).nunique()

  uk['Home'] = uk['Match'].str.split(' -').str[0]
  uk['Away'] = uk['Match'].str.split('- ').str[1]
  uk['FTHG'] = uk['Result'].str.split(' -').str[0]
  uk['FTAG'] = uk['Result'].str.split('- ').str[1]
  uk['FTHG'] = uk['FTHG'].astype(int)
  uk['FTAG'] = uk['FTAG'].astype(int)
  uk['GW'] = uk['Gameweek']
  uk['Rslt'] = 'S'
  uk['AR'] = 'W'

  for i in range(len(uk)):
    if (uk['FTHG'][i] > uk['FTAG'][i]):
      uk['Rslt'][i] = 'W'
      uk['AR'][i] = 'L'
    elif (uk['FTHG'][i] < uk['FTAG'][i]):
      uk['Rslt'][i] = 'L'
      uk['AR'][i] = 'W'
    else:
      uk['Rslt'][i] = 'D'
      uk['AR'][i] = 'D'

  for i in range(len(uk)):
    if (uk['Home'][i]!=team) and (uk['Rslt'][i]=='W'):
      uk['Rslt'][i] = 'L'
    elif (uk['Home'][i]!=team) and (uk['Rslt'][i]=='L'):
      uk['Rslt'][i] = 'W'


  uk = uk[['Date', 'GW', 'Rslt', 'Home', 'FTHG', 'FTAG', 'Away']]
  uk = uk.sort_values(by='GW').reset_index(drop=True)

  def bg_col(val):
    if val == 'W':
      color = '#7ed957'
    elif val == 'L':
      color = '#d9577e'
    elif val == 'D':
      color = '#a6a6a6'
    else:
      color = 'white'
    return 'background-color: %s' % color
  uk = uk.style.applymap(bg_col)

  return uk

def get_skuad(data, data2, team, gws):
  df = data.copy()
  db = data2.copy()

  gw_list = gws

  import datetime as dt
  from datetime import date

  today = date.today()
  db['Age'] = db['DoB'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

  df = df[df['Team']==team]
  df = df[df['Gameweek'].isin(gw_list)]
  df = df[['Player ID','MoP','Goal','Penalty Goal','Assist','Yellow Card','Red Card']]
  df['Goals'] = df['Goal']+df['Penalty Goal']

  df = df.groupby('Player ID', as_index=False).sum()
  fin = pd.merge(df, db, on='Player ID', how='left')
  fin = fin[['Nickname','Age','Nationality','Position','MoP',
             'Goals','Assist','Yellow Card','Red Card']]
  
  return fin

def get_formasi(data, data2):
  df = data.copy()
  cd = data2.copy()

  df_ft = df[df['Position (in match)'].notna()]
  df_ft = df_ft[['Gameweek', 'Name', 'Position (in match)', 'Team', 'Opponent', 'Match']]
  #df_ft = df_ft[(df_ft['Gameweek']==gw) & (df_ft['Team']==team)]
  df_ft = pd.merge(df_ft, cd, on='Position (in match)', how='left')
  df_ft['Formation'] = ''
  df_ft = df_ft.reset_index()

  x = 0
  y = 11
  for i in range(len(df_ft)):
    for j in range(x, y):
      tpr = df_ft[(df_ft['index'] >= x) & (df_ft['index'] < y)]
      a = list(tpr['Kode'].unique())
      fms = []
      old_stdout = sys.stdout
      new_stdout = io.StringIO()
      sys.stdout = new_stdout

      for m in a:
        temp = tpr[(tpr['Kode']==m)]['Kode'].count()
        fms.append(temp)

      for n in range(len(fms)):
        if (n != (len(fms)-1)):
          print(fms[n], end='-')
        else:
          print(fms[n], end='')

      df_ft['Formation'][j] = new_stdout.getvalue()
      sys.stdout = old_stdout
    x += 11
    y += 11

  fixdata = df_ft[['Gameweek', 'Team', 'Match', 'Formation']]
  fixdata = fixdata.groupby(['Gameweek', 'Match', 'Team'])['Formation'].unique().reset_index()
  for k in range(len(fixdata)):
    fixdata['Formation'][k] = list(fixdata['Formation'][k])[0]
  fixdata = fixdata[['Gameweek', 'Team', 'Match', 'Formation']]

  return fixdata

def get_radar(data1, data2, data3, pos, player):
  df1 = data1.copy()
  df2 = data2.copy()
  df3 = data3.copy()

  if (pos=='Forward'):
    temp1 = df1[posdict['fw']['metrics']]
    temp2 = df2[posdict['fw']['metrics']]
    temp3 = df3[posdict['fw']['metrics']]
  elif (pos=='Winger') or (pos=='Attacking 10'):
    temp1 = df1[posdict['cam/w']['metrics']]
    temp2 = df2[posdict['cam/w']['metrics']]
    temp3 = df3[posdict['cam/w']['metrics']]
  elif (pos=='Midfielder'):
    temp1 = df1[posdict['cm']['metrics']]
    temp2 = df2[posdict['cm']['metrics']]
    temp3 = df3[posdict['cm']['metrics']]
  elif (pos=='Fullback'):
    temp1 = df1[posdict['fb']['metrics']]
    temp2 = df2[posdict['fb']['metrics']]
    temp3 = df3[posdict['fb']['metrics']]
  elif (pos=='Center Back'):
    temp1 = df1[posdict['cb']['metrics']]
    temp2 = df2[posdict['cb']['metrics']]
    temp3 = df3[posdict['cb']['metrics']]
  elif (pos=='Goalkeeper'):
    temp1 = df1[posdict['gk']['metrics']]
    temp2 = df2[posdict['gk']['metrics']]
    temp3 = df3[posdict['gk']['metrics']]

  auxdata1 = temp1[temp1['Name']==player]
  auxdata2 = temp2[temp2['Name']==player]
  auxdata3 = temp3[temp3['Name']==player]
  auxt1 = auxdata1.transpose().reset_index()
  auxt2 = auxdata2.transpose().reset_index()
  auxt3 = auxdata3.transpose().reset_index()

  new_header = auxt1.iloc[0]
  auxt1 = auxt1[2:].reset_index(drop=True)
  auxt1.columns = new_header
  auxt1 = auxt1.reset_index(drop=True).rename(columns={'Name':'Metrics',
                                                       player:'Percentile'})
  auxt2 = auxt2[2:].reset_index(drop=True)
  auxt2.columns = new_header
  auxt2 = auxt2.reset_index(drop=True).rename(columns={'Name':'Metrics',
                                                       player:'per 90'})

  auxt3 = auxt3[2:].reset_index(drop=True)
  auxt3.columns = new_header
  auxt3 = auxt3.reset_index(drop=True).rename(columns={'Name':'Metrics',
                                                       player:'Total'})

  auxt4 = pd.merge(auxt3, auxt2, on='Metrics', how='left')
  auxt = pd.merge(auxt4, auxt1, on='Metrics', how='left')
  return auxt

def get_simi(data, data2, player, pos):
  df = data.copy()
  df = df[df['Position']==pos]
  db = data2.copy()
  
  if (pos=='Forward'):
    temp = df[posdict['fw']['metrics']].reset_index(drop=True)
  elif (pos=='Winger') or (pos=='Attacking 10'):
    temp = df[posdict['cam/w']['metrics']].reset_index(drop=True)
  elif (pos=='Midfielder'):
    temp = df[posdict['cm']['metrics']].reset_index(drop=True)
  elif (pos=='Fullback'):
    temp = df[posdict['fb']['metrics']].reset_index(drop=True)
  elif (pos=='Center Back'):
    temp = df[posdict['cb']['metrics']].reset_index(drop=True)
  elif (pos=='Goalkeeper'):
    temp = df[posdict['gk']['metrics']].reset_index(drop=True)

  dfx = temp.drop(['Name'], axis=1)

  def create_scaler_model():
    return StandardScaler()

  scaler = create_scaler_model()
  scaler.fit(dfx)
  scaled_features = scaler.transform(dfx)
  scaled_feat_df = pd.DataFrame(scaled_features)

  X = np.array(scaled_feat_df)
  calc_k_model = KMeans()
  visualizer = KElbowVisualizer(calc_k_model, k=(1,10), timings= True)
  visualizer.fit(X)

  kmeans = KMeans(n_clusters=visualizer.elbow_value_)
  kmeans.fit(X)

  scaled_feat_df['cluster'] = kmeans.predict(X)
  scaled_feat_df.insert(0, 'Name', temp['Name'])
  
  df_fin = scaled_feat_df.copy()
  clus = df_fin[df_fin['Name'] == player]['cluster']

  df_fin = df_fin[df_fin['cluster'] == int(clus)]
  df_fin.reset_index(inplace=True)
  df_fin.drop(['index'],axis=1,inplace=True)
  
  player_list = df_fin[df_fin['Name'] == player].values.tolist()
  others_list = df_fin[df_fin['Name'] != player].values.tolist()

  ind = df_fin[df_fin['Name'] == player_list[0][0]].index[0]
  df_fin['Similarity Score'] = ''

  df_fin['Similarity Score'][ind] = 0

  for elem in others_list:
    sim_score = 0
  #Calculate similarity score using Euclidian distance
    for i in range(1,len(player_list[0])-1):
      sim_score += pow(player_list[0][i] - elem[i],2)
    sim_score = math.sqrt(sim_score)
    ind = df_fin[df_fin['Name'] == elem[0]].index
    df_fin['Similarity Score'][ind] = sim_score
  df_fin = df_fin.sort_values('Similarity Score').reset_index(drop=True)
  df_fin = df_fin.iloc[1:, :].reset_index(drop=True)
  df_fin = df_fin[['Name', 'Similarity Score']]

  import datetime as dt
  from datetime import date

  today = date.today()
  db['Age'] = db['DoB'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

  df_fix = pd.merge(df_fin, db, on='Name', how='left')
  df_fix = pd.merge(df_fix, df, on='Name', how='left')
  df_fix = df_fix[['Name', 'Nickname', 'Team', 'MoP', 'Age', 'Nationality', 'Similarity Score']]
  df_fix['MoP'] = df_fix['MoP'].astype(int)

  return df_fix

def get_playerlist(data, komp, pos, mins, nat, age, arr_met):
  df = data.copy()

  ag_list = age
  nt_list = nat
  kp_list = komp

  df = df[df['Kompetisi'].isin(kp_list)]
  df = df[df['Nat. Status'].isin(nt_list)]
  df = df[df['Age Group'].isin(ag_list)]
  df = df[df['MoP']>=mins]
  df = df[df['Position']==pos].reset_index(drop=True)

  metrik = arr_met
  data = df[metrik]
  data['mean'] = round(data.mean(axis=1),2)
  data.insert(0, column='Name', value=df['Name'])
  data.insert(1, column='Team', value=df['Team'])
  data.insert(2, column='MoP', value=df['MoP'])

  data = data.sort_values(by=['mean'], ascending=False).reset_index(drop=True)
  
  return data
