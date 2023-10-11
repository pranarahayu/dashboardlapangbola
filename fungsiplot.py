import os
import pandas as pd
import glob
from datetime import date
import numpy as np
from sklearn import preprocessing

from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.font_manager as fm
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch

from PIL import Image
from tempfile import NamedTemporaryFile
import urllib
import os

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

def plot_skuad(data, data2, team):
  df = data.copy()
  db = data2.copy()

  import datetime as dt
  from datetime import date

  today = date.today()
  db['Age'] = db['DoB'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

  df = df[df['Team']==team]
  fil = df[['Player ID','MoP']]
  fil = fil.groupby('Player ID', as_index=False).sum()
  fil = fil[fil['MoP']>0]
  fil['%Min'] = round(((fil['MoP']/(df['Gameweek'].max()*90))*100),2)

  fin = pd.merge(fil, db, on='Player ID', how='left')
  fin = fin[['Nickname','Age','%Min']]

  fig, ax = plt.subplots(figsize=(10,10), dpi=500)
  fig.patch.set_facecolor('#ffffff')
  ax.set_facecolor('#ffffff')
  ax.grid(ls='--', color='#000000', alpha=0.2)

  ax.scatter(fin['Age'], fin['%Min'], color='#7ed957', s=150, edgecolors='#ffffff', zorder=10)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_color('#000000')
  ax.spines['left'].set_color('#000000')

  for t in ax.xaxis.get_ticklines(): t.set_color('#000000')
  ax.tick_params(axis='x', colors='#000000')

  for t in ax.yaxis.get_ticklines(): t.set_color('#000000')
  ax.tick_params(axis='y', colors='#000000')

  for tick in ax.get_xticklabels():
    tick.set_fontproperties(bold)

  for tick in ax.get_yticklabels():
    tick.set_fontproperties(bold)

  ax.axvline(x=23.5, color='#f2ff00', linestyle='--', alpha=0.7, zorder=2)
  ax.fill_between([15,23.5], 0, 100, color='#f2ff00', alpha=0.4)
  ax.text(19.25, 50,'U-23 PLAYER', ha='center', va='center', color='#000000',
          fontproperties=bold, size=16, rotation=90, alpha=0.7)

  ax.set_xticks([15, 20, 25, 30, 35, 40])
  ax.set_yticks([0, 20, 40, 60, 80, 100])
  ax.set_ylim(bottom=0)
  ax.set_ylim(top=100)
  ax.set_xlim(left=15)
  ax.set_xlim(right=40)

  ax.set_xlabel('Age', color='#000000', fontproperties=bold, size=11)
  ax.set_ylabel('% Mins Played', color='#000000', fontproperties=bold, size=11)

  for i in range(len(fin)):
    ax.annotate(fin['Nickname'][i], xy=(fin['Age'][i]+0.5, fin['%Min'][i]), c='#000000', va='center',
                zorder=11, ha='left', size=6, weight='bold', path_effects=path_eff)

  plt.savefig('skuadsca.jpg', dpi=500, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
  
  return fig
  

def plot_skuadbar(data, data2, team):
  df = data.copy()
  db = data2.copy()

  import datetime as dt
  from datetime import date

  today = date.today()
  db['Age'] = db['DoB'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

  df = df[df['Team']==team]
  fil = df[['Player ID','MoP']]
  fil = fil.groupby('Player ID', as_index=False).sum()
  fil = fil[fil['MoP']>0]

  fin = pd.merge(fil, db, on='Player ID', how='left')
  fin = fin[['Nickname','Age','MoP']]
  fin['Range'] = '0'

  for i in range(len(fin)):
    if (fin['Age'][i]<24):
      fin['Range'][i] = '15-24'
    elif (fin['Age'][i]>=24) and (fin['Age'][i]<31):
      fin['Range'][i] = '24-30'
    elif (fin['Age'][i]>=31) and (fin['Age'][i]<36):
      fin['Range'][i] = '31-35'
    else:
      fin['Range'][i] = '35+'
    
  fin = fin[['Range','MoP']]
  fin = fin.groupby('Range', as_index=False).sum()
  fin['%Min'] = round(((fin['MoP']/fin['MoP'].sum())*100),0)

  fig, ax = plt.subplots(figsize=(10,10), dpi=500)
  fig.patch.set_facecolor('#ffffff')
  ax.set_facecolor('#ffffff')

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_color('#000000')
  ax.spines['left'].set_color('#000000')

  for t in ax.xaxis.get_ticklines(): t.set_color('#000000')
  ax.tick_params(axis='x', colors='#000000')

  for t in ax.yaxis.get_ticklines(): t.set_color('#000000')
  ax.tick_params(axis='y', colors='#000000')

  for tick in ax.get_xticklabels():
    tick.set_fontproperties(bold)

  for tick in ax.get_yticklabels():
    tick.set_fontproperties(bold)
  
  bar = ax.barh(fin['Range'], fin['%Min'], color='#7ed957')

  for bar in bar.patches:
    ax.annotate(str(format(bar.get_width(), '.0f'))+'%',
              (bar.get_width()/2,
               bar.get_y() + bar.get_height()/2), ha='center', va='center',
              size=14, xytext=(0, 8), fontproperties=bold,
              textcoords='offset points',color='#000000')

  ax.set_xticks([0, 20, 40, 60, 80, 100])
  ax.set_xlim(right=100)
  ax.set_ylabel('Age Range', color='#000000', fontproperties=bold, size=11)
  ax.set_xlabel('% Mins Played', color='#000000', fontproperties=bold, size=11)


  plt.savefig('skuadbar.jpg', dpi=500, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
  
  return fig

def plot_form(data, data2, team, gw):
  df = data.copy()
  cf = data2.copy()

  dft = df[df['Team']==team].reset_index(drop=True)
  temp = dft[dft['Gameweek']==gw].reset_index(drop=True)
  temp['Formation'] = temp['Formation'].astype(str)
  cf = cf[cf['Formation']==temp['Formation'][0]].reset_index(drop=True)

  fig, ax = plt.subplots(figsize=(20, 20), dpi=500)
  pitch = VerticalPitch(half=True, pitch_type='wyscout', corner_arcs=True,
                        pitch_color='#ffffff', line_color='#000000',
                        stripe_color='#fcf8f7', goal_type='box', pad_bottom=5,
                        pad_right=0.5, pad_left=0.5, stripe=True, linewidth=3.5)
  pitch.draw(ax=ax)
  ax.add_patch(FancyBboxPatch((0, 45), 200, 4.5, fc='#ffffff', ec='#ffffff', lw=2))

  #cf['X'] = 100 - cf['X']
  ax.scatter(cf['Y'], cf['X'], color='#7ed957', s=2500, edgecolors='#f2ff00', lw=4)
  for i in range(len(cf)):
    pitch.annotate(cf['Position'][i], xy=(cf['X'][i], cf['Y'][i]), c='#000000', va='center', zorder=11,
                   ha='center', size=16, weight='bold', ax=ax, path_effects=path_eff)
  
  mst = dft.drop(['Gameweek','Team'], axis=1).groupby('Formation', as_index=False).count()
  mst = mst.sort_values(by='Match', ascending=False).reset_index(drop=True)
  fmt = mst['Formation'][0]
  jml = mst['Match'].max()

  ax.annotate(text='Most Used Starting Formation: '+fmt+' ('+str(jml)+')',
              size=24, xy=(0, 49), xytext=(0,-18),
              textcoords='offset points', color='black', ha='left',
              zorder=9, va='center', fontproperties=bold)
  ax.annotate(text='Match: '+temp['Match'][0], size=24, xy=(0, 102), xytext=(0,-18),
              textcoords='offset points', color='black', ha='left',
              zorder=9, va='center', fontproperties=bold)
  
  plt.savefig('stafor.jpg', dpi=500, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')

  return fig
