#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:10:20 2021

@author: ricardobortothopker
"""

import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from random import shuffle
import random
import string
import pathlib

from google.oauth2 import service_account
from googleapiclient.discovery import build
from gsheetsdb import connect

# Create a connection object.
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["gcp_service_account"],
#     scopes=[
#         "https://www.googleapis.com/auth/spreadsheets",
#     ],
# )

# result_counter = sheet.values().get(spreadsheetId=Sheet1,range="current_test_number!A1:A3").execute()
# values_counter = result_counter.get('values',[])
# df1 = pd.DataFrame(values_counter)
# st.write(df1)
# last_row = df1.iloc[2][0]
# result = sheet.values().get(spreadsheetId=Sheet0,range=f"results_streamlit!A1:E{last_row}").execute()
# values = result.get('values',[])
# df0 = pd.DataFrame(values)
# st.write(df0)


# df1.iloc[1][0] = int(df1.iloc[1][0])+1
# df_as_list = df1.values.tolist()
# dictt = {'values':df_as_list}
# request = sheet.values().update(spreadsheetId=Sheet1,
#                                range="current_test_number!A1",
#                                valueInputOption='USER_ENTERED', 
#                                body=dictt).execute()


if 'session_id' not in st.session_state:
    letters = string.ascii_lowercase
    st.session_state['session_id'] = ''.join(random.choice(letters) for i in range(30))
@st.cache()
def create_experiments():
    from itertools import combinations
    data = list(range(0,30,2))
    cc = np.array(list(combinations(data,13)))
    np.random.seed(22689)
    random = np.random.randint(2,size=np.shape(cc))
    notRandom = random==0
    problems = cc+random
    problems=np.where(problems!=29,problems,problems+1)
    problems2 = cc+notRandom
    problems2=np.where(problems2!=29,problems2,problems2+1)
    problemsTot = np.concatenate([problems.T,problems2.T],axis=1).T
    return problemsTot
problemsTot=create_experiments()
path = pathlib.Path(__file__).parents[0]
# path = r"/Users/ricardobortothopker/OneDrive - Massachusetts Institute of Technology/Classes/Thesis/excels/Points for TSP/"
file = r"TSP Problems3.pkl"
url = path / file
with open(url,'rb') as f:  # Python 3: open(..., 'rb')\n",
    xyArrDict = pickle.load(f)
if 'current_test' not in st.session_state:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    conn = connect(credentials=credentials)

    service = build('sheets','v4',credentials=credentials)
    sheet = service.spreadsheets()

    Sheet0 = st.secrets["Sheet0"]
    Sheet1 = st.secrets["Sheet1"]
    result_counter = sheet.values().get(spreadsheetId=Sheet1,range="current_test_number!A1:A3").execute()
    values_counter = result_counter.get('values',[])
    df1 = pd.DataFrame(values_counter)
    cur_test_num = int(df1.iloc[1][0])

    if cur_test_num>=len(problemsTot):
        cur_test_num=-1
    st.session_state['current_test'] = cur_test_num
    df1.iloc[1][0] = int(df1.iloc[1][0])+1
    df_as_list = df1.values.tolist()
    dictt = {'values':df_as_list}
    request = sheet.values().update(spreadsheetId=Sheet1,
                                   range="current_test_number!A1",
                                   valueInputOption='USER_ENTERED', 
                                   body=dictt).execute()

    cur_test_num = st.session_state['current_test']
    cur_test = problemsTot[cur_test_num,:].copy()
    shuffle(cur_test)
    st.session_state['current_test'] = cur_test

if 'count' not in st.session_state:
    st.session_state['count'] = 0
elif st.session_state['count'] == len(st.session_state['current_test']):
    st.write('You have reached the end of the experiment')
    st.write('Thank you for participating!')
    st.stop()
st.write(f"excercise {st.session_state['count']+1} of {len(st.session_state['current_test'])}")
cur_test = st.session_state['current_test']
test_id = cur_test[st.session_state['count']]
test_id =f"id {test_id}"
cur_id = xyArrDict[test_id]
x = cur_id[:,0].tolist()
y = cur_id[:,1].tolist()
xy = list(zip(x,y))
# path = {'x': [[4, 0], [1, 2], [3, 4], [1, 0], [2, 3]], 'y': [[16, 0], [1, 4], [9, 16], [1, 0], [4, 9]]}
def valid_path(path):
    if len(x)!=len(path['x']):
        return False
    xy_path =[]
    path_dict = {}
    for i in range(len(path['x'])):
        points = list(zip(path['x'][i],path['y'][i]))
        point0 = np.where(np.all(points[0]==np.array(xy),axis=1))[0][0]
        point1 = np.where(np.all(points[1]==np.array(xy),axis=1))[0][0]
        xy_path.append([point0,point1])
        path_dict[i] = []
    for i in xy_path:
        path_dict[i[0]].append(i[1])
        path_dict[i[1]].append(i[0])
    visited = [0]
    tf = True
    cur_node = 0
    while tf:
        # print(f'Current node: {cur_node}')
        # print(f'Visited: {visited}')
        # print(f'Possible nodes: {path_dict[cur_node]}')
        # print(f'-------')
        if path_dict[cur_node][0] not in visited:
            cur_node = path_dict[cur_node][0]
            visited.append(cur_node)
        elif path_dict[cur_node][1] not in visited:
            cur_node = path_dict[cur_node][1]
            visited.append(cur_node)
        else:
            if len(visited) == len(x) and 0 in path_dict[cur_node]:
                tf = False
                return True
            else:
                tf = False
                return False
        if len(visited)> len(x):
            tf = False
            return False
# path = {'x': [[4, 0], [1, 2], [3, 4], [1, 0], [2, 3],[2, 3],[3,2]],  
#         'y': [[16, 0], [1, 4], [9, 16], [1, 0], [4, 9], [4, 9],[9,4]]}
def remove_double_paths(path):
    if len(path['x'])>1:
        xy_path=[]
        path_dict = {}
        for i in range(len(path['x'])):
            points = list(zip(path['x'][i],path['y'][i]))
            point0 = np.where(np.all(points[0]==np.array(xy),axis=1))[0][0]
            point1 = np.where(np.all(points[1]==np.array(xy),axis=1))[0][0]
            xy_path.append([point0,point1])
        for i in range(len(x)):
            path_dict[i] = []
        x_path = []
        y_path = []
        for i in xy_path:
            if i[1] not in path_dict[i[0]]:
                path_dict[i[0]].append(i[1])
            if i[0] not in path_dict[i[1]]:
                path_dict[i[1]].append(i[0])
        visited = []
        xy_path =[]
        for key,item in path_dict.items():
            visited.append(key)
            if item !=[]:
                for i in item:
                    if i not in visited:
                        xy_path.append([key,i])
        x_path = []
        y_path = []
        for i in xy_path:
            x_path.append([x[i[0]],x[i[1]]])
            y_path.append([y[i[0]],y[i[1]]])
        outpath ={}
        outpath['x'] = x_path
        outpath['y'] = y_path
        return outpath
    return path
def path_to_point(path):
    xy_path =[]
    for i in range(len(path['x'])):
        points = list(zip(path['x'][i],path['y'][i]))
        point0 = np.where(np.all(points[0]==np.array(xy),axis=1))[0][0]
        point1 = np.where(np.all(points[1]==np.array(xy),axis=1))[0][0]
        xy_path.append([point0,point1])
    return xy_path
# remove_double_paths(path)


if 'last_point' not in st.session_state:
    st.session_state['last_point'] = []
    st.session_state['path'] = {'x':[],'y':[]}

else:
    if st.session_state['path']['x']!=[]:
        # print(f"before {st.session_state['path']}")
        new_path = remove_double_paths(st.session_state['path'])
        # print(f"new_path {new_path}")
        if new_path!=st.session_state['path']:
            st.session_state['path'] = new_path
            st.experimental_rerun()
        # print(f"after {st.session_state['path']}")
        
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = datetime.now()
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,mode='markers',text=list(range(len(x))), hovertemplate='point number:%{text}'))
fig.update_yaxes(visible=False, showticklabels=False)
fig.update_xaxes(visible=False, showticklabels=False)

for i in range(len(st.session_state['path']['x'])):
    x1 = np.linspace(st.session_state['path']['x'][i][0],st.session_state['path']['x'][i][1],10)
    y1 = np.linspace(st.session_state['path']['y'][i][0],st.session_state['path']['y'][i][1],10)
    fig.add_trace(go.Scatter(x=x1, y=y1, mode="lines",marker_color='rgba(255, 0, 0, 1)',text=list(range(len(x1))), hovertemplate='path (click to erase)'))
fig.update_layout(showlegend=False)
fig.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True)
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
selected_points = plotly_events(fig, click_event=True, hover_event=False)
# st.plotly_chart(fig,use_container_width=True)
if selected_points ==[]:
    selected_point = []
else:
    selected_point = selected_points[0]
if st.session_state['last_point'] != [] and selected_point!=[]:
    selected_point = selected_points[0]
    # print(st.session_state['last_point']['curveNumber'])
    curve0 = st.session_state['last_point']['curveNumber']
    curve1 = selected_point['curveNumber']
    if curve0 == 0 and curve1 == 0 :
        st.session_state['path']['x'].append([st.session_state['last_point']['x'],selected_point['x']])
        st.session_state['path']['y'].append([st.session_state['last_point']['y'],selected_point['y']])
        # print(st.session_state['path'])
        # print(valid_path(st.session_state['path']))
        st.experimental_rerun()
elif selected_point!=[] and selected_point['curveNumber']!=0:
    curve1 = selected_point['curveNumber']-1
    del st.session_state['path']['x'][curve1]
    del st.session_state['path']['y'][curve1]
    st.experimental_rerun()
if st.button(label='Clear all lines'):
    st.session_state['path'] = {'x':[],'y':[]}
    st.session_state['last_point'] = selected_point
    st.experimental_rerun()
st.session_state['last_point'] = selected_point
if valid_path(st.session_state['path']):
    if st.button(label='Next'):
        
        st.session_state['finished'] = datetime.now()
        st.session_state['count'] += 1
        st.session_state['last_point'] = []
        selected_points =[]
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
            ],
        )
        # url_results = path / 'results_streamlit.csv'
        # streamlit_csv = pd.read_csv(url_results)
        conn = connect(credentials=credentials)

        service = build('sheets','v4',credentials=credentials)
        sheet = service.spreadsheets()

        Sheet0 = st.secrets["Sheet0"]
        Sheet1 = st.secrets["Sheet1"]
        result_counter = sheet.values().get(spreadsheetId=Sheet1,range="current_test_number!A1:A3").execute()
        values_counter = result_counter.get('values',[])
        df1 = pd.DataFrame(values_counter)
        # st.write(df1)
        last_row = df1.iloc[2][0]
        result = sheet.values().get(spreadsheetId=Sheet0,range=f"results_streamlit!A1:E{last_row}").execute()
        values = result.get('values',[])
        streamlit_csv = pd.DataFrame(values[1:],columns=values[0])
        # st.write(streamlit_csv)
        duration = st.session_state['finished'] - st.session_state['start_time']
        
        df_temp = pd.DataFrame([{'test_id': test_id,
                                 'path':str(path_to_point(st.session_state['path'])),
                                 'duration':duration.total_seconds(),
                                 'Session_id': st.session_state['session_id'],
                                 'Finish_time':st.session_state['finished'].strftime("%Y/%m/%d, %H:%M:%S")}])
        
        
        
        st.session_state['path'] = {'x':[],'y':[]}
        # st.write(df_temp)
        streamlit_csv = pd.concat([streamlit_csv,df_temp])
        # st.write(streamlit_csv)
        streamlit_csv_as_list = streamlit_csv.values.tolist()
        streamlit_csv_as_list.insert(0,streamlit_csv.columns.tolist())
        
        dict_write = {'values':streamlit_csv_as_list}
        # print(streamlit_csv_as_list)
        st.markdown(streamlit_csv_as_list)
        request = sheet.values().update(spreadsheetId=Sheet0,
                                        range="results_streamlit!A1",
                                        valueInputOption='USER_ENTERED', 
                                        body=dict_write).execute()
        df1.iloc[2][0] = int(df1.iloc[2][0])+1
        df_as_list = df1.values.tolist()
        dictt = {'values':df_as_list}
        request = sheet.values().update(spreadsheetId=Sheet1,
                                       range="current_test_number!A1",
                                       valueInputOption='USER_ENTERED', 
                                       body=dictt).execute()
        
        
        # streamlit_csv.to_csv(url_results,index=False)
        # st.session_state['path'] = {'x':[],'y':[]}
        st.session_state['start_time'] = datetime.now()
        st.experimental_rerun()

