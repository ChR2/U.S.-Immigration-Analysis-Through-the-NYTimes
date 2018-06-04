import pickle
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import random
import datetime

# Get the model
with open('../master_df.pkl', 'rb') as f:
    df = pickle.load(f)

# Pickle removes datettimes so we need them back
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

# Manual topics labels
topics = [
    'Education Standards',
    'Trump',
    'Romney',
    'Cuban Relations',
    'Judicial',
    'Prosecution & Criminality',
    'Polling',
    'Refugees',
    'Family',
    'Democrats',
    'US/Soviet Disarmament',
    'Mexican Relations',
    'Reagan\'s Amnesty',
    'Partisanship',
    'Terrorism',
    'Congress',
    'Census',
    'Asylum',
    'Foriegn Trade',
    'Police Brutality',
    'Peace Talks',
    'California',
    'Southern Border',
    'Judicial',
    'Catholic Church',
    'Job Growth',
    'Visas',
    'Coast Guard',
    'DACA',
    'Health Care',
]

# Dates dataframe
year_df = pd.DataFrame(df.groupby('year')['article'].count())
year_x = year_df.index
year_y = year_df.article

# Topics over time
# Smaller df for faster processing?
sub_df = df[[
    # 'article', 
    'date',
    'url',
    # 'year',
    # 'month',
    # 'headline',
    'snippet',
    'topic'
    ]]
# t0 = sub_df[sub_df.topic==0]
# t0_count = t0.groupby(['date']).count()
# t0_time = t0.groupby('date')['snippet'].apply(list)
# t0_time = t0_time.reset_index()
# t0_time['Count'] = pd.Series([len(i) for i in t0_time['snippet']])

# Play with data extraction, reduce down later
def fmt_topic_over_time(topic_index):
    t = sub_df[sub_df.topic==topic_index]
    t_count = t.groupby(t.date.dt.to_period('M')).count()
    t_time = t.groupby('date')['snippet'].apply(list)
    t_time = t_time.reset_index()
    t_time['Count'] = pd.Series([len(i) for i in t_time['snippet']])
    return {'count': t_count, 'time': t_time, 'snippet': t_time['snippet']}

# rewrite to list
formatted = [fmt_topic_over_time(t) for t in range(len(topics))]

# Sentiment analysis
gr = pd.DataFrame(df.groupby('year')['sentiment'].mean()).reset_index()
x_sent, y_sent = gr.year, gr.sentiment

app = dash.Dash()

app.scripts.config.serve_locally = True

colors = {
    'background': 'white',
    'text': '#666'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Immigration model',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(
            children='Topics as related to immigration by the New York Times since January 1981.', 
            style={
                'textAlign': 'center',
                'margin-bottom': '1.6em',
                'color': colors['text']
            }),
    html.Div([
        dcc.Tabs(
            tabs=[
                {'label': 'Topics Distribution', 'value': 0},
                {'label': 'Count Articles by Year', 'value': 1},
                {'label': 'Sentiment Over Time', 'value': 2},
                {'label': 'Topic Over Time', 'value': 3}
            ],
            value=1,
            id='tabs'
        ),
        html.Div(id='tab-output')
    ], style={
        'width': '80%',
        'fontFamily': 'Sans-Serif',
        'margin-left': 'auto',
        'margin-right': 'auto'
    })])

@app.callback(Output('tab-output', 'children'), [Input('tabs', 'value')])
def display_content(value):
    data = [
    [
        {
            'x': topics,
            'y': df.groupby('topic').count()[1],
            'name': 'Topics Distribution',
            'marker': {
                'color': 'rgb(26, 118, 255)'
            },
            'type': 'bar'
        }
    ],
    [
        {
            'x': year_x,
            'y': year_y,
            'name': 'Articles by Year',
            'marker': {
                'color': 'rgb(255, 118, 26)'
            },
            'type': 'line'
        }
    ],
    [
        {
            'x': x_sent,
            'y': y_sent,
            'name': 'Sentiment over years',
            'marker': {
                'color': 'rgb(118, 255, 26)'
            },
            'type': 'line'
        }
    ],
    # [
        # {
        #     'x': t0_time.date,
        #     'y': t0_time.Count,
        #     'name': topics[0],
        #     'marker': {
        #         'color': 'rgb(118, 26, 255)'
        #     },
        #     'type': 'line'
        # },
    [
        {
            'x': formatted[t]['time'].date,
            'y': formatted[t]['time'].Count,
            'name': topics[t],
            'marker': {
                'color': 'rgb({r}, {g}, {b})'.format(r=random.randint(0,255),g=random.randint(0,255),b=random.randint(0,255))
            },
            'type': 'line',
            'text': [formatted[t]['time'].snippet[s][0][2:64] for s in range(len(formatted[t]['time'].date))]
        } for t in range(len(topics))]
    ]
    # ]]


    return html.Div([
        dcc.Graph(
            id='graph',
            figure={
                'data': data[value],
                'layout': {
                    # 'legend': {'x': 0, 'y': 1}
                }
            }
        )
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
