import datetime
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output
import time
import random
from multiprocessing import Process
import psutil
import redis
import pickle
from pynvml import *
import logging


_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
_app.layout = html.Div(
    html.Div([
        # NavBar
        dbc.Navbar(
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand("Researchlib Dashboard", className="ml-2")),
                    ],
                    align="left",
                    no_gutters=True,
                ),
                href="#",
            ),
            dark=False,
        ),
        
        # Resource monitor
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(id='live-update-pie',
                        config={
                            'displayModeBar': False
                        }),
            ])
        ),
        
        # Experiments choose (TODO)
        html.Div([
            dcc.Tabs(id="tabs", value='tab-0', 
                children=[
                    dcc.Tab(label='Loading', value='tab-0'),
                ]
            ),
            
            dbc.Card(
                dbc.CardBody([
                    html.Div(id='live-update-text'),
                    dbc.Progress(id="progress", value=0, striped=True, animated=True),
                ])
            ),

            dbc.Card(
                dbc.CardBody([
                    dcc.Graph(id='live-update-loss',
                            config={
                                'displayModeBar': False
                            }),
                    dcc.Graph(id='live-update-acc',
                            config={
                                'displayModeBar': False
                            }),
                ])
            ),
        ]),
        
        
        dcc.Interval(
            id='text-update',
            interval=1000,
            n_intervals=0
        ),
        dcc.Interval(
            id='pie-update',
            interval=1000,
            n_intervals=0
        ),
        dcc.Interval(
            id='loss-update',
            interval=1000,
            n_intervals=0
        ),
        dcc.Interval(
            id='acc-update',
            interval=1000,
            n_intervals=0
        )
    ])
)

@_app.callback([Output('live-update-text', 'children'), Output('progress', 'value'), Output('tabs', 'children')], [Input('text-update', 'n_intervals')])
def _update_desc(n):
    r = redis.Redis()
    desc = r.get('desc').decode('utf-8')
    stage = r.get('stage').decode('utf-8')
    value = int(float(r.get('progress'))*100)
    stage_color = {'train':'success', 'validate':'danger'}
    style = {'padding': '5px', 'fontSize': '16px'}
    
    tabs = []
    experiments = pickle.loads(r.get('experiment'))
    for i, name in enumerate(experiments):
        tabs.append(dcc.Tab(label=str(name), value='tab-'+str(i)))
    if len(tabs) == 0:
        tabs.append(dcc.Tab(label='Loading', value='tab-0'))
    
    
    if stage == 'stop':
        return [html.Span(f'{desc}', style=style)], value, tabs
    else:
        return [
            dbc.Spinner(color=stage_color[stage], type="grow"),
            html.Span(f'{desc}', style=style)
        ], value, tabs



def _add_trace(fig, x, y, name, row_index, col_index):
    try:
        fig.append_trace({
            'x': x,
            'y': y,
            'name': name,
            'mode': 'lines+markers',
            'type': 'scatter',
        }, row_index, col_index)
    except:
        pass
    return fig

def _add_pie(fig, values, name, row_index, col_index):
    fig.append_trace({
        'values': values,
        'labels': ['Occupy', 'Non-Occupy'],
        'sort': False,
        'name': name,
        'type': 'pie',
        'hole': 0.3,
    }, row_index, col_index)
    return fig

def _get_gpu_monitor():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    s = nvmlDeviceGetUtilizationRates(handle)
    return int(100 * (info.used / info.total)), s.gpu

# Multiple components can update everytime interval gets fired.
@_app.callback(Output('live-update-pie', 'figure'), [Input('pie-update', 'n_intervals')])
def _update_pie_live(n):
    fig = plotly.subplots.make_subplots(rows=1, cols=4, 
        subplot_titles=('CPU Memory', 'CPU Utilization', 'GPU Memory', 'GPU Utilization'),
        specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}]]
        )

    fig['layout']['margin'] = {
        'l': 20, 'r': 20, 'b': 20, 't': 20
    }

    cpu_mem_used = int(psutil.virtual_memory()[3])
    cpu_mem_free = int(psutil.virtual_memory()[4])
    cpu_util_used = float(psutil.cpu_percent())
    cpu_util_free = float(100 - cpu_util_used)
    gpu_mem_used, gpu_util_used = _get_gpu_monitor()
    gpu_mem_free = float(100 - gpu_mem_used)
    gpu_util_free = float(100 - gpu_util_used)

    fig.update_layout(autosize=True, showlegend=False, height=150)
    fig = _add_pie(fig, [cpu_mem_used, cpu_mem_free], 'CPU Memory', 1, 1)
    fig = _add_pie(fig, [cpu_util_used, cpu_util_free], 'CPU Utilization', 1, 2)
    fig = _add_pie(fig, [gpu_mem_used, gpu_mem_free], 'GPU Memory', 1, 3)
    fig = _add_pie(fig, [gpu_util_used, gpu_util_free], 'GPU Utilization', 1, 4)

    return fig

# Multiple components can update everytime interval gets fired.
@_app.callback(Output('live-update-loss', 'figure'), [Input('loss-update', 'n_intervals')])
def _update_loss_live(n):
    r = redis.Redis()
    data = pickle.loads(r.get('history'))

    fig = plotly.subplots.make_subplots(rows=1, cols=1, 
        subplot_titles=('Loss',)
        )

    fig['layout']['margin'] = {
        'l': 20, 'r': 20, 'b': 20, 't': 20
    }
    
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    fig.update_layout(autosize=True, showlegend=True, height=200)
    fig = _add_trace(fig, list(range(len(data['train_loss']))), data['train_loss'], 'Train Loss', 1, 1)
    fig = _add_trace(fig, list(range(len(data['val_loss']))), data['val_loss'], 'Validation Loss', 1, 1)
    return fig

@_app.callback(Output('live-update-acc', 'figure'), [Input('acc-update', 'n_intervals')])
def _update_acc_live(n):
    r = redis.Redis()
    data = pickle.loads(r.get('history'))

    fig = plotly.subplots.make_subplots(rows=1, cols=1, 
        subplot_titles=('Accuracy',)
        )

    fig['layout']['margin'] = {
        'l': 20, 'r': 20, 'b': 20, 't': 20
    }

    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    fig.update_layout(autosize=True, showlegend=True, height=200)

    fig = _add_trace(fig, list(range(len(data['train_acc']))), data['train_acc'], 'Train Accuracy', 1, 1)
    fig = _add_trace(fig, list(range(len(data['val_acc']))), data['val_acc'], 'Validation Accuracy', 1, 1)
    return fig




class _Dashboard:
    def __init__(self):
        self.log = logging.getLogger('werkzeug')
        self.log.disabled = True

    def start(self):
        self.flask_process = Process(target=_app.run_server, kwargs={'debug':False, 'host':'0.0.0.0'})
        self.flask_process.start()
    
    def stop(self):
        self.flask_process.terminate()
        self.flask_process.join()

if __name__ == '__main__':
    f = _Dashboard()
    f.start()
    time.sleep(3000)
    f.stop()