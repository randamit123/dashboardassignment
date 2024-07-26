import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Data preprocessing and handling
data = pd.read_csv("./data/task_4.csv").dropna()
data['recording_date'] = pd.to_datetime(data['recording_date'])

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Task 5: Sensor Data Dashboard",
            style={'textAlign': 'center'}),
    html.H4(id="sensor-header",
            style={'textAlign': 'center'}),

    # Dropdown to select sensor
    dcc.Dropdown(
        id='sensor-dropdown',
        options=[{'label': f'Sensor {i}', 'value': i}
                 for i in data['sensor_id'].unique()],
        value=data['sensor_id'].unique()[0],  # default to the first sensor
        style={'width': '50%', 'margin': 'auto'}
    ),

    html.Div([
        # Time series graphs
        html.Div([
            dcc.Graph(id='contrast-graph-mean'),
            dcc.Graph(id='energy-graph-mean'),
            dcc.Graph(id='contrast-graph-std'),
            dcc.Graph(id='energy-graph-std')
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'left'})
    ], style={'textAlign': 'center'})
])


@app.callback(
    [Output('sensor-header', 'children'),
     Output('contrast-graph-mean', 'figure'),
     Output('energy-graph-mean', 'figure'),
     Output('contrast-graph-std', 'figure'),
     Output('energy-graph-std', 'figure')],
    [Input('sensor-dropdown', 'value')]
)
def update_graphs(selected_sensor):
    df = data[data['sensor_id'] == selected_sensor]

    # Convert dates to ordinal for regression
    df['date'] = df['recording_date'].map(
        pd.Timestamp.toordinal)

    # Columns and titles
    columns = {
        'CONTRAST_SNR_mean': 'Contrast Mean',
        'ENERGY_SNR_mean': 'Energy Mean',
        'CONTRAST_std': 'Contrast STD',
        'ENERGY_SNR_std': 'Energy STD'
    }

    titles = {
        'CONTRAST_SNR_mean': 'Contrast_SNR_mean over Time',
        'ENERGY_SNR_mean': 'Energy_SNR_mean over Time',
        'CONTRAST_std': 'CONTRAST_SNR_std over Time',
        'ENERGY_SNR_std': 'ENERGY_SNR_std over Time'
    }

    # Function to fit regression and create plot
    def create_figure(col):
        X = df[['date']]
        y = df[col]
        model = LinearRegression()
        model.fit(X, y)
        df[f'{col}_regression'] = model.predict(X)

        fig = px.scatter(
            df,
            x='recording_date',
            y=col,
            title=titles[col],
            labels={'recording_date': 'Dates',
                    col: columns[col]},
            template='simple_white'
        )
        fig.add_scatter(
            x=df['recording_date'],
            y=df[f'{col}_regression'],
            mode='lines',
            name='Regression Line',
        )
        return fig

    # Create plots
    contrast_mean = create_figure('CONTRAST_SNR_mean')
    energy_mean = create_figure('ENERGY_SNR_mean')
    contrast_std = create_figure('CONTRAST_std')
    energy_std = create_figure('ENERGY_SNR_std')

    return (f'Sensor {selected_sensor}', contrast_mean, energy_mean, contrast_std, energy_std)


if __name__ == '__main__':
    app.run_server(debug=True)
