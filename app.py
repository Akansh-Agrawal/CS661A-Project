import json
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio
from prophet import Prophet
import pandas as pd
import numpy as np
from prophet.plot import plot_plotly

#loading geojson file
with open('Datasets (Preprocessed)/india.geojson') as f:
    india_geojson = json.load(f)

#dictionary representing the center coordinates and AQI marker for each states
cc = {
    'Andaman and Nicobar': [(11.7401, 92.6586),[]],
    'Telangana': [(17.3850, 78.4867), ['Hyderabad']],
    'Andhra Pradesh': [(15.9129, 79.7400),['Amaravati','Visakhapatnam']],
    'Arunachal Pradesh': [(27.0844, 93.6053),[]],
    'Assam': [(26.2006, 92.9376),['Guwahati']],
    'Bihar': [(25.0961, 85.3131),['Patna']],
    'Chandigarh': [(30.7333, 76.7794),['Chandigarh']],
    'Chhattisgarh': [(21.2787, 81.8661),[]],
    'Dadra and Nagar Haveli': [(20.1809, 73.0169),[]],
    'Daman and Diu': [(20.4283, 72.8397),[]],
    'Delhi': [(28.7041, 77.1025),['Delhi']],
    'Goa': [(15.2993, 74.1240),[]],
    'Gujarat': [(22.2587, 71.1924),['Ahmedabad']],
    'Haryana': [(29.0588, 76.0856),['Gurugram','Chandigarh']],
    'Himachal Pradesh': [(31.1048, 77.1734),[]],
    'Jammu and Kashmir': [(33.7782, 76.5762),[]],
    'Jharkhand': [(23.6102, 85.2799),['Jorapokhar']],
    'Karnataka': [(12.9716, 77.5946),['Bengaluru']],
    'Kerala': [(8.5241, 76.9366), ['Ernakulam','Thiruvananthapuram','Kochi']],
    'Lakshadweep': [(10.5667, 72.6417),[]],
    'Madhya Pradesh': [(22.9734, 78.6569),'Bhopal'],
    'Maharashtra': [(19.7515, 75.7139),['Mumbai']],
    'Manipur': [(24.6637, 93.9063),[]],
    'Meghalaya': [(25.4670, 91.3662),['Shillong']],
    'Mizoram': [(23.1645, 92.9376),'Aizwal'],
    'Nagaland': [(26.1584, 94.5624),[]],
    'Orissa': [(20.9517, 85.0985),['Brajrajnagar','Talcher']],
    'Puducherry': [(11.9139, 79.8145),[]],
    'Punjab': [(31.1471, 75.3412),['Amritsar','Chandigarh']],
    'Rajasthan': [(27.0238, 74.2179),['Jaipur']],
    'Sikkim': [(27.5330, 88.5122),[]],
    'Tamil Nadu': [(11.1271, 78.6569),['Chennai','Coimbatore']],
    'Tripura': [(23.9408, 91.9882),[]],
    'Uttar Pradesh': [(26.8467, 80.9462),['Lucknow']],
    'Uttaranchal': [(30.0668, 79.0193),[]],
    'West Bengal': [(22.9868, 87.8550),['Kolkata']]
}

#temp variables to support reset button callback function
temp=0
temp2=0
temp4=0
temp5=0
temp6=0

#26 cities of Air Quality Data
cities=['Ahmedabad', 'Aizawl','Amaravati', 'Amritsar', 'Bengaluru','Bhopal','Brajrajnagar','Chandigarh',
        'Chennai', 'Coimbatore','Delhi', 'Ernakulam','Gurugram', 'Guwahati','Hyderabad','Jaipur','Jorapokhar',
        'Kochi', 'Kolkata','Lucknow','Mumbai','Patna','Shillong','Talcher','Thiruvananthapuram','Visakhapatnam']

#AQI values for marker hover
aqi_hover=[332.8058735689398, 34.50442477876106, 90.869610935857, 115.59595959595958, 92.47934295669488, 130.10611303344868, 138.77931769722815, 96.1173245614035, 113.72473867595819, 68.92227979274611, 258.9394391903103, 93.32098765432099, 214.02481635894384, 138.72908366533864, 105.3000997008973, 132.58378216636746, 142.25634445394923, 103.98148148148148, 133.64496314496316, 209.42906918865108, 73.60511033681765, 209.04628632938645, 43.590322580645164, 153.85477477477477, 74.12230215827338, 105.33994528043776]

#latitude and longitude of markers

latitude=[23.0216238, 23.7435236, 16.5667, 31.6343083, 12.9767936, 23.2584857, 21.8498594, 30.72984395, 13.0836939,
     11.0018115, 28.6517178, 9.98408, 28.4646148, 26.1805978, 17.360589, 26.9154576, 23.7167069, 9.9674277, 22.5726459,
     26.8381, 19.0785451, 25.6093239, 25.5760446, 20.9458183, 8.4882267, 17.7231276]

longitude= [72.5797068, 92.7382905, 80.3667, 74.8736788, 77.590082, 77.401989, 83.9254698, 76.78414567016054, 80.270186,
      76.9628425, 77.2219388, 76.2741457, 77.0299194, 91.753943, 78.4740613, 75.8189817, 86.4110166, 76.2454436,88.3638953,
      80.9346001, 72.878176, 85.1235252, 91.8825282, 85.2111736, 76.947551, 83.3012842]

#units
units = {'PM10':'ug/m^3', 'PM2.5':'ug/m^3', 'NO': 'ug/m^3', 'NO2': 'ug/m^3', 'NOx':'ppb', 'NH3':'ug/m^3', 'CO':'mg/m^3', 'SO2':'ug/m^3', 'AQI':'unitless', 'Benzene':'ug/m^3', 'Toluene':'ug/m^3', 'Xylene':'ug/m^3', 'O3':'ug/m^3'}

# Loading preprocessed data
data = pd.read_csv('Datasets (Preprocessed)/AQIdatafinal.csv') # Air Quality final data
rainfall_means_data=pd.read_csv('Datasets (Preprocessed)/state_means.csv') # Rainfall mean data
df_cor=pd.read_csv('Datasets (Preprocessed)/correlation.csv') # Correlation data
dfnew=pd.read_csv('Datasets (Preprocessed)/new.csv') # Rainfall 100 years data

# Heatmap state locations and values to support map
heatmap_locations=['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh','Delhi', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Orissa', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttaranchal', 'West Bengal']
heatmap_colors=[rainfall_means_data[rainfall_means_data.State==i]['Average'].values[0] for i in heatmap_locations]

# Extract day, month, and year components of the date column
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Replacing inconsistencies in spelling
dfnew['State'] = dfnew['State'].replace('Chattisgarh', 'Chhattisgarh')
dfnew['State'] = dfnew['State'].replace('Dadra & Nagar Haveli', 'Dadra and Nagar Haveli')
dfnew['State'] = dfnew['State'].replace('Jammu & Kashmir', 'Jammu and Kashmir')
dfnew['State'] = dfnew['State'].replace('Pondicherry', 'Puducherry')

# Dictionary mapping state to cities for rainfall
state_cities = {state: list(group['City'].unique()) for state, group in dfnew.groupby('State')}

# Change datetime format for 100 years for forecatsing 
def getchange(df, oldDate, newDate, col):
    df[col] = pd.to_datetime(df[col], format=oldDate).dt.strftime(newDate)
    return df

old_date ='%d-%m-%Y'
new_date ='%Y-%m'
getchange(dfnew, old_date, new_date, "Date")

# Sort the data dataframe by day, month, and year
data = data.sort_values(['City', "Year", "Month", "Day"])

# Drop the temporary columns
data = data.drop(['Day', 'Month', 'Year'], axis=1)



# group mapping cities to AQI variables
groups = {city: group[['Date', 'AQI', 'PM10', 'PM2.5']].rename(columns={'Date': 'ds'})
          for city, group in data.groupby('City')}

dfnew['Date'] = pd.to_datetime(dfnew['Date'])

# group mapping cities to Rainfall variables
groups_2 = {city: group[['Date', 'RainfallVal', 'Month']].rename(columns={'Date': 'ds'})
          for city, group in dfnew.groupby('City')}

variables = list(data.columns)[2:-2] # To remove the "City", "Date", "AQI bucket", and "State"


#function for AQI forecast figure
def make_forecast(city, y_label):
    # Select y label
    group = groups[city]
    y = group[y_label].values

    # Split data into train and test sets
    train_size = int(len(group) * 0.8)
    train, test = group.iloc[:train_size], group.iloc[train_size:]

    # Fit Prophet model
    train=train.rename(columns={y_label:'y'})
    test=test.rename(columns={y_label:'y'})
    
    model = Prophet()
    model.fit(train)

    # Make predictions
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

    # Extract predictions for test period
    pred = forecast[-len(test):]['yhat'].values
    
    #making figure
    fig = plot_plotly(model, forecast)
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], name='Test'))
    fig.add_trace(go.Scatter(x=test['ds'], y=pred, name='Predictions'))
    
    fig.data[1].update(showlegend = False)
    fig.data[3].update(showlegend = False)
    fig.data[2].update(name = "Train Data Fit")
    
    fig.update_layout(title=f'{y_label} forecast for {city}', title_x = 0.5, title_y = 0.9)
    fig.update_layout(yaxis=dict(title = y_label, showticklabels=True, showgrid=True))
    fig.update_layout(xaxis=dict(title = "Time", showticklabels=True, showgrid=True))
    fig.update_layout(showlegend=True, width = None, autosize = True)
    
    return fig

#function for rainfall forecast figure    
def make_forecast_rainfall(city, y_label):
    # Select y label
    group = groups_2[city]
    y = group[y_label].values
    freq = pd.infer_freq(dfnew['Date'])
    # Split data into train and test sets
    train_size = int(len(group) * 0.8)
    train, test = group.iloc[:train_size], group.iloc[train_size:]

    # Fit Prophet model
    train=train.rename(columns={y_label:'y'})
    test=test.rename(columns={y_label:'y'})
   
    model = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_prior_scale=0.1, changepoint_prior_scale=0.5, n_changepoints=20, interval_width=0.95)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    model.fit(train)
    
    

    # Make predictions
    future = model.make_future_dataframe(periods=12)
    forecast = model.predict(future)

    # Extract predictions for test period
    pred = forecast[-len(test):]['yhat'].values

    #making figure
    fig = plot_plotly(model, forecast)
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], name='Test'))
    fig.add_trace(go.Scatter(x=test['ds'], y=pred, name='Predictions (Test)'))
    
    fig.data[1].update(showlegend = False)
    fig.data[3].update(showlegend = False)
    fig.data[2].update(name = "Train Data Fit")
    
    fig.update_layout(title = f'Rainfall forecast for {city}', title_x = 0.5, title_y = 0.9)
    fig.update_layout(yaxis=dict(title = "Rainfall (in cm)", showticklabels=True, showgrid=True))
    fig.update_layout(xaxis=dict(title = "Time", showticklabels=True, showgrid=True))
    fig.update_layout(showlegend=True, width = None, autosize = True)
    
    fig.update_layout(showlegend = True)
    
    return fig

#function for rainfall figure over centuries
def make_rainfall_plot(city, y_label):
    group = groups_2[city]
    group = group.drop_duplicates(subset = ['ds'])
    group = group[group['Month'] == y_label]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=group['ds'], y=group['RainfallVal'], mode='markers', marker=dict(color='black'), name='Rainfall values'))
    fig.add_trace(go.Scatter(x=group['ds'], y=group['RainfallVal'], mode='lines', line=dict(color='#1f77b4'), name='Connecting line'))
    
    fig.update_layout(title = f'Plot showing Rainfall in {city} for the month of {y_label} from 1901 to 2002', xaxis_title ='Date', yaxis_title='Rainfall (in cm)', title_x=0.5, title_y = 0.85)
    fig.update_layout(showlegend = True)
        
    return fig
    
#function for correlation figure
def make_correlation(city, vars):
    # Filter the data by city
    df_city = df_cor[df_cor['City'] == city]
    
    # Select the variables
    df_vars = df_city[vars]
    
    # Compute the correlation matrix
    corr_matrix = df_vars.corr()
    
    # Creating correlation figure
    fig = px.imshow(corr_matrix, labels=dict(x="Variable", y="Variable", color="Correlation"),
                    x=vars, y=vars, color_continuous_scale = 'plasma')
    
    # Update the layout
    fig.update_layout(title=f"Correlation Matrix for {city}", title_x = 0.48, title_y = 0.95)
    
    return fig

#function for multivariate aqi plot
def make_aqi_plot(city, var1_label, var2_label):
    
    dfay = data[data['City'] == city]
    dfay = dfay.sort_values(by = "Date")
    
    #making figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfay['Date'], y=dfay[var1_label], name = f'{var1_label} ({units[var1_label]})', line = {'color' : '#ff7f0e'}))

    
    fig.add_trace(go.Scatter(x=dfay['Date'], y=dfay[var2_label], name = f'{var2_label} ({units[var2_label]})', line = {'color' : '#1f77b4'}))

    
    fig.update_layout(autosize=True,width=None, title = "Air Quality Time Series Visualisation", title_x = 0.5, title_y = 0.9)
    fig.update_xaxes("")
    return fig

#initial plot figures
aqi_forecast_fig= make_forecast('Delhi','AQI') 
weather_forecast_fig=make_forecast_rainfall('Delhi','RainfallVal')
mrp=make_rainfall_plot('Delhi','January')
corr_fig= make_correlation('Delhi',['Rainfall(mm)'])
aqi_plot=make_aqi_plot('Delhi','PM10','AQI')

#Initial india map figure
india_fig = px.choropleth_mapbox(
    geojson=india_geojson,
    featureidkey="properties.NAME_1",
    locations=heatmap_locations,
    color=heatmap_colors,
    color_continuous_scale='bluyl',
    range_color=(30, 300),
    mapbox_style='carto-positron',
    zoom=3.3,
    center={"lat": 23.5, "lon": 80.5},
    opacity=0.5,
    labels={'color': 'Average Rainfall Over Century'},
    template='plotly_dark'
)

# Add scatter mapbox trace for the city markers
india_fig.add_trace(go.Scattermapbox(
    mode='markers',
    lat=latitude,
    lon=longitude,
    
    marker=go.scattermapbox.Marker(
        size=12,
        color="red",
        
    ),
    text=[f'City: {city}, AQI: {round(value,2)}' for city, value in zip(cities, aqi_hover)],
    hoverinfo='text'

))
india_fig.update_coloraxes(showscale=False)
india_fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0)  # Set the margin to 0 for left, right, top, and bottom
)
# Create Dash app
app = dash.Dash(__name__)

#header html container
header = html.Div(
    children=[
        html.H1("Seeing Beyond the Concrete...", className="header-title"),
        html.H2("A Visual Journey into Rainfall and Air Pollution")
        #html.P(" className="header-text"),
    ],
    className="header-container",
    style={
        "color": "white",
        "backgroundColor": "#25967C",
        "padding": "20px",
        "marginBottom": "20px",
        "textAlign": "center",
    },
)


#button css
button_css={
    'background-color': '#1e97cc',
    'border': 'none',
    'color': 'white',
    'padding': '10px 20px',
    'text-align': 'center',
    'text-decoration': 'none',
    'display': 'inline-block',
    'font-size': '16px',
    'margin': '4px 2px',
    'transition-duration': '0.4s',
    'cursor': 'pointer'
}

#seperator css
sep={
    'border-top': '8px solid #bbb',
    'border-radius': '5px',
    'margin-top': '10px',
    'margin-bottom': '20px',
    'background-color':'#962538'
}

#app layout html and css
app.layout = html.Div(children=[
    
        header,
    
        html.Div([
            dcc.Graph(id='india-map', figure=india_fig),
            html.Button('Reset', id='reset-button',n_clicks=0,style=button_css)
        ],),
    
        html.Hr(style=sep),
    
        html.Div([

            dcc.Graph(id='mrp',figure=mrp),
            dcc.Dropdown(

                id='dropdown4',
                options=sorted(dfnew['City'].unique()),
                value='Delhi'
            ),

            dcc.Dropdown(
                id='dropdown5',
                options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                value='January'
            ),
        ],),
    
        html.Hr(style=sep),
    
        html.Div([
            
            dcc.Graph(id='weather_forecast',figure=weather_forecast_fig),
            dcc.Dropdown(
                id='dropdown3',
                options=sorted(dfnew['City'].unique()),
                value='Delhi'
            )
        ],),
        
        html.Hr(style=sep),
    
        html.Div([
        
        dcc.Graph(id='aqi_plot',figure=aqi_plot),
        dcc.Dropdown(

            id='dropdown8',
            options=cities,
            value='Delhi'
        ),

        dcc.Dropdown(
            id='dropdown9',
            options=sorted(variables),
            value='PM10'
        ),
            
        dcc.Dropdown(
            id='dropdown10',
            options=sorted(variables),
            value='AQI'
        )
        ],),
        
    
        html.Hr(style=sep),

        html.Div([
            dcc.Graph(id='aqi_forecast',figure=aqi_forecast_fig),
            dcc.Dropdown(
                
                id='dropdown1',
                options=['AQI','PM10','PM2.5'],
                value='AQI'
            ),
            
            dcc.Dropdown(
                id='dropdown2',
                options=cities,
                value='Delhi'
            )
        ],),
    
        html.Hr(style=sep),
    
    
        html.Div([

        dcc.Graph(id='corr',figure=corr_fig),
        dcc.Dropdown(
            id='dropdown6',
            options=sorted(df_cor['City'].unique()),
            value=sorted(df_cor['City'].unique())[0]
        ),
        dcc.Dropdown(
            id='dropdown7',
            options=[{'label': v, 'value': v} for v in sorted(df_cor.columns[3:])],
            multi=True,
            value=[df_cor.columns[3]]
        ),
        ]),

    

    ],)


# callback functions

#callback functions to handle map interactivity
@app.callback(
    [Output('india-map', 'figure'),Output('dropdown2', 'options'),Output('dropdown2','value')],
    #Input('india-map', 'clickData'),
    [Input('india-map', 'clickData'), Input('reset-button', 'n_clicks')]
    #State('reset-button','n_clicks')
)
def display_state_map(clickData,n_clicks):
    global temp
    print(n_clicks,temp)
    if n_clicks>temp:
        temp=n_clicks
        return india_fig, cities, 'Delhi'
    
    if clickData is not None:
        
        state=clickData['points'][0]['location']
        state_fig = px.choropleth_mapbox(
            geojson=india_geojson,
            featureidkey="properties.NAME_1",
            locations=[state],
            color=[heatmap_colors[heatmap_locations.index(state)]],
            color_continuous_scale='bluyl',
            range_color=(0, 100),
            #color_continuous_scale='blues',
            #range_color=(0, 100),
            mapbox_style='carto-positron',
            zoom=5,
            center={"lat": cc[state][0][0], "lon": cc[state][0][1]},
            opacity=0.5,
            template='plotly_dark',
            labels={'color':'Average Rainfall Over Century'}
        )
        state_fig.update_layout(
            title=f"{state} Map",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        state_fig.add_trace(go.Scattermapbox(
        mode='markers',
        lat=latitude,
        lon=longitude,

        marker=go.scattermapbox.Marker(
            size=12,
            color="red",

        ),

        text=[f'City: {city}, AQI: {round(value,2)}' for city, value in zip(cities, aqi_hover)],
        hoverinfo='text'
        ))
        state_fig.update_coloraxes(showscale=False)
        state_fig.update_layout(showlegend=False)
        
        if(len(cc[state][1])!=0):
            return state_fig, cc[state][1], cc[state][1][0]
        else:
            return state_fig, cc[state][1], 'Delhi'
    else:
        return india_fig, cities, 'Delhi'

#callback functions to handle aqi forecast
@app.callback(
    Output('aqi_forecast', 'figure'),
    [Input('dropdown1', 'value'),
     Input('dropdown2', 'value')])
def update_figure(param, city):
    return make_forecast(city, param)

#callback functions to handle wather forecast interactivity
@app.callback(
    Output('weather_forecast', 'figure'),
    Input('dropdown3', 'value'))
def update_figure_2(city):
    return make_forecast_rainfall(city, 'RainfallVal')

@app.callback(
    [Output('dropdown3', 'options'),Output('dropdown3','value')],
    [Input('india-map', 'clickData'), Input('reset-button', 'n_clicks')])
def update_dropdown_3(clickData,n_clicks):
    global temp2
    print(n_clicks,temp2)
    if n_clicks>temp2:
        temp2=n_clicks
        return sorted(dfnew['City'].unique()), 'Delhi'
    
    if clickData is not None:
        
        state=clickData['points'][0]['location']
        
        if(len(state_cities[state])!=0):
            return sorted(state_cities[state]),sorted(state_cities[state])[0]
        else:
            return sorted(state_cities[state]), 'Delhi'
    else:
        return sorted(dfnew['City'].unique()), 'Delhi'

#callback functions to handle make rainfall plot interactivity
@app.callback(
    Output('mrp', 'figure'),
    [Input('dropdown4', 'value'),
     Input('dropdown5', 'value')])
def update_figure_4(city,month):
    return make_rainfall_plot(city, month)

@app.callback(
    [Output('dropdown4', 'options'),Output('dropdown4','value')],
    [Input('india-map', 'clickData'), Input('reset-button', 'n_clicks')])
def update_dropdown_4(clickData,n_clicks):
    global temp4
    #print(n_clicks,temp4)
    
    if n_clicks>temp4:
        temp4=n_clicks
        return sorted(dfnew['City'].unique()),'Delhi'
    
    if clickData is not None:
        
        state=clickData['points'][0]['location']
        if(len(state_cities[state])!=0):
            return sorted(state_cities[state]),sorted(state_cities[state])[0]
        else:
            return sorted(state_cities[state]), 'Delhi'
    
    else:
        return sorted(dfnew['City'].unique()),'Delhi'
    
#callback function to handle correlation matrix interactivity
    
@app.callback(
    Output('corr', 'figure'),
    [Input('dropdown6', 'value'),
     Input('dropdown7', 'value')])
def update_figure_correlation(city,var):
    return make_correlation(city, var)

@app.callback(
    [Output('dropdown6', 'options'),Output('dropdown6','value')],
    [Input('india-map', 'clickData'), Input('reset-button', 'n_clicks')])

def update_dropdown_6(clickData,n_clicks):
    global temp5
    #print(n_clicks,temp4)
    
    if n_clicks>temp5:
        temp5=n_clicks
        return sorted(df_cor['City'].unique()),'Delhi'
    
    if clickData is not None:
        
        state=clickData['points'][0]['location']
        common=[]
        for i in cc[state][1]:
            if i in df_cor['City'].unique():
                common.append(i)

        if(len(common)!=0):
            return sorted(common),sorted(common)[0]
        else:
            return sorted(common), 'Delhi'
    
    else:
        return sorted(df_cor['City'].unique()),'Delhi'

#callback functions to manage interactivity of aqi plot representing multiple variables on y axes
@app.callback(
    Output('aqi_plot', 'figure'),
    [Input('dropdown8', 'value'),Input('dropdown9', 'value'),Input('dropdown10', 'value')])
def update_figure_correlation(city,var1,var2):
    return make_aqi_plot(city, var1,var2)

@app.callback(
    [Output('dropdown8', 'options'),Output('dropdown8','value')],
    [Input('india-map', 'clickData'), Input('reset-button', 'n_clicks')])
def update_dropdown_6(clickData,n_clicks):
    global temp6
    #print(n_clicks,temp4)
    
    if n_clicks>temp6:
        temp6=n_clicks
        return sorted(cities),'Delhi'
    
    if clickData is not None:
        
        state=clickData['points'][0]['location']

        if(len(cc[state][1])!=0):
            return sorted(cc[state][1]),sorted(cc[state][1])[0]
        else:
            return sorted(cc[state][1]), 'Delhi'
    
    else:
        return sorted(cities),'Delhi'

# Running servers  
if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=False)
# Click on the link "http://127.0.0.1:8050" to run the file on your localhost.