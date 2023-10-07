#%%
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd

# %matplotlib inline
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.graphics.tsaplots import plot_acf

plt.rcParams["figure.figsize"] = (12, 8)

#%%
import os
os.getcwd()
# %%
# df1 = pd.read_excel('treated_ger_energ_ele_hidr.xlsx', index_col='mes_ano')
df1 = pd.read_excel('../datasets/treated_ger_energ_ele_hidr.xlsx')
serie_geracao = pd.Series(data=df1["geracao_gwh"])
# serie_geracao.plot(label = "geracao_gwh")

#%% Define the train and test datasets
df_train = df1.iloc[:-12,].copy()
df_test = df1.iloc[-12:,].copy()

#%%
def plot_chart(x, y, x_label, y_label, title):
    # Create a line plot using Seaborn
    sns.set(style="whitegrid")  # Set the style of the plot
    plt.figure(figsize=(8, 5))  # Set the size of the plot

    sns.lineplot(x=x, y=y)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add dashed red vertical lines at multiples of 12 on the x-axis
    for i in range(len(x)):
        if (i + 1) % 12 == 0:
            plt.axvline(x=x.iloc[i], color='red', linestyle='--', linewidth=1)
    plt.show()

#%%
# Plot da serie temporal completa - dataset de treino
plot_chart(
    x=df_train['mes_ano'], y=df_train['geracao_gwh'],
    title='Geração de energia elétrica hidráulica',
    x_label='Tempo (mensal)',
    y_label='Geração (GWh)'
)

#%%
# Plot dos cinco primeiros anos - dataset de treino
num_years = 10

plot_chart(
    x=df_train['mes_ano'].iloc[:num_years*12], 
    y=df_train['geracao_gwh'].iloc[:num_years*12],
    title='Geração de energia elétrica hidráulica',
    x_label='Tempo (mensal)',
    y_label='Geração (GWh)'
)


#%%
def boxplot_monthly(df, ts_time_col, ts_name_col, ts_name, title, box_color='lightgreen'):
    """"""
    df = df.copy()
    df['month'] = df[ts_time_col].apply(lambda x : x.month)
    month_order = [i + 1 for i in range(12)]

    data = [
        go.Box(
            y=df.loc[df['month'] == month][ts_name_col], 
            name=month,
            fillcolor=box_color,
            boxpoints='outliers', 
            line=dict(color='black'), 
        )
        for month in month_order
    ]

    layout = go.Layout(
        title=title,
        xaxis=dict(title="Meses do ano"),
        yaxis=dict(title=ts_name),
        showlegend=False
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()


#%%
df_train['series_first_diff'] = df_train['geracao_gwh'].diff()

#%%
# Plot da serie temporal completa - primeira diff da série (treino)
plot_chart(
    x=df_train['mes_ano'], y=df_train['series_first_diff'],
    title='Geração de energia elétrica hidráulica',
    x_label='Tempo (mensal)',
    y_label='Geração (GWh)'
)

#%%
boxplot_monthly(
    df=df_train,
    ts_time_col='mes_ano',
    ts_name_col='series_first_diff',
    ts_name='Geração (GWh)',
    title='Boxplots mensais - 1ª Diferença da série temporal',
    box_color='orange'
)

#%%
# Define each feature of the ETS models 

ets_models = {
    'ETS(A,A,N)': {
        'error_oper': 'add',
        'trend_oper': 'add',
        'damped_trend': False,
        'seasonal_oper': None,
        'seasonal_periods': None
    }, 
    'ETS(A,A,A)': {
        'error_oper': 'add',
        'trend_oper': 'add',
        'damped_trend': False,
        'seasonal_oper': 'add',
        'seasonal_periods': 12
    }, 
    'ETS(A,A,M)': {
        'error_oper': 'add',
        'trend_oper': 'add',
        'damped_trend': False,
        'seasonal_oper': 'mul',
        'seasonal_periods': 12
    }, 
    'ETS(A,Ad,A)': {
        'error_oper': 'add',
        'trend_oper': 'add',
        'damped_trend': True,
        'seasonal_oper': 'add',
        'seasonal_periods': 12        
    },
    'ETS(A,Ad,M)': {
        'error_oper': 'add',
        'trend_oper': 'add',
        'damped_trend': True,
        'seasonal_oper': 'mul',
        'seasonal_periods': 12        
    }
}


#%%
# Rename the model columns to plot the legends correctly
ets_models = {new_model_name: ets_models[model_name] for model_name, new_model_name in zip(list(ets_models), 
    ['ETS(A,A,N)', 'ETS(A,A,A)', 'ETS(A,A,M)', 'ETS(A,Ad,A)', 'ETS(A,Ad,M)'])}

#%%
# ============================ Questão 3.iii ===============================
# Estimate all of the ETS models with their own configurantion

for name in ets_models.keys():
    model = ETSModel(
        df_train['geracao_gwh'],
        error = ets_models[name]['error_oper'],
        trend = ets_models[name]['trend_oper'],
        damped_trend = ets_models[name]['damped_trend'],
        seasonal = ets_models[name]['seasonal_oper'],
        seasonal_periods = ets_models[name]['seasonal_periods']
    )
    ets_models[name]['fitted'] = model.fit()

# Get the diagnostics of the ETS model estimation
for name in ets_models.keys():
    print(ets_models[name]['fitted'].summary(), '\n\n')
    
#%%
# ============================ Questão 3.iv ===============================
# Plot the observed values and the fitted ETS model during the training period
for name in ets_models.keys():
    df_train["geracao_gwh"].plot(label = "série observada")
    ets_models[name]['fitted'].fittedvalues.plot(label = "série ajustada")
    plt.title(f'Série temporal original e série ajustada por {name} - Período de treinamento')
    plt.xlabel('Tempo (mensal)')
    plt.ylabel('Geração (GWh)')
    plt.legend(loc='upper right')
    plt.show()

# Calculate the error metrics and the AICc for all estimated ETS models 
for name in ets_models.keys():
    fitted_values = ets_models[name]['fitted'].fittedvalues
    ets_models[name]['metric_results'] = {
        'RMSE': mean_squared_error(df_train["geracao_gwh"], fitted_values, squared=False),
        'MAPE': mean_absolute_percentage_error(df_train["geracao_gwh"], fitted_values),
        'MAD': np.median(np.absolute(np.array(fitted_values) - np.median(fitted_values))),
        'AICc': ets_models[name]['fitted'].aicc
    }

# Build a dataframe as a table of error metrics for all estimated ETS models 
df_error_metrics = pd.DataFrame(data = [
    ets_models[name]['metric_results']
    for name in ets_models.keys()]
)
df_error_metrics.insert(0, 'Modelos estimados', ets_models.keys())
print(df_error_metrics, '\n')

# Look for the estimated ETS model with the lowest AICc
filter_min_aicc = df_error_metrics['AICc'] == df_error_metrics['AICc'].min()
best_model_aicc = df_error_metrics.loc[filter_min_aicc]['Modelos estimados'].iloc[0]
print('The model with lowest AICc in training period: {}'.format(best_model_aicc))

#%%
# ============================ Questão 3.v ===============================
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

# Plot the FAC chart for the residuals of all estimated ETS models
for i, name in enumerate(ets_models.keys()):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Função de autocorelação (FAC) - {name}')

    plot_acf(ets_models[name]['fitted'].resid, lags=96, ax=ax)
    ax.set_title(None)
    ax.set_xlabel("Lag")
    ax.set_ylabel("FAC")

    plt.show()

#%%
# ============================ Questão 3.vi ===============================
models_methods_predictions = {}

for name in ets_models.keys():
    fitted = ets_models[name]['fitted']
    models_methods_predictions['PRED_' + name] = (
        fitted.simulate(anchor="end", nsimulations=12, repetitions=1).tolist()
    )

df_models_predictions = pd.DataFrame(models_methods_predictions)

# Add naive predictions for method i)
df_models_predictions['PRED_NAIVE_I'] = df_train['geracao_gwh'].iloc[-1]

# Add naive predictions for method ii)
df_models_predictions['PRED_NAIVE_II'] = df_train['geracao_gwh'].iloc[-12:].values

# Add observed values in the test period
df_models_predictions.insert(0, 'OBSERVADO', df_test['geracao_gwh'].values)
df_models_predictions.index = df_test.index
df_models_predictions

# Build the dataframe with the error metric for each ETS model or naive method precitions
rows_error_metric_test = []

for col in df_models_predictions.columns[1:]:
    pred_values = df_models_predictions[col].values
    rows_error_metric_test.append({
        'Modelo': col.replace('PRED_', ''),
        'RMSE': mean_squared_error(df_models_predictions['OBSERVADO'], pred_values, squared=False),
        'MAPE': mean_absolute_percentage_error(df_models_predictions['OBSERVADO'], pred_values),
        'MAD': np.median(np.absolute(np.array(pred_values) - np.median(pred_values))),    
    })

df_models_errors_test = pd.DataFrame(rows_error_metric_test)
df_models_errors_test

# %%
# ============================ Questão 3.vii ===============================
