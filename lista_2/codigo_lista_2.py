#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import seaborn as sns
import pandas as pd

#%% Leitura da base de dados
df = pd.read_excel('../datasets/treated_ger_energ_ele_hidr.xlsx')
serie_geracao = pd.Series(data=df["geracao_gwh"])
serie_geracao.index = df['mes_ano'].values

#%% Separa as bases de treinamento e teste
serie_geracao_treino = serie_geracao[:-12]
serie_geracao_teste = serie_geracao[-12:]

#%% Obtém a primeira e segunda diferença da série original
diff_1 = serie_geracao_treino.diff().dropna()
diff_2 = diff_1.diff().dropna()

# %%  FAC E FACP DAS DIFERENÇAS
def acf_pacf_grid(ts, ts_name, num_lags=50):
    """"""
    # Create a figure and axes with a 1x2 grid
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # Calculate the ACF and plot it on the left side
    acf_fig = plot_acf(ts, lags=num_lags, ax=axs[0])
    axs[0].set_title(None)

    # Calculate the PACF and plot it on the right side
    pacf_fig = plot_pacf(ts, lags=num_lags, ax=axs[1], method='ols')
    axs[1].set_title(None)

    # Set titles for each chart
    axs[0].set_xlabel('Lag')
    axs[0].set_ylabel('FAC')
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('FACP')

    # Set a title for the grid
    fig.suptitle(f'FAC / FACP - {ts_name}')

    # Adjust the spacing between the subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

# %%
def plot_chart(x, y, x_label, y_label, title, line_color='blue'):
    # Create a line plot using Seaborn
    sns.set(style="whitegrid")  # Set the style of the plot
    plt.figure(figsize=(8, 5))  # Set the size of the plot

    sns.lineplot(x=x, y=y, color=line_color)  # Use the specified line color

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add dashed red vertical lines at multiples of 12 on the x-axis
    # for i in range(len(x)):
    #     if (i + 1) % 12 == 0:
    #         plt.axvline(x=x.iloc[i], color='red', linestyle='--', linewidth=1)
    plt.show()

#%% Plot da série original - período de treino
plot_chart(
    x=serie_geracao_treino.index, 
    y=serie_geracao_treino,
    title='Geração de energia elétrica hidráulica',
    x_label='Tempo (mensal)',
    y_label='Geração (GWh)',
    line_color='xkcd:sky blue'
)

#%% Plot da 1ª primeira diferença da série temporal - período de treino
plot_chart(
    x=diff_1.index, 
    y=diff_1,
    title='1ª Diferença da série temporal',
    x_label='Tempo (mensal)',
    y_label='Geração (GWh)',
    line_color='xkcd:goldenrod'
)

#%% Plot da 2ª primeira diferença da série temporal - período de treino
plot_chart(
    x=diff_2.index, 
    y=diff_2,
    title='2ª Diferença da série temporal',
    x_label='Tempo (mensal)',
    y_label='Geração (GWh)',
    line_color='xkcd:light purple'
)

#%% Plot da FAC e FACP da segunda diferença da série
acf_pacf_grid(diff_2, '2ª Diferença da série temporal')

# %% ARMA (P,Q)
arma_models_aicc = []
# df_arma = pd.DataFrame(columns=['P Q','aicc'])

for p in range(0,6):
    for q in range (0, 6):
        print("modelo atual: {},{}".format(p,q))
        # # Versão Joao
        model = ARIMA(diff_2.values, order=(p, 0, q))
        results = model.fit()
        model_aicc = {'(p,q)': f'{p},{q}', 'AICc': results.aicc}
        arma_models_aicc.append(model_aicc)


#%% Descobre qual dos modelos ARMA(p,q) apresenta o menor valor de AICc
df_arma = pd.DataFrame(arma_models_aicc)
best_arma_aicc = df_arma.loc[df_arma['AICc'] == df_arma['AICc'].min()]['(p,q)'].iloc[0]
print('Modelo ARMA com menor AICc: ({})'.format(best_arma_aicc))
# df_arma.to_excel('ARMAPQ.xlsx', index=False) 
#%%
