#%%
#%%
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#%%
df1 = pd.read_excel('../datasets/treated_ger_energ_ele_hidr.xlsx')
serie_geracao = pd.Series(data=df["geracao_gwh"])

serie_geracao_treino = serie_geracao[:-12]
serie_geracao_teste = serie_geracao[-12:]


serie_geracao_treino.plot(ylabel = "geracao_gwh")

#%%      primeira diferença da série:

diff_1 = serie_geracao_treino.diff()
diff_1.plot(ylabel = "1a diff", color='orange')

#%%      segunda diferença da série:

diff_2 = diff_1.diff()
diff_2.plot(ylabel = "1a diff", color='purple')


# %%   ## FAC E FACP DAS DIFERENÇAS
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


#acf_pacf_grid(diff_1[1:], '1a diferença')
acf_pacf_grid(diff_2[2:], '2a diferença')

# %% ARMA (P,Q)
df_arma = pd.DataFrame(columns=['P Q','aicc'])

for i in range(0,6):
    for j in range (0, 6):
        p = i
        q = j
        arma_model = sm.tsa.ARIMA(diff_2, order=(p, 0, q)).fit()
        serie_geracao_treino.plot(color = 'blue')
        arma_model.fittedvalues.plot(ylabel='Fitted Values')
        dict = {
            'P Q':[str(p) + ' ' + str(q)],
            'aicc': [arma_model.aicc]
        }
        df_dict = pd.DataFrame(dict)
        df_arma = pd.concat([df_arma, df_dict], axis=0, ignore_index=True)

df_arma.to_excel('ARMAPQ.xlsx', index=False)
#%%
