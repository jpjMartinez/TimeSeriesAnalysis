#%%
import pandas as pd

#%%
df1 = pd.read_excel('../datasets/treated_ger_energ_ele_hidr.xlsx')
serie_geracao = pd.Series(data=df["geracao_gwh"])

#%%
