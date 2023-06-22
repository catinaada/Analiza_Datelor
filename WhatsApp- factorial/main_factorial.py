import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import numpy as np
import seaborn
import matplotlib.pyplot as plt

prezenta_vot=pd.read_csv("prezenta_vot.csv",index_col=0)
print(prezenta_vot)

categorii=list(prezenta_vot)[1:]
print(categorii)

x=prezenta_vot[categorii].values

# test bartlett
test_bartlett=calculate_bartlett_sphericity(x)
print(test_bartlett[1])
if test_bartlett[1]>0.01:
    print("Nu exista factori")

# test kmo
etichete_kmo=list(categorii)
etichete_kmo.append('Total')
index_kmo=calculate_kmo(x)
print(index_kmo[1])
if index_kmo[1]<0.5:
    print("Nu exista factori")

tabel_kmo=pd.DataFrame(
    data={
        'kmo':np.append(index_kmo[0],index_kmo[1])
    },index=etichete_kmo
)
print(tabel_kmo)

# construire model
q=len(categorii)
model_factorial=FactorAnalyzer(q,rotation=None)
model_factorial.fit(x)

etichete_factori=['F'+str(i+1) for i in range(q)]
# corelatii factoriale
l=model_factorial.loadings_
t_l=pd.DataFrame(l,categorii,etichete_factori)
print(t_l)

# scoruri factoriale
f=model_factorial.transform(x)
t_f=pd.DataFrame(f,prezenta_vot.index,etichete_factori)
print(t_f)

# calcul comunalitati
comm=model_factorial.get_communalities()
t_comm=pd.DataFrame(
    data={
        'Comunalitati':np.round(comm,3)
    },index=categorii
)
print(t_comm)

# calcul varianta
varianta=model_factorial.get_factor_variance()
tabel_varianta=pd.DataFrame(
    data={
        'Varianta':np.round(varianta[0],3),
        'Procent varianta':np.round(varianta[1]*100,3),
        'Procent cumulat': np.round(varianta[2] * 100, 3),
    }, index=etichete_factori
)
print(tabel_varianta)

def corelograma(t,vmin=-1,titlu='Corelograma'):
    fig=plt.figure(figsize=(8,5))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu,fontsize=14)
    seaborn.heatmap(t,vmin=vmin,vmax=1,cmap='RdYlBu',ax=ax)
    plt.show()

corelograma(t_l,titlu='Corelograma corelatii fact')
corelograma(t_f,titlu='Corelograma scoruri fact')

def plot_instante(t,k1=0,k2=1):
    fig=plt.figure(figsize=(8,5))
    ax=fig.add_subplot(1,1,1)
    ax.set_title("Plot instante",fontsize=14)
    ax.set_xlabel(t.columns[k1])
    ax.set_ylabel(t.columns[k2])
    ax.scatter(t.iloc[:,k1],t.iloc[:,k2],c='m')
    for i in range(len(t)):
        ax.text(t.iloc[i,k1],t.iloc[i,k2],t.index[i])
    plt.show()

plot_instante(t_f)