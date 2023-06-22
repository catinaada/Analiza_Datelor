import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity


vot=pd.read_csv("prezenta_vot.csv",index_col=0)
print(vot)
coduri=pd.read_csv("Coduri.csv",index_col=0)
print(coduri)

# salvarea in cerinta1.csv a categoriei de alegatori pentru care
# s-a inregistrat cel mai mic procent ce prezenta la vot
# se va salva codul siruta, numele local, categoria

categorii=list(vot)[1:]
print(categorii)

def minim(t, categorii):
    x=t[categorii].values
    minimul=np.argmin(x)
    return pd.Series(data=[t['Localitate'],categorii[minimul]],index=[['Localitate','Categorie']])

df1=vot.apply(func=minim,axis=1,categorii=categorii)
df1.to_csv("Cerinta1.csv")

# salvarea in cerinta2.csv a valorilor medii la nivel de judet.
# se va salva indicativul judetului si valorile medii pt fiecare categorie

df=vot[categorii].merge(coduri,left_index=True,right_index=True)
print(df)

df2=df[categorii+['Judet']].groupby("Judet").agg(np.mean)
df2.to_csv("Cerinta2.csv")

# test relevanta
x=vot[categorii].values
test_bartlett=calculate_bartlett_sphericity(x)
print(test_bartlett)
if(test_bartlett[1]>0.01):
    print("Brtlett. Nu exista factori! ")
else:
    print("Brtlett. Exista factori! ")

# construire model
q=len(categorii)
model_factorial=FactorAnalyzer(q,rotation=None)
model_factorial.fit(x)

# soruri factoriale
etichete_factori=['F'+str(i+1) for i in range(q)]
f=model_factorial.transform(x)
t_f=pd.DataFrame(data=f,index=vot.index,columns=etichete_factori)
t_f.to_csv("f.csv")

# grafic primele 2 scoruri
def plot_scoruri(x, k1=0, k2=1, etichete=None, titlu="Plot instante (componente)",eticheta_axe="Comp"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_xlabel(eticheta_axe + str(k1+1))
    ax.set_ylabel(eticheta_axe + str(k2+1))
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    ax.scatter(x[:, k1], x[:, k2], c="r")
    if etichete is not None:
        n = len(etichete)
        for i in range(n):
            ax.text(x[i, k1], x[i, k2], etichete[i])

def plot_instante(t,k1=0,k2=1):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot instante", fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(t.columns[k1])
    ax.set_ylabel(t.columns[k2])
    ax.scatter(t.iloc[:,k1],t.iloc[:,k2],c="m")
    for i in range(len(t)):
        ax.text(t.iloc[i,k1],t.iloc[i,k2],t.index[i])

plot_scoruri(f,k1=0,k2=1,titlu="Plot scoruri (factori)",etichete=t_f.index,eticheta_axe="F")
plot_instante(t_f)
plt.show()
