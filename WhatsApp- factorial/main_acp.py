import pandas as pd
import numpy as np

vot=pd.read_csv("prezenta_vot.csv",index_col=0)
print(vot)
variabile=list(vot)[1:]
print(variabile)
x=vot[variabile].values
def acp(x, std=True, nlib=0):
    if std:
        x_ = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    else:
        x_ = x - np.mean(x, axis=0)
    n, m = np.shape(x)
    mat = (1 / (n - nlib)) * np.transpose(x_) @ x_
    valp, vecp = np.linalg.eig(mat)
    k = np.flipud(np.argsort(valp))
    alpha = valp[k]
    a = vecp[:, k]
    c = x_ @ a
    return alpha, c

alpha,c=acp(x)

m=len(variabile)
etichete=['Comp'+str(i+1) for i in range(m)]

# calcul componente
t_componente=pd.DataFrame(
    data=c,index=vot.index,columns=etichete
)
print(t_componente)

# calcul scoruri
scoruri=c/np.sqrt(alpha)
scoruri=pd.DataFrame(
    data=scoruri, index=vot.index,columns=vot.columns
)
