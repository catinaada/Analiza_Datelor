import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA

tabel=pd.read_csv("./dataIn/DataSet_34.csv",index_col=0)
print(tabel)

variabile=list(tabel)
print(variabile)

observatii=tabel.index.values
print(observatii)

# impartire set de date + standardizare
x_coloane=variabile[:4]
y_coloane=variabile[4:]
print(x_coloane)
print(y_coloane)

X=tabel[x_coloane].values
Y=tabel[y_coloane].values

def standardizare(X):
    medii=np.mean(X,axis=0)
    abateri=np.std(X,axis=0)
    return (X-medii)/abateri

Xstd=standardizare(X)
Xstd_df=pd.DataFrame(data=Xstd,index=observatii,columns=x_coloane)
print(Xstd_df)

Ystd=standardizare(Y)
Ystd_df=pd.DataFrame(data=Ystd,index=observatii,columns=y_coloane)
print(Ystd_df)

# construire model
n,p=np.shape(Xstd)
q=np.shape(Ystd)[1]
m=min(p,q)
print(n,p,q,m)

model_CCA=CCA(n_components=m)
model_CCA.fit(Xstd,Ystd)

# calcul scoruri
z,u=model_CCA.transform(Xstd,Ystd)

z_df=pd.DataFrame(data=z,index=observatii,columns=['z'+str(i+1) for i in range(p)])
print(z_df)

u_df=pd.DataFrame(data=u,index=observatii,columns=['u'+str(i+1) for i in range(q)])
print(u_df)

# calcul corelatii Rxz, Ryu
Rxz=model_CCA.x_loadings_
Rxz_df=pd.DataFrame(data=Rxz,index=x_coloane,columns=['z'+str(i+1) for i in range(p)])
print(Rxz_df)

Ryu=model_CCA.y_loadings_
Ryu_df=pd.DataFrame(data=Ryu,index=y_coloane,columns=['u'+str(i+1) for i in range(q)])
print(Ryu_df)

# calcul corelatii radacini canonice
r=np.diag(np.corrcoef(z,u,rowvar=False)[:m,m:])

# determinare relevanta rad canonice
r2=r*r


def test_bartlett(r2, n, p, q, m):
    x=1-r2
    df=[(p-k+1)*(q-k+1) for k in range(1,m+1)]
    l=np.flip(np.cumprod(np.flip(x)))
    chi2_=(-n+1+(p+q+1)/2)*np.log(l)
    return 1-chi2.cdf(chi2_,df)

p_value=test_bartlett(r2,n,p,q,m)
tabel_semnificatii=pd.DataFrame(
    data={
        'R':r,
        "R2":r2,
        'P_value':p_value
    },index=['root'+str(i+1)for i in range(m)]
)
print(tabel_semnificatii)

# nr radacini
nr=len(np.where(p_value<0.01)[0])
print(nr)

# varianta explicata
Rxz2=Rxz*Rxz
Ryu2=Ryu*Ryu
VX=np.sum(Rxz2,axis=0)
VY=np.sum(Ryu2,axis=0)
VX_T=sum(VX)
VY_T=sum(VY)

tabel_varianta=pd.DataFrame(data={
    'VX':np.append(VX,VX_T*100/p),
    'VY':np.append(VY,VY_T*100/q)
}, index=['root'+str(i+1)for i in range(m)]+['Procent'])
print(tabel_varianta)

# biplot-
def biplot(x,y,xLabel='X',yLabel='Y',titlu="Biplot"):
    fig=plt.figure(figsize=(8,5))
    ax=fig.add_subplot(1,1,1)
    ax.set_title(titlu,fontsize=14)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.scatter(x=x[:,0],y=x[:,1],c='m',label='Set X')
    ax.scatter(x=y[:,0],y=y[:,1],c='b',label='Set Y')
    ax.legend()
    plt.show()

biplot(z[:,:2],u[:,:2],xLabel='z1u1',yLabel='z2u2')