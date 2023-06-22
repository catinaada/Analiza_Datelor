import numpy as np
import pandas as pd
import sklearn.cross_decomposition as skl
import matplotlib.pyplot as plt

t_ind=pd.read_csv('./dataIn/Industrie.csv',index_col=0)
t_pop=pd.read_csv('./dataIn/PopulatieLocalitati.csv',index_col=0)

# print(t_ind)
# print(t_pop)


#cerinta 1
# Să se salveze în fișierul Cerinta1.csv cifra de afaceri pe locuitor pentru
# fiecare activitate,
# la nivel de localitate. Pentru fiecare localitate se va salva codul Siruta,
# numele localității
# și cifra de afaceri pe locuitor pentru fiecare activitate. (2 punct)

industrii=list(t_ind)[1:]
# print(industrii)


t1=t_ind.merge(right=t_pop,left_index=True,right_index=True)
# print(t1)

def caLoc(t, variabile,populatie):
    #imparte fiecare coloana la pop
    x=t[variabile].values/t[populatie]
    # print(x)
    v=list(x)
    v.insert(0,t['Localitate_x'])
    return pd.Series(data=v,index=['Localitate']+variabile)

#caLoc(t1[['Localitate_x','Populatie']+industrii],industrii,'Populatie')

# print(t2)
t2=t1[['Localitate_x','Populatie']+industrii].apply(func=caLoc,axis=1, variabile = industrii, populatie = 'Populatie')
t2.to_csv('./dataOut/Cerinta1.csv')

# Cerinta 2
# Să se calculeze și să se salveze în fișierul Cerinta2.csv
# activitatea industrială dominantă (cu cifra de afaceri cea
# mai mare) la nivel de județ. Pentru fiecare județ se va
# afișa indicativul de județ, activitatea dominantă și cifra
# de afaceri corespunzătoare

def maxCA(t):
    x=t.values
    max_linie=np.argmax(x)
    #print(max_linie)
    return pd.Series(data=[t.index[max_linie],x[max_linie]],
                     index=['Activitate','CA'])

t3 = t1[industrii+['Judet']].groupby(by='Judet').agg(sum)
# print(t3)
t4=t3[industrii].apply(func=maxCA,axis=1) #axis =1 = pe linii
#print(t4)

t4.to_csv('./dataOut/Ceinta2.csv')

# Cerinta 3

tabel=pd.read_csv('./dataIn/DataSet_34.csv',index_col=0)
print(tabel)

obs_nume=tabel.index.values
var_nume=tabel.columns.values
print(obs_nume)
print(var_nume)

x_coloane=var_nume[:4]
y_coloane=var_nume[4:]

X=tabel[x_coloane].values
Y=tabel[y_coloane].values

def standardiazre(X):
    medii=np.mean(X,axis=0)
    abateri=np.std(X,axis=0)
    return (X- medii)/abateri

Xstd=standardiazre(X)
Xstd_df=pd.DataFrame(data=Xstd,index=obs_nume,columns=x_coloane)
Xstd_df.to_csv('./dataOut/Cerinta3 - Xstd.csv')

Ystd=standardiazre(Y)
Ystd_df=pd.DataFrame(data=Ystd,index=obs_nume,columns=x_coloane)
Ystd_df.to_csv('./dataOut/Cerinta3 - Ystd.csv')

# Cerinta 4

#creare model ACC - analiza canonica
n,p = np.shape(X) # nr de linii +coloane
q=np.shape(Y)[1] #nr coloane din y
m=min(p,q)
print(n,p,q,m)

modelACC=skl.CCA(n_components=m)
modelACC.fit(X=Xstd,Y=Ystd)

#SCORURI
z,u =modelACC.transform(X=Xstd,Y=Ystd)
print(z)
print(u)

z_df=pd.DataFrame(data=z,index=obs_nume,
                  columns=['z'+str(j+1) for j in range(p)])
z_df.to_csv('./dataOut/Cerinta4 Xscores.csv')

u_df=pd.DataFrame(data=u,index=obs_nume,
                  columns=['u'+str(j+1) for j in range(q)])
u_df.to_csv('./dataOut/Cerinta4 Yscores.csv')


#Cerinta 5
Rxz=modelACC.x_loadings_  #corelatia dintre x si z
Rxz_df=pd.DataFrame(data=Rxz,index=x_coloane,
                    columns=['z'+str(j+1) for j in range(p)])
Rxz_df.to_csv('./dataOut/Rxz.csv')

Ryu=modelACC.y_loadings_  #corelatia dintre x si z
Ryu_df=pd.DataFrame(data=Ryu,index=y_coloane,
                    columns=['u'+str(j+1) for j in range(q)])
Ryu_df.to_csv('./dataOut/Ryu.csv')

#Cerinta 6 - sa facem scatterplot

def biplot(x,y,xLabel='X',yLabel='Y',titlu='Biplot',e1=None,e2=None):
    f=plt.figure(figsize=(10,7))
    ax=f.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title(titlu,fontsize=14)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.scatter(x=x[:,0],y=x[:,1],c='r',label='Set X')
    ax.scatter(x=y[:,0],y=y[:,1],c='b',label='Set Y')
    ax.legend()

biplot(z[:,:2],u[:,:2],
       titlu='Biplot tari',xLabel='z1,u1',yLabel='z2,u2')# toate liniile si primele 2 coloane
plt.show()





