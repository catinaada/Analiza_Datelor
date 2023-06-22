import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster._agglomerative import  AgglomerativeClustering
from seaborn import kdeplot


# Analiza clusteri

# calcul partitie optimala
def calcul_ierarhie(x,metoda):
    h=linkage(x,method=metoda)
    #det nr de clusteri in partitia optimala
    nr_jonctiuni=h.shape[0]
    k_max=np.argmax(h[1:,2]-h[:(nr_jonctiuni-1),2])
    k=nr_jonctiuni-k_max
    return h,k

def calcul_partitie(x,k,metoda):
    hclust=AgglomerativeClustering(k,linkage=metoda,compute_distances=True)
    hclust.fit(x)
    coduri=hclust.labels_
    etichete_clusteri=np.array(["c"+str(cod+1) for cod in coduri])
    distante=hclust.distances_
    nr_jonctiuni = len(distante)
    j=nr_jonctiuni-k
    threshold=(distante[j]+distante[j+1])/2
    return etichete_clusteri,threshold

def plot_ierarhie(h,etichete,threshold, titlu="Plot Ierarhie"):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu)
    dendrogram(h,labels=etichete,color_threshold = threshold, ax=ax)
    plt.show()

def plot_distributii(z,p,variabila):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Distributii pentru variabila "+variabila)
    kdeplot(x=z,hue=p,ax=ax)
    plt.show()


tabel_date=pd.read_csv('ADN_Tari.csv',index_col=0)
variabile=list(tabel_date)[1:]
print(variabile)
metoda="ward"

x=tabel_date[variabile].values

h,k_opt=calcul_ierarhie(x,metoda=metoda)
plot_ierarhie(h,tabel_date.index,max(h[:,2])+1)


p_opt=calcul_partitie(x,k_opt,metoda)
tabel_partitii=pd.DataFrame(
    data={"partitia_opt":p_opt[0]},
    index=tabel_date.index
)
plot_ierarhie(h,tabel_date.index,p_opt[1])


# calcul partitie oarecare
p3=calcul_partitie(x,3,metoda)
tabel_partitii["p3"]=p3[0]

for i in range(len(variabile)):
    plot_distributii(x[:,i],p3[0],variabile[i])

print(tabel_partitii)



