# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
# Importation des librairies standards:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.spatial.distance import cdist

def normalisation(df):
    return (df-df.min())/(df.max()-df.min())

def dist_euclidienne(x,y):
    return np.linalg.norm(y-x)

def centroide(df):
    return np.mean(df,axis=0)

def dist_centroides(data1, data2):
    return dist_euclidienne(centroide(data1),centroide(data2))

def initialise_CHA(data):
    return {i:[i] for i in range(len(data))}

def dist_linkage(linkage,df1,df2):
    if linkage =='centroide':
        return dist_centroides(df1,df2)
    r = cdist(df1,df2, 'euclidean')
    if linkage == 'complete':
        return np.max(r)
    if linkage == 'simple':
        return np.min(r)
    if linkage == 'average':
        return np.mean(r)
    
def fusionne(df, partition,verbose=False,linkage="centroide"):
    dist_min = +np.inf
    k1_dist_min, k2_dist_min = -1,-1
    p_new = dict(partition)
    for k1,v1 in partition.items():
        for k2,v2 in partition.items():
            if k1!=k2:
                dist = dist_linkage(linkage,df.iloc[v1], df.iloc[v2])
                if dist < dist_min:
                    dist_min = dist
                    k1_dist_min, k2_dist_min = k1, k2
    if k1_dist_min != -1:
        p_new.pop(k1_dist_min)
        p_new.pop(k2_dist_min)
        p_new[max(partition)+1] = [*partition[k1_dist_min], *partition[k2_dist_min]]
    if verbose and k1_dist_min !=-1:
        print("Distance mininimale trouvée entre ["+str(k1_dist_min) +"," +str(k2_dist_min) +"]  = "+str(dist_min))
    return p_new, k1_dist_min, k2_dist_min, dist_min

def CHA_centroid(df):
    res = []
    part = initialise_CHA(df)
    for _ in range(len(df)):
        fus=fusionne(df,part)
        part=fus[0]
        if fus[1]!=-1 and fus[2]!=-1:
            res.append([fus[1],fus[2],fus[3],len(part[max(part.keys())])])
    return res

def create_dendo(res):
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)    
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme pour notre clustering :
    scipy.cluster.hierarchy.dendrogram(res,leaf_font_size=24.)  # taille des caractères de l'axe des X)
    # Affichage du résultat obtenu:
    plt.show()
    
def CHA_centroid(df,verbose=False,dendrogramme=False):
    res = []
    part = initialise_CHA(df)
    for _ in range(len(df)):
        fus=fusionne(df,part,verbose)
        part=fus[0]
        if fus[1]!=-1 and fus[2]!=-1:
            res.append([fus[1],fus[2],fus[3],len(part[max(part.keys())])])
    if dendrogramme:
        create_dendo(res)
    return res

def CHA_complete(df,verbose=False,dendrogramme=False):
    res = []
    part = initialise_CHA(df)
    for _ in range(len(df)):
        fus=fusionne(df,part,verbose=verbose, linkage="complete")
        part=fus[0]
        if fus[1]!=-1 and fus[2]!=-1:
            res.append([fus[1],fus[2],fus[3],len(part[max(part.keys())])])
    if dendrogramme:
        create_dendo(res)
    return res

def CHA_simple(df,verbose=False,dendrogramme=False):
    res = []
    part = initialise_CHA(df)
    for _ in range(len(df)):
        fus=fusionne(df,part,verbose=verbose,linkage="simple")
        part=fus[0]
        if fus[1]!=-1 and fus[2]!=-1:
            res.append([fus[1],fus[2],fus[3],len(part[max(part.keys())])])
    if dendrogramme:
        create_dendo(res)
    return res

def CHA_average(df,verbose=False,dendrogramme=False):
    res = []
    part = initialise_CHA(df)
    for _ in range(len(df)):
        fus=fusionne(df,part,verbose=verbose,linkage="average")
        part=fus[0]
        if fus[1]!=-1 and fus[2]!=-1:
            res.append([fus[1],fus[2],fus[3],len(part[max(part.keys())])])
    if dendrogramme:
        create_dendo(res)
    return res

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    if linkage=='centroid':
        return CHA_centroid(DF,verbose,dendrogramme)
    elif linkage=='complete':
        return CHA_complete(DF,verbose,dendrogramme)
    elif linkage=='simple':
        return CHA_simple(DF,verbose,dendrogramme)
    elif linkage=='average':
        return CHA_average(DF,verbose,dendrogramme)
    else:
        raise ValueError("Le parametre linkage n est pas bon")

def affiche_resultat(Base,Centres,Affect):
    couleurs = ["b","g","c","m","y","k","w"]
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
    for elem in Affect.values():
        c=np.random.choice(couleurs)
        for i in elem:
            plt.scatter(Base.iloc[i,0],Base.iloc[i,1],color=c)

#-----------------------------
def init_kmeans(K,Ens):
    return np.array(pd.DataFrame(Ens).sample(n=K))

def inertie_cluster(Ens):
    return sum(dist_euclidienne(centroide(Ens),v)**2 for v in np.array(Ens))

def plus_proche(Exe,Centres):
    return np.argmin([dist_euclidienne(Exe,centre) for centre in Centres])

def affecte_cluster(Base,Centres):
    dict_centre={i:[] for i in range(0,len(Centres))}
    for j in range (0,len(Base)):
        dict_centre[plus_proche(np.array(Base)[j],Centres)].append(j)
        
    return dict_centre

def nouveaux_centroides(Base,U):
    Base_numpy=np.array(Base) #pour pouvoir utiliser np.mean
    result=[]
    for _,val in U.items():
        result.append(np.mean([Base_numpy[i] for i in val],axis=0))
    
    return np.array(result)

def inertie_globale(Base, U):
    Base_numpy=np.array(Base)
    return sum([inertie_cluster([Base_numpy[i] for i in valeur]) for valeur in U.values()])
    
def kmoyennes(K, Base, epsilon, iter_max, affichage = True):
    Centres=init_kmeans(K,Base)
    U=affecte_cluster(Base,Centres)
    inertie_nouv=inertie_globale(Base,U)
    if affichage:
        print("Iteration : ",1," Inertie : ",inertie_nouv," Difference : ",inertie_nouv-epsilon-1)
    for i in range(1,iter_max):
        inertie_ancien=inertie_nouv
        #Recalcul de Centres et U
        Centres=nouveaux_centroides(Base,U)
        U=affecte_cluster(Base,Centres)
        inertie_nouv=inertie_globale(Base,U)
        diff=inertie_ancien-inertie_nouv
        if affichage:
            print("Iteration : ",i+1," Inertie : ",inertie_nouv," Difference : ",diff)
        if (diff < epsilon):
            break
        
    return Centres,U
        