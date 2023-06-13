# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz as gv

# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
	""" int * int * float^2 -> tuple[ndarray, ndarray]
		Hyp: n est pair
		p: nombre de dimensions de la description
		n: nombre d'exemples de chaque classe
		les valeurs générées uniformément sont dans [binf,bsup]
	"""

	data_desc = np.random.uniform(binf,bsup,(n*p,p))

	data_label = np.asarray([-1 for i in range(0,n)] + [+1 for i in range(0,n)])
	np.random.shuffle(data_label)
	
	return data_desc,data_label
	
	
	

def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
	""" les valeurs générées suivent une loi normale
		rend un tuple (data_desc, data_labels)
	"""
	data_desc_pos = np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
	data_desc_neg = np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
	data_desc = np.vstack((data_desc_neg,data_desc_pos))
	data_label = np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
	#np.random.shuffle(data_label)
	
	return data_desc,data_label
	
	
	
	
	
def plot2DSet(desc,labels):	
	""" ndarray * ndarray -> affichage
		la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
	"""
	desc_negatifs = desc[labels == -1]
	desc_positifs = desc[labels == +1]
	
	plt.scatter(desc_negatifs[:,0],desc_negatifs[:,1],marker='.', color="red") # 'o' rouge pour la classe -1
	plt.scatter(desc_positifs[:,0],desc_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1
	
	
	
	
	
def plot_frontiere(desc_set, label_set, classifier, step=30):
	""" desc_set * label_set * Classifier * int -> NoneType
		Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
		et plus le tracé de la frontière sera précis.		
		Cette fonction affiche la frontière de décision associée au classifieur
	"""
	mmax=desc_set.max(0)
	mmin=desc_set.min(0)
	x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
	grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
	
	# calcul de la prediction pour chaque point de la grille
	res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
	res=res.reshape(x1grid.shape)
	# tracer des frontieres
	# colors[0] est la couleur des -1 et colors[1] est la couleur des +1
	plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
	
	
	



def create_XOR(n, var):
	""" int * float -> tuple[ndarray, ndarray]
		Hyp: n et var sont positifs
		n: nombre de points voulus
		var: variance sur chaque dimension
	"""
	var_carr=np.array([[var,0],[0,var]])
	data_xor_pos1 = np.random.multivariate_normal(np.array([1,0]),var_carr,n)
	data_xor_pos2 = np.random.multivariate_normal(np.array([0,1]),var_carr,n)
	
	data_xor_pos = np.vstack((data_xor_pos1,data_xor_pos2))
	
	data_xor_neg1 = np.random.multivariate_normal(np.array([0,0]),var_carr,n)
	data_xor_neg2 = np.random.multivariate_normal(np.array([1,1]),var_carr,n)
	
	data_xor_neg= np.vstack((data_xor_neg1,data_xor_neg2))
	
	data_xor= np.vstack((data_xor_neg,data_xor_pos))
	label_xor = np.asarray([-1 for i in range(0,2*n)] + [+1 for i in range(0,2*n)])
	
	return data_xor,label_xor

def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    arrL = np.array(L)
    moy = arrL.sum()/len(arrL)
    e_t = np.std(arrL, dtype = np.float64)
    return moy,e_t



class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g



def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        for attribut in LNoms: #Pour chacun des attributs
            
            index = LNoms.index(attribut) # indice de l'attribut dans LNoms
            attribut_valeurs = np.unique([x[index] for x in X]) #liste des valeurs (sans doublon) prises par l'attribut
            
            # Liste des entropies de chaque valeur pour l'attribut courant
            entropies = []
            # Liste des probabilités de chaque valeur pour l'attribut courant
            probas_val = []
            
            for v in attribut_valeurs: #pour chaque valeur prise par l'attribut
                # on construit l'ensemble des exemples de X qui possède la valeur v ainsi que l'ensemble de leurs labels
                X_v = [i for i in range(len(X)) if X[i][index] == v]
                Y_v = np.array([Y[i] for i in X_v])
                e_v = entropie(Y_v)
                entropies.append(e_v)
                probas_val.append(len(X_v)/len(X))
                
            entropie_conditionnelle = 0
            
            # On calcule l'entropie conditionnelle de l'attribut courant
            for i in range(len(attribut_valeurs)): 
                entropie_conditionnelle += probas_val[i] * entropies[i]
            
            if entropie_conditionnelle < min_entropie:
                min_entropie = entropie_conditionnelle
                i_best = LNoms.index(attribut)
                Xbest_valeurs = attribut_valeurs
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud



def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y, return_counts=True)
    return valeurs[np.argmax(nb_fois)]



def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
        rem: la fonction utilise le log dont la base correspond à la taille de P
    """
    if len(P)<=1:
        return 0.0
    hsp = np.array(P)
    hsp2 = np.where(hsp!=0,hsp,12.25)
    return abs(np.sum((np.log(hsp2)/np.log(len(hsp)))*hsp))



def entropie(Y):
    """ np.array[String] -> float
        labels correspond à une array list de labels (classes)
        rend l'entropie de la distribution des classes dans cet array
    """
    P = [] # liste de la distribution de probabilités
    
    classes, nb_fois = np.unique(Y, return_counts=True)
    
    # on complète P
    for i in range(len(classes)):
        P.append(nb_fois[i]/sum(nb_fois))
        
    return shannon(P)



def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = cl.entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = cl.entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)



def partitionne(m_desc,m_class,n,s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    # Extraire les indices des exemples pour lesquels la valeur pour la colonne `n` est inférieure ou égale à `s`
    indices_inf_s = np.where(m_desc[:, n] <= s)[0]
    
    # Extraire les indices des exemples pour lesquels la valeur pour la colonne `n` est strictement supérieure à `s`
    indices_sup_s = np.where(m_desc[:, n] > s)[0]
    
    # Partitionner les descriptions et les classes selon les indices obtenus
    desc_inf_s = m_desc[indices_inf_s, :]
    class_inf_s = m_class[indices_inf_s]
    desc_sup_s = m_desc[indices_sup_s, :]
    class_sup_s = m_class[indices_sup_s]
    
    # Retourner un tuple composé de deux tuples contenant les descriptions et les classes de chaque partition
    return ((desc_inf_s, class_inf_s), (desc_sup_s, class_sup_s))




class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.Les_fils['inf'].classifie(exemple)
        return self.Les_fils['sup'].classifie(exemple)
        
    
        
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g
    
    

def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = cl.entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(cl.classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1               # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_set = None
        
        for i in range(nb_col):
            gain = 0
            ((seuil, entropie), (_, _)) = discretise(X, Y, i)
            partition = ((X, Y), (None, None))
            if seuil is not None:
                partition = partitionne(X, Y, i, seuil)
            gain = entropie_classe - entropie
            if gain > gain_max:
                gain_max = gain
                i_best = i
                Xbest_tuple = partition
                Xbest_seuil = seuil

        if (gain_max != float('-Inf')):
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_num(left_data,left_class, epsilon, LNoms), \
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(cl.classe_majoritaire(Y))
        
    return noeud