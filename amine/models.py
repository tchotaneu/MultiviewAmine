#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
+-------------------------------------------------------------------------------------+
| This file is part of AMINE                                                          |
|                                                                                     |
| AMINE is free software: you can redistribute it and/or modify it under the terms of |
| the GNU General Public License as published by the Free Software Foundation, either |
| version 3 of the License, or (at your option) any later version.                    |
| You should have received a copy of the GNU General Public License along with AMINE. |
| If not, see <http://www.gnu.org/licenses/>.                                         |
|                                                                                     |
| Author: Claude Pasquier (I3S Laboratory, CNRS, Université Côte d'Azur)              |
| Contact: claude.pasquier@univ-cotedazur.fr                                          |
| Created on decembre 20, 2022                                                        |
+-------------------------------------------------------------------------------------+

Various graph models.

Each class defined here is instanciated with a networkx graph
After instanciation, it is possible to get, according to the model,
the closest nodes to a specific node with the method 'get_most_similar'
"""

import pathlib
from abc import ABC, abstractmethod
from typing import Union, Iterable,List
from node2vec import Node2Vec
import networkx as nx

import itertools
import numpy as np
from gensim.models import FastText, Word2Vec
from scipy.spatial import distance
from .dimension_reduction import node2vec
from .dimension_reduction.pecanpy import node2vec as n2v
from .datasets import Datasets
import matplotlib.pyplot as plt


class Model(ABC):
    """
    abstract class.

    Three implementation are proposed:
        - Node2vec,
        - RandomWalk
        - SVD.
    """

    @abstractmethod
    def get_most_similar(self, elt: str, number: int):
        """
        Collect similar nodes ; method implemented by each subclass.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """


class Node2vec(Model):
    """Node2vec model."""

    def __init__(self):
        """Declare variables."""
        self.model = None
        self.num_walks = 20  # 10
        self.walk_length = 100  # 80
        self.directed = False
        self.param_p = 1  # 4  # 0.15
        self.param_q = 1  # 2
        self.dimensions = 64  # 128
        self.window_size = 5  # 10
        self.workers = 4
        self.epoch = 10  # 10

    def init(self, G: nx.Graph, list_nodes: Iterable = None, precomputed: str = None):
        """
        Initialize the model with a weighted graph.

        Parameters
        ----------
        G           : nx.Graph
                      the graph used to initialize the model
        list_nodes  : Iterable, optional
                      specify an order of the nodes to be used, default is None
        precomputed : str or file-like object, optional
                      None or path to the precomputed model that must be used, default is None

        """
        if list_nodes is None:
            list_nodes = list(G.nodes)
        if precomputed and pathlib.Path(precomputed).is_file():
            self.model = Word2Vec.load(precomputed)
        else:
            for node in G.nodes():
                for nbr in sorted(G.neighbors(node)):
                    G[node][nbr]["weight"] = 1 - abs(
                        G.nodes[node]["weight"] - G.nodes[nbr]["weight"]
                    )
            self.compute_embedding(G, list_nodes)
            if precomputed:
                self.save(precomputed)

    def compute_embedding(self, G: nx.Graph, list_nodes: list):
        """
        Compute embedding.

        Parameters
        ----------
        G           : nx.Graph
                      the processed graph
        list_nodes  : list of nodes
                      the list of start nodes from the randomwalk

        """
        use_pecanpy = False
        if use_pecanpy:
            # from pecanpy import node2vec as n2v
            graph = n2v.SparseOTF(
                p=self.param_p,
                q=self.param_q,
                workers=self.workers,
                verbose=False,
                extend=True,
            )
            A = np.array(
                nx.adjacency_matrix(
                    G, nodelist=sorted(G.nodes), weight="weight"
                ).todense(),
                dtype=np.float_,
            )
            # isolated_nodes = np.where(~A.any(axis=1))[0]
            # print(np.where(~A.any(axis=0))[0])
            # print(nx.is_connected(G))
            # A = np.delete(A, isolated_nodes, axis=0)
            # A = np.delete(A, isolated_nodes, axis=1)
            graph.from_mat(A, sorted(G.nodes))
            walks = graph.simulate_walks(
                num_walks=self.num_walks,
                walk_length=self.walk_length,
                list_nodes=list_nodes,
            )
        else:
            graph = node2vec.Graph(G, self.directed, self.param_p, self.param_q)
            graph.preprocess_transition_probs()
            walks = graph.simulate_walks(
                self.num_walks, self.walk_length, nodes=list_nodes
            )

        # Learn embeddings by optimizing the Skipgram objective using SGD.
        walks = [list(map(str, walk)) for walk in walks]
        # import pickle
        # with open("/home/cpasquie/Téléchargements/test.txt", "wb") as fp:   #Pickling
        #     pickle.dump(walks, fp)
        # dd
        # with open("/home/cpasquie/Téléchargements/test.txt", "rb") as fp:   # Unpickling
        #     walks = pickle.load(fp)
        use_fasttext = False
        if use_fasttext:
            self.model = FastText(
                vector_size=self.dimensions,
                window=self.window_size,
                min_count=1,
                sentences=walks,
                epochs=self.epoch,
                max_n=0,
                sg=1,
            )
        else:
            self.model = Word2Vec(
                walks,
                vector_size=self.dimensions,  # size=self.dimensions,
                window=self.window_size,
                min_count=5,
                negative=5,
                sg=1,
                workers=self.workers,
                epochs=self.epoch,
            )  # iter=self.epoch)

    def save(self, fname_or_handle: str):
        """
            Save the model to file.

        Parameters
        ----------
        fname_or_handle : str or file-like object
                          path or handle to file where the model will be persisted

        """
        self.model.save(fname_or_handle)

    def load(self, fname_or_handle: str):
        """
        Load a previously saved model from a file.

        Parameters
        ----------
        fname_or_handle : str or file-like object
                          path or handle to file that contains the model

        """
        self.model = Word2Vec.load(fname_or_handle)

    def get_most_similar(self, elt: str, number: int): 
        """
        Collect similar nodes.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """
        return [int(x[0]) for x in self.model.wv.similar_by_word(str(elt), topn=number)]

    def get_distance(self, elt1: str, elt2: str):
        """
        Return the distance between two elements.

        Parameters
        ----------
        elt1 : str
            first element
        elt2 : str
            second element

        """
        return self.model.wv.distance(str(elt1), str(elt2))

    def get_vector(self, elt: Union[str, int]):
        """
        Get the vector encoding the element

        Parameters
        ----------
        elt : Union[str, int]
            the element

        Returns
        -------
        vector
            the vector encoding the element
        """
        return self.model.wv.get_vector(str(elt))


class RandomWalk(Model):
    """
    RandomWalk model.
    """

    # convergence criterion - when vector L1 norm drops below 10^(-6)
    # (this is the same as the original RWR paper)
    conv_threshold = 0.000001

    def __init__(self):
        """Declare variables."""
        self.nodelist = None
        self.walk_length = 0
        self.restart_prob = 0
        self.T = None

    def init(self, G: nx.Graph):
        """
        Initialize the model with a weighted graph.

        Parameters
        ----------
        G   : nx.Graph
              the graph used to initialize the model

        """
        self.nodelist = sorted(G.nodes)
        self.walk_length = 200
        self.restart_prob = 0.7

        for node in G.nodes():
            for nbr in sorted(G.neighbors(node)):
                G[node][nbr]["weight"] = 1 - abs(
                    G.nodes[node]["weight"] - G.nodes[nbr]["weight"]
                )

        # Create the adjacency matrix of G
        A = np.array(
            nx.adjacency_matrix(G, nodelist=self.nodelist, weight="weight").todense(),
            dtype=np.float_,
        )
        # Create the degree matrix
        D = np.diag(np.sum(A, axis=0))

        # The Laplacian matrix L, not used here is equal to D - A

        # Compute the inverse of D
        # Several solutions are possible
        #     - first solution: numpy.inverse
        #       inverse_of_d = numpy.linalg.inv(D)
        #     - second solution: numpy.solve
        #       inverse_of_d = numpy.linalg.solve(D, numpy.identity(len(nodes_list))
        #     - third solution, as the matrix is diagonal, one can use
        #       the inverse of the diagonal values
        #       inverse_of_d = np.diag(1 / np.diag(D))

        inverse_of_d = np.diag(1 / np.diag(D))

        # compute the transition matrix
        self.T = np.dot(inverse_of_d, A)

    def get_most_similar(self, elt: str, number: int):
        """
        Collect similar nodes.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """
        arr = [0] * len(self.nodelist)
        arr[elt] = 1
        p_0 = np.array(arr)
        state_matrix = p_0
        for _ in range(self.walk_length):

            # evaluate the next state vector
            p_1 = (1 - self.restart_prob) * np.dot(
                state_matrix, self.T
            ) + self.restart_prob * p_0

            # calculate L1 norm of difference between p^(t + 1) and p^(t),
            # for checking the convergence condition
            diff_norm = np.linalg.norm(np.subtract(p_1, state_matrix), 1)
            if diff_norm < RandomWalk.conv_threshold:
                break
        state_matrix = p_1
        result = sorted(
            enumerate(state_matrix.tolist()), key=lambda res: res[1], reverse=True
        )
        return [int(x[0]) for x in result][1 : number + 1]


class SVD(Model):
    """SVD model."""

    def __init__(self):
        """Declare variables."""
        self.nodelist = None
        self.most_similar = []

    def init(self, G: nx.Graph):
        """
        Initialize the model with a weighted graph.

        Parameters
        ----------
        G   : nx.Graph
              the graph used to initialize the model

        """
        self.nodelist = sorted(G.nodes)
        A = np.array(
            nx.adjacency_matrix(G, sorted(G.nodes), weight=None).todense(),
            dtype=np.float_,
        )
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        reduced_dimension = A.shape[0] // 5
        reduced_matrix = U * S
        reduced_matrix = reduced_matrix[:, 0:reduced_dimension]
        self.most_similar = []
        for ctr in range(reduced_matrix.shape[0]):
            dist = distance.cdist(
                reduced_matrix[ctr : ctr + 1], reduced_matrix[0:], "cosine"
            )
            self.most_similar.append(
                [
                    x[0]
                    for x in sorted(
                        list(enumerate(dist[0].tolist())), key=lambda x: x[1]
                    )
                ][1:]
            )

    def get_most_similar(self, elt: str, number: int):
        """
        Collect similar nodes.

        Parameters
        ----------
        elt    : str
                 node identifier
        number : int
                 number of most similar nodes to collect

        """
        return self.most_similar[elt][:number]




class Multiview:
    def __init__(self):

        self.min_count =1
        self.seed = 42
        self.workers=4
        self.modefusion = "union" # inter
        self.model1 = None
        self.model2 = None
        self.view1 = None
        self.view2 = None

    def init(self, G: nx.Graph):
        """
        Génère les deux vues à partir du graphe G et calcule leurs embeddings.
        """
        self.view1, self.view2= self.genererG(G)        #self.genererVue_Simba(G)
        print("la vue 1",self.view1)
        print("la vue 2",self.view2)
        truehits = Datasets.get_groups(G)
        noeuds=truehits[1]
        poidTrueHit=self.extraire_poids_trueHIT(G,truehits)
        stats=self.stats_poids_noeuds(G)
        self.visualiser_sousgraphe_pvalue(G,noeuds)
        print(poidTrueHit)
        print(stats)
       
        self.model1 = self.compute_embedding(self.view1,
                                            dimensions=64, #64
                                            walk_length=75,
                                            num_walks=40,
                                            p=0.25, #0.25
                                            q=2, #2 
                                            window=10,
                                            negative=5, 
                                            sg=1, 
                                            epochs=25)
        
        self.model2 = self.compute_embedding(self.view2,
                                            dimensions=64,
                                            walk_length=50,# 40 ou 50
                                            num_walks=40, #20 ou 30
                                            p=0.25,      #0.25
                                            q=2,         # 2 encourage la coherence locale
                                            window=10,
                                            negative=5, 
                                            sg=1, 
                                            epochs=25)
###############################
    def visualiser_sousgraphe_pvalue(self,G, noeuds, fichier_png='sousgraphe_truehits.png'):
        subG = G.subgraph(noeuds)
        pos = nx.spring_layout(subG, seed=42)
        labels = {n: f"{n}\n{G.nodes[n].get('weight', ''):.3f}" for n in subG.nodes}
        pvalues = [G.nodes[n].get('weight', 0.0) for n in subG.nodes]
        sizes = [300 + 1000*(0.05 - min(p, 0.05)) for p in pvalues]

        # Correction : créer explicitement un Axes
        fig, ax = plt.subplots(figsize=(10, 7))
        nodes = nx.draw_networkx_nodes(
            subG, pos, ax=ax,
            node_color=pvalues, node_size=sizes, cmap=plt.cm.viridis_r
        )
        edges = nx.draw_networkx_edges(subG, pos, ax=ax, edge_color='gray')
        nx.draw_networkx_labels(subG, pos, labels=labels, ax=ax, font_size=10)
        ax.set_title("Sous-graphe des true hits (chaque nœud avec sa p-value)")
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=plt.Normalize(vmin=min(pvalues), vmax=max(pvalues)))
        sm._A = []  # Nécessaire pour compatibilité matplotlib < 3.1
        fig.colorbar(sm, ax=ax, label='p-value')
        plt.tight_layout()
        plt.savefig(fichier_png)
        plt.close()
        print(f"Le sous-graphe a été sauvegardé dans '{fichier_png}'.")



####################################
    def stats_poids_noeuds(self,G, noeuds=None):
        """
        Calcule les statistiques descriptives des poids (p-values) sur un ensemble de nœuds du graphe.
        
        Paramètres :
        - G : nx.Graph, graphe avec attributs 'weight' sur les nœuds
        - noeuds : liste ou ensemble d'ID de nœuds à analyser (ou None pour tous les nœuds)
        
        Retourne :
        - Un dictionnaire avec moyenne, médiane, écart-type, min, max, et effectif
        """
        if noeuds is None:
            poids = [data.get('weight') for _, data in G.nodes(data=True) if data.get('weight') is not None]
        else:
            poids = [G.nodes[n].get('weight') for n in noeuds if G.nodes[n].get('weight') is not None]
        
        if not poids:
            return None  # Aucun poids trouvé
        
        stats = {
            "effectif": len(poids),
            "moyenne": np.mean(poids),
            "mediane": np.median(poids),
            "ecart_type": np.std(poids),
            "min": np.min(poids),
            "max": np.max(poids),
            "proportion_<0.05": np.mean(np.array(poids) < 0.05)
        }
        return stats

################################""

    def train_word2vec_from_walks(self, walks, dimensions, window, negative, sg, epochs):
        """
        Entraîne manuellement Word2Vec avec contrôle total des paramètres.
        """
        model = Word2Vec(
            walks,
            vector_size=dimensions,
            window=window,
            min_count=self.min_count,
            negative=negative,
            sg=sg,
            workers=self.workers,
            epochs=epochs,
            seed=self.seed)
        return model

    def compute_embedding(self,  G: nx.Graph, dimensions=64, walk_length=40, num_walks=20, p=0.25, q=2,
                           window=10, negative=5, sg=1, epochs=40):
        """
        Entraîne un modèle Word2Vec à partir d'un seul graphe, 
        possibilité d'écraser les paramètres par défaut.
        """
        

        # Générer les marches
        node2vec = Node2Vec(G, 
                            dimensions=dimensions,  
                            walk_length=walk_length,
                            num_walks=num_walks, 
                            p=p, 
                            q=q, 
                            seed=self.seed,
                            workers=self.workers)
        walks = node2vec.walks

        # Entraîner Word2Vec sur les walks
        model = self.train_word2vec_from_walks(
                            walks,
                            dimensions,
                            window, 
                            negative,
                            sg, 
                            epochs
                           )

        return model

    def genererVue_Simba(self, G: nx.Graph, seuil=0.8):
        """
        Génère Vue 1 (topologie PPI) et Vue 2 (graphe filtré par p-value, approche SIMBA).

        Retourne : (Vue1, Vue2)
        """
        ### Construction de Vue 1 : PPI topologique pur
        G_vue1 = nx.Graph()
        G_vue1.add_nodes_from(G.nodes(data=True))  # ajou
        G_vue1.add_edges_from(G.edges())

        ### Construction de Vue 2 : filtrage basé sur p-values
        t = 1 / (2 * seuil)
        seuil_normalise = 1 - np.exp(-t)

        G_vue2 = nx.Graph()
        G_vue2.add_nodes_from(G.nodes(data=True))

        for u, v in G.edges():
            p_u = G.nodes[u].get('weight', None)
            p_v = G.nodes[v].get('weight', None)

            if p_u is None or p_v is None:
                continue
            if p_u + p_v == 0:
                continue

            f = 1 - abs(p_u - p_v) / (p_u + p_v)
            f_norm = 1 - np.exp(-f)

            if f_norm >= seuil_normalise:
                G_vue2.add_edge(u, v, weight=f_norm)
        
        return G_vue1, G_vue2



    def genererVue_Simba2( self,G: nx.Graph, seuil=0.8):
        """
        Connecte toutes les paires de nœuds (u, v) en fonction du seuil de similarité.
        
        Paramètres :
        - G_vue2 : nx.Graph - Graphe filtré existant.
        - G : nx.Graph - Graphe original avec les poids des nœuds.
        - seuil : float - Seuil pour établir une nouvelle connexion.

        Retourne :
        - G_vue2 : nx.Graph - Graphe mis à jour avec les nouvelles connexions.
        """
         ### Construction de Vue 1 : PPI topologique pur
        G_vue1 = nx.Graph()
        G_vue2 = nx.Graph()
        G_vue1.add_nodes_from(G.nodes(data=True))  # 
        G_vue1.add_edges_from(G.edges())
        # Calcul du seuil normalisé
        t = 1 / (2 * seuil)
        seuil_normalise = 1 - np.exp(-t)

        # Parcours de toutes les paires de nœuds
        nodes = list(G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                # Calcul de la similarité pour toutes les paires
                p_u = G.nodes[u].get('weight', None)
                p_v = G.nodes[v].get('weight', None)

                if p_u is None or p_v is None:
                    continue
                if p_u + p_v == 0:
                    continue

                # Calcul de la similarité
                f = 1 - abs(p_u - p_v) / (p_u + p_v)
                f_norm = 1 - np.exp(-f)

                # Si la similarité est suffisante, ajouter l'arête
                if f_norm >= seuil_normalise:
                    G_vue2.add_edge(u, v, weight=f_norm)

        return G_vue1,G_vue2


    #########################
    def extraire_poids_trueHIT(self,G, truehits):
        """
        Retourne un dictionnaire {noeud: poids} pour tous les nœuds présents dans le dictionnaire truehits,
        où les poids sont extraits des attributs des nœuds du graphe G.

        Paramètres :
        - G : nx.Graph, graphe dont les nœuds portent un attribut 'weight'
        - truehits : dict, {clé: ensemble_de_noeuds}

        Retourne :
        - dict {noeud: poids}
        """
        noeud_vers_poids = {}
        for noeuds in truehits.values():
            for n in noeuds:
                poids = G.nodes[n].get('weight', None)
                noeud_vers_poids[n] = poids
        return noeud_vers_poids

    #####################

    def genererVue_Gaussian(self,G, seuil=0.9):
        """
        Construit un graphe basé sur la similarité des poids des nœuds.

        Paramètres:
        G (nx.Graph): Graphe d'origine avec les poids des nœuds.
        sigma (float): Paramètre de l'écart-type pour la fonction gaussienne.
        seuil(float): Seuil de similarité pour créer les arêtes.

        Retourne:
        nx.Graph: Nouveau graphe construit à partir des similarités.
        """
        ### Construction de Vue 1 : PPI topologique pur
        G_vue1 = nx.Graph()
        G_vue1.add_nodes_from(G.nodes(data=True))  # ajou
        G_vue1.add_edges_from(G.edges())

        ### Construction de Vue 2 :similarité Gaussian

        # Extraction des poids des nœuds
        node_weights = {node: data['weight'] for node, data in G.nodes(data=True)}
        nodes = list(G.nodes)
        
        # Calcul de sigma 
        sigma = np.std(list(node_weights.values()))

        # Création du nouveau graphe
        G_vue2 = nx.Graph()

        G_vue2.add_nodes_from(G.nodes(data=True))

        # Calcul de la similarité et ajout des arêtes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                w_i = node_weights[nodes[i]]
                w_j = node_weights[nodes[j]]
                similarity = np.exp(-((w_i - w_j) ** 2) / (2 * sigma ** 2))
                
                if similarity >= seuil:
                    G_vue2.add_edge(nodes[i], nodes[j], weight=similarity)

        return G_vue1,G_vue2
   

    def genererG( self,G: nx.Graph, p_value_seuil=0.05):
        """
        Ajoute à G_vue toutes les arêtes (u,v) existant dans G
        pour lesquelles les deux nœuds ont une p-value < p_value_seuil.

        - G_vue : nx.Graph, le graphe à mettre à jour
        - G     : nx.Graph, le graphe d'origine avec attributs 'weight' (p-value)
        - p_value_seuil : float, seuil pour la p-value (default 0.05)

        Retourne :
        - G_vue : nx.Graph, mis à jour
        """
        G_vue1 = nx.Graph()
        G_vue1.add_nodes_from(G.nodes(data=True))  # ajou
        G_vue1.add_edges_from(G.edges())
        G_vue2=nx.Graph()
        G_vue2.add_nodes_from(G.nodes(data=True))
        for u, v in G.edges():
            p_u = G.nodes[u].get('weight', None)
            p_v = G.nodes[v].get('weight', None)

            if (p_u is not None and p_v is not None 
                    and p_u <= p_value_seuil and p_v <= p_value_seuil):
                if not G_vue2.has_edge(u, v):
                    G_vue2.add_edge(u, v, weight=1.0)  # Poids maximal, modifiable selon besoin

        return G_vue1,G_vue2

    

#######################
    
    def get_most_similar_model(self, model, elt: str, number=20) -> List[int]:   #get_most_similar_model
        """
        Récupère les 'number' nœuds les plus similaires à 'elt' dans un modèle Word2Vec donné.

        Parameters
        ----------
        model : Word2Vec
            Modèle Word2Vec à utiliser.
        elt : str
            Identifiant du nœud.
        number : int
            Nombre de voisins à retourner.

        Returns
        -------
        List[int] : Liste d'identifiants de nœuds similaires.
        """
        try:
            similar = model.wv.similar_by_word(str(elt), topn=number)
            return [(int(x[0]), float(x[1])) for x in similar]
        except KeyError:
            print(f"Le nœud '{elt}' n'existe pas dans le modèle fourni.")
            return []
    

    def get_most_similar(self, elt: str, number: int = 20) -> List[int]:
        """
        Récupère les voisins similaires dans les deux vues ou dans une seule vue.

        Parameters
        ----------
        elt : str
            Le nœud pour lequel on recherche les voisins.
        number : int, optional
            Le nombre de voisins à récupérer par vue.
        top_n : int, optional
            Le nombre de voisins à retourner après fusion et tri.
        view : str, optional
            Spécifie la vue utilisée pour la similarité ('view1', 'view2', 'multi').

        Returns
        -------
        List[int] : Liste des nœuds similaires triés par similarité décroissante.

        """
    
        view='multi' #  view =('view1', 'view2', 'multi')
        voisins1 = dict(self.get_most_similar_model(self.model1, elt,7))
        voisins2 = dict(self.get_most_similar_model(self.model2, elt,7))

        if view == 'multi':
            # Similarité uniquement basée sur la vue 1
            fusionnes = list(voisins1.items())

        elif view == 'view2':
            # Similarité uniquement basée sur la vue 2
            fusionnes = list(voisins2.items())
            # print("liste fusionne type ",fusionnes)

        elif view == 'multi':
            if self.modefusion == "inter":
                communs = set(voisins1.keys()) & set(voisins2.keys())
                fusionnes = [(n, max(voisins1[n], voisins2[n])) for n in communs]

            elif self.modefusion == "union":
                tous_les_noeuds = set(voisins1.keys()) | set(voisins2.keys())
                fusionnes = []

                for n in tous_les_noeuds:
                    score1 = voisins1.get(n, -float('inf'))
                    score2 = voisins2.get(n, -float('inf'))
                    score = max(score1, score2)
                    fusionnes.append((n, score))
            else:
                raise ValueError(f"Mode '{self.modefusion}' invalide. Choisir 'union' ou 'inter'.")
        else:
            raise ValueError(f"Vue '{view}' invalide. Choisir 'view1', 'view2' ou 'multi'.")

        # Tri par score décroissant
        fusionnes_trie = sorted(fusionnes, key=lambda x: x[1], reverse=True)

        # Extraction des `top_n` meilleurs voisins
        resultat = [n for n, _ in fusionnes_trie[:number]]

        return resultat

