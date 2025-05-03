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
        self.view1, self.view2 = self.genererVue_Simba(G)
        print("la vue 1",self.view1)
        print("la vue 2",self.view2)
       
        self.model1 = self.compute_embedding(self.view1,
                                            dimensions=64,
                                            walk_length=100,
                                            num_walks=30,
                                            p=0.15, #0.25
                                            q=2,
                                            window=10,
                                            negative=5, 
                                            sg=1, 
                                            epochs=25)
        
        self.model2 = self.compute_embedding(self.view2,
                                            dimensions=64,
                                            walk_length=100,# 40 ou 50
                                            num_walks=30, #20 ou 30
                                            p=0.25,      #
                                            q=2,         # encourage la coherence locale
                                            window=10,
                                            negative=5, 
                                            sg=1, 
                                            epochs=25)

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

    def genererVue_Simba(self, G: nx.Graph, seuil_pvalue=0.8):
        """
        Génère Vue 1 (topologie PPI) et Vue 2 (graphe filtré par p-value, approche SIMBA).

        Retourne : (Vue1, Vue2)
        """
        ### Construction de Vue 1 : PPI topologique pur
        G_vue1 = nx.Graph()
        G_vue1.add_nodes_from(G.nodes(data=True))  # ajou
        G_vue1.add_edges_from(G.edges())

        ### Construction de Vue 2 : filtrage basé sur p-values
        t = 1 / (2 * seuil_pvalue)
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
    

    #########################

    def genererVue_Simba2(G: nx.Graph, seuil_pvalue=0.05):
        """
        Génère deux versions de Vue 2 à partir du graphe G (avec p-values sur les nœuds) :
        - une version pondérée par similarité des p-values (f_norm)
        - une version uniforme avec les mêmes arêtes mais poids = 1

        Paramètres
        ----------
        G : nx.Graph
            Graphe avec les nœuds possédant l'attribut 'weight' (p-value).
        seuil_pvalue : float
            Seuil biologique utilisé pour filtrer la similarité.

        Retour
        ------
        G_vue2_uniforme : nx.Graph
            Graphe avec arêtes basées sur la similarité des p-values, poids = 1.
        G_vue2_ponderee : nx.Graph
            Graphe avec arêtes basées sur la similarité des p-values, poids = f_norm.
        """
        #initialisation des vues
        G_vue1 = nx.Graph()
        G_vue2_ponderee = nx.Graph()
        G_vue2_uniforme = nx.Graph()
        # vue 1 
        G_vue1.add_nodes_from(G.nodes(data=True))  # ajou
        G_vue1.add_edges_from(G.edges(data=True))  # ajoute les arêtes 

        # definition du seuil 
        t = 1 / (2 * seuil_pvalue)
        seuil_normalise = 1 - np.exp(-t)
        
        G_vue2_ponderee.add_nodes_from(G.nodes(data=True))
        G_vue2_uniforme.add_nodes_from(G.nodes(data=True))

        for u, v in itertools.combinations(G.nodes(), 2):
            p_u = G.nodes[u].get("weight", None)
            p_v = G.nodes[v].get("weight", None)

            if p_u is None or p_v is None:
                continue
            if p_u + p_v == 0:
                continue

            f = 1 - abs(p_u - p_v) / (p_u + p_v)
            f_norm = 1 - np.exp(-f)

            if f_norm >= seuil_normalise:
                G_vue2_ponderee.add_edge(u, v, weight=f_norm)
                G_vue2_uniforme.add_edge(u, v)  # poids implicite = 1

        return G_vue1,G_vue2_uniforme, G_vue2_ponderee



#######################
    
    def get_most_similar_model(self, model, elt: str, number=20) -> List[int]:
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
    def get_most_similar1(self, elt: str, number: int) -> List[int]:
        """
        Récupère les voisins similaires dans les deux vues et les combine par intersection, 
        en gardant la meilleure similarité (max) et trié par ordre décroissant.

        Returns
        -------
        List[int] : Liste des nœuds similaires triés par score décroissant (intersection).
        """
        voisins1 = dict(self.get_most_similar_model(self.model1, elt, 50))
        voisins2 = dict(self.get_most_similar_model(self.model2, elt, 50))

        if self.modefusion == "inter":
            communs = set(voisins1.keys()) & set(voisins2.keys())

            fusionnes = [
                (n, max(voisins1[n], voisins2[n])) for n in communs
            ]

            fusionnes_trie = sorted(fusionnes, key=lambda x: x[1], reverse=True)
            resultat = [n for n, _ in fusionnes_trie[:50]]

            # print("Résultat intersection triée par max :", len(fusionnes_trie))
            # print("Résultat intersection triée par max :", len(fusionnes_trie))
            return resultat

        elif self.modefusion == "union":
            raise NotImplementedError("La fusion par union n'est pas encore codée ici.")

        else:
            raise ValueError("mode doit être 'union' ou 'intersection'.")

        
    def get_most_similar(self, elt: str, number: int) -> List[int]:
        """
        Récupère les voisins similaires dans les deux vues.
        Mode "inter" : intersection avec max des similarités.
        Mode "union" : union avec max des similarités.
        
        Returns
        -------
        List[int] : Liste des nœuds similaires triés par similarité décroissante.
        """
        voisins1 = dict(self.get_most_similar_model(self.model1, elt, 7)) # 25 est bon pour inter 
                                                                              # 10 pour union 
        voisins2 = dict(self.get_most_similar_model(self.model2, elt, 7)) # 25 est bon pou inter 

        if self.modefusion == "inter":
            communs = set(voisins1.keys()) & set(voisins2.keys())
            fusionnes = [
                (n, max(voisins1[n], voisins2[n])) for n in communs
            ]

        elif self.modefusion == "union":
            tous_les_noeuds = set(voisins1.keys()) | set(voisins2.keys())
            fusionnes = []

            for n in tous_les_noeuds:
                score1 = voisins1.get(n, None)
                score2 = voisins2.get(n, None)

                if score1 is not None and score2 is not None:
                    score = max(score1, score2)
                elif score1 is not None:
                    score = score1
                else:
                    score = score2

                fusionnes.append((n, score))

        else:
            raise ValueError("mode doit être 'union' ou 'intersection'.")

        # Trier par score décroissant
        fusionnes_trie = sorted(fusionnes, key=lambda x: x[1], reverse=True)

        # Extraire les n premiers identifiants
        resultat = [n for n, _ in fusionnes_trie[:20]]

        return resultat
