
"""
MathLab AI — Algorithm Explanations
===================================

Explications pédagogiques des principaux algorithmes du module Algorithms.

Le module fournit :
- une description de chaque algorithme ;
- son idée principale ;
- les étapes de fonctionnement ;
- sa complexité temporelle et spatiale ;
- ses avantages ;
- ses limitations ;
- un exemple simple.

Les données sont volontairement indépendantes de l'interface
Streamlit afin de pouvoir être utilisées aussi bien par les tests
que par l'interface utilisateur.
"""

from __future__ import annotations


# ============================================================================
# BASE DE DONNÉES DES EXPLICATIONS
# ============================================================================

_ALGORITHM_EXPLANATIONS = {
    # ========================================================================
    # SEARCHING
    # ========================================================================

    "linear_search": {
        "name": "Linear Search",
        "category": "Searching",
        "description": (
            "Recherche séquentielle d'une valeur en parcourant les éléments "
            "d'une liste un par un."
        ),
        "idea": (
            "Comparer la valeur recherchée avec chaque élément jusqu'à "
            "trouver une correspondance."
        ),
        "steps": [
            "Commencer au premier élément.",
            "Comparer l'élément courant avec la valeur recherchée.",
            "Si les valeurs sont égales, retourner l'indice.",
            "Sinon, passer à l'élément suivant.",
            "Si aucun élément ne correspond, retourner -1.",
        ],
        "complexity": {
            "best": "O(1)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(1)",
        },
        "advantages": [
            "Très simple à comprendre.",
            "Ne nécessite pas que les données soient triées.",
            "Fonctionne avec de nombreux types de données.",
        ],
        "limitations": [
            "Peut être lent sur de grandes listes.",
            "Explore potentiellement tous les éléments.",
        ],
        "example": {
            "input": "[4, 8, 2, 9, 5]",
            "target": "9",
            "result": "Indice 3",
        },
    },

    "binary_search": {
        "name": "Binary Search",
        "category": "Searching",
        "description": (
            "Recherche efficace d'une valeur dans une liste triée en "
            "divisant progressivement l'espace de recherche par deux."
        ),
        "idea": (
            "Comparer la cible avec l'élément central puis éliminer la moitié "
            "de la zone de recherche qui ne peut pas contenir la cible."
        ),
        "steps": [
            "Vérifier que les données sont triées.",
            "Déterminer l'élément au milieu.",
            "Comparer cet élément avec la cible.",
            "Si la cible est plus petite, rechercher dans la moitié gauche.",
            "Si la cible est plus grande, rechercher dans la moitié droite.",
            "Répéter jusqu'à trouver la cible ou épuiser la zone.",
        ],
        "complexity": {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(1)",
        },
        "advantages": [
            "Très rapide sur de grandes listes triées.",
            "Réduit fortement l'espace de recherche à chaque étape.",
        ],
        "limitations": [
            "Nécessite des données triées.",
            "Moins adaptée lorsque les données changent constamment.",
        ],
        "example": {
            "input": "[1, 3, 5, 7, 9, 11]",
            "target": "7",
            "result": "Indice 3",
        },
    },

    "jump_search": {
        "name": "Jump Search",
        "category": "Searching",
        "description": (
            "Recherche dans une liste triée en avançant par blocs avant "
            "d'effectuer une recherche linéaire dans le bloc pertinent."
        ),
        "idea": (
            "Sauter plusieurs éléments à la fois pour localiser le bloc "
            "contenant la cible."
        ),
        "steps": [
            "Choisir une taille de saut proche de √n.",
            "Avancer de bloc en bloc.",
            "S'arrêter lorsque la valeur du bloc dépasse la cible.",
            "Effectuer une recherche linéaire dans le bloc trouvé.",
        ],
        "complexity": {
            "best": "O(1)",
            "average": "O(√n)",
            "worst": "O(√n)",
            "space": "O(1)",
        },
        "advantages": [
            "Plus rapide qu'une recherche linéaire dans certains cas.",
            "Simple à implémenter.",
            "Utilise peu de mémoire supplémentaire.",
        ],
        "limitations": [
            "Nécessite une liste triée.",
            "Moins efficace que Binary Search sur des données accessibles par indice.",
        ],
        "example": {
            "input": "[1, 3, 5, 7, 9, 11, 13, 15, 17]",
            "target": "13",
            "result": "Le bloc contenant 13 est identifié puis parcouru.",
        },
    },

    "interpolation_search": {
        "name": "Interpolation Search",
        "category": "Searching",
        "description": (
            "Recherche dans une liste numérique triée en estimant la position "
            "probable de la cible."
        ),
        "idea": (
            "Contrairement à Binary Search qui examine systématiquement le "
            "milieu, Interpolation Search estime directement une position "
            "à partir des valeurs extrêmes."
        ),
        "steps": [
            "Vérifier que les données sont numériques et triées.",
            "Comparer la cible aux valeurs extrêmes.",
            "Estimer la position de la cible.",
            "Comparer la valeur estimée à la cible.",
            "Réduire la zone de recherche.",
            "Répéter jusqu'à trouver la valeur ou épuiser la zone.",
        ],
        "complexity": {
            "best": "O(1)",
            "average": "O(log log n)",
            "worst": "O(n)",
            "space": "O(1)",
        },
        "advantages": [
            "Très efficace lorsque les données sont uniformément distribuées.",
            "Peut dépasser Binary Search dans certaines distributions.",
        ],
        "limitations": [
            "Nécessite des données numériques triées.",
            "Peut devenir linéaire pour des distributions défavorables.",
        ],
        "example": {
            "input": "[10, 20, 30, 40, 50, 60]",
            "target": "50",
            "result": "La position est estimée directement à partir des valeurs.",
        },
    },

    "exponential_search": {
        "name": "Exponential Search",
        "category": "Searching",
        "description": (
            "Recherche dans une liste triée en trouvant d'abord une borne "
            "grâce à des intervalles qui doublent, puis en utilisant Binary Search."
        ),
        "idea": (
            "Augmenter rapidement la zone de recherche : 1, 2, 4, 8, 16, ..."
        ),
        "steps": [
            "Vérifier le premier élément.",
            "Doubler progressivement l'indice.",
            "S'arrêter lorsque la borne dépasse la cible.",
            "Appliquer Binary Search dans l'intervalle identifié.",
        ],
        "complexity": {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(1)",
        },
        "advantages": [
            "Efficace sur les grandes listes triées.",
            "Utile lorsque la taille exacte de la zone de recherche n'est pas connue.",
        ],
        "limitations": [
            "Nécessite des données triées.",
            "La phase finale utilise Binary Search.",
        ],
        "example": {
            "input": "[2, 4, 6, 8, 10, 12, 14, 16]",
            "target": "12",
            "result": "Une borne est trouvée par doublement puis Binary Search est appliquée.",
        },
    },

    # ========================================================================
    # SORTING
    # ========================================================================

    "bubble_sort": {
        "name": "Bubble Sort",
        "category": "Sorting",
        "description": (
            "Algorithme de tri qui compare des éléments voisins et les échange "
            "lorsqu'ils sont dans le mauvais ordre."
        ),
        "idea": (
            "Les plus grandes valeurs remontent progressivement vers la fin "
            "de la liste, comme des bulles."
        ),
        "steps": [
            "Comparer deux éléments voisins.",
            "Les échanger s'ils sont dans le mauvais ordre.",
            "Continuer jusqu'à la fin de la liste.",
            "Répéter les passages jusqu'à ce qu'aucun échange ne soit nécessaire.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n²)",
            "worst": "O(n²)",
            "space": "O(1)",
        },
        "advantages": [
            "Très simple à comprendre.",
            "Peut être effectué en place.",
        ],
        "limitations": [
            "Très lent sur de grandes listes.",
            "Peu adapté aux applications réelles nécessitant de bonnes performances.",
        ],
        "example": {
            "input": "[5, 2, 4, 1]",
            "result": "[1, 2, 4, 5]",
        },
    },

    "selection_sort": {
        "name": "Selection Sort",
        "category": "Sorting",
        "description": (
            "Algorithme de tri qui cherche le plus petit élément puis le place "
            "à la position correcte."
        ),
        "idea": "Sélectionner successivement le minimum restant.",
        "steps": [
            "Chercher le plus petit élément.",
            "L'échanger avec le premier élément non trié.",
            "Déplacer la frontière entre les parties triée et non triée.",
            "Répéter jusqu'à la fin.",
        ],
        "complexity": {
            "best": "O(n²)",
            "average": "O(n²)",
            "worst": "O(n²)",
            "space": "O(1)",
        },
        "advantages": [
            "Simple à implémenter.",
            "Utilise très peu de mémoire supplémentaire.",
        ],
        "limitations": [
            "Complexité quadratique même lorsque la liste est déjà triée.",
        ],
        "example": {
            "input": "[4, 2, 7, 1]",
            "result": "[1, 2, 4, 7]",
        },
    },

    "insertion_sort": {
        "name": "Insertion Sort",
        "category": "Sorting",
        "description": (
            "Algorithme qui construit progressivement une partie triée en "
            "insérant chaque nouvel élément à sa bonne position."
        ),
        "idea": "Construire la liste triée élément par élément.",
        "steps": [
            "Commencer avec le deuxième élément.",
            "Comparer cet élément aux éléments précédents.",
            "Décaler les éléments plus grands.",
            "Insérer l'élément à la bonne position.",
            "Répéter jusqu'au dernier élément.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n²)",
            "worst": "O(n²)",
            "space": "O(1)",
        },
        "advantages": [
            "Très efficace sur de petites listes.",
            "Excellent lorsque les données sont presque triées.",
        ],
        "limitations": [
            "Peu performant sur de grandes listes désordonnées.",
        ],
        "example": {
            "input": "[5, 2, 4, 1]",
            "result": "[1, 2, 4, 5]",
        },
    },

    "merge_sort": {
        "name": "Merge Sort",
        "category": "Sorting",
        "description": (
            "Algorithme de tri basé sur la stratégie Divide and Conquer."
        ),
        "idea": (
            "Diviser la liste en sous-listes, les trier récursivement puis "
            "fusionner les résultats."
        ),
        "steps": [
            "Diviser la liste en deux parties.",
            "Trier récursivement chaque partie.",
            "Fusionner les deux parties triées.",
            "Continuer jusqu'à obtenir la liste complète triée.",
        ],
        "complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(n)",
        },
        "advantages": [
            "Performance garantie en O(n log n).",
            "Stable pour les implémentations classiques.",
        ],
        "limitations": [
            "Utilise de la mémoire supplémentaire.",
        ],
        "example": {
            "input": "[8, 3, 5, 1]",
            "result": "[1, 3, 5, 8]",
        },
    },

    "quick_sort": {
        "name": "Quick Sort",
        "category": "Sorting",
        "description": (
            "Algorithme de tri qui choisit un pivot et partitionne les données "
            "autour de celui-ci."
        ),
        "idea": "Placer les éléments plus petits à gauche et les plus grands à droite du pivot.",
        "steps": [
            "Choisir un pivot.",
            "Partitionner les éléments autour du pivot.",
            "Trier récursivement la partie gauche.",
            "Trier récursivement la partie droite.",
        ],
        "complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n²)",
            "space": "O(log n)",
        },
        "advantages": [
            "Très performant en pratique.",
            "Bonne localité mémoire.",
        ],
        "limitations": [
            "Le mauvais choix du pivot peut conduire à O(n²).",
        ],
        "example": {
            "input": "[6, 3, 8, 2]",
            "result": "[2, 3, 6, 8]",
        },
    },

    "heap_sort": {
        "name": "Heap Sort",
        "category": "Sorting",
        "description": (
            "Algorithme de tri utilisant une structure de tas binaire."
        ),
        "idea": "Construire un maximum-heap puis extraire progressivement le maximum.",
        "steps": [
            "Construire un heap.",
            "Placer le maximum à la fin.",
            "Réduire la taille du heap.",
            "Réorganiser le heap.",
            "Répéter jusqu'au tri complet.",
        ],
        "complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(1)",
        },
        "advantages": [
            "Complexité garantie en O(n log n).",
            "Peu de mémoire supplémentaire.",
        ],
        "limitations": [
            "Généralement moins rapide que Quick Sort en pratique.",
            "Non stable.",
        ],
        "example": {
            "input": "[4, 10, 3, 5]",
            "result": "[3, 4, 5, 10]",
        },
    },

    "counting_sort": {
        "name": "Counting Sort",
        "category": "Sorting",
        "description": (
            "Algorithme de tri non comparatif qui compte le nombre "
            "d'occurrences de chaque valeur entière."
        ),
        "idea": "Compter les occurrences puis reconstruire la liste triée.",
        "steps": [
            "Identifier les valeurs minimale et maximale.",
            "Créer un tableau de comptage.",
            "Compter les occurrences.",
            "Reconstruire les valeurs dans l'ordre.",
        ],
        "complexity": {
            "best": "O(n + k)",
            "average": "O(n + k)",
            "worst": "O(n + k)",
            "space": "O(n + k)",
        },
        "advantages": [
            "Très rapide lorsque l'intervalle des valeurs est limité.",
            "Ne compare pas directement les éléments.",
        ],
        "limitations": [
            "Adapté aux entiers.",
            "Peut consommer beaucoup de mémoire si la plage des valeurs est grande.",
        ],
        "example": {
            "input": "[4, 2, 2, 8, 3]",
            "result": "[2, 2, 3, 4, 8]",
        },
    },

    "radix_sort": {
        "name": "Radix Sort",
        "category": "Sorting",
        "description": (
            "Algorithme de tri non comparatif qui trie les nombres chiffre "
            "par chiffre."
        ),
        "idea": "Trier successivement selon les unités, dizaines, centaines, etc.",
        "steps": [
            "Identifier le nombre de chiffres maximal.",
            "Trier selon le chiffre des unités.",
            "Trier selon le chiffre des dizaines.",
            "Continuer jusqu'au chiffre le plus significatif.",
        ],
        "complexity": {
            "best": "O(d(n + k))",
            "average": "O(d(n + k))",
            "worst": "O(d(n + k))",
            "space": "O(n + k)",
        },
        "advantages": [
            "Très efficace pour certains ensembles d'entiers.",
            "Peut être linéaire dans certaines configurations.",
        ],
        "limitations": [
            "Principalement adapté aux données numériques.",
            "Les performances dépendent du nombre de chiffres.",
        ],
        "example": {
            "input": "[170, 45, 75, 90]",
            "result": "[45, 75, 90, 170]",
        },
    },

    "bucket_sort": {
        "name": "Bucket Sort",
        "category": "Sorting",
        "description": (
            "Algorithme qui répartit les éléments dans plusieurs compartiments "
            "puis trie chaque compartiment."
        ),
        "idea": "Distribuer les valeurs dans des buckets avant de les trier.",
        "steps": [
            "Créer plusieurs buckets.",
            "Distribuer les éléments dans les buckets.",
            "Trier chaque bucket.",
            "Concaténer les buckets.",
        ],
        "complexity": {
            "best": "O(n + k)",
            "average": "O(n + k)",
            "worst": "O(n²)",
            "space": "O(n + k)",
        },
        "advantages": [
            "Très efficace pour certaines distributions uniformes.",
        ],
        "limitations": [
            "Dépend fortement de la distribution des données.",
            "Nécessite des buckets supplémentaires.",
        ],
        "example": {
            "input": "[0.42, 0.32, 0.73, 0.25]",
            "result": "[0.25, 0.32, 0.42, 0.73]",
        },
    },

    # ========================================================================
    # GRAPH
    # ========================================================================

    "bfs": {
        "name": "Breadth-First Search (BFS)",
        "category": "Graphs",
        "description": (
            "Parcours d'un graphe niveau par niveau à partir d'un sommet donné."
        ),
        "idea": "Explorer d'abord les voisins directs, puis leurs voisins.",
        "steps": [
            "Placer le sommet de départ dans une file.",
            "Le marquer comme visité.",
            "Retirer un sommet de la file.",
            "Ajouter ses voisins non visités.",
            "Répéter jusqu'à ce que la file soit vide.",
        ],
        "complexity": {
            "best": "O(V + E)",
            "average": "O(V + E)",
            "worst": "O(V + E)",
            "space": "O(V)",
        },
        "advantages": [
            "Trouve les plus courts chemins en nombre d'arêtes dans un graphe non pondéré.",
            "Simple et systématique.",
        ],
        "limitations": [
            "Peut consommer beaucoup de mémoire sur des graphes larges.",
        ],
        "example": {
            "input": "A → B, C ; B → D ; C → D",
            "start": "A",
            "result": "A, B, C, D",
        },
    },

    "dfs": {
        "name": "Depth-First Search (DFS)",
        "category": "Graphs",
        "description": (
            "Parcours d'un graphe en explorant aussi profondément que possible "
            "avant de revenir en arrière."
        ),
        "idea": "Explorer une branche jusqu'au bout avant d'en commencer une autre.",
        "steps": [
            "Commencer au sommet de départ.",
            "Marquer le sommet comme visité.",
            "Choisir un voisin non visité.",
            "Continuer récursivement ou avec une pile.",
            "Revenir en arrière lorsqu'une branche est terminée.",
        ],
        "complexity": {
            "best": "O(V + E)",
            "average": "O(V + E)",
            "worst": "O(V + E)",
            "space": "O(V)",
        },
        "advantages": [
            "Utile pour explorer complètement un graphe.",
            "Utilisé dans de nombreux algorithmes de graphes.",
        ],
        "limitations": [
            "Ne garantit pas le plus court chemin dans un graphe non pondéré.",
        ],
        "example": {
            "input": "A → B, C ; B → D",
            "start": "A",
            "result": "A, B, D, C",
        },
    },

    "dijkstra": {
        "name": "Dijkstra",
        "category": "Graphs",
        "description": (
            "Algorithme de calcul des plus courts chemins depuis une source "
            "dans un graphe pondéré à poids non négatifs."
        ),
        "idea": "Choisir progressivement le sommet dont la distance connue est minimale.",
        "steps": [
            "Initialiser la distance de la source à 0.",
            "Initialiser les autres distances à l'infini.",
            "Choisir le sommet non traité avec la plus petite distance.",
            "Relaxer ses arêtes.",
            "Répéter jusqu'à traiter les sommets accessibles.",
        ],
        "complexity": {
            "best": "O((V + E) log V)",
            "average": "O((V + E) log V)",
            "worst": "O((V + E) log V)",
            "space": "O(V)",
        },
        "advantages": [
            "Très efficace avec une file de priorité.",
            "Donne les distances minimales pour les poids non négatifs.",
        ],
        "limitations": [
            "Ne fonctionne pas avec des poids négatifs.",
        ],
        "example": {
            "input": "A → B (4), A → C (2), C → D (1)",
            "start": "A",
            "result": "distance[D] = 3",
        },
    },

    "bellman_ford": {
        "name": "Bellman-Ford",
        "category": "Graphs",
        "description": (
            "Algorithme de plus courts chemins capable de gérer les arêtes "
            "de poids négatif."
        ),
        "idea": "Relaxer toutes les arêtes plusieurs fois.",
        "steps": [
            "Initialiser la source à 0.",
            "Relaxer toutes les arêtes V−1 fois.",
            "Effectuer une passe supplémentaire.",
            "Si une amélioration est encore possible, un cycle négatif existe.",
        ],
        "complexity": {
            "best": "O(VE)",
            "average": "O(VE)",
            "worst": "O(VE)",
            "space": "O(V)",
        },
        "advantages": [
            "Gère les poids négatifs.",
            "Peut détecter les cycles négatifs accessibles.",
        ],
        "limitations": [
            "Plus lent que Dijkstra pour les graphes sans poids négatifs.",
        ],
        "example": {
            "input": "Graphe avec une arête de poids -2",
            "result": "Les distances minimales peuvent être calculées malgré le poids négatif.",
        },
    },

    "floyd_warshall": {
        "name": "Floyd-Warshall",
        "category": "Graphs",
        "description": (
            "Algorithme de programmation dynamique calculant les plus courts "
            "chemins entre toutes les paires de sommets."
        ),
        "idea": "Autoriser progressivement chaque sommet comme intermédiaire.",
        "steps": [
            "Initialiser la matrice des distances.",
            "Choisir un sommet intermédiaire k.",
            "Tester si passer par k améliore chaque distance.",
            "Répéter pour tous les sommets intermédiaires.",
        ],
        "complexity": {
            "best": "O(V³)",
            "average": "O(V³)",
            "worst": "O(V³)",
            "space": "O(V²)",
        },
        "advantages": [
            "Calcule toutes les distances en une seule procédure.",
            "Gère les poids négatifs sans cycle négatif.",
        ],
        "limitations": [
            "Coûteux pour les grands graphes.",
        ],
        "example": {
            "input": "Graphe de 4 sommets",
            "result": "Matrice des distances minimales entre toutes les paires.",
        },
    },

    "kruskal": {
        "name": "Kruskal",
        "category": "Graphs",
        "description": (
            "Algorithme glouton permettant de construire un arbre couvrant "
            "de poids minimal."
        ),
        "idea": "Ajouter les arêtes les moins coûteuses sans créer de cycle.",
        "steps": [
            "Trier les arêtes par poids croissant.",
            "Considérer chaque arête dans cet ordre.",
            "Ajouter l'arête si elle ne crée pas de cycle.",
            "Arrêter lorsque V−1 arêtes ont été sélectionnées.",
        ],
        "complexity": {
            "best": "O(E log E)",
            "average": "O(E log E)",
            "worst": "O(E log E)",
            "space": "O(V + E)",
        },
        "advantages": [
            "Simple avec une structure Union-Find.",
            "Très adapté aux graphes clairsemés.",
        ],
        "limitations": [
            "Suppose un graphe non orienté pour l'arbre couvrant.",
        ],
        "example": {
            "input": "Graphe pondéré connexe",
            "result": "Ensemble d'arêtes formant un MST.",
        },
    },

    "prim": {
        "name": "Prim",
        "category": "Graphs",
        "description": (
            "Algorithme glouton construisant un arbre couvrant minimal "
            "à partir d'un sommet initial."
        ),
        "idea": "Ajouter à chaque étape l'arête la moins coûteuse reliant l'arbre au reste du graphe.",
        "steps": [
            "Choisir un sommet de départ.",
            "Examiner les arêtes sortantes.",
            "Choisir l'arête de poids minimal vers un sommet non inclus.",
            "Ajouter ce sommet à l'arbre.",
            "Répéter jusqu'à inclure tous les sommets.",
        ],
        "complexity": {
            "best": "O(E log V)",
            "average": "O(E log V)",
            "worst": "O(E log V)",
            "space": "O(V + E)",
        },
        "advantages": [
            "Efficace avec une file de priorité.",
            "Construction progressive intuitive.",
        ],
        "limitations": [
            "Destiné aux graphes non orientés connexes pour un MST.",
        ],
        "example": {
            "input": "Graphe pondéré connexe",
            "result": "Un arbre couvrant de poids minimal.",
        },
    },

    # ========================================================================
    # DYNAMIC PROGRAMMING
    # ========================================================================

    "fibonacci": {
        "name": "Fibonacci",
        "category": "Dynamic Programming",
        "description": (
            "Calcul de la suite de Fibonacci en exploitant les résultats "
            "précédemment calculés."
        ),
        "idea": "Éviter de recalculer plusieurs fois les mêmes sous-problèmes.",
        "steps": [
            "Initialiser F(0) et F(1).",
            "Calculer successivement les termes suivants.",
            "Retourner F(n).",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(1)",
        },
        "advantages": [
            "Très simple.",
            "Évite l'explosion exponentielle de la version récursive naïve.",
        ],
        "limitations": [
            "Le nombre de Fibonacci devient très grand rapidement.",
        ],
        "example": {
            "input": "n = 7",
            "result": "13",
        },
    },

    "climbing_stairs": {
        "name": "Climbing Stairs",
        "category": "Dynamic Programming",
        "description": (
            "Calcule le nombre de façons de monter un escalier en avançant "
            "d'une ou deux marches."
        ),
        "idea": "Le nombre de façons d'atteindre n dépend de n−1 et n−2.",
        "steps": [
            "Initialiser les deux premiers résultats.",
            "Calculer chaque état à partir des deux précédents.",
            "Retourner le nombre de façons pour n marches.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(1)",
        },
        "advantages": [
            "Illustration classique de la programmation dynamique.",
        ],
        "limitations": [
            "Le modèle suppose uniquement des pas de 1 ou 2.",
        ],
        "example": {
            "input": "n = 4",
            "result": "5 façons",
        },
    },

    "knapsack_01": {
        "name": "0/1 Knapsack",
        "category": "Dynamic Programming",
        "description": (
            "Optimise la valeur d'un ensemble d'objets sous une contrainte "
            "de capacité, chaque objet pouvant être choisi au maximum une fois."
        ),
        "idea": "Pour chaque objet, choisir entre le prendre ou le laisser.",
        "steps": [
            "Parcourir les objets.",
            "Pour chaque capacité, comparer les deux choix.",
            "Conserver la meilleure valeur.",
            "Retourner la meilleure valeur obtenue.",
        ],
        "complexity": {
            "best": "O(nC)",
            "average": "O(nC)",
            "worst": "O(nC)",
            "space": "O(C)",
        },
        "advantages": [
            "Résout exactement de nombreux problèmes de sélection sous contrainte.",
        ],
        "limitations": [
            "La complexité dépend fortement de la capacité C.",
        ],
        "example": {
            "input": "weights=[2,3,4], values=[3,4,5], capacity=5",
            "result": "Valeur optimale = 7",
        },
    },

    "coin_change": {
        "name": "Coin Change",
        "category": "Dynamic Programming",
        "description": (
            "Détermine le nombre minimal de pièces nécessaires pour atteindre "
            "un montant donné."
        ),
        "idea": "Construire progressivement la solution optimale pour chaque montant.",
        "steps": [
            "Initialiser le montant 0 à 0 pièce.",
            "Calculer le minimum pour chaque montant.",
            "Tester chaque pièce disponible.",
            "Retourner le minimum pour le montant demandé.",
        ],
        "complexity": {
            "best": "O(nA)",
            "average": "O(nA)",
            "worst": "O(nA)",
            "space": "O(A)",
        },
        "advantages": [
            "Trouve une solution optimale lorsque le problème est bien défini.",
        ],
        "limitations": [
            "Peut être coûteux pour de très grands montants.",
        ],
        "example": {
            "input": "coins=[1,2,5], amount=11",
            "result": "3 pièces",
        },
    },

    "longest_common_subsequence": {
        "name": "Longest Common Subsequence",
        "category": "Dynamic Programming",
        "description": (
            "Recherche la plus longue sous-séquence commune à deux séquences."
        ),
        "idea": "Construire une table des meilleures solutions pour les préfixes.",
        "steps": [
            "Créer une table pour les deux séquences.",
            "Comparer les éléments courants.",
            "Si les éléments correspondent, prolonger la solution.",
            "Sinon, conserver la meilleure solution précédente.",
            "Reconstruire une sous-séquence valide.",
        ],
        "complexity": {
            "best": "O(nm)",
            "average": "O(nm)",
            "worst": "O(nm)",
            "space": "O(nm)",
        },
        "advantages": [
            "Méthode classique pour comparer des séquences.",
        ],
        "limitations": [
            "Peut utiliser beaucoup de mémoire pour de longues séquences.",
        ],
        "example": {
            "input": "\"ABCBDAB\", \"BDCABA\"",
            "result": "Une LCS valide de longueur 4",
        },
    },

    "longest_increasing_subsequence": {
        "name": "Longest Increasing Subsequence",
        "category": "Dynamic Programming",
        "description": (
            "Recherche une plus longue sous-séquence dont les éléments "
            "sont strictement croissants."
        ),
        "idea": "Conserver la meilleure sous-séquence se terminant à chaque position.",
        "steps": [
            "Examiner chaque élément.",
            "Chercher les sous-séquences précédentes compatibles.",
            "Conserver la meilleure longueur.",
            "Reconstruire une sous-séquence optimale.",
        ],
        "complexity": {
            "best": "O(n²)",
            "average": "O(n²)",
            "worst": "O(n²)",
            "space": "O(n)",
        },
        "advantages": [
            "Méthode simple et pédagogique.",
        ],
        "limitations": [
            "Une implémentation plus avancée peut atteindre O(n log n).",
        ],
        "example": {
            "input": "[10, 9, 2, 5, 3, 7, 101, 18]",
            "result": "[2, 3, 7, 101] ou autre LIS valide",
        },
    },

    "matrix_chain_multiplication": {
        "name": "Matrix Chain Multiplication",
        "category": "Dynamic Programming",
        "description": (
            "Détermine le meilleur ordre de multiplication d'une chaîne "
            "de matrices afin de minimiser le nombre d'opérations."
        ),
        "idea": "Tester les différents points de séparation et conserver le coût minimal.",
        "steps": [
            "Considérer les chaînes de longueur croissante.",
            "Tester chaque position de séparation.",
            "Calculer le coût des deux sous-chaînes.",
            "Conserver le coût minimal.",
        ],
        "complexity": {
            "best": "O(n³)",
            "average": "O(n³)",
            "worst": "O(n³)",
            "space": "O(n²)",
        },
        "advantages": [
            "Évite d'essayer naïvement toutes les parenthésisations.",
        ],
        "limitations": [
            "Le nombre de matrices peut rendre le calcul important.",
        ],
        "example": {
            "input": "dimensions = [10, 20, 30]",
            "result": "Coût minimal = 6000",
        },
    },

    # ========================================================================
    # GREEDY
    # ========================================================================

    "activity_selection": {
        "name": "Activity Selection",
        "category": "Greedy",
        "description": (
            "Sélectionne le maximum d'activités compatibles dans un ensemble "
            "d'activités ayant des heures de début et de fin."
        ),
        "idea": "Choisir l'activité qui se termine le plus tôt.",
        "steps": [
            "Trier les activités par heure de fin.",
            "Choisir la première activité.",
            "Ignorer les activités incompatibles.",
            "Choisir la prochaine activité compatible.",
            "Répéter.",
        ],
        "complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(n)",
        },
        "advantages": [
            "Simple et efficace.",
            "La stratégie gloutonne est optimale pour ce problème.",
        ],
        "limitations": [
            "La stratégie dépend de la structure particulière du problème.",
        ],
        "example": {
            "input": "[(1,2), (2,4), (3,5), (5,7)]",
            "result": "Une sélection maximale d'activités compatibles",
        },
    },

    "fractional_knapsack": {
        "name": "Fractional Knapsack",
        "category": "Greedy",
        "description": (
            "Maximise la valeur d'un sac lorsque les objets peuvent être "
            "fractionnés."
        ),
        "idea": "Prendre d'abord les objets ayant la plus grande valeur par unité de poids.",
        "steps": [
            "Calculer le ratio valeur/poids.",
            "Trier les objets selon ce ratio.",
            "Prendre les objets dans cet ordre.",
            "Prendre éventuellement une fraction du dernier objet.",
        ],
        "complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(n)",
        },
        "advantages": [
            "Stratégie gloutonne optimale pour la version fractionnaire.",
        ],
        "limitations": [
            "Ne s'applique pas directement à la version 0/1.",
        ],
        "example": {
            "input": "weights=[10,20], values=[60,100], capacity=15",
            "result": "Valeur maximale = 85",
        },
    },

    "greedy_coin_change": {
        "name": "Greedy Coin Change",
        "category": "Greedy",
        "description": (
            "Construit une solution de rendu de monnaie en utilisant "
            "toujours la plus grande pièce possible."
        ),
        "idea": "Prendre la plus grande pièce disponible qui ne dépasse pas le montant restant.",
        "steps": [
            "Trier les pièces par ordre décroissant.",
            "Prendre autant que possible de la plus grande pièce.",
            "Passer à la suivante.",
            "Continuer jusqu'à atteindre le montant ou ne plus pouvoir avancer.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(1)",
        },
        "advantages": [
            "Très simple et rapide.",
        ],
        "limitations": [
            "Ne garantit pas toujours le nombre minimal de pièces.",
        ],
        "example": {
            "input": "coins=[1,5,10,25], amount=41",
            "result": "[25, 10, 5, 1]",
        },
    },

    "huffman_coding": {
        "name": "Huffman Coding",
        "category": "Greedy",
        "description": (
            "Construit un codage préfixe optimal à partir des fréquences "
            "des symboles."
        ),
        "idea": "Fusionner progressivement les deux symboles les moins fréquents.",
        "steps": [
            "Créer un nœud pour chaque symbole.",
            "Sélectionner les deux fréquences minimales.",
            "Les fusionner.",
            "Réinsérer le nouveau nœud.",
            "Répéter jusqu'à obtenir un arbre.",
            "Attribuer 0 et 1 aux branches.",
        ],
        "complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(n)",
        },
        "advantages": [
            "Produit un codage préfixe efficace.",
            "Utilisé dans la compression de données.",
        ],
        "limitations": [
            "Nécessite les fréquences des symboles.",
        ],
        "example": {
            "input": "{A:5, B:9, C:12, D:13}",
            "result": "Un ensemble de codes binaires préfixes",
        },
    },

    "job_sequencing": {
        "name": "Job Sequencing",
        "category": "Greedy",
        "description": (
            "Sélectionne des tâches avec deadlines et profits afin de "
            "maximiser le profit total."
        ),
        "idea": "Choisir d'abord les tâches les plus profitables et les placer le plus tard possible.",
        "steps": [
            "Trier les tâches par profit décroissant.",
            "Chercher le dernier créneau disponible avant la deadline.",
            "Planifier la tâche si un créneau existe.",
            "Continuer avec les tâches suivantes.",
        ],
        "complexity": {
            "best": "O(n²)",
            "average": "O(n²)",
            "worst": "O(n²)",
            "space": "O(n)",
        },
        "advantages": [
            "Simple pour les problèmes classiques de job sequencing.",
        ],
        "limitations": [
            "Le modèle suppose une durée d'une unité par tâche.",
        ],
        "example": {
            "input": "jobs=[(A,2,100),(B,1,19),(C,2,27)]",
            "result": "Planification des tâches maximisant le profit",
        },
    },

    "interval_scheduling": {
        "name": "Interval Scheduling",
        "category": "Greedy",
        "description": (
            "Sélectionne un ensemble maximal d'intervalles non chevauchants."
        ),
        "idea": "Choisir successivement les intervalles qui se terminent le plus tôt.",
        "steps": [
            "Trier les intervalles par fin.",
            "Sélectionner le premier intervalle.",
            "Rejeter ceux qui chevauchent le dernier sélectionné.",
            "Continuer jusqu'à la fin.",
        ],
        "complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(n)",
        },
        "advantages": [
            "Méthode gloutonne simple.",
        ],
        "limitations": [
            "Optimise le nombre d'intervalles, pas nécessairement une autre mesure.",
        ],
        "example": {
            "input": "[(1,3), (2,4), (3,5), (5,7)]",
            "result": "Une sélection maximale sans chevauchement",
        },
    },

    # ========================================================================
    # BACKTRACKING
    # ========================================================================

    "n_queens": {
        "name": "N-Queens",
        "category": "Backtracking",
        "description": (
            "Place n reines sur un échiquier n×n sans qu'aucune reine "
            "ne puisse attaquer une autre."
        ),
        "idea": "Construire une solution ligne par ligne et revenir en arrière lorsqu'un conflit apparaît.",
        "steps": [
            "Choisir une ligne.",
            "Tester chaque colonne.",
            "Vérifier les conflits.",
            "Placer la reine si la position est valide.",
            "Continuer récursivement.",
            "Revenir en arrière si aucune position ne fonctionne.",
        ],
        "complexity": {
            "best": "O(n!)",
            "average": "O(n!)",
            "worst": "O(n!)",
            "space": "O(n)",
        },
        "advantages": [
            "Excellent exemple de recherche avec retour arrière.",
        ],
        "limitations": [
            "La complexité augmente rapidement avec n.",
        ],
        "example": {
            "input": "n = 4",
            "result": "[1, 3, 0, 2] est une solution valide",
        },
    },

    "solve_sudoku": {
        "name": "Sudoku Solver",
        "category": "Backtracking",
        "description": (
            "Résout une grille de Sudoku en essayant les valeurs possibles "
            "et en revenant en arrière en cas de contradiction."
        ),
        "idea": "Tester une valeur, continuer, puis annuler le choix si nécessaire.",
        "steps": [
            "Trouver une case vide.",
            "Tester les chiffres possibles.",
            "Vérifier ligne, colonne et bloc.",
            "Placer une valeur valide.",
            "Continuer récursivement.",
            "Annuler le choix si aucune solution n'est possible.",
        ],
        "complexity": {
            "best": "Variable",
            "average": "Exponentielle",
            "worst": "O(9^81)",
            "space": "O(81)",
        },
        "advantages": [
            "Méthode générale et intuitive.",
        ],
        "limitations": [
            "Peut devenir coûteuse pour certaines grilles.",
        ],
        "example": {
            "input": "Grille 9×9 partiellement remplie",
            "result": "Grille Sudoku complétée",
        },
    },

    "solve_maze": {
        "name": "Maze Solver",
        "category": "Backtracking",
        "description": (
            "Recherche un chemin dans un labyrinthe en explorant "
            "les directions possibles."
        ),
        "idea": "Avancer, marquer le chemin, puis revenir en arrière si nécessaire.",
        "steps": [
            "Commencer à l'entrée.",
            "Tester les directions possibles.",
            "Éviter les murs et les cases déjà visitées.",
            "Continuer vers la sortie.",
            "Revenir en arrière lorsqu'une voie est bloquée.",
        ],
        "complexity": {
            "best": "O(V)",
            "average": "O(V)",
            "worst": "O(V)",
            "space": "O(V)",
        },
        "advantages": [
            "Illustration claire du backtracking.",
        ],
        "limitations": [
            "Peut explorer de nombreux chemins inutiles.",
        ],
        "example": {
            "input": "Matrice binaire avec entrée et sortie",
            "result": "Liste des coordonnées du chemin",
        },
    },

    "subsets": {
        "name": "Subsets",
        "category": "Backtracking",
        "description": "Génère tous les sous-ensembles d'une séquence.",
        "idea": "Pour chaque élément, choisir de l'inclure ou non.",
        "steps": [
            "Commencer avec un sous-ensemble vide.",
            "Pour chaque élément, explorer les deux choix.",
            "Inclure l'élément.",
            "Ne pas l'inclure.",
            "Continuer jusqu'à la fin.",
        ],
        "complexity": {
            "best": "O(2ⁿ)",
            "average": "O(2ⁿ)",
            "worst": "O(2ⁿ)",
            "space": "O(2ⁿ)",
        },
        "advantages": [
            "Simple représentation des choix binaires.",
        ],
        "limitations": [
            "Le nombre de résultats est exponentiel.",
        ],
        "example": {
            "input": "[1, 2]",
            "result": "[[], [1], [2], [1, 2]]",
        },
    },

    "permutations": {
        "name": "Permutations",
        "category": "Backtracking",
        "description": "Génère les permutations uniques d'une séquence.",
        "idea": "Choisir successivement quel élément occupe chaque position.",
        "steps": [
            "Choisir un élément.",
            "Le placer dans la position courante.",
            "Générer récursivement les positions restantes.",
            "Annuler le choix.",
            "Continuer avec un autre élément.",
        ],
        "complexity": {
            "best": "O(n!)",
            "average": "O(n!)",
            "worst": "O(n!)",
            "space": "O(n!)",
        },
        "advantages": [
            "Permet d'explorer systématiquement toutes les configurations.",
        ],
        "limitations": [
            "Le nombre de permutations augmente très rapidement.",
        ],
        "example": {
            "input": "[1, 2, 3]",
            "result": "6 permutations uniques",
        },
    },

    "combination_sum": {
        "name": "Combination Sum",
        "category": "Backtracking",
        "description": (
            "Recherche les combinaisons de candidats dont la somme atteint "
            "une cible, avec réutilisation possible des candidats."
        ),
        "idea": "Explorer les choix possibles et abandonner les branches qui dépassent la cible.",
        "steps": [
            "Choisir un candidat.",
            "Réduire la cible restante.",
            "Continuer avec le même candidat ou les suivants.",
            "Ajouter la combinaison lorsqu'elle atteint la cible.",
            "Revenir en arrière pour tester d'autres choix.",
        ],
        "complexity": {
            "best": "Variable",
            "average": "Exponentielle",
            "worst": "Exponentielle",
            "space": "Exponentielle",
        },
        "advantages": [
            "Permet d'explorer efficacement un espace de combinaisons.",
        ],
        "limitations": [
            "Peut générer un grand nombre de solutions.",
        ],
        "example": {
            "input": "candidates=[2,3,6,7], target=7",
            "result": "[[2,2,3], [7]]",
        },
    },

    # ========================================================================
    # TREES
    # ========================================================================

    "preorder_traversal": {
        "name": "Preorder Traversal",
        "category": "Trees",
        "description": "Parcourt un arbre selon l'ordre racine, gauche, droite.",
        "idea": "Visiter la racine avant ses sous-arbres.",
        "steps": [
            "Visiter la racine.",
            "Parcourir le sous-arbre gauche.",
            "Parcourir le sous-arbre droit.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(n)",
        },
        "advantages": [
            "Utile pour représenter ou copier la structure d'un arbre.",
        ],
        "limitations": [
            "L'ordre obtenu dépend de la structure de l'arbre.",
        ],
        "example": {
            "input": "      1\n     / \\\n    2   3",
            "result": "[1, 2, 3]",
        },
    },

    "inorder_traversal": {
        "name": "Inorder Traversal",
        "category": "Trees",
        "description": "Parcourt un arbre selon l'ordre gauche, racine, droite.",
        "idea": "Visiter la racine entre ses deux sous-arbres.",
        "steps": [
            "Parcourir le sous-arbre gauche.",
            "Visiter la racine.",
            "Parcourir le sous-arbre droit.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(n)",
        },
        "advantages": [
            "Sur un BST, produit les valeurs dans l'ordre croissant.",
        ],
        "limitations": [
            "Nécessite de connaître la structure de l'arbre pour interpréter le résultat.",
        ],
        "example": {
            "input": "      2\n     / \\\n    1   3",
            "result": "[1, 2, 3]",
        },
    },

    "postorder_traversal": {
        "name": "Postorder Traversal",
        "category": "Trees",
        "description": "Parcourt un arbre selon l'ordre gauche, droite, racine.",
        "idea": "Visiter les enfants avant la racine.",
        "steps": [
            "Parcourir le sous-arbre gauche.",
            "Parcourir le sous-arbre droit.",
            "Visiter la racine.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(n)",
        },
        "advantages": [
            "Utile pour supprimer ou évaluer certains arbres.",
        ],
        "limitations": [
            "Le résultat n'est pas naturellement trié pour un BST.",
        ],
        "example": {
            "input": "      1\n     / \\\n    2   3",
            "result": "[2, 3, 1]",
        },
    },

    "level_order_traversal": {
        "name": "Level Order Traversal",
        "category": "Trees",
        "description": "Parcourt un arbre niveau par niveau.",
        "idea": "Utiliser une file pour traiter les nœuds dans leur ordre de profondeur.",
        "steps": [
            "Ajouter la racine à une file.",
            "Retirer un nœud.",
            "Ajouter ses enfants.",
            "Répéter jusqu'à ce que la file soit vide.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(n)",
        },
        "advantages": [
            "Parcours naturel pour les niveaux d'un arbre.",
        ],
        "limitations": [
            "Nécessite une file potentiellement importante.",
        ],
        "example": {
            "input": "      1\n     / \\\n    2   3",
            "result": "[[1], [2, 3]]",
        },
    },

    "bst_search": {
        "name": "BST Search",
        "category": "Trees",
        "description": (
            "Recherche une valeur dans un arbre binaire de recherche "
            "en exploitant son ordre."
        ),
        "idea": "Éliminer une moitié de l'arbre à chaque comparaison.",
        "steps": [
            "Comparer la cible à la racine.",
            "Si elle est plus petite, aller à gauche.",
            "Si elle est plus grande, aller à droite.",
            "Continuer jusqu'à trouver la valeur ou atteindre None.",
        ],
        "complexity": {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(n)",
            "space": "O(1)",
        },
        "advantages": [
            "Très efficace avec un arbre équilibré.",
        ],
        "limitations": [
            "Peut devenir linéaire si l'arbre est fortement déséquilibré.",
        ],
        "example": {
            "input": "BST contenant [2, 4, 6, 8]",
            "target": "6",
            "result": "Valeur trouvée",
        },
    },

    "bst_insert": {
        "name": "BST Insert",
        "category": "Trees",
        "description": "Insère une valeur dans un arbre binaire de recherche.",
        "idea": "Suivre les comparaisons jusqu'à trouver la position correcte.",
        "steps": [
            "Comparer la nouvelle valeur à la racine.",
            "Aller à gauche si elle est plus petite.",
            "Aller à droite si elle est plus grande.",
            "Créer le nouveau nœud à la position vide.",
        ],
        "complexity": {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(n)",
            "space": "O(n)",
        },
        "advantages": [
            "Insertion simple.",
        ],
        "limitations": [
            "Peut dégrader la structure si l'arbre n'est pas équilibré.",
        ],
        "example": {
            "input": "BST [2, 4, 6], valeur 5",
            "result": "5 est inséré à gauche de 6.",
        },
    },

    "bst_delete": {
        "name": "BST Delete",
        "category": "Trees",
        "description": "Supprime une valeur d'un arbre binaire de recherche.",
        "idea": "Traiter les cas feuille, un enfant ou deux enfants.",
        "steps": [
            "Rechercher le nœud.",
            "Si le nœud est une feuille, le supprimer.",
            "S'il possède un enfant, remplacer le nœud par cet enfant.",
            "Avec deux enfants, utiliser le successeur inorder.",
        ],
        "complexity": {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(n)",
            "space": "O(n)",
        },
        "advantages": [
            "Permet de maintenir la propriété du BST.",
        ],
        "limitations": [
            "Plus complexe que l'insertion.",
            "Dépend de l'équilibre de l'arbre.",
        ],
        "example": {
            "input": "BST [2, 4, 6], supprimer 4",
            "result": "Le BST est réorganisé en conservant son ordre.",
        },
    },

    "min_heap": {
        "name": "Min Heap",
        "category": "Trees",
        "description": (
            "Structure arborescente dans laquelle chaque parent est inférieur "
            "ou égal à ses enfants."
        ),
        "idea": "Conserver constamment le plus petit élément à la racine.",
        "steps": [
            "Insérer un élément.",
            "Le faire remonter si nécessaire.",
            "Lors d'une extraction, déplacer le dernier élément à la racine.",
            "Le faire descendre jusqu'à restaurer la propriété du heap.",
        ],
        "complexity": {
            "best": "O(log n)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(n)",
        },
        "advantages": [
            "Accès rapide au minimum.",
            "Base des files de priorité.",
        ],
        "limitations": [
            "L'accès à un élément arbitraire n'est pas rapide.",
        ],
        "example": {
            "input": "[5, 3, 8, 1]",
            "result": "Le minimum accessible est 1.",
        },
    },

    "priority_queue": {
        "name": "Priority Queue",
        "category": "Trees",
        "description": (
            "Structure de données qui traite les éléments selon leur priorité."
        ),
        "idea": "Toujours extraire l'élément possédant la priorité la plus élevée "
        "ou la plus faible selon la convention choisie."
        ,
        "steps": [
            "Insérer un élément avec sa priorité.",
            "Maintenir la structure du heap.",
            "Consulter ou retirer l'élément prioritaire.",
            "Répéter selon les besoins.",
        ],
        "complexity": {
            "best": "O(log n)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(n)",
        },
        "advantages": [
            "Insertion et extraction efficaces.",
            "Très utile dans les algorithmes de graphes.",
        ],
        "limitations": [
            "Les éléments ne sont pas totalement triés en permanence.",
        ],
        "example": {
            "input": "push(A, priority=2), push(B, priority=1)",
            "result": "B est extrait en premier.",
        },
    },

    # ========================================================================
    # MATHEMATICAL
    # ========================================================================

    "gcd": {
        "name": "Euclidean Algorithm (GCD)",
        "category": "Mathematical",
        "description": (
            "Calcule le plus grand commun diviseur de deux entiers."
        ),
        "idea": "Utiliser la propriété gcd(a,b) = gcd(b, a mod b).",
        "steps": [
            "Calculer le reste de la division de a par b.",
            "Remplacer a par b.",
            "Remplacer b par le reste.",
            "Répéter jusqu'à obtenir un reste nul.",
            "Le dernier diviseur non nul est le PGCD.",
        ],
        "complexity": {
            "best": "O(log n)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(1)",
        },
        "advantages": [
            "Très efficace.",
            "Algorithme fondamental en arithmétique.",
        ],
        "limitations": [
            "Concerne principalement les entiers.",
        ],
        "example": {
            "input": "gcd(48, 18)",
            "result": "6",
        },
    },

    "extended_gcd": {
        "name": "Extended Euclidean Algorithm",
        "category": "Mathematical",
        "description": (
            "Calcule le PGCD ainsi que les coefficients de Bézout."
        ),
        "idea": "Étendre l'algorithme d'Euclide pour obtenir x et y tels que ax + by = gcd(a,b).",
        "steps": [
            "Appliquer les divisions euclidiennes.",
            "Remonter les équations.",
            "Calculer les coefficients de Bézout.",
            "Retourner g, x et y.",
        ],
        "complexity": {
            "best": "O(log n)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(1)",
        },
        "advantages": [
            "Utile pour les inverses modulaires.",
            "Produit une identité de Bézout.",
        ],
        "limitations": [
            "Nécessite une compréhension de l'arithmétique euclidienne.",
        ],
        "example": {
            "input": "a=30, b=18",
            "result": "gcd=6 et 30x + 18y = 6",
        },
    },

    "generate_primes": {
        "name": "Prime Generation",
        "category": "Mathematical",
        "description": "Génère les nombres premiers jusqu'à une limite donnée.",
        "idea": "Tester ou construire progressivement les nombres premiers.",
        "steps": [
            "Parcourir les nombres candidats.",
            "Tester leur divisibilité.",
            "Ajouter les nombres premiers.",
        ],
        "complexity": {
            "best": "Variable",
            "average": "Variable",
            "worst": "O(n²)",
            "space": "O(n)",
        },
        "advantages": [
            "Simple pour générer de petites listes de nombres premiers.",
        ],
        "limitations": [
            "Des méthodes spécialisées sont plus efficaces pour de grandes limites.",
        ],
        "example": {
            "input": "limit = 10",
            "result": "[2, 3, 5, 7]",
        },
    },

    "sieve_of_eratosthenes": {
        "name": "Sieve of Eratosthenes",
        "category": "Mathematical",
        "description": (
            "Génère efficacement tous les nombres premiers jusqu'à une limite "
            "en éliminant les multiples des nombres premiers."
        ),
        "idea": "Marquer progressivement les multiples comme non premiers.",
        "steps": [
            "Créer une table indiquant les candidats.",
            "Commencer avec 2.",
            "Éliminer ses multiples.",
            "Passer au prochain nombre non éliminé.",
            "Répéter jusqu'à √n.",
        ],
        "complexity": {
            "best": "O(n log log n)",
            "average": "O(n log log n)",
            "worst": "O(n log log n)",
            "space": "O(n)",
        },
        "advantages": [
            "Très efficace pour générer de nombreux nombres premiers.",
        ],
        "limitations": [
            "Utilise un tableau proportionnel à la limite.",
        ],
        "example": {
            "input": "limit = 20",
            "result": "[2, 3, 5, 7, 11, 13, 17, 19]",
        },
    },

    "fast_power": {
        "name": "Fast Exponentiation",
        "category": "Mathematical",
        "description": (
            "Calcule rapidement une puissance en utilisant la décomposition "
            "binaire de l'exposant."
        ),
        "idea": "Réduire le nombre de multiplications grâce au carré successif.",
        "steps": [
            "Examiner l'exposant en base 2.",
            "Multiplier le résultat lorsque le bit courant vaut 1.",
            "Élever la base au carré.",
            "Diviser l'exposant par 2.",
            "Répéter jusqu'à l'exposant nul.",
        ],
        "complexity": {
            "best": "O(log n)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(1)",
        },
        "advantages": [
            "Beaucoup plus rapide que n multiplications successives.",
        ],
        "limitations": [
            "La taille des nombres peut devenir très importante.",
        ],
        "example": {
            "input": "2^10",
            "result": "1024",
        },
    },

    "modular_power": {
        "name": "Modular Exponentiation",
        "category": "Mathematical",
        "description": (
            "Calcule efficacement base^exposant modulo un entier."
        ),
        "idea": "Combiner l'exponentiation rapide avec les propriétés du modulo.",
        "steps": [
            "Initialiser le résultat à 1.",
            "Examiner l'exposant en binaire.",
            "Multiplier modulo m lorsque nécessaire.",
            "Élever la base au carré modulo m.",
            "Continuer jusqu'à l'exposant nul.",
        ],
        "complexity": {
            "best": "O(log n)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(1)",
        },
        "advantages": [
            "Évite de construire directement de très grands nombres.",
            "Fondamental en cryptographie.",
        ],
        "limitations": [
            "Le module doit être positif dans cette implémentation.",
        ],
        "example": {
            "input": "2^10 mod 1000",
            "result": "24",
        },
    },

    "factorial": {
        "name": "Factorial",
        "category": "Mathematical",
        "description": "Calcule n! = 1 × 2 × ... × n.",
        "idea": "Multiplier successivement les entiers jusqu'à n.",
        "steps": [
            "Initialiser le résultat à 1.",
            "Multiplier par chaque entier de 1 à n.",
            "Retourner le résultat.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(1)",
        },
        "advantages": [
            "Simple et fondamental en combinatoire.",
        ],
        "limitations": [
            "La valeur de n! augmente extrêmement rapidement.",
        ],
        "example": {
            "input": "5!",
            "result": "120",
        },
    },

    "pascal_triangle": {
        "name": "Pascal Triangle",
        "category": "Mathematical",
        "description": (
            "Construit le triangle de Pascal à partir des sommes des valeurs "
            "adjacentes de la ligne précédente."
        ),
        "idea": "Chaque valeur intérieure est la somme des deux valeurs au-dessus.",
        "steps": [
            "Commencer par [1].",
            "Créer chaque nouvelle ligne.",
            "Placer 1 aux extrémités.",
            "Calculer les valeurs intérieures par addition.",
        ],
        "complexity": {
            "best": "O(n²)",
            "average": "O(n²)",
            "worst": "O(n²)",
            "space": "O(n²)",
        },
        "advantages": [
            "Permet de visualiser les coefficients binomiaux.",
        ],
        "limitations": [
            "Le nombre de valeurs augmente quadratiquement avec le nombre de lignes.",
        ],
        "example": {
            "input": "rows = 4",
            "result": "[[1], [1,1], [1,2,1], [1,3,3,1]]",
        },
    },

    # ========================================================================
    # STRINGS
    # ========================================================================

    "naive_search": {
        "name": "Naive String Search",
        "category": "Strings",
        "description": (
            "Recherche un motif dans une chaîne en testant chaque position possible."
        ),
        "idea": "Comparer le motif avec la sous-chaîne commençant à chaque position.",
        "steps": [
            "Commencer à la première position.",
            "Comparer les caractères du motif.",
            "Si une différence apparaît, déplacer le motif d'une position.",
            "Répéter jusqu'à trouver une occurrence ou atteindre la fin.",
        ],
        "complexity": {
            "best": "O(n)",
            "average": "O(nm)",
            "worst": "O(nm)",
            "space": "O(1)",
        },
        "advantages": [
            "Très simple.",
            "Ne nécessite aucune préparation.",
        ],
        "limitations": [
            "Peut être lent pour de longues chaînes.",
        ],
        "example": {
            "input": "text='hello world', pattern='world'",
            "result": "Indice 6",
        },
    },

    "kmp_search": {
        "name": "Knuth-Morris-Pratt (KMP)",
        "category": "Strings",
        "description": (
            "Algorithme de recherche de motif utilisant un tableau de préfixes "
            "pour éviter les comparaisons inutiles."
        ),
        "idea": "Réutiliser l'information déjà obtenue lors des comparaisons précédentes.",
        "steps": [
            "Construire le tableau LPS du motif.",
            "Parcourir le texte.",
            "Comparer les caractères.",
            "En cas d'échec, utiliser le tableau LPS pour éviter de repartir de zéro.",
            "Continuer jusqu'à trouver le motif.",
        ],
        "complexity": {
            "best": "O(n + m)",
            "average": "O(n + m)",
            "worst": "O(n + m)",
            "space": "O(m)",
        },
        "advantages": [
            "Complexité linéaire garantie.",
            "Évite de refaire certaines comparaisons.",
        ],
        "limitations": [
            "Plus complexe que la recherche naïve.",
            "Nécessite de construire le tableau LPS.",
        ],
        "example": {
            "input": "text='ABABDABACDABABCABAB', pattern='ABABCABAB'",
            "result": "Indice 10",
        },
    },

    "rabin_karp_search": {
        "name": "Rabin-Karp",
        "category": "Strings",
        "description": (
            "Recherche un motif en comparant des valeurs de hachage avant "
            "de comparer éventuellement les caractères."
        ),
        "idea": "Utiliser un hash glissant pour éviter de comparer systématiquement les chaînes.",
        "steps": [
            "Calculer le hash du motif.",
            "Calculer le hash de la première fenêtre du texte.",
            "Comparer les hashes.",
            "Vérifier les caractères si les hashes correspondent.",
            "Mettre à jour le hash en faisant glisser la fenêtre.",
        ],
        "complexity": {
            "best": "O(n + m)",
            "average": "O(n + m)",
            "worst": "O(nm)",
            "space": "O(1)",
        },
        "advantages": [
            "Très utile pour rechercher plusieurs motifs ou comparer des fenêtres.",
        ],
        "limitations": [
            "Les collisions de hash peuvent provoquer des comparaisons supplémentaires.",
        ],
        "example": {
            "input": "text='abracadabra', pattern='cada'",
            "result": "Indice 4",
        },
    },

    "z_search": {
        "name": "Z Algorithm",
        "category": "Strings",
        "description": (
            "Utilise un tableau Z indiquant la longueur du plus long préfixe "
            "correspondant à chaque position."
        ),
        "idea": "Réutiliser les correspondances déjà calculées pour obtenir une recherche linéaire.",
        "steps": [
            "Construire la chaîne pattern + séparateur + texte.",
            "Calculer le tableau Z.",
            "Repérer les positions où la valeur Z correspond à la longueur du motif.",
            "Convertir ces positions en indices du texte.",
        ],
        "complexity": {
            "best": "O(n + m)",
            "average": "O(n + m)",
            "worst": "O(n + m)",
            "space": "O(n + m)",
        },
        "advantages": [
            "Complexité linéaire.",
            "Technique générale de traitement de chaînes.",
        ],
        "limitations": [
            "Utilise un tableau supplémentaire.",
        ],
        "example": {
            "input": "text='abcxabcdabxabcdabcdabcy', pattern='abcdabcy'",
            "result": "Une occurrence est détectée.",
        },
    },

    "levenshtein_distance": {
        "name": "Levenshtein Distance",
        "category": "Strings",
        "description": (
            "Mesure le nombre minimal d'opérations d'insertion, suppression "
            "ou substitution nécessaires pour transformer une chaîne en une autre."
        ),
        "idea": "Construire une matrice représentant les distances entre préfixes.",
        "steps": [
            "Initialiser les distances des chaînes vides.",
            "Comparer chaque paire de caractères.",
            "Choisir le minimum entre insertion, suppression et substitution.",
            "Continuer jusqu'à la dernière cellule.",
        ],
        "complexity": {
            "best": "O(nm)",
            "average": "O(nm)",
            "worst": "O(nm)",
            "space": "O(min(n, m))",
        },
        "advantages": [
            "Très utile pour mesurer la similarité de chaînes.",
            "Utilisée notamment en correction orthographique et comparaison de textes.",
        ],
        "limitations": [
            "Le coût augmente avec la longueur des chaînes.",
        ],
        "example": {
            "input": "source='kitten', target='sitting'",
            "result": "Distance = 3",
        },
    },
}


# ============================================================================
# VALIDATION INTERNE
# ============================================================================


def _validate_algorithm_name(algorithm: str) -> None:
    """Valide le nom d'un algorithme."""
    if not isinstance(algorithm, str):
        raise ValueError("algorithm doit être une chaîne de caractères.")

    if not algorithm.strip():
        raise ValueError("algorithm ne peut pas être vide.")

    if algorithm not in _ALGORITHM_EXPLANATIONS:
        raise ValueError(f"Algorithme inconnu : {algorithm}")


# ============================================================================
# API PUBLIQUE
# ============================================================================


def get_algorithm_explanation(algorithm: str) -> dict:
    """
    Retourne l'explication complète d'un algorithme.

    Parameters
    ----------
    algorithm:
        Nom technique de l'algorithme.

    Returns
    -------
    dict
        Dictionnaire contenant toutes les informations pédagogiques.

    Raises
    ------
    ValueError
        Si l'algorithme est inconnu.
    """
    _validate_algorithm_name(algorithm)

    # Retourne une copie superficielle afin d'éviter que l'appelant
    # modifie directement la base interne.
    explanation = _ALGORITHM_EXPLANATIONS[algorithm].copy()

    explanation["complexity"] = explanation["complexity"].copy()
    explanation["advantages"] = list(explanation["advantages"])
    explanation["limitations"] = list(explanation["limitations"])
    explanation["steps"] = list(explanation["steps"])
    explanation["example"] = explanation["example"].copy()

    return explanation


def get_algorithm_description(algorithm: str) -> str:
    """Retourne uniquement la description d'un algorithme."""
    return get_algorithm_explanation(algorithm)["description"]


def get_algorithm_idea(algorithm: str) -> str:
    """Retourne l'idée principale d'un algorithme."""
    return get_algorithm_explanation(algorithm)["idea"]


def get_algorithm_steps(algorithm: str) -> list[str]:
    """Retourne les étapes de fonctionnement d'un algorithme."""
    return get_algorithm_explanation(algorithm)["steps"]


def get_algorithm_complexity(algorithm: str) -> dict:
    """Retourne les complexités temporelles et spatiales."""
    return get_algorithm_explanation(algorithm)["complexity"]


def get_algorithm_example(algorithm: str) -> dict:
    """Retourne un exemple pédagogique."""
    return get_algorithm_explanation(algorithm)["example"]


def list_explained_algorithms() -> list[str]:
    """
    Retourne la liste des algorithmes disposant d'une explication.
    """
    return list(_ALGORITHM_EXPLANATIONS.keys())


def list_algorithm_categories() -> list[str]:
    """
    Retourne les catégories couvertes par le module.
    """
    return list(
        dict.fromkeys(
            explanation["category"]
            for explanation in _ALGORITHM_EXPLANATIONS.values()
        )
    )


def get_algorithms_by_category(category: str) -> list[str]:
    """
    Retourne les algorithmes appartenant à une catégorie.

    La comparaison de catégorie est insensible à la casse.
    """
    if not isinstance(category, str):
        raise ValueError("category doit être une chaîne de caractères.")

    normalized_category = category.strip().lower()

    if not normalized_category:
        raise ValueError("category ne peut pas être vide.")

    return [
        algorithm
        for algorithm, explanation in _ALGORITHM_EXPLANATIONS.items()
        if explanation["category"].lower() == normalized_category
    ]


def get_category_summary(category: str) -> dict:
    """
    Retourne un résumé pédagogique d'une catégorie.
    """
    algorithms = get_algorithms_by_category(category)

    if not algorithms:
        raise ValueError(f"Catégorie inconnue : {category}")

    return {
        "category": _ALGORITHM_EXPLANATIONS[algorithms[0]]["category"],
        "count": len(algorithms),
        "algorithms": algorithms,
    }


def search_algorithm_explanations(keyword: str) -> list[str]:
    """
    Recherche des algorithmes à partir d'un mot-clé.

    La recherche porte sur :
    - le nom ;
    - la catégorie ;
    - la description ;
    - l'idée principale.
    """
    if not isinstance(keyword, str):
        raise ValueError("keyword doit être une chaîne de caractères.")

    keyword = keyword.strip().lower()

    if not keyword:
        raise ValueError("keyword ne peut pas être vide.")

    results = []

    for algorithm, explanation in _ALGORITHM_EXPLANATIONS.items():
        searchable_text = " ".join(
            [
                algorithm,
                explanation["name"],
                explanation["category"],
                explanation["description"],
                explanation["idea"],
            ]
        ).lower()

        if keyword in searchable_text:
            results.append(algorithm)

    return results


def format_complexity(algorithm: str) -> str:
    """
    Retourne la complexité d'un algorithme sous forme lisible.
    """
    complexity = get_algorithm_complexity(algorithm)

    return (
        f"Meilleur cas : {complexity['best']}\n"
        f"Cas moyen : {complexity['average']}\n"
        f"Pire cas : {complexity['worst']}\n"
        f"Espace : {complexity['space']}"
    )


def get_full_explanation(algorithm: str) -> str:
    """
    Génère une explication textuelle complète destinée à l'interface.
    """
    explanation = get_algorithm_explanation(algorithm)

    steps = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(explanation["steps"], start=1)
    )

    advantages = "\n".join(
        f"- {item}" for item in explanation["advantages"]
    )

    limitations = "\n".join(
        f"- {item}" for item in explanation["limitations"]
    )

    complexity = explanation["complexity"]

    return (
        f"# {explanation['name']}\n\n"
        f"**Catégorie :** {explanation['category']}\n\n"
        f"## Description\n"
        f"{explanation['description']}\n\n"
        f"## Idée principale\n"
        f"{explanation['idea']}\n\n"
        f"## Étapes\n"
        f"{steps}\n\n"
        f"## Complexité\n"
        f"- Meilleur cas : {complexity['best']}\n"
        f"- Cas moyen : {complexity['average']}\n"
        f"- Pire cas : {complexity['worst']}\n"
        f"- Espace : {complexity['space']}\n\n"
        f"## Avantages\n"
        f"{advantages}\n\n"
        f"## Limitations\n"
        f"{limitations}\n\n"
        f"## Exemple\n"
        f"Entrée : {explanation['example'].get('input', '')}\n"
        f"Résultat : {explanation['example'].get('result', '')}"
    )


__all__ = [
    "get_algorithm_explanation",
    "get_algorithm_description",
    "get_algorithm_idea",
    "get_algorithm_steps",
    "get_algorithm_complexity",
    "get_algorithm_example",
    "list_explained_algorithms",
    "list_algorithm_categories",
    "get_algorithms_by_category",
    "get_category_summary",
    "search_algorithm_explanations",
    "format_complexity",
    "get_full_explanation",
]

