
import streamlit as st
import math
import plotly.graph_objects as go

from core.algorithms.algorithm_explanations import (
    list_algorithm_categories,
    get_algorithms_by_category,
    get_algorithm_explanation,
    search_algorithm_explanations,
)

from core.algorithms.searching import (
    linear_search,
    binary_search,
    jump_search,
    interpolation_search,
    exponential_search,
)

from core.algorithms.sorting import (
    bubble_sort,
    selection_sort,
    insertion_sort,
    merge_sort,
    quick_sort,
    heap_sort,
    counting_sort,
    radix_sort,
    bucket_sort,
)

from core.algorithms.graph_algorithms import (
    bfs,
    dfs,
    dijkstra,
    bellman_ford,
    floyd_warshall,
    kruskal,
    prim,
)

from core.algorithms.dynamic_programming import (
    fibonacci,
    climbing_stairs,
    knapsack_01,
    coin_change,
    longest_common_subsequence,
    longest_increasing_subsequence,
    matrix_chain_multiplication,
)


from core.algorithms.greedy import (
    activity_selection,
    fractional_knapsack,
    greedy_coin_change,
    huffman_coding,
    job_sequencing,
    interval_scheduling,
)

from core.algorithms.backtracking import (
    n_queens,
    solve_sudoku,
    solve_maze,
    subsets,
    permutations,
    combination_sum,
)

from core.algorithms.tree_algorithms import (
    TreeNode,
    preorder_traversal,
    inorder_traversal,
    postorder_traversal,
    level_order_traversal,
    bst_search,
    bst_insert,
    bst_delete,
    MinHeap,
    PriorityQueue,
)


from core.algorithms.mathematical import (
    gcd,
    extended_gcd,
    generate_primes,
    sieve_of_eratosthenes,
    fast_power,
    modular_power,
    factorial,
    pascal_triangle,
)


from core.algorithms.string_algorithms import (
    naive_string_search,
    kmp_search,
    rabin_karp_search,
    z_algorithm_search,
    levenshtein_distance,
)




# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MathLab AI — Algorithms Lab",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# FONCTIONS UTILITAIRES — VISUALISATION DES GRAPHES
# ============================================================


def create_graph_positions(nodes):
    """
    Génère automatiquement une disposition circulaire
    des sommets du graphe.
    """

    if not nodes:
        return {}

    positions = {}

    radius = 1.0
    count = len(nodes)

    for index, node in enumerate(nodes):

        angle = (
            2 * math.pi * index / count
        )

        positions[node] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
        )

    return positions


def create_graph_figure(
    graph,
    weighted=False,
    highlighted_nodes=None,
    highlighted_edges=None,
):
    """
    Construit une visualisation Plotly du graphe.

    Parameters
    ----------
    graph:
        Graphe représenté sous forme de dictionnaire.

    weighted:
        Indique si le graphe contient des poids.

    highlighted_nodes:
        Sommets à mettre en évidence.

    highlighted_edges:
        Arêtes à mettre en évidence.
    """

    highlighted_nodes = set(
        highlighted_nodes or []
    )

    highlighted_edges = {
        tuple(edge)
        for edge in (
            highlighted_edges or []
        )
    }

    nodes = list(graph.keys())

    positions = create_graph_positions(
        nodes
    )

    figure = go.Figure()

    # ========================================================
    # ARÊTES
    # ========================================================

    edge_x = []
    edge_y = []

    processed_edges = set()

    for source in nodes:

        neighbors = graph[source]

        if weighted:

            neighbor_items = neighbors.items()

        else:

            neighbor_items = (
                (target, None)
                for target in neighbors
            )

        for target, weight in neighbor_items:

            if target not in positions:
                continue

            # Pour les graphes non orientés représentés
            # dans les deux sens, éviter le doublon visuel.
            edge_key = frozenset(
                [source, target]
            )

            if edge_key in processed_edges:
                continue

            processed_edges.add(
                edge_key
            )

            x0, y0 = positions[source]
            x1, y1 = positions[target]

            edge_x.extend(
                [x0, x1, None]
            )

            edge_y.extend(
                [y0, y1, None]
            )

    if edge_x:

        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={
                    "width": 1.5,
                },
                hoverinfo="none",
                name="Arêtes",
            )
        )

    # ========================================================
    # POIDS DES ARÊTES
    # ========================================================

    if weighted:

        annotations = []

        processed_weight_edges = set()

        for source in nodes:

            for target, weight in graph[
                source
            ].items():

                if target not in positions:
                    continue

                edge_key = frozenset(
                    [source, target]
                )

                if edge_key in processed_weight_edges:
                    continue

                processed_weight_edges.add(
                    edge_key
                )

                x0, y0 = positions[source]
                x1, y1 = positions[target]

                annotations.append(
                    {
                        "x": (
                            x0 + x1
                        ) / 2,
                        "y": (
                            y0 + y1
                        ) / 2,
                        "text": str(weight),
                        "showarrow": False,
                    }
                )

        figure.update_layout(
            annotations=annotations
        )

    # ========================================================
    # ARÊTES MISES EN ÉVIDENCE
    # ========================================================

    if highlighted_edges:

        highlighted_x = []
        highlighted_y = []

        for source, target in highlighted_edges:

            if (
                source not in positions
                or target not in positions
            ):
                continue

            x0, y0 = positions[source]
            x1, y1 = positions[target]

            highlighted_x.extend(
                [x0, x1, None]
            )

            highlighted_y.extend(
                [y0, y1, None]
            )

        if highlighted_x:

            figure.add_trace(
                go.Scatter(
                    x=highlighted_x,
                    y=highlighted_y,
                    mode="lines",
                    line={
                        "width": 5,
                    },
                    hoverinfo="none",
                    name="Arêtes sélectionnées",
                )
            )

    # ========================================================
    # SOMMETS
    # ========================================================

    node_x = []
    node_y = []
    node_text = []
    node_sizes = []

    for node in nodes:

        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)
        node_text.append(
            str(node)
        )

        if node in highlighted_nodes:

            node_sizes.append(28)

        else:

            node_sizes.append(20)

    if node_x:

        figure.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_text,
                textposition="middle center",
                marker={
                    "size": node_sizes,
                },
                hovertemplate=(
                    "Sommet : %{text}"
                    "<extra></extra>"
                ),
                name="Sommets",
            )
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    figure.update_layout(
        title="Visualisation du graphe",
        showlegend=True,
        height=600,
        hovermode="closest",
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
    )

    return figure


def display_graph_visualization(
    graph,
    weighted=False,
    highlighted_nodes=None,
    highlighted_edges=None,
):
    """
    Affiche le graphe dans une visualisation Plotly.
    """

    figure = create_graph_figure(
        graph,
        weighted=weighted,
        highlighted_nodes=highlighted_nodes,
        highlighted_edges=highlighted_edges,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )



# ============================================================
# FONCTIONS — PLUS COURTS CHEMINS
# ============================================================


def reconstruct_shortest_paths(
    graph,
    start_node,
    distances,
):
    """
    Reconstruit un chemin depuis start_node vers chaque sommet
    à partir des distances calculées.

    Cette fonction est utilisée pour Dijkstra et Bellman-Ford.

    Elle détermine un prédécesseur compatible avec les distances :
        distance[u] + poids(u, v) == distance[v]

    Returns
    -------
    dict
        Dictionnaire :
            sommet -> liste représentant le chemin.
    """

    paths = {}

    if start_node not in graph:
        return paths

    paths[start_node] = [start_node]

    # --------------------------------------------------------
    # Recherche des prédécesseurs
    # --------------------------------------------------------

    predecessors = {
        node: None
        for node in graph
    }

    for source in graph:

        for target, weight in graph[source].items():

            if (
                source in distances
                and target in distances
                and distances[source] != math.inf
                and distances[target] != math.inf
            ):

                if math.isclose(
                    distances[source] + weight,
                    distances[target],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):

                    if predecessors[target] is None:
                        predecessors[target] = source

    # --------------------------------------------------------
    # Reconstruction des chemins
    # --------------------------------------------------------

    for target in graph:

        if target == start_node:
            continue

        if (
            target not in distances
            or distances[target] == math.inf
        ):
            paths[target] = []
            continue

        path = []
        current = target
        visited = set()

        while current is not None:

            if current in visited:
                path = []
                break

            visited.add(current)
            path.append(current)

            if current == start_node:
                break

            current = predecessors.get(
                current
            )

        if not path or path[-1] != start_node:

            paths[target] = []

        else:

            path.reverse()
            paths[target] = path

    return paths


def get_path_edges(path):
    """
    Convertit un chemin de sommets en liste d'arêtes.

    Exemple
    -------
    ["A", "C", "D"]

    devient :

    [("A", "C"), ("C", "D")]
    """

    if len(path) < 2:
        return []

    return [
        (
            path[index],
            path[index + 1],
        )
        for index in range(
            len(path) - 1
        )
    ]




# ============================================================
# EN-TÊTE
# ============================================================

st.title("🧠 MathLab AI — Algorithms Lab")

st.markdown(
    """
    Explorez les principaux algorithmes utilisés en informatique
    et en mathématiques discrètes.

    Vous pouvez consulter leur fonctionnement, leur complexité,
    leurs avantages, leurs limitations et des exemples pédagogiques.
    """
)

st.info(
    """
    **Catégories disponibles :**
    Searching · Sorting · Graphs · Dynamic Programming ·
    Greedy · Backtracking · Trees · Mathematical · Strings
    """
)


# ============================================================
# ÉTAPE 9 — RECHERCHE D'ALGORITHMES
# ============================================================

st.header("🔎 Rechercher un algorithme")

st.markdown(
    """
    Recherchez directement un algorithme par son **nom**, sa
    **catégorie** ou un **mot-clé**.
    
    Exemples : `binary`, `Dijkstra`, `Fibonacci`, `tri`, `graphes`
    """
)

search_query = st.text_input(
    "Rechercher",
    placeholder="Ex. binary, Dijkstra, Fibonacci, tri, graphes...",
    key="algorithm_search",
)

if search_query.strip():

    search_results = search_algorithm_explanations(
        search_query.strip()
    )

    if search_results:

        st.success(
            f"{len(search_results)} algorithme(s) trouvé(s) "
            f"pour « {search_query.strip()} »."
        )

        st.markdown("### 📚 Résultats de la recherche")

        for algorithm in search_results:

            explanation = get_algorithm_explanation(algorithm)

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.subheader(
                        f"🧠 {explanation['name']}"
                    )

                    st.caption(
                        f"Catégorie : "
                        f"**{explanation['category']}**"
                    )

                    st.write(
                        explanation["description"]
                    )

                with col2:
                    st.metric(
                        "Pire cas",
                        explanation["complexity"].get(
                            "worst",
                            "N/D",
                        ),
                    )

    else:

        st.warning(
            f"Aucun algorithme trouvé pour "
            f"« {search_query.strip()} »."
        )

        st.caption(
            "Essayez par exemple : binary, Dijkstra, "
            "Fibonacci, tri ou graphes."
        )


# ============================================================
# SÉLECTION PAR CATÉGORIE
# ============================================================

st.divider()

st.header("📂 Explorer par catégorie")

categories = list_algorithm_categories()

selected_category = st.selectbox(
    "Choisissez une catégorie",
    categories,
)

algorithms = get_algorithms_by_category(
    selected_category
)

st.caption(
    f"{len(algorithms)} algorithme(s) disponible(s) "
    f"dans la catégorie **{selected_category}**."
)


# ============================================================
# SÉLECTION DE L'ALGORITHME
# ============================================================

selected_algorithm = st.selectbox(
    "Choisissez un algorithme",
    algorithms,
)

st.success(
    f"Algorithme sélectionné : **{selected_algorithm}**"
)


# ============================================================
# RÉCUPÉRATION DES EXPLICATIONS
# ============================================================

explanation = get_algorithm_explanation(
    selected_algorithm
)


# ============================================================
# PRÉSENTATION GÉNÉRALE
# ============================================================

st.header("📖 Explication de l'algorithme")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(
        f"🧠 {explanation['name']}"
    )

with col2:
    st.metric(
        "Catégorie",
        explanation["category"],
    )


# ============================================================
# DESCRIPTION
# ============================================================

st.markdown("### 📝 Description")

st.write(
    explanation["description"]
)


# ============================================================
# IDÉE PRINCIPALE
# ============================================================

st.markdown("### 💡 Idée principale")

st.info(
    explanation["idea"]
)


# ============================================================
# ÉTAPES
# ============================================================

st.markdown("### 🔢 Étapes de l'algorithme")

for index, step in enumerate(
    explanation["steps"],
    start=1,
):
    st.markdown(
        f"**{index}.** {step}"
    )


# ============================================================
# COMPLEXITÉ
# ============================================================

st.header("📊 Complexité algorithmique")

complexity = explanation["complexity"]

best = complexity.get(
    "best",
    "Non disponible",
)

average = complexity.get(
    "average",
    "Non disponible",
)

worst = complexity.get(
    "worst",
    "Non disponible",
)

space = complexity.get(
    "space",
    "Non disponible",
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🟢 Meilleur cas",
        best,
    )

with col2:
    st.metric(
        "🔵 Cas moyen",
        average,
    )

with col3:
    st.metric(
        "🔴 Pire cas",
        worst,
    )

with col4:
    st.metric(
        "💾 Espace",
        space,
    )


with st.expander("ℹ️ Comprendre la notation Big-O"):

    st.markdown(
        """
        La notation **Big-O** permet de décrire la manière
        dont le temps ou l'espace nécessaire à un algorithme
        évolue lorsque la taille des données augmente.

        - **O(1)** → constant
        - **O(log n)** → logarithmique
        - **O(n)** → linéaire
        - **O(n log n)** → linéarithmique
        - **O(n²)** → quadratique
        - **O(2ⁿ)** → exponentiel
        - **O(n!)** → factoriel
        """
    )


# ============================================================
# AVANTAGES / LIMITATIONS
# ============================================================

st.header("⚖️ Avantages et limitations")

advantages = explanation["advantages"]
limitations = explanation["limitations"]

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "✅ Avantages",
        len(advantages),
    )

with col2:

    st.metric(
        "⚠️ Limitations",
        len(limitations),
    )

col1, col2 = st.columns(2)

with col1:

    st.markdown("### ✅ Avantages")

    if advantages:

        for index, advantage in enumerate(
            advantages,
            start=1,
        ):

            with st.container(border=True):

                st.markdown(
                    f"**Avantage {index}**"
                )

                st.write(
                    advantage
                )

    else:

        st.info(
            "Aucun avantage renseigné."
        )


with col2:

    st.markdown("### ⚠️ Limitations")

    if limitations:

        for index, limitation in enumerate(
            limitations,
            start=1,
        ):

            with st.container(border=True):

                st.markdown(
                    f"**Limitation {index}**"
                )

                st.write(
                    limitation
                )

    else:

        st.info(
            "Aucune limitation renseignée."
        )


# ============================================================
# EXEMPLE PÉDAGOGIQUE
# ============================================================

st.header("🧪 Exemple pédagogique")

example = explanation["example"]

if isinstance(example, dict):

    if "input" in example:

        st.markdown("### 📥 Entrée")

        with st.container(border=True):

            st.code(
                str(example["input"]),
                language="text",
            )

    st.markdown("### ⚙️ Démarche")

    with st.container(border=True):

        st.markdown(
            f"""
            L'algorithme **{explanation["name"]}**
            applique les étapes présentées précédemment
            sur les données d'entrée.
            """
        )

        for index, step in enumerate(
            explanation["steps"],
            start=1,
        ):

            st.markdown(
                f"**Étape {index} :** {step}"
            )

    if "result" in example:

        st.markdown("### 📤 Résultat")

        with st.container(border=True):

            st.code(
                str(example["result"]),
                language="text",
            )

            st.success(
                "Résultat obtenu par l'algorithme."
            )

    st.markdown("### 💡 Interprétation")

    with st.container(border=True):

        if "result" in example:

            st.write(
                f"""
                Pour l'exemple sélectionné,
                l'algorithme **{explanation["name"]}**
                transforme ou traite l'entrée donnée
                afin d'obtenir le résultat présenté
                ci-dessus.
                """
            )

        else:

            st.write(
                "Cet exemple illustre le fonctionnement "
                "général de l'algorithme."
            )

else:

    st.info(
        "Aucun exemple pédagogique disponible "
        "pour cet algorithme."
    )



# ============================================================
# ÉTAPE 10 — RÉSUMÉ DES CATÉGORIES
# ============================================================

st.header("📚 Résumé des catégories")

st.markdown(
    """
    Explorez rapidement les différentes familles d'algorithmes
    disponibles dans **MathLab AI**.
    """
)

# Récupération des catégories disponibles
categories = list_algorithm_categories()

# Correspondance catégorie → icône
category_icons = {
    "Searching": "🔎",
    "Sorting": "🔢",
    "Graphs": "🕸️",
    "Dynamic Programming": "🧩",
    "Greedy": "🎯",
    "Backtracking": "↩️",
    "Trees": "🌳",
    "Mathematical": "📐",
    "Strings": "🔤",
}

# Affichage sous forme de cartes
columns = st.columns(3)

for index, category in enumerate(categories):

    algorithms_in_category = get_algorithms_by_category(
        category
    )

    icon = category_icons.get(
        category,
        "🧠",
    )

    column = columns[index % 3]

    with column:

        with st.container(border=True):

            st.markdown(
                f"## {icon} {category}"
            )

            st.metric(
                "Algorithmes",
                len(algorithms_in_category),
            )

            if algorithms_in_category:

                st.caption(
                    " · ".join(
                        algorithms_in_category
                    )
                )

            else:

                st.caption(
                    "Aucun algorithme disponible."
                )


# ============================================================
# ÉTAPE 12 — EXPÉRIMENTATIONS PAR CATÉGORIE
# ============================================================

st.header("🧪 Expérimentation interactive")

st.markdown(
    """
    Exécutez réellement l'algorithme sélectionné avec vos
    propres données.

    Les résultats sont calculés directement à partir des
    implémentations présentes dans `core.algorithms`.
    """
)


# ============================================================
# FONCTION UTILITAIRE — PARSING DES LISTES
# ============================================================

def parse_algorithm_list(value: str) -> list:
    """
    Convertit une saisie utilisateur en liste de valeurs.

    Exemples :
        "10, 5, 8, 2"
        -> [10, 5, 8, 2]

        "10; 5; 8; 2"
        -> [10, 5, 8, 2]

        "10 5 8 2"
        -> [10, 5, 8, 2]
    """

    if not isinstance(value, str):
        raise TypeError(
            "Les données doivent être saisies sous forme de texte."
        )

    cleaned = value.replace(";", ",").strip()

    if not cleaned:
        raise ValueError(
            "Les données ne peuvent pas être vides."
        )

    if "," in cleaned:

        parts = [
            part.strip()
            for part in cleaned.split(",")
            if part.strip()
        ]

    else:

        parts = [
            part.strip()
            for part in cleaned.split()
            if part.strip()
        ]

    if not parts:
        raise ValueError(
            "Aucune donnée valide n'a été trouvée."
        )

    # --------------------------------------------------------
    # Conversion numérique
    # --------------------------------------------------------

    try:

        numbers = [
            float(part)
            for part in parts
        ]

        if all(
            number.is_integer()
            for number in numbers
        ):

            return [
                int(number)
                for number in numbers
            ]

        return numbers

    except ValueError:

        # Permet également de travailler avec des chaînes
        return parts


# ============================================================
# INFORMATIONS SUR L'ALGORITHME
# ============================================================

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🧠 Algorithme")

        st.write(
            explanation["name"]
        )

    with col2:

        st.markdown("### 📂 Catégorie")

        st.write(
            explanation["category"]
        )

    with col3:

        st.markdown("### ⚙️ Pire cas")

        st.write(
            explanation["complexity"].get(
                "worst",
                "N/D",
            )
        )


# ============================================================
# SEARCHING
# ============================================================

if selected_category == "Searching":

    st.subheader(
        "🔎 Expérimentation — Recherche"
    )

    st.markdown(
        """
        Entrez une liste de données puis une valeur à rechercher.

        **Linear Search** peut travailler sur une liste quelconque.

        **Binary Search**, **Jump Search**, **Interpolation Search**
        et **Exponential Search** nécessitent une liste triée
        dans l'ordre croissant.
        """
    )

    data_text = st.text_area(
        "Liste de données",
        value="10, 4, 7, 2, 15, 8",
        placeholder="Ex. 10, 4, 7, 2, 15, 8",
        key="search_data",
        height=100,
    )

    target_text = st.text_input(
        "Valeur recherchée",
        value="8",
        placeholder="Ex. 8",
        key="search_target",
    )

    st.divider()

    if st.button(
        "▶️ Rechercher",
        type="primary",
        use_container_width=True,
    ):

        try:

            data = parse_algorithm_list(
                data_text
            )

            target = target_text.strip()

            if not target:

                raise ValueError(
                    "La valeur recherchée ne peut pas être vide."
                )

            # ------------------------------------------------
            # Conversion de la cible
            # ------------------------------------------------

            try:

                numeric_target = float(target)

                if numeric_target.is_integer():

                    target = int(
                        numeric_target
                    )

                else:

                    target = numeric_target

            except ValueError:

                # Conserver une éventuelle valeur textuelle
                target = target

            # ------------------------------------------------
            # Algorithmes de recherche
            # ------------------------------------------------

            search_functions = {
                "linear_search": linear_search,
                "binary_search": binary_search,
                "jump_search": jump_search,
                "interpolation_search": interpolation_search,
                "exponential_search": exponential_search,
            }

            algorithm_function = search_functions.get(
                selected_algorithm
            )

            if algorithm_function is None:

                raise ValueError(
                    f"L'algorithme "
                    f"« {selected_algorithm} » "
                    "n'est pas disponible pour "
                    "l'expérimentation."
                )

            # ------------------------------------------------
            # Exécution réelle
            # ------------------------------------------------

            result = algorithm_function(
                data,
                target,
            )

            # ------------------------------------------------
            # Affichage
            # ------------------------------------------------

            st.success(
                "Recherche exécutée avec succès."
            )

            st.markdown("### 📊 Résultat")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Index",
                    result,
                )

            with col2:

                if result == -1:

                    st.metric(
                        "Statut",
                        "Non trouvé",
                    )

                else:

                    st.metric(
                        "Statut",
                        "Trouvé",
                    )

            with st.container(border=True):

                if result == -1:

                    st.warning(
                        f"La valeur **{target}** "
                        "n'a pas été trouvée dans la liste."
                    )

                else:

                    st.success(
                        f"La valeur **{target}** "
                        f"a été trouvée à l'index **{result}**."
                    )

            st.markdown(
                "### 📥 Données utilisées"
            )

            st.code(
                str(data),
                language="text",
            )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur de recherche : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "l'exécution de la recherche."
            )

            st.exception(error)


# ============================================================
# SORTING
# ============================================================

elif selected_category == "Sorting":

    st.subheader(
        "🔢 Expérimentation — Tri"
    )

    st.markdown(
        """
        Entrez une liste de valeurs et observez le résultat
        du tri effectué par l'algorithme sélectionné.
        """
    )

    data_text = st.text_area(
        "Liste à trier",
        value="10, 4, 7, 2, 15, 8",
        placeholder="Ex. 10, 4, 7, 2, 15, 8",
        key="sorting_data",
        height=100,
    )

    st.divider()

    if st.button(
        "▶️ Trier les données",
        type="primary",
        use_container_width=True,
    ):

        try:

            data = parse_algorithm_list(
                data_text
            )

            sorting_functions = {
                "bubble_sort": bubble_sort,
                "selection_sort": selection_sort,
                "insertion_sort": insertion_sort,
                "merge_sort": merge_sort,
                "quick_sort": quick_sort,
                "heap_sort": heap_sort,
                "counting_sort": counting_sort,
                "radix_sort": radix_sort,
                "bucket_sort": bucket_sort,
            }

            algorithm_function = sorting_functions.get(
                selected_algorithm
            )

            if algorithm_function is None:

                raise ValueError(
                    f"L'algorithme "
                    f"« {selected_algorithm} » "
                    "n'est pas disponible pour "
                    "l'expérimentation."
                )

            # ------------------------------------------------
            # Exécution réelle
            # ------------------------------------------------

            result = algorithm_function(
                data
            )

            # ------------------------------------------------
            # Affichage
            # ------------------------------------------------

            st.success(
                "Tri exécuté avec succès."
            )

            st.markdown("### 📊 Résultat")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 📥 Avant")

                with st.container(border=True):

                    st.code(
                        str(data),
                        language="text",
                    )

            with col2:

                st.markdown("### 📤 Après")

                with st.container(border=True):

                    st.code(
                        str(result),
                        language="text",
                    )

            st.divider()

            st.markdown(
                "### 🔍 Vérification"
            )

            if result == sorted(data):

                st.success(
                    "Le résultat est correctement trié "
                    "dans l'ordre croissant."
                )

            else:

                st.warning(
                    "Le résultat obtenu diffère du tri "
                    "croissant Python de référence."
                )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur de tri : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "l'exécution du tri."
            )

            st.exception(error)


elif selected_category == "Graphs":

    st.markdown("### 🕸️ Expérimentation des graphes")

    st.markdown(
        """
        Les algorithmes de graphes utilisent deux représentations
        différentes :

        - **Graphe non pondéré** pour BFS et DFS
        - **Graphe pondéré** pour Dijkstra, Bellman-Ford,
          Floyd-Warshall, Kruskal et Prim
        """
    )

    # ========================================================
    # GRAPHES D'EXEMPLE
    # ========================================================

    unweighted_graph_example = """{
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C"]
}"""

    weighted_graph_example = """{
    "A": {"B": 4, "C": 2},
    "B": {"A": 4, "D": 5},
    "C": {"A": 2, "D": 1},
    "D": {"B": 5, "C": 1}
}"""

    # ========================================================
    # TYPE DE GRAPHE
    # ========================================================

    if selected_algorithm in ["bfs", "dfs"]:

        graph_text = st.text_area(
            "Graphe",
            value=unweighted_graph_example,
            height=220,
            key="graph_unweighted_input",
            help=(
                "Utilisez un dictionnaire JSON dont chaque "
                "sommet contient une liste de voisins."
            ),
        )

        start_node = st.text_input(
            "Sommet de départ",
            value="A",
            key="graph_start_node",
        )

    else:

        graph_text = st.text_area(
            "Graphe pondéré",
            value=weighted_graph_example,
            height=220,
            key="graph_weighted_input",
            help=(
                "Utilisez un dictionnaire JSON dont chaque "
                "sommet contient un dictionnaire "
                "voisin → poids."
            ),
        )

        if selected_algorithm in [
            "dijkstra",
            "bellman_ford",
            "prim",
        ]:

            start_node = st.text_input(
                "Sommet de départ",
                value="A",
                key="graph_weighted_start_node",
            )

        else:

            start_node = None

    # ========================================================
    # EXPLICATION DU FORMAT
    # ========================================================

    with st.expander("ℹ️ Comprendre le format du graphe"):

        if selected_algorithm in ["bfs", "dfs"]:

            st.code(
                unweighted_graph_example,
                language="json",
            )

            st.caption(
                "Chaque sommet est associé à la liste de ses voisins."
            )

        else:

            st.code(
                weighted_graph_example,
                language="json",
            )

            st.caption(
                "Chaque sommet est associé à un dictionnaire "
                "voisin → poids."
            )

    
    # ========================================================
    # EXÉCUTION
    # ========================================================

    if st.button(
        "▶️ Exécuter l'algorithme",
        type="primary",
        use_container_width=True,
        key="execute_graph_algorithm",
    ):

        try:

            import json

            graph = json.loads(
                graph_text
            )

            if not isinstance(graph, dict):
                raise ValueError(
                    "Le graphe doit être un objet JSON."
                )

            # ==================================================
            # BFS
            # ==================================================

            if selected_algorithm == "bfs":

                result = bfs(
                    graph,
                    start_node,
                )

                st.success(
                    "Parcours BFS exécuté avec succès."
                )

                st.subheader("Ordre de parcours")

                st.write(
                    " → ".join(
                        str(node)
                        for node in result
                    )
                )

                st.dataframe(
                    {
                        "Étape": list(
                            range(
                                1,
                                len(result) + 1,
                            )
                        ),
                        "Sommet": result,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # VISUALISATION BFS
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation du graphe"
                )

                display_graph_visualization(
                    graph,
                    weighted=False,
                    highlighted_nodes=result,
                )

                st.caption(
                    "Les sommets visités par BFS sont "
                    "mis en évidence."
                )

            # ==================================================
            # DFS
            # ==================================================

            elif selected_algorithm == "dfs":

                result = dfs(
                    graph,
                    start_node,
                )

                st.success(
                    "Parcours DFS exécuté avec succès."
                )

                st.subheader("Ordre de parcours")

                st.write(
                    " → ".join(
                        str(node)
                        for node in result
                    )
                )

                st.dataframe(
                    {
                        "Étape": list(
                            range(
                                1,
                                len(result) + 1,
                            )
                        ),
                        "Sommet": result,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # VISUALISATION DFS
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation du graphe"
                )

                display_graph_visualization(
                    graph,
                    weighted=False,
                    highlighted_nodes=result,
                )

                st.caption(
                    "Les sommets visités par DFS sont "
                    "mis en évidence."
                )


            # ==================================================
            # DIJKSTRA
            # ==================================================

            elif selected_algorithm == "dijkstra":

                distances = dijkstra(
                    graph,
                    start_node,
                )

                st.success(
                    "Algorithme de Dijkstra exécuté avec succès."
                )

                st.subheader(
                    "Distances minimales depuis "
                    f"{start_node}"
                )

                st.dataframe(
                    {
                        "Sommet": list(
                            distances.keys()
                        ),
                        "Distance": list(
                            distances.values()
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # RECONSTRUCTION DES CHEMINS
                # ------------------------------------------------

                paths = reconstruct_shortest_paths(
                    graph,
                    start_node,
                    distances,
                )

                st.divider()

                st.subheader(
                    "🛣️ Plus courts chemins"
                )

                path_rows = []

                for target, path in paths.items():

                    if path:

                        path_rows.append(
                            {
                                "Départ": start_node,
                                "Destination": target,
                                "Chemin": " → ".join(
                                    str(node)
                                    for node in path
                                ),
                                "Distance": distances[target],
                            }
                        )

                    else:

                        path_rows.append(
                            {
                                "Départ": start_node,
                                "Destination": target,
                                "Chemin": "Aucun chemin",
                                "Distance": distances[target],
                            }
                        )

                st.dataframe(
                    path_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # CHOIX DE LA DESTINATION
                # ------------------------------------------------

                reachable_nodes = [
                    node
                    for node in paths
                    if paths[node]
                ]

                if reachable_nodes:

                    destination = st.selectbox(
                        "Destination à visualiser",
                        reachable_nodes,
                        key="dijkstra_destination",
                    )

                    selected_path = paths[
                        destination
                    ]

                    selected_edges = get_path_edges(
                        selected_path
                    )

                    st.markdown(
                        "### 🛣️ Chemin sélectionné"
                    )

                    st.success(
                        " → ".join(
                            str(node)
                            for node in selected_path
                        )
                    )

                    st.metric(
                        "Distance minimale",
                        distances[destination],
                    )

                    # ------------------------------------------------
                    # VISUALISATION
                    # ------------------------------------------------

                    st.subheader(
                        "🕸️ Visualisation du plus court chemin"
                    )

                    display_graph_visualization(
                        graph,
                        weighted=True,
                        highlighted_nodes=selected_path,
                        highlighted_edges=selected_edges,
                    )

                    st.caption(
                        "Les sommets et les arêtes du plus court "
                        "chemin sélectionné sont mis en évidence."
                    )

                else:

                    st.warning(
                        "Aucun chemin accessible depuis "
                        f"{start_node}."
                    )


            # ==================================================
            # BELLMAN-FORD
            # ==================================================

            elif selected_algorithm == "bellman_ford":

                distances = bellman_ford(
                    graph,
                    start_node,
                )

                st.success(
                    "Algorithme de Bellman-Ford exécuté avec succès."
                )

                st.subheader(
                    "Distances minimales depuis "
                    f"{start_node}"
                )

                st.dataframe(
                    {
                        "Sommet": list(
                            distances.keys()
                        ),
                        "Distance": list(
                            distances.values()
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # RECONSTRUCTION DES CHEMINS
                # ------------------------------------------------

                paths = reconstruct_shortest_paths(
                    graph,
                    start_node,
                    distances,
                )

                st.divider()

                st.subheader(
                    "🛣️ Plus courts chemins"
                )

                path_rows = []

                for target, path in paths.items():

                    if path:

                        path_rows.append(
                            {
                                "Départ": start_node,
                                "Destination": target,
                                "Chemin": " → ".join(
                                    str(node)
                                    for node in path
                                ),
                                "Distance": distances[target],
                            }
                        )

                    else:

                        path_rows.append(
                            {
                                "Départ": start_node,
                                "Destination": target,
                                "Chemin": "Aucun chemin",
                                "Distance": distances[target],
                            }
                        )

                st.dataframe(
                    path_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # CHOIX DE LA DESTINATION
                # ------------------------------------------------

                reachable_nodes = [
                    node
                    for node in paths
                    if paths[node]
                ]

                if reachable_nodes:

                    destination = st.selectbox(
                        "Destination à visualiser",
                        reachable_nodes,
                        key="bellman_ford_destination",
                    )

                    selected_path = paths[
                        destination
                    ]

                    selected_edges = get_path_edges(
                        selected_path
                    )

                    st.markdown(
                        "### 🛣️ Chemin sélectionné"
                    )

                    st.success(
                        " → ".join(
                            str(node)
                            for node in selected_path
                        )
                    )

                    st.metric(
                        "Distance minimale",
                        distances[destination],
                    )

                    # ------------------------------------------------
                    # VISUALISATION
                    # ------------------------------------------------

                    st.subheader(
                        "🕸️ Visualisation du plus court chemin"
                    )

                    display_graph_visualization(
                        graph,
                        weighted=True,
                        highlighted_nodes=selected_path,
                        highlighted_edges=selected_edges,
                    )

                    st.caption(
                        "Les sommets et les arêtes du plus court "
                        "chemin sélectionné sont mis en évidence."
                    )

                else:

                    st.warning(
                        "Aucun chemin accessible depuis "
                        f"{start_node}."
                    )



                # ------------------------------------------------
                # VISUALISATION FLOYD-WARSHALL
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation du graphe"
                )

                display_graph_visualization(
                    graph,
                    weighted=True,
                )

                st.caption(
                    "Le graphe pondéré utilisé pour calculer "
                    "les plus courtes distances entre toutes "
                    "les paires de sommets."
                )

            # ==================================================
            # KRUSKAL
            # ==================================================

            elif selected_algorithm == "kruskal":

                result = kruskal(
                    graph,
                )

                edges, total_weight = result

                st.success(
                    "Algorithme de Kruskal exécuté avec succès."
                )

                st.subheader(
                    "Arbre couvrant minimal"
                )

                st.dataframe(
                    {
                        "Arête": [
                            f"{edge[0]} — {edge[1]}"
                            for edge in edges
                        ],
                        "Poids": [
                            edge[2]
                            for edge in edges
                        ],
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.metric(
                    "Poids total",
                    total_weight,
                )

                # ------------------------------------------------
                # VISUALISATION KRUSKAL
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation de l'arbre couvrant minimal"
                )

                highlighted_edges = [
                    (
                        edge[0],
                        edge[1],
                    )
                    for edge in edges
                ]

                display_graph_visualization(
                    graph,
                    weighted=True,
                    highlighted_edges=highlighted_edges,
                )

                st.caption(
                    "Les arêtes sélectionnées par Kruskal "
                    "sont mises en évidence."
                )

            # ==================================================
            # PRIM
            # ==================================================

            elif selected_algorithm == "prim":

                result = prim(
                    graph,
                    start_node,
                )

                edges, total_weight = result

                st.success(
                    "Algorithme de Prim exécuté avec succès."
                )

                st.subheader(
                    "Arbre couvrant minimal"
                )

                st.dataframe(
                    {
                        "Arête": [
                            f"{edge[0]} — {edge[1]}"
                            for edge in edges
                        ],
                        "Poids": [
                            edge[2]
                            for edge in edges
                        ],
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.metric(
                    "Poids total",
                    total_weight,
                )

                # ------------------------------------------------
                # VISUALISATION PRIM
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation de l'arbre couvrant minimal"
                )

                highlighted_edges = [
                    (
                        edge[0],
                        edge[1],
                    )
                    for edge in edges
                ]

                display_graph_visualization(
                    graph,
                    weighted=True,
                    highlighted_edges=highlighted_edges,
                    highlighted_nodes=[
                        start_node
                    ],
                )

                st.caption(
                    "Les arêtes sélectionnées par Prim "
                    "sont mises en évidence et le sommet "
                    "de départ est également identifié."
                )

            else:

                st.info(
                    "Cet algorithme de graphe n'est pas encore "
                    "connecté à la zone d'expérimentation."
                )

        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as error:

            st.error(
                f"Erreur de graphe : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "l'exécution de l'algorithme."
            )

            st.exception(error)


# ============================================================
# DYNAMIC PROGRAMMING
# ============================================================

elif selected_category == "Dynamic Programming":

    st.subheader(
        "🧩 Expérimentation — Programmation dynamique"
    )

    st.markdown(
        """
        Cette section permet d'expérimenter plusieurs algorithmes
        classiques de **programmation dynamique** et de visualiser
        leur démarche étape par étape.
        """
    )

    st.divider()

    # ========================================================
    # FIBONACCI
    # ========================================================

    if selected_algorithm == "fibonacci":

        st.markdown(
            "### 🔢 Fibonacci"
        )

        st.markdown(
            """
            La suite de Fibonacci est définie par :

            \[
            F(0)=0,\qquad F(1)=1
            \]

            puis :

            \[
            F(n)=F(n-1)+F(n-2)
            \]

            Chaque valeur dépend donc des deux valeurs précédentes.
            """
        )

        n = st.number_input(
            "Indice n",
            min_value=0,
            max_value=1000,
            value=10,
            step=1,
            key="dp_fibonacci_n",
        )

        if st.button(
            "Calculer Fibonacci",
            type="primary",
            use_container_width=True,
            key="execute_fibonacci",
        ):

            n_value = int(n)

            try:

                result = fibonacci(n_value)

                st.success(
                    "Calcul de Fibonacci terminé."
                )

                st.metric(
                    f"F({n_value})",
                    str(result),
                )

                st.divider()

                # ==================================================
                # INTERPRÉTATION
                # ==================================================

                st.markdown(
                    "### 🧠 Interprétation"
                )

                st.info(
                    f"""
                    Pour calculer F({n_value}), l'algorithme
                    construit progressivement les valeurs précédentes
                    de la suite au lieu de recalculer plusieurs fois
                    les mêmes sous-problèmes.
                    """
                )

                st.divider()

                # ==================================================
                # CAS PARTICULIERS
                # ==================================================

                st.markdown(
                    "### 📌 Cas de base"
                )

                if n_value == 0:

                    st.latex(
                        r"F(0)=0"
                    )

                elif n_value == 1:

                    st.latex(
                        r"F(1)=1"
                    )

                else:

                    st.latex(
                        r"F(0)=0"
                    )

                    st.latex(
                        r"F(1)=1"
                    )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🔎 Calcul étape par étape"
                )

                if n_value >= 2:

                    fibonacci_values = [0, 1]

                    for index in range(
                        2,
                        n_value + 1,
                    ):

                        previous_1 = fibonacci_values[
                            index - 1
                        ]

                        previous_2 = fibonacci_values[
                            index - 2
                        ]

                        current = (
                            previous_1
                            + previous_2
                        )

                        fibonacci_values.append(
                            current
                        )

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"### Étape {index}"
                            )

                            st.latex(
                                rf"""
                                F({index})
                                =
                                F({index - 1})
                                +
                                F({index - 2})
                                """
                            )

                            st.latex(
                                rf"""
                                F({index})
                                =
                                {previous_1}
                                +
                                {previous_2}
                                =
                                {current}
                                """
                            )

                    st.divider()

                    st.markdown(
                        "### 📊 Tableau de la suite"
                    )

                    st.dataframe(
                        {
                            "Indice n": list(
                                range(
                                    n_value + 1
                                )
                            ),
                            "F(n)": fibonacci_values,
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                if n_value == 0:

                    expected = 0

                elif n_value == 1:

                    expected = 1

                else:

                    expected = fibonacci_values[-1]

                if result == expected:

                    st.success(
                        "La valeur calculée par le module "
                        "correspond au calcul pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le calcul pédagogique et le module."
                    )

            except Exception as error:

                st.error(
                    f"Une erreur est survenue lors du calcul de Fibonacci : {error}"
                )

    # ========================================================
    # CLIMBING STAIRS
    # ========================================================

    elif selected_algorithm == "climbing_stairs":

        st.markdown(
            "### 🪜 Climbing Stairs"
        )

        st.markdown(
            """
            On dispose de **n marches**.

            À chaque mouvement, on peut monter :

            - 1 marche ;
            - 2 marches.

            Le nombre de façons d'atteindre la marche `n` est donné par :

            \[
            C(n)=C(n-1)+C(n-2)
            \]

            avec :

            \[
            C(0)=1,\qquad C(1)=1
            \]

            Pourquoi ?

            - Si le dernier mouvement fait **1 marche**, il reste
              `n-1` marches.
            - Si le dernier mouvement fait **2 marches**, il reste
              `n-2` marches.

            On additionne donc les deux possibilités.
            """
        )

        n = st.number_input(
            "Nombre de marches n",
            min_value=0,
            max_value=1000,
            value=5,
            step=1,
            key="dp_climbing_stairs_n",
        )

        if st.button(
            "Calculer le nombre de façons",
            type="primary",
            use_container_width=True,
            key="execute_climbing_stairs",
        ):

            n_value = int(n)

            try:

                result = climbing_stairs(
                    n_value
                )

                st.success(
                    "Calcul de Climbing Stairs terminé."
                )

                st.metric(
                    f"C({n_value})",
                    str(result),
                )

                st.divider()

                # ==================================================
                # INTERPRÉTATION
                # ==================================================

                st.markdown(
                    "### 🧠 Interprétation"
                )

                st.info(
                    f"""
                    Il existe **{result} façon(s)** différentes
                    d'atteindre la marche {n_value} lorsque chaque
                    mouvement permet de monter 1 ou 2 marches.
                    """
                )

                st.divider()

                # ==================================================
                # CAS DE BASE
                # ==================================================

                st.markdown(
                    "### 📌 Cas de base"
                )

                st.latex(
                    r"C(0)=1"
                )

                st.latex(
                    r"C(1)=1"
                )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                if n_value >= 2:

                    st.divider()

                    st.markdown(
                        "### 🔎 Calcul étape par étape"
                    )

                    climbing_values = [
                        1,
                        1,
                    ]

                    for index in range(
                        2,
                        n_value + 1,
                    ):

                        previous_1 = climbing_values[
                            index - 1
                        ]

                        previous_2 = climbing_values[
                            index - 2
                        ]

                        current = (
                            previous_1
                            + previous_2
                        )

                        climbing_values.append(
                            current
                        )

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"### Étape {index}"
                            )

                            st.latex(
                                rf"""
                                C({index})
                                =
                                C({index - 1})
                                +
                                C({index - 2})
                                """
                            )

                            st.latex(
                                rf"""
                                C({index})
                                =
                                {previous_1}
                                +
                                {previous_2}
                                =
                                {current}
                                """
                            )

                    st.divider()

                    st.markdown(
                        "### 📊 Tableau des possibilités"
                    )

                    st.dataframe(
                        {
                            "Nombre de marches n": list(
                                range(
                                    n_value + 1
                                )
                            ),
                            "Nombre de façons C(n)": climbing_values,
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                if n_value == 0:

                    expected = 1

                elif n_value == 1:

                    expected = 1

                else:

                    expected = climbing_values[-1]

                if result == expected:

                    st.success(
                        "La valeur calculée par le module "
                        "correspond au calcul pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le calcul pédagogique et le module."
                    )

            except Exception as error:

                st.error(
                    f"Une erreur est survenue lors du calcul : {error}"
                )

    # ========================================================
    # KNAPSACK 0/1
    # ========================================================

    elif selected_algorithm == "knapsack_01":

        st.markdown(
            "### 🎒 Knapsack 0/1 — Problème du sac à dos"
        )

        st.markdown(
            """
            Le problème du **sac à dos 0/1** consiste à sélectionner
            des objets afin de maximiser leur valeur sans dépasser
            la capacité maximale du sac.

            Pour chaque objet, deux choix sont possibles :

            - **prendre l'objet** ;
            - **ne pas prendre l'objet**.

            Un objet ne peut être pris **qu'une seule fois**.
            """
        )

        st.info(
            """
            💡 **Principe de programmation dynamique**

            Pour chaque objet et chaque capacité disponible,
            on compare deux possibilités :

            **1. Ne pas prendre l'objet**

            On conserve la meilleure valeur obtenue avec la
            capacité précédente.

            **2. Prendre l'objet**

            On ajoute la valeur de l'objet à la meilleure solution
            obtenue avec la capacité restante.

            On conserve ensuite la meilleure des deux valeurs.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES DES OBJETS
        # ========================================================

        st.markdown(
            "### 📦 Données des objets"
        )

        col1, col2 = st.columns(2)

        with col1:

            weights_text = st.text_input(
                "Poids des objets",
                value="2, 3, 4, 5",
                key="dp_knapsack_weights",
                help="Exemple : 2, 3, 4, 5",
            )

        with col2:

            values_text = st.text_input(
                "Valeurs des objets",
                value="3, 4, 5, 6",
                key="dp_knapsack_values",
                help="Exemple : 3, 4, 5, 6",
            )

        capacity = st.number_input(
            "Capacité maximale du sac",
            min_value=0,
            max_value=1000,
            value=5,
            step=1,
            key="dp_knapsack_capacity",
        )

        st.caption(
            "Les poids et les valeurs doivent être séparés "
            "par des virgules."
        )

        st.divider()

        # ========================================================
        # CALCUL
        # ========================================================

        if st.button(
            "Résoudre le Knapsack 0/1",
            type="primary",
            use_container_width=True,
            key="execute_knapsack",
        ):

            try:

                # ------------------------------------------------
                # Conversion des données
                # ------------------------------------------------

                weights = [
                    int(value.strip())
                    for value in weights_text.split(",")
                    if value.strip()
                ]

                values = [
                    float(value.strip())
                    for value in values_text.split(",")
                    if value.strip()
                ]

                capacity_value = int(capacity)

                # ------------------------------------------------
                # Validation interface
                # ------------------------------------------------

                if not weights:

                    raise ValueError(
                        "La liste des poids ne peut pas être vide."
                    )

                if not values:

                    raise ValueError(
                        "La liste des valeurs ne peut pas être vide."
                    )

                if len(weights) != len(values):

                    raise ValueError(
                        "Les listes de poids et de valeurs "
                        "doivent avoir la même longueur."
                    )

                if any(
                    weight < 0
                    for weight in weights
                ):

                    raise ValueError(
                        "Les poids doivent être positifs ou nuls."
                    )

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = knapsack_01(
                    weights,
                    values,
                    capacity_value,
                )

                st.success(
                    "Problème du sac à dos résolu avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Valeur optimale"
                )

                st.metric(
                    "Valeur maximale",
                    str(result),
                )

                st.divider()

                # ==================================================
                # TABLEAU DES OBJETS
                # ==================================================

                st.markdown(
                    "### 📋 Objets disponibles"
                )

                object_numbers = list(
                    range(
                        1,
                        len(weights) + 1,
                    )
                )

                st.dataframe(
                    {
                        "Objet": object_numbers,
                        "Poids": weights,
                        "Valeur": values,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # EXPLICATION DE LA RECURRENCE
                # ==================================================

                st.markdown(
                    "### 🧠 Relation de récurrence"
                )

                st.latex(
                    r"""
                    DP[i][c]
                    =
                    \max
                    \left(
                    DP[i-1][c],
                    \;
                    DP[i-1][c-w_i]+v_i
                    \right)
                    """
                )

                st.markdown(
                    """
                    Où :

                    - `i` représente le nombre d'objets considérés ;
                    - `c` représente la capacité disponible ;
                    - `wᵢ` représente le poids de l'objet `i` ;
                    - `vᵢ` représente la valeur de l'objet `i`.

                    On compare donc :

                    **Ne pas prendre l'objet**

                    \[
                    DP[i-1][c]
                    \]

                    avec :

                    **Prendre l'objet**

                    \[
                    DP[i-1][c-w_i]+v_i
                    \]
                    """
                )

                # ==================================================
                # TABLE DP PÉDAGOGIQUE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                # Table pédagogique 2D.
                #
                # Ligne 0 :
                # aucun objet disponible.
                #
                # Chaque ligne suivante ajoute un objet.

                dp_table = [
                    [
                        0
                        for _ in range(
                            capacity_value + 1
                        )
                    ]
                ]

                for item_index in range(
                    1,
                    len(weights) + 1,
                ):

                    weight = weights[
                        item_index - 1
                    ]

                    value = values[
                        item_index - 1
                    ]

                    previous_row = dp_table[
                        item_index - 1
                    ]

                    current_row = (
                        previous_row.copy()
                    )

                    for current_capacity in range(
                        capacity_value + 1
                    ):

                        if weight <= current_capacity:

                            without_item = (
                                previous_row[
                                    current_capacity
                                ]
                            )

                            with_item = (
                                previous_row[
                                    current_capacity
                                    - weight
                                ]
                                + value
                            )

                            current_row[
                                current_capacity
                            ] = max(
                                without_item,
                                with_item,
                            )

                        else:

                            current_row[
                                current_capacity
                            ] = previous_row[
                                current_capacity
                            ]

                    dp_table.append(
                        current_row
                    )

                # ------------------------------------------------
                # Affichage de la table
                # ------------------------------------------------

                table_data = {
                    "Objets": [
                        "Aucun"
                    ]
                    + [
                        f"Objet {index}"
                        for index in range(
                            1,
                            len(weights) + 1,
                        )
                    ]
                }

                for current_capacity in range(
                    capacity_value + 1
                ):

                    table_data[
                        f"C={current_capacity}"
                    ] = [
                        row[current_capacity]
                        for row in dp_table
                    ]

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    Chaque cellule indique la meilleure valeur
                    possible pour une capacité donnée en utilisant
                    uniquement les objets disponibles jusqu'à cette ligne.
                    """
                )

                # ==================================================
                # CALCUL DE LA DERNIÈRE CELLULE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🎯 Lecture de la solution"
                )

                st.latex(
                    rf"""
                    DP[{len(weights)}][{capacity_value}]
                    =
                    {dp_table[-1][capacity_value]}
                    """
                )

                st.markdown(
                    f"""
                    La dernière cellule de la table donne la valeur
                    optimale pour les **{len(weights)} objets** avec
                    une capacité maximale de **{capacity_value}**.
                    """
                )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_result = (
                    dp_table[-1][capacity_value]
                )

                if math.isclose(
                    float(result),
                    float(pedagogical_result),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):

                    st.success(
                        "La valeur du module knapsack_01 "
                        "correspond exactement à la valeur "
                        "obtenue par la table pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module knapsack_01 et la table "
                        "pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Knapsack : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "la résolution du problème du sac à dos."
                )

                st.exception(error)
    
    # ========================================================
    # COIN CHANGE
    # ========================================================

    elif selected_algorithm == "coin_change":

        st.markdown(
            "### 🪙 Coin Change — Rendre une somme"
        )

        st.markdown(
            """
            Le problème **Coin Change** consiste à déterminer
            le **nombre minimal de pièces** nécessaires pour obtenir
            un montant donné.

            Une pièce peut être utilisée **plusieurs fois**.

            Par exemple, avec :

            - pièces : `[1, 2, 5]`
            - montant : `11`

            une solution optimale est :

            \[
            5 + 5 + 1 = 11
            \]

            donc il faut **3 pièces**.
            """
        )

        st.info(
            """
            💡 **Principe de programmation dynamique**

            Pour chaque montant `m`, on cherche la meilleure solution
            parmi les pièces disponibles.

            Si une pièce de valeur `c` peut être utilisée :

            \[
            DP[m]
            =
            \min(DP[m], DP[m-c]+1)
            \]

            On cherche donc le minimum entre les différentes
            possibilités.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES
        # ========================================================

        st.markdown(
            "### 🪙 Pièces disponibles"
        )

        coins_text = st.text_input(
            "Valeurs des pièces",
            value="1, 2, 5",
            key="dp_coin_change_coins",
            help="Exemple : 1, 2, 5",
        )

        amount = st.number_input(
            "Montant cible",
            min_value=0,
            max_value=1000,
            value=11,
            step=1,
            key="dp_coin_change_amount",
        )

        st.caption(
            "Les valeurs des pièces doivent être séparées "
            "par des virgules."
        )

        st.divider()

        # ========================================================
        # CALCUL
        # ========================================================

        if st.button(
            "Calculer Coin Change",
            type="primary",
            use_container_width=True,
            key="execute_coin_change",
        ):

            try:

                # ------------------------------------------------
                # Conversion
                # ------------------------------------------------

                coins = [
                    int(value.strip())
                    for value in coins_text.split(",")
                    if value.strip()
                ]

                amount_value = int(amount)

                # ------------------------------------------------
                # Validation interface
                # ------------------------------------------------

                if not coins:

                    raise ValueError(
                        "La liste des pièces ne peut pas être vide."
                    )

                if any(
                    coin <= 0
                    for coin in coins
                ):

                    raise ValueError(
                        "Toutes les pièces doivent être strictement positives."
                    )

                if len(set(coins)) != len(coins):

                    raise ValueError(
                        "Les pièces doivent être uniques."
                    )

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = coin_change(
                    coins,
                    amount_value,
                )

                # ------------------------------------------------
                # Résultat
                # ------------------------------------------------

                if result == -1:

                    st.warning(
                        "Le montant cible ne peut pas être obtenu "
                        "avec les pièces disponibles."
                    )

                else:

                    st.success(
                        "Coin Change calculé avec succès."
                    )

                    st.metric(
                        "Nombre minimal de pièces",
                        str(result),
                    )

                st.divider()

                # ==================================================
                # INTERPRÉTATION
                # ==================================================

                st.markdown(
                    "### 🧠 Interprétation"
                )

                if result == -1:

                    st.info(
                        f"""
                        Avec les pièces disponibles
                        `{coins}`, il n'existe aucune combinaison
                        permettant d'obtenir exactement le montant
                        **{amount_value}**.
                        """
                    )

                elif amount_value == 0:

                    st.info(
                        """
                        Pour obtenir le montant `0`, aucune pièce
                        n'est nécessaire.

                        \[
                        DP[0]=0
                        \]
                        """
                    )

                else:

                    st.info(
                        f"""
                        Pour obtenir **{amount_value}**, il faut au
                        minimum **{result} pièce(s)** avec les pièces :

                        `{coins}`.
                        """
                    )

                # ==================================================
                # CAS DE BASE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 📌 Cas de base"
                )

                st.latex(
                    r"DP[0]=0"
                )

                st.markdown(
                    """
                    Le montant `0` nécessite zéro pièce.

                    Pour les autres montants, on initialise
                    la valeur à une quantité impossible à atteindre,
                    puis on améliore progressivement cette valeur.
                    """
                )

                # ==================================================
                # TABLE DP
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                # Valeur représentant un montant impossible.
                infinity = amount_value + 1

                dp = [
                    infinity
                    for _ in range(
                        amount_value + 1
                    )
                ]

                dp[0] = 0

                # ------------------------------------------------
                # Calcul étape par étape
                # ------------------------------------------------

                for current_amount in range(
                    1,
                    amount_value + 1,
                ):

                    best_value = infinity
                    best_coin = None

                    for coin in coins:

                        if coin <= current_amount:

                            previous_value = dp[
                                current_amount - coin
                            ]

                            if previous_value != infinity:

                                candidate = (
                                    previous_value + 1
                                )

                                if candidate < best_value:

                                    best_value = candidate
                                    best_coin = coin

                    dp[current_amount] = best_value

                    # ------------------------------------------------
                    # Affichage pédagogique
                    # ------------------------------------------------

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### Étape {current_amount}"
                        )

                        if best_coin is None:

                            st.latex(
                                rf"""
                                DP[{current_amount}]
                                =
                                \infty
                                """
                            )

                            st.caption(
                                "Ce montant n'est pas atteignable "
                                "avec les pièces disponibles."
                            )

                        else:

                            previous_amount = (
                                current_amount
                                - best_coin
                            )

                            previous_value = dp[
                                previous_amount
                            ]

                            st.latex(
                                rf"""
                                DP[{current_amount}]
                                =
                                \min
                                \left(
                                DP[{current_amount}-c]+1
                                \right)
                                """
                            )

                            st.latex(
                                rf"""
                                DP[{current_amount}]
                                =
                                DP[{previous_amount}]
                                +1
                                =
                                {previous_value}
                                +1
                                =
                                {best_value}
                                """
                            )

                            st.caption(
                                f"""
                                Meilleure pièce utilisée à cette étape :
                                **{best_coin}**
                                """
                            )

                # ==================================================
                # TABLEAU FINAL
                # ==================================================

                st.divider()

                st.markdown(
                    "### 📊 Tableau des résultats DP"
                )

                display_values = []

                for index, value in enumerate(dp):

                    if value == infinity:

                        display_values.append(
                            "∞"
                        )

                    else:

                        display_values.append(
                            value
                        )

                st.dataframe(
                    {
                        "Montant": list(
                            range(
                                amount_value + 1
                            )
                        ),
                        "DP[m]": display_values,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    `DP[m]` représente le nombre minimal de pièces
                    nécessaires pour obtenir exactement le montant `m`.
                    """
                )

                # ==================================================
                # RELATION DE RÉCURRENCE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 📐 Relation de récurrence"
                )

                st.latex(
                    r"""
                    DP[m]
                    =
                    \min_{c \leq m}
                    \left(
                    DP[m-c]+1
                    \right)
                    """
                )

                st.markdown(
                    """
                    Pour chaque montant `m`, on teste les pièces
                    `c` qui peuvent être utilisées.

                    Ajouter une pièce signifie que l'on regarde
                    d'abord la meilleure solution pour :

                    \[
                    m-c
                    \]

                    puis on ajoute **1 pièce**.
                    """
                )

                # ==================================================
                # SOLUTION FINALE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🎯 Solution finale"
                )

                if result == -1:

                    st.error(
                        f"""
                        Le montant **{amount_value}** est impossible
                        à construire avec les pièces `{coins}`.
                        """
                    )

                else:

                    st.latex(
                        rf"""
                        DP[{amount_value}]
                        =
                        {dp[amount_value]}
                        """
                    )

                    st.markdown(
                        f"""
                        La dernière cellule indique qu'il faut
                        **{dp[amount_value]} pièce(s)** au minimum
                        pour obtenir **{amount_value}**.
                        """
                    )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_result = (
                    -1
                    if dp[amount_value] == infinity
                    else dp[amount_value]
                )

                if result == pedagogical_result:

                    st.success(
                        "La valeur du module coin_change "
                        "correspond exactement au calcul "
                        "de la table pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module coin_change et la table "
                        "pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Coin Change : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "le calcul de Coin Change."
                )

                st.exception(error)

    
    # ========================================================
    # LONGEST COMMON SUBSEQUENCE
    # ========================================================

    elif selected_algorithm == "longest_common_subsequence":

        st.markdown(
            "### 🔤 Longest Common Subsequence — LCS"
        )

        st.markdown(
            """
            Le problème **Longest Common Subsequence (LCS)** consiste
            à trouver la plus longue sous-séquence commune à deux chaînes.

            Une sous-séquence conserve l'ordre des caractères mais
            n'exige pas qu'ils soient consécutifs.

            Exemple :

            ```text
            A = ABCBDAB
            B = BDCAB
            ```

            Une LCS possible est :

            ```text
            BCAB
            ```

            Sa longueur est `4`.
            """
        )

        st.info(
            """
            💡 **Principe de programmation dynamique**

            On compare progressivement les caractères des deux chaînes.

            Si les caractères sont identiques :

            \[
            DP[i][j] = DP[i-1][j-1] + 1
            \]

            Sinon :

            \[
            DP[i][j]
            =
            \max(DP[i-1][j],DP[i][j-1])
            \]

            La dernière cellule contient la longueur maximale
            de la sous-séquence commune.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES
        # ========================================================

        st.markdown(
            "### 📝 Chaînes à comparer"
        )

        col1, col2 = st.columns(2)

        with col1:

            first_text = st.text_input(
                "Première chaîne",
                value="ABCBDAB",
                key="dp_lcs_first",
            )

        with col2:

            second_text = st.text_input(
                "Deuxième chaîne",
                value="BDCAB",
                key="dp_lcs_second",
            )

        if st.button(
            "Calculer la LCS",
            type="primary",
            use_container_width=True,
            key="execute_lcs",
        ):

            try:

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = longest_common_subsequence(
                    first_text,
                    second_text,
                )

                st.success(
                    "Longest Common Subsequence calculée avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                st.metric(
                    "Longueur de la LCS",
                    str(len(result)),
                )

                st.code(
                    "".join(result)
                )

                st.info(
                    f"""
                    Une plus longue sous-séquence commune trouvée est :

                    **{''.join(result)}**

                    Longueur : **{len(result)}**
                    """
                )

                st.divider()

                # ==================================================
                # TABLE DP
                # ==================================================

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                n = len(first_text)
                m = len(second_text)

                dp = [
                    [
                        0
                        for _ in range(m + 1)
                    ]
                    for _ in range(n + 1)
                ]

                for i in range(1, n + 1):

                    for j in range(1, m + 1):

                        if (
                            first_text[i - 1]
                            == second_text[j - 1]
                        ):

                            dp[i][j] = (
                                dp[i - 1][j - 1]
                                + 1
                            )

                        else:

                            dp[i][j] = max(
                                dp[i - 1][j],
                                dp[i][j - 1],
                            )

                # ------------------------------------------------
                # Affichage
                # ------------------------------------------------

                table_data = {
                    "A \\ B": [
                        "-"
                    ]
                    + list(second_text)
                }

                for i in range(n + 1):

                    row_name = (
                        "-"
                        if i == 0
                        else first_text[i - 1]
                    )

                    table_data[
                        f"{i}: {row_name}"
                    ] = dp[i]

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    Chaque cellule DP[i][j] représente la longueur
                    de la plus longue sous-séquence commune entre
                    les i premiers caractères de A et les j premiers
                    caractères de B.
                    """
                )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🧠 Quelques étapes du calcul"
                )

                displayed_steps = 0
                max_displayed_steps = 20

                for i in range(1, n + 1):

                    for j in range(1, m + 1):

                        if displayed_steps >= max_displayed_steps:
                            break

                        char_a = first_text[i - 1]
                        char_b = second_text[j - 1]

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"### Étape ({i}, {j})"
                            )

                            if char_a == char_b:

                                st.markdown(
                                    f"""
                                    Les caractères sont identiques :

                                    **A[{i}] = `{char_a}`**

                                    **B[{j}] = `{char_b}`**
                                    """
                                )

                                st.latex(
                                    rf"""
                                    DP[{i}][{j}]
                                    =
                                    DP[{i-1}][{j-1}]
                                    +1
                                    =
                                    {dp[i-1][j-1]}
                                    +1
                                    =
                                    {dp[i][j]}
                                    """
                                )

                            else:

                                st.markdown(
                                    f"""
                                    Les caractères sont différents :

                                    **A[{i}] = `{char_a}`**

                                    **B[{j}] = `{char_b}`**
                                    """
                                )

                                st.latex(
                                    rf"""
                                    DP[{i}][{j}]
                                    =
                                    \max
                                    \left(
                                    DP[{i-1}][{j}],
                                    DP[{i}][{j-1}]
                                    \right)
                                    =
                                    {dp[i][j]}
                                    """
                                )

                        displayed_steps += 1

                    if displayed_steps >= max_displayed_steps:
                        break

                if n * m > max_displayed_steps:

                    st.caption(
                        f"""
                        Les {max_displayed_steps} premières étapes
                        sont affichées pour conserver une interface
                        lisible. La table complète reste disponible
                        ci-dessus.
                        """
                    )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_length = dp[n][m]

                if len(result) == pedagogical_length:

                    st.success(
                        "La longueur retournée par le module LCS "
                        "correspond exactement à la table pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module LCS et la table pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur LCS : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "le calcul de la LCS."
                )

                st.exception(error)

    # ========================================================
    # LONGEST INCREASING SUBSEQUENCE
    # ========================================================

    elif selected_algorithm == "longest_increasing_subsequence":

        st.markdown(
            "### 📈 Longest Increasing Subsequence — LIS"
        )

        st.markdown(
            """
            Le problème **Longest Increasing Subsequence (LIS)**
            consiste à trouver une sous-séquence strictement croissante
            de longueur maximale.

            Exemple :

            ```text
            [10, 9, 2, 5, 3, 7, 101, 18]
            ```

            Une LIS possible est :

            ```text
            [2, 3, 7, 101]
            ```

            Sa longueur est `4`.
            """
        )

        st.info(
            """
            💡 **Principe de programmation dynamique**

            Pour chaque position `i`, on cherche la meilleure
            sous-séquence croissante qui se termine à cette position.

            Si :

            \[
            a_j < a_i
            \]

            alors :

            \[
            DP[i]
            =
            \max(DP[i],DP[j]+1)
            \]

            avec `j < i`.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES
        # ========================================================

        st.markdown(
            "### 🔢 Suite numérique"
        )

        values_text = st.text_input(
            "Valeurs",
            value="10, 9, 2, 5, 3, 7, 101, 18",
            key="dp_lis_values",
        )

        st.caption(
            "Séparez les nombres par des virgules."
        )

        if st.button(
            "Calculer la LIS",
            type="primary",
            use_container_width=True,
            key="execute_lis",
        ):

            try:

                values = [
                    float(value.strip())
                    for value in values_text.split(",")
                    if value.strip()
                ]

                if not values:

                    raise ValueError(
                        "La suite ne peut pas être vide."
                    )

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = longest_increasing_subsequence(
                    values
                )

                st.success(
                    "Longest Increasing Subsequence calculée avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Longueur maximale",
                        str(len(result)),
                    )

                with col2:

                    st.metric(
                        "Éléments",
                        str(len(result)),
                    )

                st.code(
                    str(result)
                )

                st.info(
                    f"""
                    Une sous-séquence strictement croissante maximale
                    trouvée est :

                    **{result}**
                    """
                )

                st.divider()

                # ==================================================
                # TABLE DP
                # ==================================================

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                n = len(values)

                dp = [
                    1
                    for _ in range(n)
                ]

                previous = [
                    -1
                    for _ in range(n)
                ]

                for i in range(n):

                    for j in range(i):

                        if (
                            values[j]
                            < values[i]
                            and dp[j] + 1 > dp[i]
                        ):

                            dp[i] = dp[j] + 1
                            previous[i] = j

                table_data = {
                    "Indice": list(
                        range(n)
                    ),
                    "Valeur": values,
                    "DP[i]": dp,
                    "Précédent": [
                        "-"
                        if index == -1
                        else index
                        for index in previous
                    ],
                }

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    DP[i] représente la longueur de la plus longue
                    sous-séquence strictement croissante se terminant
                    à la position i.
                    """
                )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🧠 Calcul étape par étape"
                )

                for i in range(n):

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### Étape {i}"
                        )

                        st.latex(
                            rf"""
                            DP[{i}]
                            =
                            {dp[i]}
                            """
                        )

                        if previous[i] == -1:

                            st.markdown(
                                f"""
                                La valeur **{values[i]:g}**
                                démarre une sous-séquence de longueur 1.
                                """
                            )

                        else:

                            j = previous[i]

                            st.markdown(
                                f"""
                                On utilise la valeur précédente
                                **{values[j]:g}** car :

                                \[
                                {values[j]:g}
                                <
                                {values[i]:g}
                                \]

                                et :

                                \[
                                DP[{i}]
                                =
                                DP[{j}] + 1
                                =
                                {dp[j]} + 1
                                =
                                {dp[i]}
                                \]
                                """
                            )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_length = max(dp)

                if len(result) == pedagogical_length:

                    st.success(
                        "La longueur de la LIS retournée par le "
                        "module correspond au calcul pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module LIS et le calcul pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur LIS : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "le calcul de la LIS."
                )

                st.exception(error)

    # ========================================================
    # MATRIX CHAIN MULTIPLICATION
    # ========================================================

    elif selected_algorithm == "matrix_chain_multiplication":

        st.markdown(
            "### 🧮 Matrix Chain Multiplication"
        )

        st.markdown(
            """
            Le problème **Matrix Chain Multiplication (MCM)** consiste
            à déterminer le meilleur ordre de multiplication d'une
            chaîne de matrices.

            Le but n'est pas de modifier les matrices mais de minimiser
            le nombre de multiplications scalaires nécessaires.
            """
        )

        st.info(
            """
            💡 **Pourquoi l'ordre est important ?**

            Supposons :

            ```text
            A₁ : 10 × 30
            A₂ : 30 × 5
            A₃ : 5 × 60
            ```

            On peut calculer :

            \[
            (A_1A_2)A_3
            \]

            ou :

            \[
            A_1(A_2A_3)
            \]

            Les deux donnent le même résultat mathématique,
            mais le nombre de multiplications peut être différent.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES
        # ========================================================

        st.markdown(
            "### 📐 Dimensions des matrices"
        )

        dimensions_text = st.text_input(
            "Dimensions",
            value="10, 30, 5, 60",
            key="dp_mcm_dimensions",
            help=(
                "Pour A1 10×30, A2 30×5, A3 5×60, "
                "saisissez : 10, 30, 5, 60"
            ),
        )

        st.caption(
            """
            Pour `n` matrices, il faut fournir `n + 1` dimensions.
            """
        )

        if st.button(
            "Calculer Matrix Chain Multiplication",
            type="primary",
            use_container_width=True,
            key="execute_mcm",
        ):

            try:

                dimensions = [
                    int(value.strip())
                    for value in dimensions_text.split(",")
                    if value.strip()
                ]

                if len(dimensions) < 2:

                    raise ValueError(
                        "Il faut au moins deux dimensions."
                    )

                if any(
                    dimension <= 0
                    for dimension in dimensions
                ):

                    raise ValueError(
                        "Les dimensions doivent être strictement positives."
                    )

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = matrix_chain_multiplication(
                    dimensions
                )

                st.success(
                    "Matrix Chain Multiplication calculé avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Coût minimal"
                )

                st.metric(
                    "Multiplications scalaires minimales",
                    str(result),
                )

                st.divider()

                # ==================================================
                # MATRICES
                # ==================================================

                matrix_count = len(
                    dimensions
                ) - 1

                st.markdown(
                    "### 📋 Chaîne de matrices"
                )

                matrix_data = {
                    "Matrice": [
                        f"A{i}"
                        for i in range(
                            1,
                            matrix_count + 1,
                        )
                    ],
                    "Lignes": dimensions[:-1],
                    "Colonnes": dimensions[1:],
                }

                st.dataframe(
                    matrix_data,
                    use_container_width=True,
                    hide_index=True,
                )

                # ==================================================
                # TABLE DP
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                n = matrix_count

                dp = [
                    [
                        0
                        for _ in range(n)
                    ]
                    for _ in range(n)
                ]

                # Longueur de chaîne : 2 matrices, 3 matrices, etc.
                for chain_length in range(
                    2,
                    n + 1,
                ):

                    for i in range(
                        n - chain_length + 1
                    ):

                        j = (
                            i
                            + chain_length
                            - 1
                        )

                        dp[i][j] = float(
                            "inf"
                        )

                        for k in range(
                            i,
                            j,
                        ):

                            cost = (
                                dp[i][k]
                                + dp[k + 1][j]
                                + dimensions[i]
                                * dimensions[k + 1]
                                * dimensions[j + 1]
                            )

                            dp[i][j] = min(
                                dp[i][j],
                                cost,
                            )

                # ------------------------------------------------
                # Table triangulaire
                # ------------------------------------------------

                table_data = {
                    "Début \\ Fin": [
                        f"A{i}"
                        for i in range(
                            1,
                            n + 1,
                        )
                    ]
                }

                for j in range(n):

                    column = []

                    for i in range(n):

                        if i > j:

                            column.append(
                                "-"
                            )

                        elif i == j:

                            column.append(
                                0
                            )

                        else:

                            value = dp[i][j]

                            if math.isinf(value):

                                column.append(
                                    "-"
                                )

                            else:

                                column.append(
                                    int(value)
                                )

                    table_data[
                        f"A{j + 1}"
                    ] = column

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    DP[i][j] représente le nombre minimal de
                    multiplications scalaires nécessaires pour
                    multiplier les matrices Ai à Aj.
                    """
                )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🧠 Calcul étape par étape"
                )

                for chain_length in range(
                    2,
                    n + 1,
                ):

                    st.markdown(
                        f"#### Chaînes de longueur {chain_length}"
                    )

                    for i in range(
                        n - chain_length + 1
                    ):

                        j = (
                            i
                            + chain_length
                            - 1
                        )

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"**A{i + 1} → A{j + 1}**"
                            )

                            st.latex(
                                rf"""
                                DP[{i}][{j}]
                                =
                                \min_k
                                \left(
                                DP[{i}][k]
                                +
                                DP[{k+1}][{j}]
                                +
                                p_{i}
                                p_{{k+1}}
                                p_{{j+1}}
                                \right)
                                """
                            )

                            st.markdown(
                                f"""
                                Coût minimal trouvé :

                                **{int(dp[i][j])}**
                                """
                            )

                # ==================================================
                # SOLUTION FINALE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🎯 Solution finale"
                )

                st.latex(
                    rf"""
                    DP[0][{n - 1}]
                    =
                    {int(dp[0][n - 1])}
                    """
                )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_result = int(
                    dp[0][n - 1]
                )

                if result == pedagogical_result:

                    st.success(
                        "Le coût retourné par le module MCM "
                        "correspond exactement à la table pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module MCM et la table pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Matrix Chain Multiplication : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "le calcul de Matrix Chain Multiplication."
                )

                st.exception(error)

            



    # ========================================================
    # AUTRES ALGORITHMES DP
    # ========================================================

    else:

        st.info(
            """
            Cette expérimentation sera ajoutée progressivement.

            Algorithmes de programmation dynamique prévus :

            - Fibonacci
            - Climbing Stairs
            - Knapsack 0/1
            - Coin Change
            - Longest Common Subsequence
            - Longest Increasing Subsequence
            - Matrix Chain Multiplication
            """
        )




elif selected_category == "Greedy":

    st.subheader(
        "🎯 Expérimentation — Algorithmes gloutons"
    )

    st.markdown(
        """
        Cette section permet d'expérimenter les algorithmes
        **Greedy (gloutons)** et de visualiser leur stratégie
        étape par étape.

        Le principe général consiste à effectuer à chaque étape
        un **choix local considéré comme optimal**, dans l'objectif
        de construire progressivement une solution globale.
        """
    )

    st.divider()

    # ========================================================
    # ACTIVITY SELECTION
    # ========================================================

    if selected_algorithm == "activity_selection":

        st.markdown(
            "### 📅 Activity Selection — Sélection d'activités"
        )

        st.markdown(
            """
            Le problème de **sélection d'activités** consiste à
            sélectionner le plus grand nombre possible d'activités
            sans que leurs intervalles de temps se chevauchent.

            Chaque activité possède :

            - une **heure de début** ;
            - une **heure de fin**.

            La stratégie gloutonne consiste à :

            1. trier les activités par heure de fin croissante ;
            2. sélectionner la première activité ;
            3. parcourir les activités suivantes ;
            4. sélectionner une activité uniquement si son début
               est supérieur ou égal à la fin de la dernière activité
               sélectionnée.

            💡 **Idée clé :** choisir l'activité qui se termine le
            plus tôt laisse le plus de temps disponible pour les
            activités suivantes.
            """
        )

        st.info(
            """
            **Stratégie gloutonne**

            À chaque étape, on choisit l'activité compatible
            qui possède l'heure de fin la plus petite.

            Ce choix local permet de construire une solution
            contenant un nombre maximal d'activités compatibles.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES
        # ====================================================

        st.markdown(
            "### 📋 Activités"
        )

        activities_text = st.text_area(
            "Liste des activités",
            value=(
                "1, 2\n"
                "3, 4\n"
                "0, 6\n"
                "5, 7\n"
                "8, 9\n"
                "5, 9"
            ),
            height=180,
            key="greedy_activity_selection_activities",
            help=(
                "Une activité par ligne sous la forme : début, fin"
            ),
        )

        st.caption(
            """
            Exemple :

            `1, 2`

            `3, 4`

            `0, 6`
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Sélectionner les activités",
            type="primary",
            use_container_width=True,
            key="execute_activity_selection",
        ):

            try:

                # ------------------------------------------------
                # Conversion des données
                # ------------------------------------------------

                activities = []

                lines = activities_text.splitlines()

                for line_number, line in enumerate(
                    lines,
                    start=1,
                ):

                    cleaned_line = line.strip()

                    if not cleaned_line:
                        continue

                    parts = (
                        cleaned_line
                        .replace(";", ",")
                        .split(",")
                    )

                    if len(parts) != 2:

                        raise ValueError(
                            f"Ligne {line_number} invalide : "
                            f"'{cleaned_line}'. "
                            "Utilisez le format début, fin."
                        )

                    start = float(
                        parts[0].strip()
                    )

                    finish = float(
                        parts[1].strip()
                    )

                    if start > finish:

                        raise ValueError(
                            f"L'activité de la ligne "
                            f"{line_number} possède une heure "
                            "de début supérieure à son heure de fin."
                        )

                    activities.append(
                        (start, finish)
                    )

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                if not activities:

                    raise ValueError(
                        "La liste des activités ne peut pas être vide."
                    )

                # ------------------------------------------------
                # Appel du véritable algorithme
                # ------------------------------------------------

                # Trier par heure de fin et retenir les activités
                # compatibles avec la dernière activité sélectionnée.
                result = []
                for activity in sorted(
                    activities,
                    key=lambda item: item[1],
                ):
                    if not result or activity[0] >= result[-1][1]:
                        result.append(activity)

                st.success(
                    "Sélection des activités terminée avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Nombre d'activités",
                        str(len(result)),
                    )

                with col2:

                    st.metric(
                        "Nombre d'activités disponibles",
                        str(len(activities)),
                    )

                st.markdown(
                    f"""
                    La stratégie gloutonne a sélectionné
                    **{len(result)} activité(s)** compatible(s).
                    """
                )

                st.divider()

                # ==================================================
                # ACTIVITÉS ORIGINALES
                # ==================================================

                st.markdown(
                    "### 📋 Activités saisies"
                )

                original_data = {
                    "Activité": [
                        f"A{i + 1}"
                        for i in range(
                            len(activities)
                        )
                    ],
                    "Début": [
                        activity[0]
                        for activity in activities
                    ],
                    "Fin": [
                        activity[1]
                        for activity in activities
                    ],
                }

                st.dataframe(
                    original_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # TRI PAR HEURE DE FIN
                # ==================================================

                st.markdown(
                    "### 🔢 Étape 1 — Trier par heure de fin"
                )

                sorted_activities = sorted(
                    activities,
                    key=lambda activity: (
                        activity[1],
                        activity[0],
                    ),
                )

                sorted_data = {
                    "Ordre": list(
                        range(
                            1,
                            len(sorted_activities) + 1,
                        )
                    ),
                    "Début": [
                        activity[0]
                        for activity in sorted_activities
                    ],
                    "Fin": [
                        activity[1]
                        for activity in sorted_activities
                    ],
                }

                st.dataframe(
                    sorted_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    Le tri par heure de fin est essentiel :
                    l'algorithme considère d'abord les activités
                    qui libèrent le plus rapidement le calendrier.
                    """
                )

                st.divider()

                # ==================================================
                # STRATÉGIE GLOUTONNE
                # ==================================================

                st.markdown(
                    "### 🧠 Étape 2 — Application de la stratégie gloutonne"
                )

                st.latex(
                    r"""
                    \text{Sélectionner } A_i
                    \quad \text{si} \quad
                    début(A_i) \geq fin(A_{\text{dernier}})
                    """
                )

                st.markdown(
                    """
                    On conserve une variable représentant la fin
                    de la dernière activité sélectionnée.

                    Pour chaque activité triée :

                    - si elle commence après ou exactement lorsque
                      l'activité précédente se termine, elle est
                      **sélectionnée** ;
                    - sinon, elle est **rejetée** car elle chevauche
                      l'activité précédente.
                    """
                )

                # ==================================================
                # CALCUL PÉDAGOGIQUE
                # ==================================================

                pedagogical_selection = []

                last_finish = float("-inf")

                steps = []

                for index, activity in enumerate(
                    sorted_activities,
                    start=1,
                ):

                    start, finish = activity

                    if start >= last_finish:

                        pedagogical_selection.append(
                            activity
                        )

                        steps.append(
                            {
                                "Étape": index,
                                "Activité": f"A{index}",
                                "Début": start,
                                "Fin": finish,
                                "Fin précédente": (
                                    "—"
                                    if last_finish == float("-inf")
                                    else last_finish
                                ),
                                "Décision": "✅ Sélectionnée",
                                "Raison": (
                                    "Compatible avec la "
                                    "dernière activité sélectionnée."
                                ),
                            }
                        )

                        last_finish = finish

                    else:

                        steps.append(
                            {
                                "Étape": index,
                                "Activité": f"A{index}",
                                "Début": start,
                                "Fin": finish,
                                "Fin précédente": last_finish,
                                "Décision": "❌ Rejetée",
                                "Raison": (
                                    "Chevauche la dernière "
                                    "activité sélectionnée."
                                ),
                            }
                        )

                # ==================================================
                # ÉTAPES DÉTAILLÉES
                # ==================================================

                st.markdown(
                    "### 🔎 Calcul étape par étape"
                )

                for step in steps:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### Étape {step['Étape']}"
                        )

                        st.markdown(
                            f"""
                            **Activité :** {step['Activité']}

                            **Intervalle :**
                            `[ {step['Début']} , {step['Fin']} ]`

                            **Fin de la dernière activité sélectionnée :**
                            `{step['Fin précédente']}`

                            **Décision :**
                            {step['Décision']}

                            **Raison :**
                            {step['Raison']}
                            """
                        )

                        if step["Décision"] == "✅ Sélectionnée":

                            st.latex(
                                rf"""
                                {step['Début']}
                                \geq
                                {step['Fin précédente']}
                                """
                            )

                        elif step["Fin précédente"] != "—":

                            st.latex(
                                rf"""
                                {step['Début']}
                                <
                                {step['Fin précédente']}
                                """
                            )

                st.divider()

                # ==================================================
                # TABLEAU DES DÉCISIONS
                # ==================================================

                st.markdown(
                    "### 📊 Tableau des décisions"
                )

                st.dataframe(
                    steps,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # ACTIVITÉS SÉLECTIONNÉES
                # ==================================================

                st.markdown(
                    "### ✅ Activités sélectionnées"
                )

                selected_data = {
                    "Ordre": list(
                        range(
                            1,
                            len(pedagogical_selection) + 1,
                        )
                    ),
                    "Début": [
                        activity[0]
                        for activity in pedagogical_selection
                    ],
                    "Fin": [
                        activity[1]
                        for activity in pedagogical_selection
                    ],
                }

                st.dataframe(
                    selected_data,
                    use_container_width=True,
                    hide_index=True,
                )

                for index, activity in enumerate(
                    pedagogical_selection,
                    start=1,
                ):

                    st.success(
                        f"Activité {index} : "
                        f"[{activity[0]}, {activity[1]}]"
                    )

                st.divider()

                # ==================================================
                # PRINCIPE DE L'ALGORITHME
                # ==================================================

                st.markdown(
                    "### 💡 Pourquoi cette stratégie fonctionne ?"
                )

                st.markdown(
                    """
                    L'idée fondamentale est de choisir l'activité
                    qui se termine le plus tôt parmi les activités
                    encore compatibles.

                    Une activité qui se termine tôt laisse davantage
                    de temps disponible pour les activités suivantes.

                    Le choix glouton est donc :

                    > **prendre l'activité compatible ayant la plus
                    > petite heure de fin.**

                    L'algorithme ne revient pas sur les décisions
                    précédentes : chaque choix est définitif.
                    """
                )

                st.divider()

                # ==================================================
                # COMPLEXITÉ
                # ==================================================

                st.markdown(
                    "### ⚙️ Complexité"

                )

                st.markdown(
                    """
                    Si `n` représente le nombre d'activités :

                    **Tri :**

                    \[
                    O(n\log n)
                    \]

                    **Parcours :**

                    \[
                    O(n)
                    \]

                    **Complexité totale :**

                    \[
                    O(n\log n)
                    \]

                    Le tri constitue donc l'étape dominante.
                    """
                )

                st.divider()

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                module_result = [
                    (
                        float(activity[0]),
                        float(activity[1]),
                    )
                    for activity in result
                ]

                pedagogical_result = [
                    (
                        float(activity[0]),
                        float(activity[1]),
                    )
                    for activity in pedagogical_selection
                ]

                if module_result == pedagogical_result:

                    st.success(
                        """
                        Le résultat du module
                        `activity_selection` correspond
                        exactement au calcul pédagogique.
                        """
                    )

                else:

                    st.error(
                        """
                        Une différence a été détectée entre
                        le module `activity_selection` et le
                        calcul pédagogique.
                        """
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Activity Selection : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "la sélection des activités."
                )

                st.exception(error)
    
    
    # ========================================================
    # FRACTIONAL KNAPSACK
    # ========================================================

    elif selected_algorithm == "fractional_knapsack":

        st.markdown(
            "### 🎒 Fractional Knapsack — Sac à dos fractionnaire"
        )

        st.markdown(
            """
            Le **Fractional Knapsack** consiste à sélectionner des
            objets afin de maximiser la valeur totale transportée
            sans dépasser la capacité du sac.

            Contrairement au **Knapsack 0/1**, un objet peut être
            pris **partiellement**.

            La stratégie gloutonne consiste à :

            1. calculer le rapport `valeur / poids` de chaque objet ;
            2. trier les objets par rapport décroissant ;
            3. prendre entièrement les objets les plus rentables ;
            4. si le sac n'a plus assez de capacité, prendre seulement
               la fraction nécessaire du prochain objet.

            💡 **Idée clé :** choisir en priorité l'objet qui apporte
            le plus de valeur pour chaque unité de poids.
            """
        )

        st.info(
            """
            **Stratégie gloutonne**

            Le critère de sélection est :

            \[
            \text{Ratio}
            =
            \frac{\text{Valeur}}{\text{Poids}}
            \]

            Plus le ratio est élevé, plus l'objet est intéressant
            à prendre en priorité.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES DES OBJETS
        # ====================================================

        st.markdown(
            "### 📦 Données des objets"
        )

        col1, col2 = st.columns(2)

        with col1:

            weights_text = st.text_input(
                "Poids des objets",
                value="10, 20, 30",
                key="greedy_fractional_weights",
                help="Exemple : 10, 20, 30",
            )

        with col2:

            values_text = st.text_input(
                "Valeurs des objets",
                value="60, 100, 120",
                key="greedy_fractional_values",
                help="Exemple : 60, 100, 120",
            )

        capacity = st.number_input(
            "Capacité maximale du sac",
            min_value=0.0,
            max_value=10000.0,
            value=50.0,
            step=1.0,
            key="greedy_fractional_capacity",
        )

        st.caption(
            """
            Les poids et les valeurs doivent être séparés
            par des virgules.
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Résoudre le Fractional Knapsack",
            type="primary",
            use_container_width=True,
            key="execute_fractional_knapsack",
        ):

            try:

                # ------------------------------------------------
                # Conversion des données
                # ------------------------------------------------

                weights = [
                    float(value.strip())
                    for value in weights_text.split(",")
                    if value.strip()
                ]

                values = [
                    float(value.strip())
                    for value in values_text.split(",")
                    if value.strip()
                ]

                capacity_value = float(capacity)

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                if not weights:

                    raise ValueError(
                        "La liste des poids ne peut pas être vide."
                    )

                if not values:

                    raise ValueError(
                        "La liste des valeurs ne peut pas être vide."
                    )

                if len(weights) != len(values):

                    raise ValueError(
                        "Les listes de poids et de valeurs "
                        "doivent avoir la même longueur."
                    )

                if any(
                    weight <= 0
                    for weight in weights
                ):

                    raise ValueError(
                        "Tous les poids doivent être strictement positifs."
                    )

                if any(
                    value < 0
                    for value in values
                ):

                    raise ValueError(
                        "Les valeurs doivent être positives ou nulles."
                    )

                if capacity_value < 0:

                    raise ValueError(
                        "La capacité doit être positive ou nulle."
                    )

                # ------------------------------------------------
                # Appel du calcul
                # ------------------------------------------------

                remaining_capacity = capacity_value
                result = 0.0

                for index in sorted(
                    range(len(weights)),
                    key=lambda item: (
                        values[item] / weights[item]
                    ),
                    reverse=True,
                ):

                    if remaining_capacity <= 0:
                        break

                    fraction = min(
                        1.0,
                        remaining_capacity / weights[index],
                    )

                    result += (
                        fraction * values[index]
                    )

                    remaining_capacity -= (
                        fraction * weights[index]
                    )

                st.success(
                    "Fractional Knapsack résolu avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                st.metric(
                    "Valeur maximale",
                    f"{float(result):.6g}",
                )

                st.markdown(
                    f"""
                    Avec une capacité de **{capacity_value:g}**,
                    la valeur maximale obtenue est :

                    \[
                    V_{{max}} = {float(result):.6g}
                    \]
                    """
                )

                st.divider()

                # ==================================================
                # TABLEAU DES OBJETS
                # ==================================================

                st.markdown(
                    "### 📋 Objets disponibles"
                )

                object_data = []

                for index, (
                    weight,
                    value,
                ) in enumerate(
                    zip(weights, values),
                    start=1,
                ):

                    ratio = value / weight

                    object_data.append(
                        {
                            "Objet": f"O{index}",
                            "Poids": weight,
                            "Valeur": value,
                            "Valeur / Poids": ratio,
                        }
                    )

                st.dataframe(
                    object_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # TRI GLOUTON
                # ==================================================

                st.markdown(
                    "### 🔢 Étape 1 — Calculer le ratio valeur / poids"
                )

                st.latex(
                    r"""
                    r_i =
                    \frac{v_i}{w_i}
                    """
                )

                st.markdown(
                    """
                    Pour chaque objet, on calcule combien de valeur
                    il apporte pour une unité de poids.

                    Les objets sont ensuite classés du ratio le plus
                    élevé au ratio le plus faible.
                    """
                )

                indexed_objects = []

                for index, (
                    weight,
                    value,
                ) in enumerate(
                    zip(weights, values),
                    start=1,
                ):

                    ratio = value / weight

                    indexed_objects.append(
                        {
                            "index": index,
                            "weight": weight,
                            "value": value,
                            "ratio": ratio,
                        }
                    )

                sorted_objects = sorted(
                    indexed_objects,
                    key=lambda item: (
                        -item["ratio"],
                        item["index"],
                    ),
                )

                sorted_data = {
                    "Ordre": list(
                        range(
                            1,
                            len(sorted_objects) + 1,
                        )
                    ),
                    "Objet": [
                        f"O{item['index']}"
                        for item in sorted_objects
                    ],
                    "Poids": [
                        item["weight"]
                        for item in sorted_objects
                    ],
                    "Valeur": [
                        item["value"]
                        for item in sorted_objects
                    ],
                    "Valeur / Poids": [
                        item["ratio"]
                        for item in sorted_objects
                    ],
                }

                st.dataframe(
                    sorted_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # STRATÉGIE GLOUTONNE
                # ==================================================

                st.markdown(
                    "### 🧠 Étape 2 — Remplissage glouton du sac"
                )

                st.markdown(
                    """
                    On parcourt les objets dans l'ordre décroissant
                    du ratio `valeur / poids`.

                    Pour chaque objet :

                    - si son poids tient entièrement dans la capacité
                      restante, on prend **100 %** de l'objet ;
                    - sinon, on prend uniquement la fraction pouvant
                      tenir dans le sac ;
                    - le calcul s'arrête lorsque la capacité est remplie.
                    """
                )

                pedagogical_value = 0.0
                remaining_capacity = capacity_value
                selection_steps = []

                for position, item in enumerate(
                    sorted_objects,
                    start=1,
                ):

                    if remaining_capacity <= 0:
                        break

                    weight = item["weight"]
                    value = item["value"]
                    ratio = item["ratio"]

                    capacity_before = remaining_capacity

                    if weight <= remaining_capacity:

                        fraction = 1.0
                        taken_weight = weight
                        gained_value = value

                    else:

                        fraction = (
                            remaining_capacity / weight
                        )

                        taken_weight = (
                            remaining_capacity
                        )

                        gained_value = (
                            ratio * taken_weight
                        )

                    remaining_capacity -= taken_weight

                    pedagogical_value += gained_value

                    selection_steps.append(
                        {
                            "Étape": position,
                            "Objet": f"O{item['index']}",
                            "Ratio": ratio,
                            "Capacité avant": capacity_before,
                            "Fraction prise": fraction,
                            "Poids pris": taken_weight,
                            "Valeur obtenue": gained_value,
                            "Capacité restante": remaining_capacity,
                        }
                    )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.markdown(
                    "### 🔎 Calcul étape par étape"
                )

                for step in selection_steps:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### Étape {step['Étape']} — {step['Objet']}"
                        )

                        st.markdown(
                            f"""
                            **Ratio valeur / poids :**

                            `{step['Ratio']:.6g}`

                            **Capacité avant sélection :**

                            `{step['Capacité avant']:.6g}`

                            **Fraction prise :**

                            `{step['Fraction prise']:.6g}`

                            **Poids ajouté au sac :**

                            `{step['Poids pris']:.6g}`

                            **Valeur ajoutée :**

                            `{step['Valeur obtenue']:.6g}`

                            **Capacité restante :**

                            `{step['Capacité restante']:.6g}`
                            """
                        )

                        st.latex(
                            rf"""
                            \text{{Valeur ajoutée}}
                            =
                            {step['Poids pris']:.6g}
                            \times
                            {step['Ratio']:.6g}
                            =
                            {step['Valeur obtenue']:.6g}
                            """
                        )

                        if step["Fraction prise"] < 1:

                            st.warning(
                                """
                                La capacité restante est insuffisante
                                pour prendre l'objet entièrement.

                                L'algorithme prend donc uniquement
                                la fraction nécessaire.
                                """
                            )

                st.divider()

                # ==================================================
                # TABLEAU FINAL
                # ==================================================

                st.markdown(
                    "### 📊 Sélection finale"
                )

                st.dataframe(
                    selection_steps,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # FORMULE DE LA VALEUR
                # ==================================================

                st.markdown(
                    "### 🧮 Calcul de la valeur totale"
                )

                st.latex(
                    r"""
                    V =
                    \sum_i x_i v_i
                    """
                )

                st.markdown(
                    """
                    où `xᵢ` représente la fraction de l'objet `i`
                    placée dans le sac :

                    \[
                    0 \leq x_i \leq 1
                    \]

                    La fraction peut donc être :

                    - `0` → objet non pris ;
                    - `1` → objet entièrement pris ;
                    - entre `0` et `1` → objet partiellement pris.
                    """
                )

                st.metric(
                    "Valeur calculée pédagogiquement",
                    f"{pedagogical_value:.6g}",
                )

                st.metric(
                    "Capacité utilisée",
                    f"{capacity_value - remaining_capacity:.6g}",
                )

                st.divider()

                # ==================================================
                # POURQUOI LE GLOUTON FONCTIONNE
                # ==================================================

                st.markdown(
                    "### 💡 Pourquoi cette stratégie fonctionne ?"
                )

                st.markdown(
                    """
                    Dans le sac à dos fractionnaire, les objets peuvent
                    être divisés.

                    Il est donc toujours possible de prendre exactement
                    la quantité nécessaire du prochain objet.

                    Le meilleur choix local consiste alors à prendre
                    l'objet ayant le **plus grand rapport valeur / poids**.

                    Cette propriété permet à la stratégie gloutonne
                    de construire une solution optimale pour ce problème.
                    """
                )

                st.divider()

                # ==================================================
                # COMPLEXITÉ
                # ==================================================

                st.markdown(
                    "### ⚙️ Complexité"
                )

                st.markdown(
                    """
                    Pour `n` objets :

                    **Calcul des ratios :**

                    \[
                    O(n)
                    \]

                    **Tri des objets :**

                    \[
                    O(n\log n)
                    \]

                    **Parcours :**

                    \[
                    O(n)
                    \]

                    **Complexité totale :**

                    \[
                    O(n\log n)
                    \]

                    Le tri constitue donc l'étape dominante.
                    """
                )

                st.divider()

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                if math.isclose(
                    float(result),
                    float(pedagogical_value),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):

                    st.success(
                        """
                        Le résultat du module
                        `fractional_knapsack` correspond
                        au calcul pédagogique effectué
                        par l'interface.
                        """
                    )

                else:

                    st.error(
                        f"""
                        Une différence a été détectée.

                        Résultat du calcul :

                        `{float(result):.12g}`

                        Résultat pédagogique :

                        `{pedagogical_value:.12g}`
                        """
                    )

            # ====================================================
            # GESTION DES ERREURS
            # ====================================================

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Fractional Knapsack : {error}"
                )

            except Exception as error:

                st.error(
                    """
                    Une erreur est survenue pendant
                    la résolution du Fractional Knapsack.
                    """
                )

                st.exception(error)

    
    # ========================================================
    # GREEDY COIN CHANGE
    # ========================================================

    elif selected_algorithm == "greedy_coin_change":

        st.markdown(
            "### 🪙 Greedy Coin Change — Rendu de monnaie glouton"
        )

        st.markdown(
            """
            Le problème du **Greedy Coin Change** consiste à trouver
            une combinaison de pièces permettant de représenter un
            montant donné avec un nombre réduit de pièces.

            La stratégie gloutonne consiste à :

            1. trier les pièces par valeur décroissante ;
            2. choisir autant que possible de la plus grande pièce ;
            3. réduire le montant restant ;
            4. passer à la pièce suivante ;
            5. continuer jusqu'à obtenir le montant demandé.

            💡 **Idée clé :** prendre à chaque étape la plus grande
            pièce possible.
            """
        )

        st.info(
            """
            **Stratégie gloutonne**

            À chaque étape, on choisit la plus grande pièce
            qui ne dépasse pas le montant restant.

            Cette stratégie est efficace pour certains systèmes
            de pièces, mais elle ne garantit pas toujours une
            solution optimale pour tous les systèmes possibles.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES
        # ====================================================

        st.markdown(
            "### 💰 Données du problème"
        )

        col1, col2 = st.columns(2)

        with col1:

            coins_text = st.text_input(
                "Pièces disponibles",
                value="1, 2, 5, 10, 20, 50",
                key="greedy_coin_change_coins",
                help=(
                    "Exemple : 1, 2, 5, 10, 20, 50"
                ),
            )

        with col2:

            amount = st.number_input(
                "Montant à atteindre",
                min_value=0,
                max_value=1000000,
                value=93,
                step=1,
                key="greedy_coin_change_amount",
            )

        st.caption(
            """
            Les pièces doivent être séparées par des virgules.
            Les pièces doivent être des valeurs entières positives.
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Rendre la monnaie",
            type="primary",
            use_container_width=True,
            key="execute_greedy_coin_change",
        ):

            try:

                # ------------------------------------------------
                # Conversion des pièces
                # ------------------------------------------------

                coins = [
                    int(value.strip())
                    for value in coins_text.split(",")
                    if value.strip()
                ]

                amount_value = int(amount)

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                if not coins:

                    raise ValueError(
                        "La liste des pièces ne peut pas être vide."
                    )

                if any(
                    coin <= 0
                    for coin in coins
                ):

                    raise ValueError(
                        "Toutes les pièces doivent être "
                        "strictement positives."
                    )

                if len(set(coins)) != len(coins):

                    raise ValueError(
                        "Les pièces doivent être uniques."
                    )

                if amount_value < 0:

                    raise ValueError(
                        "Le montant doit être positif ou nul."
                    )

                # ------------------------------------------------
                # Tri décroissant
                # ------------------------------------------------

                sorted_coins = sorted(
                    coins,
                    reverse=True,
                )

                # ------------------------------------------------
                # Appel du véritable algorithme
                # ------------------------------------------------

                result = greedy_coin_change(
                    sorted_coins,
                    amount_value,
                )

                # ------------------------------------------------
                # Validation du résultat
                # ------------------------------------------------

                if result == -1:

                    st.warning(
                        """
                        Aucun rendu de monnaie n'est possible
                        avec les pièces fournies.
                        """
                    )

                else:

                    st.success(
                        "Greedy Coin Change résolu avec succès."
                    )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                if result == -1:

                    st.error(
                        "Le montant demandé ne peut pas être représenté "
                        "avec les pièces disponibles."
                    )

                else:

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Montant demandé",
                            str(amount_value),
                        )

                    with col2:

                        st.metric(
                            "Nombre de pièces",
                            str(result),
                        )

                    with col3:

                        st.metric(
                            "Pièces disponibles",
                            str(len(coins)),
                        )

                    st.markdown(
                        f"""
                        Pour obtenir un montant de
                        **{amount_value}**,

                        la stratégie gloutonne utilise :

                        \[
                        \boxed{{{result}}}
                        \]

                        pièce(s).
                        """
                    )

                st.divider()

                # ==================================================
                # PIÈCES DISPONIBLES
                # ==================================================

                st.markdown(
                    "### 🪙 Pièces disponibles"
                )

                coin_data = {
                    "Ordre": list(
                        range(
                            1,
                            len(sorted_coins) + 1,
                        )
                    ),
                    "Pièce": sorted_coins,
                }

                st.dataframe(
                    coin_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # STRATÉGIE GLOUTONNE
                # ==================================================

                st.markdown(
                    "### 🧠 Étape 1 — Trier les pièces"
                )

                st.latex(
                    r"""
                    c_1 > c_2 > \cdots > c_n
                    """
                )

                st.markdown(
                    """
                    Les pièces sont classées par ordre décroissant.

                    L'algorithme commence donc par la pièce de plus
                    grande valeur et utilise autant de pièces que
                    possible avant de passer à la suivante.
                    """
                )

                st.dataframe(
                    {
                        "Position": list(
                            range(
                                1,
                                len(sorted_coins) + 1,
                            )
                        ),
                        "Pièce": sorted_coins,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # CALCUL PÉDAGOGIQUE
                # ==================================================

                st.markdown(
                    "### 🔎 Étape 2 — Calcul étape par étape"
                )

                remaining_amount = amount_value
                pedagogical_coins = []
                selection_steps = []

                for position, coin in enumerate(
                    sorted_coins,
                    start=1,
                ):

                    if remaining_amount <= 0:
                        break

                    count = (
                        remaining_amount // coin
                    )

                    if count > 0:

                        amount_before = remaining_amount

                        amount_taken = (
                            count * coin
                        )

                        remaining_amount -= (
                            amount_taken
                        )

                        pedagogical_coins.extend(
                            [coin] * count
                        )

                        selection_steps.append(
                            {
                                "Étape": position,
                                "Pièce": coin,
                                "Montant avant": amount_before,
                                "Nombre utilisé": count,
                                "Montant obtenu": amount_taken,
                                "Montant restant": remaining_amount,
                            }
                        )

                # ==================================================
                # AFFICHAGE DES ÉTAPES
                # ==================================================

                if selection_steps:

                    for step in selection_steps:

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"""
                                ### Étape {step['Étape']} — Pièce {step['Pièce']}

                                **Montant avant :**
                                `{step['Montant avant']}`

                                **Pièce choisie :**
                                `{step['Pièce']}`

                                **Nombre de pièces utilisées :**
                                `{step['Nombre utilisé']}`

                                **Montant obtenu :**
                                `{step['Montant obtenu']}`

                                **Montant restant :**
                                `{step['Montant restant']}`
                                """
                            )

                            st.latex(
                                rf"""
                                {step['Montant avant']}
                                -
                                ({step['Nombre utilisé']}
                                \times
                                {step['Pièce']})
                                =
                                {step['Montant restant']}
                                """
                            )

                else:

                    st.info(
                        "Aucune pièce n'est nécessaire pour un montant nul."
                    )

                st.divider()

                # ==================================================
                # TABLEAU DES DÉCISIONS
                # ==================================================

                st.markdown(
                    "### 📊 Tableau des décisions"
                )

                if selection_steps:

                    st.dataframe(
                        selection_steps,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "Aucune étape de sélection."
                    )

                st.divider()

                # ==================================================
                # SOLUTION CONSTRUITE
                # ==================================================

                st.markdown(
                    "### ✅ Solution construite"
                )

                if pedagogical_coins:

                    solution_text = " + ".join(
                        str(coin)
                        for coin in pedagogical_coins
                    )

                    st.markdown(
                        f"""
                        Le rendu de monnaie obtenu est :

                        **{solution_text}**

                        Vérification :

                        \[
                        {solution_text}
                        =
                        {sum(pedagogical_coins)}
                        \]
                        """
                    )

                    st.metric(
                        "Nombre total de pièces",
                        str(len(pedagogical_coins)),
                    )

                    st.metric(
                        "Montant restant",
                        str(remaining_amount),
                    )

                else:

                    st.info(
                        "Aucune pièce n'a été utilisée."
                    )

                st.divider()

                # ==================================================
                # FORMULE
                # ==================================================

                st.markdown(
                    "### 🧮 Formulation mathématique"
                )

                st.markdown(
                    """
                    Si `xᵢ` représente le nombre de pièces de valeur
                    `cᵢ` utilisées, le montant obtenu est :

                    \[
                    \sum_i x_i c_i
                    \]

                    avec :

                    \[
                    x_i \geq 0
                    \]

                    et :

                    \[
                    \sum_i x_i c_i = A
                    \]

                    où `A` représente le montant demandé.
                    """
                )

                st.divider()

                # ==================================================
                # POURQUOI LE GLOUTON
                # ==================================================

                st.markdown(
                    "### 💡 Pourquoi utiliser une stratégie gloutonne ?"
                )

                st.markdown(
                    """
                    La stratégie gloutonne fait toujours le choix
                    de la plus grande pièce possible.

                    Elle est simple et rapide.

                    Cependant, contrairement au **Fractional Knapsack**,
                    le choix glouton ne garantit pas l'optimalité pour
                    tous les systèmes de pièces.

                    Par exemple, avec les pièces :

                    `1, 3, 4`

                    et le montant :

                    `6`

                    la stratégie gloutonne choisit :

                    `4 + 1 + 1`

                    soit **3 pièces**.

                    Pourtant, une autre solution existe :

                    `3 + 3`

                    soit **2 pièces**.

                    Cet exemple montre qu'il faut distinguer :

                    - une solution construite par une stratégie gloutonne ;
                    - une solution réellement optimale.
                    """
                )

                st.divider()

                # ==================================================
                # COMPLEXITÉ
                # ==================================================

                st.markdown(
                    "### ⚙️ Complexité"
                )

                st.markdown(
                    """
                    Pour `n` types de pièces :

                    **Tri des pièces :**

                    \[
                    O(n\log n)
                    \]

                    **Parcours des pièces :**

                    \[
                    O(n)
                    \]

                    **Complexité totale :**

                    \[
                    O(n\log n)
                    \]

                    Si les pièces sont déjà triées dans l'ordre
                    décroissant, le parcours est en :

                    \[
                    O(n)
                    \]
                    """
                )

                st.divider()

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                if result == -1:

                    if remaining_amount != 0:

                        st.info(
                            f"""
                            Le calcul pédagogique confirme que le
                            montant restant est **{remaining_amount}**.

                            Aucun rendu complet n'est donc possible
                            avec les pièces disponibles.
                            """
                        )

                else:

                    pedagogical_count = len(
                        pedagogical_coins
                    )

                    if (
                        pedagogical_count == result
                        and remaining_amount == 0
                    ):

                        st.success(
                            """
                            Le résultat du module
                            `greedy_coin_change` correspond
                            exactement au calcul pédagogique
                            effectué par l'interface.
                            """
                        )

                    else:

                        st.error(
                            f"""
                            Une différence a été détectée.

                            Résultat du module :

                            `{result}`

                            Résultat pédagogique :

                            `{pedagogical_count}`

                            Montant restant :

                            `{remaining_amount}`
                            """
                        )

            # ====================================================
            # GESTION DES ERREURS
            # ====================================================

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Greedy Coin Change : {error}"
                )

            except Exception as error:

                st.error(
                    """
                    Une erreur est survenue pendant
                    la résolution du Greedy Coin Change.
                    """
                )

                st.exception(error)

    
    # ========================================================
    # HUFFMAN CODING
    # ========================================================

    elif selected_algorithm == "huffman_coding":

        st.markdown(
        "### 🌳 Huffman Coding — Codage de Huffman"
    )

    st.markdown(
        """
        Le **codage de Huffman** est un algorithme glouton
        utilisé pour construire un **code binaire préfixe**
        permettant de représenter des symboles avec un nombre
        réduit de bits.

        L'idée principale est de donner :

        - des codes courts aux symboles fréquents ;
        - des codes plus longs aux symboles rares.

        La construction de l'arbre suit une stratégie gloutonne :
        on fusionne toujours les **deux fréquences les plus faibles**.
        """
    )

    st.info(
        """
        **Principe glouton**

        À chaque étape :

        1. sélectionner les deux nœuds de plus faible fréquence ;
        2. les fusionner ;
        3. créer un nouveau nœud dont la fréquence est leur somme ;
        4. remettre ce nouveau nœud dans l'ensemble ;
        5. recommencer jusqu'à obtenir un seul arbre.
        """
    )

    st.divider()

    # ====================================================
    # DONNÉES
    # ====================================================

    st.markdown(
        "### 📊 Fréquences des symboles"
    )

    frequencies_text = st.text_area(
        "Fréquences",
        value=(
            "A:45\n"
            "B:13\n"
            "C:12\n"
            "D:16\n"
            "E:9\n"
            "F:5"
        ),
        height=180,
        key="greedy_huffman_frequencies",
        help=(
            "Un symbole par ligne sous la forme : A:45"
        ),
    )

    st.caption(
        """
        Format attendu :

        `symbole:fréquence`

        Exemple :

        `A:45`

        `B:13`

        `C:12`
        """
    )

    st.divider()

    # ====================================================
    # CALCUL
    # ====================================================

    if st.button(
        "Construire le code de Huffman",
        type="primary",
        use_container_width=True,
        key="execute_huffman_coding",
    ):

        try:

            # ------------------------------------------------
            # PARSING
            # ------------------------------------------------

            frequencies = {}

            for line_number, line in enumerate(
                frequencies_text.splitlines(),
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                if ":" not in line:

                    raise ValueError(
                        f"Ligne {line_number} invalide : "
                        "utilisez le format symbole:fréquence."
                    )

                symbol_text, frequency_text = (
                    line.split(":", 1)
                )

                symbol = symbol_text.strip()

                if not symbol:

                    raise ValueError(
                        f"Ligne {line_number} : "
                        "le symbole ne peut pas être vide."
                    )

                try:

                    frequency = int(
                        frequency_text.strip()
                    )

                except ValueError as error:

                    raise ValueError(
                        f"Ligne {line_number} : "
                        "la fréquence doit être un entier."
                    ) from error

                if frequency <= 0:

                    raise ValueError(
                        f"Ligne {line_number} : "
                        "la fréquence doit être strictement positive."
                    )

                if symbol in frequencies:

                    raise ValueError(
                        f"Le symbole « {symbol} » "
                        "est présent plusieurs fois."
                    )

                frequencies[symbol] = frequency

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if not frequencies:

                raise ValueError(
                    "Aucune fréquence valide n'a été saisie."
                )

            if len(frequencies) < 2:

                raise ValueError(
                    "Le codage de Huffman nécessite "
                    "au moins deux symboles."
                )

            # ------------------------------------------------
            # AFFICHAGE DES DONNÉES
            # ------------------------------------------------

            st.success(
                "Fréquences validées avec succès."
            )

            st.markdown(
                "### 📥 Fréquences initiales"
            )

            frequency_rows = [
                {
                    "Symbole": symbol,
                    "Fréquence": frequency,
                }
                for symbol, frequency in frequencies.items()
            ]

            st.dataframe(
                frequency_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # ------------------------------------------------
            # TRI DES FRÉQUENCES
            # ------------------------------------------------

            sorted_frequencies = sorted(
                frequencies.items(),
                key=lambda item: item[1],
            )

            st.markdown(
                "### 🔢 Fréquences triées"
            )

            sorted_rows = [
                {
                    "Ordre": index + 1,
                    "Symbole": symbol,
                    "Fréquence": frequency,
                }
                for index, (symbol, frequency)
                in enumerate(sorted_frequencies)
            ]

            st.dataframe(
                sorted_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # =================================================
            # CONVERSION DES FRÉQUENCES EN SÉQUENCE
            # =================================================
            #
            # IMPORTANT :
            # huffman_coding() attend une séquence de symboles
            # et non le dictionnaire frequencies.
            #
            # Exemple :
            #
            # A:3
            # B:2
            #
            # devient :
            #
            # ["A", "A", "A", "B", "B"]
            #
            # =================================================

            data = []

            for symbol, frequency in frequencies.items():

                data.extend(
                    [symbol] * frequency
                )

            # ------------------------------------------------
            # APPEL DU VÉRITABLE ALGORITHME
            # ------------------------------------------------

            result = huffman_coding(
                data
            )

            # ------------------------------------------------
            # INTERPRÉTATION DU RÉSULTAT
            # ------------------------------------------------

            if not isinstance(result, dict):

                raise TypeError(
                    "Le résultat de huffman_coding "
                    "doit être un dictionnaire de codes."
                )

            codes = result

            # ------------------------------------------------
            # RÉSULTAT
            # ------------------------------------------------

            st.markdown(
                "### 🏆 Codes de Huffman"
            )

            code_rows = []

            for symbol, frequency in sorted(
                frequencies.items(),
                key=lambda item: item[1],
                reverse=True,
            ):

                code = codes.get(
                    symbol,
                    "",
                )

                code_rows.append(
                    {
                        "Symbole": symbol,
                        "Fréquence": frequency,
                        "Code": code,
                        "Longueur": len(code),
                    }
                )

            st.dataframe(
                code_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # =================================================
            # EXPLICATION ÉTAPE PAR ÉTAPE
            # =================================================

            st.markdown(
                "### 🔎 Construction étape par étape"
            )

            st.markdown(
                """
                Huffman sélectionne toujours les deux
                fréquences les plus faibles.

                Avec les données fournies, les deux nœuds
                ayant les plus petites fréquences sont fusionnés,
                puis leur somme est réintroduite dans l'ensemble
                des nœuds.

                Le processus continue jusqu'à obtenir un seul arbre.
                """
            )

            # ------------------------------------------------
            # Simulation pédagogique de la fusion
            # ------------------------------------------------

            nodes = [
                {
                    "symbols": [symbol],
                    "frequency": frequency,
                }
                for symbol, frequency in frequencies.items()
            ]

            merge_steps = []

            step_number = 1

            while len(nodes) > 1:

                nodes.sort(
                    key=lambda node: node["frequency"]
                )

                first = nodes.pop(0)

                second = nodes.pop(0)

                merged_frequency = (
                    first["frequency"]
                    + second["frequency"]
                )

                merged_symbols = (
                    first["symbols"]
                    + second["symbols"]
                )

                merge_steps.append(
                    {
                        "Étape": step_number,
                        "Premier nœud": (
                            ", ".join(
                                first["symbols"]
                            )
                        ),
                        "Fréquence 1": (
                            first["frequency"]
                        ),
                        "Deuxième nœud": (
                            ", ".join(
                                second["symbols"]
                            )
                        ),
                        "Fréquence 2": (
                            second["frequency"]
                        ),
                        "Nouvelle fréquence": (
                            merged_frequency
                        ),
                    }
                )

                nodes.append(
                    {
                        "symbols": merged_symbols,
                        "frequency": merged_frequency,
                    }
                )

                step_number += 1

            st.dataframe(
                merge_steps,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # FORMULE
            # ------------------------------------------------

            st.markdown(
                "### 🧮 Règle de fusion"
            )

            st.latex(
                r"""
                f_{\mathrm{nouveau}}
                =
                f_{\mathrm{gauche}}
                +
                f_{\mathrm{droite}}
                """
            )

            st.markdown(
                """
                À chaque fusion, les deux plus petites
                fréquences sont remplacées par leur somme.
                """
            )

            st.divider()

            # =================================================
            # ARBRE CONCEPTUEL
            # =================================================

            st.markdown(
                "### 🌳 Structure conceptuelle"
            )

            st.markdown(
                """
                Le processus construit progressivement
                un arbre binaire :

                ```text
                              Racine
                             /     \\
                           ...     ...
                          /  \\     /  \\
                         ... ...   ... ...

                ```

                Chaque branche peut recevoir un bit :

                - branche gauche → `0`
                - branche droite → `1`

                Le code d'un symbole correspond alors au chemin
                parcouru depuis la racine jusqu'à ce symbole.
                """
            )

            st.divider()

            # =================================================
            # PROPRIÉTÉ DE PRÉFIXE
            # =================================================

            st.markdown(
                "### 🔐 Propriété de code préfixe"
            )

            st.info(
                """
                Aucun code de symbole ne doit être le préfixe
                d'un autre code.

                Cette propriété permet de décoder les données
                sans ambiguïté.
                """
            )

            st.divider()

            # =================================================
            # LONGUEUR MOYENNE
            # =================================================

            st.markdown(
                "### 📏 Longueur moyenne du code"
            )

            total_frequency = sum(
                frequencies.values()
            )

            weighted_length = sum(
                frequencies[symbol]
                * len(codes[symbol])
                for symbol in frequencies
                if symbol in codes
            )

            average_length = (
                weighted_length
                / total_frequency
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Fréquence totale",
                    f"{total_frequency:g}",
                )

            with col2:

                st.metric(
                    "Bits pondérés",
                    f"{weighted_length:g}",
                )

            with col3:

                st.metric(
                    "Longueur moyenne",
                    f"{average_length:.3f}",
                )

            st.latex(
                r"""
                L_{\mathrm{moy}}
                =
                \frac{
                    \sum_i f_i\,l_i
                }{
                    \sum_i f_i
                }
                """
            )

            st.divider()

            # =================================================
            # POURQUOI HUFFMAN EST GLOUTON
            # =================================================

            st.markdown(
                "### 💡 Pourquoi Huffman est un algorithme glouton ?"
            )

            st.write(
                """
                Huffman prend localement les deux symboles
                ou sous-arbres ayant les plus faibles fréquences,
                puis les fusionne.

                Cette décision locale permet de construire
                progressivement un arbre de codage optimal
                pour le problème classique du codage préfixe
                pondéré.
                """
            )

            st.divider()

            # =================================================
            # COMPLEXITÉ
            # =================================================

            st.markdown(
                "### ⚙️ Complexité"
            )

            st.markdown(
                r"""
                Pour `n` symboles, l'utilisation d'une
                file de priorité permet de construire
                l'arbre de Huffman en :

                \[
                O(n\log n)
                \]

                L'espace nécessaire pour stocker l'arbre
                et les codes est de l'ordre de :

                \[
                O(n)
                \]
                """
            )

            st.divider()

            # =================================================
            # VÉRIFICATION
            # =================================================

            st.markdown(
                "### ✅ Vérification"
            )

            valid_codes = all(
                isinstance(code, str)
                and len(code) > 0
                and set(code).issubset({"0", "1"})
                for code in codes.values()
            )

            symbols_match = (
                set(codes.keys())
                == set(frequencies.keys())
            )

            if valid_codes and symbols_match:

                st.success(
                    """
                    Le résultat du module
                    `huffman_coding` contient un code binaire
                    valide pour chaque symbole fourni.
                    """
                )

            else:

                st.error(
                    """
                    Une incohérence a été détectée entre
                    les fréquences fournies et les codes
                    retournés par le module.
                    """
                )

        # ====================================================
        # GESTION DES ERREURS
        # ====================================================

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur Huffman Coding : {error}"
            )

        except Exception as error:

            st.error(
                """
                Une erreur est survenue pendant
                la construction du code de Huffman.
                """
            )

            st.exception(error)

    
    # ========================================================
    # JOB SEQUENCING WITH DEADLINES
    # ========================================================

    elif selected_algorithm == "job_sequencing":

        st.markdown(
            "### ⏱️ Job Sequencing — Ordonnancement des tâches"
        )

        st.markdown(
            """
            Le **Job Sequencing with Deadlines** est un algorithme
            glouton permettant de sélectionner un ensemble de tâches
            afin de **maximiser le profit total**, sous contrainte
            que chaque tâche soit exécutée avant ou à sa date limite.

            Chaque tâche possède :

            - un identifiant ;
            - une **deadline** ;
            - un **profit**.

            L'idée gloutonne consiste à :

            1. trier les tâches par profit décroissant ;
            2. prendre la tâche la plus profitable disponible ;
            3. la placer dans le dernier créneau libre avant sa deadline ;
            4. continuer jusqu'à traiter toutes les tâches.
            """
        )

        st.info(
            """
            **Principe glouton**

            Pour chaque tâche, on cherche à préserver autant que possible
            les créneaux précédents.

            Une tâche de profit élevé est donc placée dans le **dernier
            créneau disponible avant sa deadline**, ce qui laisse les
            créneaux antérieurs disponibles pour d'autres tâches.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES
        # ====================================================

        st.markdown(
            "### 📋 Tâches"
        )

        jobs_text = st.text_area(
            "Tâches",
            value=(
                "A:2:100\n"
                "B:1:19\n"
                "C:2:27\n"
                "D:1:25\n"
                "E:3:15"
            ),
            height=180,
            key="greedy_job_sequencing_jobs",
            help=(
                "Un job par ligne sous la forme : "
                "job:deadline:profit"
            ),
        )

        st.caption(
            """
            **Format attendu :**

            `job:deadline:profit`

            Exemple :

            `A:2:100`

            `B:1:19`

            `C:2:27`

            La deadline doit être un entier positif et le profit
            un nombre positif ou nul.
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Exécuter Job Sequencing",
            type="primary",
            use_container_width=True,
            key="execute_job_sequencing",
        ):

            try:

                # =================================================
                # PARSING
                # =================================================

                jobs = []

                for line_number, line in enumerate(
                    jobs_text.splitlines(),
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split(":")

                    if len(parts) != 3:
                        raise ValueError(
                            f"Ligne {line_number} invalide : "
                            "utilisez le format job:deadline:profit."
                        )

                    job_text = parts[0].strip()
                    deadline_text = parts[1].strip()
                    profit_text = parts[2].strip()

                    if not job_text:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le nom du job ne peut pas être vide."
                        )

                    # -------------------------------------------------
                    # DEADLINE
                    # -------------------------------------------------

                    try:
                        deadline = int(deadline_text)

                    except ValueError as error:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "la deadline doit être un entier."
                        ) from error

                    if deadline <= 0:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "la deadline doit être strictement positive."
                        )

                    # -------------------------------------------------
                    # PROFIT
                    # -------------------------------------------------

                    try:
                        profit = float(profit_text)

                    except ValueError as error:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le profit doit être un nombre."
                        ) from error

                    if profit < 0:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le profit ne peut pas être négatif."
                        )

                    # -------------------------------------------------
                    # AJOUT
                    # -------------------------------------------------

                    jobs.append(
                        {
                            "job": job_text,
                            "deadline": deadline,
                            "profit": profit,
                        }
                    )

                # =================================================
                # VALIDATION
                # =================================================

                if not jobs:
                    raise ValueError(
                        "Aucune tâche valide n'a été saisie."
                    )

                job_names = [
                    job["job"]
                    for job in jobs
                ]

                if len(job_names) != len(set(job_names)):
                    raise ValueError(
                        "Deux tâches possèdent le même identifiant."
                    )

                st.success(
                    "Les tâches ont été validées avec succès."
                )

                # =================================================
                # DONNÉES INITIALES
                # =================================================

                st.markdown(
                    "### 📥 Tâches initiales"
                )

                initial_rows = [
                    {
                        "Job": job["job"],
                        "Deadline": job["deadline"],
                        "Profit": job["profit"],
                    }
                    for job in jobs
                ]

                st.dataframe(
                    initial_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # TRI PAR PROFIT DÉCROISSANT
                # =================================================

                sorted_jobs = sorted(
                    jobs,
                    key=lambda job: job["profit"],
                    reverse=True,
                )

                st.markdown(
                    "### 💰 Tâches triées par profit décroissant"
                )

                sorted_rows = [
                    {
                        "Ordre": index + 1,
                        "Job": job["job"],
                        "Deadline": job["deadline"],
                        "Profit": job["profit"],
                    }
                    for index, job in enumerate(sorted_jobs)
                ]

                st.dataframe(
                    sorted_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # PRÉPARATION DES DONNÉES POUR L'ALGORITHME
                # =================================================

                data = [
                    (
                        job["job"],
                        job["deadline"],
                        job["profit"],
                    )
                    for job in jobs
                ]

                # =================================================
                # APPEL DU VÉRITABLE ALGORITHME
                # =================================================

                result = job_sequencing(data)

                # =================================================
                # INTERPRÉTATION DU RÉSULTAT
                # =================================================

                if isinstance(result, dict):

                    selected_jobs = result.get(
                        "selected_jobs",
                        result.get("jobs", [])
                    )

                    total_profit = result.get(
                        "total_profit",
                        result.get("profit", 0)
                    )

                    schedule = result.get(
                        "schedule",
                        []
                    )

                elif isinstance(result, tuple):

                    if len(result) == 2:

                        selected_jobs = result[0]
                        total_profit = result[1]
                        schedule = selected_jobs

                    else:

                        raise TypeError(
                            "Le résultat de job_sequencing() "
                            "possède un format inattendu."
                        )

                elif isinstance(result, list):

                    selected_jobs = result
                    schedule = result

                    # Calcul local du profit
                    selected_names = set()

                    for item in result:

                        if isinstance(item, dict):

                            name = item.get(
                                "job",
                                item.get("id")
                            )

                            if name is not None:
                                selected_names.add(name)

                        elif isinstance(item, str):

                            selected_names.add(item)

                    total_profit = sum(
                        job["profit"]
                        for job in jobs
                        if job["job"] in selected_names
                    )

                else:

                    raise TypeError(
                        "Le résultat de job_sequencing() "
                        "doit être une liste, un tuple ou un dictionnaire."
                    )

                # =================================================
                # RÉSULTAT
                # =================================================

                st.markdown(
                    "### 🏆 Ordonnancement optimal"
                )

                if schedule:

                    schedule_rows = []

                    for position, item in enumerate(
                        schedule,
                        start=1,
                    ):

                        if isinstance(item, dict):

                            job_name = item.get(
                                "job",
                                item.get("id", "")
                            )

                            deadline = item.get(
                                "deadline",
                                ""
                            )

                            profit = item.get(
                                "profit",
                                ""
                            )

                        elif isinstance(item, (tuple, list)):

                            job_name = (
                                item[0]
                                if len(item) > 0
                                else ""
                            )

                            deadline = (
                                item[1]
                                if len(item) > 1
                                else ""
                            )

                            profit = (
                                item[2]
                                if len(item) > 2
                                else ""
                            )

                        else:

                            job_name = str(item)

                            matching_job = next(
                                (
                                    job
                                    for job in jobs
                                    if job["job"] == job_name
                                ),
                                None,
                            )

                            if matching_job:

                                deadline = matching_job["deadline"]
                                profit = matching_job["profit"]

                            else:

                                deadline = ""
                                profit = ""

                        schedule_rows.append(
                            {
                                "Créneau": position,
                                "Job": job_name,
                                "Deadline": deadline,
                                "Profit": profit,
                            }
                        )

                    st.dataframe(
                        schedule_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.warning(
                        "Aucune tâche n'a été sélectionnée."
                    )

                # =================================================
                # INDICATEURS
                # =================================================

                st.divider()

                st.markdown(
                    "### 📊 Résumé"
                )

                selected_count = len(
                    schedule
                ) if schedule else 0

                max_deadline = max(
                    job["deadline"]
                    for job in jobs
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Tâches initiales",
                        len(jobs),
                    )

                with col2:

                    st.metric(
                        "Tâches sélectionnées",
                        selected_count,
                    )

                with col3:

                    st.metric(
                        "Créneaux disponibles",
                        max_deadline,
                    )

                with col4:

                    st.metric(
                        "Profit total",
                        f"{total_profit:g}",
                    )

                st.divider()

                # =================================================
                # SIMULATION PÉDAGOGIQUE
                # =================================================

                st.markdown(
                    "### 🔎 Construction étape par étape"
                )

                st.markdown(
                    """
                    L'algorithme examine les tâches dans l'ordre
                    décroissant de profit.

                    Pour chaque tâche, il cherche le **dernier créneau
                    libre avant sa deadline**.

                    Si un créneau est disponible, la tâche est placée.
                    Sinon, elle est rejetée.
                    """
                )

                # -------------------------------------------------
                # SIMULATION
                # -------------------------------------------------

                simulation_slots = [
                    None
                    for _ in range(max_deadline)
                ]

                simulation_steps = []

                for job in sorted_jobs:

                    placed = False
                    selected_slot = None

                    latest_slot = min(
                        job["deadline"],
                        max_deadline,
                    )

                    for slot in range(
                        latest_slot - 1,
                        -1,
                        -1,
                    ):

                        if simulation_slots[slot] is None:

                            simulation_slots[slot] = job["job"]

                            placed = True
                            selected_slot = slot + 1

                            break

                    simulation_steps.append(
                        {
                            "Job": job["job"],
                            "Deadline": job["deadline"],
                            "Profit": job["profit"],
                            "Créneau choisi": (
                                selected_slot
                                if placed
                                else "—"
                            ),
                            "Décision": (
                                "Acceptée"
                                if placed
                                else "Rejetée"
                            ),
                        }
                    )

                st.dataframe(
                    simulation_steps,
                    use_container_width=True,
                    hide_index=True,
                )

                # =================================================
                # CALENDRIER FINAL
                # =================================================

                st.markdown(
                    "### 🗓️ Calendrier final"
                )

                calendar_rows = []

                for slot_number, job_name in enumerate(
                    simulation_slots,
                    start=1,
                ):

                    if job_name is None:

                        calendar_rows.append(
                            {
                                "Créneau": slot_number,
                                "Job": "Libre",
                                "Statut": "Disponible",
                            }
                        )

                    else:

                        matching_job = next(
                            (
                                job
                                for job in jobs
                                if job["job"] == job_name
                            ),
                            None,
                        )

                        calendar_rows.append(
                            {
                                "Créneau": slot_number,
                                "Job": job_name,
                                "Deadline": (
                                    matching_job["deadline"]
                                    if matching_job
                                    else ""
                                ),
                                "Profit": (
                                    matching_job["profit"]
                                    if matching_job
                                    else ""
                                ),
                                "Statut": "Occupé",
                            }
                        )

                st.dataframe(
                    calendar_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # FORMULE
                # =================================================

                st.markdown(
                    "### 🧮 Objectif"
                )

                st.latex(
                    r"""
                    \max \sum_{j \in S} p_j
                    """
                )

                st.markdown(
                    """
                    où :

                    - \(S\) représente l'ensemble des tâches sélectionnées ;
                    - \(p_j\) représente le profit de la tâche \(j\).

                    Une tâche sélectionnée doit respecter sa deadline.
                    """
                )

                st.divider()

                # =================================================
                # POURQUOI L'ALGORITHME EST GLOUTON
                # =================================================

                st.markdown(
                    "### 💡 Pourquoi Job Sequencing est-il glouton ?"
                )

                st.write(
                    """
                    L'algorithme prend d'abord la décision qui semble
                    la plus avantageuse localement : considérer en priorité
                    la tâche ayant le profit le plus élevé.

                    Il tente ensuite de placer cette tâche le plus tard
                    possible avant sa deadline.

                    Cette stratégie permet de conserver les créneaux
                    précédents pour les autres tâches.
                    """
                )

                st.divider()

                # =================================================
                # COMPLEXITÉ
                # =================================================

                st.markdown(
                    "### ⚙️ Complexité"
                )

                st.markdown(
                    r"""
                    Avec un tri des tâches par profit :

                    **\[ O(n \log n) \]**

                    Puis, dans l'implémentation classique où l'on recherche
                    linéairement un créneau disponible :

                    **\[ O(n \cdot d) \]**

                    où \(d\) représente le nombre maximal de créneaux.

                    L'espace utilisé pour le calendrier est de l'ordre de :

                    **\[ O(d) \]**
                    """
                )

                st.divider()

                # =================================================
                # VÉRIFICATION
                # =================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                valid_schedule = True
                used_slots = set()

                for slot_number, job_name in enumerate(
                    simulation_slots,
                    start=1,
                ):

                    if job_name is None:
                        continue

                    if slot_number in used_slots:

                        valid_schedule = False
                        break

                    used_slots.add(slot_number)

                    matching_job = next(
                        (
                            job
                            for job in jobs
                            if job["job"] == job_name
                        ),
                        None,
                    )

                    if matching_job is None:

                        valid_schedule = False
                        break

                    if slot_number > matching_job["deadline"]:

                        valid_schedule = False
                        break

                if valid_schedule:

                    st.success(
                        """
                        L'ordonnancement obtenu respecte les deadlines :
                        chaque tâche sélectionnée est exécutée dans un
                        créneau valide et chaque créneau est utilisé au
                        maximum une fois.
                        """
                    )

                else:

                    st.error(
                        """
                        Une incohérence a été détectée dans
                        l'ordonnancement retourné.
                        """
                    )

            # ====================================================
            # GESTION DES ERREURS
            # ====================================================

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Job Sequencing : {error}"
                )

            except Exception as error:

                st.error(
                    """
                    Une erreur est survenue pendant
                    l'exécution de Job Sequencing.
                    """
                )

                st.exception(error)

    # ========================================================
    # INTERVAL SCHEDULING
    # ========================================================

    elif selected_algorithm == "interval_scheduling":

        st.markdown(
            "### 📅 Interval Scheduling — Ordonnancement d'intervalles"
        )

        st.markdown(
            """
            L'**Interval Scheduling** est un algorithme glouton qui
            cherche à sélectionner le **plus grand nombre d'intervalles
            compatibles**, c'est-à-dire des activités qui ne se
            chevauchent pas.

            Chaque intervalle possède :

            - un identifiant ;
            - une heure de début ;
            - une heure de fin.

            L'idée gloutonne consiste à :

            1. trier les intervalles par heure de fin croissante ;
            2. sélectionner le premier intervalle compatible ;
            3. ignorer les intervalles qui se chevauchent ;
            4. continuer jusqu'à la fin de la liste.
            """
        )

        st.info(
            """
            **Principe glouton**

            On choisit toujours l'intervalle qui **se termine le plus tôt**
            parmi les intervalles encore compatibles.

            Pourquoi ?

            Un intervalle qui se termine tôt laisse davantage de temps
            disponible pour sélectionner d'autres intervalles.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES
        # ====================================================

        st.markdown(
            "### 📋 Intervalles"
        )

        intervals_text = st.text_area(
            "Intervalles",
            value=(
                "A:1:3\n"
                "B:2:5\n"
                "C:4:7\n"
                "D:6:9\n"
                "E:8:10\n"
                "F:9:11\n"
                "G:10:12"
            ),
            height=200,
            key="greedy_interval_scheduling_intervals",
            help=(
                "Un intervalle par ligne sous la forme : "
                "nom:début:fin"
            ),
        )

        st.caption(
            """
            **Format attendu :**

            `nom:début:fin`

            Exemple :

            `A:1:3`

            `B:2:5`

            `C:4:7`

            Un intervalle doit respecter :

            **début < fin**
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Exécuter Interval Scheduling",
            type="primary",
            use_container_width=True,
            key="execute_interval_scheduling",
        ):

            try:

                # =================================================
                # PARSING
                # =================================================

                intervals = []

                for line_number, line in enumerate(
                    intervals_text.splitlines(),
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split(":")

                    if len(parts) != 3:
                        raise ValueError(
                            f"Ligne {line_number} invalide : "
                            "utilisez le format nom:début:fin."
                        )

                    name_text = parts[0].strip()
                    start_text = parts[1].strip()
                    end_text = parts[2].strip()

                    if not name_text:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le nom de l'intervalle ne peut pas "
                            "être vide."
                        )

                    # -------------------------------------------------
                    # DÉBUT
                    # -------------------------------------------------

                    try:
                        start = float(start_text)

                    except ValueError as error:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "l'heure de début doit être un nombre."
                        ) from error

                    # -------------------------------------------------
                    # FIN
                    # -------------------------------------------------

                    try:
                        end = float(end_text)

                    except ValueError as error:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "l'heure de fin doit être un nombre."
                        ) from error

                    # -------------------------------------------------
                    # VALIDATION
                    # -------------------------------------------------

                    if start >= end:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le début doit être strictement inférieur "
                            "à la fin."
                        )

                    intervals.append(
                        {
                            "name": name_text,
                            "start": start,
                            "end": end,
                        }
                    )

                # =================================================
                # VALIDATION GLOBALE
                # =================================================

                if not intervals:
                    raise ValueError(
                        "Aucun intervalle valide n'a été saisi."
                    )

                interval_names = [
                    interval["name"]
                    for interval in intervals
                ]

                if len(interval_names) != len(
                    set(interval_names)
                ):
                    raise ValueError(
                        "Deux intervalles possèdent le même identifiant."
                    )

                if len(intervals) < 2:

                    st.warning(
                        "Un seul intervalle a été fourni. "
                        "Il sera sélectionné automatiquement."
                    )

                st.success(
                    "Les intervalles ont été validés avec succès."
                )

                # =================================================
                # INTERVALLES INITIAUX
                # =================================================

                st.markdown(
                    "### 📥 Intervalles initiaux"
                )

                initial_rows = [
                    {
                        "Intervalle": interval["name"],
                        "Début": interval["start"],
                        "Fin": interval["end"],
                        "Durée": (
                            interval["end"]
                            - interval["start"]
                        ),
                    }
                    for interval in intervals
                ]

                st.dataframe(
                    initial_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # TRI PAR FIN CROISSANTE
                # =================================================

                sorted_intervals = sorted(
                    intervals,
                    key=lambda interval: (
                        interval["end"],
                        interval["start"],
                    ),
                )

                st.markdown(
                    "### 🔢 Intervalles triés par heure de fin"
                )

                sorted_rows = [
                    {
                        "Ordre": index + 1,
                        "Intervalle": interval["name"],
                        "Début": interval["start"],
                        "Fin": interval["end"],
                    }
                    for index, interval in enumerate(
                        sorted_intervals
                    )
                ]

                st.dataframe(
                    sorted_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                
                # =================================================
                # PRÉPARATION DES DONNÉES POUR L'ALGORITHME
                # =================================================
                #
                # IMPORTANT :
                # interval_scheduling() attend des intervalles
                # contenant exactement DEUX valeurs :
                #
                # (début, fin)
                #
                # Il ne faut donc PAS envoyer :
                #
                # (nom, début, fin)
                #
                # =================================================

                data = [
                    (
                        interval["start"],
                        interval["end"],
                    )
                    for interval in intervals
                ]

                # -------------------------------------------------
                # APPEL DU VÉRITABLE ALGORITHME
                # -------------------------------------------------

                result = interval_scheduling(data)

                # =================================================
                # INTERPRÉTATION DU RÉSULTAT
                # =================================================

                if isinstance(result, dict):

                    selected_intervals = result.get(
                        "selected_intervals",
                        result.get(
                            "intervals",
                            result.get(
                                "schedule",
                                [],
                            ),
                        ),
                    )

                    count = result.get(
                        "count",
                        len(selected_intervals),
                    )

                elif isinstance(result, tuple):

                    if len(result) == 2:

                        selected_intervals = result[0]
                        count = result[1]

                    else:

                        raise TypeError(
                            "Le résultat de interval_scheduling() "
                            "possède un format inattendu."
                        )

                elif isinstance(result, list):

                    selected_intervals = result
                    count = len(result)

                else:

                    raise TypeError(
                        "Le résultat de interval_scheduling() "
                        "doit être une liste, un tuple ou "
                        "un dictionnaire."
                    )

                # =================================================
                # NORMALISATION DU RÉSULTAT
                # =================================================

                selected_names = []

                for item in selected_intervals:

                    if isinstance(item, dict):

                        name = item.get(
                            "name",
                            item.get(
                                "interval",
                                item.get(
                                    "job",
                                    item.get(
                                        "id",
                                        "",
                                    ),
                                ),
                            ),
                        )

                    elif isinstance(item, (tuple, list)):

                        name = (
                            item[0]
                            if len(item) > 0
                            else ""
                        )

                    else:

                        name = str(item)

                    if name:
                        selected_names.append(str(name))

                # =================================================
                # RÉSULTAT
                # =================================================

                st.markdown(
                    "### 🏆 Intervalles sélectionnés"
                )

                selected_rows = []

                for position, name in enumerate(
                    selected_names,
                    start=1,
                ):

                    matching_interval = next(
                        (
                            interval
                            for interval in intervals
                            if interval["name"] == name
                        ),
                        None,
                    )

                    if matching_interval:

                        selected_rows.append(
                            {
                                "Ordre": position,
                                "Intervalle": name,
                                "Début": matching_interval[
                                    "start"
                                ],
                                "Fin": matching_interval[
                                    "end"
                                ],
                                "Durée": (
                                    matching_interval["end"]
                                    - matching_interval["start"]
                                ),
                            }
                        )

                if selected_rows:

                    st.dataframe(
                        selected_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.warning(
                        "Aucun intervalle n'a été sélectionné."
                    )

                # =================================================
                # INDICATEURS
                # =================================================

                st.divider()

                st.markdown(
                    "### 📊 Résumé"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Intervalles initiaux",
                        len(intervals),
                    )

                with col2:

                    st.metric(
                        "Intervalles sélectionnés",
                        len(selected_names),
                    )

                with col3:

                    st.metric(
                        "Taux de sélection",
                        f"{(
                            len(selected_names)
                            / len(intervals)
                            * 100
                        ):.1f}%",
                    )

                st.divider()

                # =================================================
                # SIMULATION PÉDAGOGIQUE
                # =================================================

                st.markdown(
                    "### 🔎 Construction étape par étape"
                )

                st.markdown(
                    """
                    L'algorithme examine les intervalles dans l'ordre
                    croissant de leur heure de fin.

                    Pour chaque intervalle :

                    - s'il commence après ou exactement au moment où
                      le dernier intervalle sélectionné se termine,
                      il est accepté ;
                    - sinon, il est rejeté car il chevauche un intervalle
                      déjà sélectionné.
                    """
                )

                # -------------------------------------------------
                # SIMULATION
                # -------------------------------------------------

                simulation_steps = []

                selected_simulation = []

                last_end = None

                for interval in sorted_intervals:

                    if last_end is None:

                        accepted = True

                    else:

                        accepted = (
                            interval["start"]
                            >= last_end
                        )

                    if accepted:

                        selected_simulation.append(
                            interval["name"]
                        )

                        last_end = interval["end"]

                        decision = "Accepté"

                    else:

                        decision = "Rejeté"

                    simulation_steps.append(
                        {
                            "Intervalle": interval["name"],
                            "Début": interval["start"],
                            "Fin": interval["end"],
                            "Dernière fin": (
                                "—"
                                if last_end is None
                                else last_end
                            ),
                            "Décision": decision,
                        }
                    )

                st.dataframe(
                    simulation_steps,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # CALENDRIER FINAL
                # =================================================

                st.markdown(
                    "### 🗓️ Planning final"
                )

                calendar_rows = []

                for position, name in enumerate(
                    selected_simulation,
                    start=1,
                ):

                    matching_interval = next(
                        (
                            interval
                            for interval in intervals
                            if interval["name"] == name
                        ),
                        None,
                    )

                    if matching_interval:

                        calendar_rows.append(
                            {
                                "Position": position,
                                "Intervalle": name,
                                "Début": matching_interval[
                                    "start"
                                ],
                                "Fin": matching_interval[
                                    "end"
                                ],
                            }
                        )

                st.dataframe(
                    calendar_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # FORMULE
                # =================================================

                st.markdown(
                    "### 🧮 Règle de sélection"
                )

                st.latex(
                    r"""
                    s_i \geq f_{\text{dernier}}
                    """
                )

                st.markdown(
                    """
                    Un intervalle \(i\) peut être sélectionné si son
                    heure de début \(s_i\) est supérieure ou égale à
                    l'heure de fin du dernier intervalle sélectionné.
                    """
                )

                st.divider()

                # =================================================
                # POURQUOI CETTE STRATÉGIE EST GLOUTONNE
                # =================================================

                st.markdown(
                    "### 💡 Pourquoi Interval Scheduling est-il glouton ?"
                )

                st.write(
                    """
                    À chaque étape, l'algorithme choisit l'intervalle
                    compatible qui se termine le plus tôt.

                    Cette décision est locale : on ne cherche pas à
                    explorer toutes les combinaisons possibles.

                    Le choix d'un intervalle se terminant tôt laisse
                    cependant le maximum de temps disponible pour les
                    intervalles suivants.
                    """
                )

                st.divider()

                # =================================================
                # PROPRIÉTÉ
                # =================================================

                st.markdown(
                    "### 🔐 Propriété de compatibilité"
                )

                st.info(
                    """
                    Deux intervalles sont compatibles lorsqu'ils ne se
                    chevauchent pas.

                    Dans cette implémentation, un intervalle qui commence
                    exactement lorsque le précédent se termine est
                    considéré comme compatible.
                    """
                )

                st.divider()

                # =================================================
                # COMPLEXITÉ
                # =================================================

                st.markdown(
                    "### ⚙️ Complexité"
                )

                st.markdown(
                    r"""
                    Le tri des \(n\) intervalles par heure de fin nécessite :

                    **\[ O(n \log n) \]**

                    Le parcours des intervalles après le tri nécessite :

                    **\[ O(n) \]**

                    La complexité globale est donc :

                    **\[ O(n \log n) \]**

                    L'espace supplémentaire utilisé par l'algorithme
                    dépend de l'implémentation, mais le parcours glouton
                    lui-même nécessite essentiellement :

                    **\[ O(1) \]**
                    """
                )

                st.divider()

                # =================================================
                # VÉRIFICATION
                # =================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                valid_schedule = True

                previous_end = None

                for name in selected_names:

                    matching_interval = next(
                        (
                            interval
                            for interval in intervals
                            if interval["name"] == name
                        ),
                        None,
                    )

                    if matching_interval is None:

                        valid_schedule = False
                        break

                    if previous_end is not None:

                        if (
                            matching_interval["start"]
                            < previous_end
                        ):

                            valid_schedule = False
                            break

                    previous_end = matching_interval["end"]

                if valid_schedule:

                    st.success(
                        """
                        L'ordonnancement obtenu est valide :
                        les intervalles sélectionnés ne se chevauchent pas.
                        """
                    )

                else:

                    st.error(
                        """
                        Une incohérence a été détectée dans
                        l'ordonnancement retourné.
                        """
                    )

            # ====================================================
            # GESTION DES ERREURS
            # ====================================================

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Interval Scheduling : {error}"
                )

            except Exception as error:

                st.error(
                    """
                    Une erreur est survenue pendant
                    l'exécution de Interval Scheduling.
                    """
                )

                st.exception(error)
    

    # ========================================================
    # AUTRES ALGORITHMES GREEDY
    # ========================================================

    else:

        st.info(
            """
            Les expérimentations Greedy seront ajoutées
            progressivement.

            Algorithmes disponibles :

            - Activity Selection
            - Fractional Knapsack
            - Greedy Coin Change
            - Huffman Coding
            - Job Sequencing
            - Interval Scheduling
            """
        )



# ============================================================
# TREE HELPERS
# ============================================================

def parse_tree_values(text):
    """Convertit une chaîne '8, 4, 12' en liste d'entiers."""

    if not text.strip():
        raise ValueError("Veuillez entrer au moins une valeur.")

    try:
        values = [
            int(item.strip())
            for item in text.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "Toutes les valeurs doivent être des entiers."
        ) from exc

    if not values:
        raise ValueError("L'arbre ne peut pas être vide.")

    return values


def build_binary_tree(values):
    """
    Construit un arbre binaire complet à partir d'une liste.

    Exemple :

                  8
               /     \
              4       12
             / \     /  \
            2   6   10   14
    """

    if not values:
        return None

    nodes = [TreeNode(value) for value in values]

    for index, node in enumerate(nodes):

        left_index = 2 * index + 1
        right_index = 2 * index + 2

        if left_index < len(nodes):
            node.left = nodes[left_index]

        if right_index < len(nodes):
            node.right = nodes[right_index]

    return nodes[0]


def build_bst(values):
    """Construit un Binary Search Tree à partir d'une liste."""

    root = None

    for value in values:
        root = bst_insert(root, value)

    return root



import streamlit as st
import math
import plotly.graph_objects as go

from core.algorithms.algorithm_explanations import (
    list_algorithm_categories,
    get_algorithms_by_category,
    get_algorithm_explanation,
    search_algorithm_explanations,
)

from core.algorithms.searching import (
    linear_search,
    binary_search,
    jump_search,
    interpolation_search,
    exponential_search,
)

from core.algorithms.sorting import (
    bubble_sort,
    selection_sort,
    insertion_sort,
    merge_sort,
    quick_sort,
    heap_sort,
    counting_sort,
    radix_sort,
    bucket_sort,
)

from core.algorithms.graph_algorithms import (
    bfs,
    dfs,
    dijkstra,
    bellman_ford,
    floyd_warshall,
    kruskal,
    prim,
)

from core.algorithms.dynamic_programming import (
    fibonacci,
    climbing_stairs,
    knapsack_01,
    coin_change,
    longest_common_subsequence,
    longest_increasing_subsequence,
    matrix_chain_multiplication,
)


from core.algorithms.greedy import (
    activity_selection,
    fractional_knapsack,
    greedy_coin_change,
    huffman_coding,
    job_sequencing,
    interval_scheduling,
)

from core.algorithms.backtracking import (
    n_queens,
    solve_sudoku,
    solve_maze,
    subsets,
    permutations,
    combination_sum,
)

from core.algorithms.tree_algorithms import (
    TreeNode,
    preorder_traversal,
    inorder_traversal,
    postorder_traversal,
    level_order_traversal,
    bst_search,
    bst_insert,
    bst_delete,
    MinHeap,
    PriorityQueue,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MathLab AI — Algorithms Lab",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# FONCTIONS UTILITAIRES — VISUALISATION DES GRAPHES
# ============================================================


def create_graph_positions(nodes):
    """
    Génère automatiquement une disposition circulaire
    des sommets du graphe.
    """

    if not nodes:
        return {}

    positions = {}

    radius = 1.0
    count = len(nodes)

    for index, node in enumerate(nodes):

        angle = (
            2 * math.pi * index / count
        )

        positions[node] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
        )

    return positions


def create_graph_figure(
    graph,
    weighted=False,
    highlighted_nodes=None,
    highlighted_edges=None,
):
    """
    Construit une visualisation Plotly du graphe.

    Parameters
    ----------
    graph:
        Graphe représenté sous forme de dictionnaire.

    weighted:
        Indique si le graphe contient des poids.

    highlighted_nodes:
        Sommets à mettre en évidence.

    highlighted_edges:
        Arêtes à mettre en évidence.
    """

    highlighted_nodes = set(
        highlighted_nodes or []
    )

    highlighted_edges = {
        tuple(edge)
        for edge in (
            highlighted_edges or []
        )
    }

    nodes = list(graph.keys())

    positions = create_graph_positions(
        nodes
    )

    figure = go.Figure()

    # ========================================================
    # ARÊTES
    # ========================================================

    edge_x = []
    edge_y = []

    processed_edges = set()

    for source in nodes:

        neighbors = graph[source]

        if weighted:

            neighbor_items = neighbors.items()

        else:

            neighbor_items = (
                (target, None)
                for target in neighbors
            )

        for target, weight in neighbor_items:

            if target not in positions:
                continue

            # Pour les graphes non orientés représentés
            # dans les deux sens, éviter le doublon visuel.
            edge_key = frozenset(
                [source, target]
            )

            if edge_key in processed_edges:
                continue

            processed_edges.add(
                edge_key
            )

            x0, y0 = positions[source]
            x1, y1 = positions[target]

            edge_x.extend(
                [x0, x1, None]
            )

            edge_y.extend(
                [y0, y1, None]
            )

    if edge_x:

        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={
                    "width": 1.5,
                },
                hoverinfo="none",
                name="Arêtes",
            )
        )

    # ========================================================
    # POIDS DES ARÊTES
    # ========================================================

    if weighted:

        annotations = []

        processed_weight_edges = set()

        for source in nodes:

            for target, weight in graph[
                source
            ].items():

                if target not in positions:
                    continue

                edge_key = frozenset(
                    [source, target]
                )

                if edge_key in processed_weight_edges:
                    continue

                processed_weight_edges.add(
                    edge_key
                )

                x0, y0 = positions[source]
                x1, y1 = positions[target]

                annotations.append(
                    {
                        "x": (
                            x0 + x1
                        ) / 2,
                        "y": (
                            y0 + y1
                        ) / 2,
                        "text": str(weight),
                        "showarrow": False,
                    }
                )

        figure.update_layout(
            annotations=annotations
        )

    # ========================================================
    # ARÊTES MISES EN ÉVIDENCE
    # ========================================================

    if highlighted_edges:

        highlighted_x = []
        highlighted_y = []

        for source, target in highlighted_edges:

            if (
                source not in positions
                or target not in positions
            ):
                continue

            x0, y0 = positions[source]
            x1, y1 = positions[target]

            highlighted_x.extend(
                [x0, x1, None]
            )

            highlighted_y.extend(
                [y0, y1, None]
            )

        if highlighted_x:

            figure.add_trace(
                go.Scatter(
                    x=highlighted_x,
                    y=highlighted_y,
                    mode="lines",
                    line={
                        "width": 5,
                    },
                    hoverinfo="none",
                    name="Arêtes sélectionnées",
                )
            )

    # ========================================================
    # SOMMETS
    # ========================================================

    node_x = []
    node_y = []
    node_text = []
    node_sizes = []

    for node in nodes:

        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)
        node_text.append(
            str(node)
        )

        if node in highlighted_nodes:

            node_sizes.append(28)

        else:

            node_sizes.append(20)

    if node_x:

        figure.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_text,
                textposition="middle center",
                marker={
                    "size": node_sizes,
                },
                hovertemplate=(
                    "Sommet : %{text}"
                    "<extra></extra>"
                ),
                name="Sommets",
            )
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    figure.update_layout(
        title="Visualisation du graphe",
        showlegend=True,
        height=600,
        hovermode="closest",
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
    )

    return figure


def display_graph_visualization(
    graph,
    weighted=False,
    highlighted_nodes=None,
    highlighted_edges=None,
):
    """
    Affiche le graphe dans une visualisation Plotly.
    """

    figure = create_graph_figure(
        graph,
        weighted=weighted,
        highlighted_nodes=highlighted_nodes,
        highlighted_edges=highlighted_edges,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )



# ============================================================
# FONCTIONS — PLUS COURTS CHEMINS
# ============================================================


def reconstruct_shortest_paths(
    graph,
    start_node,
    distances,
):
    """
    Reconstruit un chemin depuis start_node vers chaque sommet
    à partir des distances calculées.

    Cette fonction est utilisée pour Dijkstra et Bellman-Ford.

    Elle détermine un prédécesseur compatible avec les distances :
        distance[u] + poids(u, v) == distance[v]

    Returns
    -------
    dict
        Dictionnaire :
            sommet -> liste représentant le chemin.
    """

    paths = {}

    if start_node not in graph:
        return paths

    paths[start_node] = [start_node]

    # --------------------------------------------------------
    # Recherche des prédécesseurs
    # --------------------------------------------------------

    predecessors = {
        node: None
        for node in graph
    }

    for source in graph:

        for target, weight in graph[source].items():

            if (
                source in distances
                and target in distances
                and distances[source] != math.inf
                and distances[target] != math.inf
            ):

                if math.isclose(
                    distances[source] + weight,
                    distances[target],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):

                    if predecessors[target] is None:
                        predecessors[target] = source

    # --------------------------------------------------------
    # Reconstruction des chemins
    # --------------------------------------------------------

    for target in graph:

        if target == start_node:
            continue

        if (
            target not in distances
            or distances[target] == math.inf
        ):
            paths[target] = []
            continue

        path = []
        current = target
        visited = set()

        while current is not None:

            if current in visited:
                path = []
                break

            visited.add(current)
            path.append(current)

            if current == start_node:
                break

            current = predecessors.get(
                current
            )

        if not path or path[-1] != start_node:

            paths[target] = []

        else:

            path.reverse()
            paths[target] = path

    return paths


def get_path_edges(path):
    """
    Convertit un chemin de sommets en liste d'arêtes.

    Exemple
    -------
    ["A", "C", "D"]

    devient :

    [("A", "C"), ("C", "D")]
    """

    if len(path) < 2:
        return []

    return [
        (
            path[index],
            path[index + 1],
        )
        for index in range(
            len(path) - 1
        )
    ]



# ============================================================
# RÉCUPÉRATION DES EXPLICATIONS
# ============================================================

explanation = get_algorithm_explanation(
    selected_algorithm
)


# ============================================================
# PRÉSENTATION GÉNÉRALE
# ============================================================

st.header("📖 Explication de l'algorithme")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(
        f"🧠 {explanation['name']}"
    )

with col2:
    st.metric(
        "Catégorie",
        explanation["category"],
    )


# ============================================================
# DESCRIPTION
# ============================================================

st.markdown("### 📝 Description")

st.write(
    explanation["description"]
)


# ============================================================
# IDÉE PRINCIPALE
# ============================================================

st.markdown("### 💡 Idée principale")

st.info(
    explanation["idea"]
)


# ============================================================
# ÉTAPES
# ============================================================

st.markdown("### 🔢 Étapes de l'algorithme")

for index, step in enumerate(
    explanation["steps"],
    start=1,
):
    st.markdown(
        f"**{index}.** {step}"
    )


# ============================================================
# COMPLEXITÉ
# ============================================================

st.header("📊 Complexité algorithmique")

complexity = explanation["complexity"]

best = complexity.get(
    "best",
    "Non disponible",
)

average = complexity.get(
    "average",
    "Non disponible",
)

worst = complexity.get(
    "worst",
    "Non disponible",
)

space = complexity.get(
    "space",
    "Non disponible",
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🟢 Meilleur cas",
        best,
    )

with col2:
    st.metric(
        "🔵 Cas moyen",
        average,
    )

with col3:
    st.metric(
        "🔴 Pire cas",
        worst,
    )

with col4:
    st.metric(
        "💾 Espace",
        space,
    )


with st.expander("ℹ️ Comprendre la notation Big-O"):

    st.markdown(
        """
        La notation **Big-O** permet de décrire la manière
        dont le temps ou l'espace nécessaire à un algorithme
        évolue lorsque la taille des données augmente.

        - **O(1)** → constant
        - **O(log n)** → logarithmique
        - **O(n)** → linéaire
        - **O(n log n)** → linéarithmique
        - **O(n²)** → quadratique
        - **O(2ⁿ)** → exponentiel
        - **O(n!)** → factoriel
        """
    )


# ============================================================
# AVANTAGES / LIMITATIONS
# ============================================================

st.header("⚖️ Avantages et limitations")

advantages = explanation["advantages"]
limitations = explanation["limitations"]

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "✅ Avantages",
        len(advantages),
    )

with col2:

    st.metric(
        "⚠️ Limitations",
        len(limitations),
    )

col1, col2 = st.columns(2)

with col1:

    st.markdown("### ✅ Avantages")

    if advantages:

        for index, advantage in enumerate(
            advantages,
            start=1,
        ):

            with st.container(border=True):

                st.markdown(
                    f"**Avantage {index}**"
                )

                st.write(
                    advantage
                )

    else:

        st.info(
            "Aucun avantage renseigné."
        )


with col2:

    st.markdown("### ⚠️ Limitations")

    if limitations:

        for index, limitation in enumerate(
            limitations,
            start=1,
        ):

            with st.container(border=True):

                st.markdown(
                    f"**Limitation {index}**"
                )

                st.write(
                    limitation
                )

    else:

        st.info(
            "Aucune limitation renseignée."
        )


# ============================================================
# EXEMPLE PÉDAGOGIQUE
# ============================================================

st.header("🧪 Exemple pédagogique")

example = explanation["example"]

if isinstance(example, dict):

    if "input" in example:

        st.markdown("### 📥 Entrée")

        with st.container(border=True):

            st.code(
                str(example["input"]),
                language="text",
            )

    st.markdown("### ⚙️ Démarche")

    with st.container(border=True):

        st.markdown(
            f"""
            L'algorithme **{explanation["name"]}**
            applique les étapes présentées précédemment
            sur les données d'entrée.
            """
        )

        for index, step in enumerate(
            explanation["steps"],
            start=1,
        ):

            st.markdown(
                f"**Étape {index} :** {step}"
            )

    if "result" in example:

        st.markdown("### 📤 Résultat")

        with st.container(border=True):

            st.code(
                str(example["result"]),
                language="text",
            )

            st.success(
                "Résultat obtenu par l'algorithme."
            )

    st.markdown("### 💡 Interprétation")

    with st.container(border=True):

        if "result" in example:

            st.write(
                f"""
                Pour l'exemple sélectionné,
                l'algorithme **{explanation["name"]}**
                transforme ou traite l'entrée donnée
                afin d'obtenir le résultat présenté
                ci-dessus.
                """
            )

        else:

            st.write(
                "Cet exemple illustre le fonctionnement "
                "général de l'algorithme."
            )

else:

    st.info(
        "Aucun exemple pédagogique disponible "
        "pour cet algorithme."
    )



# ============================================================
# ÉTAPE 10 — RÉSUMÉ DES CATÉGORIES
# ============================================================

st.header("📚 Résumé des catégories")

st.markdown(
    """
    Explorez rapidement les différentes familles d'algorithmes
    disponibles dans **MathLab AI**.
    """
)

# Récupération des catégories disponibles
categories = list_algorithm_categories()

# Correspondance catégorie → icône
category_icons = {
    "Searching": "🔎",
    "Sorting": "🔢",
    "Graphs": "🕸️",
    "Dynamic Programming": "🧩",
    "Greedy": "🎯",
    "Backtracking": "↩️",
    "Trees": "🌳",
    "Mathematical": "📐",
    "Strings": "🔤",
}

# Affichage sous forme de cartes
columns = st.columns(3)

for index, category in enumerate(categories):

    algorithms_in_category = get_algorithms_by_category(
        category
    )

    icon = category_icons.get(
        category,
        "🧠",
    )

    column = columns[index % 3]

    with column:

        with st.container(border=True):

            st.markdown(
                f"## {icon} {category}"
            )

            st.metric(
                "Algorithmes",
                len(algorithms_in_category),
            )

            if algorithms_in_category:

                st.caption(
                    " · ".join(
                        algorithms_in_category
                    )
                )

            else:

                st.caption(
                    "Aucun algorithme disponible."
                )


# ============================================================
# ÉTAPE 12 — EXPÉRIMENTATIONS PAR CATÉGORIE
# ============================================================

st.header("🧪 Expérimentation interactive")

st.markdown(
    """
    Exécutez réellement l'algorithme sélectionné avec vos
    propres données.

    Les résultats sont calculés directement à partir des
    implémentations présentes dans `core.algorithms`.
    """
)


# ============================================================
# FONCTION UTILITAIRE — PARSING DES LISTES
# ============================================================

def parse_algorithm_list(value: str) -> list:
    """
    Convertit une saisie utilisateur en liste de valeurs.

    Exemples :
        "10, 5, 8, 2"
        -> [10, 5, 8, 2]

        "10; 5; 8; 2"
        -> [10, 5, 8, 2]

        "10 5 8 2"
        -> [10, 5, 8, 2]
    """

    if not isinstance(value, str):
        raise TypeError(
            "Les données doivent être saisies sous forme de texte."
        )

    cleaned = value.replace(";", ",").strip()

    if not cleaned:
        raise ValueError(
            "Les données ne peuvent pas être vides."
        )

    if "," in cleaned:

        parts = [
            part.strip()
            for part in cleaned.split(",")
            if part.strip()
        ]

    else:

        parts = [
            part.strip()
            for part in cleaned.split()
            if part.strip()
        ]

    if not parts:
        raise ValueError(
            "Aucune donnée valide n'a été trouvée."
        )

    # --------------------------------------------------------
    # Conversion numérique
    # --------------------------------------------------------

    try:

        numbers = [
            float(part)
            for part in parts
        ]

        if all(
            number.is_integer()
            for number in numbers
        ):

            return [
                int(number)
                for number in numbers
            ]

        return numbers

    except ValueError:

        # Permet également de travailler avec des chaînes
        return parts


# ============================================================
# INFORMATIONS SUR L'ALGORITHME
# ============================================================

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🧠 Algorithme")

        st.write(
            explanation["name"]
        )

    with col2:

        st.markdown("### 📂 Catégorie")

        st.write(
            explanation["category"]
        )

    with col3:

        st.markdown("### ⚙️ Pire cas")

        st.write(
            explanation["complexity"].get(
                "worst",
                "N/D",
            )
        )


# ============================================================
# SEARCHING
# ============================================================

if selected_category == "Searching":

    st.subheader(
        "🔎 Expérimentation — Recherche"
    )

    st.markdown(
        """
        Entrez une liste de données puis une valeur à rechercher.

        **Linear Search** peut travailler sur une liste quelconque.

        **Binary Search**, **Jump Search**, **Interpolation Search**
        et **Exponential Search** nécessitent une liste triée
        dans l'ordre croissant.
        """
    )

    data_text = st.text_area(
        "Liste de données",
        value="10, 4, 7, 2, 15, 8",
        placeholder="Ex. 10, 4, 7, 2, 15, 8",
        key="search_data",
        height=100,
    )

    target_text = st.text_input(
        "Valeur recherchée",
        value="8",
        placeholder="Ex. 8",
        key="search_target",
    )

    st.divider()

    if st.button(
        "▶️ Rechercher",
        type="primary",
        use_container_width=True,
    ):

        try:

            data = parse_algorithm_list(
                data_text
            )

            target = target_text.strip()

            if not target:

                raise ValueError(
                    "La valeur recherchée ne peut pas être vide."
                )

            # ------------------------------------------------
            # Conversion de la cible
            # ------------------------------------------------

            try:

                numeric_target = float(target)

                if numeric_target.is_integer():

                    target = int(
                        numeric_target
                    )

                else:

                    target = numeric_target

            except ValueError:

                # Conserver une éventuelle valeur textuelle
                target = target

            # ------------------------------------------------
            # Algorithmes de recherche
            # ------------------------------------------------

            search_functions = {
                "linear_search": linear_search,
                "binary_search": binary_search,
                "jump_search": jump_search,
                "interpolation_search": interpolation_search,
                "exponential_search": exponential_search,
            }

            algorithm_function = search_functions.get(
                selected_algorithm
            )

            if algorithm_function is None:

                raise ValueError(
                    f"L'algorithme "
                    f"« {selected_algorithm} » "
                    "n'est pas disponible pour "
                    "l'expérimentation."
                )

            # ------------------------------------------------
            # Exécution réelle
            # ------------------------------------------------

            result = algorithm_function(
                data,
                target,
            )

            # ------------------------------------------------
            # Affichage
            # ------------------------------------------------

            st.success(
                "Recherche exécutée avec succès."
            )

            st.markdown("### 📊 Résultat")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Index",
                    result,
                )

            with col2:

                if result == -1:

                    st.metric(
                        "Statut",
                        "Non trouvé",
                    )

                else:

                    st.metric(
                        "Statut",
                        "Trouvé",
                    )

            with st.container(border=True):

                if result == -1:

                    st.warning(
                        f"La valeur **{target}** "
                        "n'a pas été trouvée dans la liste."
                    )

                else:

                    st.success(
                        f"La valeur **{target}** "
                        f"a été trouvée à l'index **{result}**."
                    )

            st.markdown(
                "### 📥 Données utilisées"
            )

            st.code(
                str(data),
                language="text",
            )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur de recherche : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "l'exécution de la recherche."
            )

            st.exception(error)


# ============================================================
# SORTING
# ============================================================

elif selected_category == "Sorting":

    st.subheader(
        "🔢 Expérimentation — Tri"
    )

    st.markdown(
        """
        Entrez une liste de valeurs et observez le résultat
        du tri effectué par l'algorithme sélectionné.
        """
    )

    data_text = st.text_area(
        "Liste à trier",
        value="10, 4, 7, 2, 15, 8",
        placeholder="Ex. 10, 4, 7, 2, 15, 8",
        key="sorting_data",
        height=100,
    )

    st.divider()

    if st.button(
        "▶️ Trier les données",
        type="primary",
        use_container_width=True,
    ):

        try:

            data = parse_algorithm_list(
                data_text
            )

            sorting_functions = {
                "bubble_sort": bubble_sort,
                "selection_sort": selection_sort,
                "insertion_sort": insertion_sort,
                "merge_sort": merge_sort,
                "quick_sort": quick_sort,
                "heap_sort": heap_sort,
                "counting_sort": counting_sort,
                "radix_sort": radix_sort,
                "bucket_sort": bucket_sort,
            }

            algorithm_function = sorting_functions.get(
                selected_algorithm
            )

            if algorithm_function is None:

                raise ValueError(
                    f"L'algorithme "
                    f"« {selected_algorithm} » "
                    "n'est pas disponible pour "
                    "l'expérimentation."
                )

            # ------------------------------------------------
            # Exécution réelle
            # ------------------------------------------------

            result = algorithm_function(
                data
            )

            # ------------------------------------------------
            # Affichage
            # ------------------------------------------------

            st.success(
                "Tri exécuté avec succès."
            )

            st.markdown("### 📊 Résultat")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 📥 Avant")

                with st.container(border=True):

                    st.code(
                        str(data),
                        language="text",
                    )

            with col2:

                st.markdown("### 📤 Après")

                with st.container(border=True):

                    st.code(
                        str(result),
                        language="text",
                    )

            st.divider()

            st.markdown(
                "### 🔍 Vérification"
            )

            if result == sorted(data):

                st.success(
                    "Le résultat est correctement trié "
                    "dans l'ordre croissant."
                )

            else:

                st.warning(
                    "Le résultat obtenu diffère du tri "
                    "croissant Python de référence."
                )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur de tri : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "l'exécution du tri."
            )

            st.exception(error)


elif selected_category == "Graphs":

    st.markdown("### 🕸️ Expérimentation des graphes")

    st.markdown(
        """
        Les algorithmes de graphes utilisent deux représentations
        différentes :

        - **Graphe non pondéré** pour BFS et DFS
        - **Graphe pondéré** pour Dijkstra, Bellman-Ford,
          Floyd-Warshall, Kruskal et Prim
        """
    )

    # ========================================================
    # GRAPHES D'EXEMPLE
    # ========================================================

    unweighted_graph_example = """{
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C"]
}"""

    weighted_graph_example = """{
    "A": {"B": 4, "C": 2},
    "B": {"A": 4, "D": 5},
    "C": {"A": 2, "D": 1},
    "D": {"B": 5, "C": 1}
}"""

    # ========================================================
    # TYPE DE GRAPHE
    # ========================================================

    if selected_algorithm in ["bfs", "dfs"]:

        graph_text = st.text_area(
            "Graphe",
            value=unweighted_graph_example,
            height=220,
            key="graph_unweighted_input",
            help=(
                "Utilisez un dictionnaire JSON dont chaque "
                "sommet contient une liste de voisins."
            ),
        )

        start_node = st.text_input(
            "Sommet de départ",
            value="A",
            key="graph_start_node",
        )

    else:

        graph_text = st.text_area(
            "Graphe pondéré",
            value=weighted_graph_example,
            height=220,
            key="graph_weighted_input",
            help=(
                "Utilisez un dictionnaire JSON dont chaque "
                "sommet contient un dictionnaire "
                "voisin → poids."
            ),
        )

        if selected_algorithm in [
            "dijkstra",
            "bellman_ford",
            "prim",
        ]:

            start_node = st.text_input(
                "Sommet de départ",
                value="A",
                key="graph_weighted_start_node",
            )

        else:

            start_node = None

    # ========================================================
    # EXPLICATION DU FORMAT
    # ========================================================

    with st.expander("ℹ️ Comprendre le format du graphe"):

        if selected_algorithm in ["bfs", "dfs"]:

            st.code(
                unweighted_graph_example,
                language="json",
            )

            st.caption(
                "Chaque sommet est associé à la liste de ses voisins."
            )

        else:

            st.code(
                weighted_graph_example,
                language="json",
            )

            st.caption(
                "Chaque sommet est associé à un dictionnaire "
                "voisin → poids."
            )

    
    # ========================================================
    # EXÉCUTION
    # ========================================================

    if st.button(
        "▶️ Exécuter l'algorithme",
        type="primary",
        use_container_width=True,
        key="execute_graph_algorithm",
    ):

        try:

            import json

            graph = json.loads(
                graph_text
            )

            if not isinstance(graph, dict):
                raise ValueError(
                    "Le graphe doit être un objet JSON."
                )

            # ==================================================
            # BFS
            # ==================================================

            if selected_algorithm == "bfs":

                result = bfs(
                    graph,
                    start_node,
                )

                st.success(
                    "Parcours BFS exécuté avec succès."
                )

                st.subheader("Ordre de parcours")

                st.write(
                    " → ".join(
                        str(node)
                        for node in result
                    )
                )

                st.dataframe(
                    {
                        "Étape": list(
                            range(
                                1,
                                len(result) + 1,
                            )
                        ),
                        "Sommet": result,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # VISUALISATION BFS
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation du graphe"
                )

                display_graph_visualization(
                    graph,
                    weighted=False,
                    highlighted_nodes=result,
                )

                st.caption(
                    "Les sommets visités par BFS sont "
                    "mis en évidence."
                )

            # ==================================================
            # DFS
            # ==================================================

            elif selected_algorithm == "dfs":

                result = dfs(
                    graph,
                    start_node,
                )

                st.success(
                    "Parcours DFS exécuté avec succès."
                )

                st.subheader("Ordre de parcours")

                st.write(
                    " → ".join(
                        str(node)
                        for node in result
                    )
                )

                st.dataframe(
                    {
                        "Étape": list(
                            range(
                                1,
                                len(result) + 1,
                            )
                        ),
                        "Sommet": result,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # VISUALISATION DFS
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation du graphe"
                )

                display_graph_visualization(
                    graph,
                    weighted=False,
                    highlighted_nodes=result,
                )

                st.caption(
                    "Les sommets visités par DFS sont "
                    "mis en évidence."
                )


            # ==================================================
            # DIJKSTRA
            # ==================================================

            elif selected_algorithm == "dijkstra":

                distances = dijkstra(
                    graph,
                    start_node,
                )

                st.success(
                    "Algorithme de Dijkstra exécuté avec succès."
                )

                st.subheader(
                    "Distances minimales depuis "
                    f"{start_node}"
                )

                st.dataframe(
                    {
                        "Sommet": list(
                            distances.keys()
                        ),
                        "Distance": list(
                            distances.values()
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # RECONSTRUCTION DES CHEMINS
                # ------------------------------------------------

                paths = reconstruct_shortest_paths(
                    graph,
                    start_node,
                    distances,
                )

                st.divider()

                st.subheader(
                    "🛣️ Plus courts chemins"
                )

                path_rows = []

                for target, path in paths.items():

                    if path:

                        path_rows.append(
                            {
                                "Départ": start_node,
                                "Destination": target,
                                "Chemin": " → ".join(
                                    str(node)
                                    for node in path
                                ),
                                "Distance": distances[target],
                            }
                        )

                    else:

                        path_rows.append(
                            {
                                "Départ": start_node,
                                "Destination": target,
                                "Chemin": "Aucun chemin",
                                "Distance": distances[target],
                            }
                        )

                st.dataframe(
                    path_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # CHOIX DE LA DESTINATION
                # ------------------------------------------------

                reachable_nodes = [
                    node
                    for node in paths
                    if paths[node]
                ]

                if reachable_nodes:

                    destination = st.selectbox(
                        "Destination à visualiser",
                        reachable_nodes,
                        key="dijkstra_destination",
                    )

                    selected_path = paths[
                        destination
                    ]

                    selected_edges = get_path_edges(
                        selected_path
                    )

                    st.markdown(
                        "### 🛣️ Chemin sélectionné"
                    )

                    st.success(
                        " → ".join(
                            str(node)
                            for node in selected_path
                        )
                    )

                    st.metric(
                        "Distance minimale",
                        distances[destination],
                    )

                    # ------------------------------------------------
                    # VISUALISATION
                    # ------------------------------------------------

                    st.subheader(
                        "🕸️ Visualisation du plus court chemin"
                    )

                    display_graph_visualization(
                        graph,
                        weighted=True,
                        highlighted_nodes=selected_path,
                        highlighted_edges=selected_edges,
                    )

                    st.caption(
                        "Les sommets et les arêtes du plus court "
                        "chemin sélectionné sont mis en évidence."
                    )

                else:

                    st.warning(
                        "Aucun chemin accessible depuis "
                        f"{start_node}."
                    )


            # ==================================================
            # BELLMAN-FORD
            # ==================================================

            elif selected_algorithm == "bellman_ford":

                distances = bellman_ford(
                    graph,
                    start_node,
                )

                st.success(
                    "Algorithme de Bellman-Ford exécuté avec succès."
                )

                st.subheader(
                    "Distances minimales depuis "
                    f"{start_node}"
                )

                st.dataframe(
                    {
                        "Sommet": list(
                            distances.keys()
                        ),
                        "Distance": list(
                            distances.values()
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # RECONSTRUCTION DES CHEMINS
                # ------------------------------------------------

                paths = reconstruct_shortest_paths(
                    graph,
                    start_node,
                    distances,
                )

                st.divider()

                st.subheader(
                    "🛣️ Plus courts chemins"
                )

                path_rows = []

                for target, path in paths.items():

                    if path:

                        path_rows.append(
                            {
                                "Départ": start_node,
                                "Destination": target,
                                "Chemin": " → ".join(
                                    str(node)
                                    for node in path
                                ),
                                "Distance": distances[target],
                            }
                        )

                    else:

                        path_rows.append(
                            {
                                "Départ": start_node,
                                "Destination": target,
                                "Chemin": "Aucun chemin",
                                "Distance": distances[target],
                            }
                        )

                st.dataframe(
                    path_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # CHOIX DE LA DESTINATION
                # ------------------------------------------------

                reachable_nodes = [
                    node
                    for node in paths
                    if paths[node]
                ]

                if reachable_nodes:

                    destination = st.selectbox(
                        "Destination à visualiser",
                        reachable_nodes,
                        key="bellman_ford_destination",
                    )

                    selected_path = paths[
                        destination
                    ]

                    selected_edges = get_path_edges(
                        selected_path
                    )

                    st.markdown(
                        "### 🛣️ Chemin sélectionné"
                    )

                    st.success(
                        " → ".join(
                            str(node)
                            for node in selected_path
                        )
                    )

                    st.metric(
                        "Distance minimale",
                        distances[destination],
                    )

                    # ------------------------------------------------
                    # VISUALISATION
                    # ------------------------------------------------

                    st.subheader(
                        "🕸️ Visualisation du plus court chemin"
                    )

                    display_graph_visualization(
                        graph,
                        weighted=True,
                        highlighted_nodes=selected_path,
                        highlighted_edges=selected_edges,
                    )

                    st.caption(
                        "Les sommets et les arêtes du plus court "
                        "chemin sélectionné sont mis en évidence."
                    )

                else:

                    st.warning(
                        "Aucun chemin accessible depuis "
                        f"{start_node}."
                    )



                # ------------------------------------------------
                # VISUALISATION FLOYD-WARSHALL
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation du graphe"
                )

                display_graph_visualization(
                    graph,
                    weighted=True,
                )

                st.caption(
                    "Le graphe pondéré utilisé pour calculer "
                    "les plus courtes distances entre toutes "
                    "les paires de sommets."
                )

            # ==================================================
            # KRUSKAL
            # ==================================================

            elif selected_algorithm == "kruskal":

                result = kruskal(
                    graph,
                )

                edges, total_weight = result

                st.success(
                    "Algorithme de Kruskal exécuté avec succès."
                )

                st.subheader(
                    "Arbre couvrant minimal"
                )

                st.dataframe(
                    {
                        "Arête": [
                            f"{edge[0]} — {edge[1]}"
                            for edge in edges
                        ],
                        "Poids": [
                            edge[2]
                            for edge in edges
                        ],
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.metric(
                    "Poids total",
                    total_weight,
                )

                # ------------------------------------------------
                # VISUALISATION KRUSKAL
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation de l'arbre couvrant minimal"
                )

                highlighted_edges = [
                    (
                        edge[0],
                        edge[1],
                    )
                    for edge in edges
                ]

                display_graph_visualization(
                    graph,
                    weighted=True,
                    highlighted_edges=highlighted_edges,
                )

                st.caption(
                    "Les arêtes sélectionnées par Kruskal "
                    "sont mises en évidence."
                )

            # ==================================================
            # PRIM
            # ==================================================

            elif selected_algorithm == "prim":

                result = prim(
                    graph,
                    start_node,
                )

                edges, total_weight = result

                st.success(
                    "Algorithme de Prim exécuté avec succès."
                )

                st.subheader(
                    "Arbre couvrant minimal"
                )

                st.dataframe(
                    {
                        "Arête": [
                            f"{edge[0]} — {edge[1]}"
                            for edge in edges
                        ],
                        "Poids": [
                            edge[2]
                            for edge in edges
                        ],
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.metric(
                    "Poids total",
                    total_weight,
                )

                # ------------------------------------------------
                # VISUALISATION PRIM
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "🕸️ Visualisation de l'arbre couvrant minimal"
                )

                highlighted_edges = [
                    (
                        edge[0],
                        edge[1],
                    )
                    for edge in edges
                ]

                display_graph_visualization(
                    graph,
                    weighted=True,
                    highlighted_edges=highlighted_edges,
                    highlighted_nodes=[
                        start_node
                    ],
                )

                st.caption(
                    "Les arêtes sélectionnées par Prim "
                    "sont mises en évidence et le sommet "
                    "de départ est également identifié."
                )

            else:

                st.info(
                    "Cet algorithme de graphe n'est pas encore "
                    "connecté à la zone d'expérimentation."
                )

        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as error:

            st.error(
                f"Erreur de graphe : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "l'exécution de l'algorithme."
            )

            st.exception(error)


# ============================================================
# DYNAMIC PROGRAMMING
# ============================================================

elif selected_category == "Dynamic Programming":

    st.subheader(
        "🧩 Expérimentation — Programmation dynamique"
    )

    st.markdown(
        """
        Cette section permet d'expérimenter plusieurs algorithmes
        classiques de **programmation dynamique** et de visualiser
        leur démarche étape par étape.
        """
    )

    st.divider()

    # ========================================================
    # FIBONACCI
    # ========================================================

    if selected_algorithm == "fibonacci":

        st.markdown(
            "### 🔢 Fibonacci"
        )

        st.markdown(
            """
            La suite de Fibonacci est définie par :

            \[
            F(0)=0,\qquad F(1)=1
            \]

            puis :

            \[
            F(n)=F(n-1)+F(n-2)
            \]

            Chaque valeur dépend donc des deux valeurs précédentes.
            """
        )

        n = st.number_input(
            "Indice n",
            min_value=0,
            max_value=1000,
            value=10,
            step=1,
            key="dp_fibonacci_n",
        )

        if st.button(
            "Calculer Fibonacci",
            type="primary",
            use_container_width=True,
            key="execute_fibonacci",
        ):

            n_value = int(n)

            try:

                result = fibonacci(n_value)

                st.success(
                    "Calcul de Fibonacci terminé."
                )

                st.metric(
                    f"F({n_value})",
                    str(result),
                )

                st.divider()

                # ==================================================
                # INTERPRÉTATION
                # ==================================================

                st.markdown(
                    "### 🧠 Interprétation"
                )

                st.info(
                    f"""
                    Pour calculer F({n_value}), l'algorithme
                    construit progressivement les valeurs précédentes
                    de la suite au lieu de recalculer plusieurs fois
                    les mêmes sous-problèmes.
                    """
                )

                st.divider()

                # ==================================================
                # CAS PARTICULIERS
                # ==================================================

                st.markdown(
                    "### 📌 Cas de base"
                )

                if n_value == 0:

                    st.latex(
                        r"F(0)=0"
                    )

                elif n_value == 1:

                    st.latex(
                        r"F(1)=1"
                    )

                else:

                    st.latex(
                        r"F(0)=0"
                    )

                    st.latex(
                        r"F(1)=1"
                    )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🔎 Calcul étape par étape"
                )

                if n_value >= 2:

                    fibonacci_values = [0, 1]

                    for index in range(
                        2,
                        n_value + 1,
                    ):

                        previous_1 = fibonacci_values[
                            index - 1
                        ]

                        previous_2 = fibonacci_values[
                            index - 2
                        ]

                        current = (
                            previous_1
                            + previous_2
                        )

                        fibonacci_values.append(
                            current
                        )

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"### Étape {index}"
                            )

                            st.latex(
                                rf"""
                                F({index})
                                =
                                F({index - 1})
                                +
                                F({index - 2})
                                """
                            )

                            st.latex(
                                rf"""
                                F({index})
                                =
                                {previous_1}
                                +
                                {previous_2}
                                =
                                {current}
                                """
                            )

                    st.divider()

                    st.markdown(
                        "### 📊 Tableau de la suite"
                    )

                    st.dataframe(
                        {
                            "Indice n": list(
                                range(
                                    n_value + 1
                                )
                            ),
                            "F(n)": fibonacci_values,
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                if n_value == 0:

                    expected = 0

                elif n_value == 1:

                    expected = 1

                else:

                    expected = fibonacci_values[-1]

                if result == expected:

                    st.success(
                        "La valeur calculée par le module "
                        "correspond au calcul pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le calcul pédagogique et le module."
                    )

            except Exception as error:

                st.error(
                    f"Une erreur est survenue lors du calcul de Fibonacci : {error}"
                )

    # ========================================================
    # CLIMBING STAIRS
    # ========================================================

    elif selected_algorithm == "climbing_stairs":

        st.markdown(
            "### 🪜 Climbing Stairs"
        )

        st.markdown(
            """
            On dispose de **n marches**.

            À chaque mouvement, on peut monter :

            - 1 marche ;
            - 2 marches.

            Le nombre de façons d'atteindre la marche `n` est donné par :

            \[
            C(n)=C(n-1)+C(n-2)
            \]

            avec :

            \[
            C(0)=1,\qquad C(1)=1
            \]

            Pourquoi ?

            - Si le dernier mouvement fait **1 marche**, il reste
              `n-1` marches.
            - Si le dernier mouvement fait **2 marches**, il reste
              `n-2` marches.

            On additionne donc les deux possibilités.
            """
        )

        n = st.number_input(
            "Nombre de marches n",
            min_value=0,
            max_value=1000,
            value=5,
            step=1,
            key="dp_climbing_stairs_n",
        )

        if st.button(
            "Calculer le nombre de façons",
            type="primary",
            use_container_width=True,
            key="execute_climbing_stairs",
        ):

            n_value = int(n)

            try:

                result = climbing_stairs(
                    n_value
                )

                st.success(
                    "Calcul de Climbing Stairs terminé."
                )

                st.metric(
                    f"C({n_value})",
                    str(result),
                )

                st.divider()

                # ==================================================
                # INTERPRÉTATION
                # ==================================================

                st.markdown(
                    "### 🧠 Interprétation"
                )

                st.info(
                    f"""
                    Il existe **{result} façon(s)** différentes
                    d'atteindre la marche {n_value} lorsque chaque
                    mouvement permet de monter 1 ou 2 marches.
                    """
                )

                st.divider()

                # ==================================================
                # CAS DE BASE
                # ==================================================

                st.markdown(
                    "### 📌 Cas de base"
                )

                st.latex(
                    r"C(0)=1"
                )

                st.latex(
                    r"C(1)=1"
                )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                if n_value >= 2:

                    st.divider()

                    st.markdown(
                        "### 🔎 Calcul étape par étape"
                    )

                    climbing_values = [
                        1,
                        1,
                    ]

                    for index in range(
                        2,
                        n_value + 1,
                    ):

                        previous_1 = climbing_values[
                            index - 1
                        ]

                        previous_2 = climbing_values[
                            index - 2
                        ]

                        current = (
                            previous_1
                            + previous_2
                        )

                        climbing_values.append(
                            current
                        )

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"### Étape {index}"
                            )

                            st.latex(
                                rf"""
                                C({index})
                                =
                                C({index - 1})
                                +
                                C({index - 2})
                                """
                            )

                            st.latex(
                                rf"""
                                C({index})
                                =
                                {previous_1}
                                +
                                {previous_2}
                                =
                                {current}
                                """
                            )

                    st.divider()

                    st.markdown(
                        "### 📊 Tableau des possibilités"
                    )

                    st.dataframe(
                        {
                            "Nombre de marches n": list(
                                range(
                                    n_value + 1
                                )
                            ),
                            "Nombre de façons C(n)": climbing_values,
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                if n_value == 0:

                    expected = 1

                elif n_value == 1:

                    expected = 1

                else:

                    expected = climbing_values[-1]

                if result == expected:

                    st.success(
                        "La valeur calculée par le module "
                        "correspond au calcul pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le calcul pédagogique et le module."
                    )

            except Exception as error:

                st.error(
                    f"Une erreur est survenue lors du calcul : {error}"
                )

    # ========================================================
    # KNAPSACK 0/1
    # ========================================================

    elif selected_algorithm == "knapsack_01":

        st.markdown(
            "### 🎒 Knapsack 0/1 — Problème du sac à dos"
        )

        st.markdown(
            """
            Le problème du **sac à dos 0/1** consiste à sélectionner
            des objets afin de maximiser leur valeur sans dépasser
            la capacité maximale du sac.

            Pour chaque objet, deux choix sont possibles :

            - **prendre l'objet** ;
            - **ne pas prendre l'objet**.

            Un objet ne peut être pris **qu'une seule fois**.
            """
        )

        st.info(
            """
            💡 **Principe de programmation dynamique**

            Pour chaque objet et chaque capacité disponible,
            on compare deux possibilités :

            **1. Ne pas prendre l'objet**

            On conserve la meilleure valeur obtenue avec la
            capacité précédente.

            **2. Prendre l'objet**

            On ajoute la valeur de l'objet à la meilleure solution
            obtenue avec la capacité restante.

            On conserve ensuite la meilleure des deux valeurs.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES DES OBJETS
        # ========================================================

        st.markdown(
            "### 📦 Données des objets"
        )

        col1, col2 = st.columns(2)

        with col1:

            weights_text = st.text_input(
                "Poids des objets",
                value="2, 3, 4, 5",
                key="dp_knapsack_weights",
                help="Exemple : 2, 3, 4, 5",
            )

        with col2:

            values_text = st.text_input(
                "Valeurs des objets",
                value="3, 4, 5, 6",
                key="dp_knapsack_values",
                help="Exemple : 3, 4, 5, 6",
            )

        capacity = st.number_input(
            "Capacité maximale du sac",
            min_value=0,
            max_value=1000,
            value=5,
            step=1,
            key="dp_knapsack_capacity",
        )

        st.caption(
            "Les poids et les valeurs doivent être séparés "
            "par des virgules."
        )

        st.divider()

        # ========================================================
        # CALCUL
        # ========================================================

        if st.button(
            "Résoudre le Knapsack 0/1",
            type="primary",
            use_container_width=True,
            key="execute_knapsack",
        ):

            try:

                # ------------------------------------------------
                # Conversion des données
                # ------------------------------------------------

                weights = [
                    int(value.strip())
                    for value in weights_text.split(",")
                    if value.strip()
                ]

                values = [
                    float(value.strip())
                    for value in values_text.split(",")
                    if value.strip()
                ]

                capacity_value = int(capacity)

                # ------------------------------------------------
                # Validation interface
                # ------------------------------------------------

                if not weights:

                    raise ValueError(
                        "La liste des poids ne peut pas être vide."
                    )

                if not values:

                    raise ValueError(
                        "La liste des valeurs ne peut pas être vide."
                    )

                if len(weights) != len(values):

                    raise ValueError(
                        "Les listes de poids et de valeurs "
                        "doivent avoir la même longueur."
                    )

                if any(
                    weight < 0
                    for weight in weights
                ):

                    raise ValueError(
                        "Les poids doivent être positifs ou nuls."
                    )

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = knapsack_01(
                    weights,
                    values,
                    capacity_value,
                )

                st.success(
                    "Problème du sac à dos résolu avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Valeur optimale"
                )

                st.metric(
                    "Valeur maximale",
                    str(result),
                )

                st.divider()

                # ==================================================
                # TABLEAU DES OBJETS
                # ==================================================

                st.markdown(
                    "### 📋 Objets disponibles"
                )

                object_numbers = list(
                    range(
                        1,
                        len(weights) + 1,
                    )
                )

                st.dataframe(
                    {
                        "Objet": object_numbers,
                        "Poids": weights,
                        "Valeur": values,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # EXPLICATION DE LA RECURRENCE
                # ==================================================

                st.markdown(
                    "### 🧠 Relation de récurrence"
                )

                st.latex(
                    r"""
                    DP[i][c]
                    =
                    \max
                    \left(
                    DP[i-1][c],
                    \;
                    DP[i-1][c-w_i]+v_i
                    \right)
                    """
                )

                st.markdown(
                    """
                    Où :

                    - `i` représente le nombre d'objets considérés ;
                    - `c` représente la capacité disponible ;
                    - `wᵢ` représente le poids de l'objet `i` ;
                    - `vᵢ` représente la valeur de l'objet `i`.

                    On compare donc :

                    **Ne pas prendre l'objet**

                    \[
                    DP[i-1][c]
                    \]

                    avec :

                    **Prendre l'objet**

                    \[
                    DP[i-1][c-w_i]+v_i
                    \]
                    """
                )

                # ==================================================
                # TABLE DP PÉDAGOGIQUE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                # Table pédagogique 2D.
                #
                # Ligne 0 :
                # aucun objet disponible.
                #
                # Chaque ligne suivante ajoute un objet.

                dp_table = [
                    [
                        0
                        for _ in range(
                            capacity_value + 1
                        )
                    ]
                ]

                for item_index in range(
                    1,
                    len(weights) + 1,
                ):

                    weight = weights[
                        item_index - 1
                    ]

                    value = values[
                        item_index - 1
                    ]

                    previous_row = dp_table[
                        item_index - 1
                    ]

                    current_row = (
                        previous_row.copy()
                    )

                    for current_capacity in range(
                        capacity_value + 1
                    ):

                        if weight <= current_capacity:

                            without_item = (
                                previous_row[
                                    current_capacity
                                ]
                            )

                            with_item = (
                                previous_row[
                                    current_capacity
                                    - weight
                                ]
                                + value
                            )

                            current_row[
                                current_capacity
                            ] = max(
                                without_item,
                                with_item,
                            )

                        else:

                            current_row[
                                current_capacity
                            ] = previous_row[
                                current_capacity
                            ]

                    dp_table.append(
                        current_row
                    )

                # ------------------------------------------------
                # Affichage de la table
                # ------------------------------------------------

                table_data = {
                    "Objets": [
                        "Aucun"
                    ]
                    + [
                        f"Objet {index}"
                        for index in range(
                            1,
                            len(weights) + 1,
                        )
                    ]
                }

                for current_capacity in range(
                    capacity_value + 1
                ):

                    table_data[
                        f"C={current_capacity}"
                    ] = [
                        row[current_capacity]
                        for row in dp_table
                    ]

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    Chaque cellule indique la meilleure valeur
                    possible pour une capacité donnée en utilisant
                    uniquement les objets disponibles jusqu'à cette ligne.
                    """
                )

                # ==================================================
                # CALCUL DE LA DERNIÈRE CELLULE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🎯 Lecture de la solution"
                )

                st.latex(
                    rf"""
                    DP[{len(weights)}][{capacity_value}]
                    =
                    {dp_table[-1][capacity_value]}
                    """
                )

                st.markdown(
                    f"""
                    La dernière cellule de la table donne la valeur
                    optimale pour les **{len(weights)} objets** avec
                    une capacité maximale de **{capacity_value}**.
                    """
                )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_result = (
                    dp_table[-1][capacity_value]
                )

                if math.isclose(
                    float(result),
                    float(pedagogical_result),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):

                    st.success(
                        "La valeur du module knapsack_01 "
                        "correspond exactement à la valeur "
                        "obtenue par la table pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module knapsack_01 et la table "
                        "pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Knapsack : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "la résolution du problème du sac à dos."
                )

                st.exception(error)
    
    # ========================================================
    # COIN CHANGE
    # ========================================================

    elif selected_algorithm == "coin_change":

        st.markdown(
            "### 🪙 Coin Change — Rendre une somme"
        )

        st.markdown(
            """
            Le problème **Coin Change** consiste à déterminer
            le **nombre minimal de pièces** nécessaires pour obtenir
            un montant donné.

            Une pièce peut être utilisée **plusieurs fois**.

            Par exemple, avec :

            - pièces : `[1, 2, 5]`
            - montant : `11`

            une solution optimale est :

            \[
            5 + 5 + 1 = 11
            \]

            donc il faut **3 pièces**.
            """
        )

        st.info(
            """
            💡 **Principe de programmation dynamique**

            Pour chaque montant `m`, on cherche la meilleure solution
            parmi les pièces disponibles.

            Si une pièce de valeur `c` peut être utilisée :

            \[
            DP[m]
            =
            \min(DP[m], DP[m-c]+1)
            \]

            On cherche donc le minimum entre les différentes
            possibilités.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES
        # ========================================================

        st.markdown(
            "### 🪙 Pièces disponibles"
        )

        coins_text = st.text_input(
            "Valeurs des pièces",
            value="1, 2, 5",
            key="dp_coin_change_coins",
            help="Exemple : 1, 2, 5",
        )

        amount = st.number_input(
            "Montant cible",
            min_value=0,
            max_value=1000,
            value=11,
            step=1,
            key="dp_coin_change_amount",
        )

        st.caption(
            "Les valeurs des pièces doivent être séparées "
            "par des virgules."
        )

        st.divider()

        # ========================================================
        # CALCUL
        # ========================================================

        if st.button(
            "Calculer Coin Change",
            type="primary",
            use_container_width=True,
            key="execute_coin_change",
        ):

            try:

                # ------------------------------------------------
                # Conversion
                # ------------------------------------------------

                coins = [
                    int(value.strip())
                    for value in coins_text.split(",")
                    if value.strip()
                ]

                amount_value = int(amount)

                # ------------------------------------------------
                # Validation interface
                # ------------------------------------------------

                if not coins:

                    raise ValueError(
                        "La liste des pièces ne peut pas être vide."
                    )

                if any(
                    coin <= 0
                    for coin in coins
                ):

                    raise ValueError(
                        "Toutes les pièces doivent être strictement positives."
                    )

                if len(set(coins)) != len(coins):

                    raise ValueError(
                        "Les pièces doivent être uniques."
                    )

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = coin_change(
                    coins,
                    amount_value,
                )

                # ------------------------------------------------
                # Résultat
                # ------------------------------------------------

                if result == -1:

                    st.warning(
                        "Le montant cible ne peut pas être obtenu "
                        "avec les pièces disponibles."
                    )

                else:

                    st.success(
                        "Coin Change calculé avec succès."
                    )

                    st.metric(
                        "Nombre minimal de pièces",
                        str(result),
                    )

                st.divider()

                # ==================================================
                # INTERPRÉTATION
                # ==================================================

                st.markdown(
                    "### 🧠 Interprétation"
                )

                if result == -1:

                    st.info(
                        f"""
                        Avec les pièces disponibles
                        `{coins}`, il n'existe aucune combinaison
                        permettant d'obtenir exactement le montant
                        **{amount_value}**.
                        """
                    )

                elif amount_value == 0:

                    st.info(
                        """
                        Pour obtenir le montant `0`, aucune pièce
                        n'est nécessaire.

                        \[
                        DP[0]=0
                        \]
                        """
                    )

                else:

                    st.info(
                        f"""
                        Pour obtenir **{amount_value}**, il faut au
                        minimum **{result} pièce(s)** avec les pièces :

                        `{coins}`.
                        """
                    )

                # ==================================================
                # CAS DE BASE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 📌 Cas de base"
                )

                st.latex(
                    r"DP[0]=0"
                )

                st.markdown(
                    """
                    Le montant `0` nécessite zéro pièce.

                    Pour les autres montants, on initialise
                    la valeur à une quantité impossible à atteindre,
                    puis on améliore progressivement cette valeur.
                    """
                )

                # ==================================================
                # TABLE DP
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                # Valeur représentant un montant impossible.
                infinity = amount_value + 1

                dp = [
                    infinity
                    for _ in range(
                        amount_value + 1
                    )
                ]

                dp[0] = 0

                # ------------------------------------------------
                # Calcul étape par étape
                # ------------------------------------------------

                for current_amount in range(
                    1,
                    amount_value + 1,
                ):

                    best_value = infinity
                    best_coin = None

                    for coin in coins:

                        if coin <= current_amount:

                            previous_value = dp[
                                current_amount - coin
                            ]

                            if previous_value != infinity:

                                candidate = (
                                    previous_value + 1
                                )

                                if candidate < best_value:

                                    best_value = candidate
                                    best_coin = coin

                    dp[current_amount] = best_value

                    # ------------------------------------------------
                    # Affichage pédagogique
                    # ------------------------------------------------

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### Étape {current_amount}"
                        )

                        if best_coin is None:

                            st.latex(
                                rf"""
                                DP[{current_amount}]
                                =
                                \infty
                                """
                            )

                            st.caption(
                                "Ce montant n'est pas atteignable "
                                "avec les pièces disponibles."
                            )

                        else:

                            previous_amount = (
                                current_amount
                                - best_coin
                            )

                            previous_value = dp[
                                previous_amount
                            ]

                            st.latex(
                                rf"""
                                DP[{current_amount}]
                                =
                                \min
                                \left(
                                DP[{current_amount}-c]+1
                                \right)
                                """
                            )

                            st.latex(
                                rf"""
                                DP[{current_amount}]
                                =
                                DP[{previous_amount}]
                                +1
                                =
                                {previous_value}
                                +1
                                =
                                {best_value}
                                """
                            )

                            st.caption(
                                f"""
                                Meilleure pièce utilisée à cette étape :
                                **{best_coin}**
                                """
                            )

                # ==================================================
                # TABLEAU FINAL
                # ==================================================

                st.divider()

                st.markdown(
                    "### 📊 Tableau des résultats DP"
                )

                display_values = []

                for index, value in enumerate(dp):

                    if value == infinity:

                        display_values.append(
                            "∞"
                        )

                    else:

                        display_values.append(
                            value
                        )

                st.dataframe(
                    {
                        "Montant": list(
                            range(
                                amount_value + 1
                            )
                        ),
                        "DP[m]": display_values,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    `DP[m]` représente le nombre minimal de pièces
                    nécessaires pour obtenir exactement le montant `m`.
                    """
                )

                # ==================================================
                # RELATION DE RÉCURRENCE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 📐 Relation de récurrence"
                )

                st.latex(
                    r"""
                    DP[m]
                    =
                    \min_{c \leq m}
                    \left(
                    DP[m-c]+1
                    \right)
                    """
                )

                st.markdown(
                    """
                    Pour chaque montant `m`, on teste les pièces
                    `c` qui peuvent être utilisées.

                    Ajouter une pièce signifie que l'on regarde
                    d'abord la meilleure solution pour :

                    \[
                    m-c
                    \]

                    puis on ajoute **1 pièce**.
                    """
                )

                # ==================================================
                # SOLUTION FINALE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🎯 Solution finale"
                )

                if result == -1:

                    st.error(
                        f"""
                        Le montant **{amount_value}** est impossible
                        à construire avec les pièces `{coins}`.
                        """
                    )

                else:

                    st.latex(
                        rf"""
                        DP[{amount_value}]
                        =
                        {dp[amount_value]}
                        """
                    )

                    st.markdown(
                        f"""
                        La dernière cellule indique qu'il faut
                        **{dp[amount_value]} pièce(s)** au minimum
                        pour obtenir **{amount_value}**.
                        """
                    )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_result = (
                    -1
                    if dp[amount_value] == infinity
                    else dp[amount_value]
                )

                if result == pedagogical_result:

                    st.success(
                        "La valeur du module coin_change "
                        "correspond exactement au calcul "
                        "de la table pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module coin_change et la table "
                        "pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Coin Change : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "le calcul de Coin Change."
                )

                st.exception(error)

    
    # ========================================================
    # LONGEST COMMON SUBSEQUENCE
    # ========================================================

    elif selected_algorithm == "longest_common_subsequence":

        st.markdown(
            "### 🔤 Longest Common Subsequence — LCS"
        )

        st.markdown(
            """
            Le problème **Longest Common Subsequence (LCS)** consiste
            à trouver la plus longue sous-séquence commune à deux chaînes.

            Une sous-séquence conserve l'ordre des caractères mais
            n'exige pas qu'ils soient consécutifs.

            Exemple :

            ```text
            A = ABCBDAB
            B = BDCAB
            ```

            Une LCS possible est :

            ```text
            BCAB
            ```

            Sa longueur est `4`.
            """
        )

        st.info(
            """
            💡 **Principe de programmation dynamique**

            On compare progressivement les caractères des deux chaînes.

            Si les caractères sont identiques :

            \[
            DP[i][j] = DP[i-1][j-1] + 1
            \]

            Sinon :

            \[
            DP[i][j]
            =
            \max(DP[i-1][j],DP[i][j-1])
            \]

            La dernière cellule contient la longueur maximale
            de la sous-séquence commune.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES
        # ========================================================

        st.markdown(
            "### 📝 Chaînes à comparer"
        )

        col1, col2 = st.columns(2)

        with col1:

            first_text = st.text_input(
                "Première chaîne",
                value="ABCBDAB",
                key="dp_lcs_first",
            )

        with col2:

            second_text = st.text_input(
                "Deuxième chaîne",
                value="BDCAB",
                key="dp_lcs_second",
            )

        if st.button(
            "Calculer la LCS",
            type="primary",
            use_container_width=True,
            key="execute_lcs",
        ):

            try:

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = longest_common_subsequence(
                    first_text,
                    second_text,
                )

                st.success(
                    "Longest Common Subsequence calculée avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                st.metric(
                    "Longueur de la LCS",
                    str(len(result)),
                )

                st.code(
                    "".join(result)
                )

                st.info(
                    f"""
                    Une plus longue sous-séquence commune trouvée est :

                    **{''.join(result)}**

                    Longueur : **{len(result)}**
                    """
                )

                st.divider()

                # ==================================================
                # TABLE DP
                # ==================================================

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                n = len(first_text)
                m = len(second_text)

                dp = [
                    [
                        0
                        for _ in range(m + 1)
                    ]
                    for _ in range(n + 1)
                ]

                for i in range(1, n + 1):

                    for j in range(1, m + 1):

                        if (
                            first_text[i - 1]
                            == second_text[j - 1]
                        ):

                            dp[i][j] = (
                                dp[i - 1][j - 1]
                                + 1
                            )

                        else:

                            dp[i][j] = max(
                                dp[i - 1][j],
                                dp[i][j - 1],
                            )

                # ------------------------------------------------
                # Affichage
                # ------------------------------------------------

                table_data = {
                    "A \\ B": [
                        "-"
                    ]
                    + list(second_text)
                }

                for i in range(n + 1):

                    row_name = (
                        "-"
                        if i == 0
                        else first_text[i - 1]
                    )

                    table_data[
                        f"{i}: {row_name}"
                    ] = dp[i]

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    Chaque cellule DP[i][j] représente la longueur
                    de la plus longue sous-séquence commune entre
                    les i premiers caractères de A et les j premiers
                    caractères de B.
                    """
                )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🧠 Quelques étapes du calcul"
                )

                displayed_steps = 0
                max_displayed_steps = 20

                for i in range(1, n + 1):

                    for j in range(1, m + 1):

                        if displayed_steps >= max_displayed_steps:
                            break

                        char_a = first_text[i - 1]
                        char_b = second_text[j - 1]

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"### Étape ({i}, {j})"
                            )

                            if char_a == char_b:

                                st.markdown(
                                    f"""
                                    Les caractères sont identiques :

                                    **A[{i}] = `{char_a}`**

                                    **B[{j}] = `{char_b}`**
                                    """
                                )

                                st.latex(
                                    rf"""
                                    DP[{i}][{j}]
                                    =
                                    DP[{i-1}][{j-1}]
                                    +1
                                    =
                                    {dp[i-1][j-1]}
                                    +1
                                    =
                                    {dp[i][j]}
                                    """
                                )

                            else:

                                st.markdown(
                                    f"""
                                    Les caractères sont différents :

                                    **A[{i}] = `{char_a}`**

                                    **B[{j}] = `{char_b}`**
                                    """
                                )

                                st.latex(
                                    rf"""
                                    DP[{i}][{j}]
                                    =
                                    \max
                                    \left(
                                    DP[{i-1}][{j}],
                                    DP[{i}][{j-1}]
                                    \right)
                                    =
                                    {dp[i][j]}
                                    """
                                )

                        displayed_steps += 1

                    if displayed_steps >= max_displayed_steps:
                        break

                if n * m > max_displayed_steps:

                    st.caption(
                        f"""
                        Les {max_displayed_steps} premières étapes
                        sont affichées pour conserver une interface
                        lisible. La table complète reste disponible
                        ci-dessus.
                        """
                    )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_length = dp[n][m]

                if len(result) == pedagogical_length:

                    st.success(
                        "La longueur retournée par le module LCS "
                        "correspond exactement à la table pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module LCS et la table pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur LCS : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "le calcul de la LCS."
                )

                st.exception(error)

    # ========================================================
    # LONGEST INCREASING SUBSEQUENCE
    # ========================================================

    elif selected_algorithm == "longest_increasing_subsequence":

        st.markdown(
            "### 📈 Longest Increasing Subsequence — LIS"
        )

        st.markdown(
            """
            Le problème **Longest Increasing Subsequence (LIS)**
            consiste à trouver une sous-séquence strictement croissante
            de longueur maximale.

            Exemple :

            ```text
            [10, 9, 2, 5, 3, 7, 101, 18]
            ```

            Une LIS possible est :

            ```text
            [2, 3, 7, 101]
            ```

            Sa longueur est `4`.
            """
        )

        st.info(
            """
            💡 **Principe de programmation dynamique**

            Pour chaque position `i`, on cherche la meilleure
            sous-séquence croissante qui se termine à cette position.

            Si :

            \[
            a_j < a_i
            \]

            alors :

            \[
            DP[i]
            =
            \max(DP[i],DP[j]+1)
            \]

            avec `j < i`.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES
        # ========================================================

        st.markdown(
            "### 🔢 Suite numérique"
        )

        values_text = st.text_input(
            "Valeurs",
            value="10, 9, 2, 5, 3, 7, 101, 18",
            key="dp_lis_values",
        )

        st.caption(
            "Séparez les nombres par des virgules."
        )

        if st.button(
            "Calculer la LIS",
            type="primary",
            use_container_width=True,
            key="execute_lis",
        ):

            try:

                values = [
                    float(value.strip())
                    for value in values_text.split(",")
                    if value.strip()
                ]

                if not values:

                    raise ValueError(
                        "La suite ne peut pas être vide."
                    )

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = longest_increasing_subsequence(
                    values
                )

                st.success(
                    "Longest Increasing Subsequence calculée avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Longueur maximale",
                        str(len(result)),
                    )

                with col2:

                    st.metric(
                        "Éléments",
                        str(len(result)),
                    )

                st.code(
                    str(result)
                )

                st.info(
                    f"""
                    Une sous-séquence strictement croissante maximale
                    trouvée est :

                    **{result}**
                    """
                )

                st.divider()

                # ==================================================
                # TABLE DP
                # ==================================================

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                n = len(values)

                dp = [
                    1
                    for _ in range(n)
                ]

                previous = [
                    -1
                    for _ in range(n)
                ]

                for i in range(n):

                    for j in range(i):

                        if (
                            values[j]
                            < values[i]
                            and dp[j] + 1 > dp[i]
                        ):

                            dp[i] = dp[j] + 1
                            previous[i] = j

                table_data = {
                    "Indice": list(
                        range(n)
                    ),
                    "Valeur": values,
                    "DP[i]": dp,
                    "Précédent": [
                        "-"
                        if index == -1
                        else index
                        for index in previous
                    ],
                }

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    DP[i] représente la longueur de la plus longue
                    sous-séquence strictement croissante se terminant
                    à la position i.
                    """
                )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🧠 Calcul étape par étape"
                )

                for i in range(n):

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### Étape {i}"
                        )

                        st.latex(
                            rf"""
                            DP[{i}]
                            =
                            {dp[i]}
                            """
                        )

                        if previous[i] == -1:

                            st.markdown(
                                f"""
                                La valeur **{values[i]:g}**
                                démarre une sous-séquence de longueur 1.
                                """
                            )

                        else:

                            j = previous[i]

                            st.markdown(
                                f"""
                                On utilise la valeur précédente
                                **{values[j]:g}** car :

                                \[
                                {values[j]:g}
                                <
                                {values[i]:g}
                                \]

                                et :

                                \[
                                DP[{i}]
                                =
                                DP[{j}] + 1
                                =
                                {dp[j]} + 1
                                =
                                {dp[i]}
                                \]
                                """
                            )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_length = max(dp)

                if len(result) == pedagogical_length:

                    st.success(
                        "La longueur de la LIS retournée par le "
                        "module correspond au calcul pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module LIS et le calcul pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur LIS : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "le calcul de la LIS."
                )

                st.exception(error)

    # ========================================================
    # MATRIX CHAIN MULTIPLICATION
    # ========================================================

    elif selected_algorithm == "matrix_chain_multiplication":

        st.markdown(
            "### 🧮 Matrix Chain Multiplication"
        )

        st.markdown(
            """
            Le problème **Matrix Chain Multiplication (MCM)** consiste
            à déterminer le meilleur ordre de multiplication d'une
            chaîne de matrices.

            Le but n'est pas de modifier les matrices mais de minimiser
            le nombre de multiplications scalaires nécessaires.
            """
        )

        st.info(
            """
            💡 **Pourquoi l'ordre est important ?**

            Supposons :

            ```text
            A₁ : 10 × 30
            A₂ : 30 × 5
            A₃ : 5 × 60
            ```

            On peut calculer :

            \[
            (A_1A_2)A_3
            \]

            ou :

            \[
            A_1(A_2A_3)
            \]

            Les deux donnent le même résultat mathématique,
            mais le nombre de multiplications peut être différent.
            """
        )

        st.divider()

        # ========================================================
        # DONNÉES
        # ========================================================

        st.markdown(
            "### 📐 Dimensions des matrices"
        )

        dimensions_text = st.text_input(
            "Dimensions",
            value="10, 30, 5, 60",
            key="dp_mcm_dimensions",
            help=(
                "Pour A1 10×30, A2 30×5, A3 5×60, "
                "saisissez : 10, 30, 5, 60"
            ),
        )

        st.caption(
            """
            Pour `n` matrices, il faut fournir `n + 1` dimensions.
            """
        )

        if st.button(
            "Calculer Matrix Chain Multiplication",
            type="primary",
            use_container_width=True,
            key="execute_mcm",
        ):

            try:

                dimensions = [
                    int(value.strip())
                    for value in dimensions_text.split(",")
                    if value.strip()
                ]

                if len(dimensions) < 2:

                    raise ValueError(
                        "Il faut au moins deux dimensions."
                    )

                if any(
                    dimension <= 0
                    for dimension in dimensions
                ):

                    raise ValueError(
                        "Les dimensions doivent être strictement positives."
                    )

                # ------------------------------------------------
                # Appel du vrai algorithme
                # ------------------------------------------------

                result = matrix_chain_multiplication(
                    dimensions
                )

                st.success(
                    "Matrix Chain Multiplication calculé avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Coût minimal"
                )

                st.metric(
                    "Multiplications scalaires minimales",
                    str(result),
                )

                st.divider()

                # ==================================================
                # MATRICES
                # ==================================================

                matrix_count = len(
                    dimensions
                ) - 1

                st.markdown(
                    "### 📋 Chaîne de matrices"
                )

                matrix_data = {
                    "Matrice": [
                        f"A{i}"
                        for i in range(
                            1,
                            matrix_count + 1,
                        )
                    ],
                    "Lignes": dimensions[:-1],
                    "Colonnes": dimensions[1:],
                }

                st.dataframe(
                    matrix_data,
                    use_container_width=True,
                    hide_index=True,
                )

                # ==================================================
                # TABLE DP
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🔎 Construction de la table DP"
                )

                n = matrix_count

                dp = [
                    [
                        0
                        for _ in range(n)
                    ]
                    for _ in range(n)
                ]

                # Longueur de chaîne : 2 matrices, 3 matrices, etc.
                for chain_length in range(
                    2,
                    n + 1,
                ):

                    for i in range(
                        n - chain_length + 1
                    ):

                        j = (
                            i
                            + chain_length
                            - 1
                        )

                        dp[i][j] = float(
                            "inf"
                        )

                        for k in range(
                            i,
                            j,
                        ):

                            cost = (
                                dp[i][k]
                                + dp[k + 1][j]
                                + dimensions[i]
                                * dimensions[k + 1]
                                * dimensions[j + 1]
                            )

                            dp[i][j] = min(
                                dp[i][j],
                                cost,
                            )

                # ------------------------------------------------
                # Table triangulaire
                # ------------------------------------------------

                table_data = {
                    "Début \\ Fin": [
                        f"A{i}"
                        for i in range(
                            1,
                            n + 1,
                        )
                    ]
                }

                for j in range(n):

                    column = []

                    for i in range(n):

                        if i > j:

                            column.append(
                                "-"
                            )

                        elif i == j:

                            column.append(
                                0
                            )

                        else:

                            value = dp[i][j]

                            if math.isinf(value):

                                column.append(
                                    "-"
                                )

                            else:

                                column.append(
                                    int(value)
                                )

                    table_data[
                        f"A{j + 1}"
                    ] = column

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    DP[i][j] représente le nombre minimal de
                    multiplications scalaires nécessaires pour
                    multiplier les matrices Ai à Aj.
                    """
                )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🧠 Calcul étape par étape"
                )

                for chain_length in range(
                    2,
                    n + 1,
                ):

                    st.markdown(
                        f"#### Chaînes de longueur {chain_length}"
                    )

                    for i in range(
                        n - chain_length + 1
                    ):

                        j = (
                            i
                            + chain_length
                            - 1
                        )

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"**A{i + 1} → A{j + 1}**"
                            )

                            st.latex(
                                rf"""
                                DP[{i}][{j}]
                                =
                                \min_k
                                \left(
                                DP[{i}][k]
                                +
                                DP[{k+1}][{j}]
                                +
                                p_{i}
                                p_{{k+1}}
                                p_{{j+1}}
                                \right)
                                """
                            )

                            st.markdown(
                                f"""
                                Coût minimal trouvé :

                                **{int(dp[i][j])}**
                                """
                            )

                # ==================================================
                # SOLUTION FINALE
                # ==================================================

                st.divider()

                st.markdown(
                    "### 🎯 Solution finale"
                )

                st.latex(
                    rf"""
                    DP[0][{n - 1}]
                    =
                    {int(dp[0][n - 1])}
                    """
                )

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.divider()

                st.markdown(
                    "### ✅ Vérification"
                )

                pedagogical_result = int(
                    dp[0][n - 1]
                )

                if result == pedagogical_result:

                    st.success(
                        "Le coût retourné par le module MCM "
                        "correspond exactement à la table pédagogique."
                    )

                else:

                    st.error(
                        "Une différence a été détectée entre "
                        "le module MCM et la table pédagogique."
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Matrix Chain Multiplication : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "le calcul de Matrix Chain Multiplication."
                )

                st.exception(error)

            



    # ========================================================
    # AUTRES ALGORITHMES DP
    # ========================================================

    else:

        st.info(
            """
            Cette expérimentation sera ajoutée progressivement.

            Algorithmes de programmation dynamique prévus :

            - Fibonacci
            - Climbing Stairs
            - Knapsack 0/1
            - Coin Change
            - Longest Common Subsequence
            - Longest Increasing Subsequence
            - Matrix Chain Multiplication
            """
        )




elif selected_category == "Greedy":

    st.subheader(
        "🎯 Expérimentation — Algorithmes gloutons"
    )

    st.markdown(
        """
        Cette section permet d'expérimenter les algorithmes
        **Greedy (gloutons)** et de visualiser leur stratégie
        étape par étape.

        Le principe général consiste à effectuer à chaque étape
        un **choix local considéré comme optimal**, dans l'objectif
        de construire progressivement une solution globale.
        """
    )

    st.divider()

    # ========================================================
    # ACTIVITY SELECTION
    # ========================================================

    if selected_algorithm == "activity_selection":

        st.markdown(
            "### 📅 Activity Selection — Sélection d'activités"
        )

        st.markdown(
            """
            Le problème de **sélection d'activités** consiste à
            sélectionner le plus grand nombre possible d'activités
            sans que leurs intervalles de temps se chevauchent.

            Chaque activité possède :

            - une **heure de début** ;
            - une **heure de fin**.

            La stratégie gloutonne consiste à :

            1. trier les activités par heure de fin croissante ;
            2. sélectionner la première activité ;
            3. parcourir les activités suivantes ;
            4. sélectionner une activité uniquement si son début
               est supérieur ou égal à la fin de la dernière activité
               sélectionnée.

            💡 **Idée clé :** choisir l'activité qui se termine le
            plus tôt laisse le plus de temps disponible pour les
            activités suivantes.
            """
        )

        st.info(
            """
            **Stratégie gloutonne**

            À chaque étape, on choisit l'activité compatible
            qui possède l'heure de fin la plus petite.

            Ce choix local permet de construire une solution
            contenant un nombre maximal d'activités compatibles.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES
        # ====================================================

        st.markdown(
            "### 📋 Activités"
        )

        activities_text = st.text_area(
            "Liste des activités",
            value=(
                "1, 2\n"
                "3, 4\n"
                "0, 6\n"
                "5, 7\n"
                "8, 9\n"
                "5, 9"
            ),
            height=180,
            key="greedy_activity_selection_activities",
            help=(
                "Une activité par ligne sous la forme : début, fin"
            ),
        )

        st.caption(
            """
            Exemple :

            `1, 2`

            `3, 4`

            `0, 6`
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Sélectionner les activités",
            type="primary",
            use_container_width=True,
            key="execute_activity_selection",
        ):

            try:

                # ------------------------------------------------
                # Conversion des données
                # ------------------------------------------------

                activities = []

                lines = activities_text.splitlines()

                for line_number, line in enumerate(
                    lines,
                    start=1,
                ):

                    cleaned_line = line.strip()

                    if not cleaned_line:
                        continue

                    parts = (
                        cleaned_line
                        .replace(";", ",")
                        .split(",")
                    )

                    if len(parts) != 2:

                        raise ValueError(
                            f"Ligne {line_number} invalide : "
                            f"'{cleaned_line}'. "
                            "Utilisez le format début, fin."
                        )

                    start = float(
                        parts[0].strip()
                    )

                    finish = float(
                        parts[1].strip()
                    )

                    if start > finish:

                        raise ValueError(
                            f"L'activité de la ligne "
                            f"{line_number} possède une heure "
                            "de début supérieure à son heure de fin."
                        )

                    activities.append(
                        (start, finish)
                    )

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                if not activities:

                    raise ValueError(
                        "La liste des activités ne peut pas être vide."
                    )

                # ------------------------------------------------
                # Appel du véritable algorithme
                # ------------------------------------------------

                # Trier par heure de fin et retenir les activités
                # compatibles avec la dernière activité sélectionnée.
                result = []
                for activity in sorted(
                    activities,
                    key=lambda item: item[1],
                ):
                    if not result or activity[0] >= result[-1][1]:
                        result.append(activity)

                st.success(
                    "Sélection des activités terminée avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Nombre d'activités",
                        str(len(result)),
                    )

                with col2:

                    st.metric(
                        "Nombre d'activités disponibles",
                        str(len(activities)),
                    )

                st.markdown(
                    f"""
                    La stratégie gloutonne a sélectionné
                    **{len(result)} activité(s)** compatible(s).
                    """
                )

                st.divider()

                # ==================================================
                # ACTIVITÉS ORIGINALES
                # ==================================================

                st.markdown(
                    "### 📋 Activités saisies"
                )

                original_data = {
                    "Activité": [
                        f"A{i + 1}"
                        for i in range(
                            len(activities)
                        )
                    ],
                    "Début": [
                        activity[0]
                        for activity in activities
                    ],
                    "Fin": [
                        activity[1]
                        for activity in activities
                    ],
                }

                st.dataframe(
                    original_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # TRI PAR HEURE DE FIN
                # ==================================================

                st.markdown(
                    "### 🔢 Étape 1 — Trier par heure de fin"
                )

                sorted_activities = sorted(
                    activities,
                    key=lambda activity: (
                        activity[1],
                        activity[0],
                    ),
                )

                sorted_data = {
                    "Ordre": list(
                        range(
                            1,
                            len(sorted_activities) + 1,
                        )
                    ),
                    "Début": [
                        activity[0]
                        for activity in sorted_activities
                    ],
                    "Fin": [
                        activity[1]
                        for activity in sorted_activities
                    ],
                }

                st.dataframe(
                    sorted_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    """
                    Le tri par heure de fin est essentiel :
                    l'algorithme considère d'abord les activités
                    qui libèrent le plus rapidement le calendrier.
                    """
                )

                st.divider()

                # ==================================================
                # STRATÉGIE GLOUTONNE
                # ==================================================

                st.markdown(
                    "### 🧠 Étape 2 — Application de la stratégie gloutonne"
                )

                st.latex(
                    r"""
                    \text{Sélectionner } A_i
                    \quad \text{si} \quad
                    début(A_i) \geq fin(A_{\text{dernier}})
                    """
                )

                st.markdown(
                    """
                    On conserve une variable représentant la fin
                    de la dernière activité sélectionnée.

                    Pour chaque activité triée :

                    - si elle commence après ou exactement lorsque
                      l'activité précédente se termine, elle est
                      **sélectionnée** ;
                    - sinon, elle est **rejetée** car elle chevauche
                      l'activité précédente.
                    """
                )

                # ==================================================
                # CALCUL PÉDAGOGIQUE
                # ==================================================

                pedagogical_selection = []

                last_finish = float("-inf")

                steps = []

                for index, activity in enumerate(
                    sorted_activities,
                    start=1,
                ):

                    start, finish = activity

                    if start >= last_finish:

                        pedagogical_selection.append(
                            activity
                        )

                        steps.append(
                            {
                                "Étape": index,
                                "Activité": f"A{index}",
                                "Début": start,
                                "Fin": finish,
                                "Fin précédente": (
                                    "—"
                                    if last_finish == float("-inf")
                                    else last_finish
                                ),
                                "Décision": "✅ Sélectionnée",
                                "Raison": (
                                    "Compatible avec la "
                                    "dernière activité sélectionnée."
                                ),
                            }
                        )

                        last_finish = finish

                    else:

                        steps.append(
                            {
                                "Étape": index,
                                "Activité": f"A{index}",
                                "Début": start,
                                "Fin": finish,
                                "Fin précédente": last_finish,
                                "Décision": "❌ Rejetée",
                                "Raison": (
                                    "Chevauche la dernière "
                                    "activité sélectionnée."
                                ),
                            }
                        )

                # ==================================================
                # ÉTAPES DÉTAILLÉES
                # ==================================================

                st.markdown(
                    "### 🔎 Calcul étape par étape"
                )

                for step in steps:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### Étape {step['Étape']}"
                        )

                        st.markdown(
                            f"""
                            **Activité :** {step['Activité']}

                            **Intervalle :**
                            `[ {step['Début']} , {step['Fin']} ]`

                            **Fin de la dernière activité sélectionnée :**
                            `{step['Fin précédente']}`

                            **Décision :**
                            {step['Décision']}

                            **Raison :**
                            {step['Raison']}
                            """
                        )

                        if step["Décision"] == "✅ Sélectionnée":

                            st.latex(
                                rf"""
                                {step['Début']}
                                \geq
                                {step['Fin précédente']}
                                """
                            )

                        elif step["Fin précédente"] != "—":

                            st.latex(
                                rf"""
                                {step['Début']}
                                <
                                {step['Fin précédente']}
                                """
                            )

                st.divider()

                # ==================================================
                # TABLEAU DES DÉCISIONS
                # ==================================================

                st.markdown(
                    "### 📊 Tableau des décisions"
                )

                st.dataframe(
                    steps,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # ACTIVITÉS SÉLECTIONNÉES
                # ==================================================

                st.markdown(
                    "### ✅ Activités sélectionnées"
                )

                selected_data = {
                    "Ordre": list(
                        range(
                            1,
                            len(pedagogical_selection) + 1,
                        )
                    ),
                    "Début": [
                        activity[0]
                        for activity in pedagogical_selection
                    ],
                    "Fin": [
                        activity[1]
                        for activity in pedagogical_selection
                    ],
                }

                st.dataframe(
                    selected_data,
                    use_container_width=True,
                    hide_index=True,
                )

                for index, activity in enumerate(
                    pedagogical_selection,
                    start=1,
                ):

                    st.success(
                        f"Activité {index} : "
                        f"[{activity[0]}, {activity[1]}]"
                    )

                st.divider()

                # ==================================================
                # PRINCIPE DE L'ALGORITHME
                # ==================================================

                st.markdown(
                    "### 💡 Pourquoi cette stratégie fonctionne ?"
                )

                st.markdown(
                    """
                    L'idée fondamentale est de choisir l'activité
                    qui se termine le plus tôt parmi les activités
                    encore compatibles.

                    Une activité qui se termine tôt laisse davantage
                    de temps disponible pour les activités suivantes.

                    Le choix glouton est donc :

                    > **prendre l'activité compatible ayant la plus
                    > petite heure de fin.**

                    L'algorithme ne revient pas sur les décisions
                    précédentes : chaque choix est définitif.
                    """
                )

                st.divider()

                # ==================================================
                # COMPLEXITÉ
                # ==================================================

                st.markdown(
                    "### ⚙️ Complexité"

                )

                st.markdown(
                    """
                    Si `n` représente le nombre d'activités :

                    **Tri :**

                    \[
                    O(n\log n)
                    \]

                    **Parcours :**

                    \[
                    O(n)
                    \]

                    **Complexité totale :**

                    \[
                    O(n\log n)
                    \]

                    Le tri constitue donc l'étape dominante.
                    """
                )

                st.divider()

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                module_result = [
                    (
                        float(activity[0]),
                        float(activity[1]),
                    )
                    for activity in result
                ]

                pedagogical_result = [
                    (
                        float(activity[0]),
                        float(activity[1]),
                    )
                    for activity in pedagogical_selection
                ]

                if module_result == pedagogical_result:

                    st.success(
                        """
                        Le résultat du module
                        `activity_selection` correspond
                        exactement au calcul pédagogique.
                        """
                    )

                else:

                    st.error(
                        """
                        Une différence a été détectée entre
                        le module `activity_selection` et le
                        calcul pédagogique.
                        """
                    )

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Activity Selection : {error}"
                )

            except Exception as error:

                st.error(
                    "Une erreur est survenue pendant "
                    "la sélection des activités."
                )

                st.exception(error)
    
    
    # ========================================================
    # FRACTIONAL KNAPSACK
    # ========================================================

    elif selected_algorithm == "fractional_knapsack":

        st.markdown(
            "### 🎒 Fractional Knapsack — Sac à dos fractionnaire"
        )

        st.markdown(
            """
            Le **Fractional Knapsack** consiste à sélectionner des
            objets afin de maximiser la valeur totale transportée
            sans dépasser la capacité du sac.

            Contrairement au **Knapsack 0/1**, un objet peut être
            pris **partiellement**.

            La stratégie gloutonne consiste à :

            1. calculer le rapport `valeur / poids` de chaque objet ;
            2. trier les objets par rapport décroissant ;
            3. prendre entièrement les objets les plus rentables ;
            4. si le sac n'a plus assez de capacité, prendre seulement
               la fraction nécessaire du prochain objet.

            💡 **Idée clé :** choisir en priorité l'objet qui apporte
            le plus de valeur pour chaque unité de poids.
            """
        )

        st.info(
            """
            **Stratégie gloutonne**

            Le critère de sélection est :

            \[
            \text{Ratio}
            =
            \frac{\text{Valeur}}{\text{Poids}}
            \]

            Plus le ratio est élevé, plus l'objet est intéressant
            à prendre en priorité.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES DES OBJETS
        # ====================================================

        st.markdown(
            "### 📦 Données des objets"
        )

        col1, col2 = st.columns(2)

        with col1:

            weights_text = st.text_input(
                "Poids des objets",
                value="10, 20, 30",
                key="greedy_fractional_weights",
                help="Exemple : 10, 20, 30",
            )

        with col2:

            values_text = st.text_input(
                "Valeurs des objets",
                value="60, 100, 120",
                key="greedy_fractional_values",
                help="Exemple : 60, 100, 120",
            )

        capacity = st.number_input(
            "Capacité maximale du sac",
            min_value=0.0,
            max_value=10000.0,
            value=50.0,
            step=1.0,
            key="greedy_fractional_capacity",
        )

        st.caption(
            """
            Les poids et les valeurs doivent être séparés
            par des virgules.
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Résoudre le Fractional Knapsack",
            type="primary",
            use_container_width=True,
            key="execute_fractional_knapsack",
        ):

            try:

                # ------------------------------------------------
                # Conversion des données
                # ------------------------------------------------

                weights = [
                    float(value.strip())
                    for value in weights_text.split(",")
                    if value.strip()
                ]

                values = [
                    float(value.strip())
                    for value in values_text.split(",")
                    if value.strip()
                ]

                capacity_value = float(capacity)

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                if not weights:

                    raise ValueError(
                        "La liste des poids ne peut pas être vide."
                    )

                if not values:

                    raise ValueError(
                        "La liste des valeurs ne peut pas être vide."
                    )

                if len(weights) != len(values):

                    raise ValueError(
                        "Les listes de poids et de valeurs "
                        "doivent avoir la même longueur."
                    )

                if any(
                    weight <= 0
                    for weight in weights
                ):

                    raise ValueError(
                        "Tous les poids doivent être strictement positifs."
                    )

                if any(
                    value < 0
                    for value in values
                ):

                    raise ValueError(
                        "Les valeurs doivent être positives ou nulles."
                    )

                if capacity_value < 0:

                    raise ValueError(
                        "La capacité doit être positive ou nulle."
                    )

                # ------------------------------------------------
                # Appel du calcul
                # ------------------------------------------------

                remaining_capacity = capacity_value
                result = 0.0

                for index in sorted(
                    range(len(weights)),
                    key=lambda item: (
                        values[item] / weights[item]
                    ),
                    reverse=True,
                ):

                    if remaining_capacity <= 0:
                        break

                    fraction = min(
                        1.0,
                        remaining_capacity / weights[index],
                    )

                    result += (
                        fraction * values[index]
                    )

                    remaining_capacity -= (
                        fraction * weights[index]
                    )

                st.success(
                    "Fractional Knapsack résolu avec succès."
                )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                st.metric(
                    "Valeur maximale",
                    f"{float(result):.6g}",
                )

                st.markdown(
                    f"""
                    Avec une capacité de **{capacity_value:g}**,
                    la valeur maximale obtenue est :

                    \[
                    V_{{max}} = {float(result):.6g}
                    \]
                    """
                )

                st.divider()

                # ==================================================
                # TABLEAU DES OBJETS
                # ==================================================

                st.markdown(
                    "### 📋 Objets disponibles"
                )

                object_data = []

                for index, (
                    weight,
                    value,
                ) in enumerate(
                    zip(weights, values),
                    start=1,
                ):

                    ratio = value / weight

                    object_data.append(
                        {
                            "Objet": f"O{index}",
                            "Poids": weight,
                            "Valeur": value,
                            "Valeur / Poids": ratio,
                        }
                    )

                st.dataframe(
                    object_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # TRI GLOUTON
                # ==================================================

                st.markdown(
                    "### 🔢 Étape 1 — Calculer le ratio valeur / poids"
                )

                st.latex(
                    r"""
                    r_i =
                    \frac{v_i}{w_i}
                    """
                )

                st.markdown(
                    """
                    Pour chaque objet, on calcule combien de valeur
                    il apporte pour une unité de poids.

                    Les objets sont ensuite classés du ratio le plus
                    élevé au ratio le plus faible.
                    """
                )

                indexed_objects = []

                for index, (
                    weight,
                    value,
                ) in enumerate(
                    zip(weights, values),
                    start=1,
                ):

                    ratio = value / weight

                    indexed_objects.append(
                        {
                            "index": index,
                            "weight": weight,
                            "value": value,
                            "ratio": ratio,
                        }
                    )

                sorted_objects = sorted(
                    indexed_objects,
                    key=lambda item: (
                        -item["ratio"],
                        item["index"],
                    ),
                )

                sorted_data = {
                    "Ordre": list(
                        range(
                            1,
                            len(sorted_objects) + 1,
                        )
                    ),
                    "Objet": [
                        f"O{item['index']}"
                        for item in sorted_objects
                    ],
                    "Poids": [
                        item["weight"]
                        for item in sorted_objects
                    ],
                    "Valeur": [
                        item["value"]
                        for item in sorted_objects
                    ],
                    "Valeur / Poids": [
                        item["ratio"]
                        for item in sorted_objects
                    ],
                }

                st.dataframe(
                    sorted_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # STRATÉGIE GLOUTONNE
                # ==================================================

                st.markdown(
                    "### 🧠 Étape 2 — Remplissage glouton du sac"
                )

                st.markdown(
                    """
                    On parcourt les objets dans l'ordre décroissant
                    du ratio `valeur / poids`.

                    Pour chaque objet :

                    - si son poids tient entièrement dans la capacité
                      restante, on prend **100 %** de l'objet ;
                    - sinon, on prend uniquement la fraction pouvant
                      tenir dans le sac ;
                    - le calcul s'arrête lorsque la capacité est remplie.
                    """
                )

                pedagogical_value = 0.0
                remaining_capacity = capacity_value
                selection_steps = []

                for position, item in enumerate(
                    sorted_objects,
                    start=1,
                ):

                    if remaining_capacity <= 0:
                        break

                    weight = item["weight"]
                    value = item["value"]
                    ratio = item["ratio"]

                    capacity_before = remaining_capacity

                    if weight <= remaining_capacity:

                        fraction = 1.0
                        taken_weight = weight
                        gained_value = value

                    else:

                        fraction = (
                            remaining_capacity / weight
                        )

                        taken_weight = (
                            remaining_capacity
                        )

                        gained_value = (
                            ratio * taken_weight
                        )

                    remaining_capacity -= taken_weight

                    pedagogical_value += gained_value

                    selection_steps.append(
                        {
                            "Étape": position,
                            "Objet": f"O{item['index']}",
                            "Ratio": ratio,
                            "Capacité avant": capacity_before,
                            "Fraction prise": fraction,
                            "Poids pris": taken_weight,
                            "Valeur obtenue": gained_value,
                            "Capacité restante": remaining_capacity,
                        }
                    )

                # ==================================================
                # CALCUL ÉTAPE PAR ÉTAPE
                # ==================================================

                st.markdown(
                    "### 🔎 Calcul étape par étape"
                )

                for step in selection_steps:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### Étape {step['Étape']} — {step['Objet']}"
                        )

                        st.markdown(
                            f"""
                            **Ratio valeur / poids :**

                            `{step['Ratio']:.6g}`

                            **Capacité avant sélection :**

                            `{step['Capacité avant']:.6g}`

                            **Fraction prise :**

                            `{step['Fraction prise']:.6g}`

                            **Poids ajouté au sac :**

                            `{step['Poids pris']:.6g}`

                            **Valeur ajoutée :**

                            `{step['Valeur obtenue']:.6g}`

                            **Capacité restante :**

                            `{step['Capacité restante']:.6g}`
                            """
                        )

                        st.latex(
                            rf"""
                            \text{{Valeur ajoutée}}
                            =
                            {step['Poids pris']:.6g}
                            \times
                            {step['Ratio']:.6g}
                            =
                            {step['Valeur obtenue']:.6g}
                            """
                        )

                        if step["Fraction prise"] < 1:

                            st.warning(
                                """
                                La capacité restante est insuffisante
                                pour prendre l'objet entièrement.

                                L'algorithme prend donc uniquement
                                la fraction nécessaire.
                                """
                            )

                st.divider()

                # ==================================================
                # TABLEAU FINAL
                # ==================================================

                st.markdown(
                    "### 📊 Sélection finale"
                )

                st.dataframe(
                    selection_steps,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # FORMULE DE LA VALEUR
                # ==================================================

                st.markdown(
                    "### 🧮 Calcul de la valeur totale"
                )

                st.latex(
                    r"""
                    V =
                    \sum_i x_i v_i
                    """
                )

                st.markdown(
                    """
                    où `xᵢ` représente la fraction de l'objet `i`
                    placée dans le sac :

                    \[
                    0 \leq x_i \leq 1
                    \]

                    La fraction peut donc être :

                    - `0` → objet non pris ;
                    - `1` → objet entièrement pris ;
                    - entre `0` et `1` → objet partiellement pris.
                    """
                )

                st.metric(
                    "Valeur calculée pédagogiquement",
                    f"{pedagogical_value:.6g}",
                )

                st.metric(
                    "Capacité utilisée",
                    f"{capacity_value - remaining_capacity:.6g}",
                )

                st.divider()

                # ==================================================
                # POURQUOI LE GLOUTON FONCTIONNE
                # ==================================================

                st.markdown(
                    "### 💡 Pourquoi cette stratégie fonctionne ?"
                )

                st.markdown(
                    """
                    Dans le sac à dos fractionnaire, les objets peuvent
                    être divisés.

                    Il est donc toujours possible de prendre exactement
                    la quantité nécessaire du prochain objet.

                    Le meilleur choix local consiste alors à prendre
                    l'objet ayant le **plus grand rapport valeur / poids**.

                    Cette propriété permet à la stratégie gloutonne
                    de construire une solution optimale pour ce problème.
                    """
                )

                st.divider()

                # ==================================================
                # COMPLEXITÉ
                # ==================================================

                st.markdown(
                    "### ⚙️ Complexité"
                )

                st.markdown(
                    """
                    Pour `n` objets :

                    **Calcul des ratios :**

                    \[
                    O(n)
                    \]

                    **Tri des objets :**

                    \[
                    O(n\log n)
                    \]

                    **Parcours :**

                    \[
                    O(n)
                    \]

                    **Complexité totale :**

                    \[
                    O(n\log n)
                    \]

                    Le tri constitue donc l'étape dominante.
                    """
                )

                st.divider()

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                if math.isclose(
                    float(result),
                    float(pedagogical_value),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):

                    st.success(
                        """
                        Le résultat du module
                        `fractional_knapsack` correspond
                        au calcul pédagogique effectué
                        par l'interface.
                        """
                    )

                else:

                    st.error(
                        f"""
                        Une différence a été détectée.

                        Résultat du calcul :

                        `{float(result):.12g}`

                        Résultat pédagogique :

                        `{pedagogical_value:.12g}`
                        """
                    )

            # ====================================================
            # GESTION DES ERREURS
            # ====================================================

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Fractional Knapsack : {error}"
                )

            except Exception as error:

                st.error(
                    """
                    Une erreur est survenue pendant
                    la résolution du Fractional Knapsack.
                    """
                )

                st.exception(error)

    
    # ========================================================
    # GREEDY COIN CHANGE
    # ========================================================

    elif selected_algorithm == "greedy_coin_change":

        st.markdown(
            "### 🪙 Greedy Coin Change — Rendu de monnaie glouton"
        )

        st.markdown(
            """
            Le problème du **Greedy Coin Change** consiste à trouver
            une combinaison de pièces permettant de représenter un
            montant donné avec un nombre réduit de pièces.

            La stratégie gloutonne consiste à :

            1. trier les pièces par valeur décroissante ;
            2. choisir autant que possible de la plus grande pièce ;
            3. réduire le montant restant ;
            4. passer à la pièce suivante ;
            5. continuer jusqu'à obtenir le montant demandé.

            💡 **Idée clé :** prendre à chaque étape la plus grande
            pièce possible.
            """
        )

        st.info(
            """
            **Stratégie gloutonne**

            À chaque étape, on choisit la plus grande pièce
            qui ne dépasse pas le montant restant.

            Cette stratégie est efficace pour certains systèmes
            de pièces, mais elle ne garantit pas toujours une
            solution optimale pour tous les systèmes possibles.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES
        # ====================================================

        st.markdown(
            "### 💰 Données du problème"
        )

        col1, col2 = st.columns(2)

        with col1:

            coins_text = st.text_input(
                "Pièces disponibles",
                value="1, 2, 5, 10, 20, 50",
                key="greedy_coin_change_coins",
                help=(
                    "Exemple : 1, 2, 5, 10, 20, 50"
                ),
            )

        with col2:

            amount = st.number_input(
                "Montant à atteindre",
                min_value=0,
                max_value=1000000,
                value=93,
                step=1,
                key="greedy_coin_change_amount",
            )

        st.caption(
            """
            Les pièces doivent être séparées par des virgules.
            Les pièces doivent être des valeurs entières positives.
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Rendre la monnaie",
            type="primary",
            use_container_width=True,
            key="execute_greedy_coin_change",
        ):

            try:

                # ------------------------------------------------
                # Conversion des pièces
                # ------------------------------------------------

                coins = [
                    int(value.strip())
                    for value in coins_text.split(",")
                    if value.strip()
                ]

                amount_value = int(amount)

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                if not coins:

                    raise ValueError(
                        "La liste des pièces ne peut pas être vide."
                    )

                if any(
                    coin <= 0
                    for coin in coins
                ):

                    raise ValueError(
                        "Toutes les pièces doivent être "
                        "strictement positives."
                    )

                if len(set(coins)) != len(coins):

                    raise ValueError(
                        "Les pièces doivent être uniques."
                    )

                if amount_value < 0:

                    raise ValueError(
                        "Le montant doit être positif ou nul."
                    )

                # ------------------------------------------------
                # Tri décroissant
                # ------------------------------------------------

                sorted_coins = sorted(
                    coins,
                    reverse=True,
                )

                # ------------------------------------------------
                # Appel du véritable algorithme
                # ------------------------------------------------

                result = greedy_coin_change(
                    sorted_coins,
                    amount_value,
                )

                # ------------------------------------------------
                # Validation du résultat
                # ------------------------------------------------

                if result == -1:

                    st.warning(
                        """
                        Aucun rendu de monnaie n'est possible
                        avec les pièces fournies.
                        """
                    )

                else:

                    st.success(
                        "Greedy Coin Change résolu avec succès."
                    )

                st.divider()

                # ==================================================
                # RÉSULTAT
                # ==================================================

                st.markdown(
                    "### 🏆 Résultat"
                )

                if result == -1:

                    st.error(
                        "Le montant demandé ne peut pas être représenté "
                        "avec les pièces disponibles."
                    )

                else:

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Montant demandé",
                            str(amount_value),
                        )

                    with col2:

                        st.metric(
                            "Nombre de pièces",
                            str(result),
                        )

                    with col3:

                        st.metric(
                            "Pièces disponibles",
                            str(len(coins)),
                        )

                    st.markdown(
                        f"""
                        Pour obtenir un montant de
                        **{amount_value}**,

                        la stratégie gloutonne utilise :

                        \[
                        \boxed{{{result}}}
                        \]

                        pièce(s).
                        """
                    )

                st.divider()

                # ==================================================
                # PIÈCES DISPONIBLES
                # ==================================================

                st.markdown(
                    "### 🪙 Pièces disponibles"
                )

                coin_data = {
                    "Ordre": list(
                        range(
                            1,
                            len(sorted_coins) + 1,
                        )
                    ),
                    "Pièce": sorted_coins,
                }

                st.dataframe(
                    coin_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # STRATÉGIE GLOUTONNE
                # ==================================================

                st.markdown(
                    "### 🧠 Étape 1 — Trier les pièces"
                )

                st.latex(
                    r"""
                    c_1 > c_2 > \cdots > c_n
                    """
                )

                st.markdown(
                    """
                    Les pièces sont classées par ordre décroissant.

                    L'algorithme commence donc par la pièce de plus
                    grande valeur et utilise autant de pièces que
                    possible avant de passer à la suivante.
                    """
                )

                st.dataframe(
                    {
                        "Position": list(
                            range(
                                1,
                                len(sorted_coins) + 1,
                            )
                        ),
                        "Pièce": sorted_coins,
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # ==================================================
                # CALCUL PÉDAGOGIQUE
                # ==================================================

                st.markdown(
                    "### 🔎 Étape 2 — Calcul étape par étape"
                )

                remaining_amount = amount_value
                pedagogical_coins = []
                selection_steps = []

                for position, coin in enumerate(
                    sorted_coins,
                    start=1,
                ):

                    if remaining_amount <= 0:
                        break

                    count = (
                        remaining_amount // coin
                    )

                    if count > 0:

                        amount_before = remaining_amount

                        amount_taken = (
                            count * coin
                        )

                        remaining_amount -= (
                            amount_taken
                        )

                        pedagogical_coins.extend(
                            [coin] * count
                        )

                        selection_steps.append(
                            {
                                "Étape": position,
                                "Pièce": coin,
                                "Montant avant": amount_before,
                                "Nombre utilisé": count,
                                "Montant obtenu": amount_taken,
                                "Montant restant": remaining_amount,
                            }
                        )

                # ==================================================
                # AFFICHAGE DES ÉTAPES
                # ==================================================

                if selection_steps:

                    for step in selection_steps:

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"""
                                ### Étape {step['Étape']} — Pièce {step['Pièce']}

                                **Montant avant :**
                                `{step['Montant avant']}`

                                **Pièce choisie :**
                                `{step['Pièce']}`

                                **Nombre de pièces utilisées :**
                                `{step['Nombre utilisé']}`

                                **Montant obtenu :**
                                `{step['Montant obtenu']}`

                                **Montant restant :**
                                `{step['Montant restant']}`
                                """
                            )

                            st.latex(
                                rf"""
                                {step['Montant avant']}
                                -
                                ({step['Nombre utilisé']}
                                \times
                                {step['Pièce']})
                                =
                                {step['Montant restant']}
                                """
                            )

                else:

                    st.info(
                        "Aucune pièce n'est nécessaire pour un montant nul."
                    )

                st.divider()

                # ==================================================
                # TABLEAU DES DÉCISIONS
                # ==================================================

                st.markdown(
                    "### 📊 Tableau des décisions"
                )

                if selection_steps:

                    st.dataframe(
                        selection_steps,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "Aucune étape de sélection."
                    )

                st.divider()

                # ==================================================
                # SOLUTION CONSTRUITE
                # ==================================================

                st.markdown(
                    "### ✅ Solution construite"
                )

                if pedagogical_coins:

                    solution_text = " + ".join(
                        str(coin)
                        for coin in pedagogical_coins
                    )

                    st.markdown(
                        f"""
                        Le rendu de monnaie obtenu est :

                        **{solution_text}**

                        Vérification :

                        \[
                        {solution_text}
                        =
                        {sum(pedagogical_coins)}
                        \]
                        """
                    )

                    st.metric(
                        "Nombre total de pièces",
                        str(len(pedagogical_coins)),
                    )

                    st.metric(
                        "Montant restant",
                        str(remaining_amount),
                    )

                else:

                    st.info(
                        "Aucune pièce n'a été utilisée."
                    )

                st.divider()

                # ==================================================
                # FORMULE
                # ==================================================

                st.markdown(
                    "### 🧮 Formulation mathématique"
                )

                st.markdown(
                    """
                    Si `xᵢ` représente le nombre de pièces de valeur
                    `cᵢ` utilisées, le montant obtenu est :

                    \[
                    \sum_i x_i c_i
                    \]

                    avec :

                    \[
                    x_i \geq 0
                    \]

                    et :

                    \[
                    \sum_i x_i c_i = A
                    \]

                    où `A` représente le montant demandé.
                    """
                )

                st.divider()

                # ==================================================
                # POURQUOI LE GLOUTON
                # ==================================================

                st.markdown(
                    "### 💡 Pourquoi utiliser une stratégie gloutonne ?"
                )

                st.markdown(
                    """
                    La stratégie gloutonne fait toujours le choix
                    de la plus grande pièce possible.

                    Elle est simple et rapide.

                    Cependant, contrairement au **Fractional Knapsack**,
                    le choix glouton ne garantit pas l'optimalité pour
                    tous les systèmes de pièces.

                    Par exemple, avec les pièces :

                    `1, 3, 4`

                    et le montant :

                    `6`

                    la stratégie gloutonne choisit :

                    `4 + 1 + 1`

                    soit **3 pièces**.

                    Pourtant, une autre solution existe :

                    `3 + 3`

                    soit **2 pièces**.

                    Cet exemple montre qu'il faut distinguer :

                    - une solution construite par une stratégie gloutonne ;
                    - une solution réellement optimale.
                    """
                )

                st.divider()

                # ==================================================
                # COMPLEXITÉ
                # ==================================================

                st.markdown(
                    "### ⚙️ Complexité"
                )

                st.markdown(
                    """
                    Pour `n` types de pièces :

                    **Tri des pièces :**

                    \[
                    O(n\log n)
                    \]

                    **Parcours des pièces :**

                    \[
                    O(n)
                    \]

                    **Complexité totale :**

                    \[
                    O(n\log n)
                    \]

                    Si les pièces sont déjà triées dans l'ordre
                    décroissant, le parcours est en :

                    \[
                    O(n)
                    \]
                    """
                )

                st.divider()

                # ==================================================
                # VÉRIFICATION
                # ==================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                if result == -1:

                    if remaining_amount != 0:

                        st.info(
                            f"""
                            Le calcul pédagogique confirme que le
                            montant restant est **{remaining_amount}**.

                            Aucun rendu complet n'est donc possible
                            avec les pièces disponibles.
                            """
                        )

                else:

                    pedagogical_count = len(
                        pedagogical_coins
                    )

                    if (
                        pedagogical_count == result
                        and remaining_amount == 0
                    ):

                        st.success(
                            """
                            Le résultat du module
                            `greedy_coin_change` correspond
                            exactement au calcul pédagogique
                            effectué par l'interface.
                            """
                        )

                    else:

                        st.error(
                            f"""
                            Une différence a été détectée.

                            Résultat du module :

                            `{result}`

                            Résultat pédagogique :

                            `{pedagogical_count}`

                            Montant restant :

                            `{remaining_amount}`
                            """
                        )

            # ====================================================
            # GESTION DES ERREURS
            # ====================================================

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Greedy Coin Change : {error}"
                )

            except Exception as error:

                st.error(
                    """
                    Une erreur est survenue pendant
                    la résolution du Greedy Coin Change.
                    """
                )

                st.exception(error)

    
    # ========================================================
    # HUFFMAN CODING
    # ========================================================

    elif selected_algorithm == "huffman_coding":

        st.markdown(
        "### 🌳 Huffman Coding — Codage de Huffman"
    )

    st.markdown(
        """
        Le **codage de Huffman** est un algorithme glouton
        utilisé pour construire un **code binaire préfixe**
        permettant de représenter des symboles avec un nombre
        réduit de bits.

        L'idée principale est de donner :

        - des codes courts aux symboles fréquents ;
        - des codes plus longs aux symboles rares.

        La construction de l'arbre suit une stratégie gloutonne :
        on fusionne toujours les **deux fréquences les plus faibles**.
        """
    )

    st.info(
        """
        **Principe glouton**

        À chaque étape :

        1. sélectionner les deux nœuds de plus faible fréquence ;
        2. les fusionner ;
        3. créer un nouveau nœud dont la fréquence est leur somme ;
        4. remettre ce nouveau nœud dans l'ensemble ;
        5. recommencer jusqu'à obtenir un seul arbre.
        """
    )

    st.divider()

    # ====================================================
    # DONNÉES
    # ====================================================

    st.markdown(
        "### 📊 Fréquences des symboles"
    )

    frequencies_text = st.text_area(
        "Fréquences",
        value=(
            "A:45\n"
            "B:13\n"
            "C:12\n"
            "D:16\n"
            "E:9\n"
            "F:5"
        ),
        height=180,
        key="greedy_huffman_frequencies",
        help=(
            "Un symbole par ligne sous la forme : A:45"
        ),
    )

    st.caption(
        """
        Format attendu :

        `symbole:fréquence`

        Exemple :

        `A:45`

        `B:13`

        `C:12`
        """
    )

    st.divider()

    # ====================================================
    # CALCUL
    # ====================================================

    if st.button(
        "Construire le code de Huffman",
        type="primary",
        use_container_width=True,
        key="execute_huffman_coding",
    ):

        try:

            # ------------------------------------------------
            # PARSING
            # ------------------------------------------------

            frequencies = {}

            for line_number, line in enumerate(
                frequencies_text.splitlines(),
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                if ":" not in line:

                    raise ValueError(
                        f"Ligne {line_number} invalide : "
                        "utilisez le format symbole:fréquence."
                    )

                symbol_text, frequency_text = (
                    line.split(":", 1)
                )

                symbol = symbol_text.strip()

                if not symbol:

                    raise ValueError(
                        f"Ligne {line_number} : "
                        "le symbole ne peut pas être vide."
                    )

                try:

                    frequency = int(
                        frequency_text.strip()
                    )

                except ValueError as error:

                    raise ValueError(
                        f"Ligne {line_number} : "
                        "la fréquence doit être un entier."
                    ) from error

                if frequency <= 0:

                    raise ValueError(
                        f"Ligne {line_number} : "
                        "la fréquence doit être strictement positive."
                    )

                if symbol in frequencies:

                    raise ValueError(
                        f"Le symbole « {symbol} » "
                        "est présent plusieurs fois."
                    )

                frequencies[symbol] = frequency

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if not frequencies:

                raise ValueError(
                    "Aucune fréquence valide n'a été saisie."
                )

            if len(frequencies) < 2:

                raise ValueError(
                    "Le codage de Huffman nécessite "
                    "au moins deux symboles."
                )

            # ------------------------------------------------
            # AFFICHAGE DES DONNÉES
            # ------------------------------------------------

            st.success(
                "Fréquences validées avec succès."
            )

            st.markdown(
                "### 📥 Fréquences initiales"
            )

            frequency_rows = [
                {
                    "Symbole": symbol,
                    "Fréquence": frequency,
                }
                for symbol, frequency in frequencies.items()
            ]

            st.dataframe(
                frequency_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # ------------------------------------------------
            # TRI DES FRÉQUENCES
            # ------------------------------------------------

            sorted_frequencies = sorted(
                frequencies.items(),
                key=lambda item: item[1],
            )

            st.markdown(
                "### 🔢 Fréquences triées"
            )

            sorted_rows = [
                {
                    "Ordre": index + 1,
                    "Symbole": symbol,
                    "Fréquence": frequency,
                }
                for index, (symbol, frequency)
                in enumerate(sorted_frequencies)
            ]

            st.dataframe(
                sorted_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # =================================================
            # CONVERSION DES FRÉQUENCES EN SÉQUENCE
            # =================================================
            #
            # IMPORTANT :
            # huffman_coding() attend une séquence de symboles
            # et non le dictionnaire frequencies.
            #
            # Exemple :
            #
            # A:3
            # B:2
            #
            # devient :
            #
            # ["A", "A", "A", "B", "B"]
            #
            # =================================================

            data = []

            for symbol, frequency in frequencies.items():

                data.extend(
                    [symbol] * frequency
                )

            # ------------------------------------------------
            # APPEL DU VÉRITABLE ALGORITHME
            # ------------------------------------------------

            result = huffman_coding(
                data
            )

            # ------------------------------------------------
            # INTERPRÉTATION DU RÉSULTAT
            # ------------------------------------------------

            if not isinstance(result, dict):

                raise TypeError(
                    "Le résultat de huffman_coding "
                    "doit être un dictionnaire de codes."
                )

            codes = result

            # ------------------------------------------------
            # RÉSULTAT
            # ------------------------------------------------

            st.markdown(
                "### 🏆 Codes de Huffman"
            )

            code_rows = []

            for symbol, frequency in sorted(
                frequencies.items(),
                key=lambda item: item[1],
                reverse=True,
            ):

                code = codes.get(
                    symbol,
                    "",
                )

                code_rows.append(
                    {
                        "Symbole": symbol,
                        "Fréquence": frequency,
                        "Code": code,
                        "Longueur": len(code),
                    }
                )

            st.dataframe(
                code_rows,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # =================================================
            # EXPLICATION ÉTAPE PAR ÉTAPE
            # =================================================

            st.markdown(
                "### 🔎 Construction étape par étape"
            )

            st.markdown(
                """
                Huffman sélectionne toujours les deux
                fréquences les plus faibles.

                Avec les données fournies, les deux nœuds
                ayant les plus petites fréquences sont fusionnés,
                puis leur somme est réintroduite dans l'ensemble
                des nœuds.

                Le processus continue jusqu'à obtenir un seul arbre.
                """
            )

            # ------------------------------------------------
            # Simulation pédagogique de la fusion
            # ------------------------------------------------

            nodes = [
                {
                    "symbols": [symbol],
                    "frequency": frequency,
                }
                for symbol, frequency in frequencies.items()
            ]

            merge_steps = []

            step_number = 1

            while len(nodes) > 1:

                nodes.sort(
                    key=lambda node: node["frequency"]
                )

                first = nodes.pop(0)

                second = nodes.pop(0)

                merged_frequency = (
                    first["frequency"]
                    + second["frequency"]
                )

                merged_symbols = (
                    first["symbols"]
                    + second["symbols"]
                )

                merge_steps.append(
                    {
                        "Étape": step_number,
                        "Premier nœud": (
                            ", ".join(
                                first["symbols"]
                            )
                        ),
                        "Fréquence 1": (
                            first["frequency"]
                        ),
                        "Deuxième nœud": (
                            ", ".join(
                                second["symbols"]
                            )
                        ),
                        "Fréquence 2": (
                            second["frequency"]
                        ),
                        "Nouvelle fréquence": (
                            merged_frequency
                        ),
                    }
                )

                nodes.append(
                    {
                        "symbols": merged_symbols,
                        "frequency": merged_frequency,
                    }
                )

                step_number += 1

            st.dataframe(
                merge_steps,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # FORMULE
            # ------------------------------------------------

            st.markdown(
                "### 🧮 Règle de fusion"
            )

            st.latex(
                r"""
                f_{\mathrm{nouveau}}
                =
                f_{\mathrm{gauche}}
                +
                f_{\mathrm{droite}}
                """
            )

            st.markdown(
                """
                À chaque fusion, les deux plus petites
                fréquences sont remplacées par leur somme.
                """
            )

            st.divider()

            # =================================================
            # ARBRE CONCEPTUEL
            # =================================================

            st.markdown(
                "### 🌳 Structure conceptuelle"
            )

            st.markdown(
                """
                Le processus construit progressivement
                un arbre binaire :

                ```text
                              Racine
                             /     \\
                           ...     ...
                          /  \\     /  \\
                         ... ...   ... ...

                ```

                Chaque branche peut recevoir un bit :

                - branche gauche → `0`
                - branche droite → `1`

                Le code d'un symbole correspond alors au chemin
                parcouru depuis la racine jusqu'à ce symbole.
                """
            )

            st.divider()

            # =================================================
            # PROPRIÉTÉ DE PRÉFIXE
            # =================================================

            st.markdown(
                "### 🔐 Propriété de code préfixe"
            )

            st.info(
                """
                Aucun code de symbole ne doit être le préfixe
                d'un autre code.

                Cette propriété permet de décoder les données
                sans ambiguïté.
                """
            )

            st.divider()

            # =================================================
            # LONGUEUR MOYENNE
            # =================================================

            st.markdown(
                "### 📏 Longueur moyenne du code"
            )

            total_frequency = sum(
                frequencies.values()
            )

            weighted_length = sum(
                frequencies[symbol]
                * len(codes[symbol])
                for symbol in frequencies
                if symbol in codes
            )

            average_length = (
                weighted_length
                / total_frequency
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Fréquence totale",
                    f"{total_frequency:g}",
                )

            with col2:

                st.metric(
                    "Bits pondérés",
                    f"{weighted_length:g}",
                )

            with col3:

                st.metric(
                    "Longueur moyenne",
                    f"{average_length:.3f}",
                )

            st.latex(
                r"""
                L_{\mathrm{moy}}
                =
                \frac{
                    \sum_i f_i\,l_i
                }{
                    \sum_i f_i
                }
                """
            )

            st.divider()

            # =================================================
            # POURQUOI HUFFMAN EST GLOUTON
            # =================================================

            st.markdown(
                "### 💡 Pourquoi Huffman est un algorithme glouton ?"
            )

            st.write(
                """
                Huffman prend localement les deux symboles
                ou sous-arbres ayant les plus faibles fréquences,
                puis les fusionne.

                Cette décision locale permet de construire
                progressivement un arbre de codage optimal
                pour le problème classique du codage préfixe
                pondéré.
                """
            )

            st.divider()

            # =================================================
            # COMPLEXITÉ
            # =================================================

            st.markdown(
                "### ⚙️ Complexité"
            )

            st.markdown(
                r"""
                Pour `n` symboles, l'utilisation d'une
                file de priorité permet de construire
                l'arbre de Huffman en :

                \[
                O(n\log n)
                \]

                L'espace nécessaire pour stocker l'arbre
                et les codes est de l'ordre de :

                \[
                O(n)
                \]
                """
            )

            st.divider()

            # =================================================
            # VÉRIFICATION
            # =================================================

            st.markdown(
                "### ✅ Vérification"
            )

            valid_codes = all(
                isinstance(code, str)
                and len(code) > 0
                and set(code).issubset({"0", "1"})
                for code in codes.values()
            )

            symbols_match = (
                set(codes.keys())
                == set(frequencies.keys())
            )

            if valid_codes and symbols_match:

                st.success(
                    """
                    Le résultat du module
                    `huffman_coding` contient un code binaire
                    valide pour chaque symbole fourni.
                    """
                )

            else:

                st.error(
                    """
                    Une incohérence a été détectée entre
                    les fréquences fournies et les codes
                    retournés par le module.
                    """
                )

        # ====================================================
        # GESTION DES ERREURS
        # ====================================================

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur Huffman Coding : {error}"
            )

        except Exception as error:

            st.error(
                """
                Une erreur est survenue pendant
                la construction du code de Huffman.
                """
            )

            st.exception(error)

    
    # ========================================================
    # JOB SEQUENCING WITH DEADLINES
    # ========================================================

    elif selected_algorithm == "job_sequencing":

        st.markdown(
            "### ⏱️ Job Sequencing — Ordonnancement des tâches"
        )

        st.markdown(
            """
            Le **Job Sequencing with Deadlines** est un algorithme
            glouton permettant de sélectionner un ensemble de tâches
            afin de **maximiser le profit total**, sous contrainte
            que chaque tâche soit exécutée avant ou à sa date limite.

            Chaque tâche possède :

            - un identifiant ;
            - une **deadline** ;
            - un **profit**.

            L'idée gloutonne consiste à :

            1. trier les tâches par profit décroissant ;
            2. prendre la tâche la plus profitable disponible ;
            3. la placer dans le dernier créneau libre avant sa deadline ;
            4. continuer jusqu'à traiter toutes les tâches.
            """
        )

        st.info(
            """
            **Principe glouton**

            Pour chaque tâche, on cherche à préserver autant que possible
            les créneaux précédents.

            Une tâche de profit élevé est donc placée dans le **dernier
            créneau disponible avant sa deadline**, ce qui laisse les
            créneaux antérieurs disponibles pour d'autres tâches.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES
        # ====================================================

        st.markdown(
            "### 📋 Tâches"
        )

        jobs_text = st.text_area(
            "Tâches",
            value=(
                "A:2:100\n"
                "B:1:19\n"
                "C:2:27\n"
                "D:1:25\n"
                "E:3:15"
            ),
            height=180,
            key="greedy_job_sequencing_jobs",
            help=(
                "Un job par ligne sous la forme : "
                "job:deadline:profit"
            ),
        )

        st.caption(
            """
            **Format attendu :**

            `job:deadline:profit`

            Exemple :

            `A:2:100`

            `B:1:19`

            `C:2:27`

            La deadline doit être un entier positif et le profit
            un nombre positif ou nul.
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Exécuter Job Sequencing",
            type="primary",
            use_container_width=True,
            key="execute_job_sequencing",
        ):

            try:

                # =================================================
                # PARSING
                # =================================================

                jobs = []

                for line_number, line in enumerate(
                    jobs_text.splitlines(),
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split(":")

                    if len(parts) != 3:
                        raise ValueError(
                            f"Ligne {line_number} invalide : "
                            "utilisez le format job:deadline:profit."
                        )

                    job_text = parts[0].strip()
                    deadline_text = parts[1].strip()
                    profit_text = parts[2].strip()

                    if not job_text:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le nom du job ne peut pas être vide."
                        )

                    # -------------------------------------------------
                    # DEADLINE
                    # -------------------------------------------------

                    try:
                        deadline = int(deadline_text)

                    except ValueError as error:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "la deadline doit être un entier."
                        ) from error

                    if deadline <= 0:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "la deadline doit être strictement positive."
                        )

                    # -------------------------------------------------
                    # PROFIT
                    # -------------------------------------------------

                    try:
                        profit = float(profit_text)

                    except ValueError as error:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le profit doit être un nombre."
                        ) from error

                    if profit < 0:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le profit ne peut pas être négatif."
                        )

                    # -------------------------------------------------
                    # AJOUT
                    # -------------------------------------------------

                    jobs.append(
                        {
                            "job": job_text,
                            "deadline": deadline,
                            "profit": profit,
                        }
                    )

                # =================================================
                # VALIDATION
                # =================================================

                if not jobs:
                    raise ValueError(
                        "Aucune tâche valide n'a été saisie."
                    )

                job_names = [
                    job["job"]
                    for job in jobs
                ]

                if len(job_names) != len(set(job_names)):
                    raise ValueError(
                        "Deux tâches possèdent le même identifiant."
                    )

                st.success(
                    "Les tâches ont été validées avec succès."
                )

                # =================================================
                # DONNÉES INITIALES
                # =================================================

                st.markdown(
                    "### 📥 Tâches initiales"
                )

                initial_rows = [
                    {
                        "Job": job["job"],
                        "Deadline": job["deadline"],
                        "Profit": job["profit"],
                    }
                    for job in jobs
                ]

                st.dataframe(
                    initial_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # TRI PAR PROFIT DÉCROISSANT
                # =================================================

                sorted_jobs = sorted(
                    jobs,
                    key=lambda job: job["profit"],
                    reverse=True,
                )

                st.markdown(
                    "### 💰 Tâches triées par profit décroissant"
                )

                sorted_rows = [
                    {
                        "Ordre": index + 1,
                        "Job": job["job"],
                        "Deadline": job["deadline"],
                        "Profit": job["profit"],
                    }
                    for index, job in enumerate(sorted_jobs)
                ]

                st.dataframe(
                    sorted_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # PRÉPARATION DES DONNÉES POUR L'ALGORITHME
                # =================================================

                data = [
                    (
                        job["job"],
                        job["deadline"],
                        job["profit"],
                    )
                    for job in jobs
                ]

                # =================================================
                # APPEL DU VÉRITABLE ALGORITHME
                # =================================================

                result = job_sequencing(data)

                # =================================================
                # INTERPRÉTATION DU RÉSULTAT
                # =================================================

                if isinstance(result, dict):

                    selected_jobs = result.get(
                        "selected_jobs",
                        result.get("jobs", [])
                    )

                    total_profit = result.get(
                        "total_profit",
                        result.get("profit", 0)
                    )

                    schedule = result.get(
                        "schedule",
                        []
                    )

                elif isinstance(result, tuple):

                    if len(result) == 2:

                        selected_jobs = result[0]
                        total_profit = result[1]
                        schedule = selected_jobs

                    else:

                        raise TypeError(
                            "Le résultat de job_sequencing() "
                            "possède un format inattendu."
                        )

                elif isinstance(result, list):

                    selected_jobs = result
                    schedule = result

                    # Calcul local du profit
                    selected_names = set()

                    for item in result:

                        if isinstance(item, dict):

                            name = item.get(
                                "job",
                                item.get("id")
                            )

                            if name is not None:
                                selected_names.add(name)

                        elif isinstance(item, str):

                            selected_names.add(item)

                    total_profit = sum(
                        job["profit"]
                        for job in jobs
                        if job["job"] in selected_names
                    )

                else:

                    raise TypeError(
                        "Le résultat de job_sequencing() "
                        "doit être une liste, un tuple ou un dictionnaire."
                    )

                # =================================================
                # RÉSULTAT
                # =================================================

                st.markdown(
                    "### 🏆 Ordonnancement optimal"
                )

                if schedule:

                    schedule_rows = []

                    for position, item in enumerate(
                        schedule,
                        start=1,
                    ):

                        if isinstance(item, dict):

                            job_name = item.get(
                                "job",
                                item.get("id", "")
                            )

                            deadline = item.get(
                                "deadline",
                                ""
                            )

                            profit = item.get(
                                "profit",
                                ""
                            )

                        elif isinstance(item, (tuple, list)):

                            job_name = (
                                item[0]
                                if len(item) > 0
                                else ""
                            )

                            deadline = (
                                item[1]
                                if len(item) > 1
                                else ""
                            )

                            profit = (
                                item[2]
                                if len(item) > 2
                                else ""
                            )

                        else:

                            job_name = str(item)

                            matching_job = next(
                                (
                                    job
                                    for job in jobs
                                    if job["job"] == job_name
                                ),
                                None,
                            )

                            if matching_job:

                                deadline = matching_job["deadline"]
                                profit = matching_job["profit"]

                            else:

                                deadline = ""
                                profit = ""

                        schedule_rows.append(
                            {
                                "Créneau": position,
                                "Job": job_name,
                                "Deadline": deadline,
                                "Profit": profit,
                            }
                        )

                    st.dataframe(
                        schedule_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.warning(
                        "Aucune tâche n'a été sélectionnée."
                    )

                # =================================================
                # INDICATEURS
                # =================================================

                st.divider()

                st.markdown(
                    "### 📊 Résumé"
                )

                selected_count = len(
                    schedule
                ) if schedule else 0

                max_deadline = max(
                    job["deadline"]
                    for job in jobs
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Tâches initiales",
                        len(jobs),
                    )

                with col2:

                    st.metric(
                        "Tâches sélectionnées",
                        selected_count,
                    )

                with col3:

                    st.metric(
                        "Créneaux disponibles",
                        max_deadline,
                    )

                with col4:

                    st.metric(
                        "Profit total",
                        f"{total_profit:g}",
                    )

                st.divider()

                # =================================================
                # SIMULATION PÉDAGOGIQUE
                # =================================================

                st.markdown(
                    "### 🔎 Construction étape par étape"
                )

                st.markdown(
                    """
                    L'algorithme examine les tâches dans l'ordre
                    décroissant de profit.

                    Pour chaque tâche, il cherche le **dernier créneau
                    libre avant sa deadline**.

                    Si un créneau est disponible, la tâche est placée.
                    Sinon, elle est rejetée.
                    """
                )

                # -------------------------------------------------
                # SIMULATION
                # -------------------------------------------------

                simulation_slots = [
                    None
                    for _ in range(max_deadline)
                ]

                simulation_steps = []

                for job in sorted_jobs:

                    placed = False
                    selected_slot = None

                    latest_slot = min(
                        job["deadline"],
                        max_deadline,
                    )

                    for slot in range(
                        latest_slot - 1,
                        -1,
                        -1,
                    ):

                        if simulation_slots[slot] is None:

                            simulation_slots[slot] = job["job"]

                            placed = True
                            selected_slot = slot + 1

                            break

                    simulation_steps.append(
                        {
                            "Job": job["job"],
                            "Deadline": job["deadline"],
                            "Profit": job["profit"],
                            "Créneau choisi": (
                                selected_slot
                                if placed
                                else "—"
                            ),
                            "Décision": (
                                "Acceptée"
                                if placed
                                else "Rejetée"
                            ),
                        }
                    )

                st.dataframe(
                    simulation_steps,
                    use_container_width=True,
                    hide_index=True,
                )

                # =================================================
                # CALENDRIER FINAL
                # =================================================

                st.markdown(
                    "### 🗓️ Calendrier final"
                )

                calendar_rows = []

                for slot_number, job_name in enumerate(
                    simulation_slots,
                    start=1,
                ):

                    if job_name is None:

                        calendar_rows.append(
                            {
                                "Créneau": slot_number,
                                "Job": "Libre",
                                "Statut": "Disponible",
                            }
                        )

                    else:

                        matching_job = next(
                            (
                                job
                                for job in jobs
                                if job["job"] == job_name
                            ),
                            None,
                        )

                        calendar_rows.append(
                            {
                                "Créneau": slot_number,
                                "Job": job_name,
                                "Deadline": (
                                    matching_job["deadline"]
                                    if matching_job
                                    else ""
                                ),
                                "Profit": (
                                    matching_job["profit"]
                                    if matching_job
                                    else ""
                                ),
                                "Statut": "Occupé",
                            }
                        )

                st.dataframe(
                    calendar_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # FORMULE
                # =================================================

                st.markdown(
                    "### 🧮 Objectif"
                )

                st.latex(
                    r"""
                    \max \sum_{j \in S} p_j
                    """
                )

                st.markdown(
                    """
                    où :

                    - \(S\) représente l'ensemble des tâches sélectionnées ;
                    - \(p_j\) représente le profit de la tâche \(j\).

                    Une tâche sélectionnée doit respecter sa deadline.
                    """
                )

                st.divider()

                # =================================================
                # POURQUOI L'ALGORITHME EST GLOUTON
                # =================================================

                st.markdown(
                    "### 💡 Pourquoi Job Sequencing est-il glouton ?"
                )

                st.write(
                    """
                    L'algorithme prend d'abord la décision qui semble
                    la plus avantageuse localement : considérer en priorité
                    la tâche ayant le profit le plus élevé.

                    Il tente ensuite de placer cette tâche le plus tard
                    possible avant sa deadline.

                    Cette stratégie permet de conserver les créneaux
                    précédents pour les autres tâches.
                    """
                )

                st.divider()

                # =================================================
                # COMPLEXITÉ
                # =================================================

                st.markdown(
                    "### ⚙️ Complexité"
                )

                st.markdown(
                    r"""
                    Avec un tri des tâches par profit :

                    **\[ O(n \log n) \]**

                    Puis, dans l'implémentation classique où l'on recherche
                    linéairement un créneau disponible :

                    **\[ O(n \cdot d) \]**

                    où \(d\) représente le nombre maximal de créneaux.

                    L'espace utilisé pour le calendrier est de l'ordre de :

                    **\[ O(d) \]**
                    """
                )

                st.divider()

                # =================================================
                # VÉRIFICATION
                # =================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                valid_schedule = True
                used_slots = set()

                for slot_number, job_name in enumerate(
                    simulation_slots,
                    start=1,
                ):

                    if job_name is None:
                        continue

                    if slot_number in used_slots:

                        valid_schedule = False
                        break

                    used_slots.add(slot_number)

                    matching_job = next(
                        (
                            job
                            for job in jobs
                            if job["job"] == job_name
                        ),
                        None,
                    )

                    if matching_job is None:

                        valid_schedule = False
                        break

                    if slot_number > matching_job["deadline"]:

                        valid_schedule = False
                        break

                if valid_schedule:

                    st.success(
                        """
                        L'ordonnancement obtenu respecte les deadlines :
                        chaque tâche sélectionnée est exécutée dans un
                        créneau valide et chaque créneau est utilisé au
                        maximum une fois.
                        """
                    )

                else:

                    st.error(
                        """
                        Une incohérence a été détectée dans
                        l'ordonnancement retourné.
                        """
                    )

            # ====================================================
            # GESTION DES ERREURS
            # ====================================================

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Job Sequencing : {error}"
                )

            except Exception as error:

                st.error(
                    """
                    Une erreur est survenue pendant
                    l'exécution de Job Sequencing.
                    """
                )

                st.exception(error)

    # ========================================================
    # INTERVAL SCHEDULING
    # ========================================================

    elif selected_algorithm == "interval_scheduling":

        st.markdown(
            "### 📅 Interval Scheduling — Ordonnancement d'intervalles"
        )

        st.markdown(
            """
            L'**Interval Scheduling** est un algorithme glouton qui
            cherche à sélectionner le **plus grand nombre d'intervalles
            compatibles**, c'est-à-dire des activités qui ne se
            chevauchent pas.

            Chaque intervalle possède :

            - un identifiant ;
            - une heure de début ;
            - une heure de fin.

            L'idée gloutonne consiste à :

            1. trier les intervalles par heure de fin croissante ;
            2. sélectionner le premier intervalle compatible ;
            3. ignorer les intervalles qui se chevauchent ;
            4. continuer jusqu'à la fin de la liste.
            """
        )

        st.info(
            """
            **Principe glouton**

            On choisit toujours l'intervalle qui **se termine le plus tôt**
            parmi les intervalles encore compatibles.

            Pourquoi ?

            Un intervalle qui se termine tôt laisse davantage de temps
            disponible pour sélectionner d'autres intervalles.
            """
        )

        st.divider()

        # ====================================================
        # DONNÉES
        # ====================================================

        st.markdown(
            "### 📋 Intervalles"
        )

        intervals_text = st.text_area(
            "Intervalles",
            value=(
                "A:1:3\n"
                "B:2:5\n"
                "C:4:7\n"
                "D:6:9\n"
                "E:8:10\n"
                "F:9:11\n"
                "G:10:12"
            ),
            height=200,
            key="greedy_interval_scheduling_intervals",
            help=(
                "Un intervalle par ligne sous la forme : "
                "nom:début:fin"
            ),
        )

        st.caption(
            """
            **Format attendu :**

            `nom:début:fin`

            Exemple :

            `A:1:3`

            `B:2:5`

            `C:4:7`

            Un intervalle doit respecter :

            **début < fin**
            """
        )

        st.divider()

        # ====================================================
        # CALCUL
        # ====================================================

        if st.button(
            "Exécuter Interval Scheduling",
            type="primary",
            use_container_width=True,
            key="execute_interval_scheduling",
        ):

            try:

                # =================================================
                # PARSING
                # =================================================

                intervals = []

                for line_number, line in enumerate(
                    intervals_text.splitlines(),
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split(":")

                    if len(parts) != 3:
                        raise ValueError(
                            f"Ligne {line_number} invalide : "
                            "utilisez le format nom:début:fin."
                        )

                    name_text = parts[0].strip()
                    start_text = parts[1].strip()
                    end_text = parts[2].strip()

                    if not name_text:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le nom de l'intervalle ne peut pas "
                            "être vide."
                        )

                    # -------------------------------------------------
                    # DÉBUT
                    # -------------------------------------------------

                    try:
                        start = float(start_text)

                    except ValueError as error:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "l'heure de début doit être un nombre."
                        ) from error

                    # -------------------------------------------------
                    # FIN
                    # -------------------------------------------------

                    try:
                        end = float(end_text)

                    except ValueError as error:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "l'heure de fin doit être un nombre."
                        ) from error

                    # -------------------------------------------------
                    # VALIDATION
                    # -------------------------------------------------

                    if start >= end:
                        raise ValueError(
                            f"Ligne {line_number} : "
                            "le début doit être strictement inférieur "
                            "à la fin."
                        )

                    intervals.append(
                        {
                            "name": name_text,
                            "start": start,
                            "end": end,
                        }
                    )

                # =================================================
                # VALIDATION GLOBALE
                # =================================================

                if not intervals:
                    raise ValueError(
                        "Aucun intervalle valide n'a été saisi."
                    )

                interval_names = [
                    interval["name"]
                    for interval in intervals
                ]

                if len(interval_names) != len(
                    set(interval_names)
                ):
                    raise ValueError(
                        "Deux intervalles possèdent le même identifiant."
                    )

                if len(intervals) < 2:

                    st.warning(
                        "Un seul intervalle a été fourni. "
                        "Il sera sélectionné automatiquement."
                    )

                st.success(
                    "Les intervalles ont été validés avec succès."
                )

                # =================================================
                # INTERVALLES INITIAUX
                # =================================================

                st.markdown(
                    "### 📥 Intervalles initiaux"
                )

                initial_rows = [
                    {
                        "Intervalle": interval["name"],
                        "Début": interval["start"],
                        "Fin": interval["end"],
                        "Durée": (
                            interval["end"]
                            - interval["start"]
                        ),
                    }
                    for interval in intervals
                ]

                st.dataframe(
                    initial_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # TRI PAR FIN CROISSANTE
                # =================================================

                sorted_intervals = sorted(
                    intervals,
                    key=lambda interval: (
                        interval["end"],
                        interval["start"],
                    ),
                )

                st.markdown(
                    "### 🔢 Intervalles triés par heure de fin"
                )

                sorted_rows = [
                    {
                        "Ordre": index + 1,
                        "Intervalle": interval["name"],
                        "Début": interval["start"],
                        "Fin": interval["end"],
                    }
                    for index, interval in enumerate(
                        sorted_intervals
                    )
                ]

                st.dataframe(
                    sorted_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                
                # =================================================
                # PRÉPARATION DES DONNÉES POUR L'ALGORITHME
                # =================================================
                #
                # IMPORTANT :
                # interval_scheduling() attend des intervalles
                # contenant exactement DEUX valeurs :
                #
                # (début, fin)
                #
                # Il ne faut donc PAS envoyer :
                #
                # (nom, début, fin)
                #
                # =================================================

                data = [
                    (
                        interval["start"],
                        interval["end"],
                    )
                    for interval in intervals
                ]

                # -------------------------------------------------
                # APPEL DU VÉRITABLE ALGORITHME
                # -------------------------------------------------

                result = interval_scheduling(data)

                # =================================================
                # INTERPRÉTATION DU RÉSULTAT
                # =================================================

                if isinstance(result, dict):

                    selected_intervals = result.get(
                        "selected_intervals",
                        result.get(
                            "intervals",
                            result.get(
                                "schedule",
                                [],
                            ),
                        ),
                    )

                    count = result.get(
                        "count",
                        len(selected_intervals),
                    )

                elif isinstance(result, tuple):

                    if len(result) == 2:

                        selected_intervals = result[0]
                        count = result[1]

                    else:

                        raise TypeError(
                            "Le résultat de interval_scheduling() "
                            "possède un format inattendu."
                        )

                elif isinstance(result, list):

                    selected_intervals = result
                    count = len(result)

                else:

                    raise TypeError(
                        "Le résultat de interval_scheduling() "
                        "doit être une liste, un tuple ou "
                        "un dictionnaire."
                    )

                # =================================================
                # NORMALISATION DU RÉSULTAT
                # =================================================

                selected_names = []

                for item in selected_intervals:

                    if isinstance(item, dict):

                        name = item.get(
                            "name",
                            item.get(
                                "interval",
                                item.get(
                                    "job",
                                    item.get(
                                        "id",
                                        "",
                                    ),
                                ),
                            ),
                        )

                    elif isinstance(item, (tuple, list)):

                        name = (
                            item[0]
                            if len(item) > 0
                            else ""
                        )

                    else:

                        name = str(item)

                    if name:
                        selected_names.append(str(name))

                # =================================================
                # RÉSULTAT
                # =================================================

                st.markdown(
                    "### 🏆 Intervalles sélectionnés"
                )

                selected_rows = []

                for position, name in enumerate(
                    selected_names,
                    start=1,
                ):

                    matching_interval = next(
                        (
                            interval
                            for interval in intervals
                            if interval["name"] == name
                        ),
                        None,
                    )

                    if matching_interval:

                        selected_rows.append(
                            {
                                "Ordre": position,
                                "Intervalle": name,
                                "Début": matching_interval[
                                    "start"
                                ],
                                "Fin": matching_interval[
                                    "end"
                                ],
                                "Durée": (
                                    matching_interval["end"]
                                    - matching_interval["start"]
                                ),
                            }
                        )

                if selected_rows:

                    st.dataframe(
                        selected_rows,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.warning(
                        "Aucun intervalle n'a été sélectionné."
                    )

                # =================================================
                # INDICATEURS
                # =================================================

                st.divider()

                st.markdown(
                    "### 📊 Résumé"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Intervalles initiaux",
                        len(intervals),
                    )

                with col2:

                    st.metric(
                        "Intervalles sélectionnés",
                        len(selected_names),
                    )

                with col3:

                    st.metric(
                        "Taux de sélection",
                        f"{(
                            len(selected_names)
                            / len(intervals)
                            * 100
                        ):.1f}%",
                    )

                st.divider()

                # =================================================
                # SIMULATION PÉDAGOGIQUE
                # =================================================

                st.markdown(
                    "### 🔎 Construction étape par étape"
                )

                st.markdown(
                    """
                    L'algorithme examine les intervalles dans l'ordre
                    croissant de leur heure de fin.

                    Pour chaque intervalle :

                    - s'il commence après ou exactement au moment où
                      le dernier intervalle sélectionné se termine,
                      il est accepté ;
                    - sinon, il est rejeté car il chevauche un intervalle
                      déjà sélectionné.
                    """
                )

                # -------------------------------------------------
                # SIMULATION
                # -------------------------------------------------

                simulation_steps = []

                selected_simulation = []

                last_end = None

                for interval in sorted_intervals:

                    if last_end is None:

                        accepted = True

                    else:

                        accepted = (
                            interval["start"]
                            >= last_end
                        )

                    if accepted:

                        selected_simulation.append(
                            interval["name"]
                        )

                        last_end = interval["end"]

                        decision = "Accepté"

                    else:

                        decision = "Rejeté"

                    simulation_steps.append(
                        {
                            "Intervalle": interval["name"],
                            "Début": interval["start"],
                            "Fin": interval["end"],
                            "Dernière fin": (
                                "—"
                                if last_end is None
                                else last_end
                            ),
                            "Décision": decision,
                        }
                    )

                st.dataframe(
                    simulation_steps,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # CALENDRIER FINAL
                # =================================================

                st.markdown(
                    "### 🗓️ Planning final"
                )

                calendar_rows = []

                for position, name in enumerate(
                    selected_simulation,
                    start=1,
                ):

                    matching_interval = next(
                        (
                            interval
                            for interval in intervals
                            if interval["name"] == name
                        ),
                        None,
                    )

                    if matching_interval:

                        calendar_rows.append(
                            {
                                "Position": position,
                                "Intervalle": name,
                                "Début": matching_interval[
                                    "start"
                                ],
                                "Fin": matching_interval[
                                    "end"
                                ],
                            }
                        )

                st.dataframe(
                    calendar_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =================================================
                # FORMULE
                # =================================================

                st.markdown(
                    "### 🧮 Règle de sélection"
                )

                st.latex(
                    r"""
                    s_i \geq f_{\text{dernier}}
                    """
                )

                st.markdown(
                    """
                    Un intervalle \(i\) peut être sélectionné si son
                    heure de début \(s_i\) est supérieure ou égale à
                    l'heure de fin du dernier intervalle sélectionné.
                    """
                )

                st.divider()

                # =================================================
                # POURQUOI CETTE STRATÉGIE EST GLOUTONNE
                # =================================================

                st.markdown(
                    "### 💡 Pourquoi Interval Scheduling est-il glouton ?"
                )

                st.write(
                    """
                    À chaque étape, l'algorithme choisit l'intervalle
                    compatible qui se termine le plus tôt.

                    Cette décision est locale : on ne cherche pas à
                    explorer toutes les combinaisons possibles.

                    Le choix d'un intervalle se terminant tôt laisse
                    cependant le maximum de temps disponible pour les
                    intervalles suivants.
                    """
                )

                st.divider()

                # =================================================
                # PROPRIÉTÉ
                # =================================================

                st.markdown(
                    "### 🔐 Propriété de compatibilité"
                )

                st.info(
                    """
                    Deux intervalles sont compatibles lorsqu'ils ne se
                    chevauchent pas.

                    Dans cette implémentation, un intervalle qui commence
                    exactement lorsque le précédent se termine est
                    considéré comme compatible.
                    """
                )

                st.divider()

                # =================================================
                # COMPLEXITÉ
                # =================================================

                st.markdown(
                    "### ⚙️ Complexité"
                )

                st.markdown(
                    r"""
                    Le tri des \(n\) intervalles par heure de fin nécessite :

                    **\[ O(n \log n) \]**

                    Le parcours des intervalles après le tri nécessite :

                    **\[ O(n) \]**

                    La complexité globale est donc :

                    **\[ O(n \log n) \]**

                    L'espace supplémentaire utilisé par l'algorithme
                    dépend de l'implémentation, mais le parcours glouton
                    lui-même nécessite essentiellement :

                    **\[ O(1) \]**
                    """
                )

                st.divider()

                # =================================================
                # VÉRIFICATION
                # =================================================

                st.markdown(
                    "### ✅ Vérification"
                )

                valid_schedule = True

                previous_end = None

                for name in selected_names:

                    matching_interval = next(
                        (
                            interval
                            for interval in intervals
                            if interval["name"] == name
                        ),
                        None,
                    )

                    if matching_interval is None:

                        valid_schedule = False
                        break

                    if previous_end is not None:

                        if (
                            matching_interval["start"]
                            < previous_end
                        ):

                            valid_schedule = False
                            break

                    previous_end = matching_interval["end"]

                if valid_schedule:

                    st.success(
                        """
                        L'ordonnancement obtenu est valide :
                        les intervalles sélectionnés ne se chevauchent pas.
                        """
                    )

                else:

                    st.error(
                        """
                        Une incohérence a été détectée dans
                        l'ordonnancement retourné.
                        """
                    )

            # ====================================================
            # GESTION DES ERREURS
            # ====================================================

            except (
                ValueError,
                TypeError,
            ) as error:

                st.error(
                    f"Erreur Interval Scheduling : {error}"
                )

            except Exception as error:

                st.error(
                    """
                    Une erreur est survenue pendant
                    l'exécution de Interval Scheduling.
                    """
                )

                st.exception(error)

    # ============================================================
# N-QUEENS — BACKTRACKING
# ============================================================

elif selected_algorithm == "n_queens":

    st.markdown("## ♛ N-Queens — Backtracking")

    st.markdown(
        """
        Le problème des **N-Reines** consiste à placer N reines
        sur un échiquier **N × N** de manière qu'aucune reine
        ne puisse en attaquer une autre.

        Une solution doit respecter trois contraintes :

        - aucune reine sur la même ligne ;
        - aucune reine sur la même colonne ;
        - aucune reine sur la même diagonale.
        """
    )

    st.info(
        """
        **Principe du Backtracking**

        L'algorithme construit la solution ligne par ligne.

        À chaque étape :

        1. choisir une colonne ;
        2. vérifier si la position est valide ;
        3. placer la reine ;
        4. continuer récursivement ;
        5. revenir en arrière si nécessaire.
        """
    )

    st.divider()

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.markdown("### ⚙️ Paramètres")

    n = st.number_input(
        "Nombre de reines",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        key="backtracking_n_queens_size",
    )

    n = int(n)

    st.caption(
        f"Échiquier : **{n} × {n}**"
    )

    # ========================================================
    # EXÉCUTION
    # ========================================================

    if st.button(
        "♛ Résoudre avec Backtracking",
        type="primary",
        use_container_width=True,
        key="execute_n_queens",
    ):

        try:

            solutions = n_queens(n)

            # ------------------------------------------------
            # AUCUNE SOLUTION
            # ------------------------------------------------

            if not solutions:

                st.warning(
                    f"""
                    Aucune solution n'existe pour **N = {n}**.
                    """
                )

            else:

                # ------------------------------------------------
                # INFORMATIONS GÉNÉRALES
                # ------------------------------------------------

                st.success(
                    f"""
                    ✅ Recherche terminée pour **N = {n}**.
                    """
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Nombre de solutions",
                        len(solutions),
                    )

                with col2:

                    st.metric(
                        "Taille de l'échiquier",
                        f"{n} × {n}",
                    )

                # ------------------------------------------------
                # SÉLECTION D'UNE SOLUTION
                # ------------------------------------------------

                st.markdown(
                    "### 🔎 Explorer une solution"
                )

                solution_index = st.selectbox(
                    "Choisir une solution",
                    range(len(solutions)),
                    format_func=lambda i: (
                        f"Solution {i + 1}"
                    ),
                    key="n_queens_solution_index",
                )

                solution = solutions[solution_index]

                # ------------------------------------------------
                # ÉCHIQUIER
                # ------------------------------------------------

                st.markdown(
                    "### ♟️ Échiquier"
                )

                board_rows = []

                for row in range(n):

                    row_data = {
                        "Ligne": row + 1
                    }

                    for column in range(n):

                        row_data[
                            f"C{column + 1}"
                        ] = (
                            "♛"
                            if solution[row] == column
                            else "·"
                        )

                    board_rows.append(row_data)

                st.dataframe(
                    board_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # POSITIONS
                # ------------------------------------------------

                st.markdown(
                    "### 📍 Positions des reines"
                )

                position_rows = []

                for row, column in enumerate(
                    solution,
                    start=1,
                ):

                    position_rows.append(
                        {
                            "Reine": f"Q{row}",
                            "Ligne": row,
                            "Colonne": column + 1,
                            "Position": (
                                f"({row}, {column + 1})"
                            ),
                        }
                    )

                st.dataframe(
                    position_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # CONTRAINTES
                # ------------------------------------------------

                st.markdown(
                    "### 🔐 Vérification des contraintes"
                )

                columns_valid = (
                    len(set(solution)) == n
                )

                diagonals_valid = True

                main_diagonals = set()
                secondary_diagonals = set()

                for row, column in enumerate(solution):

                    main_diagonal = row - column
                    secondary_diagonal = row + column

                    if main_diagonal in main_diagonals:
                        diagonals_valid = False
                        break

                    if secondary_diagonal in secondary_diagonals:
                        diagonals_valid = False
                        break

                    main_diagonals.add(
                        main_diagonal
                    )

                    secondary_diagonals.add(
                        secondary_diagonal
                    )

                if columns_valid and diagonals_valid:

                    st.success(
                        """
                        ✅ Solution valide.

                        - Une reine par ligne
                        - Aucune colonne répétée
                        - Aucune diagonale répétée
                        """
                    )

                else:

                    st.error(
                        "❌ La solution n'est pas valide."
                    )

                # ------------------------------------------------
                # FORMULES
                # ------------------------------------------------

                st.markdown(
                    "### 📐 Contraintes mathématiques"
                )

                st.markdown(
                    """
                    Si `ci` représente la colonne de la reine
                    placée sur la ligne `i`, deux reines doivent
                    respecter :
                    """
                )

                st.latex(
                    r"""
                    c_i \neq c_j
                    """
                )

                st.markdown(
                    "pour éviter une collision sur une colonne."
                )

                st.latex(
                    r"""
                    |c_i-c_j| \neq |i-j|
                    """
                )

                st.markdown(
                    "pour éviter une collision sur une diagonale."
                )

                # ------------------------------------------------
                # SIMULATION
                # ------------------------------------------------

                st.markdown(
                    "### 🔄 Construction de la solution"
                )

                simulation_rows = []

                for step, column in enumerate(
                    solution,
                    start=1,
                ):

                    simulation_rows.append(
                        {
                            "Étape": step,
                            "Ligne": step,
                            "Colonne": column + 1,
                            "Action": "Placer la reine",
                        }
                    )

                st.dataframe(
                    simulation_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                # ------------------------------------------------
                # EXPLICATION
                # ------------------------------------------------

                st.markdown(
                    "### 🧠 Comment fonctionne le Backtracking ?"
                )

                st.markdown(
                    """
                    Le programme commence par la première ligne.

                    Il essaie successivement les différentes colonnes.
                    Lorsqu'une position est valide, la reine est placée
                    et l'algorithme passe à la ligne suivante.

                    Si aucune colonne ne convient à une ligne donnée,
                    l'algorithme revient à la ligne précédente et retire
                    la reine précédemment placée.

                    C'est le **retour en arrière**, ou *backtracking*.
                    """
                )

                # ------------------------------------------------
                # COMPLEXITÉ
                # ------------------------------------------------

                st.markdown(
                    "### 📊 Complexité"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Temps",
                        "O(N!)",
                    )

                with col2:

                    st.metric(
                        "Espace",
                        "O(N)",
                    )

                st.caption(
                    """
                    La recherche peut explorer un grand nombre
                    de configurations. L'implémentation du core
                    utilise également des ensembles pour vérifier
                    efficacement les colonnes et diagonales utilisées.
                    """
                )

        except ValueError as error:

            st.error(
                f"❌ Erreur N-Queens : {error}"
            )

        except Exception as error:

            st.error(
                "❌ Une erreur inattendue est survenue."
            )

            st.exception(error)

    # ============================================================
# SUDOKU SOLVER — BACKTRACKING
# ============================================================

elif selected_algorithm == "sudoku":

    st.markdown("## 🔢 Sudoku Solver — Backtracking")

    st.markdown(
        """
        Le **Sudoku Solver** cherche une solution à une grille de
        Sudoku 9 × 9 en utilisant le **backtracking**.

        Le programme doit respecter trois contraintes :

        - chaque ligne contient les chiffres de 1 à 9 sans répétition ;
        - chaque colonne contient les chiffres de 1 à 9 sans répétition ;
        - chaque bloc 3 × 3 contient les chiffres de 1 à 9 sans répétition.

        La valeur `0` représente une case vide.
        """
    )

    st.info(
        """
        **Principe du Backtracking**

        L'algorithme cherche une case vide, essaie une valeur possible,
        vérifie si cette valeur est valide, puis continue récursivement.

        Si aucun choix ne permet d'obtenir une solution, il revient
        en arrière et essaie une autre possibilité.
        """
    )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    sudoku_examples = {
        "Exemple classique": [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ],
        "Grille presque complète": [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 0],
        ],
    }

    # ========================================================
    # SÉLECTION D'UN EXEMPLE
    # ========================================================

    st.markdown("### 📋 Grille d'exemple")

    example_name = st.selectbox(
        "Choisir une grille",
        list(sudoku_examples.keys()),
        key="sudoku_example_selector",
    )

    if st.button(
        "📥 Charger la grille",
        use_container_width=True,
        key="load_sudoku_example",
    ):
        st.session_state["sudoku_board"] = [
            row[:] for row in sudoku_examples[example_name]
        ]

        st.session_state["sudoku_solution"] = None

        st.rerun()

    # ========================================================
    # INITIALISATION
    # ========================================================

    if "sudoku_board" not in st.session_state:
        st.session_state["sudoku_board"] = [
            row[:] for row in sudoku_examples["Exemple classique"]
        ]

    if "sudoku_solution" not in st.session_state:
        st.session_state["sudoku_solution"] = None

    # ========================================================
    # GRILLE
    # ========================================================

    st.markdown("### 🧩 Grille Sudoku")

    st.caption(
        "Entrez un chiffre entre 0 et 9. "
        "`0` représente une case vide."
    )

    current_board = st.session_state["sudoku_board"]

    edited_board = []

    for row in range(9):

        columns = st.columns(9)

        current_row = []

        for col in range(9):

            value = columns[col].number_input(
                f"R{row + 1}C{col + 1}",
                min_value=0,
                max_value=9,
                value=int(current_board[row][col]),
                step=1,
                label_visibility="collapsed",
                key=f"sudoku_cell_{row}_{col}",
            )

            current_row.append(int(value))

        edited_board.append(current_row)

    # ========================================================
    # ACTIONS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        solve_clicked = st.button(
            "🔍 Résoudre",
            type="primary",
            use_container_width=True,
            key="execute_sudoku",
        )

    with col2:

        clear_clicked = st.button(
            "🗑️ Effacer",
            use_container_width=True,
            key="clear_sudoku",
        )

    # ========================================================
    # EFFACER
    # ========================================================

    if clear_clicked:

        st.session_state["sudoku_board"] = [
            [0] * 9
            for _ in range(9)
        ]

        st.session_state["sudoku_solution"] = None

        st.rerun()

    # ========================================================
    # EXÉCUTER SOLVE_SUDOKU
    # ========================================================

    if solve_clicked:

        input_board = [
            row[:]
            for row in edited_board
        ]

        try:

            solved_board = solve_sudoku(input_board)

            st.session_state["sudoku_solution"] = [
                row[:]
                for row in solved_board
            ]

            st.success(
                "✅ Sudoku résolu avec succès par backtracking."
            )

        except ValueError as error:

            st.session_state["sudoku_solution"] = None

            st.error(
                f"❌ Erreur Sudoku : {error}"
            )

        except Exception as error:

            st.session_state["sudoku_solution"] = None

            st.error(
                "❌ Une erreur inattendue est survenue."
            )

            st.exception(error)

    # ========================================================
    # SOLUTION
    # ========================================================

    solution = st.session_state["sudoku_solution"]

    if solution is not None:

        st.markdown("### ✅ Solution")

        solution_rows = []

        for row_index, row in enumerate(
            solution,
            start=1,
        ):

            row_data = {
                "Ligne": row_index
            }

            for col_index, value in enumerate(
                row,
                start=1,
            ):

                row_data[f"C{col_index}"] = value

            solution_rows.append(row_data)

        st.dataframe(
            solution_rows,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # STATISTIQUES
        # ====================================================

        empty_cells = sum(
            1
            for row in edited_board
            for value in row
            if value == 0
        )

        filled_cells = 81 - empty_cells

        st.markdown("### 📊 Analyse")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Cases vides",
                empty_cells,
            )

        with metric2:

            st.metric(
                "Cases initiales",
                filled_cells,
            )

        with metric3:

            st.metric(
                "Taille",
                "9 × 9",
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        st.markdown("### 🔐 Vérification")

        expected = set(range(1, 10))

        valid_rows = all(
            set(row) == expected
            for row in solution
        )

        valid_columns = all(
            set(
                solution[row][col]
                for row in range(9)
            ) == expected
            for col in range(9)
        )

        valid_boxes = True

        for box_row in range(0, 9, 3):

            for box_col in range(0, 9, 3):

                box = []

                for row in range(
                    box_row,
                    box_row + 3,
                ):

                    for col in range(
                        box_col,
                        box_col + 3,
                    ):

                        box.append(
                            solution[row][col]
                        )

                if set(box) != expected:

                    valid_boxes = False

        if (
            valid_rows
            and valid_columns
            and valid_boxes
        ):

            st.success(
                """
                ✅ Solution valide.

                - Toutes les lignes sont valides.
                - Toutes les colonnes sont valides.
                - Tous les blocs 3 × 3 sont valides.
                """
            )

        else:

            st.error(
                "❌ La solution ne respecte pas toutes les contraintes."
            )

        # ====================================================
        # FONCTIONNEMENT
        # ====================================================

        st.markdown(
            "### 🧠 Fonctionnement du Backtracking"
        )

        st.markdown(
            """
            Le solveur suit cette stratégie :

            **1. Rechercher une case vide**

            Le programme cherche une cellule contenant `0`.

            **2. Essayer une valeur**

            Il essaie les chiffres de `1` à `9`.

            **3. Vérifier**

            Le chiffre doit être absent de la ligne,
            de la colonne et du bloc 3 × 3.

            **4. Continuer**

            Si le choix est valide, l'algorithme passe
            à la prochaine case vide.

            **5. Revenir en arrière**

            Si aucun choix ne mène à une solution,
            l'algorithme annule le dernier choix et
            essaie une autre valeur.
            """
        )

        # ====================================================
        # PSEUDO-CODE
        # ====================================================

        st.markdown("### 📐 Pseudo-code")

        st.code(
            """solve(board):

    trouver une case vide

    si aucune case vide:
        retourner True

    pour chiffre de 1 à 9:

        si chiffre valide:

            placer chiffre

            si solve(board):
                retourner True

            retirer chiffre

    retourner False""",
            language="text",
        )

        # ====================================================
        # COMPLEXITÉ
        # ====================================================

        st.markdown("### ⏱️ Complexité")

        complexity1, complexity2 = st.columns(2)

        with complexity1:

            st.metric(
                "Pire cas",
                "O(9^81)",
            )

        with complexity2:

            st.metric(
                "Espace",
                "O(81)",
            )

        st.caption(
            """
            Dans le pire cas théorique, chaque case vide peut
            avoir jusqu'à 9 possibilités. En pratique, les
            contraintes du Sudoku réduisent fortement l'espace
            de recherche.
            """
        )

    # ============================================================
# MAZE SOLVER — BACKTRACKING
# ============================================================

elif selected_algorithm == "maze":

    st.markdown("## 🧩 Maze Solver — Backtracking")

    st.markdown(
        """
        Le **Maze Solver** utilise le backtracking pour rechercher
        un chemin entre une position de départ et une destination.

        La grille utilise :

        - `0` → case libre ;
        - `1` → case bloquée.

        Le déplacement s'effectue dans quatre directions :

        **↑ Haut · → Droite · ↓ Bas · ← Gauche**
        """
    )

    st.info(
        """
        **Principe du Backtracking**

        Le programme avance dans le labyrinthe en essayant les
        directions possibles.

        Lorsqu'un chemin mène à une impasse, l'algorithme revient
        à la position précédente et essaie une autre direction.
        """
    )

    st.divider()

    # ========================================================
    # LABYRINTHES D'EXEMPLE
    # ========================================================

    maze_examples = {
        "Petit labyrinthe": [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],

        "Labyrinthe moyen": [
            [0, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
        ],

        "Labyrinthe ouvert": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    }

    # ========================================================
    # CHOIX DU LABYRINTHE
    # ========================================================

    st.markdown("### 📋 Labyrinthe d'exemple")

    maze_example_name = st.selectbox(
        "Choisir un labyrinthe",
        list(maze_examples.keys()),
        key="maze_example_selector",
    )

    if st.button(
        "📥 Charger le labyrinthe",
        use_container_width=True,
        key="load_maze_example",
    ):

        st.session_state["maze_grid"] = [
            row[:]
            for row in maze_examples[maze_example_name]
        ]

        st.session_state["maze_path"] = None

        st.rerun()

    # ========================================================
    # INITIALISATION
    # ========================================================

    if "maze_grid" not in st.session_state:

        st.session_state["maze_grid"] = [
            row[:]
            for row in maze_examples["Petit labyrinthe"]
        ]

    if "maze_path" not in st.session_state:
        st.session_state["maze_path"] = None

    maze = st.session_state["maze_grid"]

    rows = len(maze)
    cols = len(maze[0])

    # ========================================================
    # PARAMETRES
    # ========================================================

    st.markdown("### ⚙️ Paramètres")

    col1, col2 = st.columns(2)

    with col1:

        start_row = st.number_input(
            "Ligne de départ",
            min_value=0,
            max_value=rows - 1,
            value=0,
            step=1,
            key="maze_start_row",
        )

        start_col = st.number_input(
            "Colonne de départ",
            min_value=0,
            max_value=cols - 1,
            value=0,
            step=1,
            key="maze_start_col",
        )

    with col2:

        end_row = st.number_input(
            "Ligne d'arrivée",
            min_value=0,
            max_value=rows - 1,
            value=rows - 1,
            step=1,
            key="maze_end_row",
        )

        end_col = st.number_input(
            "Colonne d'arrivée",
            min_value=0,
            max_value=cols - 1,
            value=cols - 1,
            step=1,
            key="maze_end_col",
        )

    start = (
        int(start_row),
        int(start_col),
    )

    end = (
        int(end_row),
        int(end_col),
    )

    st.caption(
        f"Départ : **{start}** · Arrivée : **{end}**"
    )

    # ========================================================
    # GRILLE
    # ========================================================

    st.markdown("### 🧩 Grille du labyrinthe")

    st.caption(
        "Modifiez les cases : `0` = libre · `1` = bloquée."
    )

    edited_maze = []

    for row in range(rows):

        columns_ui = st.columns(cols)

        current_row = []

        for col in range(cols):

            value = columns_ui[col].number_input(
                f"R{row}C{col}",
                min_value=0,
                max_value=1,
                value=int(maze[row][col]),
                step=1,
                label_visibility="collapsed",
                key=f"maze_cell_{row}_{col}",
            )

            current_row.append(int(value))

        edited_maze.append(current_row)

    # ========================================================
    # ACTIONS
    # ========================================================

    action1, action2 = st.columns(2)

    with action1:

        solve_maze_clicked = st.button(
            "🧩 Trouver le chemin",
            type="primary",
            use_container_width=True,
            key="execute_maze",
        )

    with action2:

        clear_maze_clicked = st.button(
            "🗑️ Effacer",
            use_container_width=True,
            key="clear_maze",
        )

    # ========================================================
    # EFFACER
    # ========================================================

    if clear_maze_clicked:

        st.session_state["maze_grid"] = [
            [0 for _ in range(cols)]
            for _ in range(rows)
        ]

        st.session_state["maze_path"] = None

        st.rerun()

    # ========================================================
    # EXECUTION
    # ========================================================

    if solve_maze_clicked:

        input_maze = [
            row[:]
            for row in edited_maze
        ]

        try:

            path = solve_maze(
                input_maze,
                start=start,
                end=end,
            )

            st.session_state["maze_path"] = [
                tuple(position)
                for position in path
            ]

            st.session_state["maze_grid"] = [
                row[:]
                for row in input_maze
            ]

            st.success(
                "✅ Un chemin a été trouvé avec le backtracking."
            )

        except ValueError as error:

            st.session_state["maze_path"] = None

            st.error(
                f"❌ Erreur Maze Solver : {error}"
            )

        except Exception as error:

            st.session_state["maze_path"] = None

            st.error(
                "❌ Une erreur inattendue est survenue."
            )

            st.exception(error)

    # ========================================================
    # AFFICHAGE DU CHEMIN
    # ========================================================

    path = st.session_state["maze_path"]

    if path is not None:

        st.markdown("### 🗺️ Chemin trouvé")

        path_set = set(path)

        for row in range(rows):

            columns_ui = st.columns(cols)

            for col in range(cols):

                position = (row, col)

                if position == start:
                    symbol = "🟢"

                elif position == end:
                    symbol = "🏁"

                elif position in path_set:
                    symbol = "🟡"

                elif edited_maze[row][col] == 1:
                    symbol = "⬛"

                else:
                    symbol = "⬜"

                columns_ui[col].markdown(
                    f"""
                    <div style="
                        border: 1px solid #999;
                        padding: 8px;
                        text-align: center;
                        font-size: 20px;
                    ">
                        {symbol}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(
            "🟢 Départ · 🏁 Arrivée · 🟡 Chemin · "
            "⬛ Obstacle · ⬜ Case libre"
        )

        # ====================================================
        # STATISTIQUES
        # ====================================================

        st.markdown("### 📊 Analyse")

        obstacle_count = sum(
            1
            for row in edited_maze
            for value in row
            if value == 1
        )

        path_length = len(path)

        free_cells = (
            rows * cols
            - obstacle_count
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Longueur du chemin",
                path_length,
            )

        with metric2:
            st.metric(
                "Obstacles",
                obstacle_count,
            )

        with metric3:
            st.metric(
                "Cases libres",
                free_cells,
            )

        # ====================================================
        # POSITIONS
        # ====================================================

        st.markdown("### 📍 Étapes du chemin")

        path_rows = []

        for step, position in enumerate(
            path,
            start=1,
        ):

            path_rows.append(
                {
                    "Étape": step,
                    "Ligne": position[0],
                    "Colonne": position[1],
                    "Position": str(position),
                }
            )

        st.dataframe(
            path_rows,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # VERIFICATION
        # ====================================================

        st.markdown("### 🔐 Vérification")

        valid_path = True

        if path[0] != start:
            valid_path = False

        if path[-1] != end:
            valid_path = False

        for position in path:

            row, col = position

            if (
                row < 0
                or row >= rows
                or col < 0
                or col >= cols
            ):
                valid_path = False
                break

            if edited_maze[row][col] != 0:
                valid_path = False
                break

        for index in range(len(path) - 1):

            current = path[index]
            next_position = path[index + 1]

            distance = (
                abs(current[0] - next_position[0])
                + abs(current[1] - next_position[1])
            )

            if distance != 1:
                valid_path = False
                break

        if valid_path:

            st.success(
                """
                ✅ Le chemin est valide.

                - Il commence au point de départ.
                - Il termine au point d'arrivée.
                - Il ne traverse aucun obstacle.
                - Chaque déplacement se fait vers une case voisine.
                """
            )

        else:

            st.error(
                "❌ Le chemin trouvé n'est pas valide."
            )

        # ====================================================
        # FONCTIONNEMENT
        # ====================================================

        st.markdown(
            "### 🧠 Fonctionnement du Backtracking"
        )

        st.markdown(
            """
            Le solveur explore le labyrinthe récursivement.

            **1. Position actuelle**

            Le programme commence au point de départ.

            **2. Explorer les voisins**

            Il essaie les quatre directions possibles.

            **3. Vérifier**

            Une case est utilisable si elle est dans la grille,
            libre et non encore visitée.

            **4. Continuer**

            Le programme avance vers cette case.

            **5. Backtracking**

            Si la position actuelle ne permet pas d'atteindre
            l'arrivée, le programme revient à la position précédente
            et explore une autre direction.
            """
        )

        # ====================================================
        # PSEUDO-CODE
        # ====================================================

        st.markdown("### 📐 Pseudo-code")

        st.code(
            """
solve_maze(maze, current):

    si current == destination:
        retourner le chemin

    marquer current comme visitée

    pour chaque voisin:

        si voisin est valide:

            ajouter voisin au chemin

            si solve_maze(maze, voisin):
                retourner succès

            retirer voisin du chemin

    retourner échec
            """,
            language="text",
        )

        # ====================================================
        # COMPLEXITE
        # ====================================================

        st.markdown("### ⏱️ Complexité")

        complexity1, complexity2 = st.columns(2)

        with complexity1:
            st.metric(
                "Temps",
                "O(R × C)",
            )

        with complexity2:
            st.metric(
                "Espace",
                "O(R × C)",
            )

        st.caption(
            """
            Pour une grille de R lignes et C colonnes, les cellules
            explorées sont suivies afin d'éviter les revisites.
            """
        )


# ============================================================
# SUBSETS — BACKTRACKING
# ============================================================

elif selected_algorithm == "subsets":

    st.markdown("## 📦 Subsets — Backtracking")

    st.caption(
        "Génération de tous les sous-ensembles d'une séquence"
    )

    st.markdown(
        """
        L'algorithme **Subsets** génère tous les sous-ensembles
        possibles d'une séquence.

        Pour chaque élément, deux décisions sont possibles :

        - ❌ exclure l'élément ;
        - ✅ inclure l'élément.

        Pour `n` éléments, le nombre de sous-ensembles est :

        **2ⁿ**
        """
    )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    subset_examples = {
        "Exemple simple": "1, 2, 3",
        "Deux éléments": "A, B",
        "Quatre éléments": "1, 2, 3, 4",
        "Nombres pairs": "2, 4, 6",
    }

    example_name = st.selectbox(
        "Choisir un exemple",
        list(subset_examples.keys()),
        key="subsets_example",
    )

    if st.button(
        "📥 Charger l'exemple",
        use_container_width=True,
        key="load_subsets_example",
    ):

        st.session_state["subsets_input"] = (
            subset_examples[example_name]
        )

        st.rerun()

    # ========================================================
    # SAISIE
    # ========================================================

    if "subsets_input" not in st.session_state:
        st.session_state["subsets_input"] = "1, 2, 3"

    input_text = st.text_input(
        "Séquence d'entrée",
        value=st.session_state["subsets_input"],
        key="subsets_input_widget",
        help="Séparez les éléments par des virgules.",
    )

    st.session_state["subsets_input"] = input_text

    st.caption(
        "Exemple : `1, 2, 3`"
    )

    # ========================================================
    # PARSING
    # ========================================================

    def parse_subsets_input(value):

        if not value.strip():
            raise ValueError(
                "La séquence ne peut pas être vide."
            )

        items = [
            item.strip()
            for item in value.split(",")
        ]

        if any(item == "" for item in items):
            raise ValueError(
                "Chaque élément doit contenir une valeur."
            )

        parsed = []

        for item in items:

            try:

                if "." in item:
                    parsed.append(float(item))

                else:
                    parsed.append(int(item))

            except ValueError:

                parsed.append(item)

        return parsed

    st.divider()

    # ========================================================
    # ACTIONS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        execute_subsets = st.button(
            "📦 Générer les sous-ensembles",
            type="primary",
            use_container_width=True,
            key="execute_subsets",
        )

    with col2:

        clear_subsets = st.button(
            "🗑️ Effacer",
            use_container_width=True,
            key="clear_subsets",
        )

    if clear_subsets:

        st.session_state.pop(
            "subsets_result",
            None,
        )

        st.session_state.pop(
            "subsets_data",
            None,
        )

        st.session_state["subsets_input"] = ""

        st.rerun()

    # ========================================================
    # EXECUTION
    # ========================================================

    if execute_subsets:

        try:

            data = parse_subsets_input(
                input_text
            )

            result = subsets(data)

            st.session_state["subsets_result"] = result

            st.session_state["subsets_data"] = data

        except ValueError as error:

            st.error(
                f"❌ Donnée invalide : {error}"
            )

        except Exception as error:

            st.error(
                f"❌ Erreur Subsets : {error}"
            )

    # ========================================================
    # RESULTATS
    # ========================================================

    if "subsets_result" in st.session_state:

        result = st.session_state["subsets_result"]

        data = st.session_state["subsets_data"]

        n = len(data)

        expected_count = 2 ** n

        st.divider()

        st.success(
            f"✅ {len(result)} sous-ensembles générés."
        )

        # ====================================================
        # METRIQUES
        # ====================================================

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Éléments",
                n,
            )

        with metric2:

            st.metric(
                "Sous-ensembles",
                len(result),
            )

        with metric3:

            st.metric(
                "2ⁿ attendu",
                expected_count,
            )

        if len(result) == expected_count:

            st.success(
                "✓ Le nombre de sous-ensembles correspond à 2ⁿ."
            )

        else:

            st.warning(
                "⚠️ Le nombre de résultats est différent de 2ⁿ."
            )

        # ====================================================
        # RESULTATS
        # ====================================================

        st.markdown("### 📋 Sous-ensembles générés")

        max_display = 128

        for index, subset_result in enumerate(
            result[:max_display]
        ):

            if subset_result:

                representation = (
                    "{ "
                    + ", ".join(
                        str(value)
                        for value in subset_result
                    )
                    + " }"
                )

            else:

                representation = "{ ∅ }"

            st.write(
                f"**{index + 1}.** `{representation}`"
            )

        if len(result) > max_display:

            st.info(
                f"{len(result)} sous-ensembles ont été générés. "
                f"Seuls les {max_display} premiers sont affichés."
            )

        # ====================================================
        # VERIFICATIONS
        # ====================================================

        st.markdown("### 🔐 Vérification")

        count_valid = len(result) == 2 ** n

        empty_valid = [] in result

        individual_valid = all(
            [element] in result
            for element in data
        )

        if (
            count_valid
            and empty_valid
            and individual_valid
        ):

            st.success(
                """
                ✅ Vérification réussie :

                - le nombre de résultats est correct ;
                - le sous-ensemble vide est présent ;
                - chaque élément possède son sous-ensemble individuel.
                """
            )

        else:

            st.warning(
                "⚠️ Certaines propriétés doivent être vérifiées."
            )

        # ====================================================
        # TABLEAU
        # ====================================================

        st.markdown("### 🔎 Structure des résultats")

        subset_table = []

        for index, subset_result in enumerate(result):

            subset_table.append(
                {
                    "Index": index + 1,
                    "Sous-ensemble": (
                        "{ "
                        + ", ".join(
                            str(value)
                            for value in subset_result
                        )
                        + " }"
                        if subset_result
                        else "{ ∅ }"
                    ),
                    "Taille": len(subset_result),
                }
            )

        st.dataframe(
            subset_table,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # EXPLICATION
        # ====================================================

        st.divider()

        st.markdown(
            "### 🧠 Fonctionnement du Backtracking"
        )

        st.markdown(
            """
            Pour chaque élément, l'algorithme crée deux branches :

            **❌ Branche 1 : exclure**

            L'élément n'est pas ajouté au sous-ensemble courant.

            **✅ Branche 2 : inclure**

            L'élément est ajouté au sous-ensemble courant.

            Ces deux branches sont explorées récursivement jusqu'à
            ce que tous les éléments aient été traités.
            """
        )

        if n <= 4:

            st.markdown("### 🌳 Arbre de décision")

            st.code(
                """
                         Départ
                       /       \\
                  exclure    inclure
                     |           |
                  élément     élément
                     |           |
                    ...         ...

                Chaque élément produit
                deux possibilités :

                ❌ Exclure
                ✅ Inclure

                Nombre de feuilles = 2ⁿ
                """,
                language="text",
            )

        # ====================================================
        # PSEUDOCODE
        # ====================================================

        st.markdown("### 📝 Pseudocode")

        st.code(
            """
BACKTRACK(index, current):

    si index == n:
        ajouter current aux résultats
        retourner

    # Exclure
    BACKTRACK(index + 1, current)

    # Inclure
    ajouter data[index] à current

    BACKTRACK(index + 1, current)

    retirer data[index] de current
            """,
            language="text",
        )

        # ====================================================
        # COMPLEXITE
        # ====================================================

        st.markdown("### 📊 Complexité")

        complexity1, complexity2 = st.columns(2)

        with complexity1:

            st.markdown(
                """
                **Temps**

                **O(n × 2ⁿ)**

                car il existe `2ⁿ` sous-ensembles et chaque résultat
                peut contenir jusqu'à `n` éléments.
                """
            )

        with complexity2:

            st.markdown(
                """
                **Espace**

                Profondeur de récursion :

                **O(n)**

                Espace total des résultats :

                **O(n × 2ⁿ)**
                """
            )


# ============================================================
# PERMUTATIONS — BACKTRACKING
# ============================================================

elif selected_algorithm == "permutations":

    st.markdown("## 🔄 Permutations — Backtracking")

    st.caption(
        "Génération de toutes les permutations uniques"
    )

    st.markdown(
        """
        L'algorithme **Permutations** construit toutes les façons
        possibles d'organiser les éléments d'une séquence.

        Pour `n` éléments distincts :

        **n! permutations**
        """
    )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    permutation_examples = {
        "Deux éléments": "1, 2",
        "Trois éléments": "1, 2, 3",
        "Quatre éléments": "1, 2, 3, 4",
        "Avec doublon": "1, 1, 2",
        "Lettres": "A, B, C",
    }

    example_name = st.selectbox(
        "Choisir un exemple",
        list(permutation_examples.keys()),
        key="permutations_example",
    )

    if st.button(
        "📥 Charger l'exemple",
        use_container_width=True,
        key="load_permutations_example",
    ):

        st.session_state["permutations_input"] = (
            permutation_examples[example_name]
        )

        st.rerun()

    # ========================================================
    # SAISIE
    # ========================================================

    if "permutations_input" not in st.session_state:
        st.session_state["permutations_input"] = "1, 2, 3"

    input_text = st.text_input(
        "Séquence d'entrée",
        value=st.session_state["permutations_input"],
        key="permutations_input_widget",
        help="Séparez les éléments par des virgules.",
    )

    st.session_state["permutations_input"] = input_text

    # ========================================================
    # PARSING
    # ========================================================

    def parse_permutations_input(value):

        if not value.strip():
            raise ValueError(
                "La séquence ne peut pas être vide."
            )

        items = [
            item.strip()
            for item in value.split(",")
        ]

        if any(item == "" for item in items):

            raise ValueError(
                "Chaque élément doit contenir une valeur."
            )

        parsed = []

        for item in items:

            try:

                if "." in item:
                    parsed.append(float(item))

                else:
                    parsed.append(int(item))

            except ValueError:

                parsed.append(item)

        return parsed

    st.divider()

    # ========================================================
    # ACTIONS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        execute_permutations = st.button(
            "🔄 Générer les permutations",
            type="primary",
            use_container_width=True,
            key="execute_permutations",
        )

    with col2:

        clear_permutations = st.button(
            "🗑️ Effacer",
            use_container_width=True,
            key="clear_permutations",
        )

    if clear_permutations:

        st.session_state.pop(
            "permutations_result",
            None,
        )

        st.session_state.pop(
            "permutations_data",
            None,
        )

        st.session_state["permutations_input"] = ""

        st.rerun()

    # ========================================================
    # EXECUTION
    # ========================================================

    if execute_permutations:

        try:

            data = parse_permutations_input(
                input_text
            )

            if len(data) > 8:

                st.warning(
                    "⚠️ Plus de 8 éléments peuvent produire "
                    "un très grand nombre de permutations."
                )

            result = permutations(data)

            st.session_state["permutations_result"] = result

            st.session_state["permutations_data"] = data

        except ValueError as error:

            st.error(
                f"❌ Donnée invalide : {error}"
            )

        except Exception as error:

            st.error(
                f"❌ Erreur Permutations : {error}"
            )

    # ========================================================
    # RESULTATS
    # ========================================================

    if "permutations_result" in st.session_state:

        result = st.session_state["permutations_result"]

        data = st.session_state["permutations_data"]

        n = len(data)

        distinct_count = len(set(data))

        if distinct_count == n:

            expected_count = math.factorial(n)

        else:

            from collections import Counter

            frequencies = Counter(data)

            expected_count = math.factorial(n)

            for frequency in frequencies.values():

                expected_count //= math.factorial(
                    frequency
                )

        st.divider()

        st.success(
            f"✅ {len(result)} permutation(s) unique(s) générée(s)."
        )

        # ====================================================
        # METRIQUES
        # ====================================================

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Éléments",
                n,
            )

        with metric2:

            st.metric(
                "Permutations",
                len(result),
            )

        with metric3:

            st.metric(
                "Attendu",
                expected_count,
            )

        if len(result) == expected_count:

            st.success(
                "✓ Toutes les permutations uniques attendues "
                "ont été générées."
            )

        else:

            st.warning(
                "⚠️ Le nombre de permutations est incorrect."
            )

        # ====================================================
        # AFFICHAGE
        # ====================================================

        st.markdown("### 📋 Permutations générées")

        max_display = 120

        for index, permutation_result in enumerate(
            result[:max_display]
        ):

            representation = (
                "["
                + ", ".join(
                    str(value)
                    for value in permutation_result
                )
                + "]"
            )

            st.write(
                f"**{index + 1}.** `{representation}`"
            )

        if len(result) > max_display:

            st.info(
                f"{len(result)} permutations ont été générées. "
                f"Seules les {max_display} premières sont affichées."
            )

        # ====================================================
        # VERIFICATION
        # ====================================================

        st.markdown("### 🔐 Vérification")

        all_same_length = all(
            len(item) == n
            for item in result
        )

        all_contain_same_elements = all(
            sorted(item, key=str)
            == sorted(data, key=str)
            for item in result
        )

        all_unique = (
            len(
                {
                    tuple(item)
                    for item in result
                }
            )
            == len(result)
        )

        if (
            all_same_length
            and all_contain_same_elements
            and all_unique
            and len(result) == expected_count
        ):

            st.success(
                """
                ✅ Vérification réussie :

                - toutes les permutations ont la bonne taille ;
                - elles contiennent les mêmes éléments ;
                - elles sont uniques ;
                - le nombre total est correct.
                """
            )

        else:

            st.warning(
                "⚠️ Une propriété des permutations n'est pas respectée."
            )

        # ====================================================
        # TABLEAU
        # ====================================================

        st.markdown("### 🔎 Structure des résultats")

        permutation_table = []

        for index, item in enumerate(
            result[:max_display]
        ):

            permutation_table.append(
                {
                    "Index": index + 1,
                    "Permutation": str(item),
                    "Taille": len(item),
                }
            )

        st.dataframe(
            permutation_table,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # EXPLICATION
        # ====================================================

        st.divider()

        st.markdown(
            "### 🧠 Fonctionnement du Backtracking"
        )

        st.markdown(
            """
            L'algorithme construit progressivement une permutation.

            À chaque niveau :

            1. choisir un élément non utilisé ;
            2. l'ajouter à la permutation courante ;
            3. continuer récursivement ;
            4. retirer l'élément pour explorer une autre branche.

            Le mécanisme est :

            **choisir → explorer → retirer**
            """
        )

        if n <= 4:

            st.markdown("### 🌳 Arbre de décision")

            st.code(
                """
                    Départ
                  /   |   \\
                 1    2    3
                / \\  / \\  / \\
               2   3 1  3 1  2
               |   | |  | |  |
               3   2 3  1 2  1

                Chaque niveau choisit
                un élément disponible.

                Total = n!
                """,
                language="text",
            )

        # ====================================================
        # PSEUDOCODE
        # ====================================================

        st.markdown("### 📝 Pseudocode")

        st.code(
            """
        BACKTRACK(current):

    si current contient tous les éléments:
        ajouter current aux résultats
        retourner

    pour chaque élément disponible:

        choisir l'élément

        ajouter l'élément à current

        BACKTRACK(current)

        retirer l'élément de current
            """,
            language="text",
        )

        # ====================================================
        # COMPLEXITE
        # ====================================================

        st.markdown("### 📊 Complexité")

        complexity1, complexity2 = st.columns(2)

        with complexity1:

            st.markdown(
                """
                **Temps**

                Pour `n` éléments distincts :

                **O(n × n!)**

                car il existe `n!` résultats et chaque permutation
                contient `n` éléments.
                """
            )

        with complexity2:

            st.markdown(
                """
                **Espace**

                Profondeur de récursion :

                **O(n)**

                Stockage des résultats :

                **O(n × n!)**
                """
            )

        # ====================================================
        # FORMULE
        # ====================================================

        st.markdown("### 🧮 Formule")

        if distinct_count == n:

            st.latex(
                rf"{n}! = {expected_count}"
            )

        else:

            from collections import Counter

            frequencies = Counter(data)

            denominator = " × ".join(
                f"{frequency}!"
                for frequency in frequencies.values()
                if frequency > 1
            )

            st.markdown(
                "Avec des éléments répétés :"
            )

            st.latex(
                rf"\frac{{{n}!}}{{{denominator}}}"
                rf" = {expected_count}"
            )


# ============================================================
# COMBINATION SUM — BACKTRACKING
# ============================================================

elif selected_algorithm == "combination_sum":

    st.markdown("## 🎯 Combination Sum — Backtracking")

    st.caption(
        "Recherche de toutes les combinaisons permettant d'atteindre une cible"
    )

    st.markdown(
        """
        **Combination Sum** recherche toutes les combinaisons de candidats
        dont la somme est égale à une cible.

        Un même candidat peut être utilisé **plusieurs fois**.

        Exemple :

        `candidats = [2, 3, 6, 7]`

        `cible = 7`

        Résultats :

        - `[2, 2, 3]`
        - `[7]`
        """
    )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    combination_examples = {
        "Exemple classique": {
            "candidates": "2, 3, 6, 7",
            "target": 7,
        },

        "Petite cible": {
            "candidates": "2, 3, 5",
            "target": 8,
        },

        "Pièces": {
            "candidates": "1, 2, 5",
            "target": 5,
        },

        "Aucune solution": {
            "candidates": "4, 6",
            "target": 7,
        },

        "Plusieurs solutions": {
            "candidates": "2, 3, 5",
            "target": 10,
        },
    }

    example_name = st.selectbox(
        "Choisir un exemple",
        list(combination_examples.keys()),
        key="combination_sum_example",
    )

    if st.button(
        "📥 Charger l'exemple",
        use_container_width=True,
        key="load_combination_sum_example",
    ):

        example = combination_examples[example_name]

        st.session_state["combination_sum_candidates"] = (
            example["candidates"]
        )

        st.session_state["combination_sum_target"] = (
            example["target"]
        )

        st.rerun()

    # ========================================================
    # INITIALISATION
    # ========================================================

    if "combination_sum_candidates" not in st.session_state:

        st.session_state["combination_sum_candidates"] = (
            "2, 3, 6, 7"
        )

    if "combination_sum_target" not in st.session_state:

        st.session_state["combination_sum_target"] = 7

    # ========================================================
    # SAISIE
    # ========================================================

    candidates_text = st.text_input(
        "Candidats",
        value=st.session_state[
            "combination_sum_candidates"
        ],
        key="combination_sum_candidates_widget",
        help="Entiers positifs séparés par des virgules.",
    )

    target = st.number_input(
        "Cible",
        min_value=0,
        value=int(
            st.session_state[
                "combination_sum_target"
            ]
        ),
        step=1,
        key="combination_sum_target_widget",
    )

    st.session_state[
        "combination_sum_candidates"
    ] = candidates_text

    st.session_state[
        "combination_sum_target"
    ] = int(target)

    st.divider()

    # ========================================================
    # PARSING
    # ========================================================

    def parse_combination_candidates(value):

        if not value.strip():

            raise ValueError(
                "La liste des candidats ne peut pas être vide."
            )

        items = [
            item.strip()
            for item in value.split(",")
        ]

        if any(item == "" for item in items):

            raise ValueError(
                "Chaque candidat doit contenir une valeur."
            )

        candidates = []

        for item in items:

            try:

                number = int(item)

            except ValueError:

                raise ValueError(
                    f"'{item}' n'est pas un entier valide."
                )

            if number <= 0:

                raise ValueError(
                    f"Le candidat {number} doit être "
                    "strictement positif."
                )

            candidates.append(number)

        return candidates

    # ========================================================
    # ACTIONS
    # ========================================================

    action1, action2 = st.columns(2)

    with action1:

        execute_combination_sum = st.button(
            "🎯 Rechercher les combinaisons",
            type="primary",
            use_container_width=True,
            key="execute_combination_sum",
        )

    with action2:

        clear_combination_sum = st.button(
            "🗑️ Effacer",
            use_container_width=True,
            key="clear_combination_sum",
        )

    # ========================================================
    # EFFACER
    # ========================================================

    if clear_combination_sum:

        st.session_state.pop(
            "combination_sum_result",
            None,
        )

        st.session_state.pop(
            "combination_sum_data",
            None,
        )

        st.session_state.pop(
            "combination_sum_target_result",
            None,
        )

        st.session_state[
            "combination_sum_candidates"
        ] = ""

        st.session_state[
            "combination_sum_target"
        ] = 0

        st.rerun()

    # ========================================================
    # EXECUTION
    # ========================================================

    if execute_combination_sum:

        try:

            candidates = parse_combination_candidates(
                candidates_text
            )

            target_value = int(target)

            result = combination_sum(
                candidates,
                target_value,
            )

            st.session_state[
                "combination_sum_result"
            ] = result

            st.session_state[
                "combination_sum_data"
            ] = candidates

            st.session_state[
                "combination_sum_target_result"
            ] = target_value

        except ValueError as error:

            st.error(
                f"❌ Donnée invalide : {error}"
            )

        except Exception as error:

            st.error(
                f"❌ Erreur Combination Sum : {error}"
            )

    # ========================================================
    # RESULTATS
    # ========================================================

    if "combination_sum_result" in st.session_state:

        result = st.session_state[
            "combination_sum_result"
        ]

        candidates = st.session_state[
            "combination_sum_data"
        ]

        target_value = st.session_state[
            "combination_sum_target_result"
        ]

        st.divider()

        # ====================================================
        # MESSAGE
        # ====================================================

        if result:

            st.success(
                f"✅ {len(result)} combinaison(s) trouvée(s) "
                f"pour une cible de {target_value}."
            )

        else:

            st.warning(
                f"⚠️ Aucune combinaison ne permet d'atteindre "
                f"la cible {target_value}."
            )

        # ====================================================
        # METRIQUES
        # ====================================================

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Candidats",
                len(candidates),
            )

        with metric2:

            st.metric(
                "Cible",
                target_value,
            )

        with metric3:

            st.metric(
                "Solutions",
                len(result),
            )

        # ====================================================
        # CANDIDATS
        # ====================================================

        normalized_candidates = sorted(
            set(candidates)
        )

        st.markdown("### 🔢 Candidats normalisés")

        st.code(
            str(normalized_candidates),
            language="python",
        )

        # ====================================================
        # RESULTATS
        # ====================================================

        st.markdown("### 📋 Combinaisons trouvées")

        if result:

            for index, combination in enumerate(result):

                total = sum(combination)

                representation = (
                    "["
                    + ", ".join(
                        str(value)
                        for value in combination
                    )
                    + "]"
                )

                st.write(
                    f"**{index + 1}.** "
                    f"`{representation}` "
                    f"→ **Somme = {total}**"
                )

        else:

            st.info(
                "Aucune combinaison valide n'a été trouvée."
            )

        # ====================================================
        # TABLEAU
        # ====================================================

        if result:

            st.markdown(
                "### 🔎 Vérification des solutions"
            )

            combination_table = []

            for index, combination in enumerate(result):

                combination_table.append(
                    {
                        "Index": index + 1,
                        "Combinaison": str(combination),
                        "Nombre d'éléments": len(combination),
                        "Somme": sum(combination),
                        "Cible": target_value,
                        "Valide": (
                            sum(combination)
                            == target_value
                        ),
                    }
                )

            st.dataframe(
                combination_table,
                use_container_width=True,
                hide_index=True,
            )

        # ====================================================
        # VERIFICATIONS
        # ====================================================

        st.markdown("### 🔐 Vérification")

        all_correct_sum = all(
            sum(combination) == target_value
            for combination in result
        )

        candidate_set = set(candidates)

        all_values_valid = all(
            all(
                value in candidate_set
                for value in combination
            )
            for combination in result
        )

        all_sorted = all(
            combination == sorted(combination)
            for combination in result
        )

        all_unique = (
            len(
                {
                    tuple(combination)
                    for combination in result
                }
            )
            == len(result)
        )

        if (
            all_correct_sum
            and all_values_valid
            and all_sorted
            and all_unique
        ):

            st.success(
                """
                ✅ Toutes les solutions respectent les propriétés
                attendues :

                - somme correcte ;
                - candidats valides ;
                - ordre croissant ;
                - solutions uniques.
                """
            )

        else:

            st.warning(
                "⚠️ Une ou plusieurs propriétés ne sont pas respectées."
            )

        # ====================================================
        # EXPLICATION
        # ====================================================

        st.divider()

        st.markdown(
            "### 🧠 Fonctionnement du Backtracking"
        )

        st.markdown(
            """
            L'algorithme construit progressivement une combinaison.

            À chaque étape :

            **1. Choisir** un candidat.

            **2. Ajouter** le candidat à la combinaison courante.

            **3. Calculer** la somme restante.

            **4. Continuer** si la cible n'est pas encore dépassée.

            **5. Revenir en arrière** lorsque la branche ne peut
            plus produire une solution.

            **6. Enregistrer** la combinaison lorsque la somme
            atteint exactement la cible.

            Un candidat peut être réutilisé plusieurs fois.
            """
        )

        # ====================================================
        # EXEMPLE DE TRACE
        # ====================================================

        if result:

            st.markdown(
                "### 🔬 Trace d'une combinaison"
            )

            selected_index = st.selectbox(
                "Choisir une solution",
                range(len(result)),
                format_func=lambda i: (
                    f"Combinaison {i + 1}"
                ),
                key="combination_sum_solution_index",
            )

            selected_combination = result[
                selected_index
            ]

            trace_rows = []

            current_sum = 0

            for step, value in enumerate(
                selected_combination,
                start=1,
            ):

                current_sum += value

                trace_rows.append(
                    {
                        "Étape": step,
                        "Candidat choisi": value,
                        "Somme courante": current_sum,
                        "Cible": target_value,
                        "État": (
                            "🎯 Cible atteinte"
                            if current_sum == target_value
                            else "➡️ Continuer"
                        ),
                    }
                )

            st.dataframe(
                trace_rows,
                use_container_width=True,
                hide_index=True,
            )

        # ====================================================
        # PSEUDOCODE
        # ====================================================

        st.markdown("### 📝 Pseudocode")

        st.code(
            """
BACKTRACK(start, remaining, current):

    si remaining == 0:
        ajouter current aux résultats
        retourner

    si remaining < 0:
        retourner

    pour i de start jusqu'à la fin:

        ajouter candidates[i] à current

        BACKTRACK(
            i,
            remaining - candidates[i],
            current
        )

        retirer candidates[i] de current
            """,
            language="text",
        )

        # ====================================================
        # COMPLEXITE
        # ====================================================

        st.markdown("### 📊 Complexité")

        complexity1, complexity2 = st.columns(2)

        with complexity1:

            st.markdown(
                """
                **Temps**

                Le nombre de branches peut croître
                exponentiellement avec la cible.

                **Complexité exponentielle**
                """
            )

        with complexity2:

            st.markdown(
                """
                **Espace**

                La profondeur maximale dépend de :

                `target / min(candidates)`

                Les résultats générés nécessitent également
                un espace supplémentaire.
                """
            )

        # ====================================================
        # PRINCIPE MATHEMATIQUE
        # ====================================================

        st.markdown("### 🧮 Principe mathématique")

        st.markdown(
            f"""
            Chaque solution vérifie :

            `a₁ + a₂ + ... + aₖ = {target_value}`

            avec chaque `aᵢ` appartenant aux candidats.

            L'objectif est de trouver toutes les solutions sans
            générer deux fois la même combinaison.
            """
        )         


    

    # ========================================================
    # AUTRES ALGORITHMES GREEDY
    # ========================================================

    else:

        st.info(
            """
            Les expérimentations Greedy seront ajoutées
            progressivement.

            Algorithmes disponibles :

            - Activity Selection
            - Fractional Knapsack
            - Greedy Coin Change
            - Huffman Coding
            - Job Sequencing
            - Interval Scheduling
            """
        )

# ============================================================
# TREE HELPERS
# ============================================================

def parse_tree_values(text):
    """Convertit une chaîne '8, 4, 12' en liste d'entiers."""

    if not text.strip():
        raise ValueError("Veuillez entrer au moins une valeur.")

    try:
        values = [
            int(item.strip())
            for item in text.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "Toutes les valeurs doivent être des entiers."
        ) from exc

    if not values:
        raise ValueError("L'arbre ne peut pas être vide.")

    return values


def build_binary_tree(values):
    """
    Construit un arbre binaire complet à partir d'une liste.

    Exemple :

                  8
               /     \
              4       12
             / \     /  \
            2   6   10   14
    """

    if not values:
        return None

    nodes = [TreeNode(value) for value in values]

    for index, node in enumerate(nodes):

        left_index = 2 * index + 1
        right_index = 2 * index + 2

        if left_index < len(nodes):
            node.left = nodes[left_index]

        if right_index < len(nodes):
            node.right = nodes[right_index]

    return nodes[0]


def build_bst(values):
    """Construit un Binary Search Tree à partir d'une liste."""

    root = None

    for value in values:
        root = bst_insert(root, value)

    return root

# ============================================================
# TREES
# ============================================================

if selected_category == "Trees":

    # ========================================================
    # PREORDER TRAVERSAL
    # ========================================================

    if selected_algorithm == "preorder_traversal":

        st.markdown("## 🌳 Preorder Traversal")
        st.caption(
            "Parcours d'un arbre binaire : "
            "Root → Left → Right"
        )

        st.markdown("""
        **Principe :**

        1. Visiter le nœud courant
        2. Parcourir le sous-arbre gauche
        3. Parcourir le sous-arbre droit

        **Complexité :** O(n)
        """)

        values_text = st.text_input(
            "Valeurs de l'arbre",
            value="8, 4, 12, 2, 6, 10, 14",
            key="tree_preorder_input",
            help="Entrez les valeurs séparées par des virgules.",
        )

        if st.button(
            "▶️ Exécuter Preorder",
            key="run_preorder",
            use_container_width=True,
        ):

            try:

                values = parse_tree_values(values_text)

                # Construction du TreeNode
                root = build_binary_tree(values)

                # Exécution de l'algorithme
                result = preorder_traversal(root)

                st.success(
                    "Parcours Preorder exécuté avec succès."
                )

                st.markdown("### 🌳 Structure")

                st.code("""
                  8
               /     \\
              4       12
             / \\     /  \\
            2   6   10   14
                """)

                st.markdown("### 📥 Entrée")

                st.code(
                    " → ".join(map(str, values))
                )

                st.markdown("### 📋 Résultat")

                st.code(
                    " → ".join(map(str, result))
                )

                st.metric(
                    "Nombre de nœuds",
                    len(result),
                )

                if len(result) == len(values):

                    st.success(
                        "✓ Tous les nœuds ont été parcourus."
                    )

                else:

                    st.error(
                        "✗ Le nombre de nœuds retournés "
                        "est incorrect."
                    )

            except Exception as e:

                st.error(
                    f"Erreur Preorder Traversal : {e}"
                )


    # ========================================================
    # INORDER TRAVERSAL
    # ========================================================

    elif selected_algorithm == "inorder_traversal":

        st.markdown("## 🌿 Inorder Traversal")
        st.caption(
            "Parcours : Left → Root → Right"
        )

        st.markdown("""
        **Principe :**

        1. Parcourir le sous-arbre gauche
        2. Visiter le nœud courant
        3. Parcourir le sous-arbre droit

        Pour un BST, le parcours Inorder produit
        les valeurs dans l'ordre croissant.

        **Complexité :** O(n)
        """)

        values_text = st.text_input(
            "Valeurs de l'arbre",
            value="8, 4, 12, 2, 6, 10, 14",
            key="tree_inorder_input",
        )

        if st.button(
            "▶️ Exécuter Inorder",
            key="run_inorder",
            use_container_width=True,
        ):

            try:

                values = parse_tree_values(values_text)

                # Construction du TreeNode
                root = build_binary_tree(values)

                # Exécution
                result = inorder_traversal(root)

                st.success(
                    "Parcours Inorder exécuté avec succès."
                )

                st.markdown("### 📥 Entrée")

                st.code(
                    " → ".join(map(str, values))
                )

                st.markdown("### 📋 Résultat")

                st.code(
                    " → ".join(map(str, result))
                )

                st.metric(
                    "Nombre de nœuds",
                    len(result),
                )

                if len(result) == len(values):

                    st.success(
                        "✓ Tous les nœuds ont été parcourus."
                    )

            except Exception as e:

                st.error(
                    f"Erreur Inorder Traversal : {e}"
                )


    # ========================================================
    # POSTORDER TRAVERSAL
    # ========================================================

    elif selected_algorithm == "postorder_traversal":

        st.markdown("## 🍃 Postorder Traversal")
        st.caption(
            "Parcours : Left → Right → Root"
        )

        st.markdown("""
        **Principe :**

        1. Parcourir le sous-arbre gauche
        2. Parcourir le sous-arbre droit
        3. Visiter le nœud courant

        **Complexité :** O(n)
        """)

        values_text = st.text_input(
            "Valeurs de l'arbre",
            value="8, 4, 12, 2, 6, 10, 14",
            key="tree_postorder_input",
        )

        if st.button(
            "▶️ Exécuter Postorder",
            key="run_postorder",
            use_container_width=True,
        ):

            try:

                values = parse_tree_values(values_text)

                root = build_binary_tree(values)

                result = postorder_traversal(root)

                st.success(
                    "Parcours Postorder exécuté avec succès."
                )

                st.markdown("### 📥 Entrée")

                st.code(
                    " → ".join(map(str, values))
                )

                st.markdown("### 📋 Résultat")

                st.code(
                    " → ".join(map(str, result))
                )

                st.metric(
                    "Nombre de nœuds",
                    len(result),
                )

                if len(result) == len(values):

                    st.success(
                        "✓ Tous les nœuds ont été parcourus."
                    )

            except Exception as e:

                st.error(
                    f"Erreur Postorder Traversal : {e}"
                )


    # ========================================================
    # LEVEL ORDER TRAVERSAL
    # ========================================================

    elif selected_algorithm == "level_order_traversal":

        st.markdown("## 📊 Level Order Traversal")
        st.caption(
            "Parcours niveau par niveau — "
            "Breadth-First Traversal"
        )

        st.markdown("""
        **Principe :**

        L'arbre est parcouru niveau par niveau,
        de gauche à droite.

        **Structure utilisée :** Queue

        **Complexité :** O(n)
        """)

        values_text = st.text_input(
            "Valeurs de l'arbre",
            value="8, 4, 12, 2, 6, 10, 14",
            key="tree_level_input",
        )

        if st.button(
            "▶️ Exécuter Level Order",
            key="run_level_order",
            use_container_width=True,
        ):

            try:

                values = parse_tree_values(values_text)

                root = build_binary_tree(values)

                result = level_order_traversal(root)

                st.success(
                    "Parcours Level Order exécuté avec succès."
                )

                st.markdown("### 📥 Entrée")

                st.code(
                    " → ".join(map(str, values))
                )

                st.markdown("### 📋 Résultat")

                st.code(
                    " → ".join(map(str, result))
                )

                st.metric(
                    "Nombre de nœuds",
                    len(result),
                )

            except Exception as e:

                st.error(
                    f"Erreur Level Order Traversal : {e}"
                )


    # ========================================================
    # BST SEARCH
    # ========================================================

    elif selected_algorithm == "bst_search":

        st.markdown("## 🔎 BST Search")

        st.caption(
            "Recherche d'une valeur dans un "
            "Binary Search Tree"
        )

        st.markdown("""
        **Règle d'un BST :**

        - valeur plus petite → gauche
        - valeur plus grande → droite
        - valeur égale → trouvée

        **Complexité moyenne :** O(log n)

        **Pire cas :** O(n)
        """)

        values_text = st.text_input(
            "Valeurs du BST",
            value="8, 4, 12, 2, 6, 10, 14",
            key="bst_search_values",
        )

        target = st.number_input(
            "Valeur recherchée",
            value=6,
            step=1,
            key="bst_search_target",
        )

        if st.button(
            "🔎 Rechercher",
            key="run_bst_search",
            use_container_width=True,
        ):

            try:

                values = parse_tree_values(values_text)

                # Construction du BST
                root = build_bst(values)

                # Recherche dans le BST
                result = bst_search(
                    root,
                    int(target),
                )

                if result is not None:

                    st.success(
                        f"✓ La valeur {int(target)} "
                        "a été trouvée."
                    )

                else:

                    st.info(
                        f"La valeur {int(target)} "
                        "n'est pas présente."
                    )

                st.markdown("### 🌳 BST — Inorder")

                inorder_result = inorder_traversal(root)

                st.code(
                    " → ".join(
                        map(str, inorder_result)
                    )
                )

                st.metric(
                    "Valeur recherchée",
                    int(target),
                )

            except Exception as e:

                st.error(
                    f"Erreur BST Search : {e}"
                )


    # ========================================================
    # BST INSERT
    # ========================================================

    elif selected_algorithm == "bst_insert":

        st.markdown("## ➕ BST Insert")
        st.caption(
            "Insertion d'une nouvelle valeur dans un BST"
        )

        values_text = st.text_input(
            "Valeurs initiales",
            value="8, 4, 12, 2, 6, 10, 14",
            key="bst_insert_values",
        )

        new_value = st.number_input(
            "Valeur à insérer",
            value=7,
            step=1,
            key="bst_insert_value",
        )

        if st.button(
            "➕ Insérer",
            key="run_bst_insert",
            use_container_width=True,
        ):

            try:

                values = parse_tree_values(values_text)

                root = build_bst(values)

                # Insertion
                root = bst_insert(
                    root,
                    int(new_value),
                )

                # Affichage du BST avec Inorder
                result = inorder_traversal(root)

                st.success(
                    "Insertion exécutée avec succès."
                )

                st.markdown("### 📥 Valeurs initiales")

                st.code(
                    " → ".join(map(str, values))
                )

                st.markdown("### ➕ Valeur insérée")

                st.code(str(int(new_value)))

                st.markdown("### 🌳 BST après insertion")

                st.code(
                    " → ".join(map(str, result))
                )

                st.metric(
                    "Nombre de nœuds",
                    len(result),
                )

                if int(new_value) in result:

                    st.success(
                        "✓ La valeur est présente dans le BST."
                    )

            except Exception as e:

                st.error(
                    f"Erreur BST Insert : {e}"
                )


    # ========================================================
    # BST DELETE
    # ========================================================

    elif selected_algorithm == "bst_delete":

        st.markdown("## ➖ BST Delete")
        st.caption(
            "Suppression d'une valeur dans un BST"
        )

        st.markdown("""
        La suppression d'un nœud peut correspondre
        à trois situations :

        1. Le nœud est une feuille.
        2. Le nœud possède un seul enfant.
        3. Le nœud possède deux enfants.

        **Complexité moyenne :** O(log n)

        **Pire cas :** O(n)
        """)

        values_text = st.text_input(
            "Valeurs initiales",
            value="8, 4, 12, 2, 6, 10, 14",
            key="bst_delete_values",
        )

        delete_value = st.number_input(
            "Valeur à supprimer",
            value=6,
            step=1,
            key="bst_delete_value",
        )

        if st.button(
            "➖ Supprimer",
            key="run_bst_delete",
            use_container_width=True,
        ):

            try:

                values = parse_tree_values(values_text)

                root = build_bst(values)

                # Suppression
                root = bst_delete(
                    root,
                    int(delete_value),
                )

                if root is None:

                    result = []

                else:

                    result = inorder_traversal(root)

                st.success(
                    "Suppression exécutée avec succès."
                )

                st.markdown("### 📥 Valeurs initiales")

                st.code(
                    " → ".join(map(str, values))
                )

                st.markdown("### ➖ Valeur supprimée")

                st.code(str(int(delete_value)))

                st.markdown("### 🌳 BST après suppression")

                if result:

                    st.code(
                        " → ".join(map(str, result))
                    )

                else:

                    st.info("Le BST est maintenant vide.")

                st.metric(
                    "Nombre de nœuds",
                    len(result),
                )

                if int(delete_value) not in result:

                    st.success(
                        "✓ La valeur a été supprimée du BST."
                    )

            except Exception as e:

                st.error(
                    f"Erreur BST Delete : {e}"
                )


    # ========================================================
    # MIN HEAP
    # ========================================================

    elif selected_algorithm == "min_heap":

        st.markdown("## ⛰️ Min Heap")
        st.caption(
            "Construction et manipulation d'un tas minimal"
        )

        st.markdown("""
        **Principe :**

        Dans un Min Heap, chaque parent est
        inférieur ou égal à ses enfants.

        **Formules :**

        - enfant gauche : `2i + 1`
        - enfant droit : `2i + 2`
        - parent : `(i - 1) // 2`

        **Construction :** O(n)
        """)

        values_text = st.text_input(
            "Valeurs",
            value="9, 4, 7, 1, 3, 6, 2",
            key="min_heap_input",
        )

        if st.button(
            "⛰️ Construire le Min Heap",
            key="run_min_heap",
            use_container_width=True,
        ):

            try:

                values = parse_tree_values(values_text)

                # Utilisation de la classe MinHeap
                heap = MinHeap(values)

                result = heap.to_list()

                st.success(
                    "Min Heap construit avec succès."
                )

                st.markdown("### 📥 Entrée")

                st.code(str(values))

                st.markdown("### ⛰️ Min Heap")

                st.code(str(result))

                st.metric(
                    "Nombre d'éléments",
                    heap.size(),
                )

                st.metric(
                    "Minimum",
                    heap.peek(),
                )

                # Vérification
                valid_heap = True

                for i in range(len(result)):

                    left = 2 * i + 1
                    right = 2 * i + 2

                    if (
                        left < len(result)
                        and result[i] > result[left]
                    ):
                        valid_heap = False

                    if (
                        right < len(result)
                        and result[i] > result[right]
                    ):
                        valid_heap = False

                if valid_heap:

                    st.success(
                        "✓ Propriété Min Heap respectée."
                    )

                else:

                    st.error(
                        "✗ Propriété Min Heap non respectée."
                    )

                # Test du pop
                if not heap.is_empty():

                    minimum = heap.pop()

                    st.markdown("### 📤 Extraction")

                    st.code(str(minimum))

            except Exception as e:

                st.error(
                    f"Erreur Min Heap : {e}"
                )


    # ========================================================
    # PRIORITY QUEUE
    # ========================================================

    elif selected_algorithm == "priority_queue":

        st.markdown("## 🚦 Priority Queue")
        st.caption(
            "File de priorité"
        )

        st.markdown("""
        Une Priority Queue traite les éléments
        selon leur priorité.

        Une priorité numérique plus petite est
        extraite en premier.

        **Complexité :**

        - insertion : O(log n)
        - peek : O(1)
        - pop : O(log n)
        """)

        col1, col2 = st.columns(2)

        with col1:

            item1 = st.text_input(
                "Élément 1",
                value="Urgent",
                key="priority_item_1",
            )

            priority1 = st.number_input(
                "Priorité 1",
                value=1,
                step=1,
                key="priority_value_1",
            )

            item2 = st.text_input(
                "Élément 2",
                value="Important",
                key="priority_item_2",
            )

            priority2 = st.number_input(
                "Priorité 2",
                value=2,
                step=1,
                key="priority_value_2",
            )

        with col2:

            item3 = st.text_input(
                "Élément 3",
                value="Normal",
                key="priority_item_3",
            )

            priority3 = st.number_input(
                "Priorité 3",
                value=5,
                step=1,
                key="priority_value_3",
            )

        if st.button(
            "🚦 Exécuter Priority Queue",
            key="run_priority_queue",
            use_container_width=True,
        ):

            try:

                queue = PriorityQueue()

                queue.push(
                    item1,
                    priority=int(priority1),
                )

                queue.push(
                    item2,
                    priority=int(priority2),
                )

                queue.push(
                    item3,
                    priority=int(priority3),
                )

                st.success(
                    "Priority Queue exécutée avec succès."
                )

                st.markdown("### 🔝 Élément prioritaire")

                st.code(str(queue.peek()))

                extraction_order = []

                while not queue.is_empty():

                    extraction_order.append(
                        queue.pop()
                    )

                st.markdown("### 📤 Ordre d'extraction")

                st.code(
                    " → ".join(
                        map(str, extraction_order)
                    )
                )

                st.metric(
                    "Nombre d'éléments",
                    len(extraction_order),
                )

            except Exception as e:

                st.error(
                    f"Erreur Priority Queue : {e}"
                )


    # ========================================================
    # TREES FALLBACK
    # ========================================================

    else:

        st.info(
            "Sélectionnez un algorithme Trees "
            "pour lancer son expérimentation interactive."
        )

    
elif selected_category == "Mathematical":

    # ========================================================
    # GCD — ALGORITHME D'EUCLIDE
    # ========================================================

    if selected_algorithm == "gcd":

        st.subheader("🔢 Greatest Common Divisor — GCD")

        col1, col2 = st.columns(2)

        with col1:
            gcd_a = st.number_input(
                "Entier a",
                value=48,
                step=1,
                key="mathematical_gcd_a",
            )

        with col2:
            gcd_b = st.number_input(
                "Entier b",
                value=18,
                step=1,
                key="mathematical_gcd_b",
            )

        if st.button(
            "▶ Calculer le GCD",
            key="mathematical_gcd_button",
        ):
            try:
                result = gcd(int(gcd_a), int(gcd_b))

                st.success(
                    f"PGCD({int(gcd_a)}, {int(gcd_b)}) = **{result}**"
                )

                st.code(
                    f"{int(gcd_a)} mod {int(gcd_b)} → ... → {result}",
                    language="text",
                )

            except ValueError as exc:
                st.error(f"Erreur GCD : {exc}")


    # ========================================================
    # EXTENDED GCD
    # ========================================================

    elif selected_algorithm == "extended_gcd":

        st.subheader("🔢 Extended Euclidean Algorithm")

        col1, col2 = st.columns(2)

        with col1:
            egcd_a = st.number_input(
                "Entier a",
                value=30,
                step=1,
                key="mathematical_egcd_a",
            )

        with col2:
            egcd_b = st.number_input(
                "Entier b",
                value=12,
                step=1,
                key="mathematical_egcd_b",
            )

        if st.button(
            "▶ Calculer Extended GCD",
            key="mathematical_egcd_button",
        ):
            try:
                result_g, result_x, result_y = extended_gcd(
                    int(egcd_a),
                    int(egcd_b),
                )

                st.success(
                    f"gcd = **{result_g}**"
                )

                st.info(
                    f"Coefficients : x = **{result_x}**, "
                    f"y = **{result_y}**"
                )

                st.code(
                    f"{int(egcd_a)} × {result_x} + "
                    f"{int(egcd_b)} × {result_y} = {result_g}",
                    language="text",
                )

            except ValueError as exc:
                st.error(f"Erreur Extended GCD : {exc}")


    # ========================================================
    # GENERATE PRIMES
    # ========================================================

    elif selected_algorithm == "generate_primes":

        st.subheader("🔢 Génération de nombres premiers")

        prime_limit = st.number_input(
            "Limite supérieure",
            min_value=0,
            value=50,
            step=1,
            key="mathematical_generate_primes_limit",
        )

        if st.button(
            "▶ Générer les nombres premiers",
            key="mathematical_generate_primes_button",
        ):
            try:
                result = generate_primes(int(prime_limit))

                st.success(
                    f"{len(result)} nombre(s) premier(s) trouvé(s)."
                )

                if result:
                    st.code(
                        " → ".join(map(str, result)),
                        language="text",
                    )
                else:
                    st.info(
                        "Aucun nombre premier dans cet intervalle."
                    )

            except ValueError as exc:
                st.error(
                    f"Erreur génération des nombres premiers : {exc}"
                )


    # ========================================================
    # SIEVE OF ERATOSTHENES
    # ========================================================

    elif selected_algorithm == "sieve_of_eratosthenes":

        st.subheader("🔢 Crible d'Ératosthène")

        sieve_limit = st.number_input(
            "Limite supérieure",
            min_value=0,
            value=50,
            step=1,
            key="mathematical_sieve_limit",
        )

        if st.button(
            "▶ Exécuter le crible",
            key="mathematical_sieve_button",
        ):
            try:
                result = sieve_of_eratosthenes(
                    int(sieve_limit)
                )

                st.success(
                    f"{len(result)} nombre(s) premier(s) trouvé(s)."
                )

                if result:
                    st.code(
                        " → ".join(map(str, result)),
                        language="text",
                    )
                else:
                    st.info(
                        "Aucun nombre premier trouvé."
                    )

            except ValueError as exc:
                st.error(
                    f"Erreur Crible d'Ératosthène : {exc}"
                )


    # ========================================================
    # FAST POWER
    # ========================================================

    elif selected_algorithm == "fast_power":

        st.subheader("⚡ Exponentiation rapide")

        col1, col2 = st.columns(2)

        with col1:
            power_base = st.number_input(
                "Base",
                value=2.0,
                step=1.0,
                key="mathematical_fast_power_base",
            )

        with col2:
            power_exponent = st.number_input(
                "Exposant entier",
                value=10,
                step=1,
                key="mathematical_fast_power_exponent",
            )

        if st.button(
            "▶ Calculer la puissance",
            key="mathematical_fast_power_button",
        ):
            try:
                exponent = int(power_exponent)

                result = fast_power(
                    power_base,
                    exponent,
                )

                st.success(
                    f"{power_base}^{exponent} = **{result}**"
                )

                st.caption(
                    "L'exponentiation rapide utilise "
                    "la technique d'exponentiation par carré."
                )

            except ValueError as exc:
                st.error(
                    f"Erreur exponentiation rapide : {exc}"
                )


    # ========================================================
    # MODULAR POWER
    # ========================================================

    elif selected_algorithm == "modular_power":

        st.subheader("⚡ Exponentiation modulaire")

        col1, col2, col3 = st.columns(3)

        with col1:
            modular_base = st.number_input(
                "Base",
                value=2,
                step=1,
                key="mathematical_modular_base",
            )

        with col2:
            modular_exponent = st.number_input(
                "Exposant",
                min_value=0,
                value=10,
                step=1,
                key="mathematical_modular_exponent",
            )

        with col3:
            modular_modulus = st.number_input(
                "Modulo",
                min_value=1,
                value=1000,
                step=1,
                key="mathematical_modular_modulus",
            )

        if st.button(
            "▶ Calculer",
            key="mathematical_modular_button",
        ):
            try:
                result = modular_power(
                    int(modular_base),
                    int(modular_exponent),
                    int(modular_modulus),
                )

                st.success(
                    f"({int(modular_base)}^{int(modular_exponent)}) "
                    f"mod {int(modular_modulus)} = **{result}**"
                )

            except ValueError as exc:
                st.error(
                    f"Erreur exponentiation modulaire : {exc}"
                )


    # ========================================================
    # FACTORIAL
    # ========================================================

    elif selected_algorithm == "factorial":

        st.subheader("✖ Factorielle")

        factorial_n = st.number_input(
            "Entier n",
            min_value=0,
            value=5,
            step=1,
            key="mathematical_factorial_n",
        )

        if st.button(
            "▶ Calculer n!",
            key="mathematical_factorial_button",
        ):
            try:
                n = int(factorial_n)
                result = factorial(n)

                st.success(
                    f"{n}! = **{result}**"
                )

                if n <= 20:
                    expression = " × ".join(
                        str(value)
                        for value in range(n, 0, -1)
                    )

                    if expression:
                        st.code(
                            f"{n}! = {expression} = {result}",
                            language="text",
                        )
                    else:
                        st.code(
                            "0! = 1",
                            language="text",
                        )

            except ValueError as exc:
                st.error(
                    f"Erreur factorielle : {exc}"
                )


    # ========================================================
    # PASCAL TRIANGLE
    # ========================================================

    elif selected_algorithm == "pascal_triangle":

        st.subheader("🔺 Triangle de Pascal")

        rows = st.number_input(
            "Nombre de lignes",
            min_value=0,
            value=5,
            step=1,
            key="mathematical_pascal_rows",
        )

        if st.button(
            "▶ Générer le triangle",
            key="mathematical_pascal_button",
        ):
            try:
                triangle = pascal_triangle(
                    int(rows)
                )

                if triangle:
                    st.success(
                        f"{len(triangle)} ligne(s) générée(s)."
                    )

                    for index, row in enumerate(triangle):
                        st.code(
                            " ".join(map(str, row)),
                            language="text",
                        )
                else:
                    st.info(
                        "Aucune ligne à afficher."
                    )

            except ValueError as exc:
                st.error(
                    f"Erreur Triangle de Pascal : {exc}"
                )


    
elif selected_category == "Strings":

    # ========================================================
    # NAIVE STRING SEARCH
    # ========================================================

    if selected_algorithm == "naive_string_search":

        st.subheader("🔎 Naive String Search")

        st.caption(
            "Recherche naïve de toutes les occurrences "
            "d'un motif dans un texte."
        )

        text = st.text_area(
            "Texte",
            value="ababcabc",
            key="strings_naive_text",
        )

        pattern = st.text_input(
            "Motif",
            value="abc",
            key="strings_naive_pattern",
        )

        if st.button(
            "▶ Rechercher",
            key="strings_naive_button",
        ):
            try:
                result = naive_string_search(
                    text,
                    pattern,
                )

                if result:
                    st.success(
                        f"{len(result)} occurrence(s) trouvée(s)."
                    )
                else:
                    st.info("Aucune occurrence trouvée.")

                st.code(
                    str(result),
                    language="text",
                )

            except ValueError as exc:
                st.error(
                    f"Erreur Naive String Search : {exc}"
                )


    # ========================================================
    # KMP
    # ========================================================

    elif selected_algorithm == "kmp_search":

        st.subheader("🔎 KMP — Knuth-Morris-Pratt")

        st.caption(
            "Recherche efficace d'un motif avec la table LPS."
        )

        text = st.text_area(
            "Texte",
            value="ababcabc",
            key="strings_kmp_text",
        )

        pattern = st.text_input(
            "Motif",
            value="abc",
            key="strings_kmp_pattern",
        )

        if st.button(
            "▶ Rechercher avec KMP",
            key="strings_kmp_button",
        ):
            try:
                result = kmp_search(
                    text,
                    pattern,
                )

                if result:
                    st.success(
                        f"{len(result)} occurrence(s) trouvée(s)."
                    )
                else:
                    st.info("Aucune occurrence trouvée.")

                st.code(
                    str(result),
                    language="text",
                )

            except ValueError as exc:
                st.error(
                    f"Erreur KMP : {exc}"
                )


    # ========================================================
    # RABIN-KARP
    # ========================================================

    elif selected_algorithm == "rabin_karp_search":

        st.subheader("🔎 Rabin-Karp")

        st.caption(
            "Recherche de motif utilisant une fonction de "
            "hachage et une fenêtre glissante."
        )

        text = st.text_area(
            "Texte",
            value="ababcabc",
            key="strings_rabin_karp_text",
        )

        pattern = st.text_input(
            "Motif",
            value="abc",
            key="strings_rabin_karp_pattern",
        )

        if st.button(
            "▶ Rechercher avec Rabin-Karp",
            key="strings_rabin_karp_button",
        ):
            try:
                result = rabin_karp_search(
                    text,
                    pattern,
                )

                if result:
                    st.success(
                        f"{len(result)} occurrence(s) trouvée(s)."
                    )
                else:
                    st.info("Aucune occurrence trouvée.")

                st.code(
                    str(result),
                    language="text",
                )

            except ValueError as exc:
                st.error(
                    f"Erreur Rabin-Karp : {exc}"
                )


    # ========================================================
    # Z ALGORITHM
    # ========================================================

    elif selected_algorithm == "z_algorithm_search":

        st.subheader("🔎 Z Algorithm")

        st.caption(
            "Recherche de motif basée sur le tableau Z."
        )

        text = st.text_area(
            "Texte",
            value="ababcabc",
            key="strings_z_text",
        )

        pattern = st.text_input(
            "Motif",
            value="abc",
            key="strings_z_pattern",
        )

        if st.button(
            "▶ Rechercher avec Z Algorithm",
            key="strings_z_button",
        ):
            try:
                result = z_algorithm_search(
                    text,
                    pattern,
                )

                if result:
                    st.success(
                        f"{len(result)} occurrence(s) trouvée(s)."
                    )
                else:
                    st.info("Aucune occurrence trouvée.")

                st.code(
                    str(result),
                    language="text",
                )

            except ValueError as exc:
                st.error(
                    f"Erreur Z Algorithm : {exc}"
                )


    # ========================================================
    # LEVENSHTEIN DISTANCE
    # ========================================================

    elif selected_algorithm == "levenshtein_distance":

        st.subheader("📝 Levenshtein Distance")

        st.caption(
            "Distance minimale d'édition entre deux chaînes."
        )

        col1, col2 = st.columns(2)

        with col1:
            first = st.text_input(
                "Première chaîne",
                value="kitten",
                key="strings_levenshtein_first",
            )

        with col2:
            second = st.text_input(
                "Deuxième chaîne",
                value="sitting",
                key="strings_levenshtein_second",
            )

        if st.button(
            "▶ Calculer la distance",
            key="strings_levenshtein_button",
        ):
            try:
                result = levenshtein_distance(
                    first,
                    second,
                )

                st.success(
                    f"Distance de Levenshtein : **{result}**"
                )

                st.code(
                    f'"{first}" → "{second}" = {result}',
                    language="text",
                )

            except ValueError as exc:
                st.error(
                    f"Erreur Levenshtein : {exc}"
                )


    # ========================================================
    # ALGORITHME NON RECONNU
    # ========================================================

    else:

        st.info(
            "Sélectionnez un algorithme de traitement "
            "des chaînes de caractères."
        )




    
    
