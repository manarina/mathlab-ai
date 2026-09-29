
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
# AUTRES CATÉGORIES
# ============================================================

else:

    st.info(
        f"""
        L'expérimentation interactive de la catégorie
        **{selected_category}** sera ajoutée dans
        les prochaines sous-étapes.

        Pour le moment, **Searching** et **Sorting**
        sont entièrement connectés aux fonctions
        de `core.algorithms`.
        """
    )

