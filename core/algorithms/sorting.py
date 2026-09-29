
"""
Algorithmes de tri pour MathLab AI.

Ce module contient plusieurs algorithmes classiques de tri :

- Bubble Sort
- Selection Sort
- Insertion Sort
- Merge Sort
- Quick Sort
- Heap Sort
- Counting Sort
- Radix Sort
- Bucket Sort

Convention générale
-------------------
Chaque fonction :

- accepte une séquence de valeurs ;
- retourne une nouvelle liste triée ;
- ne modifie jamais la séquence originale.

Les algorithmes généraux fonctionnent avec des valeurs comparables.
Les algorithmes Counting Sort et Radix Sort sont réservés aux entiers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real


# ============================================================
# VALIDATION
# ============================================================


def _validate_data(data: Sequence) -> None:
    """
    Vérifie que data est une séquence exploitable.

    Les chaînes de caractères et les bytes sont volontairement
    refusés afin d'éviter de les traiter comme des séquences
    de caractères.
    """
    if isinstance(data, (str, bytes)):
        raise TypeError("Les données doivent être une séquence de valeurs.")

    if not isinstance(data, Sequence):
        raise TypeError("Les données doivent être une séquence de valeurs.")


def _validate_comparable_data(data: Sequence) -> None:
    """
    Vérifie que les éléments de data peuvent être comparés
    entre eux.
    """
    _validate_data(data)

    for index in range(len(data) - 1):
        try:
            _ = data[index] < data[index + 1]
            _ = data[index] > data[index + 1]
        except TypeError as exc:
            raise TypeError(
                "Les éléments de la séquence doivent être comparables."
            ) from exc


def _validate_integer_data(data: Sequence) -> None:
    """
    Vérifie que les données contiennent uniquement des entiers.
    """
    _validate_data(data)

    for value in data:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                "Les données doivent contenir uniquement des entiers."
            )


def _validate_numeric_data(data: Sequence) -> None:
    """
    Vérifie que les données contiennent uniquement des nombres
    réels finis.
    """
    _validate_data(data)

    for value in data:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "Les données doivent contenir uniquement des valeurs numériques."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                "Les données doivent contenir uniquement des valeurs finies."
            )


# ============================================================
# 1. BUBBLE SORT
# ============================================================


def bubble_sort(data: Sequence) -> list:
    """
    Trie les éléments avec l'algorithme Bubble Sort.

    Le principe consiste à comparer les éléments voisins et à
    les échanger lorsqu'ils sont dans le mauvais ordre.

    Complexité temporelle :
        Meilleur cas : O(n)
        Cas moyen    : O(n²)
        Pire cas     : O(n²)

    Complexité spatiale :
        O(1) hors copie de sortie.

    Returns
    -------
    list
        Nouvelle liste triée.
    """
    _validate_comparable_data(data)

    result = list(data)
    n = len(result)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True

        if not swapped:
            break

    return result


# ============================================================
# 2. SELECTION SORT
# ============================================================


def selection_sort(data: Sequence) -> list:
    """
    Trie les éléments avec l'algorithme Selection Sort.

    À chaque étape, le plus petit élément de la partie non triée
    est recherché puis placé à sa position définitive.

    Complexité temporelle :
        O(n²)

    Complexité spatiale :
        O(1) hors copie de sortie.
    """
    _validate_comparable_data(data)

    result = list(data)
    n = len(result)

    for i in range(n):
        minimum_index = i

        for j in range(i + 1, n):
            if result[j] < result[minimum_index]:
                minimum_index = j

        if minimum_index != i:
            result[i], result[minimum_index] = (
                result[minimum_index],
                result[i],
            )

    return result


# ============================================================
# 3. INSERTION SORT
# ============================================================


def insertion_sort(data: Sequence) -> list:
    """
    Trie les éléments avec l'algorithme Insertion Sort.

    Les éléments sont insérés progressivement dans une partie
    déjà triée de la liste.

    Complexité temporelle :
        Meilleur cas : O(n)
        Cas moyen    : O(n²)
        Pire cas     : O(n²)

    Complexité spatiale :
        O(1) hors copie de sortie.
    """
    _validate_comparable_data(data)

    result = list(data)

    for i in range(1, len(result)):
        key = result[i]
        j = i - 1

        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1

        result[j + 1] = key

    return result


# ============================================================
# 4. MERGE SORT
# ============================================================


def merge_sort(data: Sequence) -> list:
    """
    Trie les éléments avec l'algorithme Merge Sort.

    L'algorithme divise récursivement la séquence en deux parties,
    trie chaque partie puis fusionne les résultats.

    Complexité temporelle :
        O(n log n)

    Complexité spatiale :
        O(n)
    """
    _validate_comparable_data(data)

    result = list(data)

    if len(result) <= 1:
        return result

    middle = len(result) // 2

    left = merge_sort(result[:middle])
    right = merge_sort(result[middle:])

    return _merge(left, right)


def _merge(left: list, right: list) -> list:
    """Fusionne deux listes déjà triées."""
    merged = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


# ============================================================
# 5. QUICK SORT
# ============================================================


def quick_sort(data: Sequence) -> list:
    """
    Trie les éléments avec l'algorithme Quick Sort.

    Un pivot est choisi puis les éléments sont répartis autour
    de celui-ci avant de trier récursivement les deux partitions.

    Complexité temporelle :
        Meilleur cas : O(n log n)
        Cas moyen    : O(n log n)
        Pire cas     : O(n²)

    Complexité spatiale :
        O(n) dans cette implémentation fonctionnelle.
    """
    _validate_comparable_data(data)

    result = list(data)

    if len(result) <= 1:
        return result

    pivot = result[len(result) // 2]

    lower = [value for value in result if value < pivot]
    equal = [value for value in result if value == pivot]
    greater = [value for value in result if value > pivot]

    return quick_sort(lower) + equal + quick_sort(greater)


# ============================================================
# 6. HEAP SORT
# ============================================================


def heap_sort(data: Sequence) -> list:
    """
    Trie les éléments avec l'algorithme Heap Sort.

    L'algorithme construit un max-heap puis extrait
    successivement le plus grand élément.

    Complexité temporelle :
        O(n log n)

    Complexité spatiale :
        O(1) hors copie de sortie.
    """
    _validate_comparable_data(data)

    result = list(data)
    n = len(result)

    def sift_down(start: int, heap_size: int) -> None:
        root = start

        while True:
            child = 2 * root + 1

            if child >= heap_size:
                break

            if child + 1 < heap_size and result[child] < result[child + 1]:
                child += 1

            if result[root] >= result[child]:
                break

            result[root], result[child] = result[child], result[root]
            root = child

    # Construction du max-heap.
    for start in range(n // 2 - 1, -1, -1):
        sift_down(start, n)

    # Extraction successive du maximum.
    for end in range(n - 1, 0, -1):
        result[0], result[end] = result[end], result[0]
        sift_down(0, end)

    return result


# ============================================================
# 7. COUNTING SORT
# ============================================================


def counting_sort(data: Sequence) -> list[int]:
    """
    Trie une séquence d'entiers avec Counting Sort.

    L'algorithme compte le nombre d'occurrences de chaque valeur.

    Les valeurs négatives sont supportées.

    Complexité temporelle :
        O(n + k)

    où k représente l'étendue des valeurs.

    Complexité spatiale :
        O(n + k)
    """
    _validate_integer_data(data)

    if not data:
        return []

    minimum = min(data)
    maximum = max(data)

    range_size = maximum - minimum + 1

    counts = [0] * range_size

    for value in data:
        counts[value - minimum] += 1

    result = []

    for offset, count in enumerate(counts):
        if count:
            result.extend([offset + minimum] * count)

    return result


# ============================================================
# 8. RADIX SORT
# ============================================================


def radix_sort(data: Sequence) -> list[int]:
    """
    Trie une séquence d'entiers avec Radix Sort.

    Les nombres positifs et négatifs sont supportés.

    Complexité temporelle :
        O(d(n + k))

    où :
        d = nombre de chiffres ;
        k = base utilisée.

    Complexité spatiale :
        O(n + k)
    """
    _validate_integer_data(data)

    if not data:
        return []

    positives = [value for value in data if value >= 0]
    negatives = [-value for value in data if value < 0]

    sorted_positives = _radix_sort_non_negative(positives)
    sorted_negatives = _radix_sort_non_negative(negatives)

    # Pour les valeurs négatives, l'ordre doit être inversé.
    sorted_negatives = [-value for value in reversed(sorted_negatives)]

    return sorted_negatives + sorted_positives


def _radix_sort_non_negative(data: list[int]) -> list[int]:
    """
    Trie une liste d'entiers positifs ou nuls.
    """
    if not data:
        return []

    result = list(data)
    maximum = max(result)

    exponent = 1

    while maximum // exponent > 0:
        buckets = [[] for _ in range(10)]

        for value in result:
            digit = (value // exponent) % 10
            buckets[digit].append(value)

        result = [
            value
            for bucket in buckets
            for value in bucket
        ]

        exponent *= 10

    return result


# ============================================================
# 9. BUCKET SORT
# ============================================================


def bucket_sort(data: Sequence) -> list[float]:
    """
    Trie des nombres réels avec Bucket Sort.

    L'implémentation utilise des intervalles de taille uniforme
    entre la valeur minimale et la valeur maximale.

    Complexité moyenne :
        O(n + k)

    Pire cas :
        O(n²)

    Complexité spatiale :
        O(n + k)

    Parameters
    ----------
    data : Sequence[Real]
        Valeurs numériques réelles et finies.
    """
    _validate_numeric_data(data)

    if not data:
        return []

    values = [float(value) for value in data]

    if len(values) == 1:
        return values

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return values

    bucket_count = len(values)

    buckets = [[] for _ in range(bucket_count)]

    interval = maximum - minimum

    for value in values:
        index = int(
            (value - minimum)
            / interval
            * (bucket_count - 1)
        )

        buckets[index].append(value)

    result = []

    for bucket in buckets:
        result.extend(insertion_sort(bucket))

    return result

