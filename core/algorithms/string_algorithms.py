
"""
String Algorithms
=================

Implémentation d'algorithmes classiques de traitement et de recherche
de chaînes de caractères.

Algorithmes disponibles
-----------------------
- Naive String Search
- KMP (Knuth-Morris-Pratt)
- Rabin-Karp
- Z Algorithm
- Levenshtein Distance

Conventions
-----------
Les algorithmes de recherche retournent la liste des indices de début
de toutes les occurrences du motif dans le texte.

Exemple
-------
>>> naive_string_search("ababcabc", "abc")
[2, 5]
"""

from __future__ import annotations


# ============================================================================
# VALIDATION
# ============================================================================


def _validate_strings(text: str, pattern: str) -> None:
    """Validate text and pattern inputs."""
    if not isinstance(text, str):
        raise ValueError("text must be a string.")

    if not isinstance(pattern, str):
        raise ValueError("pattern must be a string.")


def _validate_single_string(value: str, name: str) -> None:
    """Validate a single string input."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")


# ============================================================================
# NAIVE STRING SEARCH
# ============================================================================


def naive_string_search(text: str, pattern: str) -> list[int]:
    """
    Find all occurrences of pattern in text using naive string search.

    Parameters
    ----------
    text:
        String in which to search.
    pattern:
        String to search for.

    Returns
    -------
    list[int]
        Starting indices of all occurrences.

    Notes
    -----
    Overlapping occurrences are included.

    An empty pattern returns an empty list.

    Examples
    --------
    >>> naive_string_search("hello world", "world")
    [6]

    >>> naive_string_search("aaaa", "aa")
    [0, 1, 2]

    >>> naive_string_search("abc", "xyz")
    []
    """
    _validate_strings(text, pattern)

    if not pattern or len(pattern) > len(text):
        return []

    matches: list[int] = []
    pattern_length = len(pattern)
    text_length = len(text)

    for i in range(text_length - pattern_length + 1):
        match = True

        for j in range(pattern_length):
            if text[i + j] != pattern[j]:
                match = False
                break

        if match:
            matches.append(i)

    return matches


# ============================================================================
# KMP — KNUTH-MORRIS-PRATT
# ============================================================================


def _build_lps(pattern: str) -> list[int]:
    """
    Build the LPS (Longest Prefix Suffix) table for KMP.

    lps[i] contains the length of the longest proper prefix of
    pattern[:i + 1] which is also a suffix of that substring.
    """
    lps = [0] * len(pattern)

    length = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1

    return lps


def kmp_search(text: str, pattern: str) -> list[int]:
    """
    Find all occurrences of pattern in text using KMP.

    Parameters
    ----------
    text:
        String in which to search.
    pattern:
        String to search for.

    Returns
    -------
    list[int]
        Starting indices of all occurrences.

    Notes
    -----
    KMP runs in O(n + m) time, where n is the text length and
    m is the pattern length.

    Overlapping occurrences are included.

    Examples
    --------
    >>> kmp_search("ababcabc", "abc")
    [2, 5]

    >>> kmp_search("aaaa", "aa")
    [0, 1, 2]
    """
    _validate_strings(text, pattern)

    if not pattern or len(pattern) > len(text):
        return []

    lps = _build_lps(pattern)

    matches: list[int] = []

    text_index = 0
    pattern_index = 0

    while text_index < len(text):
        if text[text_index] == pattern[pattern_index]:
            text_index += 1
            pattern_index += 1

            if pattern_index == len(pattern):
                matches.append(text_index - pattern_index)

                # Continue searching to detect overlapping matches.
                pattern_index = lps[pattern_index - 1]

        elif pattern_index > 0:
            pattern_index = lps[pattern_index - 1]
        else:
            text_index += 1

    return matches


# ============================================================================
# RABIN-KARP
# ============================================================================


def rabin_karp_search(text: str, pattern: str) -> list[int]:
    """
    Find all occurrences of pattern in text using Rabin-Karp.

    Parameters
    ----------
    text:
        String in which to search.
    pattern:
        String to search for.

    Returns
    -------
    list[int]
        Starting indices of all occurrences.

    Notes
    -----
    A rolling polynomial hash is used to identify candidate matches.
    A direct character comparison is performed after a hash collision
    to guarantee correctness.

    Average-case complexity is O(n + m).

    Overlapping occurrences are included.

    Examples
    --------
    >>> rabin_karp_search("ababcabc", "abc")
    [2, 5]

    >>> rabin_karp_search("aaaa", "aa")
    [0, 1, 2]
    """
    _validate_strings(text, pattern)

    if not pattern or len(pattern) > len(text):
        return []

    pattern_length = len(pattern)
    text_length = len(text)

    # Polynomial rolling hash parameters.
    base = 256
    modulus = 1_000_000_007

    pattern_hash = 0
    window_hash = 0

    # base^(m-1) modulo modulus.
    highest_power = pow(base, pattern_length - 1, modulus)

    for i in range(pattern_length):
        pattern_hash = (
            pattern_hash * base + ord(pattern[i])
        ) % modulus

        window_hash = (
            window_hash * base + ord(text[i])
        ) % modulus

    matches: list[int] = []

    for i in range(text_length - pattern_length + 1):
        if pattern_hash == window_hash:
            # Verify the candidate to protect against hash collisions.
            if text[i : i + pattern_length] == pattern:
                matches.append(i)

        if i < text_length - pattern_length:
            outgoing = ord(text[i])
            incoming = ord(text[i + pattern_length])

            window_hash = (
                (
                    (window_hash - outgoing * highest_power)
                    * base
                    + incoming
                )
                % modulus
            )

    return matches


# ============================================================================
# Z ALGORITHM
# ============================================================================


def _build_z_array(value: str) -> list[int]:
    """
    Build the Z-array for a string.

    z[i] is the length of the longest substring starting at i
    that is also a prefix of value.
    """
    n = len(value)

    if n == 0:
        return []

    z = [0] * n
    z[0] = n

    left = 0
    right = 0

    for i in range(1, n):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])

        while (
            i + z[i] < n
            and value[z[i]] == value[i + z[i]]
        ):
            z[i] += 1

        if i + z[i] - 1 > right:
            left = i
            right = i + z[i] - 1

    return z


def z_algorithm_search(text: str, pattern: str) -> list[int]:
    """
    Find all occurrences of pattern in text using the Z algorithm.

    Parameters
    ----------
    text:
        String in which to search.
    pattern:
        String to search for.

    Returns
    -------
    list[int]
        Starting indices of all occurrences.

    Notes
    -----
    The pattern and text are combined as:

        pattern + separator + text

    A separator not present in either input is selected dynamically.

    Complexity is O(n + m).

    Overlapping occurrences are included.

    Examples
    --------
    >>> z_algorithm_search("ababcabc", "abc")
    [2, 5]

    >>> z_algorithm_search("aaaa", "aa")
    [0, 1, 2]
    """
    _validate_strings(text, pattern)

    if not pattern or len(pattern) > len(text):
        return []

    # Select a separator that cannot appear in either string.
    separator = "\0"

    while separator in pattern or separator in text:
        separator += "\0"

    combined = pattern + separator + text
    z = _build_z_array(combined)

    pattern_length = len(pattern)
    offset = pattern_length + len(separator)

    matches: list[int] = []

    for i in range(offset, len(combined)):
        if z[i] >= pattern_length:
            matches.append(i - offset)

    return matches


# ============================================================================
# LEVENSHTEIN DISTANCE
# ============================================================================


def levenshtein_distance(first: str, second: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    The distance is the minimum number of single-character operations
    required to transform one string into the other.

    Allowed operations
    ------------------
    - insertion
    - deletion
    - substitution

    Parameters
    ----------
    first:
        First string.
    second:
        Second string.

    Returns
    -------
    int
        Minimum edit distance.

    Complexity
    ----------
    Time: O(n * m)
    Space: O(min(n, m))

    Examples
    --------
    >>> levenshtein_distance("kitten", "sitting")
    3

    >>> levenshtein_distance("flaw", "lawn")
    2

    >>> levenshtein_distance("", "abc")
    3
    """
    _validate_single_string(first, "first")
    _validate_single_string(second, "second")

    # Use the shorter string as the columns to minimize memory.
    if len(first) < len(second):
        first, second = second, first

    if not second:
        return len(first)

    previous_row = list(range(len(second) + 1))

    for i, char_first in enumerate(first, start=1):
        current_row = [i]

        for j, char_second in enumerate(second, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1

            substitution = previous_row[j - 1] + (
                char_first != char_second
            )

            current_row.append(
                min(insertion, deletion, substitution)
            )

        previous_row = current_row

    return previous_row[-1]


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "naive_string_search",
    "kmp_search",
    "rabin_karp_search",
    "z_algorithm_search",
    "levenshtein_distance",
]

