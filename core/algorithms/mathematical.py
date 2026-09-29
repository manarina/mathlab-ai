
"""
MathLab AI - Mathematical Algorithms.

Collection d'algorithmes mathématiques classiques :
- Algorithme d'Euclide (GCD)
- Algorithme d'Euclide étendu
- Génération de nombres premiers
- Crible d'Ératosthène
- Exponentiation rapide
- Exponentiation modulaire
- Factorielle
- Suite de Fibonacci
- Triangle de Pascal
"""

from __future__ import annotations

from numbers import Integral
from typing import List, Tuple


def _validate_integer(value: object, name: str) -> None:
    """Validate that a value is an integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")


def _validate_non_negative_integer(value: object, name: str) -> None:
    """Validate that a value is a non-negative integer."""
    _validate_integer(value, name)

    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_integer(value: object, name: str) -> None:
    """Validate that a value is a positive integer."""
    _validate_integer(value, name)

    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def gcd(a: int, b: int) -> int:
    """
    Calculate the greatest common divisor using Euclid's algorithm.

    The result is always non-negative.

    Examples
    --------
    >>> gcd(48, 18)
    6
    >>> gcd(-48, 18)
    6
    >>> gcd(0, 5)
    5
    """
    _validate_integer(a, "a")
    _validate_integer(b, "b")

    a = abs(a)
    b = abs(b)

    while b != 0:
        a, b = b, a % b

    return a


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Calculate the extended Euclidean algorithm.

    Returns (g, x, y) such that:

        a*x + b*y = g

    where g = gcd(a, b) >= 0.

    Examples
    --------
    >>> extended_gcd(30, 12)
    (6, 1, -2)
    """
    _validate_integer(a, "a")
    _validate_integer(b, "b")

    original_a = a
    original_b = b

    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r

        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    g = old_r

    if original_a < 0:
        old_s = -old_s

    if original_b < 0:
        old_t = -old_t

    return g, old_s, old_t


def generate_primes(limit: int) -> List[int]:
    """
    Generate all prime numbers less than or equal to limit.

    Parameters
    ----------
    limit:
        Upper bound, inclusive.

    Returns
    -------
    list[int]
        Primes in ascending order.

    Examples
    --------
    >>> generate_primes(10)
    [2, 3, 5, 7]
    """
    _validate_non_negative_integer(limit, "limit")

    if limit < 2:
        return []

    primes: List[int] = []

    for number in range(2, limit + 1):
        is_prime = True

        if number > 2 and number % 2 == 0:
            is_prime = False
        else:
            divisor = 3

            while divisor * divisor <= number:
                if number % divisor == 0:
                    is_prime = False
                    break

                divisor += 2

        if is_prime:
            primes.append(number)

    return primes


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """
    Generate primes up to limit using the Sieve of Eratosthenes.

    Parameters
    ----------
    limit:
        Upper bound, inclusive.

    Returns
    -------
    list[int]
        Prime numbers in ascending order.

    Examples
    --------
    >>> sieve_of_eratosthenes(20)
    [2, 3, 5, 7, 11, 13, 17, 19]
    """
    _validate_non_negative_integer(limit, "limit")

    if limit < 2:
        return []

    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False

    candidate = 2

    while candidate * candidate <= limit:
        if is_prime[candidate]:
            start = candidate * candidate

            for multiple in range(start, limit + 1, candidate):
                is_prime[multiple] = False

        candidate += 1

    return [
        number
        for number in range(2, limit + 1)
        if is_prime[number]
    ]


def fast_power(base: int | float, exponent: int) -> int | float:
    """
    Calculate base raised to an integer exponent using exponentiation
    by squaring.

    Negative exponents are supported.

    Examples
    --------
    >>> fast_power(2, 10)
    1024
    >>> fast_power(2, -3)
    0.125
    """
    _validate_integer(exponent, "exponent")

    if exponent == 0:
        return 1

    if base == 0 and exponent < 0:
        raise ValueError("Cannot raise zero to a negative exponent.")

    negative = exponent < 0
    exponent = abs(exponent)

    result = 1
    current = base

    while exponent > 0:
        if exponent % 2 == 1:
            result *= current

        current *= current
        exponent //= 2

    if negative:
        return 1 / result

    return result


def modular_power(base: int, exponent: int, modulus: int) -> int:
    """
    Calculate (base ** exponent) % modulus efficiently.

    Negative exponents are not supported.

    Parameters
    ----------
    base:
        Integer base.
    exponent:
        Non-negative integer exponent.
    modulus:
        Positive integer modulus.

    Examples
    --------
    >>> modular_power(2, 10, 1000)
    24
    """
    _validate_integer(base, "base")
    _validate_non_negative_integer(exponent, "exponent")
    _validate_positive_integer(modulus, "modulus")

    base %= modulus
    result = 1 % modulus

    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus

        base = (base * base) % modulus
        exponent //= 2

    return result


def factorial(n: int) -> int:
    """
    Calculate n! iteratively.

    Parameters
    ----------
    n:
        Non-negative integer.

    Examples
    --------
    >>> factorial(5)
    120
    >>> factorial(0)
    1
    """
    _validate_non_negative_integer(n, "n")

    result = 1

    for value in range(2, n + 1):
        result *= value

    return result


def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.

    Convention:
        F(0) = 0
        F(1) = 1

    Parameters
    ----------
    n:
        Non-negative integer.

    Examples
    --------
    >>> fibonacci(0)
    0
    >>> fibonacci(10)
    55
    """
    _validate_non_negative_integer(n, "n")

    if n == 0:
        return 0

    if n == 1:
        return 1

    previous = 0
    current = 1

    for _ in range(2, n + 1):
        previous, current = current, previous + current

    return current


def pascal_triangle(rows: int) -> List[List[int]]:
    """
    Generate the first `rows` rows of Pascal's triangle.

    The first row is [1].

    Parameters
    ----------
    rows:
        Number of rows to generate.

    Returns
    -------
    list[list[int]]

    Examples
    --------
    >>> pascal_triangle(5)
    [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    """
    _validate_non_negative_integer(rows, "rows")

    triangle: List[List[int]] = []

    for row_index in range(rows):
        if row_index == 0:
            triangle.append([1])
            continue

        previous_row = triangle[-1]
        current_row = [1]

        for index in range(len(previous_row) - 1):
            current_row.append(
                previous_row[index] + previous_row[index + 1]
            )

        current_row.append(1)
        triangle.append(current_row)

    return triangle

