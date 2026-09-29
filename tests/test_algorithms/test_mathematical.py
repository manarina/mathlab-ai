
"""
Tests for MathLab AI mathematical algorithms.

Coverage:
- gcd
- extended_gcd
- generate_primes
- sieve_of_eratosthenes
- fast_power
- modular_power
- factorial
- fibonacci
- pascal_triangle
"""

import math

import pytest

from core.algorithms.mathematical import (
    extended_gcd,
    factorial,
    fast_power,
    fibonacci,
    gcd,
    generate_primes,
    modular_power,
    pascal_triangle,
    sieve_of_eratosthenes,
)


# ============================================================
# GCD
# ============================================================


class TestGCD:
    """Tests for Euclid's algorithm."""

    def test_gcd_basic(self):
        assert gcd(48, 18) == 6

    def test_gcd_coprime_numbers(self):
        assert gcd(17, 13) == 1

    def test_gcd_equal_numbers(self):
        assert gcd(25, 25) == 25

    def test_gcd_with_zero_first(self):
        assert gcd(0, 5) == 5

    def test_gcd_with_zero_second(self):
        assert gcd(5, 0) == 5

    def test_gcd_both_zero(self):
        assert gcd(0, 0) == 0

    def test_gcd_negative_first(self):
        assert gcd(-48, 18) == 6

    def test_gcd_negative_second(self):
        assert gcd(48, -18) == 6

    def test_gcd_both_negative(self):
        assert gcd(-48, -18) == 6

    def test_gcd_one(self):
        assert gcd(1, 1000) == 1

    def test_gcd_large_numbers(self):
        assert gcd(1234567890, 987654321) == 9

    def test_gcd_is_commutative(self):
        a = 84
        b = 126

        assert gcd(a, b) == gcd(b, a)

    def test_gcd_divides_both_numbers(self):
        a = 252
        b = 105

        result = gcd(a, b)

        assert a % result == 0
        assert b % result == 0

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (12, 8, 4),
            (100, 40, 20),
            (81, 27, 27),
            (35, 64, 1),
            (144, 60, 12),
            (270, 192, 6),
        ],
    )
    def test_gcd_parametrized(self, a, b, expected):
        assert gcd(a, b) == expected

    @pytest.mark.parametrize(
        "a,b",
        [
            (True, 5),
            (5, True),
            (False, 5),
            (5, False),
            (1.5, 3),
            (3, 1.5),
            ("12", 6),
            (12, "6"),
            (None, 5),
            (5, None),
        ],
    )
    def test_gcd_rejects_invalid_values(self, a, b):
        with pytest.raises(ValueError):
            gcd(a, b)


# ============================================================
# EXTENDED GCD
# ============================================================


class TestExtendedGCD:
    """Tests for the extended Euclidean algorithm."""

    def test_extended_gcd_basic(self):
        g, x, y = extended_gcd(30, 12)

        assert g == 6
        assert 30 * x + 12 * y == g

    def test_extended_gcd_returns_gcd(self):
        g, x, y = extended_gcd(48, 18)

        assert g == gcd(48, 18)
        assert 48 * x + 18 * y == g

    def test_extended_gcd_coprime(self):
        g, x, y = extended_gcd(17, 13)

        assert g == 1
        assert 17 * x + 13 * y == 1

    def test_extended_gcd_equal_numbers(self):
        g, x, y = extended_gcd(25, 25)

        assert g == 25
        assert 25 * x + 25 * y == g

    def test_extended_gcd_zero_first(self):
        g, x, y = extended_gcd(0, 5)

        assert g == 5
        assert 0 * x + 5 * y == g

    def test_extended_gcd_zero_second(self):
        g, x, y = extended_gcd(5, 0)

        assert g == 5
        assert 5 * x + 0 * y == g

    def test_extended_gcd_both_zero(self):
        g, x, y = extended_gcd(0, 0)

        assert g == 0
        assert 0 * x + 0 * y == g

    def test_extended_gcd_negative_first(self):
        g, x, y = extended_gcd(-30, 12)

        assert g == 6
        assert -30 * x + 12 * y == g

    def test_extended_gcd_negative_second(self):
        g, x, y = extended_gcd(30, -12)

        assert g == 6
        assert 30 * x - 12 * y == g

    def test_extended_gcd_both_negative(self):
        g, x, y = extended_gcd(-30, -12)

        assert g == 6
        assert -30 * x - 12 * y == g

    def test_extended_gcd_large_numbers(self):
        a = 1234567890
        b = 987654321

        g, x, y = extended_gcd(a, b)

        assert g == gcd(a, b)
        assert a * x + b * y == g

    def test_extended_gcd_bezout_identity(self):
        test_cases = [
            (99, 78),
            (240, 46),
            (391, 299),
            (12345, 54321),
        ]

        for a, b in test_cases:
            g, x, y = extended_gcd(a, b)

            assert a * x + b * y == g
            assert g == gcd(a, b)

    @pytest.mark.parametrize(
        "a,b",
        [
            (True, 5),
            (5, True),
            (1.5, 3),
            (3, 1.5),
            ("12", 6),
            (12, "6"),
            (None, 5),
            (5, None),
        ],
    )
    def test_extended_gcd_rejects_invalid_values(self, a, b):
        with pytest.raises(ValueError):
            extended_gcd(a, b)


# ============================================================
# GENERATE PRIMES
# ============================================================


class TestGeneratePrimes:
    """Tests for prime generation."""

    def test_generate_primes_below_two(self):
        assert generate_primes(0) == []
        assert generate_primes(1) == []

    def test_generate_primes_two(self):
        assert generate_primes(2) == [2]

    def test_generate_primes_three(self):
        assert generate_primes(3) == [2, 3]

    def test_generate_primes_ten(self):
        assert generate_primes(10) == [2, 3, 5, 7]

    def test_generate_primes_twenty(self):
        assert generate_primes(20) == [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
        ]

    def test_generate_primes_includes_limit_when_prime(self):
        assert generate_primes(13)[-1] == 13

    def test_generate_primes_excludes_limit_when_composite(self):
        assert 14 not in generate_primes(14)

    def test_generate_primes_are_sorted(self):
        result = generate_primes(100)

        assert result == sorted(result)

    def test_generate_primes_all_are_prime(self):
        result = generate_primes(100)

        for number in result:
            assert number >= 2

            for divisor in range(2, math.isqrt(number) + 1):
                assert number % divisor != 0

    def test_generate_primes_large_limit(self):
        result = generate_primes(1000)

        assert result[0] == 2
        assert result[-1] == 997
        assert len(result) == 168

    @pytest.mark.parametrize(
        "limit,expected",
        [
            (0, []),
            (1, []),
            (2, [2]),
            (5, [2, 3, 5]),
            (11, [2, 3, 5, 7, 11]),
            (15, [2, 3, 5, 7, 11, 13]),
        ],
    )
    def test_generate_primes_parametrized(self, limit, expected):
        assert generate_primes(limit) == expected

    @pytest.mark.parametrize(
        "limit",
        [
            True,
            False,
            1.5,
            "10",
            None,
            [],
        ],
    )
    def test_generate_primes_rejects_invalid_values(self, limit):
        with pytest.raises(ValueError):
            generate_primes(limit)

    def test_generate_primes_does_not_include_one(self):
        assert 1 not in generate_primes(100)

    def test_generate_primes_does_not_include_zero(self):
        assert 0 not in generate_primes(100)


# ============================================================
# SIEVE OF ERATOSTHENES
# ============================================================


class TestSieveOfEratosthenes:
    """Tests for the Sieve of Eratosthenes."""

    def test_sieve_below_two(self):
        assert sieve_of_eratosthenes(0) == []
        assert sieve_of_eratosthenes(1) == []

    def test_sieve_two(self):
        assert sieve_of_eratosthenes(2) == [2]

    def test_sieve_ten(self):
        assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]

    def test_sieve_twenty(self):
        assert sieve_of_eratosthenes(20) == [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
        ]

    def test_sieve_includes_prime_limit(self):
        assert sieve_of_eratosthenes(13)[-1] == 13

    def test_sieve_excludes_composite_limit(self):
        assert 15 not in sieve_of_eratosthenes(15)

    def test_sieve_are_sorted(self):
        result = sieve_of_eratosthenes(100)

        assert result == sorted(result)

    def test_sieve_all_are_prime(self):
        result = sieve_of_eratosthenes(100)

        for number in result:
            for divisor in range(2, math.isqrt(number) + 1):
                assert number % divisor != 0

    def test_sieve_large_limit(self):
        result = sieve_of_eratosthenes(1000)

        assert result[0] == 2
        assert result[-1] == 997
        assert len(result) == 168

    def test_sieve_matches_generate_primes(self):
        for limit in [0, 1, 2, 10, 50, 100, 250]:
            assert sieve_of_eratosthenes(limit) == generate_primes(limit)

    @pytest.mark.parametrize(
        "limit",
        [
            True,
            False,
            1.5,
            "20",
            None,
            [],
        ],
    )
    def test_sieve_rejects_invalid_values(self, limit):
        with pytest.raises(ValueError):
            sieve_of_eratosthenes(limit)


# ============================================================
# FAST POWER
# ============================================================


class TestFastPower:
    """Tests for exponentiation by squaring."""

    def test_fast_power_basic(self):
        assert fast_power(2, 10) == 1024

    def test_fast_power_zero_exponent(self):
        assert fast_power(10, 0) == 1

    def test_fast_power_exponent_one(self):
        assert fast_power(7, 1) == 7

    def test_fast_power_negative_exponent(self):
        assert fast_power(2, -3) == pytest.approx(0.125)

    def test_fast_power_negative_base(self):
        assert fast_power(-2, 4) == 16

    def test_fast_power_negative_base_odd_exponent(self):
        assert fast_power(-2, 3) == -8

    def test_fast_power_negative_base_negative_exponent(self):
        assert fast_power(-2, -3) == pytest.approx(-0.125)

    def test_fast_power_zero_base_positive_exponent(self):
        assert fast_power(0, 5) == 0

    def test_fast_power_zero_to_zero(self):
        assert fast_power(0, 0) == 1

    def test_fast_power_zero_negative_exponent_rejected(self):
        with pytest.raises(ValueError):
            fast_power(0, -1)

    def test_fast_power_float_base(self):
        assert fast_power(2.5, 2) == pytest.approx(6.25)

    def test_fast_power_float_negative_exponent(self):
        assert fast_power(2.0, -2) == pytest.approx(0.25)

    def test_fast_power_matches_builtin(self):
        test_cases = [
            (2, 8),
            (3, 5),
            (-2, 7),
            (5, 0),
            (10, 3),
            (1.5, 4),
        ]

        for base, exponent in test_cases:
            assert fast_power(base, exponent) == pytest.approx(
                base**exponent
            )

    def test_fast_power_large_exponent(self):
        assert fast_power(2, 30) == 2**30

    @pytest.mark.parametrize(
        "base,exponent,expected",
        [
            (2, 5, 32),
            (3, 4, 81),
            (10, 3, 1000),
            (-2, 4, 16),
            (-2, 5, -32),
            (5, 0, 1),
        ],
    )
    def test_fast_power_parametrized(self, base, exponent, expected):
        assert fast_power(base, exponent) == expected

    @pytest.mark.parametrize(
        "exponent",
        [
            True,
            False,
            1.5,
            "5",
            None,
            [],
        ],
    )
    def test_fast_power_rejects_invalid_exponent(self, exponent):
        with pytest.raises(ValueError):
            fast_power(2, exponent)


# ============================================================
# MODULAR POWER
# ============================================================


class TestModularPower:
    """Tests for modular exponentiation."""

    def test_modular_power_basic(self):
        assert modular_power(2, 10, 1000) == 24

    def test_modular_power_zero_exponent(self):
        assert modular_power(10, 0, 7) == 1

    def test_modular_power_exponent_one(self):
        assert modular_power(17, 1, 5) == 2

    def test_modular_power_base_larger_than_modulus(self):
        assert modular_power(17, 3, 5) == pow(17, 3, 5)

    def test_modular_power_negative_base(self):
        assert modular_power(-2, 5, 7) == pow(-2, 5, 7)

    def test_modular_power_zero_base(self):
        assert modular_power(0, 5, 7) == 0

    def test_modular_power_zero_base_zero_exponent(self):
        assert modular_power(0, 0, 7) == 1

    def test_modular_power_modulus_one(self):
        assert modular_power(12345, 100, 1) == 0

    def test_modular_power_large_exponent(self):
        result = modular_power(2, 1000, 1000000007)

        assert result == pow(2, 1000, 1000000007)

    def test_modular_power_matches_builtin(self):
        test_cases = [
            (2, 10, 1000),
            (3, 20, 17),
            (-5, 13, 19),
            (123, 50, 97),
            (7, 0, 13),
        ]

        for base, exponent, modulus in test_cases:
            assert modular_power(base, exponent, modulus) == pow(
                base,
                exponent,
                modulus,
            )

    @pytest.mark.parametrize(
        "base,exponent,modulus,expected",
        [
            (2, 5, 7, 4),
            (3, 4, 5, 1),
            (10, 3, 6, 4),
            (-2, 3, 5, 2),
            (5, 0, 3, 1),
        ],
    )
    def test_modular_power_parametrized(
        self,
        base,
        exponent,
        modulus,
        expected,
    ):
        assert modular_power(base, exponent, modulus) == expected

    @pytest.mark.parametrize(
        "base,exponent,modulus",
        [
            (2, -1, 5),
            (2, -5, 7),
        ],
    )
    def test_modular_power_rejects_negative_exponent(
        self,
        base,
        exponent,
        modulus,
    ):
        with pytest.raises(ValueError):
            modular_power(base, exponent, modulus)

    @pytest.mark.parametrize(
        "modulus",
        [
            0,
            -1,
            -10,
            True,
            False,
            1.5,
            "7",
            None,
        ],
    )
    def test_modular_power_rejects_invalid_modulus(self, modulus):
        with pytest.raises(ValueError):
            modular_power(2, 5, modulus)

    @pytest.mark.parametrize(
        "base",
        [
            True,
            False,
            1.5,
            "2",
            None,
        ],
    )
    def test_modular_power_rejects_invalid_base(self, base):
        with pytest.raises(ValueError):
            modular_power(base, 5, 7)


# ============================================================
# FACTORIAL
# ============================================================


class TestFactorial:
    """Tests for factorial."""

    def test_factorial_zero(self):
        assert factorial(0) == 1

    def test_factorial_one(self):
        assert factorial(1) == 1

    def test_factorial_two(self):
        assert factorial(2) == 2

    def test_factorial_five(self):
        assert factorial(5) == 120

    def test_factorial_ten(self):
        assert factorial(10) == 3628800

    def test_factorial_large_value(self):
        assert factorial(20) == math.factorial(20)

    def test_factorial_matches_math_factorial(self):
        for n in range(21):
            assert factorial(n) == math.factorial(n)

    def test_factorial_positive_integer(self):
        result = factorial(6)

        assert result > 0
        assert result == 720

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (4, 24),
            (5, 120),
            (6, 720),
            (7, 5040),
        ],
    )
    def test_factorial_parametrized(self, n, expected):
        assert factorial(n) == expected

    @pytest.mark.parametrize(
        "n",
        [
            -1,
            -5,
            -100,
        ],
    )
    def test_factorial_rejects_negative(self, n):
        with pytest.raises(ValueError):
            factorial(n)

    @pytest.mark.parametrize(
        "n",
        [
            True,
            False,
            1.5,
            "5",
            None,
            [],
        ],
    )
    def test_factorial_rejects_invalid_values(self, n):
        with pytest.raises(ValueError):
            factorial(n)

    def test_factorial_recursive_identity(self):
        for n in range(1, 10):
            assert factorial(n) == n * factorial(n - 1)


# ============================================================
# FIBONACCI
# ============================================================


class TestFibonacci:
    """Tests for Fibonacci numbers."""

    def test_fibonacci_zero(self):
        assert fibonacci(0) == 0

    def test_fibonacci_one(self):
        assert fibonacci(1) == 1

    def test_fibonacci_two(self):
        assert fibonacci(2) == 1

    def test_fibonacci_three(self):
        assert fibonacci(3) == 2

    def test_fibonacci_ten(self):
        assert fibonacci(10) == 55

    def test_fibonacci_twenty(self):
        assert fibonacci(20) == 6765

    def test_fibonacci_large_value(self):
        assert fibonacci(50) == 12586269025

    def test_fibonacci_sequence(self):
        expected = [
            0,
            1,
            1,
            2,
            3,
            5,
            8,
            13,
            21,
            34,
        ]

        result = [fibonacci(n) for n in range(10)]

        assert result == expected

    def test_fibonacci_recurrence(self):
        for n in range(2, 20):
            assert fibonacci(n) == fibonacci(n - 1) + fibonacci(n - 2)

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 5),
            (6, 8),
            (7, 13),
            (8, 21),
            (9, 34),
            (10, 55),
            (15, 610),
        ],
    )
    def test_fibonacci_parametrized(self, n, expected):
        assert fibonacci(n) == expected

    @pytest.mark.parametrize(
        "n",
        [
            -1,
            -5,
            -100,
        ],
    )
    def test_fibonacci_rejects_negative(self, n):
        with pytest.raises(ValueError):
            fibonacci(n)

    @pytest.mark.parametrize(
        "n",
        [
            True,
            False,
            1.5,
            "10",
            None,
            [],
        ],
    )
    def test_fibonacci_rejects_invalid_values(self, n):
        with pytest.raises(ValueError):
            fibonacci(n)

    def test_fibonacci_ratio_approaches_golden_ratio(self):
        value_n = fibonacci(30)
        value_previous = fibonacci(29)

        ratio = value_n / value_previous
        golden_ratio = (1 + math.sqrt(5)) / 2

        assert ratio == pytest.approx(golden_ratio, rel=1e-4)


# ============================================================
# PASCAL TRIANGLE
# ============================================================


class TestPascalTriangle:
    """Tests for Pascal's triangle."""

    def test_pascal_zero_rows(self):
        assert pascal_triangle(0) == []

    def test_pascal_one_row(self):
        assert pascal_triangle(1) == [[1]]

    def test_pascal_two_rows(self):
        assert pascal_triangle(2) == [
            [1],
            [1, 1],
        ]

    def test_pascal_three_rows(self):
        assert pascal_triangle(3) == [
            [1],
            [1, 1],
            [1, 2, 1],
        ]

    def test_pascal_five_rows(self):
        assert pascal_triangle(5) == [
            [1],
            [1, 1],
            [1, 2, 1],
            [1, 3, 3, 1],
            [1, 4, 6, 4, 1],
        ]

    def test_pascal_ten_rows(self):
        triangle = pascal_triangle(10)

        assert len(triangle) == 10
        assert triangle[9] == [
            1,
            9,
            36,
            84,
            126,
            126,
            84,
            36,
            9,
            1,
        ]

    def test_pascal_row_lengths(self):
        triangle = pascal_triangle(10)

        for index, row in enumerate(triangle):
            assert len(row) == index + 1

    def test_pascal_rows_start_and_end_with_one(self):
        triangle = pascal_triangle(10)

        for row in triangle:
            assert row[0] == 1
            assert row[-1] == 1

    def test_pascal_is_symmetric(self):
        triangle = pascal_triangle(15)

        for row in triangle:
            assert row == row[::-1]

    def test_pascal_binomial_coefficients(self):
        triangle = pascal_triangle(10)

        for row_index, row in enumerate(triangle):
            for column_index, value in enumerate(row):
                expected = math.comb(row_index, column_index)

                assert value == expected

    def test_pascal_row_sums_are_powers_of_two(self):
        triangle = pascal_triangle(12)

        for row_index, row in enumerate(triangle):
            assert sum(row) == 2**row_index

    def test_pascal_does_not_mutate_previous_rows(self):
        triangle = pascal_triangle(6)

        original_first_row = triangle[0].copy()

        triangle[-1][0] = 999

        assert triangle[0] == original_first_row

    @pytest.mark.parametrize(
        "rows,expected",
        [
            (0, []),
            (1, [[1]]),
            (2, [[1], [1, 1]]),
            (3, [[1], [1, 1], [1, 2, 1]]),
        ],
    )
    def test_pascal_parametrized(self, rows, expected):
        assert pascal_triangle(rows) == expected

    @pytest.mark.parametrize(
        "rows",
        [
            -1,
            -5,
            -100,
        ],
    )
    def test_pascal_rejects_negative(self, rows):
        with pytest.raises(ValueError):
            pascal_triangle(rows)

    @pytest.mark.parametrize(
        "rows",
        [
            True,
            False,
            1.5,
            "5",
            None,
            [],
        ],
    )
    def test_pascal_rejects_invalid_values(self, rows):
        with pytest.raises(ValueError):
            pascal_triangle(rows)


# ============================================================
# CROSS-ALGORITHM CONSISTENCY
# ============================================================


class TestMathematicalAlgorithmsConsistency:
    """Cross-check mathematical relationships between algorithms."""

    def test_gcd_and_extended_gcd_are_consistent(self):
        test_cases = [
            (48, 18),
            (99, 78),
            (240, 46),
            (391, 299),
            (12345, 54321),
        ]

        for a, b in test_cases:
            g = gcd(a, b)
            extended_result, x, y = extended_gcd(a, b)

            assert extended_result == g
            assert a * x + b * y == g

    def test_prime_generation_methods_are_consistent(self):
        for limit in [2, 10, 50, 100, 250, 500]:
            assert generate_primes(limit) == sieve_of_eratosthenes(limit)

    def test_fast_power_and_modular_power_are_consistent(self):
        test_cases = [
            (2, 10, 1000),
            (3, 7, 17),
            (5, 8, 13),
            (-2, 9, 19),
        ]

        for base, exponent, modulus in test_cases:
            expected = fast_power(base, exponent) % modulus

            assert modular_power(
                base,
                exponent,
                modulus,
            ) == expected

    def test_factorial_related_to_pascal(self):
        triangle = pascal_triangle(8)

        for n in range(1, 8):
            for k in range(n + 1):
                expected = factorial(n) // (
                    factorial(k) * factorial(n - k)
                )

                assert triangle[n][k] == expected

    def test_fibonacci_values_are_integers(self):
        values = [fibonacci(n) for n in range(20)]

        assert all(isinstance(value, int) for value in values)

    def test_prime_count_for_100(self):
        primes = sieve_of_eratosthenes(100)

        assert len(primes) == 25
        assert primes[-1] == 97

    def test_pascal_contains_expected_binomial_coefficients(self):
        triangle = pascal_triangle(7)

        assert triangle[6] == [
            math.comb(6, k)
            for k in range(7)
        ]


"""
Tests for MathLab AI mathematical algorithms.

Coverage:
- gcd
- extended_gcd
- generate_primes
- sieve_of_eratosthenes
- fast_power
- modular_power
- factorial
- fibonacci
- pascal_triangle
"""

import math

import pytest

from core.algorithms.mathematical import (
    extended_gcd,
    factorial,
    fast_power,
    fibonacci,
    gcd,
    generate_primes,
    modular_power,
    pascal_triangle,
    sieve_of_eratosthenes,
)


# ============================================================
# GCD
# ============================================================


class TestGCD:
    """Tests for Euclid's algorithm."""

    def test_gcd_basic(self):
        assert gcd(48, 18) == 6

    def test_gcd_coprime_numbers(self):
        assert gcd(17, 13) == 1

    def test_gcd_equal_numbers(self):
        assert gcd(25, 25) == 25

    def test_gcd_with_zero_first(self):
        assert gcd(0, 5) == 5

    def test_gcd_with_zero_second(self):
        assert gcd(5, 0) == 5

    def test_gcd_both_zero(self):
        assert gcd(0, 0) == 0

    def test_gcd_negative_first(self):
        assert gcd(-48, 18) == 6

    def test_gcd_negative_second(self):
        assert gcd(48, -18) == 6

    def test_gcd_both_negative(self):
        assert gcd(-48, -18) == 6

    def test_gcd_one(self):
        assert gcd(1, 1000) == 1

    def test_gcd_large_numbers(self):
        assert gcd(1234567890, 987654321) == 9

    def test_gcd_is_commutative(self):
        a = 84
        b = 126

        assert gcd(a, b) == gcd(b, a)

    def test_gcd_divides_both_numbers(self):
        a = 252
        b = 105

        result = gcd(a, b)

        assert a % result == 0
        assert b % result == 0

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (12, 8, 4),
            (100, 40, 20),
            (81, 27, 27),
            (35, 64, 1),
            (144, 60, 12),
            (270, 192, 6),
        ],
    )
    def test_gcd_parametrized(self, a, b, expected):
        assert gcd(a, b) == expected

    @pytest.mark.parametrize(
        "a,b",
        [
            (True, 5),
            (5, True),
            (False, 5),
            (5, False),
            (1.5, 3),
            (3, 1.5),
            ("12", 6),
            (12, "6"),
            (None, 5),
            (5, None),
        ],
    )
    def test_gcd_rejects_invalid_values(self, a, b):
        with pytest.raises(ValueError):
            gcd(a, b)


# ============================================================
# EXTENDED GCD
# ============================================================


class TestExtendedGCD:
    """Tests for the extended Euclidean algorithm."""

    def test_extended_gcd_basic(self):
        g, x, y = extended_gcd(30, 12)

        assert g == 6
        assert 30 * x + 12 * y == g

    def test_extended_gcd_returns_gcd(self):
        g, x, y = extended_gcd(48, 18)

        assert g == gcd(48, 18)
        assert 48 * x + 18 * y == g

    def test_extended_gcd_coprime(self):
        g, x, y = extended_gcd(17, 13)

        assert g == 1
        assert 17 * x + 13 * y == 1

    def test_extended_gcd_equal_numbers(self):
        g, x, y = extended_gcd(25, 25)

        assert g == 25
        assert 25 * x + 25 * y == g

    def test_extended_gcd_zero_first(self):
        g, x, y = extended_gcd(0, 5)

        assert g == 5
        assert 0 * x + 5 * y == g

    def test_extended_gcd_zero_second(self):
        g, x, y = extended_gcd(5, 0)

        assert g == 5
        assert 5 * x + 0 * y == g

    def test_extended_gcd_both_zero(self):
        g, x, y = extended_gcd(0, 0)

        assert g == 0
        assert 0 * x + 0 * y == g

    def test_extended_gcd_negative_first(self):
        g, x, y = extended_gcd(-30, 12)

        assert g == 6
        assert -30 * x + 12 * y == g

    def test_extended_gcd_negative_second(self):
        g, x, y = extended_gcd(30, -12)

        assert g == 6
        assert 30 * x - 12 * y == g

    def test_extended_gcd_both_negative(self):
        g, x, y = extended_gcd(-30, -12)

        assert g == 6
        assert -30 * x - 12 * y == g

    def test_extended_gcd_large_numbers(self):
        a = 1234567890
        b = 987654321

        g, x, y = extended_gcd(a, b)

        assert g == gcd(a, b)
        assert a * x + b * y == g

    def test_extended_gcd_bezout_identity(self):
        test_cases = [
            (99, 78),
            (240, 46),
            (391, 299),
            (12345, 54321),
        ]

        for a, b in test_cases:
            g, x, y = extended_gcd(a, b)

            assert a * x + b * y == g
            assert g == gcd(a, b)

    @pytest.mark.parametrize(
        "a,b",
        [
            (True, 5),
            (5, True),
            (1.5, 3),
            (3, 1.5),
            ("12", 6),
            (12, "6"),
            (None, 5),
            (5, None),
        ],
    )
    def test_extended_gcd_rejects_invalid_values(self, a, b):
        with pytest.raises(ValueError):
            extended_gcd(a, b)


# ============================================================
# GENERATE PRIMES
# ============================================================


class TestGeneratePrimes:
    """Tests for prime generation."""

    def test_generate_primes_below_two(self):
        assert generate_primes(0) == []
        assert generate_primes(1) == []

    def test_generate_primes_two(self):
        assert generate_primes(2) == [2]

    def test_generate_primes_three(self):
        assert generate_primes(3) == [2, 3]

    def test_generate_primes_ten(self):
        assert generate_primes(10) == [2, 3, 5, 7]

    def test_generate_primes_twenty(self):
        assert generate_primes(20) == [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
        ]

    def test_generate_primes_includes_limit_when_prime(self):
        assert generate_primes(13)[-1] == 13

    def test_generate_primes_excludes_limit_when_composite(self):
        assert 14 not in generate_primes(14)

    def test_generate_primes_are_sorted(self):
        result = generate_primes(100)

        assert result == sorted(result)

    def test_generate_primes_all_are_prime(self):
        result = generate_primes(100)

        for number in result:
            assert number >= 2

            for divisor in range(2, math.isqrt(number) + 1):
                assert number % divisor != 0

    def test_generate_primes_large_limit(self):
        result = generate_primes(1000)

        assert result[0] == 2
        assert result[-1] == 997
        assert len(result) == 168

    @pytest.mark.parametrize(
        "limit,expected",
        [
            (0, []),
            (1, []),
            (2, [2]),
            (5, [2, 3, 5]),
            (11, [2, 3, 5, 7, 11]),
            (15, [2, 3, 5, 7, 11, 13]),
        ],
    )
    def test_generate_primes_parametrized(self, limit, expected):
        assert generate_primes(limit) == expected

    @pytest.mark.parametrize(
        "limit",
        [
            True,
            False,
            1.5,
            "10",
            None,
            [],
        ],
    )
    def test_generate_primes_rejects_invalid_values(self, limit):
        with pytest.raises(ValueError):
            generate_primes(limit)

    def test_generate_primes_does_not_include_one(self):
        assert 1 not in generate_primes(100)

    def test_generate_primes_does_not_include_zero(self):
        assert 0 not in generate_primes(100)


# ============================================================
# SIEVE OF ERATOSTHENES
# ============================================================


class TestSieveOfEratosthenes:
    """Tests for the Sieve of Eratosthenes."""

    def test_sieve_below_two(self):
        assert sieve_of_eratosthenes(0) == []
        assert sieve_of_eratosthenes(1) == []

    def test_sieve_two(self):
        assert sieve_of_eratosthenes(2) == [2]

    def test_sieve_ten(self):
        assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]

    def test_sieve_twenty(self):
        assert sieve_of_eratosthenes(20) == [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
        ]

    def test_sieve_includes_prime_limit(self):
        assert sieve_of_eratosthenes(13)[-1] == 13

    def test_sieve_excludes_composite_limit(self):
        assert 15 not in sieve_of_eratosthenes(15)

    def test_sieve_are_sorted(self):
        result = sieve_of_eratosthenes(100)

        assert result == sorted(result)

    def test_sieve_all_are_prime(self):
        result = sieve_of_eratosthenes(100)

        for number in result:
            for divisor in range(2, math.isqrt(number) + 1):
                assert number % divisor != 0

    def test_sieve_large_limit(self):
        result = sieve_of_eratosthenes(1000)

        assert result[0] == 2
        assert result[-1] == 997
        assert len(result) == 168

    def test_sieve_matches_generate_primes(self):
        for limit in [0, 1, 2, 10, 50, 100, 250]:
            assert sieve_of_eratosthenes(limit) == generate_primes(limit)

    @pytest.mark.parametrize(
        "limit",
        [
            True,
            False,
            1.5,
            "20",
            None,
            [],
        ],
    )
    def test_sieve_rejects_invalid_values(self, limit):
        with pytest.raises(ValueError):
            sieve_of_eratosthenes(limit)


# ============================================================
# FAST POWER
# ============================================================


class TestFastPower:
    """Tests for exponentiation by squaring."""

    def test_fast_power_basic(self):
        assert fast_power(2, 10) == 1024

    def test_fast_power_zero_exponent(self):
        assert fast_power(10, 0) == 1

    def test_fast_power_exponent_one(self):
        assert fast_power(7, 1) == 7

    def test_fast_power_negative_exponent(self):
        assert fast_power(2, -3) == pytest.approx(0.125)

    def test_fast_power_negative_base(self):
        assert fast_power(-2, 4) == 16

    def test_fast_power_negative_base_odd_exponent(self):
        assert fast_power(-2, 3) == -8

    def test_fast_power_negative_base_negative_exponent(self):
        assert fast_power(-2, -3) == pytest.approx(-0.125)

    def test_fast_power_zero_base_positive_exponent(self):
        assert fast_power(0, 5) == 0

    def test_fast_power_zero_to_zero(self):
        assert fast_power(0, 0) == 1

    def test_fast_power_zero_negative_exponent_rejected(self):
        with pytest.raises(ValueError):
            fast_power(0, -1)

    def test_fast_power_float_base(self):
        assert fast_power(2.5, 2) == pytest.approx(6.25)

    def test_fast_power_float_negative_exponent(self):
        assert fast_power(2.0, -2) == pytest.approx(0.25)

    def test_fast_power_matches_builtin(self):
        test_cases = [
            (2, 8),
            (3, 5),
            (-2, 7),
            (5, 0),
            (10, 3),
            (1.5, 4),
        ]

        for base, exponent in test_cases:
            assert fast_power(base, exponent) == pytest.approx(
                base**exponent
            )

    def test_fast_power_large_exponent(self):
        assert fast_power(2, 30) == 2**30

    @pytest.mark.parametrize(
        "base,exponent,expected",
        [
            (2, 5, 32),
            (3, 4, 81),
            (10, 3, 1000),
            (-2, 4, 16),
            (-2, 5, -32),
            (5, 0, 1),
        ],
    )
    def test_fast_power_parametrized(self, base, exponent, expected):
        assert fast_power(base, exponent) == expected

    @pytest.mark.parametrize(
        "exponent",
        [
            True,
            False,
            1.5,
            "5",
            None,
            [],
        ],
    )
    def test_fast_power_rejects_invalid_exponent(self, exponent):
        with pytest.raises(ValueError):
            fast_power(2, exponent)


# ============================================================
# MODULAR POWER
# ============================================================


class TestModularPower:
    """Tests for modular exponentiation."""

    def test_modular_power_basic(self):
        assert modular_power(2, 10, 1000) == 24

    def test_modular_power_zero_exponent(self):
        assert modular_power(10, 0, 7) == 1

    def test_modular_power_exponent_one(self):
        assert modular_power(17, 1, 5) == 2

    def test_modular_power_base_larger_than_modulus(self):
        assert modular_power(17, 3, 5) == pow(17, 3, 5)

    def test_modular_power_negative_base(self):
        assert modular_power(-2, 5, 7) == pow(-2, 5, 7)

    def test_modular_power_zero_base(self):
        assert modular_power(0, 5, 7) == 0

    def test_modular_power_zero_base_zero_exponent(self):
        assert modular_power(0, 0, 7) == 1

    def test_modular_power_modulus_one(self):
        assert modular_power(12345, 100, 1) == 0

    def test_modular_power_large_exponent(self):
        result = modular_power(2, 1000, 1000000007)

        assert result == pow(2, 1000, 1000000007)

    def test_modular_power_matches_builtin(self):
        test_cases = [
            (2, 10, 1000),
            (3, 20, 17),
            (-5, 13, 19),
            (123, 50, 97),
            (7, 0, 13),
        ]

        for base, exponent, modulus in test_cases:
            assert modular_power(base, exponent, modulus) == pow(
                base,
                exponent,
                modulus,
            )

    @pytest.mark.parametrize(
        "base,exponent,modulus,expected",
        [
            (2, 5, 7, 4),
            (3, 4, 5, 1),
            (10, 3, 6, 4),
            (-2, 3, 5, 2),
            (5, 0, 3, 1),
        ],
    )
    def test_modular_power_parametrized(
        self,
        base,
        exponent,
        modulus,
        expected,
    ):
        assert modular_power(base, exponent, modulus) == expected

    @pytest.mark.parametrize(
        "base,exponent,modulus",
        [
            (2, -1, 5),
            (2, -5, 7),
        ],
    )
    def test_modular_power_rejects_negative_exponent(
        self,
        base,
        exponent,
        modulus,
    ):
        with pytest.raises(ValueError):
            modular_power(base, exponent, modulus)

    @pytest.mark.parametrize(
        "modulus",
        [
            0,
            -1,
            -10,
            True,
            False,
            1.5,
            "7",
            None,
        ],
    )
    def test_modular_power_rejects_invalid_modulus(self, modulus):
        with pytest.raises(ValueError):
            modular_power(2, 5, modulus)

    @pytest.mark.parametrize(
        "base",
        [
            True,
            False,
            1.5,
            "2",
            None,
        ],
    )
    def test_modular_power_rejects_invalid_base(self, base):
        with pytest.raises(ValueError):
            modular_power(base, 5, 7)


# ============================================================
# FACTORIAL
# ============================================================


class TestFactorial:
    """Tests for factorial."""

    def test_factorial_zero(self):
        assert factorial(0) == 1

    def test_factorial_one(self):
        assert factorial(1) == 1

    def test_factorial_two(self):
        assert factorial(2) == 2

    def test_factorial_five(self):
        assert factorial(5) == 120

    def test_factorial_ten(self):
        assert factorial(10) == 3628800

    def test_factorial_large_value(self):
        assert factorial(20) == math.factorial(20)

    def test_factorial_matches_math_factorial(self):
        for n in range(21):
            assert factorial(n) == math.factorial(n)

    def test_factorial_positive_integer(self):
        result = factorial(6)

        assert result > 0
        assert result == 720

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (4, 24),
            (5, 120),
            (6, 720),
            (7, 5040),
        ],
    )
    def test_factorial_parametrized(self, n, expected):
        assert factorial(n) == expected

    @pytest.mark.parametrize(
        "n",
        [
            -1,
            -5,
            -100,
        ],
    )
    def test_factorial_rejects_negative(self, n):
        with pytest.raises(ValueError):
            factorial(n)

    @pytest.mark.parametrize(
        "n",
        [
            True,
            False,
            1.5,
            "5",
            None,
            [],
        ],
    )
    def test_factorial_rejects_invalid_values(self, n):
        with pytest.raises(ValueError):
            factorial(n)

    def test_factorial_recursive_identity(self):
        for n in range(1, 10):
            assert factorial(n) == n * factorial(n - 1)


# ============================================================
# FIBONACCI
# ============================================================


class TestFibonacci:
    """Tests for Fibonacci numbers."""

    def test_fibonacci_zero(self):
        assert fibonacci(0) == 0

    def test_fibonacci_one(self):
        assert fibonacci(1) == 1

    def test_fibonacci_two(self):
        assert fibonacci(2) == 1

    def test_fibonacci_three(self):
        assert fibonacci(3) == 2

    def test_fibonacci_ten(self):
        assert fibonacci(10) == 55

    def test_fibonacci_twenty(self):
        assert fibonacci(20) == 6765

    def test_fibonacci_large_value(self):
        assert fibonacci(50) == 12586269025

    def test_fibonacci_sequence(self):
        expected = [
            0,
            1,
            1,
            2,
            3,
            5,
            8,
            13,
            21,
            34,
        ]

        result = [fibonacci(n) for n in range(10)]

        assert result == expected

    def test_fibonacci_recurrence(self):
        for n in range(2, 20):
            assert fibonacci(n) == fibonacci(n - 1) + fibonacci(n - 2)

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 5),
            (6, 8),
            (7, 13),
            (8, 21),
            (9, 34),
            (10, 55),
            (15, 610),
        ],
    )
    def test_fibonacci_parametrized(self, n, expected):
        assert fibonacci(n) == expected

    @pytest.mark.parametrize(
        "n",
        [
            -1,
            -5,
            -100,
        ],
    )
    def test_fibonacci_rejects_negative(self, n):
        with pytest.raises(ValueError):
            fibonacci(n)

    @pytest.mark.parametrize(
        "n",
        [
            True,
            False,
            1.5,
            "10",
            None,
            [],
        ],
    )
    def test_fibonacci_rejects_invalid_values(self, n):
        with pytest.raises(ValueError):
            fibonacci(n)

    def test_fibonacci_ratio_approaches_golden_ratio(self):
        value_n = fibonacci(30)
        value_previous = fibonacci(29)

        ratio = value_n / value_previous
        golden_ratio = (1 + math.sqrt(5)) / 2

        assert ratio == pytest.approx(golden_ratio, rel=1e-4)


# ============================================================
# PASCAL TRIANGLE
# ============================================================


class TestPascalTriangle:
    """Tests for Pascal's triangle."""

    def test_pascal_zero_rows(self):
        assert pascal_triangle(0) == []

    def test_pascal_one_row(self):
        assert pascal_triangle(1) == [[1]]

    def test_pascal_two_rows(self):
        assert pascal_triangle(2) == [
            [1],
            [1, 1],
        ]

    def test_pascal_three_rows(self):
        assert pascal_triangle(3) == [
            [1],
            [1, 1],
            [1, 2, 1],
        ]

    def test_pascal_five_rows(self):
        assert pascal_triangle(5) == [
            [1],
            [1, 1],
            [1, 2, 1],
            [1, 3, 3, 1],
            [1, 4, 6, 4, 1],
        ]

    def test_pascal_ten_rows(self):
        triangle = pascal_triangle(10)

        assert len(triangle) == 10
        assert triangle[9] == [
            1,
            9,
            36,
            84,
            126,
            126,
            84,
            36,
            9,
            1,
        ]

    def test_pascal_row_lengths(self):
        triangle = pascal_triangle(10)

        for index, row in enumerate(triangle):
            assert len(row) == index + 1

    def test_pascal_rows_start_and_end_with_one(self):
        triangle = pascal_triangle(10)

        for row in triangle:
            assert row[0] == 1
            assert row[-1] == 1

    def test_pascal_is_symmetric(self):
        triangle = pascal_triangle(15)

        for row in triangle:
            assert row == row[::-1]

    def test_pascal_binomial_coefficients(self):
        triangle = pascal_triangle(10)

        for row_index, row in enumerate(triangle):
            for column_index, value in enumerate(row):
                expected = math.comb(row_index, column_index)

                assert value == expected

    def test_pascal_row_sums_are_powers_of_two(self):
        triangle = pascal_triangle(12)

        for row_index, row in enumerate(triangle):
            assert sum(row) == 2**row_index

    def test_pascal_does_not_mutate_previous_rows(self):
        triangle = pascal_triangle(6)

        original_first_row = triangle[0].copy()

        triangle[-1][0] = 999

        assert triangle[0] == original_first_row

    @pytest.mark.parametrize(
        "rows,expected",
        [
            (0, []),
            (1, [[1]]),
            (2, [[1], [1, 1]]),
            (3, [[1], [1, 1], [1, 2, 1]]),
        ],
    )
    def test_pascal_parametrized(self, rows, expected):
        assert pascal_triangle(rows) == expected

    @pytest.mark.parametrize(
        "rows",
        [
            -1,
            -5,
            -100,
        ],
    )
    def test_pascal_rejects_negative(self, rows):
        with pytest.raises(ValueError):
            pascal_triangle(rows)

    @pytest.mark.parametrize(
        "rows",
        [
            True,
            False,
            1.5,
            "5",
            None,
            [],
        ],
    )
    def test_pascal_rejects_invalid_values(self, rows):
        with pytest.raises(ValueError):
            pascal_triangle(rows)


# ============================================================
# CROSS-ALGORITHM CONSISTENCY
# ============================================================


class TestMathematicalAlgorithmsConsistency:
    """Cross-check mathematical relationships between algorithms."""

    def test_gcd_and_extended_gcd_are_consistent(self):
        test_cases = [
            (48, 18),
            (99, 78),
            (240, 46),
            (391, 299),
            (12345, 54321),
        ]

        for a, b in test_cases:
            g = gcd(a, b)
            extended_result, x, y = extended_gcd(a, b)

            assert extended_result == g
            assert a * x + b * y == g

    def test_prime_generation_methods_are_consistent(self):
        for limit in [2, 10, 50, 100, 250, 500]:
            assert generate_primes(limit) == sieve_of_eratosthenes(limit)

    def test_fast_power_and_modular_power_are_consistent(self):
        test_cases = [
            (2, 10, 1000),
            (3, 7, 17),
            (5, 8, 13),
            (-2, 9, 19),
        ]

        for base, exponent, modulus in test_cases:
            expected = fast_power(base, exponent) % modulus

            assert modular_power(
                base,
                exponent,
                modulus,
            ) == expected

    def test_factorial_related_to_pascal(self):
        triangle = pascal_triangle(8)

        for n in range(1, 8):
            for k in range(n + 1):
                expected = factorial(n) // (
                    factorial(k) * factorial(n - k)
                )

                assert triangle[n][k] == expected

    def test_fibonacci_values_are_integers(self):
        values = [fibonacci(n) for n in range(20)]

        assert all(isinstance(value, int) for value in values)

    def test_prime_count_for_100(self):
        primes = sieve_of_eratosthenes(100)

        assert len(primes) == 25
        assert primes[-1] == 97

    def test_pascal_contains_expected_binomial_coefficients(self):
        triangle = pascal_triangle(7)

        assert triangle[6] == [
            math.comb(6, k)
            for k in range(7)
        ]

