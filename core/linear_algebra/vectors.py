from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


# ============================================================
# TYPES
# ============================================================

Number = int | float


# ============================================================
# VECTOR
# ============================================================

@dataclass(frozen=True)
class Vector:
    """
    Représente un vecteur mathématique de dimension finie.
    """

    values: tuple[Number, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(
                "Un vecteur doit contenir au moins une composante."
            )

    @property
    def dimension(self) -> int:
        """
        Retourne la dimension du vecteur.
        """

        return len(self.values)

    def __add__(self, other: Vector) -> Vector:
        """
        Additionne deux vecteurs.
        """

        _check_same_dimension(self, other)

        return Vector(
            tuple(
                a + b
                for a, b in zip(
                    self.values,
                    other.values,
                )
            )
        )

    def __sub__(self, other: Vector) -> Vector:
        """
        Soustrait deux vecteurs.
        """

        _check_same_dimension(self, other)

        return Vector(
            tuple(
                a - b
                for a, b in zip(
                    self.values,
                    other.values,
                )
            )
        )

    def __mul__(self, scalar: Number) -> Vector:
        """
        Multiplie le vecteur par un scalaire.
        """

        if not isinstance(scalar, (int, float)):
            raise TypeError(
                "Le scalaire doit être un nombre."
            )

        return Vector(
            tuple(
                scalar * value
                for value in self.values
            )
        )

    def __rmul__(self, scalar: Number) -> Vector:
        """
        Permet également scalar * vector.
        """

        return self.__mul__(scalar)

    def dot(self, other: Vector) -> Number:
        """
        Calcule le produit scalaire.
        """

        _check_same_dimension(self, other)

        return sum(
            a * b
            for a, b in zip(
                self.values,
                other.values,
            )
        )

    def norm(self) -> float:
        """
        Calcule la norme euclidienne du vecteur.
        """

        return sqrt(
            sum(
                value**2
                for value in self.values
            )
        )

    def distance_to(self, other: Vector) -> float:
        """
        Calcule la distance euclidienne entre deux vecteurs.
        """

        return (self - other).norm()

    def to_tuple(self) -> tuple[Number, ...]:
        """
        Retourne les composantes sous forme de tuple.
        """

        return self.values


# ============================================================
# UTILITAIRES
# ============================================================

def _check_same_dimension(
    first: Vector,
    second: Vector,
) -> None:
    """
    Vérifie que deux vecteurs ont la même dimension.
    """

    if first.dimension != second.dimension:
        raise ValueError(
            "Les vecteurs doivent avoir la même dimension."
        )


def create_vector(
    values: Iterable[Number],
) -> Vector:
    """
    Crée un vecteur à partir d'une collection de nombres.
    """

    values_tuple = tuple(values)

    if not values_tuple:
        raise ValueError(
            "Un vecteur doit contenir au moins une composante."
        )

    if not all(
        isinstance(value, (int, float))
        for value in values_tuple
    ):
        raise TypeError(
            "Toutes les composantes doivent être numériques."
        )

    return Vector(values_tuple)


# ============================================================
# OPERATIONS PUBLIQUES
# ============================================================

def add_vectors(
    first: Vector,
    second: Vector,
) -> Vector:
    """
    Additionne deux vecteurs.
    """

    return first + second


def subtract_vectors(
    first: Vector,
    second: Vector,
) -> Vector:
    """
    Soustrait deux vecteurs.
    """

    return first - second


def scalar_multiply(
    vector: Vector,
    scalar: Number,
) -> Vector:
    """
    Multiplie un vecteur par un scalaire.
    """

    return vector * scalar


def dot_product(
    first: Vector,
    second: Vector,
) -> Number:
    """
    Calcule le produit scalaire de deux vecteurs.
    """

    return first.dot(second)


def vector_norm(
    vector: Vector,
) -> float:
    """
    Calcule la norme euclidienne d'un vecteur.
    """

    return vector.norm()


def vector_distance(
    first: Vector,
    second: Vector,
) -> float:
    """
    Calcule la distance entre deux vecteurs.
    """

    return first.distance_to(second)