from core.ai.ai_explainer import ai_explain


def main():

    print("=" * 60)
    print("MATHLAB AI — TEST OPENAI")
    print("=" * 60)

    print()
    print("Test : explication d'une dérivée")
    print()

    try:

        explanation = ai_explain(
            operation="derivative",
            expression="x^2 + 3*x",
            result="2*x + 3",
            steps=[
                "On dérive x^2.",
                "La dérivée de x^2 est 2*x.",
                "On dérive 3*x.",
                "La dérivée de 3*x est 3.",
                "On additionne les résultats.",
            ],
            level="lycée",
        )

        print("Réponse de l'IA :")
        print("-" * 60)
        print(explanation)
        print("-" * 60)

        print()
        print("✅ Test OpenAI réussi.")

    except Exception as error:

        print()
        print("❌ Erreur OpenAI :")
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()