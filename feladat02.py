import pandas as pd
from scipy.stats import chi2_contingency


def main():
    """A program fő függvénye."""
    # Adatok beolvasása
    data = pd.read_csv("bead11.2.csv")

    osszehasonlitas(data, "Csoport 1", "Csoport 2")
    osszehasonlitas(data, "Csoport 1", "Csoport 3")
    osszehasonlitas(data, "Csoport 1", "Csoport 4")
    osszehasonlitas(data, "Csoport 2", "Csoport 3")
    osszehasonlitas(data, "Csoport 2", "Csoport 4")
    osszehasonlitas(data, "Csoport 3", "Csoport 4")

    # Chi-négyzet teszt


def osszehasonlitas(data, param1, param2):
    contingency_table = pd.crosstab(data[param1], data[param2], margins=False)
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Eredmény kiírása
    print(f"Chi-négyzet érték: {chi2}")
    print(f"P-érték: {p}")

    # Szignifikancia ellenőrzése
    alpha = 0.05
    if p < alpha:
        print(
            "Van szignifikáns különbség a csoportok között a mosolygós emojik használatában.("
            + param1
            + " "
            + param2
            + ")"
        )
    else:
        print(
            "Nincs szignifikáns különbség a csoportok között a mosolygós emojik használatában.("
            + param1
            + " "
            + param2
            + ")"
        )


main()
