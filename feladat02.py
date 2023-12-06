import pandas as pd
from scipy.stats import f_oneway


def main():
    # Adatok beolvasása CSV-fájlból
    df = pd.read_csv("bead11.2.csv")

    # ANOVA teszt végrehajtása
    stat, p_value = f_oneway(
        df["Csoport 1"], df["Csoport 2"], df["Csoport 3"], df["Csoport 4"]
    )

    # Kiértékelés
    alpha = 0.05

    print(f"Statisztika: {stat}, p-érték: {p_value}")

    if p_value < alpha:
        print(
            "Van szignifikáns különbség a csoportok között a mosolygós emojik használatában."
        )
    else:
        print(
            "Nincs szignifikáns különbség a csoportok között a mosolygós emojik használatában."
        )


main()
