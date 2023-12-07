import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import shapiro
from scipy.stats import levene


def main():
    # Adatok beolvasása CSV-fájlból
    df = pd.read_csv("bead11.2.csv")

    # A normalitás tesztje minden csoportra
    for group in ["Csoport 1", "Csoport 2", "Csoport 3", "Csoport 4"]:
        stat, p_value = shapiro(df[group])

        print(f"{group}: Statisztika = {stat}, p-érték = {p_value}")

        alpha = 0.05
        if p_value > alpha:
            print(f"A {group} adatsor normális eloszlású.\n")
        else:
            print(f"A {group} adatsor nem normális eloszlású.\n")

    # Levene teszt a homogenitásra minden csoport között
    stat, p_value = levene(
        df["Csoport 1"], df["Csoport 2"], df["Csoport 3"], df["Csoport 4"]
    )

    print(f"Levene teszt statisztika: {stat}, p-érték: {p_value}")

    alpha = 0.05
    if p_value > alpha:
        print(
            "Nincs szignifikáns különbség a csoportok közötti varianciákban (homogenitás)."
        )
    else:
        print(
            "Van szignifikáns különbség a csoportok közötti varianciákban (homogenitás hiánya)."
        )

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
