import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro


def main():
    """The main function of the program."""
    # Step 1: Read the dataset from the CSV file
    data = pd.read_csv("bead11.1.csv")

    # Step 2: Extract the independent and dependent variables
    X = data[["Üzenet hossza", "Mosolygós emojik száma"]]
    X = sm.add_constant(X)  # Állandó hozzáadása a modellhez
    y = data["Boldogságszint"]

    model = sm.OLS(y, X).fit()

    # Regressziós modell paramétereinek kiírása
    print(model.summary())

    # 2. feladat: Többszörös determinációs együttható kiszámítása
    rsq = model.rsquared
    print("Többszörös determinációs együttható:", rsq)

    # 3. feladat: Regressziós modell megbízhatóságának tesztelése
    residuals = model.resid
    shapiro_test_stat, shapiro_test_p_value = shapiro(residuals)
    print("Shapiro-Wilk teszt statisztika:", shapiro_test_stat)
    print("P-érték:", shapiro_test_p_value)

    # 4. feladat: Paraméterek intervallumbecslése
    conf_int = model.conf_int()
    print("Paraméterek 95%-os megbízhatósági intervallumai:")
    print(conf_int)

    # 5. feladat: Előrejelzés és intervallumbecslés
    uj_uzenet = pd.DataFrame(
        {"const": 1, "Üzenet hossza": [130], "Mosolygós emojik száma": [3]}
    )
    predict_value = model.get_prediction(uj_uzenet).summary_frame()
    print("Előrejelzés:", predict_value["mean"][0])
    print(
        "95%-os megbízhatóságú intervallum:",
        predict_value["mean_ci_lower"][0],
        "-",
        predict_value["mean_ci_upper"][0],
    )


main()
