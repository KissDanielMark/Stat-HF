import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def main():
    """A program fő függvénye."""
    # CSV fájl beolvasása
    file_path = "bead11.3.csv"
    data = pd.read_csv(file_path)
    data["Év-hónap"] = pd.to_datetime(data["Év-hónap"])
    data.set_index("Év-hónap", inplace=True)

    # 3/a Idősor diagram-------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(data["Mosolygós emojik használata (darab)"], marker="o")
    plt.title("Mosolygós emojik használata az évek során")
    plt.xlabel("Év-hónap")
    plt.ylabel("Darabszám")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Tapasztalati autokorrelációs függvény (ACF)
    plt.figure(figsize=(10, 6))
    plot_acf(data["Mosolygós emojik használata (darab)"], lags=30)
    plt.title("Tapasztalati autokorrelációs függvény (ACF)")
    plt.xlabel("Lag")
    plt.ylabel("Autokorreláció")
    plt.show()

    # Parciális autokorrelációs függvény (PACF)
    plt.figure(figsize=(10, 6))
    plot_pacf(data["Mosolygós emojik használata (darab)"], lags=12)  # Módosított érték
    plt.title("Parciális autokorrelációs függvény (PACF)")
    plt.xlabel("Lag")
    plt.ylabel("Parciális autokorreláció")
    plt.show()

    # 3/b --------------------------------------------------------------------------------------------------------------

    # Adatok megjelenítése
    """plt.figure(figsize=(12, 6))
    plt.plot(data["Mosolygós emojik használata (darab)"], label="Eredeti adatok")
    plt.title("Mosolygós emojik használata az idő függvényében")
    plt.legend()
    plt.show()"""

    # Holt-Winters Exponential Smoothing model
    model = ExponentialSmoothing(
        data["Mosolygós emojik használata (darab)"], seasonal="add", seasonal_periods=12
    )
    fit = model.fit()

    # Illeszkedés vizsgálata
    plt.figure(figsize=(12, 6))
    plt.plot(data["Mosolygós emojik használata (darab)"], label="Eredeti adatok")
    plt.plot(fit.fittedvalues, color="red", label="Illeszkedett model")
    plt.title("Holt-Winters Exponential Smoothing illesztése")
    plt.legend()
    plt.show()

    result = sm.tsa.seasonal_decompose(
        data["Mosolygós emojik használata (darab)"], model="additive", period=12
    )
    residual = result.resid.dropna()

    # ARIMA modell illesztése a reziduálisokra
    order = (1, 1, 1)  # Példaérték, a modellrend megválasztása
    model = sm.tsa.ARIMA(residual, order=order)
    results = model.fit()
    # Illesztés eredményeinek megjelenítése
    print(results.summary())
    # Illesztett értékek kiszámítása
    fitted_values = results.fittedvalues
    # Illesztett és eredeti értékek összehasonlítása
    plt.figure(figsize=(10, 6))
    plt.plot(residual, label="Eredeti reziduálisok")
    plt.plot(fitted_values, color="red", label="Illesztett reziduálisok")
    plt.legend()
    plt.title("ARIMA Modell illesztése a reziduálisokra")
    plt.show()

    # 3/c --------------------------------------------------------------------------------------------------------------
    # Jövőbeli hónapokra történő előrejelzés
    future_months = 6  # Válaszd meg, hány hónapot szeretnél előrejelzést készíteni
    forecast = fit.forecast(steps=future_months)

    # Eredeti és előrejelzett adatok megjelenítése
    plt.figure(figsize=(12, 6))
    plt.plot(data["Mosolygós emojik használata (darab)"], label="Eredeti adatok")
    plt.plot(fit.fittedvalues, color="red", label="Illeszkedett model")
    plt.plot(
        fit.forecast(steps=future_months),
        linestyle="dashed",
        color="green",
        label=f"{future_months} hónap előrejelzés",
    )
    plt.title(
        "Holt-Winters Exponential Smoothing model - Előrejelzés a mosolygós emojik használatára"
    )
    plt.legend()
    plt.show()

    # Előrejelzött értékek kinyomtatása
    print("Előrejelzés a következő hónapokra:")
    print(forecast)


main()
