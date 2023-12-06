import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def main():
    """A program fő függvénye."""
    # CSV fájl beolvasása
    file_path = "bead11.3.csv"
    data = pd.read_csv(file_path)
    data["Év-hónap"] = pd.to_datetime(data["Év-hónap"])
    data.set_index("Év-hónap", inplace=True)

    # Idősor diagram
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

    # Szezonális dekompozíció
    result = sm.tsa.seasonal_decompose(
        data["Mosolygós emojik használata (darab)"], model="additive", period=12
    )
    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    residual = result.resid.dropna()

    # Ábrázolás
    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(data["Mosolygós emojik használata (darab)"], label="Eredeti")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(trend, label="Trend")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label="Szezonális")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(residual, label="Reziduális")
    plt.legend()

    plt.show()

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

    # Előrejelzés a következő hónapokra
    forecast_steps = 12
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(
        data.index[-1], periods=forecast_steps + 1, freq="M"
    )[1:]

    # Előrejelzés megjelenítése
    plt.figure(figsize=(10, 6))
    plt.plot(data["Mosolygós emojik használata (darab)"], label="Tényleges adatok")
    plt.plot(forecast_index, forecast.predicted_mean, color="red", label="Előrejelzés")
    plt.title("Mosolygós emojik használata és előrejelzés")
    plt.xlabel("Dátum")
    plt.ylabel("Emoji használat (darab)")
    plt.legend()
    plt.show()


main()
