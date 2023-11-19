import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


def main():
    """A program fő függvénye."""
    # Adatok beolvasása
    data = pd.read_csv("bead11.3.csv")
    data["Év-hónap"] = pd.to_datetime(data["Év-hónap"])
    data.set_index("Év-hónap", inplace=True)

    # Idősor diagram
    plt.figure(figsize=(10, 6))
    plt.plot(
        data["Mosolygós emojik használata (darab)"], label="Mosolygós emojik használata"
    )
    plt.title("Mosolygós emojik használata az időben")
    plt.xlabel("Dátum")
    plt.ylabel("Emoji használat (darab)")
    plt.legend()
    plt.show()

    # Autokorrelációk számolása
    lag_acf = acf(data["Mosolygós emojik használata (darab)"], nlags=20)
    lag_pacf = pacf(data["Mosolygós emojik használata (darab)"], nlags=20, method="ols")

    # Modellezés és előrejelzés
    model = SARIMAX(
        data["Mosolygós emojik használata (darab)"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
    )
    results = model.fit(disp=-1)

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
