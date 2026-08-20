import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import sys

# CONFIGURACIÓN
TICKERS = {
    "SPY": "S&P 500 (proxy)",
    "QQQ": "Nasdaq 100 / NAS100 (proxy)",
}
CONTRACT_MULTIPLIER = 100
RISK_FREE_RATE = 0.045
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "gex_historico.csv")

# Ventana horaria válida en hora de Nueva York (mercado cierra 16:00 ET)
HORA_MIN_NY = 16
HORA_MAX_NY = 17  # tolerancia de 1 hora para que el workflow tenga margen


# 0. VERIFICAR QUE ES LA EJECUCIÓN CORRECTA (por el tema de DST)
def es_horario_valido():
    # Si se disparó manualmente (botón "Run workflow"), siempre corre,
    # sin importar la hora -> útil para pruebas.
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        print("Ejecución manual (workflow_dispatch): se salta el chequeo de horario.")
        return True

    ahora_ny = datetime.now(ZoneInfo("America/New_York"))
    if ahora_ny.weekday() >= 5:  # sábado=5, domingo=6
        print(f"Hoy es fin de semana en NY ({ahora_ny}), mercado cerrado. Saliendo.")
        return False
    if not (HORA_MIN_NY <= ahora_ny.hour < HORA_MAX_NY):
        print(f"Hora actual en NY: {ahora_ny.strftime('%H:%M')} - fuera de la "
              f"ventana válida ({HORA_MIN_NY}:00-{HORA_MAX_NY}:00). "
              f"Esta debe ser la ejecución 'duplicada' por el cambio de "
              f"horario de verano. Saliendo sin guardar nada.")
        return False
    return True


# 1. FÓRMULA DE GAMMA (Black-Scholes)
def calcular_gamma_bs(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


# 2. DESCARGA DE LA CADENA DE OPCIONES Y CÁLCULO DE GEX
def calcular_gex_ticker(ticker_symbol, max_vencimientos=6):
    tk = yf.Ticker(ticker_symbol)
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"No se pudo obtener precio spot de {ticker_symbol}")
    spot = hist["Close"].iloc[-1]

    vencimientos_todos = tk.options[:max_vencimientos + 1]
    hoy_fecha = datetime.now(ZoneInfo("America/New_York")).date()
    # Se excluyen los vencimientos 0DTE (mismo día de la recolección):
    # Black-Scholes no está definido en T=0, y forzar T a un valor mínimo
    # arbitrario introduciría un sesgo no justificado. Esta exclusión se
    # declara como limitación explícita en la metodología de la tesis.
    vencimientos = [
        v for v in vencimientos_todos
        if datetime.strptime(v, "%Y-%m-%d").date() > hoy_fecha
    ][:max_vencimientos]

    if not vencimientos:
        raise ValueError(f"No hay vencimientos disponibles para {ticker_symbol}")

    hoy = datetime.now()
    gex_total = 0.0
    filas_detalle = []

    # ------------------------------------------------------------------
    # Paso previo: calcular la volatilidad implícita mediana del día,
    # usando solo contratos con IV válida (>0). Sirve de respaldo para
    # contratos poco líquidos donde Yahoo Finance reporta IV=0 (dato
    # obsoleto/no calculado), en vez de perder su contribución al GEX
    # pese a tener open interest real.
    # ------------------------------------------------------------------
    ivs_validas = []
    for exp in vencimientos:
        chain = tk.option_chain(exp)
        for df in (chain.calls, chain.puts):
            ivs_validas.extend(df.loc[df["impliedVolatility"] > 0, "impliedVolatility"].tolist())
    iv_mediana = float(np.median(ivs_validas)) if ivs_validas else 0.20  # 0.20 como último recurso

    for exp in vencimientos:
        chain = tk.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        T = max((datetime.strptime(exp, "%Y-%m-%d") - hoy).days, 0) / 365.0

        for _, row in calls.iterrows():
            sigma = row["impliedVolatility"] if row["impliedVolatility"] > 0 else iv_mediana
            gamma = calcular_gamma_bs(spot, row["strike"], T, RISK_FREE_RATE, sigma)
            oi = row["openInterest"] if not np.isnan(row["openInterest"]) else 0
            contrib = gamma * oi * CONTRACT_MULTIPLIER * (spot ** 2) * 0.01
            gex_total += contrib
            filas_detalle.append({"tipo": "call", "strike": row["strike"],
                                   "vencimiento": exp, "oi": oi,
                                   "gamma": gamma, "gex_contrib": contrib})

        for _, row in puts.iterrows():
            sigma = row["impliedVolatility"] if row["impliedVolatility"] > 0 else iv_mediana
            gamma = calcular_gamma_bs(spot, row["strike"], T, RISK_FREE_RATE, sigma)
            oi = row["openInterest"] if not np.isnan(row["openInterest"]) else 0
            contrib = -gamma * oi * CONTRACT_MULTIPLIER * (spot ** 2) * 0.01
            gex_total += contrib
            filas_detalle.append({"tipo": "put", "strike": row["strike"],
                                   "vencimiento": exp, "oi": oi,
                                   "gamma": gamma, "gex_contrib": contrib})

    detalle_df = pd.DataFrame(filas_detalle)
    return spot, gex_total, detalle_df


# 3. GUARDAR SNAPSHOT DIARIO
def main():
    if not es_horario_valido():
        sys.exit(0)  # sale sin error para que el workflow no se marque como fallido

    os.makedirs(DATA_DIR, exist_ok=True)

    fecha = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    hora = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S")

    # evita duplicar si por algún motivo ya se guardó hoy
    if os.path.exists(OUTPUT_FILE):
        historico_actual = pd.read_csv(OUTPUT_FILE)
        if fecha in historico_actual["fecha"].astype(str).values:
            print(f"Ya existe un snapshot para {fecha}. Saliendo sin duplicar.")
            sys.exit(0)

    filas = []
    for ticker, nombre in TICKERS.items():
        try:
            spot, gex, detalle = calcular_gex_ticker(ticker)
            filas.append({
                "fecha": fecha, "hora": hora, "ticker": ticker,
                "nombre": nombre, "spot": round(spot, 2), "gex_neto": gex,
            })
            detalle_path = os.path.join(DATA_DIR, f"detalle_{ticker}_{fecha}.csv")
            detalle.to_csv(detalle_path, index=False)
            print(f"[OK] {ticker}: spot={spot:.2f}  GEX neto={gex:,.0f}")
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    nuevo_df = pd.DataFrame(filas)
    if nuevo_df.empty:
        print("No se obtuvo ningún dato válido hoy.")
        sys.exit(0)

    if os.path.exists(OUTPUT_FILE):
        historico = pd.read_csv(OUTPUT_FILE)
        historico = pd.concat([historico, nuevo_df], ignore_index=True)
    else:
        historico = nuevo_df

    historico.to_csv(OUTPUT_FILE, index=False)
    print(f"\nGuardado en {OUTPUT_FILE}. Total de snapshots acumulados: {len(historico)}")


if __name__ == "__main__":
    main()
