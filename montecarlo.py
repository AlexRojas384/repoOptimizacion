import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# -----------------------------
# Función Monte Carlo
# -----------------------------
def montecarlo_satelite(n=10000, num_variables=5, criterio=4, generar=True, a=1000, b=5000, datos_excel=None):
    tiempos = []
    experimentos = []

    for _ in range(n):
        # Generar datos aleatorios
        if generar:
            paneles = np.random.uniform(a, b, size=num_variables)
        else:
            if datos_excel is None:
                raise ValueError("Se eligió 'Entradas' pero no se proporcionó archivo Excel")
            # Elegir una fila aleatoria de los datos del Excel
            fila = datos_excel[np.random.randint(0, datos_excel.shape[0])]
            paneles = fila[:num_variables]  # solo tomar las primeras columnas como variables

        # Aplicar criterio de sorteo
        paneles_ordenados = np.sort(paneles)
        t_falla = paneles_ordenados[criterio - 1]

        tiempos.append(t_falla)
        experimentos.append(list(paneles) + [t_falla])

    tiempos = np.array(tiempos)
    x_hat = np.mean(tiempos)
    sigma = np.std(tiempos, ddof=1)

    # Crear DataFrame de experimentos
    cols = [f'Panel {i+1}' for i in range(num_variables)] + ['Tiempo falla (criterio)']
    df_experimentos = pd.DataFrame(experimentos, columns=cols)

    return x_hat, sigma, tiempos, df_experimentos

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Simulación Monte Carlo - Satélite")
st.sidebar.header("Parámetros de simulación")

num_variables = st.sidebar.number_input("Número de Variables Aleatorias Independientes", min_value=1, value=5)
n = st.sidebar.number_input("Tamaño de la muestra (n)", min_value=1, value=10000)
criterio = st.sidebar.number_input("Criterio de sorteo (k-ésimo menor)", min_value=1, value=4)
modo = st.sidebar.radio("Origen de los datos", options=["Generar Datos", "Entradas Excel"])

if modo == "Generar Datos":
    generar = True
    a = st.sidebar.number_input("Valor mínimo", value=1000)
    b = st.sidebar.number_input("Valor máximo", value=5000)
    datos_excel = None
else:
    generar = False
    archivo = st.sidebar.file_uploader("Subir archivo Excel", type=["xlsx"])
    datos_excel = None
    if archivo:
        df_excel = pd.read_excel(archivo)
        # Tomar solo las columnas de paneles (las primeras `num_variables`)
        datos_excel = df_excel.iloc[:, :num_variables].values

if st.button("Ejecutar simulación"):
    if not generar and datos_excel is None:
        st.error("Debe subir un archivo Excel para ejecutar la simulación")
    else:
        media, desv, tiempos, df_experimentos = montecarlo_satelite(
            n=n,
            num_variables=num_variables,
            criterio=criterio,
            generar=generar,
            a=a if generar else None,
            b=b if generar else None,
            datos_excel=datos_excel
        )

        st.success("Simulación completada 🎉")
        st.write(f"**Media estimada (x̂):** {media:.2f}")
        st.write(f"**Desviación estándar (σ):** {desv:.2f}")

        # Mostrar tabla de experimentos
        st.subheader("Tabla de Experimentos")
        st.dataframe(df_experimentos)

        # -----------------------------
        # Gráficos
        # -----------------------------
        st.subheader("Gráficos")

        fig, axs = plt.subplots(1,2, figsize=(15,5))

        # Histograma
        axs[0].hist(tiempos, bins=30, edgecolor='black')
        axs[0].set_title("Distribución del tiempo de falla")
        axs[0].set_xlabel("Horas hasta falla")
        axs[0].set_ylabel("Frecuencia")

        # Convergencia de la media
        medias_parciales = np.cumsum(tiempos) / np.arange(1, len(tiempos)+1)
        axs[1].plot(medias_parciales)
        axs[1].axhline(media, color='red', linestyle='--')
        axs[1].set_title("Convergencia de la media estimada")
        axs[1].set_xlabel("Número de simulaciones")
        axs[1].set_ylabel("Media acumulada")

        plt.tight_layout()
        st.pyplot(fig)

        # Botón para descargar tabla
        towrite = io.BytesIO()
        df_experimentos.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(
            label="Descargar tabla de experimentos",
            data=towrite,
            file_name="experimentos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
