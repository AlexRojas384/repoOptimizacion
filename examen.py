import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Estimación de Integral por Montecarlo")

# --- Inputs en la barra lateral ---
st.sidebar.header("Parámetros de la simulación")
a = st.sidebar.number_input("Límite inferior a", value=-6.0)
b = st.sidebar.number_input("Límite superior b", value=6.0)
n = st.sidebar.number_input("Número de muestras n", min_value=1, value=50, step=1)
opcion = st.sidebar.radio("Seleccione función f(x):", ("Opción a", "Opción b"))
if opcion == "Opción a":
    st.markdown("Función seleccionada:")
    st.latex(r"f(x) = \frac{1}{e^x + e^{-x}}")
else:
    st.markdown("Función seleccionada:")
    st.latex(r"f(x) = \frac{2}{e^x + e^{-x}}")
opcion = 1 if opcion == "Opción a" else 2

# --- Funciones ---
def f(x, opcion=1):
    if opcion == 1:
        return 1 / (np.exp(x) + np.exp(-x))
    else:
        return 2 / (np.exp(x) + np.exp(-x))

def muestra(a, b, n):
    return np.random.uniform(a, b, n)

def metodo_montecarlo(a, b, n, opcion=1):
    x = muestra(a, b, n)
    y = f(x, opcion)
    integral = (b - a) / n * np.sum(y)
    return x, y, integral

# --- Botón para ejecutar ---
if st.sidebar.button("Ejecutar Montecarlo"):
    x, y, integral = metodo_montecarlo(a, b, n, opcion)
    areas = (b - a) / n * y

    # Tabla de resultados
    tabla = pd.DataFrame({
        "x_i": np.round(x, 3),
        "f(x_i)": np.round(y, 3),
        "Área parcial": np.round(areas, 3)
    })
    st.subheader("Tabla de resultados")
    st.dataframe(tabla)

    # Integral estimada grande
    st.subheader("Integral estimada")
    st.markdown(f"<h1 style='color:blue'>{round(integral, 4)}</h1>", unsafe_allow_html=True)

    # Gráfica
    st.subheader("Gráfica de la función y muestras Montecarlo")
    fig, ax = plt.subplots(figsize=(10,5))
    X = np.linspace(a, b, 1000)
    ax.plot(X, f(X, opcion), label="f(x)")
    ax.scatter(x, y, color="red", label="Muestras Montecarlo")
    for xi, yi in zip(x, y):
        ax.bar(xi, yi, width=(b-a)/n, alpha=0.2, color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    st.pyplot(fig)
