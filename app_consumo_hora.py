import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
)

st.set_page_config(page_title="ML: Consumo Total vs Hora (InfluxDB)", layout="wide")

# ---------------------------------------------------------------
# Datos sintéticos de respaldo (patrón realista: dos picos de consumo)
# ---------------------------------------------------------------

def generar_datos_sinteticos(n=1000, semilla=42):
    rng = np.random.default_rng(semilla)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    hora = timestamps.hour
    patron = (
        1.5
        + 2.0 * np.exp(-((hora - 8) ** 2) / 8)
        + 3.0 * np.exp(-((hora - 20) ** 2) / 10)
    )
    ruido = rng.normal(0, 0.3, n)
    consumo = patron + ruido
    return pd.DataFrame({"timestamp": timestamps, "Consumo_total": consumo})


# ---------------------------------------------------------------
# Conexión a InfluxDB (solo Consumo_total)
# ---------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Consultando InfluxDB...")
def consultar_influx(url, token, org, bucket, rango_dias):
    from influxdb_client import InfluxDBClient

    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()

    flux_query = f'''
    from(bucket: "{bucket}")
      |> range(start: -{rango_dias}d)
      |> filter(fn: (r) => r["_field"] == "Consumo_total")
      |> keep(columns: ["_time", "_value"])
    '''

    tables = query_api.query(flux_query, org=org)
    registros = []
    for table in tables:
        for record in table.records:
            registros.append({"timestamp": record.get_time(), "Consumo_total": record.get_value()})

    client.close()

    df = pd.DataFrame(registros)
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Calculando método del codo...")
def calcular_inercias_codo(consumo_values, k_max=8):
    X = consumo_values.reshape(-1, 1)
    inercias = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inercias.append(km.inertia_)
    return inercias


# ---------------------------------------------------------------
# Sidebar: conexión
# ---------------------------------------------------------------

st.sidebar.header("🔌 Conexión a InfluxDB")
usar_datos_ejemplo = st.sidebar.checkbox("Usar datos de ejemplo (sin conexión)", value=True)

df = None

if not usar_datos_ejemplo:
    influx_url = st.sidebar.text_input("URL", placeholder="https://us-east-1-1.aws.cloud2.influxdata.com/")
    influx_token = st.sidebar.text_input("Token", type="password")
    influx_org = st.sidebar.text_input("Organización")
    influx_bucket = st.sidebar.text_input("Bucket", value="Consumo_elec")
    rango_dias = st.sidebar.slider("Rango de datos (días)", 1, 90, 30)

    if st.sidebar.button("Conectar y consultar"):
        if not (influx_url and influx_token and influx_org and influx_bucket):
            st.sidebar.error("Completa todos los campos de conexión.")
        else:
            try:
                df = consultar_influx(influx_url, influx_token, influx_org, influx_bucket, rango_dias)
                if df.empty:
                    st.sidebar.warning("La consulta no devolvió datos. Revisa el rango o el nombre del bucket.")
                else:
                    st.sidebar.success(f"{len(df)} registros cargados ✅")
            except Exception as e:
                st.sidebar.error(f"Error de conexión: {e}")

    if df is None:
        st.info("Configura la conexión en la barra lateral y presiona **Conectar y consultar**, "
                "o activa 'Usar datos de ejemplo' para explorar la app sin conexión real.")
        st.stop()
else:
    df = generar_datos_sinteticos()
    st.sidebar.info("Mostrando datos sintéticos con dos picos de consumo (mañana y noche).")

st.title("Machine Learning: Consumo Total vs Hora del Día")
st.caption("K-Means · KNN · SVM · Regresión Lineal — todo sobre la relación Consumo_total vs Hora")

# ---------------------------------------------------------------
# Única variable derivada del timestamp
# ---------------------------------------------------------------
# InfluxDB Cloud devuelve los timestamps en UTC. Colombia está en UTC-5 todo el año
# (sin horario de verano), así que convertimos antes de extraer la hora local;
# si no se hiciera esto, todo el patrón de consumo quedaría desplazado 5 horas.

_ts = pd.to_datetime(df["timestamp"])
if _ts.dt.tz is not None:
    _ts = _ts.dt.tz_convert("America/Bogota")
df["hora"] = _ts.dt.hour
umbral = df["Consumo_total"].median()
df["consumo_alto"] = (df["Consumo_total"] > umbral).astype(int)

tab_datos, tab_lineal, tab_kmeans, tab_knn, tab_svm, tab_comparacion = st.tabs(
    ["📊 Datos", "📈 Regresión Lineal", "🟢 K-Means", "🔵 KNN", "🟠 SVM", "⚖️ Comparación"]
)

# ---------------------------------------------------------------
# TAB: DATOS
# ---------------------------------------------------------------
with tab_datos:
    st.subheader("La relación central: Consumo_total vs Hora")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df["hora"], df["Consumo_total"], alpha=0.3, s=15)
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Consumo total")
        ax.set_title("Consumo total vs. Hora del día")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        promedio_por_hora = df.groupby("hora")["Consumo_total"].mean()
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(promedio_por_hora.index, promedio_por_hora.values)
        ax2.set_xlabel("Hora del día")
        ax2.set_ylabel("Consumo total promedio")
        ax2.set_title("Patrón de consumo promedio por hora")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown(f"**Umbral de 'consumo alto'** (mediana): `{umbral:.2f}`")

# ---------------------------------------------------------------
# TAB: REGRESIÓN LINEAL
# ---------------------------------------------------------------
with tab_lineal:
    st.header("Regresión Lineal")
    st.markdown("¿Existe una tendencia lineal simple entre `hora` y `Consumo_total`?")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        test_size_lin = st.slider("Proporción de test", 0.1, 0.5, 0.3, step=0.05, key="test_lin")

    X = df[["hora"]].values
    y = df["Consumo_total"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_lin, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    with col_plot:
        orden = X_test[:, 0].argsort()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(df["hora"], df["Consumo_total"], alpha=0.3, s=15, label="Datos reales")
        ax.plot(X_test[orden], pred[orden], color="red", linewidth=2, label="Regresión Lineal")
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Consumo total")
        ax.set_title("Una línea recta vs. el patrón real")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_score(y_test, pred):.3f}")
    c2.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
    c3.metric("RMSE", f"{mean_squared_error(y_test, pred) ** 0.5:.3f}")

    with st.expander("📘 Conceptos clave"):
        st.markdown(
            """
            - Si el patrón diario tiene más de un pico (ej. mañana y noche), una sola línea
              recta **no puede** capturarlo bien — por eso el R² suele ser bajo aquí.
            - Esto motiva usar modelos más flexibles (K-Means, KNN, SVM) para el resto del análisis.
            """
        )

# ---------------------------------------------------------------
# TAB: K-MEANS
# ---------------------------------------------------------------
with tab_kmeans:
    st.header("K-Means — niveles de consumo (bajo, medio, alto)")
    st.markdown(
        """
        Agrupamos usando **solo `Consumo_total`** (no `hora` + `Consumo_total` juntos).
        Mezclar ambas variables con la misma escala tiende a producir clusters confusos,
        porque los valores atípicos de consumo dominan la distancia y la hora queda como
        una señal casi decorativa. Agrupando solo por consumo, los clusters representan
        **niveles reales** (bajo/medio/alto) — y usamos `hora` después, solo para interpretar.
        """
    )

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        k_elegido = st.slider("Número de clusters (k)", 2, 8, 3)
        mostrar_codo = st.checkbox("Mostrar método del codo", value=True)

    X_km = df[["Consumo_total"]].values

    kmeans = KMeans(n_clusters=k_elegido, random_state=42, n_init=10)
    labels_raw = kmeans.fit_predict(X_km)

    # Reordenar etiquetas para que 0 = consumo más bajo
    orden_clusters = pd.Series(df["Consumo_total"].values).groupby(labels_raw).mean().sort_values().index
    mapa_orden = {viejo: nuevo for nuevo, viejo in enumerate(orden_clusters)}
    df["cluster"] = pd.Series(labels_raw).map(mapa_orden).values

    with col_plot:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        scatter = ax.scatter(df["hora"], df["Consumo_total"], c=df["cluster"], cmap="viridis", s=25, alpha=0.8)
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Consumo total")
        ax.set_title(f"Niveles de consumo (k={k_elegido}) — coloreados por hora para interpretar")
        plt.colorbar(scatter, ax=ax, label="Nivel (0=bajo)")
        st.pyplot(fig)
        plt.close(fig)

    st.metric("Inercia", f"{kmeans.inertia_:.1f}")

    if mostrar_codo:
        inercias = calcular_inercias_codo(df["Consumo_total"].values)
        rango_k = range(1, len(inercias) + 1)
        fig_codo, ax_codo = plt.subplots(figsize=(6.5, 3.2))
        ax_codo.plot(list(rango_k), inercias, marker="o")
        ax_codo.axvline(k_elegido, color="red", linestyle="--", alpha=0.6, label="k elegido")
        ax_codo.set_xlabel("k")
        ax_codo.set_ylabel("Inercia")
        ax_codo.set_title("Método del codo (solo Consumo_total)")
        ax_codo.legend()
        st.pyplot(fig_codo)
        plt.close(fig_codo)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Rangos de consumo por nivel**")
        st.dataframe(
            df.groupby("cluster")["Consumo_total"].agg(["mean", "min", "max", "count"]).round(2),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**¿A qué hora ocurre cada nivel?** (interpretación, no input del modelo)")
        st.dataframe(
            df.groupby("cluster")["hora"].agg(["mean", "min", "max"]).round(1),
            use_container_width=True,
        )

    st.markdown("**Distribución de horas dentro de cada nivel**")
    fig_hist, axes = plt.subplots(1, k_elegido, figsize=(4 * k_elegido, 3), sharey=True)
    if k_elegido == 1:
        axes = [axes]
    for cluster_id, ax in enumerate(axes):
        subset = df[df["cluster"] == cluster_id]
        ax.hist(subset["hora"], bins=24, range=(0, 24), color=plt.cm.viridis(cluster_id / max(k_elegido - 1, 1)))
        ax.set_title(f"Nivel {cluster_id} (n={len(subset)})")
        ax.set_xlabel("Hora")
    axes[0].set_ylabel("Frecuencia")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    with st.expander("📘 Conceptos clave"):
        st.markdown(
            """
            - K-Means agrupa aquí **solo por nivel de consumo** — los rangos de consumo entre
              clusters no se solapan, por construcción.
            - La hora **no es input del modelo**, se usa solo para interpretar después: si el
              histograma de un nivel se concentra en ciertas horas, hay relación hora-consumo;
              si se ve disperso en las 24 horas, el nivel de consumo depende de otros factores.
            - Esto es más honesto que forzar a K-Means a usar hora y consumo juntos, lo cual
              puede producir "franjas horarias" que en realidad no son consistentes.
            """
        )

# ---------------------------------------------------------------
# Función compartida para KNN y SVM: mapa de clasificación por hora
# ---------------------------------------------------------------

def plot_clasificacion_por_hora(modelo, df, umbral, titulo):
    horas_grid = np.arange(0, 24, 0.1).reshape(-1, 1)
    pred_grid = modelo.predict(horas_grid)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(len(horas_grid) - 1):
        color = "#fde0dd" if pred_grid[i] == 0 else "#c6dbef"
        ax.axvspan(horas_grid[i, 0], horas_grid[i + 1, 0], color=color, alpha=0.6, linewidth=0)

    ax.scatter(df["hora"], df["Consumo_total"], c=df["consumo_alto"], cmap="coolwarm",
               edgecolor="k", s=20, alpha=0.8)
    ax.axhline(umbral, color="black", linestyle="--", alpha=0.5, label="Umbral (mediana)")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Consumo total")
    ax.set_title(titulo)
    ax.legend()
    return fig


# ---------------------------------------------------------------
# TAB: KNN
# ---------------------------------------------------------------
with tab_knn:
    st.header("KNN — clasificar consumo alto/bajo según horas vecinas")
    st.markdown("Para predecir si una hora tendrá consumo alto, KNN mira qué pasó en las horas más parecidas.")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        k_vecinos = st.slider("Número de vecinos (k)", 1, 30, 5)
        test_size_knn = st.slider("Proporción de test", 0.1, 0.5, 0.3, step=0.05, key="test_knn")

    X_clas = df[["hora"]].values
    y_clas = df["consumo_alto"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_clas, y_clas, test_size=test_size_knn, random_state=42, stratify=y_clas
    )

    knn = KNeighborsClassifier(n_neighbors=k_vecinos)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, pred_knn)

    knn_full = KNeighborsClassifier(n_neighbors=k_vecinos)
    knn_full.fit(X_clas, y_clas)

    with col_plot:
        fig = plot_clasificacion_por_hora(knn_full, df, umbral, f"KNN (k={k_vecinos}): horas alto (azul) vs bajo (rosa)")
        st.pyplot(fig)
        plt.close(fig)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc_knn:.2%}")
    with c2:
        cm = confusion_matrix(y_test, pred_knn)
        fig_cm, ax_cm = plt.subplots(figsize=(3.2, 3.2))
        ConfusionMatrixDisplay(cm, display_labels=["Bajo", "Alto"]).plot(ax=ax_cm, cmap="Blues", colorbar=False)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with st.expander("📘 Conceptos clave"):
        st.markdown(
            """
            - Con `k` bajo, el mapa de franjas puede verse muy irregular (sensible al ruido).
            - Con `k` alto, las franjas se suavizan, pero pueden perder detalle si son demasiado altas.
            - El fondo de color muestra qué predice el modelo para cada hora — si aparecen dos
              franjas azules separadas, el modelo capturó bien los dos picos de consumo.
            """
        )

# ---------------------------------------------------------------
# TAB: SVM
# ---------------------------------------------------------------
with tab_svm:
    st.header("SVM — mismo problema, fronteras más flexibles")
    st.markdown("Comparación directa contra KNN, usando el mismo par de franjas alto/bajo.")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        C = st.select_slider("C", options=[0.01, 0.1, 1, 10, 100], value=1)
        gamma_valor = st.select_slider("gamma", options=["scale", 0.01, 0.1, 1, 10], value="scale")
        test_size_svm = st.slider("Proporción de test", 0.1, 0.5, 0.3, step=0.05, key="test_svm")

    X_svm = df[["hora"]].values
    y_svm = df["consumo_alto"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_svm, y_svm, test_size=test_size_svm, random_state=42, stratify=y_svm
    )

    modelo_svm = SVC(kernel=kernel, C=C, gamma=gamma_valor)
    modelo_svm.fit(X_train, y_train)
    pred_svm = modelo_svm.predict(X_test)
    acc_svm = accuracy_score(y_test, pred_svm)

    svm_full = SVC(kernel=kernel, C=C, gamma=gamma_valor)
    svm_full.fit(X_svm, y_svm)

    with col_plot:
        fig = plot_clasificacion_por_hora(svm_full, df, umbral, f"SVM ({kernel}): horas alto (azul) vs bajo (rosa)")
        st.pyplot(fig)
        plt.close(fig)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc_svm:.2%}")
    c2.metric("Vectores de soporte", int(modelo_svm.support_vectors_.shape[0]))

    with st.expander("📘 Conceptos clave"):
        st.markdown(
            """
            - Con kernel `linear`, SVM solo puede definir **una** franja de corte — si hay dos
              picos de consumo separados, fallará en capturar ambos correctamente.
            - Con kernel `rbf`, SVM puede definir varias franjas, similar a KNN.
            - Compara el accuracy y la forma del mapa aquí contra la pestaña de KNN.
            """
        )

# ---------------------------------------------------------------
# TAB: COMPARACIÓN
# ---------------------------------------------------------------
with tab_comparacion:
    st.header("Comparación de modelos")
    st.markdown(
        """
        | Algoritmo | Tipo de problema | ¿Usa etiquetas? | Qué responde en este caso |
        |---|---|---|---|
        | Regresión Lineal | Regresión (valor continuo) | Sí | ¿Cuánto será el consumo según la hora? (relación simple) |
        | K-Means | Clustering | No | ¿Qué franjas horarias de consumo existen, sin definirlas de antemano? |
        | KNN | Clasificación | Sí | ¿Esta hora tendrá consumo alto o bajo, según horas parecidas? |
        | SVM | Clasificación | Sí | ¿Esta hora tendrá consumo alto o bajo? (frontera global, más flexible) |
        """
    )
    st.info("Ajusta parámetros en cada pestaña y vuelve aquí para comparar tus propias métricas anotadas.")

st.divider()
st.caption("App educativa · ML sobre Consumo_total vs Hora (InfluxDB) · Regresión Lineal, K-Means, KNN, SVM")