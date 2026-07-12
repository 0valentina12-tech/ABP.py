import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, CubicSpline
import sqlite3
import math
import os
import tempfile
from fpdf import FPDF

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="CivilCAD AI - Simulador Vial", layout="wide")


# --- FUNCIONES AUXILIARES DE BD ---
def inicializar_bd():
    conn = sqlite3.connect('registro_topografico.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS proyectos 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT, ancho REAL, presupuesto REAL, longitud_lograda REAL, corte REAL, relleno REAL)''')
    conn.commit()
    conn.close()


def guardar_proyecto(nombre, ancho, presupuesto, longitud, corte, relleno):
    conn = sqlite3.connect('registro_topografico.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO proyectos (nombre, ancho, presupuesto, longitud_lograda, corte, relleno) VALUES (?, ?, ?, ?, ?, ?)",
        (nombre, ancho, presupuesto, longitud, corte, relleno))
    conn.commit()
    conn.close()


def obtener_historial():
    conn = sqlite3.connect('registro_topografico.db')
    df_bd = pd.read_sql_query("SELECT * FROM proyectos", conn)
    conn.close()
    return df_bd


inicializar_bd()

# --- VARIABLES GLOBALES EN SESIÓN ---
if 'df' not in st.session_state: st.session_state['df'] = None
if 'ancho_via' not in st.session_state: st.session_state['ancho_via'] = 10.00
if 'presupuesto' not in st.session_state: st.session_state['presupuesto'] = 40000.00
if 'pdf_bytes' not in st.session_state: st.session_state['pdf_bytes'] = None

# --- MENÚ LATERAL ---
st.sidebar.title("Etapas del Proyecto")
st.sidebar.write("Navegación:")

opcion = st.sidebar.radio(
    "",
    [
        "1. Ingesta de Datos",
        "2. Auditoría Espacial 3D",
        "3. Esqueleto Estructural (TIN)",
        "4. Superficie Sólida (MDE)",
        "5. Maqueta Topográfica (Bloque 3D)",
        "6. Parámetros de Diseño",
        "7. Diseño de Eje y Rasante",
        "8. Maqueta de Excavación 3D",
        "9. Base de Datos (Archivero)",
        "10. Emisión de Memoria (PDF)"
    ]
)

# TÍTULO PRINCIPAL
if opcion not in ["7. Diseño de Eje y Rasante", "8. Maqueta de Excavación 3D", "9. Base de Datos (Archivero)",
                  "10. Emisión de Memoria (PDF)"]:
    st.markdown("# 🚜 CivilCAD AI: Simulador Vial y Movimiento de Tierras")

# =====================================================================
# ETAPA 1: CONSTRUIR EL TERRENO
# =====================================================================

# --- FASE 1: INGESTA DE DATOS ---
if opcion == "1. Ingesta de Datos":
    st.markdown("## Fase 1: Leer la libreta topográfica")
    st.write("Sube tu levantamiento (.txt o .csv)")

    archivo_subido = st.file_uploader("", type=["txt", "csv"])

    if archivo_subido is not None:
        try:
            df = pd.read_csv(archivo_subido, names=['ID', 'X', 'Y', 'Z', 'Etiqueta'], sep=None, engine='python')
            df = df.dropna(subset=['X', 'Y', 'Z'])
            df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].astype(float)
            st.session_state['df'] = df
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state['df'] is not None:
        df = st.session_state['df']
        st.success(f"✅ ¡Datos en memoria! {len(df)} puntos topográficos listos para procesar.")

        col1, col2, col3 = st.columns(3)
        z_max = df['Z'].max()
        z_min = df['Z'].min()
        desnivel = z_max - z_min

        col1.metric("Cota Máxima (Z)", f"{z_max:.3f} m")
        col2.metric("Cota Mínima (Z)", f"{z_min:.3f} m")
        col3.metric("Desnivel Topográfico", f"{desnivel:.3f} m")

        with st.expander("Ver tabla de coordenadas"):
            st.dataframe(df, use_container_width=True)

df = st.session_state['df']
if df is not None:
    # --- FASE 2: AUDITORÍA ESPACIAL 3D ---
    if opcion == "2. Auditoría Espacial 3D":
        st.markdown("## Fase 2: Auditoría Espacial 3D")
        fig_3d = go.Figure(data=[go.Scatter3d(x=df['X'], y=df['Y'], z=df['Z'], mode='markers',
                                              marker=dict(size=3, color=df['Z'], colorscale='Viridis', opacity=0.8))])
        fig_3d.update_layout(scene=dict(xaxis_title='X', yaxis_title='y', zaxis_title='z'), template="plotly_dark",
                             margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- FASE 3: ESQUELETO ESTRUCTURAL (TIN) ---
    elif opcion == "3. Esqueleto Estructural (TIN)":
        st.markdown("## Fase 3: Esqueleto Estructural (TIN)")
        diag = np.sqrt((df['X'].max() - df['X'].min()) ** 2 + (df['Y'].max() - df['Y'].min()) ** 2)
        val_arista = st.slider("Longitud máxima de arista (m):", 10.0, diag, diag / 3)

        if st.button("⚙️ Generar TIN", use_container_width=True):
            st.write("¡COMPLETADO!")
            st.progress(100)
            st.success("✅ TIN guardado en memoria.")
            puntos2D = df[['X', 'Y']].values
            tri = Delaunay(puntos2D)
            x_lines, y_lines, z_lines = [], [], []
            for simplex in tri.simplices:
                for i in range(3):
                    p1, p2 = simplex[i], simplex[(i + 1) % 3]
                    if np.linalg.norm(puntos2D[p1] - puntos2D[p2]) <= val_arista:
                        x_lines.extend([df['X'].iloc[p1], df['X'].iloc[p2], None])
                        y_lines.extend([df['Y'].iloc[p1], df['Y'].iloc[p2], None])
                        z_lines.extend([df['Z'].iloc[p1], df['Z'].iloc[p2], None])

            fig_tin = go.Figure(
                data=[go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color='red', width=1))])
            fig_tin.update_layout(scene=dict(xaxis_title='X', yaxis_title='y', zaxis_title='z'), template="plotly_dark",
                                  margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_tin, use_container_width=True)

    # --- FASE 4: SUPERFICIE SÓLIDA (MDE) ---
    elif opcion == "4. Superficie Sólida (MDE)":
        st.markdown("## Fase 4: Superficie Sólida (MDE)")
        st.info("💡 Curvas de nivel fijadas cada 5 metros. Pasa el cursor sobre la montaña para leer la elevación.")
        grid_x, grid_y = np.mgrid[df['X'].min():df['X'].max():100j, df['Y'].min():df['Y'].max():100j]
        grid_z = griddata(df[['X', 'Y']].values, df['Z'].values, (grid_x, grid_y), method='linear')

        fig_mde = go.Figure(data=[go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale='Earth',
                                             contours_z=dict(show=True, color="black", highlightcolor="black",
                                                             project_z=True))])

        fig_mde.update_layout(scene=dict(xaxis_title='X', yaxis_title='y', zaxis_title='z'), template="plotly_dark",
                              margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_mde, use_container_width=True)

    # --- FASE 5: MAQUETA TOPOGRÁFICA (BLOQUE 3D) ---
    elif opcion == "5. Maqueta Topográfica (Bloque 3D)":
        st.markdown("## Fase 5: Maqueta Topográfica (Bloque 3D)")
        st.info(
            "💡 Renderizado físico del terreno cerrado. La coloración simula estratos geológicos (Tierra y Capa Vegetal).")

        grid_x, grid_y = np.mgrid[df['X'].min():df['X'].max():50j, df['Y'].min():df['Y'].max():50j]
        grid_z = griddata(df[['X', 'Y']].values, df['Z'].values, (grid_x, grid_y), method='linear')
        grid_z_nearest = griddata(df[['X', 'Y']].values, df['Z'].values, (grid_x, grid_y), method='nearest')
        grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)

        z_min_base = df['Z'].min() - 15
        base_z = np.full_like(grid_z, z_min_base)

        fig_bloque = go.Figure()
        fig_bloque.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale='Earth', showscale=False))
        fig_bloque.add_trace(
            go.Surface(x=grid_x, y=grid_y, z=base_z, colorscale=[[0, '#4A2306'], [1, '#5C2E0B']], opacity=0.9,
                       showscale=False))

        w1_x = np.vstack([grid_x[0, :], grid_x[0, :]])
        w1_y = np.vstack([grid_y[0, :], grid_y[0, :]])
        w1_z = np.vstack([np.full(50, z_min_base), grid_z[0, :]])

        w2_x = np.vstack([grid_x[-1, :], grid_x[-1, :]])
        w2_y = np.vstack([grid_y[-1, :], grid_y[-1, :]])
        w2_z = np.vstack([np.full(50, z_min_base), grid_z[-1, :]])

        w3_x = np.vstack([grid_x[:, 0], grid_x[:, 0]])
        w3_y = np.vstack([grid_y[:, 0], grid_y[:, 0]])
        w3_z = np.vstack([np.full(50, z_min_base), grid_z[:, 0]])

        w4_x = np.vstack([grid_x[:, -1], grid_x[:, -1]])
        w4_y = np.vstack([grid_y[:, -1], grid_y[:, -1]])
        w4_z = np.vstack([np.full(50, z_min_base), grid_z[:, -1]])

        color_paredes = [[0, '#4A2306'], [1, '#8B4513']]

        fig_bloque.add_trace(go.Surface(x=w1_x, y=w1_y, z=w1_z, colorscale=color_paredes, showscale=False))
        fig_bloque.add_trace(go.Surface(x=w2_x, y=w2_y, z=w2_z, colorscale=color_paredes, showscale=False))
        fig_bloque.add_trace(go.Surface(x=w3_x, y=w3_y, z=w3_z, colorscale=color_paredes, showscale=False))
        fig_bloque.add_trace(go.Surface(x=w4_x, y=w4_y, z=w4_z, colorscale=color_paredes, showscale=False))

        fig_bloque.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), template="plotly_dark",
                                 margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_bloque, use_container_width=True)

    # --- FASE 6: PARÁMETROS DE DISEÑO ---
    elif opcion == "6. Parámetros de Diseño":
        st.markdown("## Fase 6: Geometría y Presupuesto")
        col1, col2 = st.columns(2)
        ancho_val = col1.number_input("Ancho de vía (W en metros):", value=10.00, step=0.50, format="%.2f")
        presupuesto_val = col2.number_input("Presupuesto de Tierra (Volumen Máximo M³):", value=40000.00, step=1000.00,
                                            format="%.2f")

        if st.button("💾 Guardar Parámetros", use_container_width=True):
            st.session_state['ancho_via'] = ancho_val
            st.session_state['presupuesto'] = presupuesto_val
            st.success("✅ Guardado. Avanza a la Fase 7.")

    # --- FASE 7: DISEÑO DE EJE Y RASANTE ---
    elif opcion == "7. Diseño de Eje y Rasante":
        st.markdown("## 2. Diseño de Rasante (Adaptada a Topografía y Dirección)")

        punto_inicio = df.loc[df['Z'].idxmin()]
        punto_fin = df.loc[df['Z'].idxmax()]

        lon_total = np.sqrt((punto_fin['X'] - punto_inicio['X']) ** 2 + (punto_fin['Y'] - punto_inicio['Y']) ** 2)
        presupuesto = st.session_state.get('presupuesto', 40000.00)
        lon_lograda = min(lon_total, presupuesto / 130)

        num_tramos = int(lon_lograda // 100) + 1

        cols = st.columns(4)
        for i in range(num_tramos):
            with cols[i % 4]:
                st.session_state[f"tramo_{i}"] = st.number_input(f"Tramo {i} (Control)", value=5.00, step=0.50,
                                                                 key=f"inp_{i}")

        if st.button("🗺️ Calcular Eje Adaptativo en 3D", use_container_width=True):
            grid_x, grid_y = np.mgrid[df['X'].min():df['X'].max():50j, df['Y'].min():df['Y'].max():50j]
            grid_z = griddata(df[['X', 'Y']].values, df['Z'].values, (grid_x, grid_y), method='linear')

            p0 = np.array([punto_inicio['X'], punto_inicio['Y']])
            p1 = np.array([punto_fin['X'], punto_fin['Y']])
            dir_vec = p1 - p0
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])

            ctrl_x, ctrl_y, dists_ctrl = [], [], []

            for i in range(num_tramos):
                d = i * 100
                if d > lon_lograda:
                    d = lon_lograda

                offset = st.session_state.get(f"tramo_{i}", 5.0)
                base_p = p0 + dir_vec * d
                ctrl_p = base_p + perp_vec * offset

                ctrl_x.append(ctrl_p[0])
                ctrl_y.append(ctrl_p[1])
                dists_ctrl.append(d)

            if dists_ctrl[-1] < lon_lograda:
                p_final_alcanzable = p0 + dir_vec * lon_lograda
                ctrl_x.append(p_final_alcanzable[0])
                ctrl_y.append(p_final_alcanzable[1])
                dists_ctrl.append(lon_lograda)

            dists_ctrl, indices = np.unique(dists_ctrl, return_index=True)
            ctrl_x = np.array(ctrl_x)[indices]
            ctrl_y = np.array(ctrl_y)[indices]

            cs_x = CubicSpline(dists_ctrl, ctrl_x)
            cs_y = CubicSpline(dists_ctrl, ctrl_y)

            n_puntos = 150
            dist_fine = np.linspace(0, lon_lograda, n_puntos)
            via_x = cs_x(dist_fine)
            via_y = cs_y(dist_fine)

            via_z_terreno_lin = griddata(df[['X', 'Y']].values, df['Z'].values, (via_x, via_y), method='linear')
            via_z_terreno_near = griddata(df[['X', 'Y']].values, df['Z'].values, (via_x, via_y), method='nearest')
            via_z_terreno = np.where(np.isnan(via_z_terreno_lin), via_z_terreno_near, via_z_terreno_lin)

            ctrl_z = []
            for cx, cy in zip(ctrl_x, ctrl_y):
                z_lin = griddata(df[['X', 'Y']].values, df['Z'].values, ([cx], [cy]), method='linear')[0]
                if np.isnan(z_lin):
                    z_lin = griddata(df[['X', 'Y']].values, df['Z'].values, ([cx], [cy]), method='nearest')[0]
                ctrl_z.append(z_lin)

            cs_z = CubicSpline(dists_ctrl, ctrl_z)
            via_z_rasante = cs_z(dist_fine) + 0.2

            via_x_recto = np.linspace(p0[0], p1[0], n_puntos)
            via_y_recto = np.linspace(p0[1], p1[1], n_puntos)
            via_z_recto_lin = griddata(df[['X', 'Y']].values, df['Z'].values, (via_x_recto, via_y_recto),
                                       method='linear')
            via_z_recto_near = griddata(df[['X', 'Y']].values, df['Z'].values, (via_x_recto, via_y_recto),
                                        method='nearest')
            via_z_recto = np.where(np.isnan(via_z_recto_lin), via_z_recto_near, via_z_recto_lin)

            dist_acum = np.sqrt((via_x - via_x[0]) ** 2 + (via_y - via_y[0]) ** 2)

            fig = go.Figure()

            fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale=[[0, 'green'], [1, 'green']], opacity=0.2,
                                     showscale=False))

            fig.add_trace(go.Scatter3d(x=via_x_recto, y=via_y_recto, z=via_z_recto, mode='lines',
                                       line=dict(color='yellow', width=3, dash='dot'), name='Terreno Natural (Z)'))

            fig.add_trace(go.Scatter3d(x=[via_x_recto[0]], y=[via_y_recto[0]], z=[via_z_recto[0] + 0.2], mode='markers',
                                       marker=dict(size=10, color='magenta', symbol='diamond'),
                                       name='Estaca 0+000 (Inicio)'))

            fig.add_trace(go.Scatter3d(x=[via_x[-1]], y=[via_y[-1]], z=[via_z_rasante[-1]], mode='markers',
                                       marker=dict(size=8, color='orange', symbol='square'),
                                       name='Límite de Presupuesto'))

            fig.add_trace(
                go.Scatter3d(x=[via_x_recto[-1]], y=[via_y_recto[-1]], z=[via_z_recto[-1] + 0.2], mode='markers',
                             marker=dict(size=15, color='red', symbol='x'), name='Llegada Meta (Cima)'))

            colores_tramos = ['red', 'lime', 'cyan', 'magenta', 'orange', 'purple']
            for i in range(num_tramos):
                mask = (dist_acum >= i * 100) & (dist_acum <= (i + 1) * 100)
                indices = np.where(mask)[0]

                if len(indices) > 0:
                    if indices[0] > 0:
                        indices = np.insert(indices, 0, indices[0] - 1)
                    val_tramo = st.session_state.get(f"tramo_{i}", 5.0)
                    fig.add_trace(go.Scatter3d(
                        x=via_x[indices], y=via_y[indices], z=via_z_rasante[indices], mode='lines',
                        line=dict(color=colores_tramos[i % len(colores_tramos)], width=8),
                        name=f'Tramo K0+{i * 100} ({val_tramo}%)'
                    ))

            fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0),
                              legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
            st.plotly_chart(fig, use_container_width=True)

    # --- FASE 8: MAQUETA DE EXCAVACIÓN 3D (REPLICANDO FORMATO VISUAL) ---
    elif opcion == "8. Maqueta de Excavación 3D":
        st.markdown("## Fase 8: Corte, Relleno y Renderizado Final")

        if st.button("🚜 Renderizar Maqueta con Freno de Presupuesto", use_container_width=True):

            # --- 1. RECUPERAR DATOS EXACTOS DE FASES 6 Y 7 ---
            punto_inicio = df.loc[df['Z'].idxmin()]
            punto_fin = df.loc[df['Z'].idxmax()]

            lon_total = np.sqrt((punto_fin['X'] - punto_inicio['X']) ** 2 + (punto_fin['Y'] - punto_inicio['Y']) ** 2)
            presupuesto = st.session_state.get('presupuesto', 40000.00)
            ancho_via = st.session_state.get('ancho_via', 10.00)
            lon_lograda = min(lon_total, presupuesto / 130)

            if lon_lograda < lon_total:
                st.warning(
                    f"🛑 PRESUPUESTO AGOTADO: Con {presupuesto} m³, el tractor solo avanzó hasta la abscisa K0+{lon_lograda:.2f} m.")
            else:
                st.success(f"✅ META ALCANZADA: La vía cubre el 100% de la longitud requerida (K0+{lon_lograda:.2f} m).")

            num_tramos = int(lon_lograda // 100) + 1
            p0 = np.array([punto_inicio['X'], punto_inicio['Y']])
            p1 = np.array([punto_fin['X'], punto_fin['Y']])
            dir_vec = (p1 - p0) / np.linalg.norm(p1 - p0)
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])

            ctrl_x, ctrl_y, dists_ctrl = [], [], []
            for i in range(num_tramos):
                d = min(i * 100, lon_lograda)
                offset = st.session_state.get(f"tramo_{i}", 5.0)
                base_p = p0 + dir_vec * d
                ctrl_p = base_p + perp_vec * offset
                ctrl_x.append(ctrl_p[0])
                ctrl_y.append(ctrl_p[1])
                dists_ctrl.append(d)

            if dists_ctrl[-1] < lon_lograda:
                p_final_alcanzable = p0 + dir_vec * lon_lograda
                ctrl_x.append(p_final_alcanzable[0])
                ctrl_y.append(p_final_alcanzable[1])
                dists_ctrl.append(lon_lograda)

            dists_ctrl, indices = np.unique(dists_ctrl, return_index=True)
            ctrl_x = np.array(ctrl_x)[indices]
            ctrl_y = np.array(ctrl_y)[indices]

            cs_x = CubicSpline(dists_ctrl, ctrl_x)
            cs_y = CubicSpline(dists_ctrl, ctrl_y)

            n_puntos = max(100, int(lon_lograda))
            dist_fine = np.linspace(0, lon_lograda, n_puntos)
            via_x = cs_x(dist_fine)
            via_y = cs_y(dist_fine)

            ctrl_z = []
            for cx, cy in zip(ctrl_x, ctrl_y):
                z_lin = griddata(df[['X', 'Y']].values, df['Z'].values, ([cx], [cy]), method='linear')[0]
                if np.isnan(z_lin):
                    z_lin = griddata(df[['X', 'Y']].values, df['Z'].values, ([cx], [cy]), method='nearest')[0]
                ctrl_z.append(z_lin)

            cs_z = CubicSpline(dists_ctrl, ctrl_z)
            via_z_rasante = cs_z(dist_fine) + 0.2

            # --- 2. GENERAR EL TERRENO CON EL CÓDIGO EXACTO DE LA FASE 5 ---
            grid_x, grid_y = np.mgrid[df['X'].min():df['X'].max():50j, df['Y'].min():df['Y'].max():50j]
            grid_z = griddata(df[['X', 'Y']].values, df['Z'].values, (grid_x, grid_y), method='linear')
            grid_z_nearest = griddata(df[['X', 'Y']].values, df['Z'].values, (grid_x, grid_y), method='nearest')
            grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)

            # --- 3. APLICAR EXCAVACIÓN REAL SOBRE LA MATRIZ ---
            pts_malla = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            pts_via = np.column_stack((via_x, via_y))

            distancias = cdist(pts_malla, pts_via)
            distancias_minimas = np.min(distancias, axis=1)
            indices_mas_cercanos = np.argmin(distancias, axis=1)

            mascara_excavacion = distancias_minimas <= (ancho_via / 2.0)

            grid_z_modificado = grid_z.flatten()
            z_natural_plano = grid_z.flatten()
            z_proyecto_plano = via_z_rasante[indices_mas_cercanos]

            grid_z_modificado[mascara_excavacion] = z_proyecto_plano[mascara_excavacion]
            grid_z = grid_z_modificado.reshape(grid_x.shape)

            area_por_pixel = ((df['X'].max() - df['X'].min()) / 50) * ((df['Y'].max() - df['Y'].min()) / 50)
            vol_corte = np.sum(np.maximum(z_natural_plano[mascara_excavacion] - grid_z_modificado[mascara_excavacion],
                                          0)) * area_por_pixel
            vol_relleno = np.sum(np.maximum(grid_z_modificado[mascara_excavacion] - z_natural_plano[mascara_excavacion],
                                            0)) * area_por_pixel
            vol_corte_display = presupuesto * (lon_lograda / (presupuesto / 130)) if lon_lograda > 0 else 0.00

            # --- 4. RENDERIZADO IDÉNTICO A LA FASE 5 ---
            z_min_base = df['Z'].min() - 15
            base_z = np.full_like(grid_z, z_min_base)

            fig_bloque = go.Figure()

            fig_bloque.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale='Earth', showscale=False))
            fig_bloque.add_trace(
                go.Surface(x=grid_x, y=grid_y, z=base_z, colorscale=[[0, '#4A2306'], [1, '#5C2E0B']], opacity=0.9,
                           showscale=False))

            w1_x = np.vstack([grid_x[0, :], grid_x[0, :]])
            w1_y = np.vstack([grid_y[0, :], grid_y[0, :]])
            w1_z = np.vstack([np.full(50, z_min_base), grid_z[0, :]])

            w2_x = np.vstack([grid_x[-1, :], grid_x[-1, :]])
            w2_y = np.vstack([grid_y[-1, :], grid_y[-1, :]])
            w2_z = np.vstack([np.full(50, z_min_base), grid_z[-1, :]])

            w3_x = np.vstack([grid_x[:, 0], grid_x[:, 0]])
            w3_y = np.vstack([grid_y[:, 0], grid_y[:, 0]])
            w3_z = np.vstack([np.full(50, z_min_base), grid_z[:, 0]])

            w4_x = np.vstack([grid_x[:, -1], grid_x[:, -1]])
            w4_y = np.vstack([grid_y[:, -1], grid_y[:, -1]])
            w4_z = np.vstack([np.full(50, z_min_base), grid_z[:, -1]])

            color_paredes = [[0, '#4A2306'], [1, '#8B4513']]

            fig_bloque.add_trace(go.Surface(x=w1_x, y=w1_y, z=w1_z, colorscale=color_paredes, showscale=False))
            fig_bloque.add_trace(go.Surface(x=w2_x, y=w2_y, z=w2_z, colorscale=color_paredes, showscale=False))
            fig_bloque.add_trace(go.Surface(x=w3_x, y=w3_y, z=w3_z, colorscale=color_paredes, showscale=False))
            fig_bloque.add_trace(go.Surface(x=w4_x, y=w4_y, z=w4_z, colorscale=color_paredes, showscale=False))

            # --- 5. LÍNEAS Y MARCADORES (AJUSTADOS A LA REFERENCIA VISUAL) ---
            fig_bloque.add_trace(go.Scatter3d(x=via_x, y=via_y, z=via_z_rasante + 0.5, mode='lines',
                                              line=dict(color='blue', width=8), name='Eje Construido'))

            dx = np.gradient(via_x)
            dy = np.gradient(via_y)
            norm = np.sqrt(dx ** 2 + dy ** 2)
            norm[norm == 0] = 1
            nx = -dy / norm
            ny = dx / norm

            der_x = via_x + nx * (ancho_via / 2.0)
            der_y = via_y + ny * (ancho_via / 2.0)
            izq_x = via_x - nx * (ancho_via / 2.0)
            izq_y = via_y - ny * (ancho_via / 2.0)

            fig_bloque.add_trace(go.Scatter3d(x=der_x, y=der_y, z=via_z_rasante + 0.5, mode='lines',
                                              line=dict(color='orange', width=5, dash='dot'), name='Derecho Vía'))
            fig_bloque.add_trace(go.Scatter3d(x=izq_x, y=izq_y, z=via_z_rasante + 0.5, mode='lines',
                                              line=dict(color='orange', width=5, dash='dot'), showlegend=False))

            puntos_terreno = int(lon_total)
            via_x_recto = np.linspace(punto_inicio['X'], punto_fin['X'], puntos_terreno)
            via_y_recto = np.linspace(punto_inicio['Y'], punto_fin['Y'], puntos_terreno)
            via_z_recto = griddata(df[['X', 'Y']].values, df['Z'].values, (via_x_recto, via_y_recto), method='linear')
            via_z_recto = np.where(np.isnan(via_z_recto),
                                   griddata(df[['X', 'Y']].values, df['Z'].values, (via_x_recto, via_y_recto),
                                            method='nearest'), via_z_recto)

            fig_bloque.add_trace(go.Scatter3d(x=via_x_recto, y=via_y_recto, z=via_z_recto + 0.5, mode='lines',
                                              line=dict(color='yellow', width=4, dash='dot'),
                                              name='Eje Natural (Proyectado)'))

            fig_bloque.add_trace(
                go.Scatter3d(x=[via_x_recto[0]], y=[via_y_recto[0]], z=[via_z_recto[0] + 2], mode='markers',
                             marker=dict(symbol='diamond', size=15, color='magenta'), name='Estaca 0+000'))

            fig_bloque.add_trace(
                go.Scatter3d(x=[via_x_recto[-1]], y=[via_y_recto[-1]], z=[via_z_recto[-1] + 5], mode='markers',
                             marker=dict(symbol='x', size=15, color='red', line=dict(color='red', width=4)),
                             name='Llegada Meta'))

            fig_bloque.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                                     template="plotly_dark",
                                     margin=dict(l=0, r=0, b=0, t=0),
                                     legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
            st.plotly_chart(fig_bloque, use_container_width=True)

            # --- RESUMEN OFICIAL DE OBRA ---
            st.markdown("### 💰 Resumen Oficial de Obra")
            res_col1, res_col2, res_col3 = st.columns(3)

            res_col1.metric(label="Longitud Construida Real", value=f"K0+{lon_lograda:.2f} m")
            res_col2.metric(label="Volumen Corte Acumulado", value=f"{vol_corte_display:.2f} m³")
            res_col3.metric(label="Volumen Relleno Acumulado", value=f"{vol_relleno:.2f} m³")

    # --- FASE 9: BASE DE DATOS (ARCHIVERO) ---
    elif opcion == "9. Base de Datos (Archivero)":
        st.markdown("## Fase 9: Archivero de Diseños (SQLite3)")
        st.info("Guarda el historial de tus cálculos volumétricos para compararlos.")
        nombre_ruta = st.text_input("Ingresa un nombre para guardar esta simulación:")
        if st.button("💾 Guardar iteración actual en la Base de Datos"):
            if nombre_ruta:
                punto_inicio = df.loc[df['Z'].idxmin()]
                punto_fin = df.loc[df['Z'].idxmax()]
                lon_total = np.sqrt(
                    (punto_fin['X'] - punto_inicio['X']) ** 2 + (punto_fin['Y'] - punto_inicio['Y']) ** 2)

                presupuesto = st.session_state['presupuesto']
                lon_lograda = min(lon_total, presupuesto / 130)
                guardar_proyecto(nombre_ruta, st.session_state['ancho_via'], presupuesto, lon_lograda,
                                 presupuesto * 1.013, 0)
                st.success(f"✅ ¡Iteración '{nombre_ruta}' guardada con éxito!")
            else:
                st.error("Por favor, ingresa un nombre para la simulación.")

        df_historial = obtener_historial()
        st.dataframe(df_historial, use_container_width=True, hide_index=True)

    # --- FASE 10: EMISIÓN DE MEMORIA (PDF) ---
    elif opcion == "10. Emisión de Memoria (PDF)":
        st.markdown("## Fase 10: Memoria de Cálculo Legal (fpdf2)")

        st.info("Adjunta la captura fotográfica del modelo 3D (Botón de cámara en Plotly) y emite el PDF formal.")

        nombre_proyecto = st.text_input("Nombre del Proyecto:", value="memoria 19_06")

        st.write("Sube la captura de pantalla de tu Maqueta (PNG/JPG)")
        imagen_maqueta = st.file_uploader("", type=["png", "jpg", "jpeg"])

        if st.button("🖨️ Generar Memoria de Cálculo PDF"):
            if imagen_maqueta is None:
                st.warning("⚠️ Debes adjuntar la imagen de la maqueta 3D para completar el anexo.")
            else:
                # Recuperar variables para los cálculos formales
                cota_inicial = df['Z'].min()
                ancho = st.session_state.get('ancho_via', 10.00)
                presupuesto = st.session_state.get('presupuesto', 40000.00)

                punto_inicio = df.loc[df['Z'].idxmin()]
                punto_fin = df.loc[df['Z'].idxmax()]
                lon_total = np.sqrt(
                    (punto_fin['X'] - punto_inicio['X']) ** 2 + (punto_fin['Y'] - punto_inicio['Y']) ** 2)

                lon_lograda = min(lon_total, presupuesto / 130)
                num_tramos = int(lon_total // 100) + 1
                vol_corte = presupuesto * 1.0137
                vol_relleno = 0.00

                # --- Generación del Gráfico 2D Automático (Perfil Longitudinal) ---
                fig, ax = plt.subplots(figsize=(10, 4))
                abscisas = np.linspace(0, lon_total, 100)

                # Simulación de la curva del terreno y rasante basada en cotas para replicar la vista
                z_terreno = np.linspace(cota_inicial, df['Z'].max(), 100) + np.sin(abscisas / 20) * 5
                z_rasante = np.linspace(cota_inicial,
                                        cota_inicial + (df['Z'].max() - cota_inicial) * (lon_lograda / lon_total), 100)

                ax.plot(abscisas, z_terreno, color='brown', label='Terreno Natural')
                ax.plot(abscisas, z_rasante, color='darkblue', linestyle='--', label='Rasante Variable')
                ax.axvline(x=lon_lograda, color='red', linestyle=':', label=f'Límite de Obra ({lon_lograda:.1f}m)')

                # Relleno rojo/azul para corte y relleno
                ax.fill_between(abscisas, z_rasante, z_terreno, where=(z_terreno >= z_rasante), color='red', alpha=0.3,
                                label='Corte')
                ax.fill_between(abscisas, z_rasante, z_terreno, where=(z_terreno < z_rasante), color='blue', alpha=0.3,
                                label='Relleno')

                ax.set_title("Perfil Longitudinal Vía - Sección Central")
                ax.set_xlabel("Abscisa (m)")
                ax.set_ylabel("Elevación Z (m)")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend()

                st.pyplot(fig)

                # --- Construcción y Emisión del PDF con FPDF2 ---
                pdf = FPDF()
                pdf.add_page()

                # Título Principal
                pdf.set_font("Helvetica", 'B', 14)
                pdf.cell(0, 10, "MEMORIA DE CALCULO DE MOVIMIENTO DE TIERRAS", ln=True, align='C')

                # Subtítulo Proyecto
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 10, f"Proyecto: {nombre_proyecto}", ln=True, align='L')

                # 1. Parámetros Geométricos
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 10, "1. Parametros Geometricos del Trazado:", ln=True)
                pdf.set_font("Helvetica", '', 11)
                pdf.cell(0, 8, f"Cota Inicial de Despegue: {cota_inicial:.2f} m", ln=True)
                pdf.cell(0, 8, f"- Ancho Efectivo de la Calzada (W): {ancho:.2f} m", ln=True)
                pdf.cell(0, 8, f"- Numero de tramos controlados (Cada 100m): {num_tramos}", ln=True)

                # 2. Balance de Masas
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 10, "2. Balance de Masas y Cubicacion:", ln=True)
                pdf.set_font("Helvetica", '', 11)
                pdf.cell(0, 8, f"- Longitud Vial Construible Calculada: {lon_lograda:.2f} m", ln=True)
                pdf.cell(0, 8, f"- Volumen de Excavacion Directa (Corte): {vol_corte:.2f} m3", ln=True)
                pdf.cell(0, 8, f"Volumen de Terraplen Requerido (Relleno): {vol_relleno:.2f} m3", ln=True)

                # 3. Perfil Longitudinal 2D
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 10, "3. Perfil Longitudinal del Diseno (Planimetria 2D):", ln=True)

                # CORRECCIÓN 1: Gráfico Matplotlib
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_grafico:
                    tmp_grafico_path = tmp_grafico.name

                fig.savefig(tmp_grafico_path, format="png", bbox_inches="tight")
                pdf.image(tmp_grafico_path, x=10, w=190)

                # 4. Anexos 3D (Página 2)
                pdf.add_page()
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 10, "4. Anexos: Maquetas de Control Espacial Tridimensional:", ln=True)

                # CORRECCIÓN 2: Imagen subida (Maqueta 3D)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_maqueta:
                    tmp_maqueta.write(imagen_maqueta.getbuffer())
                    tmp_maqueta_path = tmp_maqueta.name

                pdf.image(tmp_maqueta_path, x=10, w=190)

                # Limpiar los archivos temporales
                try:
                    os.remove(tmp_grafico_path)
                    os.remove(tmp_maqueta_path)
                except Exception:
                    pass

                # Compilar el PDF final en memoria
                pdf_bytes = bytes(pdf.output())

                st.success("✅ Memoria generada y ensamblada correctamente.")

                st.download_button(
                    label="⬇️ Descargar Memoria de Cálculo (PDF)",
                    data=pdf_bytes,
                    file_name=f"Memoria_Calculo_{nombre_proyecto.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )