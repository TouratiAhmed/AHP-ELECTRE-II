import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import openpyxl
import json
import time

# Configuration de la page
st.set_page_config(page_title="AHP & ELECTRE II", layout="wide")

# Initialisation du state
if 'poids_ahp' not in st.session_state:
    st.session_state.poids_ahp = None
if 'noms_criteres_ahp' not in st.session_state:
    st.session_state.noms_criteres_ahp = None
if 'page_actuelle' not in st.session_state:
    st.session_state.page_actuelle = "🏠 Accueil"
if 'mode_saisie' not in st.session_state:
    st.session_state.mode_saisie = None
if 'donnees_chargees' not in st.session_state:
    st.session_state.donnees_chargees = {}
if 'matrice_ahp' not in st.session_state:
    st.session_state.matrice_ahp = None
if 'poids_ajustes' not in st.session_state:
    st.session_state.poids_ajustes = None
# NOUVEAUX états pour gérer les données importées
if 'alternatives_importees' not in st.session_state:
    st.session_state.alternatives_importees = None
if 'criteres_importes' not in st.session_state:
    st.session_state.criteres_importes = None
if 'performances_importees' not in st.session_state:
    st.session_state.performances_importees = None
if 'orientation_donnees' not in st.session_state:
    st.session_state.orientation_donnees = None

# Titre principal
st.title("Analyse de Décision Multicritere-MCDA")
st.markdown("### AHP et ELECTRE II")

# Sidebar pour navigation
options_menu = [" Accueil", " Choix des données", " AHP - Analytic Hierarchy Process", " ELECTRE II"]

# Déterminer l'index actuel de manière sécurisée
try:
    index_actuel = options_menu.index(st.session_state.page_actuelle)
except (ValueError, AttributeError):
    index_actuel = 0

methode = st.sidebar.selectbox(
    "Choisir la méthode",
    options_menu,
    index=index_actuel
)
st.session_state.page_actuelle = methode

# ============= FONCTIONS AHP =============
def calculer_poids_ahp(matrice):
    """Calcule les poids par la méthode des valeurs propres"""
    n = len(matrice)
    valeurs_propres, vecteurs_propres = np.linalg.eig(matrice)
    idx_max = np.argmax(valeurs_propres.real)
    vecteur_principal = vecteurs_propres[:, idx_max].real
    poids = vecteur_principal / vecteur_principal.sum()
    lambda_max = valeurs_propres[idx_max].real
    return poids, lambda_max

def calculer_coherence(matrice, lambda_max):
    """Calcule l'indice et le ratio de cohérence"""
    n = len(matrice)
    CI = (lambda_max - n) / (n - 1)
    
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    CR = CI / RI.get(n, 1.49) if n > 2 else 0
    
    return CI, CR

def reconstruire_matrice_depuis_poids(poids):
    """Reconstruit une matrice de comparaison approximative depuis les poids"""
    n = len(poids)
    matrice = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrice[i][j] = poids[i] / poids[j]
    return matrice

def create_draggable_bar_chart(noms_criteres, poids_pourcentage):
    """Crée un graphique à barres interactif avec drag-and-drop"""
    
    # Préparer les données pour le JavaScript
    data_json = json.dumps({
        'labels': noms_criteres,
        'values': [float(p) for p in poids_pourcentage]
    })
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: 'Arial', sans-serif;
                background-color: #0e1117;
            }}
            #chart {{
                width: 100%;
                height: 500px;
            }}
            .info-box {{
                background-color: #262730;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #4CAF50;
            }}
            .controls {{
                margin-top: 20px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .control-item {{
                background-color: #262730;
                padding: 10px 15px;
                border-radius: 5px;
                color: white;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .control-item input {{
                width: 80px;
                padding: 5px;
                border: 1px solid #4CAF50;
                border-radius: 3px;
                background-color: #0e1117;
                color: white;
            }}
            button {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
            }}
            button:hover {{
                background-color: #45a049;
            }}
        </style>
    </head>
    <body>
        <div class="info-box">
            <strong>💡 Mode interactif :</strong> 
            <ol style="margin: 10px 0;">
                <li>Cliquez sur une barre et déplacez-la verticalement</li>
                <li>Ou utilisez les champs ci-dessous pour entrer une valeur précise</li>
                <li>Les poids sont automatiquement normalisés à 100%</li>
            </ol>
        </div>
        
        <div id="chart"></div>
        
        <div class="controls" id="controls"></div>
        
        <div style="margin-top: 20px;">
            <button onclick="resetWeights()"> Réinitialiser</button>
            <button onclick="equalWeights()"> Poids égaux</button>
            <button onclick="exportWeights()"> Exporter les poids</button>
        </div>
        
        <script>
            let chartData = {data_json};
            let currentValues = [...chartData.values];
            let isDragging = false;
            let dragIndex = -1;
            
            function normalizeValues(values) {{
                const sum = values.reduce((a, b) => a + b, 0);
                return sum > 0 ? values.map(v => (v / sum) * 100) : values;
            }}
            
            function createChart(values) {{
                const normalizedValues = normalizeValues(values);
                
                const trace = {{
                    x: chartData.labels,
                    y: normalizedValues,
                    type: 'bar',
                    marker: {{
                        color: normalizedValues,
                        colorscale: 'Viridis',
                        showscale: true,
                        colorbar: {{
                            title: 'Poids (%)',
                            titlefont: {{ color: 'white' }},
                            tickfont: {{ color: 'white' }}
                        }}
                    }},
                    text: normalizedValues.map(v => v.toFixed(1) + '%'),
                    textposition: 'outside',
                    textfont: {{ color: 'white', size: 14 }},
                    hovertemplate: '<b>%{{x}}</b><br>Poids: %{{y:.2f}}%<extra></extra>'
                }};
                
                const layout = {{
                    title: {{
                        text: 'Distribution des poids (%) - Glissez les barres pour ajuster',
                        font: {{ color: 'white', size: 18 }}
                    }},
                    xaxis: {{
                        title: 'Critère',
                        titlefont: {{ color: 'white' }},
                        tickfont: {{ color: 'white' }},
                        gridcolor: '#2e2e2e'
                    }},
                    yaxis: {{
                        title: 'Pourcentage (%)',
                        titlefont: {{ color: 'white' }},
                        tickfont: {{ color: 'white' }},
                        range: [0, Math.max(...normalizedValues) + 10],
                        gridcolor: '#2e2e2e'
                    }},
                    plot_bgcolor: '#0e1117',
                    paper_bgcolor: '#0e1117',
                    font: {{ color: 'white' }},
                    showlegend: false,
                    dragmode: 'pan',
                    hovermode: 'closest'
                }};
                
                const config = {{
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                    displaylogo: false
                }};
                
                Plotly.newPlot('chart', [trace], layout, config);
                
                // Ajouter les événements de drag
                const chartDiv = document.getElementById('chart');
                
                chartDiv.on('plotly_click', function(data) {{
                    dragIndex = data.points[0].pointIndex;
                    isDragging = true;
                }});
                
                chartDiv.on('plotly_hover', function(data) {{
                    if (isDragging && dragIndex !== -1) {{
                        const yValue = data.points[0].y;
                        currentValues[dragIndex] = Math.max(0, Math.min(100, yValue));
                        updateChart(currentValues);
                        updateControls(normalizeValues(currentValues));
                    }}
                }});
                
                document.addEventListener('mouseup', function() {{
                    isDragging = false;
                    dragIndex = -1;
                }});
            }}
            
            function updateChart(values) {{
                const normalizedValues = normalizeValues(values);
                
                Plotly.update('chart', {{
                    y: [normalizedValues],
                    text: [normalizedValues.map(v => v.toFixed(1) + '%')],
                    'marker.color': [normalizedValues]
                }});
            }}
            
            function createControls() {{
                const controlsDiv = document.getElementById('controls');
                controlsDiv.innerHTML = '';
                
                chartData.labels.forEach((label, index) => {{
                    const controlItem = document.createElement('div');
                    controlItem.className = 'control-item';
                    controlItem.innerHTML = `
                        <label>${{label}}:</label>
                        <input type="number" 
                               id="input_${{index}}" 
                               value="${{currentValues[index].toFixed(1)}}" 
                               min="0" 
                               max="100" 
                               step="0.5"
                               onchange="updateFromInput(${{index}}, this.value)">
                        <span>%</span>
                    `;
                    controlsDiv.appendChild(controlItem);
                }});
            }}
            
            function updateControls(values) {{
                values.forEach((value, index) => {{
                    const input = document.getElementById(`input_${{index}}`);
                    if (input && document.activeElement !== input) {{
                        input.value = value.toFixed(1);
                    }}
                }});
            }}
            
            function updateFromInput(index, value) {{
                currentValues[index] = parseFloat(value) || 0;
                updateChart(currentValues);
                const normalized = normalizeValues(currentValues);
                updateControls(normalized);
            }}
            
            function resetWeights() {{
                currentValues = [...chartData.values];
                updateChart(currentValues);
                updateControls(normalizeValues(currentValues));
            }}
            
            function equalWeights() {{
                const equalValue = 100 / chartData.labels.length;
                currentValues = chartData.labels.map(() => equalValue);
                updateChart(currentValues);
                updateControls(normalizeValues(currentValues));
            }}
            
            function exportWeights() {{
                const normalized = normalizeValues(currentValues);
                const exportData = {{}};
                chartData.labels.forEach((label, index) => {{
                    exportData[label] = normalized[index].toFixed(4);
                }});
                
                alert('Poids normalisés:\\n' + JSON.stringify(exportData, null, 2));
                console.log('Exported weights:', exportData);
            }}
            
            // Initialisation
            createChart(currentValues);
            createControls();
        </script>
    </body>
    </html>
    """
    
    return html_code

# ============= FONCTIONS ELECTRE II =============

def normaliser_performances(performances, sens_optimisation):
    """
    Normalise la matrice de performances selon les formules données:
    
    Pour critères à MINIMISER:
    g*ⱼ(a) = [maxᵦ gⱼ(b) - gⱼ(a)] / [maxᵦ gⱼ(b) - minᵦ gⱼ(b)]
    → Plus gⱼ(a) est faible, plus g*ⱼ(a) est élevé (meilleure alternative)
    
    Pour critères à MAXIMISER:
    g*ⱼ(a) = [gⱼ(a) - minᵦ gⱼ(b)] / [maxᵦ gⱼ(b) - minᵦ gⱼ(b)]
    → Plus gⱼ(a) est élevé, plus g*ⱼ(a) est élevé (meilleure alternative)
    
    Parameters:
    -----------
    performances : array-like, matrice (n_alternatives x m_criteres)
    sens_optimisation : list, 'max' pour critères à maximiser, 'min' pour critères à minimiser
    
    Returns:
    --------
    performances_norm : array normalisées (valeurs entre 0 et 1)
    """
    performances = np.array(performances, dtype=float)
    n_alt, m_crit = performances.shape
    performances_norm = np.zeros((n_alt, m_crit))
    
    for j in range(m_crit):
        col = performances[:, j]
        max_val = np.max(col)
        min_val = np.min(col)
        etendue = max_val - min_val
        
        if etendue > 0:
            if sens_optimisation[j] == 'max':
                # Pour maximiser: g*ⱼ(a) = [gⱼ(a) - min] / [max - min]
                performances_norm[:, j] = (col - min_val) / etendue
            else:  # 'min'
                # Pour minimiser: g*ⱼ(a) = [max - gⱼ(a)] / [max - min]
                performances_norm[:, j] = (max_val - col) / etendue
        else:
            # Si tous les critères ont la même valeur, on met 0.5 (neutralité)
            performances_norm[:, j] = 0.5
    
    return performances_norm

def calculer_concordance_electre2(performances, poids):
    """
    Calcule la matrice de concordance selon ELECTRE II
    Pour chaque paire (a,b), on somme les poids des critères où a >= b
    
    Parameters:
    -----------
    performances : matrice des performances
    poids : vecteur des poids des critères
    
    Returns:
    --------
    concordance : matrice de concordance C(a,b)
    """
    n = len(performances)
    m = len(performances[0])
    concordance = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                c_ij = 1  # Diagonale = 1
            else:
                c_ij = 0
                for k in range(m):
                    if performances[i][k] >= performances[j][k]:
                        c_ij += poids[k]
            concordance[i][j] = c_ij
    
    return concordance

def calculer_discordance_electre2(performances):
    """
    Calcule la matrice de discordance selon ELECTRE II
    Pour chaque paire (a,b), on prend la différence maximale normalisée où b > a
    
    Parameters:
    -----------
    performances : matrice des performances
    
    Returns:
    --------
    discordance : matrice de discordance D(a,b)
    """
    n = len(performances)
    m = len(performances[0])
    discordance = np.zeros((n, n))
    
    # Calculer les étendues pour normaliser
    etendues = []
    for k in range(m):
        col = [performances[i][k] for i in range(n)]
        etendue = max(col) - min(col)
        etendues.append(etendue if etendue > 0 else 1.0)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                max_discordance = 0
                for k in range(m):
                    # Si j est meilleur que i sur le critère k
                    if performances[j][k] > performances[i][k]:
                        diff_normalisee = (performances[j][k] - performances[i][k]) / etendues[k]
                        max_discordance = max(max_discordance, diff_normalisee)
                discordance[i][j] = max_discordance
    
    return discordance

def construire_surclassements_electre2(concordance, discordance, c1, d1, c2, d2):
    """
    Construit les graphes de surclassement fort et faible selon ELECTRE II
    
    Parameters:
    -----------
    concordance : matrice de concordance
    discordance : matrice de discordance
    c1, d1 : seuils pour surclassement FORT (c1 = seuil concordance, d1 = seuil discordance)
    c2, d2 : seuils pour surclassement FAIBLE (c2 = seuil concordance, d2 = seuil discordance)
    
    Relations:
    - c1 > c2 (seuil fort plus exigeant)
    - d1 < d2 (seuil fort plus strict)
    
    Returns:
    --------
    S_fort : matrice binaire du surclassement fort S⁺
    S_faible : matrice binaire du surclassement faible S⁻
    """
    n = len(concordance)
    S_fort = np.zeros((n, n), dtype=int)
    S_faible = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                c_ij = concordance[i][j]
                d_ij = discordance[i][j]
                
                # Surclassement FORT : concordance élevée ET discordance faible
                if c_ij >= c1 and d_ij <= d1:
                    S_fort[i][j] = 1
                
                # Surclassement FAIBLE : conditions moins strictes
                if c_ij >= c2 and d_ij <= d2:
                    S_faible[i][j] = 1
    
    return S_fort, S_faible

def deduire_preordre_electre2(S_fort, S_faible, noms_alternatives):
    """
    Déduit un préordre partiel selon ELECTRE II
    
    Relations:
    - a P b (préférence stricte) : a S⁺ b et non b S⁻ a
    - a I b (indifférence) : a S⁺ b et b S⁺ a
    - a R b (incomparabilité) : sinon 
    
    Parameters:
    -----------
    S_fort : matrice de surclassement fort
    S_faible : matrice de surclassement faible
    noms_alternatives : noms des alternatives
    
    Returns:
    --------
    relations : DataFrame des relations entre alternatives
    matrice_relations : matrice des relations (P, I, R)
    scores : scores pour classement
    """
    n = len(S_fort)
    
    # Matrice des relations (0=R incomparable, 1=P préférence, 2=I indifférence)
    matrice_relations = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # a S⁺ b (a surclasse fortement b)
                a_fort_b = S_fort[i][j] == 1
                # b S⁺ a (b surclasse fortement a)
                b_fort_a = S_fort[j][i] == 1
                # a S⁻ b (a surclasse faiblement b)
                a_faible_b = S_faible[i][j] == 1
                # b S⁻ a (b surclasse faiblement a)
                b_faible_a = S_faible[j][i] == 1
                
                # Indifférence : a S⁺ b ET b S⁺ a
                if a_fort_b and b_fort_a:
                    matrice_relations[i][j] = 2  # I
                
                # Préférence stricte : a S⁺ b mais pas b S⁻ a
                elif a_fort_b and not b_faible_a:
                    matrice_relations[i][j] = 1  # P (i préféré à j)
                
                # Incomparabilité : ni a S⁻ b ni b S⁻ a
                elif not a_faible_b and not b_faible_a:
                    matrice_relations[i][j] = 0  # R
    
    # Construire le DataFrame des relations
    relations = []
    for i in range(n):
        for j in range(i+1, n):
            rel_ij = matrice_relations[i][j]
            rel_ji = matrice_relations[j][i]
            
            if rel_ij == 2:  # Indifférence
                relations.append({
                    'Alternative 1': noms_alternatives[i],
                    'Relation': 'I (Indifférence)',
                    'Alternative 2': noms_alternatives[j]
                })
            elif rel_ij == 1:  # i préféré à j
                relations.append({
                    'Alternative 1': noms_alternatives[i],
                    'Relation': 'P (Préférence)',
                    'Alternative 2': noms_alternatives[j]
                })
            elif rel_ji == 1:  # j préféré à i
                relations.append({
                    'Alternative 1': noms_alternatives[j],
                    'Relation': 'P (Préférence)',
                    'Alternative 2': noms_alternatives[i]
                })
            else:  # Incomparabilité
                relations.append({
                    'Alternative 1': noms_alternatives[i],
                    'Relation': 'R (Incomparable)',
                    'Alternative 2': noms_alternatives[j]
                })
    
    df_relations = pd.DataFrame(relations)
    
    # Calculer un score pour le classement (basé sur le nombre de préférences)
    scores = np.zeros(n)
    for i in range(n):
        # Nombre d'alternatives que i domine (surclassement fort)
        domine = S_fort[i, :].sum()
        # Nombre d'alternatives qui dominent i
        domine_par = S_fort[:, i].sum()
        # Score net
        scores[i] = domine - domine_par
    
    df_scores = pd.DataFrame({
        'Alternative': noms_alternatives,
        'Score': scores,
        'Domine': [int(S_fort[i, :].sum()) for i in range(n)],
        'Dominé_par': [int(S_fort[:, i].sum()) for i in range(n)]
    }).sort_values('Score', ascending=False).reset_index(drop=True)
    df_scores['Rang'] = range(1, n + 1)
    
    return df_relations, matrice_relations, df_scores

def construire_noyau_electre2(S_fort, S_faible, noms_alternatives):
    """
    Construit le noyau (ensemble des alternatives non dominées) selon ELECTRE II
    
    Parameters:
    -----------
    S_fort : matrice de surclassement fort
    S_faible : matrice de surclassement faible
    noms_alternatives : noms des alternatives
    
    Returns:
    --------
    noyau : liste des alternatives dans le noyau
    est_dans_noyau : vecteur binaire indiquant si chaque alternative est dans le noyau
    """
    n = len(S_fort)
    est_dans_noyau = np.ones(n, dtype=int)
    
    for i in range(n):
        # Une alternative est hors du noyau si elle est fortement dominée par au moins une autre
        for j in range(n):
            if i != j and S_fort[j][i] == 1 and S_fort[i][j] == 0:
                est_dans_noyau[i] = 0
                break
    
    noyau = [noms_alternatives[i] for i in range(n) if est_dans_noyau[i] == 1]
    
    return noyau, est_dans_noyau

def creer_graphe_surclassement(surclassement_fort, surclassement_faible, noms_alternatives, df_scores):
    """
    Crée une visualisation du graphe de surclassement avec flèches directionnelles
    
    Parameters:
    -----------
    surclassement_fort : matrice de surclassement fort
    surclassement_faible : matrice de surclassement faible
    noms_alternatives : noms des alternatives
    df_scores : DataFrame avec les scores pour positionner les nœuds
    
    Returns:
    --------
    fig : figure Plotly du graphe avec flèches
    """
    n = len(noms_alternatives)
    
    # Positionner les nœuds selon leur score (verticalement) et circulairement (horizontalement)
    # Créer un mapping des alternatives vers leurs scores
    score_dict = dict(zip(df_scores['Alternative'].values, df_scores['Score'].values))
    scores = [score_dict.get(nom, 0) for nom in noms_alternatives]
    
    # Position verticale basée sur le score (meilleurs en haut)
    max_score = max(scores) if max(scores) != min(scores) else 1
    min_score = min(scores)
    score_range = max_score - min_score if max_score != min_score else 1
    
    # Positions: cercle pour l'horizontal, score pour le vertical
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 2.0
    pos_x = radius * np.cos(angles)
    pos_y = [3 * (score - min_score) / score_range for score in scores]  # Position verticale selon score
    
    # Créer les traces pour les flèches
    fig = go.Figure()
    
    # Annoter les flèches
    annotations = []
    
    for i in range(n):
        for j in range(n):
            if i != j:
                x0, y0 = pos_x[i], pos_y[i]
                x1, y1 = pos_x[j], pos_y[j]
                
                # Calculer le point d'arrivée ajusté (pour ne pas cacher le nœud)
                dx = x1 - x0
                dy = y1 - y0
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Réduire la longueur de la flèche pour ne pas cacher les nœuds
                    factor = 0.85
                    x1_adj = x0 + dx * factor
                    y1_adj = y0 + dy * factor
                else:
                    x1_adj, y1_adj = x1, y1
                
                # Surclassement fort (flèche rouge épaisse)
                if surclassement_fort[i][j] == 1:
                    fig.add_trace(go.Scatter(
                        x=[x0, x1_adj],
                        y=[y0, y1_adj],
                        mode='lines',
                        line=dict(color='red', width=3),
                        hoverinfo='text',
                        hovertext=f'{noms_alternatives[i]} ➜ {noms_alternatives[j]} (Fort)',
                        showlegend=False
                    ))
                    
                    # Ajouter une annotation pour la flèche
                    annotations.append(
                        dict(
                            x=x1_adj,
                            y=y1_adj,
                            ax=x0,
                            ay=y0,
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowwidth=2,
                            arrowcolor='red',
                            opacity=0.8
                        )
                    )
                
                # Surclassement faible (flèche orange fine et pointillée)
                elif surclassement_faible[i][j] == 1:
                    fig.add_trace(go.Scatter(
                        x=[x0, x1_adj],
                        y=[y0, y1_adj],
                        mode='lines',
                        line=dict(color='orange', width=1.5, dash='dash'),
                        hoverinfo='text',
                        hovertext=f'{noms_alternatives[i]} ➜ {noms_alternatives[j]} (Faible)',
                        showlegend=False
                    ))
                    
                    annotations.append(
                        dict(
                            x=x1_adj,
                            y=y1_adj,
                            ax=x0,
                            ay=y0,
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor='orange',
                            opacity=0.6
                        )
                    )
    
    # Créer les nœuds (alternatives)
    # Colorer selon le score
    node_colors = ['#2ecc71' if s > 0 else '#e74c3c' if s < 0 else '#95a5a6' for s in scores]
    
    fig.add_trace(go.Scatter(
        x=pos_x,
        y=pos_y,
        mode='markers+text',
        marker=dict(
            size=40,
            color=node_colors,
            line=dict(color='darkblue', width=2),
            opacity=0.9
        ),
        text=noms_alternatives,
        textposition='middle center',
        textfont=dict(size=11, color='white', family='Arial Black'),
        hoverinfo='text',
        hovertext=[f"{nom}<br>Score: {score}" for nom, score in zip(noms_alternatives, scores)],
        showlegend=False
    ))
    
    # Mise en page
    fig.update_layout(
        title={
            'text': "Graphe de Surclassement ELECTRE II<br><sub>Position verticale = Score (↑ meilleur)</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-3, 3]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            showticklabels=True,
            title="Score (alternatives meilleures en haut)"
        ),
        plot_bgcolor='white',
        height=700,
        annotations=annotations,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Ajouter une légende manuelle
    fig.add_annotation(
        text="🔴 <b>Flèche rouge</b>: Surclassement FORT<br>🟠 <b>Flèche orange</b>: Surclassement FAIBLE<br>🟢 <b>Vert</b>: Score positif | 🔴 <b>Rouge</b>: Score négatif",
        xref="paper", yref="paper",
        x=0.5, y=-0.08,
        showarrow=False,
        font=dict(size=11),
        bgcolor="lightyellow",
        bordercolor="gray",
        borderwidth=1,
        xanchor='center'
    )
    
    return fig

# ============= FONCTIONS DE CHARGEMENT =============
def charger_donnees_excel(file):
    """Charge les données depuis un fichier Excel"""
    try:
        xls = pd.ExcelFile(file)
        donnees = {}
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet_name)
            donnees[sheet_name] = df
        
        return donnees
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel : {e}")
        return None

def charger_donnees_csv(file):
    """Charge les données depuis un fichier CSV"""
    try:
        df = pd.read_csv(file)
        return {"data": df}
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
        return None

# ============= NOUVELLE FONCTION: Extraction intelligente des données =============
def extraire_alternatives_et_criteres(df, orientation='colonnes'):
    """
    Extrait les alternatives, critères et performances depuis un DataFrame
    
    Parameters:
    -----------
    df : DataFrame avec les données
    orientation : 'colonnes' si les critères sont en colonnes, 'lignes' si en lignes
    
    Returns:
    --------
    dict avec 'alternatives', 'criteres', 'performances'
    """
    if orientation == 'colonnes':
        # Les critères sont les noms de colonnes (sauf la première qui contient les noms d'alternatives)
        alternatives = df.iloc[:, 0].tolist()
        criteres = df.columns[1:].tolist()
        performances = df.iloc[:, 1:].values.tolist()
        
    else:  # orientation == 'lignes'
        # Les critères sont dans la première colonne, les alternatives en en-têtes
        criteres = df.iloc[:, 0].tolist()
        alternatives = df.columns[1:].tolist()
        # Transposer pour avoir alternatives en lignes
        performances = df.iloc[:, 1:].values.T.tolist()
    
    return {
        'alternatives': alternatives,
        'criteres': criteres,
        'performances': performances
    }

# ============= PAGE D'ACCUEIL =============
if methode == " Accueil":
    st.markdown("""
    ## Application d'aide à la décision multicritère
    
    Cette application implémente deux méthodes puissantes :
    
    ###  AHP (Analytic Hierarchy Process)
    - Méthode de pondération des critères par comparaisons par paires
    - Calcul automatique des poids
    - Graphique interactif avec drag-and-drop
    - Vérification de la cohérence des jugements
    - Visualisations interactives
    
    ###  ELECTRE II (ÉLimination Et Choix Traduisant la RÉalité)
    - Méthode de classement des alternatives par surclassement fort/faible
    - **Étape 1 : Normalisation** selon les formules mathématiques standards
      - Pour critères à **maximiser**: g*ⱼ(a) = [gⱼ(a) - min] / [max - min]
      - Pour critères à **minimiser**: g*ⱼ(a) = [max - gⱼ(a)] / [max - min]
    - **Étape 2 : Matrice de concordance** (comme ELECTRE I)
    - **Étape 3 : Matrice de discordance** (comme ELECTRE I)
    - **Étape 4 : Deux graphes de surclassement** :
      - **S⁺ (Surclassement FORT)** avec seuils (c₁, d₁) stricts
      - **S⁻ (Surclassement FAIBLE)** avec seuils (c₂, d₂) souples
    - **Étape 5 : Préordre partiel** avec relations :
      - **P** (Préférence stricte) : a S⁺ b et non b S⁻ a
      - **I** (Indifférence) : a S⁺ b et b S⁺ a
      - **R** (Incomparabilité) : sinon
    - **Étape 6 : Construction du noyau** (alternatives non dominées)
    - **Étape 7 : Graphe de surclassement** avec **flèches directionnelles**
    
    ###  Fonctionnalités
    - **Import de données** depuis fichiers (Excel, CSV,...)
    - **Détection** des critères et alternatives
    - **Saisie manuelle** flexible
    - **Graphiques drag-and-drop** pour ajuster les poids
    - **Graphe** pour visualiser les relations de surclassement
    - **Analyse détaillée** des résultats
    
    ---
    
    **Instructions :** 
    1. Allez dans ** Choix des données** pour charger ou saisir vos données
    2. Utilisez ** AHP** pour déterminer les poids de vos critères
    3. **Glissez les barres du graphique** pour ajuster interactivement
    4. Utilisez ** ELECTRE II** pour classer vos alternatives avec surclassement fort/faible
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Workflow recommandé :**
        1.  Choix des données
        2.  AHP (avec drag-and-drop)
        3.  ELECTRE II (surclassement fort/faible)
        4. Analysez les résultats !
        """)
    
    with col2:
        if st.session_state.poids_ahp is not None:
            st.success("""
            ✅ **Poids AHP disponibles !**
            
            Vous avez déjà calculé des poids avec AHP.
            Allez dans ELECTRE II pour les utiliser.
            """)
        else:
            st.warning("""
            ⚠️ **Aucun poids AHP calculé**
            
            Commencez par la méthode AHP pour
            déterminer les poids de vos critères.
            """)
    
    if st.button(" Commencer : Choix des données", type="primary", use_container_width=True):
        st.session_state.page_actuelle = " Choix des données"
        st.rerun()

# ============= PAGE CHOIX DES DONNÉES (AMÉLIORÉE) =============
elif methode == " Choix des données":
    st.header(" Choix du mode de saisie des données")
    
    st.markdown("""
    ### Comment souhaitez-vous fournir vos données ?
    
    Vous avez trois options :
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📁 Import de fichier")
        st.markdown("""
        Chargez vos données depuis :
        - Excel (.xlsx, .xls)
        - CSV (.csv)
        - Formats structurés
        """)
        if st.button("📁 Importer un fichier", use_container_width=True, type="primary"):
            st.session_state.mode_saisie = "fichier"
    
    with col2:
        st.markdown("#### ✍️ Saisie manuelle")
        st.markdown("""
        Entrez vos données
        directement dans
        l'application
        """)
        if st.button("✍️ Saisie manuelle", use_container_width=True, type="secondary"):
            st.session_state.mode_saisie = "manuel"
            st.session_state.page_actuelle = " AHP - Analytic Hierarchy Process"
            st.rerun()
    
    with col3:
        st.markdown("#### ⏭️ Passer")
        st.markdown("""
        Utiliser les données
        déjà saisies ou
        continuer plus tard
        """)
        if st.button("⏭️ Passer cette étape", use_container_width=True):
            st.session_state.page_actuelle = "⚖️ AHP - Analytic Hierarchy Process"
            st.rerun()
    
    st.markdown("---")
    
    # Mode Import de fichier (AMÉLIORÉ)
    if st.session_state.mode_saisie == "fichier":
        st.subheader("📁 Import de fichier")
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier",
            type=['xlsx', 'xls', 'csv', 'txt'],
            help="Formats supportés : Excel (.xlsx, .xls), CSV (.csv), Texte (.txt)"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Fichier chargé : {uploaded_file.name}")
            
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['xlsx', 'xls']:
                donnees = charger_donnees_excel(uploaded_file)
            elif file_extension == 'csv':
                donnees = charger_donnees_csv(uploaded_file)
            elif file_extension == 'txt':
                try:
                    df = pd.read_csv(uploaded_file, sep=None, engine='python')
                    donnees = {"data": df}
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier texte : {e}")
                    donnees = None
            else:
                st.error("Format de fichier non supporté")
                donnees = None
            
            if donnees:
                st.session_state.donnees_chargees = donnees
                
                st.subheader(" Aperçu et Configuration des données chargées")
                
                for sheet_name, df in donnees.items():
                    with st.expander(f" Feuille : {sheet_name}", expanded=True):
                        st.markdown("##### Aperçu des données")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("#####  Configuration de l'orientation des données")
                        
                        st.info("""
                        **Comment sont organisées vos données ?**
                        - **Critères en colonnes** : Chaque colonne représente un critère, chaque ligne une alternative
                        - **Critères en lignes** : Chaque ligne représente un critère, chaque colonne une alternative
                        """)
                        
                        orientation = st.radio(
                            "Orientation des données",
                            options=['colonnes', 'lignes'],
                            index=0,
                            key=f"orient_{sheet_name}",
                            horizontal=True,
                            help="Sélectionnez comment vos données sont organisées"
                        )
                        
                        # Aperçu de l'extraction
                        try:
                            donnees_extraites = extraire_alternatives_et_criteres(df, orientation)
                            
                            col_preview1, col_preview2 = st.columns(2)
                            
                            with col_preview1:
                                st.markdown("** Alternatives détectées:**")
                                st.write(donnees_extraites['alternatives'])
                                st.metric("Nombre", len(donnees_extraites['alternatives']))
                            
                            with col_preview2:
                                st.markdown("** Critères détectés:**")
                                st.write(donnees_extraites['criteres'])
                                st.metric("Nombre", len(donnees_extraites['criteres']))
                            
                            st.markdown("** Matrice des performances:**")
                            df_perf_preview = pd.DataFrame(
                                donnees_extraites['performances'],
                                columns=donnees_extraites['criteres'],
                                index=donnees_extraites['alternatives']
                            )
                            st.dataframe(df_perf_preview.style.background_gradient(cmap="YlGn"), use_container_width=True)
                            
                            st.markdown("---")
                            st.markdown("#####  Utiliser ces données pour :")
                            
                            # MODIFICATION PRINCIPALE: Un seul bouton pour charger dans les deux méthodes
                            if st.button(f" Charger pour AHP et ELECTRE II - {sheet_name}", key=f"both_{sheet_name}", use_container_width=True, type="primary"):
                                # Sauvegarder TOUTES les données pour les deux méthodes
                                st.session_state.alternatives_importees = donnees_extraites['alternatives']
                                st.session_state.criteres_importes = donnees_extraites['criteres']
                                st.session_state.performances_importees = donnees_extraites['performances']
                                st.session_state.orientation_donnees = orientation
                                st.session_state.noms_criteres_ahp = donnees_extraites['criteres']
                                
                                st.success(f"""
                                 **Données chargées pour AHP et ELECTRE II !**
                                - {len(donnees_extraites['alternatives'])} alternatives
                                - {len(donnees_extraites['criteres'])} critères
                                - Prêt pour AHP et ELECTRE II
                                """)
                                st.balloons()
                                time.sleep(1)
                                st.session_state.page_actuelle = "⚖️ AHP - Analytic Hierarchy Process"
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"Erreur lors de l'extraction des données : {e}")
                            st.info("🔍 Vérifiez que votre fichier contient bien une première colonne/ligne avec les noms")
                
                st.markdown("---")
                if st.button("➡️ Continuer vers AHP", type="primary", use_container_width=True):
                    st.session_state.page_actuelle = " AHP - Analytic Hierarchy Process"
                    st.rerun()

# ============= PAGE AHP =============
elif methode == " AHP - Analytic Hierarchy Process":
    st.header(" Méthode AHP")
    
    st.markdown("""
    ### Étape 1️⃣ : Configuration
    Définissez vos critères de décision.
    """)
    
    # Utiliser les critères importés si disponibles
    if st.session_state.criteres_importes:
        st.success(f"✅ {len(st.session_state.criteres_importes)} critères importés depuis le fichier !")
        utiliser_importes = st.checkbox("Utiliser les critères importés", value=True)
        
        if utiliser_importes:
            noms_criteres = st.session_state.criteres_importes
            n_criteres = len(noms_criteres)
            st.info(f"Critères : {', '.join(noms_criteres)}")
        else:
            n_criteres = st.number_input("Nombre de critères", min_value=2, max_value=10, value=4)
            st.subheader("Noms des critères")
            cols = st.columns(min(n_criteres, 4))
            noms_criteres = []
            
            for i in range(n_criteres):
                with cols[i % 4]:
                    nom = st.text_input(f"Critère {i+1}", value=f"Critère {i+1}", key=f"crit_{i}")
                    noms_criteres.append(nom)
    else:
        n_criteres = st.number_input("Nombre de critères", min_value=2, max_value=10, value=4)
        
        st.subheader("Noms des critères")
        cols = st.columns(min(n_criteres, 4))
        noms_criteres = []
        
        for i in range(n_criteres):
            with cols[i % 4]:
                nom = st.text_input(f"Critère {i+1}", value=f"Critère {i+1}", key=f"crit_{i}")
                noms_criteres.append(nom)
    
    st.markdown("---")
    st.markdown("""
    ### Étape 2️⃣ : Comparaisons par paires
    
    **Échelle de Saaty :**
    - **1** = Égale importance | **3** = Importance modérée | **5** = Forte importance
    - **7** = Très forte importance | **9** = Importance extrême
    """)
    
    matrice = np.ones((n_criteres, n_criteres))
    
    st.subheader("🔀 Comparaisons")
    
    for i in range(n_criteres):
        for j in range(i+1, n_criteres):
            st.markdown(f"**Comparaison {i+1}/{j+1}**")
            
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                preference = st.radio(
                    "Quel critère préférez-vous ?",
                    options=[noms_criteres[i], "Égal", noms_criteres[j]],
                    index=1,
                    key=f"pref_{i}_{j}",
                    horizontal=True
                )
            
            with col2:
                intensite = st.number_input(
                    "Intensité (1-9)",
                    min_value=1.0,
                    max_value=9.0,
                    value=1.0,
                    step=0.1,
                    key=f"int_{i}_{j}"
                )
            
            with col3:
                if preference == noms_criteres[i]:
                    valeur = intensite
                    st.info(f"✓ {noms_criteres[i]} est {intensite}x plus important")
                elif preference == noms_criteres[j]:
                    valeur = 1.0 / intensite
                    st.info(f"✓ {noms_criteres[j]} est {intensite}x plus important")
                else:
                    valeur = 1.0
                    st.info(f"✓ Importance égale")
                
                matrice[i][j] = valeur
                matrice[j][i] = 1.0 / valeur
            
            st.markdown("---")
    
    if st.button(" Calculer les poids et la cohérence", type="primary", use_container_width=True):
        st.session_state.matrice_ahp = matrice
        st.markdown("### Étape 3️⃣ : Résultats")
        
        st.subheader(" Matrice de comparaisons complète")
        df_matrice = pd.DataFrame(matrice, columns=noms_criteres, index=noms_criteres)
        st.dataframe(df_matrice.style.format("{:.3f}").background_gradient(cmap="YlOrRd"), use_container_width=True)
        
        poids, lambda_max = calculer_poids_ahp(matrice)
        CI, CR = calculer_coherence(matrice, lambda_max)
        
        st.session_state.poids_ahp = poids
        st.session_state.noms_criteres_ahp = noms_criteres
        st.session_state.poids_ajustes = poids.copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Poids des critères")
            df_poids = pd.DataFrame({
                'Critère': noms_criteres,
                'Poids': poids,
                'Pourcentage': poids * 100
            })
            df_poids = df_poids.sort_values('Poids', ascending=False)
            st.dataframe(df_poids.style.format({'Poids': '{:.4f}', 'Pourcentage': '{:.2f}%'}), use_container_width=True)
            
            fig = px.bar(df_poids, x='Critère', y='Pourcentage', 
                        title="Distribution des poids (%) - Calculés AHP",
                        color='Pourcentage',
                        color_continuous_scale='Viridis')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(" Analyse de cohérence")
            
            st.metric("λ max", f"{lambda_max:.4f}")
            st.metric("CI", f"{CI:.4f}")
            st.metric("CR", f"{CR:.4f}")
            
            if CR < 0.10:
                st.success("✅ Cohérence acceptable (CR < 0.10)")
            elif CR < 0.15:
                st.warning("⚠️ Cohérence marginale")
            else:
                st.error("❌ Cohérence insuffisante")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = CR,
                title = {'text': "Ratio de Cohérence"},
                gauge = {
                    'axis': {'range': [0, 0.20]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.10], 'color': "lightgreen"},
                        {'range': [0.10, 0.15], 'color': "yellow"},
                        {'range': [0.15, 0.20], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.10
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("---")
        
        # ========== GRAPHIQUE DRAG-AND-DROP INTERACTIF ==========
        st.subheader(" Ajustement Interactif des Poids (Drag-and-Drop)")
        
        st.markdown("""
        **✨ Mode interactif activé !**
        
        Trois façons d'ajuster les poids :
        1.  **Cliquez et glissez** les barres du graphique verticalement
        2.  **Entrez des valeurs** dans les champs sous le graphique
        3.  **Utilisez les curseurs** dans l'onglet ci-dessous
        
        Les poids sont **automatiquement normalisés** pour totaliser 100%.
        """)
        
        # Créer le graphique interactif
        html_chart = create_draggable_bar_chart(noms_criteres, st.session_state.poids_ajustes * 100)
        components.html(html_chart, height=800, scrolling=True)
        
        st.markdown("---")
        
        # Alternative avec curseurs
        with st.expander(" Ajustement alternatif avec curseurs"):
            cols_sliders = st.columns(min(n_criteres, 3))
            poids_temp = []
            
            for i in range(n_criteres):
                with cols_sliders[i % 3]:
                    valeur_initiale = float(st.session_state.poids_ajustes[i] * 100)
                    nouveau_poids = st.slider(
                        f"{noms_criteres[i]}",
                        min_value=0.0,
                        max_value=100.0,
                        value=valeur_initiale,
                        step=0.5,
                        key=f"slider_poids_{i}"
                    )
                    poids_temp.append(nouveau_poids)
            
            somme_poids_temp = sum(poids_temp)
            if somme_poids_temp > 0:
                poids_ajustes_norm = np.array([p / somme_poids_temp for p in poids_temp])
            else:
                poids_ajustes_norm = st.session_state.poids_ajustes
            
            st.session_state.poids_ajustes = poids_ajustes_norm
            
            # Affichage des poids ajustés
            df_ajuste = pd.DataFrame({
                'Critère': noms_criteres,
                'Poids': poids_ajustes_norm,
                'Pourcentage': poids_ajustes_norm * 100
            })
            st.dataframe(df_ajuste.style.format({'Poids': '{:.4f}', 'Pourcentage': '{:.2f}%'}), use_container_width=True)
        
        # Comparaison
        st.markdown("###  Comparaison : Poids calculés vs Poids ajustés")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("#### Poids calculés (AHP)")
            df_original = pd.DataFrame({
                'Critère': noms_criteres,
                'Poids': poids,
                'Pourcentage': poids * 100
            }).sort_values('Poids', ascending=False)
            st.dataframe(df_original.style.format({'Poids': '{:.4f}', 'Pourcentage': '{:.2f}%'}), use_container_width=True)
        
        with col_comp2:
            st.markdown("#### Poids ajustés (Interactif)")
            df_ajuste = pd.DataFrame({
                'Critère': noms_criteres,
                'Poids': st.session_state.poids_ajustes,
                'Pourcentage': st.session_state.poids_ajustes * 100
            }).sort_values('Poids', ascending=False)
            st.dataframe(df_ajuste.style.format({'Poids': '{:.4f}', 'Pourcentage': '{:.2f}%'}), use_container_width=True)
        
        # Graphique comparatif
        df_comparaison = pd.DataFrame({
            'Critère': noms_criteres * 2,
            'Type': ['Calculé AHP'] * n_criteres + ['Ajusté'] * n_criteres,
            'Pourcentage': list(poids * 100) + list(st.session_state.poids_ajustes * 100)
        })
        
        fig_comp = px.bar(df_comparaison, x='Critère', y='Pourcentage', color='Type',
                         barmode='group',
                         title="Comparaison : AHP vs Ajusté",
                         color_discrete_map={'Calculé AHP': '#636EFA', 'Ajusté': '#EF553B'})
        fig_comp.update_layout(height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Boutons d'action
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("✅ Utiliser les poids ajustés", type="primary", use_container_width=True):
                st.session_state.poids_ahp = st.session_state.poids_ajustes
                st.success("✅ Poids ajustés sauvegardés !")
                st.balloons()
        
        with col_btn2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.poids_ajustes = poids.copy()
                st.session_state.poids_ahp = poids
                st.info("🔄 Réinitialisé aux valeurs AHP")
                st.rerun()
        
        with col_btn3:
            if st.button("➡️ ELECTRE II", type="secondary", use_container_width=True):
                st.session_state.page_actuelle = " ELECTRE II"
                st.rerun()

# ============= PAGE ELECTRE II (AMÉLIORÉE) =============
elif methode == " ELECTRE II":
    st.header(" Méthode ELECTRE II - Surclassement Fort et Faible")
    
    st.info("""
    ** ELECTRE II - Procédure avec normalisation mathématique :**
    1. **Normalisation** selon les formules standards :
       - **Maximiser** : g*ⱼ(a) = [gⱼ(a) - minᵦ gⱼ(b)] / [maxᵦ gⱼ(b) - minᵦ gⱼ(b)]
       - **Minimiser** : g*ⱼ(a) = [maxᵦ gⱼ(b) - gⱼ(a)] / [maxᵦ gⱼ(b) - minᵦ gⱼ(b)]
    2. **Calcul des indices** de concordance et discordance
    3. **Construction de deux graphes** de surclassement (S⁺ fort et S⁻ faible)
    4. **Déduction d'un préordre partiel** avec relations P, I, R
    5. **Graphe avec flèches directionnelles** pour visualiser les relations
    """)
    
    # Vérifier si données importées disponibles
    donnees_importees_disponibles = (
        st.session_state.alternatives_importees is not None and
        st.session_state.criteres_importes is not None and
        st.session_state.performances_importees is not None
    )
    
    if donnees_importees_disponibles:
        st.success(f"""
        ✅ **Données importées détectées !**
        - {len(st.session_state.alternatives_importees)} alternatives
        - {len(st.session_state.criteres_importes)} critères
        """)
        utiliser_donnees_importees = st.checkbox("Utiliser les données importées", value=True)
    else:
        utiliser_donnees_importees = False
    
    # Vérifier si poids AHP disponibles
    if st.session_state.poids_ahp is not None:
        st.success(f"✅ Poids AHP détectés pour {len(st.session_state.noms_criteres_ahp)} critères !")
        utiliser_ahp = st.checkbox("Utiliser les poids calculés avec AHP", value=True)
    else:
        st.info("ℹ️ Utilisez d'abord AHP pour calculer les poids des critères.")
        utiliser_ahp = False
    
    st.markdown("---")
    st.markdown("### Étape 1️⃣ : Configuration")
    
    # Configuration selon les données disponibles
    if utiliser_donnees_importees:
        # Utiliser les données importées
        noms_alternatives = st.session_state.alternatives_importees
        noms_criteres = st.session_state.criteres_importes
        performances = st.session_state.performances_importees
        
        n_alternatives = len(noms_alternatives)
        n_criteres = len(noms_criteres)
        
        st.info(f"""
        **Données chargées depuis le fichier :**
        - Alternatives : {', '.join(noms_alternatives)}
        - Critères : {', '.join(noms_criteres)}
        """)
        
        # Afficher la matrice des performances
        st.markdown("####  Matrice des performances importées")
        df_perf_import = pd.DataFrame(
            performances,
            columns=noms_criteres,
            index=noms_alternatives
        )
        st.dataframe(df_perf_import.style.background_gradient(cmap="YlGn"), use_container_width=True)
        
    else:
        # Saisie manuelle - SUPPRESSION DE LA LIMITE max_value
        col1, col2 = st.columns(2)
        
        with col1:
            n_alternatives = st.number_input("Nombre d'alternatives", min_value=2, value=4)  # Pas de max_value
        
        with col2:
            if utiliser_ahp:
                n_criteres = len(st.session_state.noms_criteres_ahp)
                st.info(f"Nombre de critères (depuis AHP) : {n_criteres}")
            else:
                n_criteres = st.number_input("Nombre de critères", min_value=2, max_value=10, value=3, key="electre_crit")
        
        # Noms des alternatives
        st.subheader("Noms des alternatives")
        cols = st.columns(min(n_alternatives, 4))
        noms_alternatives = []
        
        for i in range(n_alternatives):
            with cols[i % 4]:
                nom = st.text_input(f"Alternative {i+1}", value=f"Option {i+1}", key=f"alt_{i}")
                noms_alternatives.append(nom)
        
        # Noms des critères
        if utiliser_ahp:
            st.subheader("Critères (depuis AHP)")
            noms_criteres = st.session_state.noms_criteres_ahp
            st.info(f"Critères : {', '.join(noms_criteres)}")
        else:
            st.subheader("Noms des critères")
            cols = st.columns(min(n_criteres, 4))
            noms_criteres = []
            
            for i in range(n_criteres):
                with cols[i % 4]:
                    nom = st.text_input(f"Critère {i+1}", value=f"Critère {i+1}", key=f"crit_e_{i}")
                    noms_criteres.append(nom)
    
    st.markdown("---")
    
    # Sens d'optimisation
    st.markdown("### Étape 2️⃣ : Sens d'optimisation des critères")
    st.info("📌 Indiquez si chaque critère doit être **maximisé** (plus c'est mieux) ou **minimisé** (moins c'est mieux)")
    
    cols_sens = st.columns(min(n_criteres, 4))
    sens_optimisation = []
    
    for i in range(n_criteres):
        with cols_sens[i % 4]:
            sens = st.radio(
                f"{noms_criteres[i]}",
                options=['max', 'min'],
                index=0,
                key=f"sens_{i}",
                horizontal=True
            )
            sens_optimisation.append(sens)
    
    st.markdown("---")
    
    # Poids des critères
    st.markdown("### Étape 3️⃣ : Poids des critères")
    
    if utiliser_ahp:
        poids = st.session_state.poids_ahp.tolist()
        
        df_poids_display = pd.DataFrame({
            'Critère': noms_criteres,
            'Poids': poids,
            'Pourcentage': [p*100 for p in poids]
        })
        st.dataframe(df_poids_display.style.format({'Poids': '{:.4f}', 'Pourcentage': '{:.2f}%'}), use_container_width=True)
        
        st.success("✅ Poids importés depuis AHP !")
        
    else:
        st.info("ℹ️ Entrez manuellement les poids ou utilisez AHP.")
        
        cols = st.columns(n_criteres)
        poids = []
        
        for i in range(n_criteres):
            with cols[i]:
                p = st.number_input(
                    f"Poids {noms_criteres[i]}", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=1.0/n_criteres,
                    step=0.01,
                    key=f"poids_{i}"
                )
                poids.append(p)
        
        somme_poids = sum(poids)
        poids = [p/somme_poids for p in poids]
        st.info(f"✓ Poids normalisés : {[f'{p:.3f}' for p in poids]}")
    
    st.markdown("---")
    
    # Performances (seulement si pas importées)
    if not utiliser_donnees_importees:
        st.markdown("### Étape 4️⃣ : Performances des alternatives")
        st.subheader("Tableau des performances (valeurs brutes)")
        
        performances = []
        
        for i in range(n_alternatives):
            st.markdown(f"**{noms_alternatives[i]}**")
            cols = st.columns(n_criteres)
            perf_alt = []
            
            for j in range(n_criteres):
                with cols[j]:
                    val = st.number_input(
                        noms_criteres[j],
                        value=50.0,
                        step=1.0,
                        key=f"perf_{i}_{j}"
                    )
                    perf_alt.append(val)
            
            performances.append(perf_alt)
        
        st.markdown("---")
    
    # Seuils ELECTRE II (2 couples)
    st.markdown("### Étape 5️⃣ : Seuils de surclassement ELECTRE II")
    
    st.info("""
    **📌 Deux couples de seuils :**
    - **(c₁, d₁)** : Seuils pour le **surclassement FORT** (S⁺)
      - **c₁** : seuil de concordance élevé (ex: 0.75)
      - **d₁** : seuil de discordance strict (ex: 0.20)
    
    - **(c₂, d₂)** : Seuils pour le **surclassement FAIBLE** (S⁻)
      - **c₂** : seuil de concordance modéré (ex: 0.60)
      - **d₂** : seuil de discordance souple (ex: 0.35)
    
    **Relations** : c₁ > c₂ et d₁ < d₂
    """)
    
    col_seuil1, col_seuil2 = st.columns(2)
    
    with col_seuil1:
        st.markdown("#### 🔴 Surclassement FORT (S⁺)")
        st.markdown("Conditions strictes")
        
        c1 = st.slider(
            "Seuil de concordance c₁", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.75,
            step=0.05,
            key="c1",
            help="Concordance minimale pour surclassement fort"
        )
        
        d1 = st.slider(
            "Seuil de discordance d₁", 
            min_value=0.0, 
            max_value=0.5, 
            value=0.20,
            step=0.05,
            key="d1",
            help="Discordance maximale pour surclassement fort"
        )
    
    with col_seuil2:
        st.markdown("#### 🟠 Surclassement FAIBLE (S⁻)")
        st.markdown("Conditions souples")
        
        c2 = st.slider(
            "Seuil de concordance c₂", 
            min_value=0.3, 
            max_value=1.0, 
            value=0.60,
            step=0.05,
            key="c2",
            help="Concordance minimale pour surclassement faible"
        )
        
        d2 = st.slider(
            "Seuil de discordance d₂", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.35,
            step=0.05,
            key="d2",
            help="Discordance maximale pour surclassement faible"
        )
    
    # Validation des seuils
    if c1 <= c2:
        st.error("⚠️ Le seuil c₁ doit être strictement supérieur à c₂ (surclassement fort plus exigeant)")
    
    if d1 >= d2:
        st.error("⚠️ Le seuil d₁ doit être strictement inférieur à d₂ (surclassement fort plus strict)")
    
    st.markdown("---")
    
    # CALCUL ELECTRE II
    if st.button(" Calculer le classement ELECTRE II", type="primary", use_container_width=True):
        
        if c1 <= c2 or d1 >= d2:
            st.error("❌ Veuillez corriger les seuils : c₁ > c₂ et d₁ < d₂")
        else:
            # Étape 1 : Normalisation AVEC NOUVELLES FORMULES
            st.markdown("---")
            st.markdown("##  RÉSULTATS ELECTRE II")
            
            st.subheader("1️⃣ Normalisation des performances")
            
            st.info("""
            **Formules de normalisation appliquées :**
            - **Pour critères à maximiser** : g*ⱼ(a) = [gⱼ(a) - min] / [max - min]
            - **Pour critères à minimiser** : g*ⱼ(a) = [max - gⱼ(a)] / [max - min]
            """)
            
            performances_norm = normaliser_performances(performances, sens_optimisation)
            
            col_norm1, col_norm2 = st.columns(2)
            
            with col_norm1:
                st.markdown("**Performances brutes**")
                df_perf_brut = pd.DataFrame(performances, columns=noms_criteres, index=noms_alternatives)
                st.dataframe(df_perf_brut.style.background_gradient(cmap="RdYlGn", axis=0), use_container_width=True)
            
            with col_norm2:
                st.markdown("**Performances normalisées (0-1)**")
                df_perf_norm = pd.DataFrame(performances_norm, columns=noms_criteres, index=noms_alternatives)
                st.dataframe(df_perf_norm.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=0), use_container_width=True)
                st.caption(" Valeurs normalisées : 1 = meilleur, 0 = moins bon (pour tous les critères)")
            
            st.markdown("---")
            
            # Étape 2 : Concordance
            st.subheader("2️⃣ Matrice de concordance")
            st.info("Pour chaque paire (a,b), C(a,b) = somme des poids des critères où a ≥ b")
            
            concordance = calculer_concordance_electre2(performances_norm, poids)
            
            df_conc = pd.DataFrame(concordance, columns=noms_alternatives, index=noms_alternatives)
            st.dataframe(df_conc.style.format("{:.3f}").background_gradient(cmap="Greens"), use_container_width=True)
            
            st.markdown("**Interprétation :** C(a,b) représente le degré de concordance du surclassement de a sur b")
            
            st.markdown("---")
            
            # Étape 3 : Discordance
            st.subheader("3️⃣ Matrice de discordance")
            st.info("Pour chaque paire (a,b), D(a,b) = max des différences normalisées où b > a")
            
            discordance = calculer_discordance_electre2(performances_norm)
            
            df_disc = pd.DataFrame(discordance, columns=noms_alternatives, index=noms_alternatives)
            st.dataframe(df_disc.style.format("{:.3f}").background_gradient(cmap="Reds"), use_container_width=True)
            
            st.markdown("**Interprétation :** D(a,b) mesure l'opposition maximale au surclassement de a sur b")
            
            st.markdown("---")
            
            # Étape 4 : Construction des surclassements
            st.subheader("4️⃣ Graphes de surclassement FORT et FAIBLE")
            
            S_fort, S_faible = construire_surclassements_electre2(concordance, discordance, c1, d1, c2, d2)
            
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("#### 🔴 Surclassement FORT (S⁺)")
                st.markdown(f"**Conditions** : C(a,b) ≥ {c1} ET D(a,b) ≤ {d1}")
                df_s_fort = pd.DataFrame(S_fort, columns=noms_alternatives, index=noms_alternatives)
                st.dataframe(df_s_fort.style.background_gradient(cmap="Reds"), use_container_width=True)
                
                nb_fort = S_fort.sum()
                st.metric("Nombre de relations fortes", int(nb_fort))
            
            with col_s2:
                st.markdown("#### 🟠 Surclassement FAIBLE (S⁻)")
                st.markdown(f"**Conditions** : C(a,b) ≥ {c2} ET D(a,b) ≤ {d2}")
                df_s_faible = pd.DataFrame(S_faible, columns=noms_alternatives, index=noms_alternatives)
                st.dataframe(df_s_faible.style.background_gradient(cmap="Oranges"), use_container_width=True)
                
                nb_faible = S_faible.sum()
                st.metric("Nombre de relations faibles", int(nb_faible))
            
            st.info("""
            **Légende :**
            - **1** = relation de surclassement existe (a surclasse b)
            - **0** = pas de relation
            - Le surclassement fort est **plus exigeant** que le surclassement faible
            """)
            
            st.markdown("---")
            
            # Étape 5 : Préordre partiel
            st.subheader("5️⃣ Préordre partiel et relations")
            
            df_relations, matrice_relations, df_scores = deduire_preordre_electre2(S_fort, S_faible, noms_alternatives)
            
            tab1, tab2, tab3 = st.tabs([" Relations", " Classement", " Noyau"])
            
            with tab1:
                st.markdown("**Relations entre alternatives :**")
                st.markdown("""
                - **P (Préférence)** : a S⁺ b mais pas b S⁻ a → a est strictement préféré à b
                - **I (Indifférence)** : a S⁺ b ET b S⁺ a → a et b sont équivalents
                - **R (Incomparabilité)** : ni a S⁻ b ni b S⁻ a → a et b sont incomparables
                """)
                
                if len(df_relations) > 0:
                    st.dataframe(df_relations, use_container_width=True)
                else:
                    st.info("Aucune relation identifiée")
            
            with tab2:
                st.markdown("**Classement basé sur le score net (Domine - Dominé par) :**")
                st.dataframe(df_scores, use_container_width=True)
                
                fig_scores = px.bar(df_scores, x='Alternative', y='Score',
                                   title="Scores de surclassement net",
                                   color='Score',
                                   color_continuous_scale='RdYlGn',
                                   text='Score')
                fig_scores.update_traces(texttemplate='%{text}', textposition='outside')
                fig_scores.update_layout(height=500)
                st.plotly_chart(fig_scores, use_container_width=True)
                
                meilleure = df_scores.iloc[0]
                st.success(f"""
                ###  **Meilleure alternative : {meilleure['Alternative']}**
                - Score net : {int(meilleure['Score'])}
                - Domine : {int(meilleure['Domine'])} alternatives
                - Dominé par : {int(meilleure['Dominé_par'])} alternatives
                """)
            
            with tab3:
                st.markdown("**Noyau (alternatives non dominées) :**")
                st.info("Le noyau contient les alternatives qui ne sont fortement dominées par aucune autre")
                
                noyau, est_dans_noyau = construire_noyau_electre2(S_fort, S_faible, noms_alternatives)
                
                if len(noyau) > 0:
                    st.success(f"**Alternatives dans le noyau :** {', '.join(noyau)}")
                    
                    df_noyau = pd.DataFrame({
                        'Alternative': noms_alternatives,
                        'Dans_le_noyau': [' Oui' if est_dans_noyau[i] == 1 else ' Non' for i in range(len(noms_alternatives))]
                    })
                    st.dataframe(df_noyau, use_container_width=True)
                else:
                    st.warning("Aucune alternative dans le noyau")
            
            st.markdown("---")
            
            # Étape 6 : Graphe de surclassement AVEC FLÈCHES
            st.subheader("6️⃣ Graphe de surclassement avec flèches directionnelles")
            fig_graphe = creer_graphe_surclassement(S_fort, S_faible, noms_alternatives, df_scores)
            st.plotly_chart(fig_graphe, use_container_width=True)
            
            st.info("""
            **Comment lire le graphe :**
            - **Flèches rouges épaisses** → Surclassement FORT (a S⁺ b : a domine fortement b)
            - **Flèches oranges pointillées** → Surclassement FAIBLE (a S⁻ b : a domine faiblement b)
            - **Position verticale** → Basée sur le score net (meilleures alternatives en haut)
            - **Couleur des nœuds** :
              - 🟢 Vert = Score positif (domine plus qu'elle n'est dominée)
              - 🔴 Rouge = Score négatif (dominée plus qu'elle ne domine)
              - ⚪ Gris = Score neutre
            """)
            
            st.markdown("---")
            
            # Résumé final
            st.markdown("###  Résumé ELECTRE II")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Relations fortes (S⁺)", int(nb_fort))
                st.metric("Relations faibles (S⁻)", int(nb_faible))
            
            with col_res2:
                nb_preferences = len(df_relations[df_relations['Relation'] == 'P (Préférence)'])
                nb_indifferences = len(df_relations[df_relations['Relation'] == 'I (Indifférence)'])
                nb_incomparables = len(df_relations[df_relations['Relation'] == 'R (Incomparable)'])
                
                st.metric("Préférences (P)", nb_preferences)
                st.metric("Indifférences (I)", nb_indifferences)
                st.metric("Incomparabilités (R)", nb_incomparables)
            
            with col_res3:
                st.metric("Taille du noyau", len(noyau))
                if len(noyau) > 0:
                    st.info(f"**Noyau** : {', '.join(noyau[:3])}" + ("..." if len(noyau) > 3 else ""))
            
            st.markdown("---")
            
            # ============= EXPORT DES RÉSULTATS =============
            st.markdown("###  Export des résultats")
            
            st.info(" Téléchargez les résultats de votre analyse ELECTRE II dans différents formats")
            
            tab_export1, tab_export2, tab_export3 = st.tabs([" Excel", " Rapport TXT", " JSON"])
            
            with tab_export1:
                st.markdown("#### Export Excel complet")
                st.markdown("Génère un fichier Excel avec toutes les matrices et résultats sur différentes feuilles")
                
                # Créer le fichier Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Feuille 1: Données brutes
                    df_perf_brut.to_excel(writer, sheet_name='Performances_Brutes', index=True)
                    
                    # Feuille 2: Performances normalisées
                    df_perf_norm.to_excel(writer, sheet_name='Performances_Normalisées', index=True)
                    
                    # Feuille 3: Poids des critères
                    df_poids_export = pd.DataFrame({
                        'Critère': noms_criteres,
                        'Poids': poids,
                        'Pourcentage (%)': [p*100 for p in poids],
                        'Sens': sens_optimisation
                    })
                    df_poids_export.to_excel(writer, sheet_name='Poids_Critères', index=False)
                    
                    # Feuille 4: Matrice de concordance
                    df_conc.to_excel(writer, sheet_name='Concordance', index=True)
                    
                    # Feuille 5: Matrice de discordance
                    df_disc.to_excel(writer, sheet_name='Discordance', index=True)
                    
                    # Feuille 6: Surclassement fort
                    df_s_fort.to_excel(writer, sheet_name='Surclassement_Fort', index=True)
                    
                    # Feuille 7: Surclassement faible
                    df_s_faible.to_excel(writer, sheet_name='Surclassement_Faible', index=True)
                    
                    # Feuille 8: Relations
                    if len(df_relations) > 0:
                        df_relations.to_excel(writer, sheet_name='Relations', index=False)
                    
                    # Feuille 9: Classement final
                    df_scores.to_excel(writer, sheet_name='Classement', index=False)
                    
                    # Feuille 10: Noyau
                    df_noyau.to_excel(writer, sheet_name='Noyau', index=False)
                    
                    # Feuille 11: Paramètres
                    df_params = pd.DataFrame({
                        'Paramètre': ['Seuil concordance fort (c1)', 'Seuil discordance fort (d1)', 
                                     'Seuil concordance faible (c2)', 'Seuil discordance faible (d2)',
                                     'Nombre alternatives', 'Nombre critères'],
                        'Valeur': [c1, d1, c2, d2, n_alternatives, n_criteres]
                    })
                    df_params.to_excel(writer, sheet_name='Paramètres', index=False)
                
                output.seek(0)
                
                st.download_button(
                    label=" Télécharger Excel (.xlsx)",
                    data=output,
                    file_name="resultats_electre2.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )
                
                st.success("""
                 **Contenu du fichier Excel :**
                - Performances brutes et normalisées
                - Poids des critères
                - Matrices de concordance et discordance
                - Surclassements fort et faible
                - Relations et classement
                - Noyau et paramètres
                """)
            
            with tab_export2:
                st.markdown("#### Export Rapport (TXT)")
                st.markdown("Génère un rapport texte complet avec tableaux et résumé")
                
                # Créer un rapport texte pour PDF
                rapport_text = f"""
═══════════════════════════════════════════════════════════
    RAPPORT D'ANALYSE MULTICRITÈRE - ELECTRE II
═══════════════════════════════════════════════════════════

Date de génération: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────────────────
1. PARAMÈTRES DE L'ANALYSE
─────────────────────────────────────────────────────────────

Nombre d'alternatives: {n_alternatives}
Nombre de critères: {n_criteres}

Seuils ELECTRE II:
  • Surclassement FORT:  c₁ = {c1:.2f}, d₁ = {d1:.2f}
  • Surclassement FAIBLE: c₂ = {c2:.2f}, d₂ = {d2:.2f}

─────────────────────────────────────────────────────────────
2. CRITÈRES ET POIDS
─────────────────────────────────────────────────────────────

{df_poids_export.to_string(index=False)}

─────────────────────────────────────────────────────────────
3. PERFORMANCES DES ALTERNATIVES
─────────────────────────────────────────────────────────────

3.1 Performances brutes:
{df_perf_brut.to_string()}

3.2 Performances normalisées (0-1):
{df_perf_norm.to_string()}

─────────────────────────────────────────────────────────────
4. MATRICES DE CONCORDANCE ET DISCORDANCE
─────────────────────────────────────────────────────────────

4.1 Matrice de concordance:
{df_conc.to_string()}

4.2 Matrice de discordance:
{df_disc.to_string()}

─────────────────────────────────────────────────────────────
5. SURCLASSEMENTS
─────────────────────────────────────────────────────────────

5.1 Surclassement FORT (S⁺):
{df_s_fort.to_string()}

Nombre de relations fortes: {int(nb_fort)}

5.2 Surclassement FAIBLE (S⁻):
{df_s_faible.to_string()}

Nombre de relations faibles: {int(nb_faible)}

─────────────────────────────────────────────────────────────
6. RELATIONS ET CLASSEMENT
─────────────────────────────────────────────────────────────

6.1 Relations entre alternatives:
{df_relations.to_string(index=False) if len(df_relations) > 0 else "Aucune relation identifiée"}

6.2 Classement final:
{df_scores.to_string(index=False)}

─────────────────────────────────────────────────────────────
7. NOYAU (ALTERNATIVES NON DOMINÉES)
─────────────────────────────────────────────────────────────

Alternatives dans le noyau: {', '.join(noyau) if len(noyau) > 0 else 'Aucune'}

{df_noyau.to_string(index=False)}

─────────────────────────────────────────────────────────────
8. CONCLUSION
─────────────────────────────────────────────────────────────

 MEILLEURE ALTERNATIVE: {meilleure['Alternative']}
   • Score net: {int(meilleure['Score'])}
   • Domine: {int(meilleure['Domine'])} alternatives
   • Dominé par: {int(meilleure['Dominé_par'])} alternatives

Statistiques des relations:
   • Préférences (P): {nb_preferences}
   • Indifférences (I): {nb_indifferences}
   • Incomparabilités (R): {nb_incomparables}

═══════════════════════════════════════════════════════════
Analyse réalisée avec MCDA App - AHP & ELECTRE II
Développé par: SERHANE Firdaws, BOURCHAK Mehdi, TOURATI Ahmed
Encadré par: Pr. BOUANANI Hicham
═══════════════════════════════════════════════════════════
"""
                
                st.download_button(
                    label=" Télécharger Rapport (.txt)",
                    data=rapport_text,
                    file_name="rapport_electre2.txt",
                    mime="text/plain",
                    use_container_width=True,
                    type="primary"
                )
                
                st.info("💡 **Note**: Le fichier .txt peut être converti en PDF avec n'importe quel éditeur de texte ou outil en ligne")
            
            with tab_export3:
                st.markdown("#### Export JSON (données structurées)")
                st.markdown("Format JSON pour intégration avec d'autres outils ou scripts Python")
                
                # Créer un dictionnaire avec tous les résultats
                resultats_json = {
                    "metadata": {
                        "date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "methode": "ELECTRE II",
                        "n_alternatives": n_alternatives,
                        "n_criteres": n_criteres
                    },
                    "parametres": {
                        "seuils": {
                            "c1": c1,
                            "d1": d1,
                            "c2": c2,
                            "d2": d2
                        }
                    },
                    "criteres": {
                        "noms": noms_criteres,
                        "poids": poids,
                        "sens_optimisation": sens_optimisation
                    },
                    "alternatives": {
                        "noms": noms_alternatives,
                        "performances_brutes": performances,
                        "performances_normalisees": performances_norm.tolist()
                    },
                    "matrices": {
                        "concordance": concordance.tolist(),
                        "discordance": discordance.tolist(),
                        "surclassement_fort": S_fort.tolist(),
                        "surclassement_faible": S_faible.tolist()
                    },
                    "resultats": {
                        "classement": df_scores.to_dict('records'),
                        "relations": df_relations.to_dict('records') if len(df_relations) > 0 else [],
                        "noyau": noyau,
                        "meilleure_alternative": {
                            "nom": meilleure['Alternative'],
                            "score": int(meilleure['Score']),
                            "domine": int(meilleure['Domine']),
                            "domine_par": int(meilleure['Dominé_par'])
                        }
                    },
                    "statistiques": {
                        "nb_relations_fortes": int(nb_fort),
                        "nb_relations_faibles": int(nb_faible),
                        "nb_preferences": nb_preferences,
                        "nb_indifferences": nb_indifferences,
                        "nb_incomparables": nb_incomparables,
                        "taille_noyau": len(noyau)
                    }
                }
                
                json_string = json.dumps(resultats_json, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label=" Télécharger JSON (.json)",
                    data=json_string,
                    file_name="resultats_electre2.json",
                    mime="application/json",
                    use_container_width=True,
                    type="primary"
                )
                
                st.success("✅ **Format JSON** idéal pour :")
                st.markdown("""
                - Intégration avec Python/R
                - Archivage des résultats
                - Comparaison avec d'autres analyses
                - Traitement automatisé
                """)
                
                # Aperçu du JSON
                with st.expander(" Aperçu du JSON"):
                    st.json(resultats_json)
            
            st.markdown("---")
            
            # Export du graphique
            st.markdown("####  Export du graphique de surclassement")
            
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                # Export PNG
                try:
                    img_bytes_png = fig_graphe.to_image(format="png", width=1200, height=800, scale=2)
                    st.download_button(
                        label=" Télécharger Graphique (.png)",
                        data=img_bytes_png,
                        file_name="graphe_surclassement.png",
                        mime="image/png",
                        use_container_width=True
                    )
                except:
                    st.warning("⚠️ Export PNG nécessite kaleido: `pip install kaleido`")
            
            with col_graph2:
                # Export HTML interactif
                html_bytes = fig_graphe.to_html(include_plotlyjs='cdn').encode()
                st.download_button(
                    label=" Télécharger Graphique Interactif (.html)",
                    data=html_bytes,
                    file_name="graphe_surclassement.html",
                    mime="text/html",
                    use_container_width=True
                )

st.markdown("---")
st.markdown("💡 **Astuce** : Importez vos données → AHP  → ELECTRE II ")
st.markdown("Développé avec SERHANE Firdaws - BOURCHAK Mehdi - TOURATI Ahmed | Encadré par Pr. BOUANANI Hicham | 2025-2026")
st.markdown("firdaws.serh@gmail.com | mhdbourchak@gmail.com | ahmedtourati7@gmail.com")
