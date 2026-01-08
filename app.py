import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go

# --- MOTEUR DE CALCUL (Modèle Black-Scholes) ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calcule la valeur théorique d'une option européenne."""
    if T <= 1e-6: # Cas de l'expiration
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == "call":
        return S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(page_title="Option Strategy Visualizer", layout="wide")
st.title("📊 Analyseur de Stratégies d'Options Multi-Jambes")
st.markdown("Visualisez l'impact du temps (Theta) et de la volatilité sur vos positions.")

# --- SIDEBAR : PARAMÈTRES MARCHÉ ---
with st.sidebar:
    st.header("⚙️ Paramètres Marché")
    S0 = st.number_input("Prix actuel du Sous-jacent", value=100.0, step=1.0)
    vol = st.slider("Volatilité Implicite (%)", 5, 100, 25) / 100
    rate = st.slider("Taux d'intérêt (%)", 0.0, 10.0, 2.0) / 100
    
    st.divider()
    st.header("⏳ Temps & Échéance")
    total_days = st.number_input("Durée totale du trade (jours)", value=45, min_value=1)
    days_passed = st.slider("Jours écoulés", 0, int(total_days), 0)
    
    T_init = total_days / 365
    T_current = (total_days - days_passed) / 365

# --- CONSTRUCTION DE LA STRATÉGIE (JUSQU'À 4 JAMBES) ---
st.subheader("🛠️ Configuration des Jambes (Legs)")
legs = []
cols = st.columns(4)

for i in range(4):
    with cols[i]:
        st.markdown(f"**Jambe {i+1}**")
        is_active = st.checkbox("Activer", value=(i==0), key=f"active_{i}")
        if is_active:
            side = st.selectbox("Position", ["Achat", "Vente"], key=f"side_{i}")
            opt_type = st.selectbox("Type", ["Call", "Put"], key=f"type_{i}")
            strike = st.number_input("Strike", value=100.0 + (i*2 if i<2 else -i*2), key=f"strike_{i}")
            premium = st.number_input("Prix payé/reçu", value=2.0, key=f"premium_{i}")
            
            # Stockage des paramètres
            qty = 1 if side == "Achat" else -1
            legs.append({
                "type": opt_type.lower(),
                "k": strike,
                "p": premium,
                "q": qty
            })

# --- FONCTION DE CALCUL DU PNL GLOBAL ---
def get_strategy_pnl(S_vec, t_remaining, is_expiry=False):
    total_pnl = np.zeros_like(S_vec)
    for leg in legs:
        leg_pnl = []
        for s in S_vec:
            if is_expiry:
                val = max(0, s - leg['k']) if leg['type'] == "call" else max(0, leg['k'] - s)
            else:
                val = black_scholes(s, leg['k'], t_remaining, rate, vol, leg['type'])
            
            # PnL = (Valeur de l'option - Coût d'entrée) * Quantité
            leg_pnl.append((val - leg['p']) * leg['q'])
        total_pnl += np.array(leg_pnl)
    return total_pnl

# --- PRÉPARATION DES DONNÉES GRAPHIQUES ---
S_range = np.linspace(S0 * 0.7, S0 * 1.3, 300)

# Courbes fixes (Échéance et Sélection)
pnl_expiry = get_strategy_pnl(S_range, 0, is_expiry=True)
pnl_current = get_strategy_pnl(S_range, T_current)

# Courbes temporelles intermédiaires (10%, 30%, 50%, 80%)
paliers = [0.1, 0.3, 0.5, 0.8]
intermediate_curves = {}
for p in paliers:
    t_step = T_init * (1 - p)
    intermediate_curves[p] = get_strategy_pnl(S_range, t_step)

# --- GÉNÉRATION DU GRAPHIQUE ---
fig = go.Figure()

# 1. Tracés des paliers temporels (finesse pour la lisibilité)
for p, pnl_vals in intermediate_curves.items():
    fig.add_trace(go.Scatter(
        x=S_range, y=pnl_vals, 
        name=f"Temps écoulé : {int(p*100)}%",
        line=dict(width=1, dash='dot'),
        opacity=0.4
    ))

# 2. Tracé à l'échéance (La cible)
fig.add_trace(go.Scatter(
    x=S_range, y=pnl_expiry, 
    name="Valeur à l'échéance", 
    line=dict(color='#ff3860', width=2, dash='dash')
))

# 3. Tracé actuel (La position en direct)
fig.add_trace(go.Scatter(
    x=S_range, y=pnl_current, 
    name="Valeur à T+choisi", 
    line=dict(color='#00d1b2', width=4)
))

# Mise en forme
fig.add_hline(y=0, line_color="white", opacity=0.3)
fig.add_vline(x=S0, line_color="yellow", line_dash="dot", opacity=0.5, annotation_text="Prix Marché")

fig.update_layout(
    height=650,
    template="plotly_dark",
    hovermode="x unified",
    title="Évolution du Profil PnL avec le Temps",
    xaxis_title="Prix du Sous-jacent",
    yaxis_title="Profit / Perte ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- RÉCAPITULATIF FINANCIER ---
st.divider()
col_m1, col_m2, col_m3 = st.columns(3)

net_cost = sum(l['p'] * l['q'] for l in legs)
type_trade = "Crédit reçu" if net_cost < 0 else "Débit payé"

with col_m1:
    st.metric("Coût/Crédit Net", f"{abs(net_cost):.2f} $", delta=type_trade, delta_color="normal")
with col_m2:
    current_val = sum((black_scholes(S0, l['k'], T_current, rate, vol, l['type']) * l['q']) for l in legs)
    st.metric("Valeur Actuelle Totale", f"{current_val:.2f} $")
with col_m3:
    pnl_total = current_val - net_cost
    st.metric("P&L Latent", f"{pnl_total:.2f} $", delta=f"{pnl_total:.2f} $")

st.info("💡 Note : Les courbes en pointillés montrent comment votre profit théorique 'aspire' ou 's'écrase' vers la ligne rouge à mesure que le temps passe (Theta Decay).")
