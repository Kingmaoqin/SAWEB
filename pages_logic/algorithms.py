import streamlit as st
import numpy as np
import plotly.graph_objects as go

def render_mysa_tutorial():
    st.header("TEXGISA — Algorithm Guide & How-To")

    st.markdown(
        """
**What is TEXGISA?**  
TEXGISA is an interpretable and interactive **deep discrete-time survival** framework with **Time-dependent EXtreme Gradient Integration (TEXGI)** for explanations, plus **expert priors** built into training. It aims to deliver accurate **time-to-event** prediction while keeping explanations stable, time-aware, and aligned with domain knowledge. :contentReference[oaicite:0]{index=0}
        """
    )

    st.subheader("1) Discrete-time survival basics")
    st.markdown(
        """
We partition follow-up into `T_max` intervals and predict **hazards** per interval. Survival up to interval *j* is the product of survival factors over previous hazards (Markov formulation):
        """
    )
    st.latex(r"S^{(j)}(x) \;=\; \prod_{k=1}^{j}\big(1 - \hat h_k(x)\big)")
    st.caption("Monotonic survival (non-increasing in j) is guaranteed by the multiplicative form. :contentReference[oaicite:1]{index=1}")

    st.subheader("2) Network design (Dense/Attention/Residual + Hazard head)")
    st.markdown(
        """
The model learns a time-indexed sequence of hazards: Dense blocks extract interactions, optional Table Attention re-weights features, Residual shared layers stabilize deep training, then a hazard head outputs per-interval logits → hazards. This captures complex tabular interactions while preserving time structure. :contentReference[oaicite:2]{index=2}
        """
    )

    st.subheader("3) TEXGI: Time-dependent EXtreme Gradient Integration")
    st.markdown(
        """
**Why TEXGI?** Standard IG depends on the baseline. TEXGI replaces arbitrary/zero baselines with **extreme baselines** sampled from a high-risk tail (e.g., generalized Pareto sampling), then integrates gradients **from extreme → actual input** per time interval, producing **time-specific attributions** that focus on clinically/semantically critical regions. :contentReference[oaicite:3]{index=3}
        """
    )
    st.latex(r"\mathrm{TEXGI}^{(j)}_l \;=\; \int_0^1 \frac{\partial f_{\theta_j}\big(x' + \alpha(x-x')\big)}{\partial x_l}\,(x_l - x'_l)\, d\alpha")
    st.caption("x' is an **extreme baseline** at interval j; the path integral emphasizes high-risk regions. :contentReference[oaicite:4]{index=4}")

    st.subheader("4) Loss: prediction + temporal smoothness + expert priors")
    st.markdown(
        """
The training objective adds two regularizers to the negative log-likelihood (NLL):

- **Temporal smoothness on attributions** (Total Variation over time) keeps explanations stable:  
        """
    )
    st.latex(r"\Omega_{\text{smooth}}(\Phi) \;=\; \sum_{i,l}\sum_{j=1}^{T_{\max}-1} \big|\phi_{i,l}^{(j+1)} - \phi_{i,l}^{(j)}\big|")
    st.caption("Prevents erratic, hard-to-trust fluctuations across adjacent time bins. :contentReference[oaicite:5]{index=5}")

    st.markdown(
        """
- **Expert prior** encourages expert-important features to have **above-average** attribution energy and suppresses unimportant ones:  
        """
    )
    st.latex(
        r"""\Omega_{\text{expert}}(\Phi) \;=\; \lambda_1 \sum_{l'\in I} 
\mathrm{ReLU}\!\left(\frac{1}{p}\sum_{l=1}^p\|\Phi_l\|_2 - \|\Phi_{l'}\|_2\right)
\;+\; \lambda_2 \sum_{l''\notin I}\|\Phi_{l''}\|_2"""
    )
    st.caption("Softly enforces “≥ average” for expert-important features; robust to small expert noise. :contentReference[oaicite:6]{index=6}")

    st.markdown(
        """
**Overall objective** (conceptual):
        """
    )
    st.latex(r"\mathcal{L} \;=\; \mathrm{NLL} \;+\; \lambda_{\text{smooth}}\Omega_{\text{smooth}} \;+\; \Omega_{\text{expert}}")
    st.caption("Balances predictive fit and human-aligned interpretability. :contentReference[oaicite:7]{index=7}")

    st.subheader("5) How to operate in the UI (recommended flow)")
    st.markdown(
        """
1. **Upload & map columns**: choose *duration* (time), *event* (0/1), and *features*.  
2. **Preview FI (no priors)**: run a quick TEXGI preview to see top-k features; sanity-check directions.  
3. **Set priors**: in *Expert Rules*, mark important features (and direction if applicable), adjust per-feature **weight**.  
4. **Tune regularizers**:  
   - `λ_smooth` (temporal smoothness on attributions): typical `0.005–0.05`.  
   - `λ_expert` (overall strength of expert prior panel): start `0.05–0.2`.  
5. **Advanced TEXGI / generator** (optional): adjust `τ` (tail probability), IG `steps (M)`, and extreme generator capacity (fast mode for speed).  
6. **Train with priors**: monitor C-index; use KM plots (overall/groups) and predicted trajectories to validate risk stratification. :contentReference[oaicite:8]{index=8}
        """
    )

    st.subheader("6) Reading model outputs")
    st.markdown(
        """
- **Predicted survival trajectories**: spread & crossings hint at heterogeneity.  
- **KM (overall)**: step-downs at events, flat tails often imply heavy censoring late.  
- **KM (by risk group)**: vertical separation = better discrimination.  
- **Feature Importance (TEXGI Top-k)**: bar length = importance; **color** encodes **direction**  
  (**blue → positive hazard direction → higher risk**, **red → negative hazard direction → lower risk**), based on directional mean.  
  Cross-check with expert priors; misaligned features are candidates for stronger priors or data QA. :contentReference[oaicite:9]{index=9}
        """
    )

    st.subheader("7) Practical tips")
    st.markdown(
        """
- Start with modest intervals, then increase bins if trajectories look too coarse.  
- If FI looks noisy across time, increase `λ_smooth`.  
- If domain-critical features are undervalued, increase `λ_expert` and/or their row weight in the Expert Rules table.  
- For preview speed, enable *fast expert mode*; for final training, use full generator & larger `M` for TEXGI. :contentReference[oaicite:10]{index=10}
        """
    )

    st.info(
        "References: The full method (discrete-time survival, TEXGI, temporal smoothness prior, and expert-"
        "informed regularization) comes from the provided manuscript. See details and equations inside. "
        ":contentReference[oaicite:11]{index=11}"
    )

def show():
    st.title("Advanced Survival Analysis Guide")
    st.caption("You are here: Algorithm Tutor")

    # ====================== Core Concepts ======================
    with st.expander("📚 Core Survival Analysis Concepts", expanded=True):
        st.markdown(r"""
        **Mathematical Foundations:**
        
        | Symbol | Name | Definition | Clinical Interpretation |
        |--------|------|------------|-------------------------|
        | $T$    | Survival Time | Time until event occurrence | Patient's survival duration/progression-free time |
        | $S(t)$ | Survival Function | $P(T > t)$ | Probability of surviving beyond time t |
        | $h(t)$ | Hazard Function | $\lim_{\Delta t \to 0} \frac{P(t \le T < t+\Delta t \mid T \ge t)}{\Delta t}$ | Instantaneous risk rate at time t |
        | $X$    | Covariates | Feature vector $(x_1,x_2,...,x_p)$ | Predictive factors (age, biomarkers, treatment) |
        | $h_0(t)$ | Baseline Hazard | Underlying risk when all covariates are zero | Reference risk level for population |
        """)

        st.markdown("""
        **Clinical Example Context:**
        - **Breast Cancer Study**: T=Time from diagnosis to recurrence, X=(Tumor size, ER status, Treatment regimen)
        - **Cardiology Study**: T=Time from discharge to readmission, X=(BP, Cholesterol levels, Medication adherence)
        """)
        
        # Interactive survival curve generator
        st.subheader("🔬 Survival Curve Simulator")
        col1, col2 = st.columns(2)
        with col1:
            median_survival = st.slider("Median Survival (months)", 6, 60, 24)
        with col2:
            hazard_ratio = st.slider("Hazard Ratio", 0.5, 2.0, 1.0)
        
        t = np.linspace(0, 60, 100)
        # Using a Weibull survival function for simulation
        # scale = median_survival / (np.log(2)**(1/1.5))
        # survival = np.exp(-(t / scale)**1.5)
        # A simpler exponential model for clarity
        rate = np.log(2) / median_survival
        survival_base = np.exp(-rate * t)
        survival_hr = np.exp(-rate * hazard_ratio * t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=survival_base, name='Baseline Survival', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=t, y=survival_hr, name=f'Survival with HR={hazard_ratio}', line=dict(color='firebrick')))
        fig.update_layout(title="Simulated Survival Curve",
                          xaxis_title='Time (months)',
                          yaxis_title='S(t) - Survival Probability')
        st.plotly_chart(fig, use_container_width=True)

    # ====================== Algorithm Selection ======================
    st.markdown("---")
    selected_algo = st.selectbox(
        "🔍 Choose an Algorithm to Explore:",
        options=["Select Algorithm", "CoxTime", "DeepSurv", "DeepHit", "TEXGISA"],
        index=0
    )

    # ====================== Algorithm Details ======================
    if selected_algo != "Select Algorithm":
        with st.expander(f"📖 {selected_algo} Detailed Explanation", expanded=True):
            if selected_algo == "CoxTime":
                st.markdown(r"""
                **Time-Dependent Cox Model with Neural Networks**
                
                **Mathematical Formulation:**
                $$
                h(t|X) = h_0(t) \cdot \exp(\beta(t)X)
                $$
                
                **Symbol Definitions:**
                - $\beta(t)$: Time-varying coefficient vector (learned via neural network)
                - $X$: Patient covariate vector (static features)
                - $h_0(t)$: Non-parametric baseline hazard
                
                **Key Components:**
                1. Temporal Attention Layer: Learns time-dependent feature importance
                2. Hazard Projection: Combines static and temporal effects
                3. Partial Likelihood Optimization: Maximizes:
                   $$
                   L(\theta) = \prod_{i=1}^n \frac{\exp(\beta(t_i)X_i)}{\sum_{j \in R(t_i)} \exp(\beta(t_i)X_j)}
                   $$
                   Where $R(t_i)$ is the risk set at time $t_i$
                
                **Clinical Application:**
                Monitoring dynamic treatment effects in chemotherapy regimens
                """)
                
                # Time-varying coefficients visualization
                st.subheader("⏳ Temporal Coefficient Dynamics")
                time_points = np.linspace(0, 24, 100)
                treatment_effect = 0.8 + 0.2 * np.sin(time_points/3)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_points, y=treatment_effect, 
                                         line=dict(color='firebrick', width=3)))
                fig.update_layout(title="Treatment Effect Over Time",
                                  xaxis_title='Treatment Duration (months)',
                                  yaxis_title='β(t) Coefficient Value',
                                  annotations=[
                                      dict(x=4.7, y=1.0, text="Peak effectiveness", showarrow=True, arrowhead=1),
                                      dict(x=14.1, y=0.6, text="Reduced impact", showarrow=True, arrowhead=1)
                                  ])
                st.plotly_chart(fig, use_container_width=True)

            elif selected_algo == "DeepSurv":
                st.markdown(r"""
                **Deep Extension of Cox Proportional Hazards**
                
                **Mathematical Formulation:**
                $$
                h(t|X) = h_0(t) \cdot \exp(f_{NN}(X;\theta))
                $$
                
                **Symbol Definitions:**
                - $f_{NN}(X;\theta)$: Neural network output (log-risk score)
                - $\theta$: Network parameters (weights & biases)
                - $h_0(t)$: Baseline hazard (estimated via Breslow's method)
                
                **Architecture Details:**
                ```python
                DeepSurv(
                    input_dim: int,      # Number of features
                    hidden_layers: list, # E.g., [32, 16]
                    dropout: float,      # Regularization
                    activation: str      # 'relu', 'selu', etc.
                )
                ```
                
                **Clinical Implementation Workflow:**
                1. Feature standardization (mean=0, std=1)
                2. Network pretraining with simulated data
                3. Fine-tuning with partial likelihood:
                   $$
                   \mathcal{L} = -\sum_{i=1}^n \delta_i[f_{NN}(X_i) - \log\sum_{j \in R(t_i)} \exp(f_{NN}(X_j))]
                   $$
                   Where $\delta_i$ is event indicator
                4. Risk stratification using output scores
                """)
                
                # Feature importance visualization
                st.subheader("📊 Biomarker Impact Analysis")
                features = ["Age", "Tumor Size", "ER Status", "HER2"]
                coefs = [0.32, 0.41, -0.28, 0.18]
                fig = go.Figure(go.Bar(x=features, y=coefs, 
                                       marker_color=['#636efa', '#ef553b', '#00cc96', '#ab63fa']))
                fig.update_layout(title="Feature Impacts on Survival Risk",
                                  yaxis_title="Normalized Coefficient Value",
                                  xaxis_title="Clinical Features")
                st.plotly_chart(fig, use_container_width=True)

            elif selected_algo == "DeepHit":
                st.markdown(r"""
                **Multi-Event Deep Survival Model**
                
                **Mathematical Formulation:**
                $$
                P(T=k, t) = \sigma\left(f_{k}(t,X;\theta)\right)
                $$
                Where:
                - $k \in \{1,...,K\}$: Event types
                - $\sigma$: Softmax activation over time bins
                - $f_{k}$: Network branch for event type $k$
                
                **Key Components:**
                1. Discrete Time Grid: $t_1 < t_2 < \cdots < t_m$
                2. Competing Risks Layer: $\sum_{k=1}^K P(T=k,t) = 1$
                3. Joint Loss Function:
                   $$
                   \mathcal{L} = \alpha\mathcal{L}_1 + (1-\alpha)\mathcal{L}_2
                   $$
                   Where:
                   - $\mathcal{L}_1$: Event type classification loss
                   - $\mathcal{L}_2$: Time prediction ranking loss
                
                **Clinical Implementation:**
                Predicting cardiovascular events (MI, stroke, death) with shared risk factors
                """)
                
                # Competing risks visualization
                st.subheader("⚠️ Competing Risks Projection")
                time_intervals = [f"{i}-{i+6}" for i in range(0, 30, 6)]
                prob_matrix = np.array([
                    [0.02, 0.01, 0.005], # 0-6 months
                    [0.05, 0.03, 0.01],  # 6-12 months
                    [0.08, 0.06, 0.02],  # 12-18 months
                    [0.10, 0.09, 0.04],  # 18-24 months
                    [0.12, 0.11, 0.07],  # 24-30 months
                ])
                
                fig = go.Figure()
                events = ["Cardiac Death", "MI (Myocardial Infarction)", "Stroke"]
                for i, event in enumerate(events):
                    fig.add_trace(go.Bar(
                        x=time_intervals,
                        y=prob_matrix[:, i],
                        name=event
                    ))
                fig.update_layout(barmode='stack', 
                                  title="Cumulative Incidence Function (CIF) by Interval",
                                  xaxis_title='Follow-up Interval (months)',
                                  yaxis_title='Cumulative Probability of Event',
                                  hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            
            elif selected_algo == "TEXGISA":
                st.markdown("""
                ### TEXGISA: Time-aware Explainable Gradient-based Inference for Survival Analysis
                A novel deep learning model designed for high interpretability by incorporating expert knowledge.
                """)
                
                tabs = st.tabs([
                    "💡 Overview & How-To",
                    "🧮 Mathematics & Losses",
                    "🔬 Explanations (TEXGI)",
                    "🧑‍🏫 Expert Priors",
                    "❓ FAQ"
                ])

                with tabs[0]:
                    st.subheader("Overview")
                    st.markdown("TEXGISA is a discrete-time survival model that integrates a unique explainability component (TEXGI) and allows for the encoding of domain knowledge as soft constraints during training, improving both performance and trust.")
                    
                    st.subheader("How-To Guide")
                    st.markdown("""
                    1.  **Data Preparation**: Format data with columns for time-to-event, event status, and covariates. Discretize the time axis into intervals.
                    2.  **Define Expert Priors**: In the 'Expert Priors' tab, specify features known to be important and assign weights to guide the model's attention.
                    3.  **Train Model**: Run the training script. The loss function will include terms for prediction accuracy, smoothness, and adherence to expert priors.
                    4.  **Generate Explanations**: Use the TEXGI module post-training to compute feature importance scores for individual predictions across time intervals.
                    5.  **Interpret Results**: Analyze the model's C-index, loss curves, and the time-aware feature importance plots to gain clinical insights.
                    """)

                with tabs[1]:
                    st.subheader("Mathematical Formulation")
                    st.markdown("---")
                    st.markdown("**Discrete-Time Survival Prediction**")
                    st.markdown("The survival probability at time interval $j$ for a patient with features $x$ is the product of conditional survival probabilities up to that point:")
                    st.latex(r"S^{(j)}(x) \;=\; \prod_{k=1}^{j}\big(1 - \hat h_k(x)\big)")
                    st.markdown("where $\hat h_k(x)$ is the model's predicted hazard (probability of event) in interval $k$.")
                    
                    st.markdown("**Combined Objective (Loss) Function**")
                    st.markdown("The model is trained by minimizing a composite loss function:")
                    st.latex(r"\mathcal{L} \;=\; \mathcal{L}_{\text{NLL}} \;+\; \lambda_{\text{smooth}}\Omega_{\text{smooth}} \;+\; \lambda_{\text{expert}}\Omega_{\text{expert}}")
                    st.markdown(r"""
                    - $\mathcal{L}_{\text{NLL}}$: The Negative Log-Likelihood for survival prediction accuracy.
                    - $\Omega_{\text{smooth}}$: A regularization term that encourages smoother changes in feature attributions over time.
                    - $\Omega_{\text{expert}}$: A term that penalizes the model if its feature attributions deviate from predefined expert knowledge.
                    """)

                with tabs[2]:
                    st.subheader("TEXGI in Practice")
                    st.markdown(
                        """
                        TEXGI (Time-aware Explainable Gradient-based Inference) is a form of integrated gradients adapted for this model.
                        - **Use extreme baselines** (e.g., all-zero or mean vectors) or tail sampling for more salient, time-specific explanations. A good baseline is crucial for meaningful attribution.
                        - **Increase IG steps `M`** for smoother, more accurate attributions, but be aware that this increases computation time.
                        - A **fast preview mode** (with fewer steps) is available to quickly check attribution trends before committing to a full, high-resolution run.
                        """
                    )

                with tabs[3]:
                    st.subheader("Setting Expert Priors")
                    st.markdown(
                        """
                        This feature allows you to inject domain knowledge directly into the model.
                        - **Mark domain-important features** in your configuration (e.g., a simple table or dictionary) and assign positive **weights** to them.
                        - **Gradually increase `λ_expert`**: This hyperparameter controls the strength of the expert prior. Start small and increase it to pull the model's learned attributions for those features towards a higher value (a soft constraint).
                        - **Penalize unimportant features**: You can also assign negative priors to features expected to be irrelevant, encouraging sparsity in explanations.
                        """
                    )

                with tabs[4]:
                    st.subheader("Frequently Asked Questions")
                    st.markdown(
                        """
                        **Q. My feature importance (FI) directions (positive/negative impact) seem counter-intuitive. What should I do?** A. First, re-check your feature scaling and column mapping. If a feature's impact still disagrees with domain knowledge, consider increasing its `λ_expert` weight in the expert priors to guide the model correctly.

                        **Q. Kaplan-Meier curves for different risk groups predicted by the model overlap significantly.** A. This may indicate the model isn't distinguishing risk well. Try using stronger expert priors, increasing the number of time intervals, or inspecting the feature importances to see which features are failing to drive risk separation.

                        **Q. The Concordance-index (C-index) is plateauing during training.** A. Try a more aggressive learning rate schedule (e.g., a higher initial rate followed by decay). Alternatively, moderately increase model capacity (more layers/neurons) or the number of time bins if your data has sufficient granularity.
                        """
                    )

    st.markdown("---")
    with st.expander("🧭 Algorithm Comparison Guide (click to expand)", expanded=False):

        # Minimal CSS to improve readability
        st.markdown("""
        <style>
        table.alg {width:100%; border-collapse:collapse; margin:0.5rem 0 1rem;}
        table.alg th, table.alg td {border:1px solid rgba(127,127,127,.25); padding:8px 10px; vertical-align:top;}
        table.alg th {background:rgba(127,127,127,.12); text-align:left;}
        table.alg td.metric {white-space:nowrap; font-weight:600;}
        table.alg td.explain {width:38%; color:var(--text-color-secondary,#bbb);}
        .ok,.no,.partial {display:inline-block; font-weight:700; padding:2px 8px; border-radius:10px;}
        .ok      {background:rgba(76,175,80,.18);  color:#9BE38C;}
        .no      {background:rgba(244,67,54,.18); color:#FF8A80;}
        .partial {background:rgba(255,193,7,.18); color:#FFD54F;}
        .muted {color:var(--text-color-secondary,#bbb); font-size:0.95rem; margin:.25rem 0 .75rem;}
        </style>
        """, unsafe_allow_html=True)

        def _badge(value, reason):
            """
            value: True / False / 'partial'
            reason: short hover text explaining WHY
            """
            if value is True:
                return f"<span class='ok' title=\"{reason}\">✓</span>"
            if value == "partial":
                return f"<span class='partial' title=\"{reason}\">◎</span>"
            return f"<span class='no' title=\"{reason}\">✗</span>"

        rows = [
            dict(
                metric="Handles Censoring",
                coxtime = _badge(True,  "Cox partial likelihood naturally incorporates right-censored samples."),
                deepsurv= _badge(True,  "Same Cox-style partial likelihood with a neural risk function."),
                deephit = _badge(True,  "Discrete-time likelihood explicitly accounts for censoring."),
                texgisa = _badge(True,  "TEXGISA optimizes a survival loss supporting right-censoring."),
                explain="Whether right-censored cases contribute correctly to the objective so that estimates remain unbiased."
            ),
            dict(
                metric="Interpretability",
                coxtime = _badge(True,  "Coefficients/attributions interpretable; time-dependence reduces transparency slightly."),
                deepsurv= _badge(False, "Non-linear black box; relies on post-hoc explanations."),
                deephit = _badge(False, "Multi-output NN; requires post-hoc interpretation."),
                texgisa = _badge(True,  "High by design: time-aware feature importance and rule-based priors in TEXGISA."),
                explain="How directly the model communicates which features drive risk over time (without heavy post-hoc tooling)."
            ),
            dict(
                metric="Expert Knowledge",
                coxtime = _badge(False, "No native prior channel besides regularization."),
                deepsurv= _badge(False, "No built-in prior mechanism."),
                deephit = _badge(False, "No built-in prior mechanism."),
                texgisa = _badge(True,  "Supports explicit expert priors (rules/weights) via λ_expert during training."),
                explain="Whether domain priors (e.g., monotonicity, signed rules) can be injected to guide learning."
            ),
        ]

        header = """
        <thead>
            <tr>
            <th>Metric</th>
            <th>CoxTime</th>
            <th>DeepSurv</th>
            <th>DeepHit</th>
            <th>TEXGISA</th>
            <th>Explanation</th>
            </tr>
        </thead>
        """
        body = []
        for r in rows:
            body.append(
                f"<tr>"
                f"<td class='metric'>{r['metric']}</td>"
                f"<td>{r['coxtime']}</td>"
                f"<td>{r['deepsurv']}</td>"
                f"<td>{r['deephit']}</td>"
                f"<td>{r['texgisa']}</td>"
                f"<td class='explain'>{r['explain']}</td>"
                f"</tr>"
            )
        st.markdown(f"<table class='alg'>{header}<tbody>{''.join(body)}</tbody></table>", unsafe_allow_html=True)

        st.markdown(
            "",
            unsafe_allow_html=True,
        )

        st.markdown("""
        """)

if __name__ == "__main__":
    show()