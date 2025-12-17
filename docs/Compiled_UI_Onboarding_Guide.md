# DHAI Survival Analysis Platform (Compiled UI) Onboarding Guide
- Version: written against the compiled UI; last updated: 2025-05-07 (matches current release)
- Target users: first-time clinicians/researchers/students using survival analysis or this platform
- You will learn:
  - Run a 30-second loop: upload → map columns → train → read metrics/curves → export
  - Match archetypes and scenarios to the right pages/modules
  - Locate navigation, sidebar, upload, and algorithm controls quickly
  - Follow page-by-page steps with checkpoints to verify results
  - Troubleshoot common errors on data, parameters, and environment

> Shortcut reading: want one successful run? Read **0. Quick Start** + **4.3 Model Training** + first two FAQ items. Want concepts? Read **2. Interface Overview** + **4.2 Algorithm Tutor**. Need screenshots? Jump to **3. Screenshot Checklist**.

> What this platform does NOT do: it does not replace clinical decisions; outputs are for research/demo and need professional judgment.

## 0. Quick Start (30s)
1. Upload CSV (default via **Model Training**): choose file in the upload area.
   - Checkpoint: `Loaded <filename> · shape = rows × cols` or column preview appears.
2. Map columns: in **Model Training** mapping table select `duration` (time) and `event` (0/1).
   - Checkpoint: dropdowns show the selected time and event columns.
3. Pick algorithm and run: choose **TEXGISA / CoxTime / DeepSurv / DeepHit** → click Run.
   - Checkpoint: progress/success message; no errors.
4. Read results: in result cards, view C-index, predicted survival curves, KM curve; expand Feature Importance when present.
   - Checkpoint: C-index appears; charts render without blank areas.
5. Export/replicate: download tables/feature importance, or note time/event columns, algorithm, and epochs.
   - Checkpoint: file saves or parameters are captured.

Optional path: if you prefer chat-driven steps, use **Chat with Agent** (Upload CSV + Direct Run); follow the same flow as above.

## 1. User Archetypes & Typical Scenarios
### 1.1 Archetypes
- Role: Clinical PI / Physician
  - Goal: verify model separates risk groups and highlights key risk factors.
  - Pain points: unfamiliar with survival params; needs interpretable outputs.
  - Pages used most: Model Training, Chat with Agent, Algorithm Tutor
  - Outputs cared most: C-index, KM curves, Feature Importance (direction + weight)
- Role: Data Analyst / Student
  - Goal: run pipeline, tune hyperparameters, compare algorithms.
  - Pain points: wrong time/event mapping; unclear parameter meanings.
  - Pages used most: Model Training, Algorithm Tutor
  - Outputs cared most: training logs, metric tables, curves, exported configs
- Role: Demo Specialist / Implementation Consultant
  - Goal: present an end-to-end demo and interpretation.
  - Pain points: stable steps, clean screenshots, avoid errors.
  - Pages used most: Home, Model Training, Chat with Agent, Publications
  - Outputs cared most: page titles + step lists, result cards, export links
- Role: Instructor / TA
  - Goal: explain concepts and algorithm differences; guide student practice.
  - Pain points: students unfamiliar with metrics/curves; need visuals.
  - Pages used most: Algorithm Tutor, Home, Model Training
  - Outputs cared most: hazard/survival animations, algorithm tables, KM curves

### 1.2 Scenarios
- Scenario: Clinical PI checks model usability
  - Goal: see risk separation and variable explanations
  - Page: Model Training → Results
  - Action: upload data, map duration/event, select TEXGISA, enable Feature Importance (if present), click Run
  - Output: C-index, predicted survival curves, KM, Top-k Feature Importance
  - Next: download feature importance for discussion
- Scenario: Analyst compares algorithms
  - Goal: compare CoxTime / DeepSurv / DeepHit / TEXGISA metrics
  - Page: Chat with Agent (quick buttons) or Model Training (form)
  - Action: run four algorithms on same dataset, record C-index
  - Output: multiple result cards with metrics
  - Next: pick best algorithm, adjust epochs/lr, rerun
- Scenario: Demo specialist needs screenshots
  - Goal: collect high-quality captures
  - Page: Home, Algorithm Tutor, Model Training, Chat with Agent, Publications, Contact
  - Action: expand Quick Start, move hazard ratio sliders, run one model and show results
  - Output: titles + descriptions, animation charts, metric cards
  - Next: capture per checklist and build slides
- Scenario: TA teaches concepts
  - Goal: animate hazard/survival, explain KM
  - Page: Algorithm Tutor
  - Action: adjust Median Survival & Hazard Ratio sliders, play animation
  - Output: Hazard & Survival charts, trend explanations
  - Next: ask students to upload sample data in Model Training for practice

## 2. Interface Overview (Compiled)
- Top navigation (button text): 🏠 Home | 📘 Algorithm Tutor | 🧪 Model Training (Run Models) | 🤖 Chat with Agent | 📄 Publications | ✉️ Contact
- Sidebar:
  - User Center dropdown: Home / Our Datasets / Our Experiments / Our Results / History / Favorite Models (pages beyond Home show a user-center placeholder)
  - Settings: Theme Switch, Multi-language Support (Under Development)
  - Help Bot: type questions to get guidance
- Global concepts:
  - duration: follow-up/survival time column (numeric)
  - event: event indicator (1=event, 0=censored)
  - censoring: observation ended without event; not the same as event
  - C-index: concordance metric; closer to 1 is better
  - KM (Kaplan–Meier): stepwise empirical survival curve; long flat segments often mean censoring or no events
  - hazard/survival: instant risk vs. survival probability; higher hazard → faster survival drop

## 3. Screenshot Checklist (what to capture)
| Fig | Page | Region to capture (precise module/card/control) | Purpose | Notes |
|-----|------|-----------------------------------------------|---------|-------|
| Fig1 | Home | Page title + caption + expanded "🚀 How to Use This System" 5-step list | Show quick-start entry | Expand the accordion |
| Fig2 | Algorithm Tutor | "📚 Core Survival Analysis Concepts" + hazard/survival animation (play button + sliders) | Explain concepts and interactive controls | Animation controls visible |
| Fig3 | Algorithm Tutor | "🔍 Choose an Algorithm to Explore" with TEXGISA tab selected | Show algorithm selection and info entry | TEXGISA tab expanded |
| Fig4 | Model Training | Upload/mapping area: upload control + duration/event dropdown + features list | Demonstrate data prep | Include mapping table if present |
| Fig5 | Model Training | Training params: Algorithm dropdown + epochs/lr/batch_size inputs + Run button | Show run entry | Pre-run state is fine |
| Fig6 | Model Training | Results: C-index/metric cards + predicted survival curve (or KM) | Show output reading | Ensure charts rendered |
| Fig7 | Chat with Agent | Left chat history + "⚡ Quick Actions" row + right "Upload CSV" card | Show chat and quick runs | Better with loaded data |
| Fig8 | Chat with Agent | Right "🧪 Direct Run" form (algorithm dropdown, time/event selectors, Run now) | Show non-LLM run path | Column dropdowns visible |
| Fig9 | Publications | Publications table (Title + Paper Link) | Show resource entry | Full table |
| Fig10 | Contact | Contact card (address/phone/email) | Show contact info | Keep title + caption |

## 4. Page-by-Page Walkthrough
### 4.1 Home
- Purpose: read platform intro and quick-start steps.
- Prepare: none; just know your time/event column names.
- Steps:
  1. Expand "🚀 How to Use This System" accordion → read the 5-step loop.
     - Checkpoint: see upload → map → choose algorithm → interpret → metrics list.
  2. For concept recap, scroll to "What is Survival Analysis?" and research direction accordions.
     - Checkpoint: accordions expand and text is readable.
- Output reading: no metrics here; this page is just entry and context.
- Export/replicate: capture accordion content for training materials.

### 4.2 Algorithm Tutor
- Purpose: learn survival concepts, view TEXGISA details and algorithm comparisons, and generate interactive curves.
- Prepare: none; browsing only.
- Steps:
  1. Expand "📚 Core Survival Analysis Concepts" → adjust "Median Survival" and "Hazard Ratio" sliders.
     - Checkpoint: Hazard vs. Time and Survival vs. Time charts change together; play/pause works.
     - Common issue: slider beyond range (6–60 months, 0.5–2.0) may stop responding → refresh page.
  2. Open "🔍 Choose an Algorithm to Explore" and select an algorithm (CoxTime/DeepSurv/DeepHit/TEXGISA).
     - Checkpoint: matching descriptions and charts appear; if not, ensure selection is not "Select Algorithm".
  3. Expand "🧭 Algorithm Comparison Guide" to review badge/tag differences.
- Output reading:
  - Animation: follow the legend for colors/direction; if no legend, focus on trend (higher hazard → faster survival drop).
  - Algorithm sections: read formulas/charts to understand model traits.
- Export/replicate: capture animations and algorithm descriptions per checklist; no file export needed.

### 4.3 Model Training (Run Models)
- Purpose: upload data, map columns, choose algorithm, train, and view metrics/curves.
- Prepare:
  - CSV with at least header example: `duration,event,feature1,feature2`
  - Event allowed values: 0/1 (convert True/False or Yes/No to 1/0 before upload).
  - duration must be numeric; non-numeric features are ignored or need prior encoding; fill missing values first.
- Steps (example: TEXGISA):
  1. Click "Upload CSV" to choose file.
     - Checkpoint: shows row/column counts and column names; otherwise an error appears.
     - Common issue: encoding/delimiter errors → ensure UTF-8 and comma-separated.
  2. In mapping area, set Time column = duration; Event column = event (or manual choice).
     - Checkpoint: dropdowns show selected columns; event mapped to 0/1.
     - Common issue: event column not binary → preprocess to 0/1 before upload.
  3. Choose Algorithm (TEXGISA/CoxTime/DeepSurv/DeepHit); set Epochs, Learning rate, Batch size.
     - Checkpoint: numeric boxes accept input; out-of-range values constrained.
     - Common issue: too-high LR diverges → start with 0.001–0.01.
  4. Click Run; wait for completion.
     - Checkpoint: no errors; results show metrics and charts.
  5. Expand Feature Importance (if supported) to see top-k directions and weights.
- Output reading:
  - C-index: >0.6 suggests separability; near 0.5 → check data/features.
  - Predicted survival curves: larger separation → stronger risk heterogeneity.
  - KM curves: stepwise drops; long flat tail often means heavy censoring.
  - Feature Importance: follow the legend for colors/directions; if no legend, read only the ranking, not direction.
- Export/replicate: use download buttons in results for tables/feature importance; log chosen time/event columns, algorithm, and params.

### 4.4 Chat with Agent
- Purpose: ask guidance via chat, run algorithms directly, view structured outputs.
- Prepare: CSV and knowledge of time/event column names.
- Steps:
  1. Upload data in the right "Upload CSV" card.
     - Checkpoint: success message with filename and shape; expand Show columns if available.
     - Common issue: load fails → verify CSV format and no empty headers.
  2. Use left "⚡ Quick Actions" if columns auto-detected; click Run TEXGISA / CoxTime / DeepSurv / DeepHit.
     - Checkpoint: chat shows result summary; if disabled, data not loaded yet.
  3. Use right "🧪 Direct Run" form: pick Algorithm, Time column, Event column, Epochs/LR/Batch size → Run now.
     - Checkpoint: result card shows C-index, survival curve, KM; if empty, recheck column mapping.
  4. For natural-language commands, type “run deepsurv time=duration event=event” in chat.
     - Checkpoint: agent replies with run status or asks for missing columns.
- Output reading: same as Model Training; chat replies summarize metrics and chart locations.
- Export/replicate: download via result cards; copy chat history if needed.

### 4.5 Publications
- Purpose: view paper links.
- Steps: no upload; read table and click Paper Link.
- Output reading: Link column is clickable.
- Export/replicate: copy links or capture table.

### 4.6 Contact
- Purpose: access contact info.
- Steps: read address/phone/email directly.
- Output reading: use for calls or emails.
- Export/replicate: screenshot or copy details.

## 5. FAQ & Troubleshooting
- Symptom: columns not detected after upload
  - Cause: delimiter/encoding issues or missing header
  - Fix: ensure UTF-8, comma-separated, first row has column names; re-upload
- Symptom: wrong duration/event mapping
  - Cause: unclear naming or mixed characters
  - Fix: manually pick correct columns; ensure event is 0/1 only
- Symptom: metrics empty or errors
  - Cause: missing values, non-numeric features, or invalid params
  - Fix: clean NA, keep numeric columns; reset LR/batch to defaults; retry
- Symptom: KM curve looks odd (heavy censoring)
  - Cause: very low event rate or short follow-up
  - Fix: check event rate; extend follow-up or filter anomalies; confirm C-index usability
- Symptom: Feature Importance lacks color/direction
  - Cause: run without directional info or algorithm unsupported
  - Fix: choose TEXGISA and enable directional explanations if present; otherwise mark TODO
- Symptom: LLM/Agent unavailable
  - Cause: offline or missing token
  - Fix: use right "🧪 Direct Run" or Model Training page to run directly
- Symptom: Runs are slow
  - Cause: too many epochs, tiny batch_size, heavy explanations
  - Fix: start with epochs 80–120, batch_size ≥32, turn off heavy explanation; check hardware
- Symptom: Feature directions conflict with domain knowledge
  - Cause: no standardization or data quality issues
  - Fix: review preprocessing; if supported, increase smoothing/prior weight; otherwise note for further analysis

## 6. Glossary
- Survival Time / Survival Time / time until event / Algorithm Tutor animations, column mapping
- Event Indicator / Event / 1=event, 0=censored / mapping dropdowns, Direct Run form
- Censoring / Censoring / samples without observed event / KM interpretation, algorithm input
- Concordance Index / C-index / ranking agreement metric; higher is better / result cards, chat summary
- Kaplan–Meier Curve / KM / empirical survival curve with steps / results charts
- Hazard / Hazard / instantaneous risk; higher means faster events / Algorithm Tutor, algorithm descriptions
- Survival Probability / Survival / 1 - cumulative risk / Algorithm Tutor, results curves
- Feature Importance / Feature Importance / variable contribution to risk / TEXGISA results
- Epochs / Epochs / full passes over training data / Model Training, Direct Run form
- Learning Rate / LR / step size for updates / Model Training, Direct Run form
