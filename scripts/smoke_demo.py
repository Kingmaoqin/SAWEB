"""Exercise the Streamlit survival workbench with the bundled synthetic data."""

import argparse
import time

import pandas as pd
from streamlit.testing.v1 import AppTest


def errors(app):
    return [entry.message for entry in app.exception] + [entry.message for entry in app.error]


def require_clean(app, label):
    found = errors(app)
    if found:
        raise AssertionError(f"{label}: {found}")


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["CoxTime", "DeepSurv", "DeepHit", "TEXGISA"])
    parser.add_argument("--cf", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mode", choices=["normal", "preview", "expert", "playback"], default="normal")
    args = parser.parse_args()

    data = pd.read_csv("demo_data/synthetic_survival_demo.csv")
    app = AppTest.from_file("main.py", default_timeout=240)
    app.session_state["clinical_data"] = data
    app.session_state["current_page"] = "Run Models"
    app.run()
    require_clean(app, "load")
    assert widget(app, "selectbox", "Duration column").value == "duration"
    assert widget(app, "selectbox", "Event column").value == "event"

    widget(app, "selectbox", "Algorithm").set_value(args.model).run()
    require_clean(app, "select model")
    widget(app, "number_input", "Epochs").set_value(args.epochs).run()
    require_clean(app, "set epochs")
    if args.model == "TEXGISA":
        widget(app, "number_input", "Generator epochs").set_value(1).run()
        require_clean(app, "set generator epochs")
        widget(app, "checkbox", "Show animation during training (slower)").set_value(False).run()
        require_clean(app, "disable animation")
        button = {
            "preview": "👀 Preview Feature Importance (no expert priors)",
            "expert": "🚀 Train with Expert Priors",
            "playback": "🎬 Train + Hazard Playback",
            "normal": "🚀 Train with Expert Priors",
        }[args.mode]
    else:
        button = f"🚀 Train {args.model}"

    start = time.monotonic()
    widget(app, "button", button).click().run()
    require_clean(app, "train")
    result = app.session_state["results"]
    assert isinstance(result, dict), "No training result was saved"
    print(f"TRAIN {args.model} {args.mode}: {time.monotonic()-start:.1f}s; "
          f"keys={list(result)}; "
          f"success={[entry.value for entry in app.success]}", flush=True)

    for chart in ["Predicted Survival Trajectories", "Kaplan–Meier (overall)",
                  "Kaplan–Meier by risk groups"]:
        widget(app, "selectbox", "Select a chart to display").set_value(chart).run()
        require_clean(app, chart)
        print(f"CHART {chart}: OK", flush=True)

    if args.cf:
        if args.model == "TEXGISA":
            widget(app, "slider", "Gradient steps").set_value(60).run()
            button = "Generate TEXGI counterfactual"
        else:
            button = "Generate CF suggestions"
        start = time.monotonic()
        widget(app, "button", button).click().run()
        require_clean(app, "counterfactual")
        print(f"COUNTERFACTUAL {args.model}: {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
