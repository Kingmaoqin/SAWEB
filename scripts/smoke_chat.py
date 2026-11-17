"""Exercise each Chat with Agent route with the bundled synthetic CSV."""

import sys

from streamlit.testing.v1 import AppTest

MODELS = ("CoxTime", "DeepSurv", "DeepHit", "TEXGISA")


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def check(app, step):
    issues = [item.message for item in app.error] + [item.message for item in app.exception]
    assert not issues, f"{step}: {issues}"


def new_app():
    app = AppTest.from_file("main.py", default_timeout=120)
    app.session_state["current_page"] = "Chat with Agent"
    app.run()
    check(app, "empty chat")
    widget(app, "button", "Load synthetic demo in chat").click().run()
    check(app, "load demo")
    assert app.session_state["qa_time_col"] == "duration"
    assert app.session_state["qa_event_col"] == "event"
    return app


def confirm(app):
    assert app.session_state["pending_action"] is not None
    app.chat_input[0].set_value("yes").run()
    check(app, "confirm")
    result = app.session_state["last_results"]
    assert result.get("metrics"), "No numeric model metrics"
    assert "survival_curves" in result.get("artifacts", {}), "No predicted survival curves"
    assert any(item.value == "Kaplan–Meier" for item in app.subheader)
    return result


def main():
    # Help Bot and free-text concepts should work with no API key.
    app = new_app()
    app.chat_input[0].set_value("what is survival analysis").run()
    assert "how long" in app.session_state["chat_messages"][-1].content.lower()
    app.chat_input[0].set_value("compare the models").run()
    assert "CoxTime" in app.session_state["chat_messages"][-1].content
    app.chat_input[0].set_value("what columns are in my dataset?").run()
    assert "duration" in app.session_state["chat_messages"][-1].content
    check(app, "chat questions")
    print("QUESTIONS: OK", flush=True)

    for model in MODELS:
        app = new_app()
        widget(app, "button", f"Run {model}" if model != "TEXGISA" else "Run TEXGISA (no expert)").click().run()
        assert "event=event" in app.session_state["pending_action"]["label"]
        result = confirm(app)
        if model == "TEXGISA":
            assert "feature_importance" in result["artifacts"]
        print(f"QUICK {model}: OK", flush=True)

    for model in MODELS:
        app = new_app()
        widget(app, "selectbox", "Algorithm").set_value(model).run()
        widget(app, "number_input", "Epochs").set_value(2).run()
        widget(app, "button", "Run now").click().run()
        confirm(app)
        print(f"DIRECT {model}: OK", flush=True)

    for model in MODELS:
        app = new_app()
        app.chat_input[0].set_value(f"run {model} time=duration event=event epochs=2").run()
        assert model in app.session_state["pending_action"]["label"]
        confirm(app)
        print(f"COMMAND {model}: OK", flush=True)

    app = new_app()
    app.chat_input[0].set_value("preview feature importance time=duration event=event epochs=2").run()
    result = confirm(app)
    assert "feature_importance" in result["artifacts"]
    print("PREVIEW: OK", flush=True)

    app = new_app()
    widget(app, "button", "Run CoxTime").click().run()
    app.chat_input[0].set_value("no").run()
    assert app.session_state.filtered_state.get("pending_action") is None
    assert app.session_state.filtered_state.get("last_results") is None
    print("CANCEL: OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
