"""Headless UI test: every page renders and every interactive control works without exceptions.

Run:  python -m pytest tests/test_app.py -q      (≈ 1–2 min; trains all models once)
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def at():
    a = AppTest.from_file(APP, default_timeout=600)
    a.run()
    assert not a.exception, [e.message for e in a.exception]
    return a


def _goto(at, name):
    opt = next(o for o in at.sidebar.radio[0].options if name in o)
    at.sidebar.radio[0].set_value(opt).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _every(at, sb):
    for i in range(len(sb.options)):
        sb.select_index(i).run()
        assert not at.exception, [e.message for e in at.exception]


def test_all_pages_render(at):
    for opt in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(opt).run()
        assert not at.exception, (opt, [e.message for e in at.exception])
        assert not at.error, (opt, [e.value for e in at.error])


def test_leaderboard_every_metric(at):
    _goto(at, "Model Leaderboard")
    _every(at, at.selectbox[0])


def test_confusion_lab_controls(at):
    _goto(at, "Confusion Matrix Lab")
    at.radio[0].set_value("Row percentages").run()
    _every(at, at.selectbox[0])
    at.radio[0].set_value("Counts").run()


def test_boundary_explorer(at):
    _goto(at, "PCA & Decision Boundaries")
    _every(at, at.selectbox[0])
    for t in at.toggle:
        t.set_value(False)
    at.run()
    assert not at.exception


def test_scaling_page_controls(at):
    _goto(at, "Scaling & Feature Insights")
    at.radio[0].set_value("Scaled").run()
    _every(at, at.selectbox[0])
    at.radio[1].set_value("Accuracy").run()
    _every(at, at.selectbox[1])


def test_significance_page_controls(at):
    _goto(at, "Significance & Thresholds")
    for i in range(len(at.radio[0].options)):          # patient subset
        at.radio[0].set_value(at.radio[0].options[i]).run()
        assert not at.exception, [e.message for e in at.exception]
    at.toggle[0].set_value(False).run()                 # unadjusted p-values
    _every(at, at.selectbox[2])                          # CV t-test metric
    at.toggle[1].set_value(True).run()                  # include linear models
    _every(at, at.selectbox[3])                          # threshold model
    at.radio[1].set_value("Reach a target recall").run()
    for v in (0.95, 0.99, 1.0):
        at.select_slider[0].set_value(v).run()
        assert not at.exception, [e.message for e in at.exception]
    at.selectbox[0].select_index(2).run()               # pair inspector
    at.selectbox[1].select_index(2).run()               # same model on both sides -> info, no crash
    assert not at.exception


def test_combinations_page_and_explorer(at):
    _goto(at, "Model Combinations")
    ms = at.multiselect[0]
    for members in (["Random Forest"], ["Naive Bayes", "Decision Tree", "Random Forest"], []):
        ms.set_value(members).run()
        assert not at.exception, [e.message for e in at.exception]
    at.multiselect[0].set_value(["Naive Bayes", "SVM (RBF)", "Random Forest"]).run()
    for t in (0.05, 0.4, 0.95):
        at.select_slider[0].set_value(t).run()
        assert not at.exception, [e.message for e in at.exception]


def test_methodology_model_internals(at):
    _goto(at, "Methodology")
    assert len(at.tabs) >= 6                                    # one tab per required model
    assert any("Decision Tree" in str(t.label) for t in at.tabs)


def _recommended(at):
    card = next(m.value for m in at.markdown if "Recommended for MediScan" in m.value)
    return card.split("class='name'>")[1].split("<")[0]


def test_recommendation_controls_and_consistency(at):
    _goto(at, "Clinical Recommendation")
    at.radio[0].set_value(at.radio[0].options[1]).run()  # CV evidence
    assert not at.exception
    at.slider[2].set_value(0).run()                       # FN tolerance 0 -> strictly fewest FN
    strict = _recommended(at)
    at.slider[2].set_value(1).run()
    default = _recommended(at)
    # with tolerance 0 the pick must be a fewest-FN model
    assert strict == "Naive Bayes"
    assert default in {"Random Forest", "Naive Bayes", "Decision Tree", "SVM (RBF)"}
    para = next(m.value for m in at.markdown if "class='para'" in m.value)
    assert default in para and "educational" in para.lower()
    assert "soft-voting ensemble" in para                      # two-level recommendation is in the paragraph


def test_start_here_route(at):
    _goto(at, "Start Here")
    labels = [str(t.label) for t in at.tabs]
    assert labels[:5] == ["① The ring", "② Line vs curve", "③ Scaling", "④ Combinations", "⑤ Recommendation"]
    cards = [m.value for m in at.markdown if "class='rec-card'" in m.value]
    assert len(cards) == 2 and "Random Forest" in cards[0] and "NB + SVM-RBF + RF @ 0.40" in cards[1]


def test_calibration_controls(at):
    _goto(at, "Significance & Thresholds")
    at.radio[2].set_value("Held-out test (200)").run()          # calibration: test-set view
    assert not at.exception, [e.message for e in at.exception]
    at.multiselect[0].set_value(["Logistic Regression", "Naive Bayes"]).run()
    assert not at.exception, [e.message for e in at.exception]
    at.multiselect[0].set_value([]).run()                       # empty selection -> info, no crash
    assert not at.exception
    assert any("squeezed toward" in m.value for m in at.markdown)


def test_team_names_everywhere(at):
    from mediscan.config import TEAM
    _goto(at, "Clinical Recommendation")
    para = next(m.value for m in at.markdown if "class='para'" in m.value)
    for who in TEAM:
        assert who in para, who
    assert "calibration" in para.lower()
    _goto(at, "Methodology")
    assert any("Sparsh" in str(t.label) for t in at.tabs)
