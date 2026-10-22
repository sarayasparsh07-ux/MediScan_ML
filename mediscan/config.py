"""Project-wide constants. Every value here comes from the assignment specification."""

ID_COL = "patient_id"
TARGET = "diagnosis"
FEATURES = [
    "cell_size",
    "cell_shape",
    "clump_thickness",
    "uniformity",
    "bland_chromatin",
    "mitoses",
    "age",
    "marker_level",
]
EXPECTED_COLUMNS = [ID_COL, *FEATURES, TARGET]
CLASS_NAMES = {0: "Benign", 1: "Malignant"}
POS_LABEL = 1  # malignant is ALWAYS the positive class

# Expected properties from 00_EDA_Report.md (verified at start-up, never assumed)
EXPECTED_N_ROWS = 800
EXPECTED_CLASS_COUNTS = {0: 430, 1: 370}

# Train/test split exactly as in 02_Student_Guide.md (Part A3)
TEST_SIZE = 0.25
SPLIT_RANDOM_STATE = 1

# Human-readable feature labels
FEATURE_LABELS = {
    "cell_size": "Cell Size",
    "cell_shape": "Cell Shape",
    "clump_thickness": "Clump Thickness",
    "uniformity": "Uniformity",
    "bland_chromatin": "Bland Chromatin",
    "mitoses": "Mitoses",
    "age": "Age",
    "marker_level": "Marker Level",
}
FEATURE_UNITS = {
    "cell_size": "biopsy measurement units",
    "cell_shape": "irregularity score",
    "clump_thickness": "score (1–10)",
    "uniformity": "score (1–10)",
    "bland_chromatin": "score (1–10)",
    "mitoses": "count of dividing cells",
    "age": "years",
    "marker_level": "blood-marker reading",
}

# Baseline model order (as listed in the guide's suggested team split)
MODEL_ORDER = [
    "Logistic Regression",
    "Naive Bayes",
    "SVM (Linear)",
    "SVM (RBF)",
    "Decision Tree",
    "Random Forest",
]
MODEL_SPEC = {
    "Logistic Regression": "LogisticRegression(max_iter=2000)",
    "Naive Bayes": "GaussianNB()",
    "SVM (Linear)": "SVC(kernel='linear')",
    "SVM (RBF)": "SVC(kernel='rbf', C=3)",
    "Decision Tree": "DecisionTreeClassifier(max_depth=6, random_state=0)",
    "Random Forest": "RandomForestClassifier(n_estimators=200, random_state=0)",
}
MODEL_FAMILY = {
    "Logistic Regression": "Linear boundary",
    "Naive Bayes": "Probabilistic (curved boundary)",
    "SVM (Linear)": "Linear boundary",
    "SVM (RBF)": "Kernel (curved boundary)",
    "Decision Tree": "Rule-based splits",
    "Random Forest": "Tree ensemble",
}
LINEAR_MODELS = {"Logistic Regression", "SVM (Linear)"}

# Team of five (the brief suggests 3–4 members; the professor approved five verbally).
# Order = the order the team lists itself. Each required model has exactly one owner.
TEAM = {
    "Sparsh": ["Random Forest", "Decision Tree"],
    "Aradhana": ["SVM (Linear)"],
    "Shashaank": ["SVM (RBF)"],
    "Shirley": ["Naive Bayes"],
    "Mahi": ["Logistic Regression"],
}
MODEL_OWNER = {m: who for who, models in TEAM.items() for m in models}
assert sorted(MODEL_OWNER) == sorted(MODEL_ORDER), "every required model needs exactly one owner"

# ---------------- MediScan visual identity ----------------
NAVY = "#16324F"
TEAL = "#168C86"
MINT = "#DDF3EC"
PALE_BLUE = "#EAF2FA"
OFF_WHITE = "#F6F8FB"
SLATE = "#334155"
AMBER = "#E9B44C"
CORAL = "#E76F51"
GRID = "#E2E8F0"
MUTED = "#64748B"

# Class colours — teal/navy family for benign, coral for malignant.
# Colour is never the only cue: markers and labels also differ.
BENIGN_COLOR = "#2A7DB8"
MALIGNANT_COLOR = CORAL
CLASS_COLORS = {"Benign": BENIGN_COLOR, "Malignant": MALIGNANT_COLOR}
CLASS_SYMBOLS = {"Benign": "circle", "Malignant": "diamond"}

# One fixed colour per model, used everywhere a model is identified by colour.
# Validated with the dataviz palette checker (lightness band, chroma floor, CVD and
# normal-vision separation all PASS). Two adjacent pairs sit in the CVD 6–8 band and
# three hues are < 3:1 vs the surface, so every model colour is ALWAYS paired with a
# secondary cue: a distinct dash style, a direct label or axis label, and a table view.
MODEL_COLORS = {
    "Logistic Regression": "#2A78D6",
    "Naive Bayes": "#E87BA4",
    "SVM (Linear)": "#1BAF7A",
    "SVM (RBF)": "#C98500",
    "Decision Tree": "#8A63D2",
    "Random Forest": "#0F7A2E",
}
MODEL_SYMBOLS = {
    "Logistic Regression": "square",
    "Naive Bayes": "triangle-up",
    "SVM (Linear)": "x",
    "SVM (RBF)": "diamond",
    "Decision Tree": "pentagon",
    "Random Forest": "circle",
}
MODEL_DASH = {
    "Logistic Regression": "dot",
    "Naive Bayes": "dash",
    "SVM (Linear)": "dashdot",
    "SVM (RBF)": "longdash",
    "Decision Tree": "longdashdot",
    "Random Forest": "solid",
}
