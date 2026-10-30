# AI-Confidence-and-Verification-System
Machine Learning based text classification and fact verification system using Python and Streamlit.
A small, beginner-friendly machine learning project built with Python, scikit-learn, and Streamlit.

You type a short statement. The app shows two independent results:

Model confidence – which topic the classifier predicts (Science, Technology, Health, Geography, or Finance) and how strongly it favours that topic.
Reference verification – whether a small, transparent local knowledge base Supports, Contradicts, or has Insufficient Evidence about the claim.
Confidence and verification are deliberately kept apart. A high confidence score means "the wording looks like this topic", not "this statement is true".

No paid APIs, no API keys, no internet connection required.
All paths are relative to the project folder.
Everything runs locally on your own computer.
1. What the project demonstrates
Part	What it does	File
Training script	Trains TF-IDF + Logistic Regression on a labelled sample dataset, saves the model	train_model.py
Classifier module	Loads the model, returns the predicted topic and a probability per topic	text_classifier.py
Verification module	Rule-based check against a small reference CSV, returns Supported / Contradicted / Insufficient Evidence	verification.py
Streamlit app	User interface that displays both results side by side	app.py
Sample data	180 labelled example sentences (5 topics × 36)	data/training_data.csv
Reference data	17 transparent reference rows with sources	data/knowledge_base.csv
Tests	Unit tests for the verifier, the model, and a headless app smoke test	tests/
2. Windows + VS Code setup (step by step)
In a hurry? Two helper scripts in the project folder do steps 4–7 for you: double-click setup.bat (creates .venv, installs requirements.txt, trains the model), then double-click run.bat to launch the app. The manual steps below explain what they do.

Step 1 – Install Python
Download Python 3.10 or newer from https://www.python.org/downloads/windows/.
Run the installer and tick "Add python.exe to PATH" on the first screen.
Finish the install.
Check it worked. Open Command Prompt or PowerShell and run:

py --version
You should see something like Python 3.12.4. If not, reinstall and make sure the PATH checkbox is ticked.

Step 2 – Open the project in VS Code
Install Visual Studio Code.
Install the Python extension (by Microsoft) from the Extensions panel.
In VS Code choose File → Open Folder… and select the folder that contains app.py, train_model.py, requirements.txt, and the data folder.
Step 3 – Open a terminal in VS Code
Use Terminal → New Terminal. The terminal should already be inside the project folder. You can confirm with:

dir
You should see app.py in the listing.

Step 4 – Create and activate a virtual environment
py -m venv .venv
.venv\Scripts\activate
Your prompt now starts with (.venv).

PowerShell error? If activating is blocked by the execution policy, run this once in the same window and try again:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
Alternatively use Command Prompt (cmd) instead of PowerShell.

Step 5 – Install the dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Step 6 – Train the model
python train_model.py
This prints the dataset size, the cross-validated accuracy, and the strongest words per label, then writes:

model/classifier.joblib – the trained pipeline
model/metrics.json – accuracy and other training details
You only need to run this when you change the dataset. The app also trains automatically if the model file is missing.

Step 7 – Launch the app
python -m streamlit run app.py
Streamlit prints a local URL (normally http://localhost:8501) and your browser opens automatically. If it does not, open that URL manually.

Stop the app with Ctrl+C in the terminal.

Step 8 – Run the tests (optional but recommended)
python -m pytest -q
All tests should pass. They use only the bundled sample data and never call the network.

Suggested VS Code interpreter
Press Ctrl+Shift+P → "Python: Select Interpreter" and choose .venv\Scripts\python.exe. Then VS Code uses the same environment in its terminal and test panel.

3. How to use the app
Type a statement in the text box, or pick one of the examples.
Click Analyse statement.
Read the left card, Model confidence: the predicted topic, the confidence percentage with a progress bar, a bar chart of the probability for every topic, and the runner-up.
Read the right card, Reference verification: the coloured verdict badge (✅ Supported / ❌ Contradicted / ⚠️ Insufficient Evidence), a plain-language explanation, and the matching reference record with its expected value, alternatives, source, and matched keywords.
Expand All class probabilities inside the left card for the numeric table.
Expand 📚 Reference facts under the results to see every reference row, and ℹ️ Project information and model details for the pipeline, model metrics, top words per topic, the file map, and the limitations.
Click Retrain model in the sidebar if you edit data/training_data.csv.
Open the Why these two results are independent section to see the separation rule.
The two cards sit side by side on a normal screen and stack on a narrow window.

What each card contains
Left card – Model confidence	Right card – Reference verification
Predicted topic	Coloured verdict badge
Confidence percentage + progress bar	Plain-language explanation
Bar chart of every topic's probability	Matching reference record (row id, topic, statement)
Runner-up topic and score	Expected value, alternatives, source, matched keywords
Expandable numeric probability table	Note that the classifier was not consulted
Example statements to try:

Statement	Expected verification
Water boils at 100 degrees Celsius at sea level.	Supported
Water boils at 90 degrees Celsius at sea level.	Contradicted
The capital of France is Paris.	Supported
The capital of France is Berlin.	Contradicted
The capital of France is not Paris.	Contradicted
Water boils when it is heated.	Insufficient Evidence
The height of Mount Everest is 8849 feet.	Insufficient Evidence
My favourite colour is blue.	Insufficient Evidence
4. How the machine learning part works
text_classifier.py builds a scikit-learn Pipeline with two steps:

TfidfVectorizer – turns the sentence into numbers. TF-IDF gives high weight to words that are frequent in one document but rare across the dataset. Single words and word pairs (ngram_range=(1, 2)) are used, English stop words are removed, and sublinear term frequency scaling dampens very long documents.
LogisticRegression – a linear classifier that outputs one probability per topic label. predict_proba gives the probability distribution; the highest value is shown as the confidence.
Training details:

Data: data/training_data.csv with columns text,label (180 examples, 36 per topic).
Validation: 5-fold stratified cross-validation, so the reported accuracy comes from data the model did not train on in that fold.
Output: model/classifier.joblib (vectoriser + classifier) and model/metrics.json.
In the bundled setup the 5-fold cross-validated accuracy is about 82% (versus 20% for random guessing). Expect small changes between library versions, and remember that this number measures topic classification only, not factual accuracy.

Confidence is not calibrated truth. With a tiny synthetic dataset, probabilities mainly reflect how similar the wording is to the training examples. They are useful for teaching, not for decisions.

engine.py is the glue: load_engine() loads the saved model and the reference data, and engine.analyse(text) returns one Analysis object holding both a text_classifier.Prediction and a verification.VerificationResult. The two remain separate fields; the engine never lets one influence the other.

5. How verification works (and why it is separate)
verification.py is intentionally not machine learning. It uses a handful of readable rules against data/knowledge_base.csv:

Column	Meaning
id	Unique row id, shown in the app
topic	Short human-readable topic
statement	The reference fact
keywords	Semicolon-separated phrases that must all appear in the text
expected_value	The value the reference supports (100, paris, …)
unit	Unit for numeric values (c, m, m/s, hours, …)
alternatives	Known-wrong values for text rows, separated by ;
source	Where the reference value comes from
Decision rules:

A row is a candidate only if every keyword appears in the text (a small suffix-stripping step lets boils, boiling, and boiled match the keyword boil).
If two rows match equally strongly, the verdict is Insufficient Evidence (ambiguous evidence).
Numeric rows: the text must contain a number in a compatible unit.
Matching number → Supported
Different number → Contradicted (the reason quotes both numbers)
No comparable number, a different unit, or an unrecognised unit → Insufficient Evidence
Text rows: the expected value → Supported; a listed alternative → Contradicted.
A negated claim ("water does not boil at 100 °C") → Contradicted.
Everything else → Insufficient Evidence.
Because the reference data is tiny, "Insufficient Evidence" is the normal answer for most real sentences – and that is the honest outcome.

Adding your own reference rows
Append a row to data/knowledge_base.csv using the same columns. Examples:

kb-018,Capital of Italy,"The capital of Italy is Rome.",capital;italy,rome,,milan;naples;turin,"World fact reference"
kb-019,Boiling point of ethanol,"Ethanol boils at about 78 degrees Celsius at sea level.",ethanol;boil,78,c,,"CRC Handbook of Chemistry and Physics"
Restart or reload the app. Numbers must be written with digits (for example 78 °C) to be compared; written-out numbers such as "seventy-eight" are treated as no number.

6. Project structure
.
├── app.py                    # Streamlit user interface (two result cards)
├── engine.py                 # Loads the model + reference data, returns both signals
├── train_model.py            # Command-line training script
├── text_classifier.py        # TF-IDF + logistic regression helpers
├── verification.py           # Rule-based reference verification
├── requirements.txt
├── setup.bat                 # Windows: create .venv, install dependencies, train
├── run.bat                   # Windows: launch the Streamlit app
├── README.md
├── data/
│   ├── training_data.csv     # 180 labelled example sentences
│   └── knowledge_base.csv    # 17 reference rows with sources
├── model/                    # Created by train_model.py (classifier.joblib, metrics.json)
└── tests/
    ├── conftest.py
    ├── test_text_classifier.py
    ├── test_verification.py
    ├── test_engine.py        # engine + knowledge-base path resolution
    └── test_app_smoke.py     # headless UI: cards, chart, expanders, verdicts
Relative paths are resolved from the project folder in which the files live, so you can start the app from that folder in VS Code, PowerShell, or Command Prompt.

The reference data ships as data/knowledge_base.csv. engine.py also accepts a knowledge_base.csv placed directly in the project root, so either layout works; the sidebar shows which file was loaded. Keep only one copy so the two cannot drift apart.

7. Troubleshooting
Problem	Fix
'python' is not recognized	Reinstall Python with "Add python.exe to PATH", or use py instead of python.
Activate.ps1 cannot be loaded	Run Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned, or use Command Prompt.
ModuleNotFoundError: streamlit	The virtual environment is not active, or dependencies are missing – run .venv\Scripts\activate then python -m pip install -r requirements.txt.
Model file not found	Run python train_model.py (the app can also train automatically on first load).
streamlit command not found	Always use python -m streamlit run app.py.
Port 8501 already in use	Run python -m streamlit run app.py --server.port 8502.
Browser does not open	Open http://localhost:8501 manually.
Tests cannot import verification	Run them from the project folder with python -m pytest -q.
8. Limitations and responsible use
The training data is a small, hand-written teaching sample, so accuracy numbers are optimistic and the model knows only five topics.
Confidence is a model score, not a probability that a sentence is factually correct.
The knowledge base covers 17 facts only. Absence of a verdict says nothing about truth.
Numeric verification understands only the units listed in verification.py; other units intentionally produce Insufficient Evidence.
Negation detection is a simple keyword rule and can be fooled by unusual phrasing.
Nothing here is a substitute for real fact checking, and the project must not be used for medical, legal, or financial decisions.
9. Quick command reference
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py

REM optional: both signals from the command line, no browser
python engine.py "Water boils at 100 degrees Celsius at sea level."
python -m pytest -q
Or, from Explorer, just double-click setup.bat and then run.bat.
