@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python walk_forward.py
streamlit run app.py
