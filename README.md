# Global Poverty and Economic Inequality

https://globalpovertyandeconomicineuapplity-3vcsoixqkwqpwfjqeagxwj.streamlit.app/

Professional notebook report and Streamlit dashboard for `global_poverty_economic_inequality.csv`.

Created by Hieu Nguyen

## What is included

- `Global_Poverty_Economic_Inequality_Report.ipynb` - executed notebook with setup, data quality checks, dashboard visuals, segmentation, and strategic insights.
- `app.py` - Streamlit web dashboard with filters, Plotly visuals, rankings, archetypes, and strategic insights.
- `global_poverty_economic_inequality.csv` - source dataset.
- `requirements.txt` - reproducible Python dependencies.
- `share/jupyter/kernels/global-poverty-venv/kernel.json` - local kernel spec that points to the project `venv`.

## Run Streamlit locally

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\streamlit.exe run app.py
```

## Run the notebook locally

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m ipykernel install --prefix . --name global-poverty-venv --display-name "Python (Global Poverty venv)"
```

Then open the notebook and select `Python (Global Poverty venv)`.

## Deploy to Streamlit Community Cloud

Use this GitHub repository as the source and set the main file path to:

```text
app.py
```

