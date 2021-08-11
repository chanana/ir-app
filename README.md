# ir-app

Visualize clustered data from IR studies

## 1. Clone repo, setup directory, create new environment, and install requirements
```bash
git clone https://github.com/chanana/ir-app.git
cd ir-app
mkdir img # make directory to save png images
python -m venv env # make virtual environment
. env/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt # install requirements
```

## 2. Run a python server in the background and then run the app
```bash
python -m http.server & # run server in background
python ir_dash_2.0.py # run app
```
This takes about 30 seconds before the app actually launches so please be patient.

## 3. View the app
navigate to http://127.0.0.1:8050/

## 4. Exit the app 
```bash
^C # stops the app
fg # brings python server to foreground
^C # stops server
```