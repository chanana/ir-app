# ir-app

Visualize clustered data from IR studies

1. Clone repo, setup directory, create new environment, and install requirements
```bash
git clone https://github.com/chanana/ir-app.git
cd ir-app
mkdir img
python -m venv env
. env/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

2. run a python server in the background and then run the app
```bash
python -m http.server &
python ir_dash_2.0.py
```

Step 2 takes about 30 seconds before the app actually launches so please be patient.

3. View the app by navigating to http://127.0.0.1:8050/

4. Exiting the app 
```bash
^C # stops the app
fg # brings python server to foreground
^C # stops server
```