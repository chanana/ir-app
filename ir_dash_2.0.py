# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
import json
import pathlib

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from dash.dependencies import Input, Output, State
from dash_table import DataTable
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

smiles = pd.read_csv(
    "csv_data/smiles_fixed.csv", encoding="latin-1", header=0, index_col=0
)
colors = px.colors.qualitative.Plotly  # plotly's default colors

blank_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
default_structure_image = "data:image/png;base64,{}".format(blank_image)
website_url = "http://0.0.0.0:8000"
host = "127.0.0.1"


def mol2png(sample_name):
    smile = smiles.loc[sample_name, "SMILES"] if sample_name in smiles.index else "CC"
    mol = Chem.MolFromSmiles(smile)
    Chem.Draw.MolToFile(mol, "img/" + str(sample_name) + ".png", size=(400, 400))


def add_text_to_file(sample_name):
    img = Image.open("img/" + sample_name + ".png")
    canvas = Image.new("RGB", (400, 450), (255, 255, 255))
    canvas.paste(img, (0, 0))
    canvas_draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=18)
    if sample_name not in smiles.index:
        canvas_draw.text(
            (0, 400),
            sample_name + ": Not Found",
            outline="black",
            fill="black",
            font=font,
        )
    else:
        canvas_draw.text(
            (20, 400),
            sample_name + ": " + smiles.loc[sample_name, "Trivial name"],
            outline="black",
            fill="black",
            font=font,
        )
    canvas.save("img/" + sample_name + ".png")


def get_png_data(sample_name):
    image = base64.b64encode(open("img/" + str(sample_name) + ".png", "rb").read())
    image_data = "data:image/png;base64,{}".format(image.decode())
    return image_data


def smiles_to_grid_png(sample_name_list):
    smile_list = [
        smiles.loc[i, "SMILES"] if i in smiles.index else "CC" for i in sample_name_list
    ]
    mols = [Chem.MolFromSmiles(i) for i in smile_list]
    legends = [
        smiles.loc[i, "Trivial name"] if i in smiles.index else "None"
        for i in sample_name_list
    ]
    grid = Chem.Draw.MolsToGridImage(mols=mols, legends=legends, molsPerRow=5)
    grid.save("img/mol.png")


def json_to_sample_names_list(json_data):
    return [
        json_data["points"][i]["customdata"] for i in range(0, len(json_data["points"]))
    ]


def pattern_in_list(list, pattern):
    truth = [True if pattern in i else False for i in list]
    return truth


app.layout = dbc.Tabs(
    [
        dbc.Tab(
            id="clustering-tab",
            label="Clustering",
            children=[
                dbc.Container(
                    [
                        html.Div(
                            [  # dataset selection + submit button
                                dbc.Container(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Collection"),
                                                        dcc.Dropdown(
                                                            id="collection-choice",
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": i.upper(),
                                                                    "value": i,
                                                                }
                                                                for i in [
                                                                    "az",
                                                                    "mtp",
                                                                    "mtp and az",
                                                                ]
                                                            ],
                                                            value="mtp and az",
                                                        ),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Region of Spectrum"
                                                        ),
                                                        dcc.Dropdown(
                                                            id="region-choice",
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": i.title(),
                                                                    "value": i,
                                                                }
                                                                for i in [
                                                                    "full",
                                                                    "fingerprint",
                                                                ]
                                                            ],
                                                            value="fingerprint",
                                                        ),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Scaling"),
                                                        dcc.Dropdown(
                                                            id="scaling-choice",
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": i.title(),
                                                                    "value": i,
                                                                }
                                                                for i in [
                                                                    "scaled",
                                                                    "normalized",
                                                                ]
                                                            ],
                                                            value="scaled",
                                                        ),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Visualization"),
                                                        dcc.Dropdown(
                                                            id="visualization-choice",
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": i.upper(),
                                                                    "value": i,
                                                                }
                                                                for i in ["tsne", "pca"]
                                                            ],
                                                            value="tsne",
                                                        ),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Button(
                                                    id="load-dataset-button",
                                                    children="Load",
                                                    className="btn btn-primary",
                                                ),
                                            ]
                                        )
                                    ],
                                    fluid=True,
                                ),
                                # Clustering
                                dbc.Container(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "Clustering",
                                                                style={
                                                                    "text-align": "center"
                                                                },
                                                            ),
                                                            dcc.Graph(id="clustering"),
                                                        ]
                                                    ),
                                                    width=8,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                id="selected-data-title",
                                                                children="Selected Spectra",
                                                                style={
                                                                    "text-align": "center"
                                                                },
                                                            ),
                                                            dcc.Graph(
                                                                id="selected-data-graph"
                                                            ),
                                                        ]
                                                    ),
                                                    width=4,
                                                ),
                                            ]
                                        )
                                    ],
                                    className="border border-secondary",
                                    fluid=True,
                                ),
                                # Table of structures selected
                                dbc.Container(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        DataTable(
                                                            id="table",
                                                            columns=[
                                                                {
                                                                    "name": i,
                                                                    "id": i,
                                                                    "presentation": "markdown",
                                                                }
                                                                for i in [
                                                                    "sample",
                                                                    "structure",
                                                                ]
                                                            ],
                                                            data=[],
                                                            style_table={
                                                                "overflow": "auto",
                                                                "display": "block",
                                                                "height": "800px",
                                                            },
                                                        )
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                dcc.RadioItems(
                                                                    id="display-trace-1-radio",
                                                                    options=[
                                                                        {
                                                                            "label": i,
                                                                            "value": i,
                                                                        }
                                                                        for i in [
                                                                            "Display trace",
                                                                            "Hide",
                                                                        ]
                                                                    ],
                                                                    value="Display trace",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="sample-name-dropdown-1",
                                                                    clearable=False,
                                                                    placeholder="loading...",
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="add-trace-1-clustering",
                                                                    options=[
                                                                        {
                                                                            "label": i,
                                                                            "value": i,
                                                                        }
                                                                        for i in [
                                                                            "Display in clustering",
                                                                            "Hide",
                                                                        ]
                                                                    ],
                                                                    value="Hide",
                                                                ),
                                                                html.Div(
                                                                    id="trace-1",
                                                                    children=[
                                                                        dcc.RadioItems(
                                                                            id="display-structure-1-radio",
                                                                            options=[
                                                                                {
                                                                                    "label": i,
                                                                                    "value": i,
                                                                                }
                                                                                for i in [
                                                                                    "Display structure",
                                                                                    "Hide",
                                                                                ]
                                                                            ],
                                                                            value="Display structure",
                                                                        ),
                                                                        html.Img(
                                                                            id="structure-1"
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    width=3,
                                                    className="border border-secondary",
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                dcc.RadioItems(
                                                                    id="display-trace-2-radio",
                                                                    options=[
                                                                        {
                                                                            "label": i,
                                                                            "value": i,
                                                                        }
                                                                        for i in [
                                                                            "Display trace",
                                                                            "Hide",
                                                                        ]
                                                                    ],
                                                                    value="Display trace",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="sample-name-dropdown-2",
                                                                    clearable=False,
                                                                    placeholder="loading...",
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="add-trace-2-clustering",
                                                                    options=[
                                                                        {
                                                                            "label": i,
                                                                            "value": i,
                                                                        }
                                                                        for i in [
                                                                            "Display in clustering",
                                                                            "Hide",
                                                                        ]
                                                                    ],
                                                                    value="Hide",
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="display-structure-2-radio",
                                                                    options=[
                                                                        {
                                                                            "label": i,
                                                                            "value": i,
                                                                        }
                                                                        for i in [
                                                                            "Display structure",
                                                                            "Hide",
                                                                        ]
                                                                    ],
                                                                    value="Display structure",
                                                                ),
                                                                html.Img(
                                                                    id="structure-2"
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    width=3,
                                                    className="border border-secondary",
                                                ),
                                            ]
                                        )
                                    ],
                                    fluid=True,
                                ),
                                # Spectral traces
                                dbc.Container(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H3(
                                                                    "Spectra",
                                                                    style={
                                                                        "text-align": "center"
                                                                    },
                                                                ),
                                                                dcc.Graph(id="spectra"),
                                                            ]
                                                        )
                                                    ],
                                                    width=12,
                                                )
                                            ]
                                        )
                                    ],
                                    fluid=True,
                                    className="border border-secondary",
                                ),
                                # Debug text area
                                dbc.Container(
                                    [
                                        dbc.Row(
                                            [
                                                html.Div(
                                                    [
                                                        dcc.Textarea(
                                                            id="debug-area",
                                                            value="selectedData",
                                                        )
                                                    ],
                                                    style={"display": "none"},
                                                )
                                            ]
                                        )
                                    ],
                                    fluid=True,
                                ),
                                # DCC Store - stores the dataset to be loaded - really important, not sure why it's all the way down here
                                dcc.Store(id="dataset-clustering"),
                                dcc.Store(id="dataset-collection"),
                                dcc.Store(id="dataset-collection-index"),
                            ]
                        )
                    ],
                    fluid=True,
                    className="border border-secondary",
                )
            ],
        ),
        dbc.Tab(
            id="dendrogram-tab",
            label="Dendrogram",
            children=[dbc.Container([html.H1("COMING SOON!")])],
        ),
    ]
)


# load the data
@app.callback(
    Output("dataset-clustering", "data"),
    Output("dataset-collection", "data"),
    Output("dataset-collection-index", "data"),
    Input("collection-choice", "value"),
    Input("region-choice", "value"),
    Input("scaling-choice", "value"),
    Input("visualization-choice", "value"),
)
def display_clustering(collection, region, scaling, visualization):
    mapping = {
        "az": "az-daughters",
        "mtp": "mtp_reduced",
        "mtp and az": "mtp-az",
        "fingerprint": "finger",
        "normalized": "normed",
    }
    dataset_clustering = (
        "-".join(
            [
                mapping[i] if i in mapping.keys() else i
                for i in [region, scaling, visualization, collection]
            ]
        )
        + ".csv"
    )
    dataset_collection = mapping[collection] + ".csv"
    # results in a file name like "finger-scaled-tsne-az-daughters.csv"

    path_clustering = pathlib.Path("csv_data/" + dataset_clustering)
    path_collection = pathlib.Path("csv_data/" + dataset_collection)

    if path_clustering.exists() and path_collection.exists():
        df_clustering = pd.read_csv(path_clustering, header=0, index_col=0)
        df_collection = pd.read_csv(path_collection, header=0, index_col=0)
        return (
            df_clustering.to_json(orient="split"),
            df_collection.to_json(orient="split"),
            json.dumps(df_collection.index.to_list()),
        )
    else:
        print(f"either {dataset_clustering} or {dataset_collection} don't exist...")


# populate dropdowns
@app.callback(
    Output("sample-name-dropdown-1", "options"),
    Output("sample-name-dropdown-1", "value"),
    Output("sample-name-dropdown-2", "options"),
    Output("sample-name-dropdown-2", "value"),
    Input("dataset-collection-index", "data"),
)
def change_dropdown_options(indices_as_json):
    indices = json.loads(indices_as_json)
    options = [{"label": i, "value": i} for i in indices]
    return options, indices[-1], options, indices[-2]


# draw spectra
@app.callback(
    Output("spectra", "figure"),
    Input("sample-name-dropdown-1", "value"),
    Input("sample-name-dropdown-2", "value"),
    Input("display-trace-1-radio", "value"),
    Input("display-trace-2-radio", "value"),
    Input("dataset-collection", "data"),
)
def update_spectrum(
    sample_name, sample_name_2, trace_1, trace_2, dataset_collection_json
):
    spectra = pd.read_json(dataset_collection_json, orient="split")

    signal_1 = spectra.loc[sample_name, :]
    signal_2 = spectra.loc[sample_name_2, :]

    x = [int(i) for i in spectra.columns]
    fig = go.Figure()
    if "D" in trace_1:
        fig.add_trace(
            go.Scatter(
                x=x, y=signal_1, mode="lines", name=sample_name, line_color=colors[0]
            )
        )
    if "D" in trace_2:
        fig.add_trace(
            go.Scatter(
                x=x, y=signal_2, mode="lines", name=sample_name_2, line_color=colors[1]
            )
        )
    fig.update_layout(transition_duration=500, transition_easing="linear")
    return fig


# draw clustering
@app.callback(
    Output("clustering", "figure"),
    Input("sample-name-dropdown-1", "value"),
    Input("sample-name-dropdown-2", "value"),
    Input("add-trace-1-clustering", "value"),
    Input("add-trace-2-clustering", "value"),
    Input("dataset-clustering", "data"),
    Input("dataset-collection-index", "data"),
)
def update_clustering(
    sample_1, sample_2, trace_1, trace_2, data_as_json, indices_as_json
):
    df = pd.read_json(data_as_json, orient="split")
    indices = json.loads(indices_as_json)

    fig = go.Figure()
    x = df.tsne_x.to_list()
    y = df.tsne_y.to_list()
    df["symbols"] = "circle"
    names = df.names.to_list()
    df["colors"] = "#000000"

    # check if MTP data present
    truth_mtp = pattern_in_list(names, "mtp")
    if any(truth_mtp):
        df.loc[truth_mtp, "colors"] = colors[-4]

    # check if AZ data present
    truth_az = pattern_in_list(names, "16210300")
    if any(truth_az):
        df.loc[truth_az, "colors"] = colors[-3]

    if "D" in trace_1:
        df.loc[(df["names"] == sample_1), "colors"] = colors[0]
        df.loc[(df["names"] == sample_1), "symbols"] = "star"
    #     loc_1 = indices.index(sample_1)
    #     symbols[loc_1] = "star"
    #     df.loc[loc_1, "colors"] = colors[0]
    if "D" in trace_2:
        df.loc[(df["names"] == sample_2), "colors"] = colors[1]
        df.loc[(df["names"] == sample_2), "symbols"] = "cross"
    #     loc_2 = indices.index(sample_2)
    #     symbols[loc_2] = "cross"
    #     df.loc[loc_2] = colors[1]
    fig.add_trace(
        go.Scatter(
            x=df.loc[truth_mtp, "tsne_x"].to_list(),
            y=df.loc[truth_mtp, "tsne_y"].to_list(),
            name="MTP",
            mode="markers",
            marker_symbol=df.loc[truth_mtp, "symbols"].to_list(),
            marker_color=df.loc[truth_mtp, "colors"].to_list(),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[truth_az, "tsne_x"].to_list(),
            y=df.loc[truth_az, "tsne_y"].to_list(),
            name="AZ",
            mode="markers",
            marker_symbol=df.loc[truth_az, "symbols"].to_list(),
            marker_color=df.loc[truth_az, "colors"].to_list(),
        )
    )
    # fig = px.scatter(df, x='tsne_x', y='tsne_y', color='colors', labels={})
    fig.update_traces(
        marker_size=8,
        marker_line_color="black",
        marker_line_width=1,
        customdata=df.names.to_list(),
        hovertemplate="<br></br><b>%{customdata}<b><br></br><extra></extra>",
    )
    fig.update_layout(transition_duration=250, transition_easing="circle-in-out")
    return fig


# spectra of selected points in clustering to clustering
@app.callback(
    Output("selected-data-graph", "figure"),
    Input("clustering", "selectedData"),
    State("dataset-collection", "data"),
)
def draw_selected_data(selected_data, collection_data):
    if selected_data:
        # only take the top 10 selected elements
        sample_name_list = json_to_sample_names_list(selected_data)[0:10]
        spectra = pd.read_json(collection_data, orient="split")
        dff = spectra.loc[sample_name_list]
        dff = dff.transpose()
        fig = px.line(dff)
    else:
        fig = go.Figure(go.Scatter())
    return fig


# Structures
@app.callback(
    Output("structure-1", "src"),
    Output("structure-2", "src"),
    Input("sample-name-dropdown-1", "value"),
    Input("sample-name-dropdown-2", "value"),
    Input("display-structure-1-radio", "value"),
    Input("display-structure-2-radio", "value"),
)
def display_png(sample_name_1, sample_name_2, struct_1, struct_2):
    samples = [sample_name_1, sample_name_2]
    list(map(mol2png, samples))
    list(map(add_text_to_file, samples))

    image_1_data = (
        get_png_data(sample_name_1) if "D" in struct_1 else default_structure_image
    )
    image_2_data = (
        get_png_data(sample_name_2) if "D" in struct_2 else default_structure_image
    )

    return image_1_data, image_2_data


# structures of selected points to table
@app.callback(
    Output("table", "data"),
    Output("table", "style_data_conditional"),
    Input("clustering", "selectedData"),
)
def table_display(selected_data):
    if selected_data:
        table_data = []
        sample_names = json_to_sample_names_list(selected_data)
        for s in sample_names:
            mol2png(s)
            image_url = "![" + s + "](" + website_url + "/img/" + str(s) + ".png)"
            chemical_name = (
                smiles.loc[s, "Trivial name"] if s in smiles.index else "None"
            )
            row = {"sample": s + ": " + chemical_name, "structure": image_url}
            table_data.append(row)
        # each row is a dict
        style_data = [
            {
                "if": {"row_index": c[0], "column_id": "structure"},
                "border": "2px solid " + c[1],
            }
            for c in enumerate(colors)
        ]
    else:
        mol2png("none")  # draws a – in image
        image_urls = ["![None](" + website_url + "/img/none.png)"] * 2
        table_data = [{"sample": "none", "structure": url} for url in image_urls]
        style_data = []
    return table_data, style_data


# @app.callback(
#     Output('debug-area', 'value'),
#     Input('selection-stuff', 'children')
# )
# def debug_area_display(input):
#     # import json
#     print(input)


if __name__ == "__main__":
    app.run_server(debug=True, host=host)
