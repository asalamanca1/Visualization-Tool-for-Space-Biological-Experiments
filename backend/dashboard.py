from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, Experiment
import os
import json
import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import re
from PIL import Image
import glob
# For AI Image Generation
from openai import AzureOpenAI
import openai
import requests

os.environ["AZURE_OPENAI_API_KEY"] = '98375ee658e14f8c9e97b085af12b564'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://iav-imagegen.openai.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01'

def sanitize_filename(value):
    # Replace any character that is not a word character (alphanumeric) or underscore
    return re.sub(r'[^\w\s-]', '', value).replace(" ", "_").lower()

def load_and_process_data(experiment_id):
    # Retrieve the experiment
    experiment = Experiment.query.get_or_404(experiment_id)
    experiment_title = experiment.url.split('/')[-1]
    samples_csv = f'datasets/{experiment_title}/{experiment_title}_samples.csv'
    assays_csv = f'datasets/{experiment_title}/{experiment_title}_assays.csv'
    
    # Load DataFrames
    samples_df = pd.read_csv(samples_csv)
    assays_df = pd.read_csv(assays_csv)
    
    # Merge DataFrames
    merged_df = pd.merge(samples_df, assays_df, on='Sample Name', how='inner')
    
    # Convert 'Parameter Value:' columns to numeric where possible
    parameter_columns = [col for col in merged_df.columns if col.startswith('Parameter Value:')]
    
    def extract_numeric_value(series):
        series = series.astype(str)  # Ensure the series is of string type
        extracted = series.str.extract(r'(\d+\.?\d*)')[0]
        return pd.to_numeric(extracted, errors='coerce')
    
    for col in parameter_columns:
        merged_df[col] = extract_numeric_value(merged_df[col])
    
    # Identify numeric columns
    numeric_columns = merged_df.select_dtypes(include=['number']).columns.tolist()
    numeric_parameter_columns = [col for col in parameter_columns if col in numeric_columns]
    
    return merged_df, samples_df, assays_df, parameter_columns, numeric_parameter_columns

def create_results_layout(samples_csv, assays_csv):

    # Check if files exist
    if not os.path.exists(samples_csv) or not os.path.exists(assays_csv):
        return html.Div(f"Data files not found for this experiment.")
    
    samples_df = pd.read_csv(samples_csv)
    assays_df = pd.read_csv(assays_csv)
    merged_df = pd.merge(samples_df, assays_df, on='Sample Name', how='inner')
    
    # Convert 'Parameter Value:' columns to numeric
    parameter_columns = [col for col in merged_df.columns if col.startswith('Parameter Value:')]
    
    def extract_numeric_value(series):
        series = series.astype(str)  # Ensure the series is of string type
        extracted = series.str.extract(r'(\d+\.?\d*)')[0]
        return pd.to_numeric(extracted, errors='coerce')
    
    for col in parameter_columns:
        merged_df[col] = extract_numeric_value(merged_df[col])
    
    numeric_columns = merged_df.select_dtypes(include=['number']).columns.tolist()
    numeric_parameter_columns = [col for col in parameter_columns if col in numeric_columns]
    
    # Extract columns for plotting
    characteristics_columns = [col for col in samples_df.columns if col.startswith('Characteristics:')]
    factor_columns = [col for col in samples_df.columns if col.startswith('Factor Value:')]
    
    # Create the layout
    layout = html.Div(
        style={
            'background-image': 'linear-gradient(to bottom, #04142c, #083343, #2d5153, #596f64, #888d7e)',
            'min-height': '100vh',
            'padding': '20px'
        },
        children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H1("Experiment Results", style={'color': 'white'}), className='mb-4')
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Factor Variable for Box Plot", style={'color': 'white'}),
                        dcc.Dropdown(
                            id='box-factor-variable',
                            options=[{'label': col, 'value': col} for col in factor_columns],
                            value=factor_columns[0] if factor_columns else None
                        ),
                        html.Label("Select Numeric Parameter for Box Plot", style={'color': 'white'}),
                        dcc.Dropdown(
                            id='box-numeric-variable',
                            options=[{'label': col, 'value': col} for col in numeric_parameter_columns],
                            value=numeric_parameter_columns[0] if numeric_parameter_columns else None
                        ),
                        dcc.Graph(id='box-plot')
                    ], md=6),
                    dbc.Col([
                        html.Label("Select Variable for Pie Chart", style={'color': 'white'}),
                        dcc.Dropdown(
                            id='pie-variable',
                            options=[{'label': col, 'value': col} for col in characteristics_columns + factor_columns],
                            value=(factor_columns[0] if factor_columns else (characteristics_columns[0] if characteristics_columns else None))
                        ),
                        dcc.Graph(id='sample-composition-pie')
                    ], md=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Factor Variable for Bar Chart", style={'color': 'white'}),
                        dcc.Dropdown(
                            id='grouped-bar-factor-variable',
                            options=[{'label': col, 'value': col} for col in factor_columns],
                            value=factor_columns[0] if factor_columns else None
                        ),
                        html.Label("Select Numeric Parameter for Bar Chart", style={'color': 'white'}),
                        dcc.Dropdown(
                            id='grouped-bar-numeric-variable',
                            options=[{'label': col, 'value': col} for col in numeric_parameter_columns],
                            value=numeric_parameter_columns[0] if numeric_parameter_columns else None
                        ),
                        dcc.Graph(id='grouped-bar-chart')
                    ], md=6),
                    dbc.Col([
                        html.Label("Select a Factor to Generate Heatmap", style={'color': 'white'}),
                        dcc.Dropdown(
                            id='heatmap-trigger',
                            options=[{'label': 'Generate Heatmap', 'value': 'generate'}],
                            value='generate'
                        ),
                        dcc.Graph(id='heatmap')
                    ], md=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Search Samples", style={'color': 'white'}),
                        dcc.Input(id='table-search', type='text', placeholder='Search...'),
                        dash_table.DataTable(
                            id='data-table',
                            columns=[{'name': col, 'id': col} for col in samples_df.columns],
                            data=samples_df.to_dict('records'),
                            filter_action='native',
                            sort_action='native',
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_header={'backgroundColor': '#04142c', 'color': 'white'},
                            style_cell={'backgroundColor': '#2d5153', 'color': 'white'}
                        )
                    ], md=12),
                ]),
            ], fluid=True)
        ]
    )

    return layout


def get_experiment_image(factor_value):    
    formatted_name = sanitize_filename(factor_value)
    formatted_name = formatted_name.replace(" ", "").lower()
    image_dir = 'assets'
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"{formatted_name}.png")

    # Check if image already exists
    if os.path.exists(image_path):
        return image_path

    # Set up the AzureOpenAI client
    client = AzureOpenAI(
        api_version="2024-02-01",  # Update this if needed
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Generate the prompt
    PROMPT = f"Generate an icon without text for the following: {factor_value}"

    # Generate the image
    result = client.images.generate(
        model="dalle3",  # Adjust the model name if necessary
        prompt=PROMPT,
        n=1,
        size="1024x1024"  # Specify the image size
    )

    # Parse the JSON response
    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]  

    # Download the image
    img_response = requests.get(image_url)

    # Save the image as a PNG file
    with open(image_path, 'wb') as img_file:
        img_file.write(img_response.content)

    # Return the saved image path
    return image_path
        

def generate_all_images(df):
    # Collect all unique values needing images
    factor_columns = [col for col in df.columns if 'Factor Value' in col and 'Duration' not in col]
    unique_values = set()
    for factor in factor_columns:
        values = df[factor].dropna().unique()
        unique_values.update(values)

    # Include organism names
    organism_values = df['Characteristics: Organism'].dropna().unique()
    unique_values.update(organism_values)

    # Generate images for all unique values
    for value in unique_values:
        get_experiment_image(value)

def create_dash_layout(csv_file_path, experiment_id):
    df = pd.read_csv(csv_file_path)

    # Determine the initial factor order
    initial_factor_columns = [col for col in df.columns if 'Factor Value' in col and 'Duration' not in col]

    # Store the initial order of factors in dcc.Store
    factor_order_store = dcc.Store(id='factor-order-store', data=initial_factor_columns)

    # Store the initial order for reset purposes
    initial_order_store = dcc.Store(id='initial-order-store', data=initial_factor_columns)

    # Create filter components with reordering buttons
    filter_components = html.Div(
        id='filter-container',
        style={'display': 'flex', 'flex-wrap': 'wrap', 'align-items': 'flex-start'}
    )

    # Cytoscape component
    cytoscape_graph = cyto.Cytoscape(
        id='cytoscape-tree',
        elements=[],
        layout={
            'name': 'dagre',
            'rankDir': 'LR',
            'nodeSep': 70,
            'edgeSep': 10,
            'rankSep': 100,
            'ranker': 'network-simplex'
        },
        style={'width': '100%', 'height': '700px'},
        stylesheet=[
            # Root node styling
            {
                'selector': '.root-node',
                'style': {
                    'background-color': '#000000',
                    'shape': 'round-rectangle',
                    'width': 150,
                    'height': 150,
                    'label': 'data(label)',
                    'text-halign': 'center',
                    'text-valign': 'bottom',
                    'text-margin-y': 5,
                    'font-size': 12,
                    'color': 'black',
                    'background-image': 'data(image)',
                    'background-fit': 'cover',
                    'background-clip': 'node',
                    'background-opacity': 1.0,
                }
            },
            # No data node styling
            {
                'selector': '.no-data-node',
                'style': {
                    'background-color': '#000000',
                    'shape': 'round-rectangle',
                    'width': 200,
                    'height': 30,
                    'label': 'data(label)',
                    'text-halign': 'center',
                    'text-valign': 'bottom',
                    'text-margin-y': 5,
                    'font-size': 12,
                    'color': 'white',
                    'background-image': 'data(image)',
                    'background-fit': 'cover',
                    'background-clip': 'node',
                    'background-opacity': 1.0,
                }
            },
            # General node styling
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'text-halign': 'center',
                    'text-valign': 'bottom',
                    'text-margin-y': 5,
                    'font-size': 10,
                    'color': 'black',
                    'background-color': '#FFFFFF',
                    'shape': 'round-rectangle',
                    'width': 70,
                    'height': 70,
                    'padding': '10px',
                    'text-wrap': 'wrap',
                    'text-max-width': '80px',
                    'background-image': 'data(image)',
                    'background-fit': 'cover',
                    'background-clip': 'node',
                    'background-opacity': 1.0,
                }
            },
            # Edge styling
            {
                'selector': 'edge',
                'style': {
                    'width': 2,
                    'line-color': '#000000',
                    'target-arrow-color': '#000000',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': 'data(label)',  # Label based on edge data
                    'text-rotation': 'autorotate',  # Keeps the label aligned with the edge
                    'font-size': 9,
                    'text-background-opacity': 0,  # Optional background for the text
                    'text-background-color': '#ffffff',  # Optional background color
                    'text-background-padding': '2px',  # Optional padding for better visibility
                    'text-margin-y': -10,  # Moves the text above the edge
                }
            }
        ]
    )

    
    layout = html.Div([
    factor_order_store,
    initial_order_store,
    # Container for the reset button with improved styling
    html.Div([
        html.Button('Reset Plot', id='reset-order-button', n_clicks=0,
                    className='btn btn-primary', style={
                        'margin': '10px 0',  # Adds margin above and below the button
                        'width': '300px',    # Sets a fixed width for the button
                        'fontSize': '16px',  # Increases the font size for better visibility
                        'borderRadius': '5px', # Adds rounded corners
                        'cursor': 'pointer',   # Changes cursor to pointer on hover
                        'fontFamily': 'Helvetica, sans-serif'
                    }),
        html.Button("Save as png", id="btn-get-png",className='btn btn-primary', style={
                        'margin': '10px 0',  # Adds margin above and below the button
                        'width': '300px',
                        'margin-left': '40px',    # Sets a fixed width for the button
                        'fontSize': '16px',  # Increases the font size for better visibility
                        'borderRadius': '5px', # Adds rounded corners
                        'cursor': 'pointer',   # Changes cursor to pointer on hover
                        'fontFamily': 'Helvetica, sans-serif'
                    }),
    ], style={
        'display': 'flex',          # Use flexbox layout
        'flexDirection': 'row',     # Align buttons in a row
        'justifyContent': 'center', # Center the buttons horizontally
        'alignItems': 'center',     # Vertically center the buttons
        'textAlign': 'center',      # Align text to the center
        'marginBottom': '20px'      # Adds space below the buttons
    }),
    cytoscape_graph,
    filter_components,
    html.Div(id='node-link', style={'display': 'none'})
])


    return layout

# Helper function to extract numeric value from a duration string like "38 day"
def extract_duration_value(duration_str):
    match = re.search(r'(\d+)', str(duration_str))
    return int(match.group(1)) if match else 0

def create_cytoscape_elements(df, filters, factor_order, experiment_id):
    factor_columns = factor_order

    # Check for duration in factor or parameter columns
    duration_factor_col = next((col for col in df.columns if 'factor value' in col.lower() and 'duration' in col.lower()), None)
    duration_param_col = next((col for col in df.columns if 'duration' in col.lower()), None)

    organism_values = df['Characteristics: Organism'].unique()

    nodes = []
    edges = []

    filtered_df = df.copy()
    for factor, selected_values in filters.items():
        if selected_values is not None:
            filtered_df = filtered_df[filtered_df[factor].isin(selected_values)]

    # Recursive function to create nodes and edges
    def create_nodes_and_edges(parent_node_id, current_depth, current_df):
        if current_depth >= len(factor_columns):
            if duration_param_col:
                # If Duration is a parameter value, create one duration node for each rightmost node
                duration_value = current_df[duration_param_col].unique()[0] if not current_df.empty else 'N/A'
                duration_node_id = f"{parent_node_id}_duration_{sanitize_filename(duration_value)}"

                # Add duration node (as a triangle)
                nodes.append({
                    'data': {
                        'id': duration_node_id,
                        'label': "Results",
                        'url': f'http://localhost:5001/dash/results/{experiment_id}',
                        'image': 'assets/Results.png'
                    }
                })

                # Add an edge from the rightmost factor node to the duration node
                edges.append({
                    'data': {'source': parent_node_id, 'target': duration_node_id, 'label': f'Duration: {duration_value}'}
                })
            elif duration_factor_col:
                # Handle duration as a factor value
                for duration_value in current_df[duration_factor_col].unique():
                    duration_node_id = f"{parent_node_id}_duration_{sanitize_filename(duration_value)}"

                    # Add duration node for each unique duration value
                    nodes.append({
                        'data': {'id': duration_node_id, 'label': "Results", 'url': f'http://localhost:5001/dash/results/{experiment_id}', 'image': 'assets/Results.png'}
                    })

                    # Add an edge to each duration node
                    edges.append({
                        'data': {'source': parent_node_id, 'target': duration_node_id, 'label': f'Duration: {duration_value}'}
                    })
            else:
                # No duration column found in either factor or parameter values
                no_duration_node_id = f"{parent_node_id}_no_duration"

                # Add a node to represent the absence of duration data
                nodes.append({
                    'data': {
                        'id': no_duration_node_id,
                        'label': "Results", 'url': f'http://localhost:5001/dash/results/{experiment_id}', 'image': 'assets/Results.png'}
                })

                # Add an edge from the rightmost factor node to this node
                edges.append({
                    'data': {'source': parent_node_id, 'target': no_duration_node_id}
                })
            return

        factor = factor_columns[current_depth]
        unique_values = current_df[factor].dropna().unique()
        factor_formatted = factor.replace("Factor Value: ", "")
        for value in unique_values:
            sub_filtered_df = current_df[current_df[factor] == value]
            filtered_count = sub_filtered_df.shape[0]
            node_id = f"{parent_node_id}->{factor}_{sanitize_filename(value)}_{current_depth}"
            node_label = f"{factor_formatted} of {parent_node_id.split('_')[1]}:\n{value}\nSamples: {filtered_count}"

            formatted_value = sanitize_filename(value)

            image_mapping = {
                value: f'assets/{formatted_value}.png',
            }

            # Logic to assign images
            if value in image_mapping and os.path.exists(image_mapping[value]):
                image_url = image_mapping[value]
            else:
                image_url = 'assets/default_image.png'

            nodes.append({
                'data': {
                    'id': node_id,
                    'label': node_label,
                    'image': image_url
                }
            })

            edges.append({
                'data': {'source': parent_node_id, 'target': node_id}
            })

            # Recurse for next depth level
            create_nodes_and_edges(node_id, current_depth + 1, sub_filtered_df)

    # Loop over each organism and add corresponding nodes and edges
    for organism_name in organism_values:
        organism_id = f"Organism_{sanitize_filename(organism_name)}"
        organism_df = filtered_df[filtered_df['Characteristics: Organism'] == organism_name]
        sample_count = organism_df.shape[0]
        organism_label = f"{organism_name}\nSamples: {sample_count}"

        # Add root node for the organism
        nodes.append({
            'data': {
                'id': organism_id,
                'label': organism_label,
                'image': get_experiment_image(organism_name)
            },
            'classes': 'root-node'
        })

        # Handle case where there are no matches for filters
        if organism_df.empty:
            no_data_id = f"{organism_id}_no_data"
            nodes.append({
                'data': {
                    'id': no_data_id,
                    'label': 'No data matches the selected filters.',
                    'image': 'assets/default_image.png'
                },
                'classes': 'no-data-node'
            })
            edges.append({
                'data': {'source': organism_id, 'target': no_data_id}
            })
            continue

        # Call recursive function to create nodes and edges for each organism
        create_nodes_and_edges(organism_id, 0, organism_df)

    elements = nodes + edges
    return elements