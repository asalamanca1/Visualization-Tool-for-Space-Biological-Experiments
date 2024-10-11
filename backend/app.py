from flask import Flask, request, jsonify, redirect
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
from scraper import download_and_append_csv, scrape_data
from dashboard import load_and_process_data, create_results_layout, create_cytoscape_elements, create_dash_layout, generate_all_images 
import plotly.express as px
import plotly.graph_objects as go
# For AI Image Generation
from openai import OpenAI
import openai
import requests
import sqlite3

# Chatbot integration
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings # Changed import to Hugging Face
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import os
from transformers import AutoTokenizer
import shutil
from langchain.schema import Document
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggingfaceh4/zephyr-7b-alpha")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///experiments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database and CORS
db.init_app(app)
CORS(app)

# Store vector stores and chat history in memory (could be optimized for scaling)
vector_stores = {}
chat_histories = {}
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.chat_template = "<|USER|>: {user_input}\n<|BOT|>: {bot_output}\n"

# Initialize Dash app
dash_app = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dash/',
    external_stylesheets=[dbc.themes.LUX]
)

# Load extra layouts for Cytoscape
cyto.load_extra_layouts()

# Define the Dash app layout with a placeholder for page content
dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
],
    style={'backgroundColor': '#e6e8e6'}, 
    )

@dash_app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname is None:
        # No experiment selected
        return html.Div("No experiment selected.")
    else:
        # Match the results page URL
        match_results = re.match(r'/dash/results/(\d+)', pathname)
        if match_results:
            experiment_id = int(match_results.group(1))
            # Retrieve the experiment from the database
            experiment = Experiment.query.get_or_404(experiment_id)
            # Construct the CSV file paths
            # Return the results layout
            csv_file_path = f'datasets/{experiment.url.split("/")[-1]}/{experiment.url.split("/")[-1]}_samples.csv'
            assay_file_path = f'datasets/{experiment.url.split("/")[-1]}/{experiment.url.split("/")[-1]}_assays.csv'
            return create_results_layout(csv_file_path, assay_file_path)
        # Match the existing Cytoscape graph page
        match_graph = re.match(r'/dash/(\d+)', pathname)
        if match_graph:
            experiment_id = int(match_graph.group(1))
            # Retrieve the experiment from the database
            experiment = Experiment.query.get_or_404(experiment_id)
            # Construct the CSV file path
            csv_file_path = f'datasets/{experiment.url.split("/")[-1]}/{experiment.url.split("/")[-1]}_samples.csv'
            # Return the Cytoscape graph layout
            return create_dash_layout(csv_file_path, experiment_id)
        else:
            return html.Div("Invalid experiment ID.")

@dash_app.callback(
    Output('sample-composition-pie', 'figure'),
    [Input('pie-variable', 'value')],
    [State('url', 'pathname')]
)
def update_pie_chart(selected_variable, pathname):
    experiment_id = int(re.match(r'/dash/results/(\d+)', pathname).group(1))
    _, samples_df, _, _, _ = load_and_process_data(experiment_id)
    if selected_variable not in samples_df.columns:
        return go.Figure(layout={'title': f"Variable '{selected_variable}' not found in data."})
    fig = px.pie(samples_df, names=selected_variable)
    fig.update_layout(title='Sample Composition')
    return fig

@dash_app.callback(
    Output('box-plot', 'figure'),
    [Input('box-factor-variable', 'value'),
     Input('box-numeric-variable', 'value')],
    [State('url', 'pathname')]
)
def update_box_plot(factor_variable, numeric_variable, pathname):
    experiment_id = int(re.match(r'/dash/results/(\d+)', pathname).group(1))
    merged_df, _, _, _, _ = load_and_process_data(experiment_id)
    if factor_variable not in merged_df.columns or numeric_variable not in merged_df.columns:
        return go.Figure(layout={'title': 'Selected variables not found in data.'})
    fig = px.box(
        merged_df,
        x=factor_variable,
        y=numeric_variable,
        points='all'  # Show all points
    )
    fig.update_layout(title=f'{numeric_variable} Distribution by {factor_variable}')
    return fig
## Grouped Bar Chart Callback
@dash_app.callback(
    Output('grouped-bar-chart', 'figure'),
    [Input('grouped-bar-factor-variable', 'value'),
     Input('grouped-bar-numeric-variable', 'value')],
    [State('url', 'pathname')]
)
def update_grouped_bar_chart(factor_variable, numeric_variable, pathname):
    experiment_id = int(re.match(r'/dash/results/(\d+)', pathname).group(1))
    merged_df, _, _, _, _ = load_and_process_data(experiment_id)
    if factor_variable not in merged_df.columns or numeric_variable not in merged_df.columns:
        return go.Figure(layout={'title': 'Selected variables not found in data.'})
    grouped = merged_df.groupby(factor_variable)[numeric_variable].mean().reset_index()
    fig = px.bar(
        grouped,
        x=factor_variable,
        y=numeric_variable
    )
    fig.update_layout(title=f'Mean {numeric_variable} by {factor_variable}')
    return fig

## Heatmap Callback
## Heatmap Callback
@dash_app.callback(
    Output('heatmap', 'figure'),
    [Input('heatmap-trigger', 'value'),  # Trigger Input
     State('url', 'pathname')]  # URL State
)
def update_heatmap(trigger, pathname):
    # Ensure the trigger is active (i.e., value is 'generate')
    if trigger != 'generate':
        return go.Figure(layout={'title': 'Heatmap not generated.'})
    
    # Extract experiment ID from the URL
    experiment_id = int(re.match(r'/dash/results/(\d+)', pathname).group(1))
    merged_df, _, _, _, numeric_parameter_columns = load_and_process_data(experiment_id)
    
    # Check if we have numeric data for the heatmap
    if not numeric_parameter_columns:
        return go.Figure(layout={'title': 'No numeric parameters available for heatmap.'})

    numeric_data = merged_df[numeric_parameter_columns]
    if numeric_data.empty or len(numeric_data.columns) < 2:
        return go.Figure(layout={'title': 'Not enough numeric data available for heatmap.'})

    # Generate the correlation matrix
    corr_matrix = numeric_data.corr()
    shortened_columns = {col: col[:15] + '...' if len(col) > 15 else col for col in corr_matrix.columns}
    # Create the heatmap using Plotly Express
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='Viridis'
    )
    fig.update_xaxes(tickvals=list(range(len(corr_matrix.columns))), ticktext=[shortened_columns[col] for col in corr_matrix.columns])
    fig.update_yaxes(tickvals=list(range(len(corr_matrix.columns))), ticktext=[shortened_columns[col] for col in corr_matrix.columns])

    # Rotate x-axis labels and adjust font size
    fig.update_layout(
        title='Correlation Heatmap of Numeric Parameters',
        xaxis_tickangle=-45,  # Rotate x-axis labels
        height=510,  # Increase heatmap size
        font=dict(size=10)  # Adjust font size
    )

    return fig


@dash_app.callback(
    Output('filter-container', 'children'),
    [Input('factor-order-store', 'data')],
    [State('url', 'pathname')]
)
def update_filter_components(factor_order, pathname):
    # Extract experiment_id from the URL
    match = re.match(r'/dash/(\d+)', pathname)
    if not match:
        raise PreventUpdate

    experiment_id = int(match.group(1))
    experiment = Experiment.query.get_or_404(experiment_id)
    csv_file_path = f'datasets/{experiment.url.split("/")[-1]}/{experiment.url.split("/")[-1]}_samples.csv'
    df = pd.read_csv(csv_file_path)

    # Identify if Duration exists as a factor value
    duration_factor_col = next((col for col in df.columns if 'duration' in col.lower()), None)

    unique_values_dict = {
        factor: df[factor].dropna().unique()
        for factor in factor_order
    }

    filter_components = []

    # Add filter components for all factors
    for idx, factor in enumerate(factor_order):
        left_disabled = idx == 0
        right_disabled = idx == len(factor_order) - 1

        filter_component = html.Div([
            html.Div([
                html.Label(factor),
                dcc.Dropdown(
                    id={'type': 'filter-dropdown', 'index': idx},
                    options=[{'label': str(value), 'value': str(value)} for value in unique_values_dict[factor]],
                    multi=True,
                    placeholder=f"Select {factor}"
                )
            ]),
            html.Div([
                html.Button('←', id={'type': 'move-left', 'index': idx}, n_clicks=0, disabled=left_disabled, className="btn btn-primary btn-sm"),
                html.Button('→', id={'type': 'move-right', 'index': idx}, n_clicks=0, disabled=right_disabled, className="btn btn-primary btn-sm"),
            ], style={'display': 'flex', 'justify-content': 'center', 'margin-top': '5px', 'margin-bottom': '5px'})
        ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '20px'})
        filter_components.append(filter_component)

    # Always include duration-dropdown
    if duration_factor_col:
        duration_values = df[duration_factor_col].dropna().unique()
        duration_disabled = False
        duration_options = [{'label': str(value), 'value': str(value)} for value in duration_values]
    else:
        duration_disabled = True
        duration_options = []

    duration_component = html.Div([
        html.Label('Duration'),
        dcc.Dropdown(
            id='duration-dropdown',
            options=duration_options,
            multi=True,
            placeholder="Select Duration",
            disabled=duration_disabled
        )
    ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '20px', 'width': '160px'})
    filter_components.append(duration_component)

    return filter_components
@app.route('/api/experiments/<int:experiment_id>/organism_characteristics', methods=['GET'])
def get_organism_characteristics(experiment_id):
    experiment = Experiment.query.get_or_404(experiment_id)
    csv_file_path = f'datasets/{experiment.url.split("/")[-1]}/{experiment.url.split("/")[-1]}_samples.csv'
    
    try:
        df = pd.read_csv(csv_file_path)
        
        # Find all columns that contain the word "characteristics" (case-insensitive) in the header
        characteristic_cols = [col for col in df.columns if 'characteristics' in col.lower()]
        
        if not characteristic_cols:
            return jsonify({'message': 'No characteristics found in the data.'}), 404
        
        # Drop duplicate rows based on these characteristics columns
        df_unique = df[characteristic_cols].drop_duplicates()

        # Convert the unique rows to dictionary format for the API response
        characteristics_data = df_unique.to_dict(orient='records')
        
        return jsonify({
            'columns': characteristic_cols,
            'data': characteristics_data
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dash_app.callback(
    Output('factor-order-store', 'data'),
    [Input({'type': 'move-left', 'index': ALL}, 'n_clicks'),
     Input({'type': 'move-right', 'index': ALL}, 'n_clicks'),
     Input('reset-order-button', 'n_clicks')],
    [State('factor-order-store', 'data'),
     State('initial-order-store', 'data')]
)
def reorder_factors(left_clicks, right_clicks, reset_clicks, current_order, initial_order):
    ctx = dash.callback_context

    # Check if any callback was triggered, and if not, simulate the "Reset Filter Order"
    if not ctx.triggered:
        # Simulate reset button click (return initial order)
        return initial_order

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'reset-order-button':
        # Reset to initial order stored in initial_order_store
     
        return initial_order

    else:
        # Parse the triggered_id
        triggered_id = json.loads(triggered_id)
        index = triggered_id['index']
        action = triggered_id['type']

        new_order = current_order.copy()
        if action == 'move-left' and index > 0:
            new_order[index - 1], new_order[index] = new_order[index], new_order[index - 1]
        elif action == 'move-right' and index < len(new_order) - 1:
            new_order[index], new_order[index + 1] = new_order[index + 1], new_order[index]

        return new_order

@dash_app.callback(
    Output('cytoscape-tree', 'elements'),
    [Input({'type': 'filter-dropdown', 'index': ALL}, 'value'),
     Input('duration-dropdown', 'value'),
     Input('factor-order-store', 'data')],
    [State('url', 'pathname')]
)
def update_cytoscape(filter_values_list, selected_duration, factor_order, pathname):
    # Extract experiment_id from the URL
    match = re.match(r'/dash/(\d+)', pathname)
    if not match:
        raise PreventUpdate

    experiment_id = int(match.group(1))
    experiment = Experiment.query.get_or_404(experiment_id)
    csv_file_path = f'datasets/{experiment.url.split("/")[-1]}/{experiment.url.split("/")[-1]}_samples.csv'
    df = pd.read_csv(csv_file_path)

    if not filter_values_list:
        # Initialize with None if filter_values_list is empty
        filter_values_list = [None] * len(factor_order)

    # Build the filters dictionary
    filters = {}
    for i, factor in enumerate(factor_order):
        selected_values = filter_values_list[i]
        filters[factor] = selected_values or None  # Ensure that None is set if no values are selected
    
    duration_col = next((col for col in df.columns if 'duration' in col.lower()), None)
    if duration_col and selected_duration:
        df = df[df[duration_col].isin(selected_duration)]

    elements = create_cytoscape_elements(df, filters, factor_order, experiment_id)

    # If there are no nodes, ensure we return an empty graph with a message
    if not elements:
        return [{'data': {'id': 'no_data', 'label': 'No data available'}, 'position': {'x': 0, 'y': 0}}]
    
    return elements

@dash_app.callback(
    Output("cytoscape-tree", "generateImage"),
    [Input("btn-get-png", "n_clicks")]
)
def get_image(get_png_clicks):
    if get_png_clicks:
        return {
            'type': 'png',      # File type: png
            'action': 'download' # Action: download the PNG
        }
    return {}

@app.route('/experiment/<int:experiment_id>/results')
def experiment_results(experiment_id):
    # Redirect to the Dash app URL
    return redirect(f'/dash/results/{experiment_id}')


@app.route('/api/experiments/<int:experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    experiment = Experiment.query.get_or_404(experiment_id)
    return jsonify({
        'id': experiment.id,
        'url': experiment.url,
        'text_content': experiment.text_content,
        'created_at': experiment.created_at.isoformat()
        
    })

@app.route('/api/experiments/<int:experiment_id>/factor_values', methods=['GET'])
def get_factor_values(experiment_id):
    experiment = Experiment.query.get_or_404(experiment_id)
    csv_file_path = f'datasets/{experiment.url.split("/")[-1]}/{experiment.url.split("/")[-1]}_samples.csv'
    print(csv_file_path)
    try:
        df = pd.read_csv(csv_file_path)
        # Look for columns that likely represent organism characteristics (modify this logic as needed)
        factor_cols = [col for col in df.columns if 'factor value' in col.lower()]
        if factor_cols:
            factor_data = {col: df[col].dropna().unique().tolist() for col in factor_cols}
            return jsonify({
                'columns': factor_cols,
                'data': factor_data
            }), 200
        else:
            return jsonify({
                'message': 'No factors found.'
            }), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/submit_url', methods=['POST'])
def submit_url():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    ###### CSV AREA ######
    experiment_title = url.split('/')[-1]
    dataset_path = f"datasets/{experiment_title}"
    desired_csv_path = f'datasets/{experiment_title}/{experiment_title}_samples.csv'
    assays_csv_path = f'datasets/{experiment_title}/{experiment_title}_assays.csv'

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Directory {dataset_path} created")
    else:
        print(f"Directory {dataset_path} already exists")

    if os.path.exists(desired_csv_path) and os.path.exists(assays_csv_path):
        pass
    else:
        download_and_append_csv(url, desired_csv_path, assays_csv_path)

    try:
        experiment = Experiment.query.filter_by(url=url).first()
        if experiment:
            csv_file_path = desired_csv_path
            df = pd.read_csv(csv_file_path)
            generate_all_images(df)
            initialize_chatbot(experiment.text_content, experiment_id=experiment.id)
            
            return jsonify({
                'message': 'Data already exists in the database',
                'experiment_id': experiment.id,
                'text_content': experiment.text_content,
                'dashboard_url': f'http://localhost:5001/dash/{experiment.id}'
            }), 200
        else:
            scraped_data = scrape_data(url)
            print(scraped_data)

            experiment = Experiment(
                url=url,
                text_content=scraped_data
            )

            db.session.add(experiment)
            db.session.commit()
            
            # Generate images before sending response
            csv_file_path = desired_csv_path
            df = pd.read_csv(csv_file_path)
            generate_all_images(df)
            #initialize_chatbot(experiment.text_content, experiment_id=experiment.id)
            initialize_chatbot(experiment.text_content, experiment_id=experiment.id)
            return jsonify({
                'message': 'URL processed successfully',
                'experiment_id': experiment.id,
                'dashboard_url': f'http://localhost:5001/dash/{experiment.id}'
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def get_vectorstore_from_text(text):
    table_data = text['table_data']
    mission_dates = text['mission_dates']
    formatted_mission_dates = ", ".join([f"{mission['Identifier']}: {mission['Start Date']} - {mission['End Date']}" for mission in mission_dates])
    # Format table data into a readable string (if necessary)
    formatted_table_data = "\n".join([f"{item[0]}: {', '.join(item[1:])}" for item in table_data])

    document = Document(
        metadata={
            'title': text['project_info']['Project Title'],
            'organism': text['organism'],
            'table_data': formatted_table_data,
            'mission_dates': formatted_mission_dates,
            'language': 'English'  # Assuming English, modify if necessary
        },
        page_content=text['header']
    )
    
    documents_diff = [document]
    
    text_splitter = RecursiveCharacterTextSplitter()
    
    document_chunks = text_splitter.split_documents(documents_diff)

    
    persist_directory = "instance/chatbot_db"

    #clear_specific_data(f"{persist_directory}/chroma.sqlite3",persist_directory)
    
    # Create the vector store
    vector_store = Chroma.from_documents(document_chunks, HuggingFaceEmbeddings(), persist_directory=persist_directory)
    vector_store.persist()  # Save the embeddings to disk
    return vector_store

def get_context_retriever_chain(vector_store):
    
     # Create an LLM instance first
    bot = HuggingFaceEndpoint(
        repo_id="huggingfaceh4/zephyr-7b-alpha", 
        
        huggingfacehub_api_token="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",  # Use your token here
        task="text-generation"  # Specify the task if needed
    )
    
    # Initialize the ChatHuggingFace with the llm
    llm = ChatHuggingFace(llm=bot, tokenizer=tokenizer )  # Pass the LLM instance here
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation") 
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
     # Create an LLM instance first
    bot = HuggingFaceEndpoint(
        repo_id="huggingfaceh4/zephyr-7b-alpha",
        huggingfacehub_api_token="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",  # Use your token here
        task="text-generation"  # Specify the task if needed
    )
    
    # Initialize the ChatHuggingFace with the llm
    llm = ChatHuggingFace(llm=bot, tokenizer=tokenizer )  # Pass the LLM instance here
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def initialize_chatbot(text_content, experiment_id):
    if text_content:
        vector_store = get_vectorstore_from_text(text_content)
        vector_stores[experiment_id] = vector_store
        chat_histories[experiment_id] = [AIMessage(content="Hello, I am a bot. How can I help you?")]
        return experiment_id  # Return the session ID


@app.route('/chat/<int:experiment_id>', methods=['POST'])
def chat(experiment_id):
    user_input = request.json.get('input')

    print(f"Received chat request for experiment_id: {experiment_id}")
    print(f"vector_stores keys: {list(vector_stores.keys())}")
    print(f"user_input: {user_input}")

    if experiment_id in vector_stores and user_input:
        retriever_chain = get_context_retriever_chain(vector_stores[experiment_id])
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        
        response = conversation_rag_chain.invoke({
            "chat_history": chat_histories[experiment_id],
            "input": user_input
        })
        
        # Append user and AI responses to chat history
        chat_histories[experiment_id].append(HumanMessage(content=user_input))
        chat_histories[experiment_id].append(AIMessage(content=response['answer']))
        
        return jsonify({"response": response['answer']}), 200
    else:
        return jsonify({"error": "Invalid session or input"}), 400
    
dash_app.clientside_callback(
    """
    function(nodeData) {
        if (nodeData && nodeData.url) {
            window.open(nodeData.url, '_blank');
        }
        return '';
    }
    """,
    Output('node-link', 'children'),
    Input('cytoscape-tree', 'tapNodeData')
)

def clear_specific_data(db_path, persist_directory):
    # Connect to the SQLite database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Specify the tables you want to clear
    tables_to_clear = [
        'embeddings', 
        'collections', 
        'embedding_metadata', 
        'segments', 
        'segment_metadata'
    ]

    # Clear data from specified tables
    for table in tables_to_clear:
        try:
            cursor.execute(f"DELETE FROM {table}")
            print(f"Cleared data from table: {table}")
        except Exception as e:
            print(f"Error clearing table {table}: {e}")

    # Commit changes
    connection.commit()
    connection.close()

    # Clear additional directories
    for item in os.listdir(persist_directory):
        item_path = os.path.join(persist_directory, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the directory if it exists

    print("Cleared specific data and additional directories.")


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5001)
