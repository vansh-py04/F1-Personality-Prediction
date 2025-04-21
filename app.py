import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os
import gdown
import zipfile

# Load model and tokenizer
model_dir = "model"
model_zip = "modelzip.zip"
gdrive_file_id = "1Axvk4tun6rX9yQJRlVA2GCnMQPTjF8Ay" 
if not os.path.exists(model_dir):
    print("Downloading model.zip from Google Drive...")
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, model_zip, quiet=False)

    print("Extracting model.zip...")
    with zipfile.ZipFile(model_zip, 'r') as zip_ref:
        zip_ref.extractall(".")

# model_path = "model" # for local execution. 
print("Files in model dir:", os.listdir("model"))

tokenizer_path = "tokenizer"
model = AutoModelForSequenceClassification.from_pretrained(model_dir,
                                                           local_files_only=True, 
                                                           from_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model.eval()

# Load target encoder
with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

# Define driver map
driver_map = {
    "INTJ": {"name" : "Sebastian Vettel", "image":"/assets/INTJ.jpeg"},
    "INTP": {"name":"Charles Leclerc","image":"/assets/INTP.jpeg"},
    "ENTJ": {"name":"Lewis Hamilton","image":"/assets/ENTJ.jpg"},
    "ENTP": {"name":"Lando Norris","image":"/assets/ENTP.jpeg"},
    "INFJ": {"name":"George Russell","image":"/assets/INFJ.jpeg"},
    "INFP": {"name":"Daniel Ricciardo","image":"/assets/INFP.jpeg"},
    "ENFJ": {"name":"Fernando Alonso","image":"/assets/ENFJ.jpeg"},
    "ENFP": {"name":"Pierre Gasly","image":"/assets/ENFP.jpg"},
    "ISTJ": {"name":"Kimi Räikkönen","image":"/assets/ISTJ.jpg"},
    "ISFJ": {"name":"Valtteri Bottas","image":"/assets/ISFJ.jpeg"},
    "ESTJ": {"name":"Michael Schumacher","image":"/assets/ESTJ.jpeg"},
    "ESFJ": {"name":"Sergio Pérez","image":"/assets/ESFJ.jpeg"},
    "ISTP": {"name":"Max Verstappen","image":"/assets/ISTP.jpg"},
    "ISFP": {"name":"Carlos Sainz","image":"/assets/ISFP.jpg"},
    "ESTP": {"name":"Nico Rosberg","image":"/assets/ESTP.jpg"},
    "ESFP": {"name":"Yuki Tsunoda","image":"/assets/ESFP.jpeg"}
}

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "F1 Personality Predictor"

app.layout = html.Div(
    style={
        'backgroundColor': '#121212',
        'color': 'white',
        'fontFamily': 'Arial, sans-serif',
        'padding': '40px',
        'textAlign': 'center'
    },
    children=[
        html.H1("F1 Personality Predictor", style={'color': '#e10600', 'fontSize': '3em'}),

        html.Div([
            dcc.Textarea(
                id='input-text',
                placeholder='Describe yourself in a few sentences...',
                style={
                    'width': '50%',
                    'height': 150,
                    'padding': '10px',
                    'fontSize': '16px',
                    'border': '2px solid #e10600',
                    'borderRadius': '10px',
                    'backgroundColor': '#1e1e1e',
                    'color': 'white',
                    'margin': 'auto'
                }
            ),
            html.Br(),
            html.Button('Predict', id='submit-button', n_clicks=0, style={
                'marginTop': '20px',
                'padding': '12px 30px',
                'fontSize': '18px',
                'color': 'white',
                'backgroundColor': '#e10600',
                'border': 'none',
                'borderRadius': '10px',
                'cursor': 'pointer'
            })
        ], style={'textAlign': 'center'}),

        html.Div(id='output-container', style={'marginTop': '40px'})
    ]
)

@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0 and value:
        inputs = tokenizer(value, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            personality = target_encoder.inverse_transform([pred])[0]

        driver_info = driver_map[personality]

        return html.Div([
            html.H2(f"Predicted Personality Type: {personality}", style={'color': '#e10600'}),
            html.H3(f"Associated F1 Driver: {driver_info['name']}", style={'color': 'white'}),
            html.Img(src=driver_info['image'], style={
                'width': '300px',
                'marginTop': '20px',
                'borderRadius': '15px',
                'boxShadow': '0 4px 8px rgba(255, 255, 255, 0.2)'
            })
        ], style={'textAlign': 'center'})

    return None

if __name__ == '__main__':
    app.run(debug=True)
