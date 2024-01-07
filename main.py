from io import StringIO
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
from analysis import analyze_credit_data

app = Flask(__name__)

# Configuration settings
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'sampledatasets')
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['SECRET_KEY'] = 'secretkey' 

# Checks for correct file (.csv)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

# Route for main page
@app.route('/')
def start():
    return render_template('start.html')

# Route for handling file uploads
@app.route('/upload', methods=['POST'])
def upload():
    # Check if file field is present in request
    if 'file' not in request.files:
        return redirect(url_for('start'))

    # Get uploaded file from request
    file = request.files['file']

    # Check if file is empty or has an allowed extension
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('start'))

    # Get filename and save to the designated folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Analyze the credit data
    df, accuracy_data, predictions_data = analyze_credit_data(file_path)

    # Render results to index.html
    return render_template('index.html', df=df, accuracy=accuracy_data, predictions=predictions_data)

# Route for index page
@app.route('/index')
def index():
    # Get data and accuracy information from the request parameters
    data = request.args.get('data', '')
    accuracy_data = {
        'accuracy': request.args.get('accuracy', ''),
        'classification_report': request.args.get('classification_report', '')
    }

    # If data is present, read it into a DataFrame and render the index.html
    if data:
        df = pd.read_html(StringIO(data), index_col=0)[0]
        return render_template('index.html', df=df, accuracy=accuracy_data, predictions=request.args.get('predictions', ''))
    else:
        # Redirect to the start.html if no data is present
        return redirect(url_for('start'))


if __name__ == '__main__':
    app.run(debug=True)
