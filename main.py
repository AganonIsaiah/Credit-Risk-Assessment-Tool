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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('start'))

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('start'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    df, accuracy_data, predictions_data = analyze_credit_data(file_path)

    print("Accuracy Table:")
    print(f"Accuracy: {accuracy_data['accuracy']}")
    print(f"Classification Report:\n{accuracy_data['classification_report']}")

    return render_template('index.html', df=df, accuracy=accuracy_data, predictions=predictions_data)


@app.route('/index')
def index():
    data = request.args.get('data', '')
    accuracy_data = {
        'accuracy': request.args.get('accuracy', ''),
        'classification_report': request.args.get('classification_report', '')
    }

    if data:
        df = pd.read_html(StringIO(data), index_col=0)[0]
        return render_template('index.html', df=df, accuracy=accuracy_data, predictions=request.args.get('predictions', ''))
    else:
        return redirect(url_for('start'))


if __name__ == '__main__':
    app.run(debug=True)
