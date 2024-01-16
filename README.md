# Credit Risk Assessment Tool

The Credit Risk Predictor is a Python tool designed to assess credit risk based on various individual attributes. This tool utilizes a machine learning model trained on a dataset to predict whether an individual is likely to default on credit.

## Author

Isaiah Aganon

Email: IsaiahAganon@cmail.carleton.ca

## Overview

The tool takes into account the following features for credit risk assessment:
- Age
- Gender
- Marital Status
- Income
- Employment Status
- Credit Score
- Open Credit Lines
- Total Credit Limit
- Late Payments
- Total Debt
- Monthly Debt Payments
- Asset Value

## Dataset Requirements

To use the Credit Risk Predictor, ensure that your dataset meets the following criteria:
- The dataset should have a minimum of two entries.
- At least one entry should have a 'DefaultStatus' of '1' (indicating a credit default) and another with '0' (indicating no default).

### Prerequisites

- Python 3.x
- Flask
- Pandas
- scikit-learn
- Werkzeug

### Installation
1. Clone the repository: git clone https://github.com/AganonIsaiah/Credit-Risk-Assessment-Tool.git
2. Install dependencies: pip install -r requirements.txt
3. Run app: python main.py
4. Open: http://127.0.0.1:5000 

