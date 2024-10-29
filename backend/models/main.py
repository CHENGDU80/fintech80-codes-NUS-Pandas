from flask import Flask, request, jsonify
import pandas as pd
import numpy as np 
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import shap
import json
import boto3
from decimal import Decimal
import logging
import base64
from openai import OpenAI
from pathlib import Path
from io import StringIO, BytesIO
import csv
import cv2
from dotenv import load_dotenv
import os
import pickle
import tempfile
from datetime import datetime
from PIL import Image


from backend.models import app

load_dotenv()

# Load the model
model = XGBClassifier()
model.load_model('./backend/models/xgboost_model.json')
if model:
    print("XGBoost Model Loaded")


openai_api_key = os.environ.get("OPENAI_API_KEY")
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Set your region and AWS credentials
region = 'ap-southeast-1'
s3 = boto3.resource(
    's3',
    region_name=region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

s3_client = boto3.client(
    's3',
    region_name=region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)


# Initialize the DynamoDB resource (not client)
dynamodb = boto3.resource(
    'dynamodb',
    region_name=region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

cognito = boto3.client(
    'cognito-idp',
    region_name=region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)    

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

accidentRecordsTable = dynamodb.Table('pandaAccidentRecordsDynamo-dev')

imageBucket = 'accident-explanation-images1b896-dev'

bucket = s3.Bucket(imageBucket)

bucket_name = 'chengdu-final'

policy_file_key = "Policy_details.csv"

object_key = 'xgboost_model.json'

obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)

policy_file_path = "./data/input_data_RiskModel.xlsx"
model_file_path = "./model/stacking_regressor_model.pkl"

# Create a temporary file to load the model
with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
    # Write the content to the temporary file
    temp_file.write(obj['Body'].read())
    temp_file.flush()  # Ensure the content is written

    # Load the model from the temporary file
    model = XGBClassifier()
    model.load_model(temp_file.name)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def preprocessing(df):
    print(df.head())
    
    # we have other variable related to time, so remove raw 

    # categorical -> int
    le_vin = LabelEncoder()
    df['VIN'] = le_vin.fit_transform(df['VIN'])
    
    le_ctr = LabelEncoder()
    df['Country'] = le_ctr.fit_transform(df['Country'])

    le_vmk = LabelEncoder()
    df['Vehicle_Make'] = le_vmk.fit_transform(df['Vehicle_Make'])
    
    le_mdl = LabelEncoder()
    df['Vehicle_Model'] = le_mdl.fit_transform(df['Vehicle_Model'])

    le_mdl = LabelEncoder()
    df['Vehicle_Model'] = le_mdl.fit_transform(df['Vehicle_Model'])

    df['Autonomy_Level'] = df['Autonomy_Level'].apply(lambda x: int(x.replace('Level ','')))

    return df

def risk_level(proba):
    if (proba <= 66):
        return 'Low'
    elif (proba <= 82):
        return 'Medium'
    else:
        return 'High'

def local_explain(X, explainer, VIN, timestamp, display=False):
    # Get shap values and explanation for the input sample
    shap_explanation = explainer(X)
    shap_values = shap_explanation.values[0]         # For a single prediction
    base_value = shap_explanation.base_values[0]
    data = X.iloc[0] if hasattr(X, 'iloc') else X    # Handle both DataFrame and array input

    if display:
        shap.waterfall_plot(shap.Explanation(values=shap_values, 
                                             base_values=base_value, 
                                             data=data))
    else:
        plt.figure()
        shap.waterfall_plot(shap.Explanation(values=shap_values, 
                                             base_values=base_value, 
                                             data=data),
                                             show=False)
        # Save as image
        plt.savefig('./shap_local_plot.png', dpi=300, bbox_inches='tight')
        plt.close()    
        
        # Upload to S3
        bucket.upload_file('./shap_local_plot.png', f'{VIN}-{timestamp}/shap_local_plot.png')
        
        return f'{VIN}/shap_local_plot.png'

def generate_image_context(image_path, client):
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                            "You use the provided image to generate a detailed description of the vehicle's damage. "
                            "Focus on identifying the point of impact, the severity and extent of the damage, and any visible issues. "
                            "The description should be suitable for filing an insurance claim."
                            ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_path}"},
                    },
                ],
            },
            
        ],
        max_tokens=1024,
    )

    response_text = response.choices[0].message.content
    return response_text

def severity_level(image_context, collision_description, client):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                            You are an assistant that assesses vehicle damage severity based on image and collision descriptions. 

                            **Instructions:**

                            1. **Match Verification:**
                            - First, determine if the image description matches the collision description.
                            - If they do not match, respond with: "The image and description do not match."

                            2. **Severity Classification:**
                            - If they match, classify the severity of the accident into one of the following categories:
                                - **High**: Extensive damage (major dents, broken parts, vehicle inoperable).
                                - **Medium**: Moderate damage (noticeable dents, scratches, but vehicle operable).
                                - **Low**: Minor damage (small scratches, light dents, cosmetic issues).
                                - **Not Severe**: No visible damage or extremely minor issues.

                            **Response Format:**

                            - Provide **only one word** indicating the severity level: "High", "Medium", "Low", or "Not Severe".
                            - Do not include any additional information or explanations.
                            """,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": image_context,
                    },
                    {
                        "type": "text",
                        "text": collision_description,
                    },
                ],
            },
        ],
        max_tokens=10,
        temperature=0.0,
    )

    response_text = response.choices[0].message.content
    return response_text

def get_risk_class_and_image(vin, timestamp):
    try:
        response = accidentRecordsTable.get_item(Key={'VIN': vin, 'timeStamp': timestamp})
        if 'Item' in response:
            item = response['Item']
            risk_class = item.get('risk_level')
            image_url = item.get('image_location')
            return risk_class, image_url
        else:
            logger.info(f"No data found for VIN: {vin}")
            return None, None
    except Exception as e:
        logger.error(f"Error querying DynamoDB for VIN {vin}: {e}")
        return None, None
    
def download_image_from_s3(image_key):
    """Downloads an image from S3 given its key."""
    # Define the local file path to save the downloaded image
    local_image_path = f"/tmp/{image_key.split('/')[-1]}"
    s3_client.download_file(imageBucket, image_key, local_image_path)
    logger.info(f"Image downloaded successfully: {local_image_path}")
    return local_image_path

def shap_image_context(local_image_path, client):
    # generate presigned url
    base64_image = encode_image(local_image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                            "You are an AI assistant specializing in accident analysis using explainable AI techniques. "
                            "Your task is to interpret the data presented in the image and explain the key factors that contributed to the accident. "
                            "Focus on discussing how each factor influenced the outcome without mentioning SHAP values, SHAP graphs, or any specific tools. "
                            "Provide a clear and concise explanation suitable for someone seeking to understand the reasons behind the accident."
                            )            
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        max_tokens=512,
        temperature=0.0,
    )

    response_text = response.choices[0].message.content
    return response_text

def claim_eligibility(risk_class, severity_level, client):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                        You are an assistant that determines claim eligibility based on risk class and severity level.

                        **Instructions:**

                        - Compare the provided **risk class** and **severity level**.
                        - If the risk class **matches** the severity level, respond with: **"Eligible for claim"**.
                        - If they do **not match**, respond with: **"Not eligible for claim"**.

                        **Response Format:**

                        - Provide **only** one of the two responses:
                        - **Eligible for claim**
                        - **Not eligible for claim**
                        - Do **not** include any additional text or explanations.
                        """,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": risk_class,
                    },
                    {
                        "type": "text",
                        "text": severity_level,
                    },
                ],
            },
        ],
        max_tokens=10,
        temperature=0.0,
    )

    response_text = response.choices[0].message.content
    return response_text

def reimbursement_claim(VIN, risk_class): 
    try:
        # Path to the local CSV file
        policy_file_path = "./data/Policy_details.csv"  # Update with the correct file path

        # Initialize default values for reimbursement_amount and reason
        reimbursement_amount = 0  # Default reimbursement amount
        reason = "No matching policy found or other issue."  # Default reason if no matching condition
        
        # Read the policy data from the local CSV file
        with open(policy_file_path, mode='r') as file:
            policy_data = csv.reader(file)
            headers = next(policy_data)  # Read the headers
            
            for row in policy_data:
                policy = dict(zip(headers, row))
                if policy['VIN'] == VIN:
                    if policy['Policy_Status'] == 'Active':
                        coverage = policy['Coverage_Type']
                        deductible = float(policy['Deductible'])
                        claim_amount = float(policy.get('Claim_Amount', 0))
                        
                        # Calculate reimbursement based on policy type and risk class
                        if coverage == "Basic":
                            # Basic policy covers only Low severity accidents
                            if risk_class == "Low":
                                reimbursement_amount = max(0, (claim_amount - deductible) * 0.5)
                                reason = "Basic policy covers only Low severity accidents."
                            else:
                                reason = "Basic policy does not cover Medium or High severity accidents."
                        elif coverage == "Collision":
                            # Collision policy covers Low and Medium severity accidents
                            if risk_class == "Low":
                                reimbursement_amount = max(0, (claim_amount - deductible) * 0.75)
                                reason = "Collision policy covers Low severity accidents."
                            elif risk_class == "Medium":
                                reimbursement_amount = max(0, (claim_amount - deductible) * 0.85)
                                reason = "Collision policy covers Medium severity accidents."
                            else:
                                reason = "Collision policy does not cover High severity accidents."
                        elif coverage == "Comprehensive":
                            # Comprehensive policy covers all severity levels
                            if risk_class == "Low":
                                reimbursement_amount = max(0, (claim_amount - deductible) * 0.9)
                                reason = "Comprehensive policy covers Low severity accidents."
                            elif risk_class == "Medium":
                                reimbursement_amount = max(0, (claim_amount - deductible) * 0.95)
                                reason = "Comprehensive policy covers Medium severity accidents."
                            elif risk_class == "High":
                                reimbursement_amount = max(0, claim_amount - deductible)
                                reason = "Comprehensive policy covers High severity accidents."
                        else:
                            reason = "Unknown coverage type."
                        break  # Exit the loop after processing the matching VIN
                    else:
                        reason = "Policy is inactive or expired."
                        break

    except Exception as e:
        logger.error(f"Error accessing policy data from local file: {e}")
        reimbursement_amount = 0
        reason = "Error accessing policy data."
    
    return reimbursement_amount, reason

def summary(shape_context, severity_level, claim_eligibility, reimbursement_amount, reason, client):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                                "You are an assistant that provides users with a comprehensive summary of their claim assessment. "
                                "Your summary should include:\n\n"
                                "1. **Accident Analysis**: Briefly explain the factors that contributed to the accident based on the provided analysis.\n"
                                "2. **Severity Level**: State the determined severity level of the accident.\n"
                                "3. **Claim Eligibility**: Inform the user whether their claim is eligible for processing.\n"
                                "4. **Reimbursement Amount**: If eligible, specify the reimbursement amount they will receive.\n"
                                "5. **Reason**: Provide a clear explanation for the eligibility decision.\n\n"
                                "Present the information in a clear, polite, and professional manner suitable for customer communication."
                            ),            
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": shape_context,
                    },
                    {
                        "type": "text",
                        "text": severity_level,
                    },
                    {
                        "type": "text",
                        "text": claim_eligibility,
                    },
                    {
                        "type": "text",
                        "text": str(reimbursement_amount),
                    },
                    {
                        "type": "text",
                        "text": reason,
                    },
                ],
            },
        ],
        max_tokens=2156,
        temperature=0.0,
    )

    response_text = response.choices[0].message.content
    return response_text

def country_specific_rules(country, client):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                                "You are an expert assistant specializing in vehicle accident procedures and insurance claim processes around the world. "
                                "Your task is to provide detailed guidance to users involved in a vehicle accident in a specific country. "
                                "Include information about legal requirements, steps to take immediately after an accident, how to report the incident, and any country-specific considerations that may affect the insurance claim. "
                                "Present the information in a clear, empathetic, and professional manner."
                            )            
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": country,
                    },
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.0
    )

    response_text = response.choices[0].message.content
    return response_text

def load_model_from_local(model_path):
    # Load the model file from a local path
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model

def calculate_risk_score(file_path, target_policy_id, model_path):
    # Load the Excel file from the local path into a DataFrame
    data = pd.read_excel(file_path)
    
    # Define columns to select
    selected_columns = [
        "Age", "Gender", "Country", "Vehicle_Make", "Vehicle_Model",
        "Vehicle_Year", "Autonomy_Level", "Coverage_Type", "Annual_Premium",
        "Deductible", "Claim_History", "Claim_Amount", "Safety_Score",
        "Num_Accidents", "IoT_Monitoring", "Past_Fraud_Record"
    ]
    
    # Filter data for the specific Policy_ID and required columns
    filtered_data = data[data["Policy_ID"] == target_policy_id][selected_columns]
    
    # Load the saved model
    loaded_model = load_model_from_local(model_path)
    
    # Make predictions using the loaded model
    filtered_data["ML_Risk_Score"] = loaded_model.predict(filtered_data)
    
    # Apply the rule-based criteria to calculate the Dynamic Risk Score
    def calculate_dynamic_risk_score(row):
        risk_score = row["ML_Risk_Score"]
        if row["Num_Accidents"] >= 3:
            risk_score += 10
        if row["Past_Fraud_Record"] == 1:
            risk_score += 20
        return risk_score
    
    filtered_data["Dynamic_Risk_Score"] = filtered_data.apply(calculate_dynamic_risk_score, axis=1)
    
    # Categorize the Dynamic Risk Score
    def categorize_risk(drs):
        if drs < 40:
            return "LOW"
        elif 40 <= drs < 60:
            return "MEDIUM"
        elif 60 <= drs < 70:
            return "HIGH A"
        elif 70 <= drs < 80:
            return "HIGH B"
        else:
            return "VERY HIGH"
    
    filtered_data["Risk_Category"] = filtered_data["Dynamic_Risk_Score"].apply(categorize_risk)
    
    # Display the prediction results
    print(filtered_data[["ML_Risk_Score", "Dynamic_Risk_Score", "Risk_Category"]])
    return filtered_data[["ML_Risk_Score", "Dynamic_Risk_Score", "Risk_Category"]]


# # Create Flask app
# app = Flask(__name__)

if app:
    print("FLASK APP ACTIVE")

# Health check method
@app.route("/", methods=['GET'])
def health():
    return "CGPT Active"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        timestamp = request.json['timestamp']
        data = request.json['data']
        
        df = pd.DataFrame([data])
        df = preprocessing(df)
        
        # Convert all columns to float
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                pass
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        if predictions[0] == 1:
            risk = risk_level(probabilities[0][1] * 100) if predictions[0] == 1 else 'No Risk'
            
            # Get SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df)
            
            # Convert probabilities and shap_values to standard float
            probabilities = float(probabilities[0][1])  # Convert to float
            shap_values = [float(value) for value in shap_values[0]]  # Convert to list of floats
            
            # convert the float to Decimal
            probabilities = Decimal(probabilities)
            shap_values = [Decimal(value) for value in shap_values]
            
            # Generate image and get location
            image_location = local_explain(df, explainer, data['VIN'], timestamp)

        
            # timestamp in string format
            timestamp = str(timestamp)

            # Create item for DynamoDB
            accidentRecordsTable.put_item(
                Item={
                    'VIN': data['VIN'],
                    'timeStamp': timestamp,
                    'accident_class': int(predictions[0]),
                    'probabilities': round(probabilities * 100, 3),  # Keep it as float, round if needed
                    'risk_level': risk,
                    'shap_values': shap_values,
                    'image_location': image_location
                }
            )
            
            return jsonify({
                    'VIN': data['VIN'],
                    'timeStamp': timestamp,
                    'accident_class': int(predictions[0]),
                    'probabilities': round(probabilities * 100, 3),  # Keep it as float, round if needed
                    'risk_level': risk,
                    'shap_values': shap_values,
                    'image_location': image_location
                }), 200
        else:
            return jsonify({
                    'VIN': data['VIN'],
                    'timeStamp': timestamp,
                    'accident_class': int(predictions[0]),
                    'probabilities': 0,  # Keep it as float, round if needed
                    'risk_level': 'No Risk',
                    'shap_values': [],
                    'image_location': ''
                }), 200

    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

dynamodb = boto3.resource('dynamodb')
table_name = "pandaClaimsDataDynamo-dev"  # Update with your DynamoDB table name
table = dynamodb.Table(table_name)

@app.route('/cgpt', methods=['POST'])
def cgpt():
    try:
        # Initialize variables and logging setup
        llm_history = []
        image_context, severity_level_response, risk_class, image_url, shap_context = None, None, None, None, None
        claim_eligibility_response, summary_response = None, None
        country, country_specific_rules_response = None, None
        reimbursement_amount, reason = 0, None  # Default values for reimbursement amount and reason
        logging.basicConfig(level=logging.INFO)

        # Client setup
        client = OpenAI(api_key=openai_api_key)
        
        # Retrieve options and other form data
        data = request.get_json()  # This will load the JSON data from the raw body

        # Access the fields from the JSON data
        option = data.get('option')
        collision_description = data.get('collision_description')
        # vin = "1HGCM82633A815208"  # Static VIN for testing
        vin = data.get('vin')
        # timestamp = "1730198023600"
        image = data.get('image')

        # Generate timestamp (in dd/mm format for display) and unique timestamp for sort key
        date_stamp = datetime.now().strftime("%d/%m")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp for sort key


        # image_path = data.get('image_path')  # Get image location from JSON payload
        # if image_path:
        #     try:
        #         # Open and process the image from the provided path
        #         image = Image.open(image_path)
        #         logging.info(f"Image loaded successfully from {image_path}.")
        #         # Perform any further processing on the `image` object as needed
        #     except FileNotFoundError:
        #         logging.error(f"Image not found at {image_path}.")
        # else:
        #     logging.warning("No image path provided in the request.")

        # Retrieve the image file
        # image = request.files.get('image')
        # if image:
        #     filename = f"../crash_images/image_{timestamp}.jpg"
        #     image.save(filename)
        #     logging.info(f"Image saved successfully as {filename}.")
        # else:
        #     logging.warning("No image provided in the request.")

        if option == "Claim_Assessment":
            # Image context generation
            logging.info("Generating image context...")
            image_context = generate_image_context(image, client)
            logging.info(f"Image Context: {image_context}")
            
            # Severity level based on image context and collision description
            if image_context:
                severity_level_response = severity_level(image_context, collision_description, client)
                logging.info(f"Severity Level: {severity_level_response}")

            # Retrieve risk class (hardcoded here for testing)
            # risk_class, image_url = get_risk_class_and_image(vin, timestamp)
            # logging.info(f"Risk Class: {risk_class}, Image URL: {image_url}")

            # # Download the image from S3
            # local_image_path = download_image_from_s3(image_url)
            # logging.info(f"Local image path: {local_image_path}")

            # # Generate SHAP context based on the image
            # shap_context = shap_image_context(local_image_path, client)
            # logging.info(f"SHAP Context: {shap_context}")
                
            risk_class = "Medium"
            
            claim_eligibility_response = claim_eligibility(risk_class, severity_level_response, client)
            logging.info(f"Claim Eligibility: {claim_eligibility_response}")

            if claim_eligibility_response == "Eligible for claim":
                reimbursement_amount, reason = reimbursement_claim(vin, risk_class)
                logging.info(f"Reimbursement Amount: {reimbursement_amount}, Reason: {reason}")
                # Convert reimbursement_amount to Decimal for DynamoDB
                reimbursement_amount = Decimal(str(reimbursement_amount))
            else:
                reason = "Not eligible for claim"
                reimbursement_amount = Decimal(0)

            # Generate summary of the assessment
            summary_response = summary(shap_context, severity_level_response, claim_eligibility_response, str(reimbursement_amount), reason, client)
            return jsonify({"status": "success", "summary": summary_response})
            #logging.info(f"Summary: {summary_response}")

        elif option == "Help":
            # Get country-specific rules if "Help" is selected
            country = "Singapore"
            if country:
                country_specific_rules_response = country_specific_rules(country, client)
                return jsonify({"status": "success", "country_specific_rules": country_specific_rules_response})

                #logging.info(f"Country Specific Rules: {country_specific_rules_response}")

        # Append results to llm_history
        llm_history.append({
            "VIN": vin,  # Partition key
            "timeStamp": timestamp,  # Sort key for DynamoDB
            "Date Stamp": date_stamp,  # For display in logs if needed
            "Image Context": image_context,
            "Collision Description": collision_description,
            "Severity Level": severity_level_response,
            "Risk Class": risk_class,
            "Image URL": image_url,
            "SHAP Context": shap_context,
            "Claim Eligibility": claim_eligibility_response,
            "Reimbursement Amount": reimbursement_amount,  # Now a Decimal
            "Reason": reason,
            "Summary": summary_response,
            "Country": country,
            "Country Specific Rules": country_specific_rules_response
        })

        # Store each entry in llm_history in DynamoDB with VIN and timestamp as keys
        for record in llm_history:
            table.put_item(Item=record)
            logging.info("Record stored in DynamoDB successfully.")

        #return jsonify({"status": "success", "history": llm_history})
    
    except Exception as e:
        logging.error(f"Error in /cgpt: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/risk_score', methods=['POST'])
def risk_score():
    try:
        target_policy_id = request.json['target_policy_id']
        # Pass the local file paths instead of S3 parameters
        risk_score_df = calculate_risk_score(policy_file_path, target_policy_id, model_file_path)
        return jsonify(risk_score_df.to_dict(orient='records')), 200
    except Exception as e:
        logging.error(f"Error in /risk_score: {e}")
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)
