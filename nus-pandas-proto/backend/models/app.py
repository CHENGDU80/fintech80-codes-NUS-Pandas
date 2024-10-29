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
from io import StringIO
import csv
import cv2
from dotenv import load_dotenv
import os

load_dotenv()


# Load the model
model = XGBClassifier()
model.load_model('xgboost_model.json')

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def preprocessing(df):
    print(df.head())
    
    # we have other variable related to time, so remove raw 
    df = df.drop(columns='Time')

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
        
        return f'{VIN}-{timestamp}/shap_local_plot.png'

def generate_image_context(image_path, client):

    base64_image = encode_image(image_path)
    
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
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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

def get_risk_class_and_image(vin):
    try:
        response = accidentRecordsTable.get_item(Key={'VIN': vin})
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

def shap_image_context(image_url, client):
    # generate presigned url
    presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': imageBucket, 'Key': image_url}, ExpiresIn=3600)
    
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
                        "image_url": {"url": presigned_url},
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
    bucket_name = "chengdu-final"
    policy_file_key = "Policy_details.csv"
    
    try:
        s3_object = s3.get_object(Bucket=bucket_name, Key=policy_file_key)
        csv_data = s3_object['Body'].read().decode('utf-8')
        policy_data = csv.reader(StringIO(csv_data))
        
        headers = next(policy_data)

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
                        else:
                            reason = "Basic policy covers only Low severity accidents."
                    elif coverage == "Collision":
                        # Collision policy covers Low and Medium severity accidents
                        if risk_class == "Low":
                            reimbursement_amount = max(0, (claim_amount - deductible) * 0.75)
                        elif risk_class == "Medium":
                            reimbursement_amount = max(0, (claim_amount - deductible) * 0.85)
                        else:
                            reason = "Collision policy does not cover High severity accidents."
                    elif coverage == "Comprehensive":
                        # Comprehensive policy covers all severity levels
                        if risk_class == "Low":
                            reimbursement_amount = max(0, (claim_amount - deductible) * 0.9)
                        elif risk_class == "Medium":
                            reimbursement_amount = max(0, (claim_amount - deductible) * 0.95)
                        elif risk_class == "High":
                            reimbursement_amount = max(0, claim_amount - deductible)
                    else:
                        reason = "Unknown coverage type."
                    # Break the loop after processing the matching VIN
                    break
                else:
                    reason = "Policy is inactive or expired."
                    break

    except Exception as e:
        logger.error(f"Error accessing policy data from S3: {e}")
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
                        "text": reimbursement_amount,
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

# Create Flask app
app = Flask(__name__)

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
        
        print(f"Predictions: {predictions[0]==1}")
        
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
            return {}, 200
            # return jsonify({
            #         'VIN': data['VIN'],
            #         'timeStamp': timestamp,
            #         'accident_class': int(predictions[0]),
            #         'probabilities': 0,  # Keep it as float, round if needed
            #         'risk_level': 'No Risk',
            #         'shap_values': [],
            #         'image_location': ''
            #     }), 200

    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/cgpt', methods=['POST'])
def cgpt():
    try:
        llm_history = []
        client = OpenAI(api_key=openai_api_key)
        option = request.json['option']
        username = request.json['username']
        image = request.json['image']
        vin = cognito.get_user(UserPoolId='ap-southeast-1_Kb6e27O18', Username=username)['UserAttributes'][0]['custom:vin']
        
        # save the image into the directory
        with open('image.jpg', 'wb') as f:
            f.write(base64.b64decode(image))
        
        while True:
            if option == "Claim Assessment":
                print("Enter the path of the image file: ")
                
                # get the image path
                image_path = 'image.jpg'
                image_context = generate_image_context(image_path, client)
                print(f"Image Context: {image_context}")

                print("Enter the collision description: ")
                collision_description = input()
                severity_level_response = severity_level(image_context, collision_description, client)
                print(f"Severity Level: {severity_level_response}")

                print("Enter the VIN number: ")
                risk_class, image_url = get_risk_class_and_image(vin)
                if risk_class:
                    print(f"Risk Class: {risk_class}")
                    print(f"Image URL: {image_url}")
                    shap_context = shap_image_context(image_url, client)
                    print(f"SHAP Context: {shap_context}")

                    claim_eligibility_response = claim_eligibility(risk_class, severity_level_response, client)
                    print(f"Claim Eligibility: {claim_eligibility_response}")

                    if claim_eligibility_response == "Eligible for claim":
                        reimbursement_amount, reason = reimbursement_claim(vin, risk_class)
                        print(f"Reimbursement Amount: {reimbursement_amount}")
                        print(f"Reason: {reason}")
                    else:
                        reimbursement_amount = 0
                        reason = "Not eligible for claim"

                    summary_response = summary(shap_context, severity_level_response, claim_eligibility_response, reimbursement_amount, reason, client)
                    print(f"Summary: {summary_response}")

            elif option == "Help":
                print("Enter the country: ")
                country = input()
                country_specific_rules_response = country_specific_rules(country, client)
                print(f"Country Specific Rules: {country_specific_rules_response}")

            llm_history.append({
                "Image Context": image_context,
                "Collision Description": collision_description,
                "Severity Level": severity_level_response,
                "Risk Class": risk_class,
                "Image URL": image_url,
                "SHAP Context": shap_context,
                "Claim Eligibility": claim_eligibility_response,
                "Reimbursement Amount": reimbursement_amount,
                "Reason": reason,
                "Summary": summary_response,
                "Country": country,
                "Country Specific Rules": country_specific_rules_response,
            })

            print("Do you want to continue? (Y/N)")
            continue_option = input()
            if continue_option.lower() == "n":
                break
            else:
                
                continue
    except Exception as e:
        logging.error(f"Error in /cgpt: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
