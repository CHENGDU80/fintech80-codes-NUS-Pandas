import base64
import time
from openai import OpenAI
import cv2
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
import boto3
import logging
from io import StringIO
import csv

# Load variables from .env file into the environment
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# Initialize DynamoDB resource (already in your code)
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

# Initialize the table for vehicle risk data
vehicle_risk_table = dynamodb.Table('XAIData')

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
        response = vehicle_risk_table.get_item(Key={'VIN': vin})
        if 'Item' in response:
            item = response['Item']
            risk_class = item.get('RISK_CLASS')
            image_url = item.get('image_url')
            return risk_class, image_url
        else:
            logger.info(f"No data found for VIN: {vin}")
            return None, None
    except Exception as e:
        logger.error(f"Error querying DynamoDB for VIN {vin}: {e}")
        return None, None
    
def shap_image_context(image_url, client):
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
                        "image_url": {"url": image_url},
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

def main():
    llm_history = []
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    while True:
        if option == "Claim Assessment":
            print("Enter the path of the image file: ")
            image_path = input()
            image_context = generate_image_context(image_path, client)
            print(f"Image Context: {image_context}")

            print("Enter the collision description: ")
            collision_description = input()
            severity_level_response = severity_level(image_context, collision_description, client)
            print(f"Severity Level: {severity_level_response}")

            print("Enter the VIN number: ")
            vin = input()
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



if __name__ == "__main__":
    main()
