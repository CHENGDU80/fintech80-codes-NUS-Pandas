import React from 'react'
import styled from 'styled-components'

const PolicyCardContainer = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 5px;
    background-color: #484b6a;
    padding: 20px;
    width: 100%;
    border-radius: 15px;
    margin: 10px 0;
    color: white;

    h1{
        color: white;
        font-size: 24px;
        font-weight: 600;
        margin: 0;
        text-align: left;
    }

    h2 {
        color: white;
        font-size: 20px;
        font-weight: 600;
        margin: 0;
        text-align: left;
    }

    p {
        margin: 0;
        font-size: 16px;
    }
    .section-container {
        display: flex;
        flex-direction: row;
        gap: 10px;
    }

    .section {
        margin-top: 10px;
        text-align: left;
    }
    
    .section-title {
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 5px;
        color: #adb5bd;
    }
`;

const PolicyCard = ({
    policy,
    riskScore,
}) => {
    return (
        <PolicyCardContainer>
            <h1>Policy ID: {policy.Policy_ID}</h1>
            <h2>Policy holder: {policy.Policyholder_Name}</h2>
            <h2>Dynamic Risk Score: {riskScore?.Dynamic_Risk_Score}</h2>
            <h2>Risk Category: {riskScore?.Risk_Category}</h2>

            <div className='section-container'>
                <div className="section">
                    <p className="section-title">Policy Details</p>
                    <p>Status: {policy.Policy_Status}</p>
                    <p>Type: {policy.Coverage_Type}</p>
                    <p>Annual Premium: ${policy.Annual_Premium}</p>
                    <p>Start Date: {policy.Policy_Start_Date}</p>
                    <p>End Date: {policy.Policy_End_Date}</p>
                </div>

                <div className="section">
                    <p className="section-title">Vehicle Information</p>
                    <p>Make: {policy.Vehicle_Make}</p>
                    <p>Model: {policy.Vehicle_Model}</p>
                    <p>Year: {policy.Vehicle_Year}</p>
                    <p>VIN: {policy.VIN}</p>
                    <p>Autonomy Level: {policy.Autonomy_Level}</p>
                </div>

                <div className="section">
                    <p className="section-title">Claim Information</p>
                    <p>Last Claim Date: {policy.Last_Claim_Date}</p>
                    <p>Total Claims: {policy.Claim_History}</p>
                    <p>Claim Amount: ${policy.Claim_Amount}</p>
                    <p>Deductible: ${policy.Deductible}</p>
                </div>

                <div className="section">
                    <p className="section-title">Additional Details</p>
                    <p>Country: {policy.Country}</p>
                    <p>Gender: {policy.Gender}</p>
                    <p>Age: {policy.Age}</p>
                    <p>Safety Score: {policy.Safety_Score}</p>
                    <p>IoT Monitoring: {policy.IoT_Monitoring ? 'Enabled' : 'Disabled'}</p>
                </div>
            </div>
        </PolicyCardContainer>
    )
}

export default PolicyCard;
