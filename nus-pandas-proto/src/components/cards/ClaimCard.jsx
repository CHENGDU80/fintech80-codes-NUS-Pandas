import React from 'react'
import styled from 'styled-components'

const ClaimCardContainer = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 10px;
    background-color: #3c3f58;
    padding: 20px;
    width: auto;
    height: auto;
    border-radius: 15px;
    color: white;

    h1, h2 {
        font-size: 22px;
        font-weight: 600;
        margin: 0;
        color: white;
    }

    p {
        margin: 5px 0;
        font-size: 16px;
        text-align: left;
    }

    .section {
        margin-top: 10px;
    }
    
    .section-title {
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 5px;
        color: #adb5bd;
    }

    .section-container {
        display: flex;
        flex-direction: column;
        flex-wrap: wrap;
        gap: 10px;
    }
`;

const ClaimCard = ({
    claim
}) => {
    const renderValue = (value) => value !== null ? value : "Not Available";

    return (
        <ClaimCardContainer>
            <h1>Claim for VIN: {renderValue(claim.VIN)}</h1>
            <h2>Date: {renderValue(claim["Date Stamp"])}</h2>

            <div className='section-container'>
                <div className="section">
                    <p className="section-title">Claim Details</p>
                    <p>Severity Level: {renderValue(claim["Severity Level"])}</p>
                    <p>Collision Description: {renderValue(claim["Collision Description"])}</p>
                    <p>Claim Eligibility: {renderValue(claim["Claim Eligibility"])}</p>
                    <p>Reimbursement Amount: ${claim["Reimbursement Amount"]}</p>
                </div>

                <div className="section">
                    <p className="section-title">Risk Assessment</p>
                    <p>Risk Class: {renderValue(claim["Risk Class"])}</p>
                    <p>Reason: {renderValue(claim["Reason"])}</p>
                    <p>Country Specific Rules: {renderValue(claim["Country Specific Rules"])}</p>
                </div>

                <div className="section">
                    <p className="section-title">Additional Information</p>
                    <p>Country: {renderValue(claim["Country"])}</p>
                    <p>SHAP Context: {renderValue(claim["SHAP Context"])}</p>
                </div>
            </div>

            <p>Image Context: {renderValue(claim["Image Context"])}</p>
        </ClaimCardContainer>
    )
}

export default ClaimCard;
