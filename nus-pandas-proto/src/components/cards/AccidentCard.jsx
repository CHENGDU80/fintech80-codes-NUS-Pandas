import React, { useState } from 'react';
import styled from 'styled-components';

const CardContainer = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 5px;
    background-color: #484b6a;
    border-radius: 15px;
    padding: 10px;
    width: calc(100% - 20px);
    z-index: 1;

    h1, h2 {
        color: white;
        font-size: 24px;
        font-weight: 600;
        margin: 0;
        text-align: left;
    }

    span {
        color: white;
        font-size: 20px;
        font-weight: 400;
        margin: 0;
        text-align: left;
    }

    p {
        color: white;
        font-size: 18px;
        font-weight: 400;
        margin: 0;
        text-align: left;
    }

    .record {
        cursor: pointer;
        padding: 10px;
        background-color: #595d82;
        border-radius: 10px;
    }

    .details {
        border-top: 1px solid white;
        margin-top: 10px;
        padding-top: 10px;
    }
`;

const AccidentCard = ({ vin_number, iot_records, onClick }) => {
    const [expandedIndex, setExpandedIndex] = useState(0); // Default open for the first record

    const toggleExpand = (index) => {
        setExpandedIndex(index === expandedIndex ? null : index); // Collapse if clicking the same index, otherwise open the clicked index
    };

    return (
        <CardContainer onClick={onClick}>
            <h1>VIN: <span>{vin_number}</span></h1>
            <p><strong>Total Records:</strong> {iot_records.length}</p>
            <p><strong>Risk Level:</strong> {iot_records[0].risk_level}</p>
            <p><strong>Probability:</strong> {iot_records[0].probabilities}%</p>

            {iot_records.map((record, index) => (
                <div
                    key={index}
                    className="record"
                    onClick={() => toggleExpand(index)}
                >
                    <p><strong>Record {index + 1}</strong> - Risk Level: {record.risk_level}</p>
                    <p><strong>Probability:</strong> {record.probabilities}%</p>

                    {expandedIndex === index && (
                        <div className="details">
                            <p><strong>Accident Class:</strong> {record.accident_class}</p>
                            <p><strong>Timestamp:</strong> {new Date(parseInt(record.timeStamp)).toLocaleString()}</p>
                        </div>
                    )}
                </div>
            ))}
        </CardContainer>
    );
}

export default AccidentCard;
