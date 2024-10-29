import React from 'react';
import styled from 'styled-components';

const InputContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: #484b6a;
    border-radius: 15px;
    width: calc(100% - 20px);
    padding: 10px;
    position: relative;
    overflow: hidden;
    transition: background-color 0.5s ease;
    z-index: 1;

    label {
        color: white;
        font-size: 18px;
        font-weight: 700;
        margin: 0 0 5px;
        transition: color 0.5s ease;
        width: 100%;
        text-align: left;
    }

    input {
        width: calc(100% - 12px);
        padding: 6px;
        border: none;
        border-radius: 8px;
        background-color: #fff;
        font-size: 16px;
        color: #333;
        transition: border 0.5s ease;
        font-family: 'Montserrat', sans-serif;
        border: 2px solid #fff;


        &:focus {
            outline: none;
            border: 2px solid #fff;
            transition: border 0.5s ease;
        }
    }

    &:hover {
        background-color: #3e4160;
    }
`;

const MainInput = ({ label, type, value, onChange }) => {
    return (
        <InputContainer>
            <label>{label}</label>
            <input
                type={type}
                value={value}
                onChange={onChange}
                placeholder="Enter text..."
            />
        </InputContainer>
    );
};

export default MainInput;
