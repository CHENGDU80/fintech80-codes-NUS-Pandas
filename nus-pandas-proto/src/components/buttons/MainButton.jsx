import React from 'react';
import styled from 'styled-components';

const Container = styled.div`
    display: flex;
    flex: 1 1 0;
    flex-direction: column;
    align-items: center;
    background-color: #484b6a;
    border-radius: 15px;
    height: 50px;
    justify-content: center;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    padding: 10px 20px;
    min-height: 30px;
    max-height: 50px;

    p {
        color: white;
        font-size: 18px;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
        transition: color 0.5s ease;
    }

    &:hover .filler {
        width: 100%;
    }

    &:hover p {
        color: #484b6a;
    }
`;

const Filler = styled.div`
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background-color: white;
    z-index: 0;
    width: 0;
    transition: width 0.5s ease;
`;

const MainButton = ({ onClick, header }) => {
    return (
        <Container onClick={onClick}>
            <Filler className="filler" />
            <p>{header}</p>
        </Container>
    );
};

export default MainButton;
