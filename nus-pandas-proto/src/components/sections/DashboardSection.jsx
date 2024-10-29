import React from 'react'
import styled from 'styled-components'

const Container = styled.div`
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;
    height: 90vh;
    width: 90vw;
    padding: 10vh 5vw 0 5vw;
    text-align: center;
    overflow: hidden; 

    h1 {
        font-weight: 700;
        z-index: 2;
        text-align: left;
    }

    p {
        font-size: 22px;
        font-weight: 400;
        z-index: 2;
        margin-top: 10px;
    }

    .container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 20px;
        width: 100%;
        height: 100%;
        z-index: 2;
    }

    .inner-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        width: 50%;
        height: 100%;
        z-index: 2;

        h4 {
            font-weight: 600;
            margin: 0;
            font-size: 24px;
        }
    }

    .search-container {
        display: flex;
        flex-direction: row;
        gap: 10px;
        width: 100%;
        height: auto;
        z-index: 2;

        .button-container {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .search-button {
            background-color: #484b6a;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 18px;
            font-weight: 600;
        }
    }

    .claim-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        width: 100%;
        height: auto;
        z-index: 2;
        overflow-y: auto;

        ::-webkit-scrollbar {
            display: none;  
        }
        -ms-overflow-style: none;

    }

    .dashboard-accident-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        width: auto;
        height: 70vh;

        ::-webkit-scrollbar {
            display: none;
        }
        -ms-overflow-style: none;
        scrollbar-width: none;

        overflow-y: auto;
    }
`;

const LineOne = styled.div`
    position: fixed;
    top: 0;
    left: 50%;
    height: 300vh;
    width: 4px;
    background-color: #484b6a;
    transform: rotate(45deg);

    z-index: 1;
`;

const LineTwo = styled.div`
    position: fixed;
    top: -50%;
    left: -30%;
    height: 300vh;
    width: 4px;
    background-color: #484b6a;
    transform: rotate(45deg);

    z-index: 1;
`;


const DashboardSection = ({ children }) => {
    return (
        <Container>
            <LineOne />
            <LineTwo />
            {children}
        </Container>
    )
}

export default DashboardSection
