import React from 'react'
import styled from 'styled-components'

const Background = styled.div`
    display: fixed;
    background-color: #e4e5f1;
    height: 100vh;
    width: 100%;
    display: flex;
    justify-content: center;
`;

const ChildrenContainer = styled.div`
    display: flex;
    height: 100vh;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    overflow-y: auto;
    overflow-x: hidden;
`;



const Background1 = ({ children }) => {
    return (
        <Background>
            <ChildrenContainer>
                {children}
            </ChildrenContainer>
        </Background>
    )
}

export default Background1
