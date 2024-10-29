import React from 'react';
import styled from 'styled-components';
import MainButton from '../buttons/MainButton';

const Container = styled.div`
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    padding: 10px;
    width: calc(100% - 20px);
    border-radius: 15px;
    background-color: #484b6a;
    color: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    min-height: 240px;
    max-height: 240px;
`;

const ImageContainer = styled.div`
    margin-right: 20px;
    max-width: 240px;

    img {
        width: 240px;
        height: 240px;
        border-radius: 10px;
        object-fit: cover;
    }
`;

const InfoContainer = styled.div`
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;

    text-align: left;
`;

const Title = styled.h3`
    margin: 0;
    font-size: 20px;
`;

const Description = styled.p`
    margin: 0;
    font-size: 14px;
    color: #dcdcdc;
`;

const Dates = styled.div`
    font-size: 12px;
    color: #a0a0a0;
`;

const Status = styled.div`
    margin-top: 10px;
    font-weight: bold;
    color: ${({ status }) => (status === 'Active' ? '#4caf50' : '#f44336')};
`;

const Button = styled.button`
    align-self: flex-start;
    margin-top: 15px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
    color: #484b6a;
    background-color: #ffffff;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s;

    &:hover {
        background-color: #e0e0e0;
    }
`;

const ReferendaCard = ({
    RefNumber,
    RefTitle,
    RefDescription,
    RefStartDate,
    RefEndDate,
    RefStatus,
    RefImage,
}) => {
    return (
        <Container>
            <ImageContainer>
                <img src={RefImage} alt={RefTitle} />
            </ImageContainer>
            <InfoContainer>
                <Title>#{RefNumber}: {RefTitle}</Title>
                <Description>{RefDescription}</Description>
                <Dates>
                    <div>Start: {RefStartDate}</div>
                    <div>End: {RefEndDate}</div>
                </Dates>
                <Status status={RefStatus}>{RefStatus}</Status>
                <Button>Vote Now</Button>
            </InfoContainer>
        </Container>
    );
};

export default ReferendaCard;
