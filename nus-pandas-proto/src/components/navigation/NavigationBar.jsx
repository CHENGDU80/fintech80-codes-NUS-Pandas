import React, { useEffect } from 'react'
import styled from 'styled-components'
import { Outlet } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';

import { signOut } from 'aws-amplify/auth';

const Container = styled.div`
    position: fixed;
    top: 20px;
    left: 5vw;
    width: auto;
    height: auto;
    border-radius: 30px;
    background-color: #484b6a;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 20px;
    gap: 50px;
    z-index: 100;
`;

const AccountContainer = styled.div`
    position: fixed;
    top: 20px;
    right: 5vw;
    width: auto;
    height: auto;
    border-radius: 30px;
    background-color: #484b6a;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 20px;
    gap: 50px;
    z-index: 100;
`;

const LogoContainer = styled.div`
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    margin: 0;
    font-weight: 600;
`;

const LinksContainer = styled.div`
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0;
    font-weight: 400;
    font-size: 18px;
    gap: 20px;

    a {
        color: white;
        text-decoration: none;
        margin: 0 10px;
        cursor: pointer;
        font-weight: 400;
        font-size: 18px;
    }

    .role {
        text-transform: capitalize;
    }

    p {
        color: white;
        margin: 10px;
        cursor: pointer;
        font-weight: 400;
        font-size: 18px;
    }

    a:hover {
        color: #f1f1f1;
    }
`;


const NavigationBar = () => {
    const { user, openSignupModal, checkUserAuth } = useAuth();
    const navigate = useNavigate();

    const handleSignOut = async () => {
        try {
            await signOut();
            await checkUserAuth();
        } catch (error) {
            console.error('Error signing out: ', error);
        }
    }

    useEffect(() => {
        checkUserAuth();
    }, [checkUserAuth]);

    const handleRoleClick = (role) => {
        if (role === 'insurer') {
            navigate('/insurer');
        }
    }

    return (
        <div>
            {user && user['custom:user_role'] === 'insurer' &&
                <Container>
                    <LogoContainer>
                        NUS Pandas
                    </LogoContainer>
                    <LinksContainer>
                        <a href="/claims">Claims</a>
                        <a href="/policies">Policies</a>
                    </LinksContainer>
                </Container>
            }
            {!user &&
                <Container>
                    <LogoContainer>
                        NUS Pandas
                    </LogoContainer>
                </Container>
            }
            <AccountContainer>
                {!user && <LinksContainer onClick={openSignupModal}>
                    <p>
                        Sign Up
                    </p>
                </LinksContainer>
                }
                {user &&
                    <LinksContainer>
                        <p className='role' onClick={() => handleRoleClick(user['custom:user_role'])}>
                            {user['custom:user_role']}
                        </p>
                        <p onClick={handleSignOut}>
                            Sign Out
                        </p>
                    </LinksContainer>}
            </AccountContainer>
            <Outlet />
        </div>
    )
}

export default NavigationBar
