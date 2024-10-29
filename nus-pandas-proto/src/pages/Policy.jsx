import React, { useState } from 'react'
import Background1 from '../components/backgrounds/Background1'
import DashboardSection from '../components/sections/DashboardSection'
import MainInput from '../components/inputs/MainInput'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSearch } from '@fortawesome/free-solid-svg-icons'

import PolicyCard from '../components/cards/PolicyCard'

import { fetchAuthSession } from 'aws-amplify/auth';
import axios from 'axios'
import { get } from 'aws-amplify/api'

const Policy = () => {
    const [policies, setPolicies] = useState([])
    const [riskScores, setRiskScores] = useState([])
    const [vin, setVin] = useState('')

    const getRiskScore = async (policy_id) => {
        const session = await fetchAuthSession();
        if (!session || !session.tokens || !session.tokens.idToken) {
            console.error('No valid session or token found.');
            return;
        }

        // policy id must be number
        const policy_id_int = parseInt(policy_id);

        const token = session.tokens.idToken;
        const payload = {
            headers: {
                Authorization: `Bearer ${token}`
            },
            body: {
                target_policy_id: policy_id_int,
            },
        }

        const response = await axios.post(
            'https://d-gpt.cognidex.ai/risk_score',
            payload.body,
            { headers: payload.headers }
        );

        const riskScore = response.data;
        setRiskScores(prev => [...prev, riskScore[0]])
    }

    const getPolicies = async (vin) => {
        try {
            const { body } = await get({
                apiName: 'pandaPolicyApi',
                path: `/policy/${vin}`
            }).response;

            const data = await body.json();

            console.log(data)

            setPolicies(data)  // Assuming you want to store the policies in state

            data.forEach(policy => {
                getRiskScore(policy.Policy_ID)
            })
        }
        catch (error) {
            console.error(error)
        }
    }

    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            getPolicies(vin)
        }
    }

    return (
        <Background1>
            <DashboardSection>
                <h1>Policies</h1>
                <div className='search-container'>
                    <MainInput
                        type='text'
                        label='Search VIN Policy'
                        placeholder='Search Policies'
                        value={vin}
                        onChange={(e) => setVin(e.target.value)}
                        onKeyDown={handleKeyPress} // Add this line
                    />
                    <div className='button-container'>
                        <div className='search-button' onClick={() => getPolicies(vin)}>
                            <FontAwesomeIcon icon={faSearch} size='2x' />
                        </div>
                    </div>
                </div>
                <div className='container'>
                    {policies.map((policy, index) => (
                        <PolicyCard key={index} policy={policy} riskScore={riskScores[index]} />
                    ))}
                </div>
            </DashboardSection>
        </Background1>
    )
}

export default Policy
