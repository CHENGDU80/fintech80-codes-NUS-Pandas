import React, { useCallback, useEffect, useState } from 'react'
import Background1 from '../components/backgrounds/Background1'
import DashboardSection from '../components/sections/DashboardSection'

import ClaimCard from '../components/cards/ClaimCard'

import { get } from 'aws-amplify/api'

const Claim = () => {
    const [claims, setClaims] = useState([])

    const getClaims = useCallback(async () => {
        try {
            const { body } = await get({
                apiName: 'pandaClaimsDataApi',
                path: '/claims'
            }).response;

            const data = await body.json();

            setClaims(data)

        } catch (error) {
            console.error(error);
        }
    }, [])

    useEffect(() => {
        getClaims()
    }, [])

    return (
        <Background1>
            <DashboardSection>
                <h1>Claim</h1>
                <div className="claim-container">
                    {claims.map(claim => <ClaimCard key={claim.id} claim={claim} />)}
                </div>
            </DashboardSection>
        </Background1>
    )
}

export default Claim
