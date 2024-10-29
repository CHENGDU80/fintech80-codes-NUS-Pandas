import React from 'react'
import CGPTBackground from '../components/backgrounds/CGPTBackground'
import CGPTSection from '../components/sections/CGPTSection'

import ChatBox from '../components/ChatBox/ChatBox'


const CGPT = () => {
    return (
        <CGPTBackground>
            <CGPTSection>
                <ChatBox />
            </CGPTSection>
        </CGPTBackground>
    )
}

export default CGPT