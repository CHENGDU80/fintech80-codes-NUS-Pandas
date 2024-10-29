import React, { useState, useRef, useEffect, useLayoutEffect } from 'react'
import { faPaperPlane } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { fetchAuthSession } from 'aws-amplify/auth';
import axios from 'axios'
import styled from 'styled-components'
import ReactMarkdown from 'react-markdown';

import { faPaperclip } from '@fortawesome/free-solid-svg-icons';

import { useAuth } from '../../context/AuthContext';

const ChatContainer = styled.div.attrs({ className: 'chat-container' })`
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    border-radius: 10px;
    padding: 8px;
    height: calc(100% - 16px);
    width: calc(100% - 16px);
    border: 2px solid black;
    backdrop-filter: blur(3px);
    transition: all 0.3s;

`

const MessageContainer = styled.div.attrs({ className: 'message-container' })`
    flex-grow: 1;
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow-y: auto; /* Make this section scrollable */
    overflow-x: hidden;
    gap: 12px;
    width: auto;
    padding: 4px 4px 4px 4px;

    ::-webkit-scrollbar {
        width: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #555;
        border-radius: 5px;
        width: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background-color: #888;
    }

`

const Message = styled.div.attrs({ className: 'message' })`
    padding: 10px;
    margin: 0;
    background-color: black;
    color: white;
    border-radius: 10px;
    max-width: 90%;
    height: auto;
    text-align: left;
    align-self: flex-start;
    display: flex;
    flex-direction: column;

    &.user-message {
        align-self: flex-end;
        text-align: left;
        // font-weight: 600;
    }

    &.bot-message {
        border-bottom-left-radius: 0px;
        border-bottom-right-radius: 10px;
    }

    &.user-message {
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 0px;
    }
`

const ChatInputContainer = styled.div.attrs({
    className: 'chatinput-container',
})`
    display: flex;
    padding: 10px;
    width: auto;
    align-items: flex-end;
    border: 0.3px solid #555;
    border-radius: 10px;
`

const Textarea = styled.textarea.attrs({ className: 'chatinput-textarea' })`
    flex-grow: 1;
    border: none;
    background: none;
    color: #000;
    border-radius: 10px;
    outline: none;
    resize: none;
    min-height: 45px;
    max-height: 150px;
    overflow-y: auto;
    font-size: 20px;
    font-family: 'Montserrat', sans-serif;
    scrollbar-width: none;
`

const SendButton = styled.button.attrs({ className: 'send-button' })`
    display: flex;
    height: 45px;
    width: 45px;
    background-color: black;
    color: #fff;
    border: none;
    border-radius: 5px;
    margin-left: 10px;
    cursor: pointer;
    justify-content: center;
    align-items: center;
`

const UploadButton = styled.button.attrs({ className: 'upload-button' })`
    display: flex;
    height: 45px;
    width: 45px;
    background-color: black;
    color: #fff;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    justify-content: center;
    align-items: center;
`;


const DotLoader = styled.div.attrs({ className: 'dot-loader' })`
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px;

    > div {
        width: 8px;
        height: 8px;
        background-color: black;
        border-radius: 50%;
        margin: 0 2.5px;
        animation: dot-bounce 1.2s infinite ease-in-out both;

        &:nth-child(1) {
            animation-delay: -0.32s;
        }

        &:nth-child(2) {
            animation-delay: -0.16s;
        }
    }

    @keyframes dot-bounce {
        0%,
        80%,
        100% {
            transform: scale(0);
        }
        40% {
            transform: scale(1);
        }
    }
`

const PromptsContainer = styled.div.attrs({ className: 'prompts-container' })`
    display: grid;
    width: 100%;
    grid-template-columns: 1fr 1fr; /* 2x2 layout */
    gap: 12px;
    border-radius: 5px;
    margin-bottom: 10px;
`

const Prompt = styled.div.attrs({ className: 'prompt' })`
    padding: 5px 10px;
    border-radius: 7px;
    border: 0.3px solid #555;
    color: #000;
    cursor: pointer;
    transition: all 0.3s;
    display: flex;
    flex-direction: column;
    text-align: left;
    gap: 5px;
    font-size: 18px;
    backdrop-filter: blur(5px);

    &:hover {
        box-shadow: 0 0 5px rgba(255, 255, 255, 0.7);
    }

    span {
        color: #000;
        font-weight: 600;
        font-size: 20px;
    }
`

const ChatBox = () => {
    const { user } = useAuth();
    const [messages, setMessages] = useState([])
    const [newMessage, setNewMessage] = useState('')
    const [loading, setLoading] = useState(false)
    const [prompts, setPrompts] = useState([
        ['Walk me through the procedure', 'After an accident', 'Help'],
        ['What are the steps to follow', 'When filing a claim'],
    ])

    const textareaRef = useRef(null)
    const messagesEndRef = useRef(null)
    const [selectedImage, setSelectedImage] = useState(null);

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();

            reader.onloadend = () => {
                let base64String = reader.result;
                // Remove the prefix
                base64String = base64String.replace(/^data:image\/[a-z]+;base64,/, '');
                setSelectedImage(base64String); // Store the cleaned base64 string in state
                console.log('Base64 encoded image without prefix:', base64String);
            };

            reader.readAsDataURL(file); // Convert file to base64
        }
    };

    useLayoutEffect(() => {
        // Adjusting the textarea height dynamically
        textareaRef.current.style.height = 'inherit'
        const currentScrollHeight = textareaRef.current.scrollHeight
        textareaRef.current.style.height = `${Math.min(currentScrollHeight, 150)}px`
    }, [newMessage])

    const handleSendMessage = () => {
        if (newMessage.trim() === '') return;

        addMessage(newMessage, 'You')
        replyMessage(newMessage)
        setNewMessage('')
    }

    const handlePromptMessage = (prompt) => {
        const promptMessage = `${prompt[0]} ${prompt[1]}`
        const option = prompt[2];
        addMessage(promptMessage, 'You')
        replyMessage(promptMessage, option)
        setPrompts([])
    }

    const addMessage = (text, user) => {
        setMessages(prev => [...prev, { text, user }])
        scrollToBottom()
    }

    const replyMessage = async (userMessage, option) => {
        if (!user) {
            return;
        }

        let option_msg = '';

        if (option === 'Help') {
            option_msg = 'Help'
        } else {
            option_msg = 'Claim_Assessment'
        }

        setLoading(true)
        try {
            const session = await fetchAuthSession();
            if (!session || !session.tokens || !session.tokens.idToken) {
                console.error('No valid session or token found.');
                return;
            }

            const vin = user['custom:vin'];
            const token = session.tokens.idToken;
            const payload = {
                headers: {
                    Authorization: `Bearer ${token}`
                },
                body: {
                    option: option_msg,
                    vin: vin,
                    collision_description: userMessage,
                    image: selectedImage,
                },
            }

            const response = await axios.post(
                'https://d-gpt.cognidex.ai/cgpt',
                payload.body,
                { headers: payload.headers }
            );

            const botReply = response.data;
            if (botReply.country_specific_rules) {
                addMessage(botReply.country_specific_rules, 'Bot')
                return;
            } else {
                addMessage(botReply.summary, 'Bot')
            }
        } catch (error) {
            console.error('Error:', error)
        } finally {
            setLoading(false)
        }
    }


    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    }

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    return (
        <div className='chat-container'>
            <ChatContainer>
                <MessageContainer>
                    {messages.map((message, index) => (
                        <Message
                            key={index}
                            className={message.user === 'You' ? 'user-message' : 'bot-message'}
                        >
                            <ReactMarkdown>{message.text}</ReactMarkdown>
                        </Message>
                    ))}
                    {loading && (
                        <DotLoader>
                            <div></div><div></div><div></div>
                        </DotLoader>
                    )}
                    <div ref={messagesEndRef} />
                </MessageContainer>
                {prompts.length > 0 && (
                    <PromptsContainer>
                        {prompts.map((prompt, index) => (
                            <Prompt key={index} onClick={() => handlePromptMessage(prompt)}>
                                <span>{prompt[0]}</span>{prompt[1]}
                            </Prompt>
                        ))}
                    </PromptsContainer>
                )}
                <ChatInputContainer>
                    <Textarea
                        ref={textareaRef}
                        placeholder={'How can I help you...'}
                        value={newMessage}
                        onChange={(e) => setNewMessage(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                    <UploadButton onClick={() => document.getElementById('fileInput').click()}>
                        <FontAwesomeIcon icon={faPaperclip} color="white" size="2x" />
                        <input
                            id="fileInput"
                            type="file"
                            accept="image/*"
                            style={{ display: 'none' }}
                            onChange={handleImageUpload}
                        />
                    </UploadButton>
                    <SendButton onClick={handleSendMessage}>
                        <FontAwesomeIcon icon={faPaperPlane} color='white' size='2x' />
                    </SendButton>
                </ChatInputContainer>
            </ChatContainer>
        </div>
    )
}


export default ChatBox