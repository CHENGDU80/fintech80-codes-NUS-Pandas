import React, { useCallback, useEffect, useState } from 'react'
import Background1 from '../components/backgrounds/Background1'
import SectionHeader from '../components/sections/SectionHeader'
import DashboardSection from '../components/sections/DashboardSection'
import MainButton from '../components/buttons/MainButton'
import AccidentCard from '../components/cards/AccidentCard'
import { useNavigate } from 'react-router-dom'

import { useAuth } from '../context/AuthContext'
import { getUrl } from 'aws-amplify/storage';
import { get } from 'aws-amplify/api'

const Home = () => {
    const { user } = useAuth()
    const navigate = useNavigate()
    const [accidents, setAccidents] = useState([])
    const [accidentCount, setAccidentCount] = useState(0)

    const [accident, setAccident] = useState({})

    const [imageUrl, setImageUrl] = useState();

    const getImage = useCallback(async (image) => {
        try {
            console.log(image)
            const url = await getUrl({
                path: image,
                options: {
                    expiresIn: 2000
                }
            });
            console.log(url.url.href)
            setImageUrl(url.url.href);
        } catch (error) {
            console.error(error);
        }
    }, []);


    const getAccidents = useCallback(async () => {
        try {
            const { body } = await get({
                apiName: 'pandaAccidentRecordsApi',
                path: '/accidents'
            }).response;

            const data = await body.json();

            setAccidentCount(data.length)

            // group the accidents by VIN
            const groupedByVIN = data.reduce((acc, vehicle) => {
                const { VIN } = vehicle;
                if (!acc[VIN]) {
                    acc[VIN] = []; // Initialize an array if it doesn't exist for this VIN
                }
                acc[VIN].push(vehicle);
                return acc;
            }, {});

            // Set accident as the first VIN's group if available
            const firstVIN = Object.keys(groupedByVIN)[0];
            if (firstVIN) {
                setAccident(groupedByVIN[firstVIN]);
            }

            console.log(accident)

            setAccidents(groupedByVIN)

            if (data.length > 0) {
                getImage(data[0].image_location)
            }
        } catch (error) {
            console.log(error)
        }
    }, [])

    useEffect(() => {
        if (user && user['custom:user_role'] === 'insurer') {
            getAccidents()
        }
    }, [user, getAccidents])

    return (
        <Background1>
            {user && user['custom:user_role'] === 'insurer' &&
                <DashboardSection>
                    <h1>
                        Dashboard
                    </h1>
                    <p>
                        Accidents Today: {accidentCount}
                    </p>
                    <div className='container'>
                        <div className="inner-container">
                            <div className="dashboard-accident-container">
                                {accidents && Object.keys(accidents).map((VIN, index) => {
                                    const vehicleAccidents = accidents[VIN];
                                    return (
                                        <AccidentCard
                                            key={index}
                                            vin_number={VIN}
                                            iot_records={vehicleAccidents}
                                            onClick={() => {
                                                setAccident(vehicleAccidents);
                                                getImage(vehicleAccidents[0].image_location);
                                            }}
                                        />
                                    )
                                })}
                            </div>
                        </div>
                        <div className="inner-container">
                            <h4>Accident Details</h4>
                            <img src={imageUrl} alt="Accident" />
                            <p>The above image is the SHAP plot of the accident.</p>
                            <p>Click on the accident card on the left to view the details.</p>
                        </div>
                    </div>
                </DashboardSection>
            }
            {!user &&
                <SectionHeader>
                    <h1>
                        VAR for Insurancing <br></br>Autonomous Vehicles
                    </h1>
                    <p>
                        Insuring Autonomous Vehicle with Trust <br></br>decide insurance claims, premiums and policies fairly
                    </p>
                    <div className="home-page-button-container">
                        <MainButton
                            header="Claim-GPT"
                            onClick={() => navigate('/cgpt')}
                        />
                    </div>
                </SectionHeader>
            }
        </Background1>
    )
}

export default Home
