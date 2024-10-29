const { TextEncoder } = require('util');
const { ApiPromise, WsProvider, Keyring } = require('@polkadot/api');
const axios = require('axios');
const crypto = require('crypto');
const fs = require('fs');
const csv = require('csv-parser');
const results = [];

// Pinata API credentials (replace with your actual credentials)
const PINATA_API_KEY = 'a49a46ca8070f440a476';
const PINATA_SECRET_API_KEY = 'c877c47f195671e90d404118224493adbaa1805927f874e2e5efd66ec0e6814a';

const multibase = require('multibase'); // Add this module
const CID = require('cids'); // Add this module

function hashValue(value) {
    return crypto.createHash('sha256').update(value).digest();
}

function toBoundedVec(bytes, length) {
    if (bytes.length > length) {
        throw new Error(`Bounded Vec length exceeded: ${length}`);
    }
    const boundedVec = new Uint8Array(length);
    boundedVec.set(bytes);
    return Array.from(boundedVec); // Convert to a regular array to match Substrate's Vec<u8> format
}

async function uploadToIPFS(data) {
    const url = `https://api.pinata.cloud/pinning/pinJSONToIPFS`;
    const body = {
        pinataContent: data
    };

    try {
        const response = await axios.post(url, body, {
            headers: {
                'Content-Type': 'application/json',
                pinata_api_key: PINATA_API_KEY,
                pinata_secret_api_key: PINATA_SECRET_API_KEY,
            },
        });
        return response.data.IpfsHash; // This is the CID
    } catch (error) {
        console.error('Error uploading to IPFS:', error);
        throw error;
    }
}

async function connectSubstrate() {
    const wsProvider = new WsProvider('ws://127.0.0.1:9944');
    const api = await ApiPromise.create({ provider: wsProvider });
    return api;
}

async function main(vehicle_data) {
    try {
        console.log('Vehicle Data:', vehicle_data);

        if (!vehicle_data.Vehicle_Make || !vehicle_data.Vehicle_Model) {
            throw new Error('Manufacturer and model are required');
        }

        const cidString = await uploadToIPFS(vehicle_data);
        console.log(`Data uploaded to IPFS with CID: ${cidString}`);

        const api = await connectSubstrate();
        console.log('Connected to Substrate node');

        const keyring = new Keyring({ type: 'sr25519' });
        const sender = keyring.addFromUri('//Alice');

        const encoder = new TextEncoder();
        const manufacturerBytes = encoder.encode(vehicle_data.Vehicle_Make);
        const modelBytes = encoder.encode(vehicle_data.Vehicle_Model);
        const vinBytes = encoder.encode(vehicle_data.VIN);

        const manufacturerHash = hashValue(manufacturerBytes);
        const modelHash = hashValue(modelBytes);
        const vinHash = hashValue(vinBytes);

        const maxLength = 32; // Adjust based on your pallet's requirements
        const cidMaxLength = 59; // Adjust based on your pallet's requirements

        const boundedManufacturer = toBoundedVec(manufacturerHash, maxLength);
        const boundedModel = toBoundedVec(modelHash, maxLength);
        const boundedVIN = toBoundedVec(vinHash, maxLength);

        // Convert CID string to bytes
        const cid = new CID(cidString);
        const cidBytes = cid.bytes; // This gives you the byte representation of the CID

        const boundedIdentifier = toBoundedVec(cidBytes, cidMaxLength);

        console.log('Adding vehicle identifier to the blockchain...');
        console.log('Bounded Manufacturer:', boundedManufacturer);
        console.log('Bounded Model:', boundedModel);
        console.log('Bounded Identifier:', boundedIdentifier);

        const unsub = await api.tx.iotStore
            .addVehicleIdentifier(
                boundedManufacturer,
                boundedModel,
                boundedIdentifier,
                boundedVIN
            )
            .signAndSend(sender, (result) => {
                if (result.status.isInBlock) {
                    console.log(`Transaction included at blockHash ${result.status.asInBlock}`);
                } else if (result.status.isFinalized) {
                    console.log(`Transaction finalized at blockHash ${result.status.asFinalized}`);
                    unsub();
                }
            });
    } catch (error) {
        console.error('Error:', error);
    }
}

function get_data() {
    // Get Data from csv file
    fs.createReadStream('synthetic_vehicle_data.csv')
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => {
            // convert data to JSON
            // main(results[0]);
            // console.log(vehicle_data);
            // simulate data streaming
            for (let i = 0; i < results.length; i++) {
                setTimeout(() => {
                    main(results[i]);
                }, i * 6000);
            }
        });
}

get_data();