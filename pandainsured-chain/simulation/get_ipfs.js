const { ApiPromise, WsProvider, Keyring } = require('@polkadot/api');
const CID = require('cids');
const crypto = require('crypto');
const axios = require('axios');


async function connectSubstrate() {
    const wsProvider = new WsProvider('ws://127.0.0.1:9944');
    const api = await ApiPromise.create({ provider: wsProvider });
    return api;
}

function formatIdentifierToCID(hexIdentifier) {
    // Remove the `0x` prefix and convert to byte array
    const hexString = hexIdentifier.slice(2);
    const bytes = Buffer.from(hexString, 'hex');

    // Extract only the first 34 bytes (CIDv1 for SHA-256)
    const trimmedBytes = bytes.slice(0, 34); // 2 bytes for prefix + 32-byte hash

    // Create CID object from the correctly sized byte array
    const cid = new CID(1, 'dag-pb', trimmedBytes); // 'dag-pb' is typical; use 'raw' if unstructured data

    return cid.toString();
}

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

// Pinata API credentials (replace with your actual credentials)
const PINATA_API_KEY = 'a49a46ca8070f440a476';
const PINATA_SECRET_API_KEY = 'c877c47f195671e90d404118224493adbaa1805927f874e2e5efd66ec0e6814a';


async function fetchFromPinata(cidString) {
    try {
        const response = await axios.get(`https://gateway.pinata.cloud/ipfs/${cidString}`, {
            headers: {
                'pinata_api_key': PINATA_API_KEY,
                'pinata_secret_api_key': PINATA_SECRET_API_KEY
            }
        });
        console.log('Data from Pinata:', response.data);
    } catch (error) {
        console.error('Error fetching from Pinata:', error.message);
    }
}


async function getVehicleIdentifiers(manufacturer, model, vinPrefix) {
    const api = await connectSubstrate();

    // Encode the parameters as per your runtime's expectations
    const encoder = new TextEncoder();
    const manufacturerBytes = encoder.encode(manufacturer);
    const modelBytes = encoder.encode(model);
    const vinPrefixBytes = encoder.encode(vinPrefix);

    // hash the vehicle data
    const manufacturerHash = hashValue(manufacturerBytes);
    const modelHash = hashValue(modelBytes);
    const vinPrefixHash = hashValue(vinPrefixBytes);

    const maxLength = 32; // Adjust as per your runtime configuration   

    // Create bounded vectors (adjust max lengths as per your runtime configuration)
    const boundedManufacturer = toBoundedVec(manufacturerHash, maxLength);
    const boundedModel = toBoundedVec(modelHash, maxLength);
    const boundedVinPrefix = toBoundedVec(vinPrefixHash, maxLength);

    const keyring = new Keyring({ type: 'sr25519' });
    const signer = keyring.addFromUri('//Alice');

    console.log('Submitting extrinsic to get vehicle identifiers...');

    let allIdentifiers = [];

    // Submit the extrinsic
    await api.tx.iotStore
        .getVehicleIdentifiers(boundedManufacturer, boundedModel, boundedVinPrefix)
        .signAndSend(signer, ({ status, events }) => {
            console.log('Transaction status:', status.type);
            if (status.isInBlock) {
                console.log('Transaction included in block:', status.asInBlock.toHex());
                // Process events here if needed
            } else if (status.isFinalized) {
                console.log('Transaction finalized in block:', status.asFinalized.toHex());
                // Process events and handle the events emitted by the transaction
                events.forEach(({ event }) => {
                    if (event.section === 'iotStore') {
                        if (event.method === 'VehicleIdentifierRetrieved') {
                            const [who, matchingIdentifiers, vinPrefix] = event.data;
                            console.log(`Account: ${who}`);
                            console.log(`All matching identifiers for VIN prefix ${vinPrefix.toHuman()}: ${matchingIdentifiers.map((id) => id)}`);

                            matchingIdentifiers.forEach(async ([identifier, vin]) => {
                                const cidString = formatIdentifierToCID(identifier.toHuman()); // Assume identifier is a CID
                                console.log(`Identifier: ${cidString}, VIN Prefix: ${vin.toHuman()}`);

                                // Fetch the data from Pinata using the identifier
                                await fetchFromPinata(cidString);
                            });
                            allIdentifiers = matchingIdentifiers;

                        } else if (event.method === 'VinPrefixNotFound') {
                            console.log('No matching VIN prefix found.');
                        } else if (event.method === 'NoIdentifiersFound') {
                            console.log('No identifiers found for the given vehicle information.');
                        }
                    }
                });
            }
        });
}

getVehicleIdentifiers(
    'Cruise',
    'Bolt',
    '1HGCM82633A337392'
)
