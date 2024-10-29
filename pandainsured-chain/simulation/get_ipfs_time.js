const { ApiPromise, WsProvider, Keyring } = require('@polkadot/api');
const CID = require('cids');
const crypto = require('crypto');
const axios = require('axios');

async function connectSubstrate() {
    const wsProvider = new WsProvider('ws://127.0.0.1:9944');
    const api = await ApiPromise.create({ provider: wsProvider });
    return api;
}


// Fetch block hash closest to the specified timestamp
async function getBlockHashAtTimestamp(api, targetTimestamp) {
    let currentBlock = await api.rpc.chain.getHeader();
    let currentBlockNumber = currentBlock.number.toNumber();
    let currentBlockTime = (await api.query.timestamp.now.at(currentBlock.hash)).toNumber();

    // Binary search to find the closest block
    let low = 1;
    let high = currentBlockNumber;
    let closestBlock = currentBlock;

    while (low <= high) {
        let mid = Math.floor((low + high) / 2);
        const blockHash = await api.rpc.chain.getBlockHash(mid);
        const blockTime = (await api.query.timestamp.now.at(blockHash)).toNumber();

        if (Math.abs(blockTime - targetTimestamp) < Math.abs(currentBlockTime - targetTimestamp)) {
            closestBlock = blockHash;
            currentBlockTime = blockTime;
        }

        if (blockTime < targetTimestamp) {
            low = mid + 1;
        } else if (blockTime > targetTimestamp) {
            high = mid - 1;
        } else {
            return blockHash;
        }
    }
    return closestBlock;
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



// Fetch data at a specific block state
async function getVehicleIdentifiers(manufacturer, model, vinPrefix, targetDate) {
    const api = await connectSubstrate();

    // Convert date to timestamp in milliseconds
    const targetTimestamp = targetDate.getTime();

    // Fetch block hash closest to the target date
    const blockHash = await getBlockHashAtTimestamp(api, targetTimestamp);
    const historicalApi = await api.at(blockHash);

    console.log(`Fetching data at block hash: ${blockHash}`);

    const encoder = new TextEncoder();
    const manufacturerBytes = encoder.encode(manufacturer);
    const modelBytes = encoder.encode(model);
    const vinPrefixBytes = encoder.encode(vinPrefix);

    const manufacturerHash = hashValue(manufacturerBytes);
    const modelHash = hashValue(modelBytes);
    const vinPrefixHash = hashValue(vinPrefixBytes);

    const maxLength = 32;

    const boundedManufacturer = toBoundedVec(manufacturerHash, maxLength);
    const boundedModel = toBoundedVec(modelHash, maxLength);
    const boundedVinPrefix = toBoundedVec(vinPrefixHash, maxLength);

    const keyring = new Keyring({ type: 'sr25519' });
    const signer = keyring.addFromUri('//Alice');

    console.log('Submitting extrinsic to get vehicle identifiers at historical state...');

    let allIdentifiers = [];

    console.log(Object.keys(historicalApi.events.iotStore.VehicleIdentifierAdded.meta.registry));

    await historicalApi.tx.iotStore
        .getVehicleIdentifiers(boundedManufacturer, boundedModel, boundedVinPrefix)
        .signAndSend(signer, ({ status, events }) => {
            if (status.isFinalized) {
                console.log(`Data at block hash ${blockHash}:`);
                events.forEach(({ event }) => {
                    if (event.section === 'iotStore') {
                        if (event.method === 'VehicleIdentifierRetrieved') {
                            const [who, matchingIdentifiers, vinPrefix] = event.data;
                            console.log(`Account: ${who}`);
                            console.log(`All matching identifiers for VIN prefix ${vinPrefix.toHuman()}: ${matchingIdentifiers.map((id) => id)}`);

                            matchingIdentifiers.forEach(async ([identifier, vin]) => {
                                const cidString = formatIdentifierToCID(identifier.toHuman());
                                await fetchFromPinata(cidString);
                            });
                            allIdentifiers = matchingIdentifiers;
                        }
                    }
                });
            }
        });
}

async function getEventsForVINAtTimestamp(targetVinPrefix, targetDate) {
    const api = await connectSubstrate();
    const targetTimestamp = targetDate.getTime();

    try {
        const blockHash = await getBlockHashAtTimestamp(api, targetTimestamp);
        const blockEvents = await api.query.system.events.at(blockHash);

        console.log(`Events for VIN prefix "${targetVinPrefix}" at block hash ${blockHash}:`);

        blockEvents.forEach((record) => {
            const { event } = record;

            if (event.section === 'iotStore' && event.method === 'VehicleIdentifierAdded') {
                const [who, matchingIdentifiers, vinPrefix] = event.data;

                // hash the target VIN prefix
                const encoder = new TextEncoder();

                const targetVinPrefixBytes = encoder.encode(targetVinPrefix);
                const targetVinPrefixHash = hashValue(targetVinPrefixBytes).toString('hex');

                // remove the 0x prefix and convert to string
                eventVinPrefix = vinPrefix.toHuman().slice(2);

                console.log(`Matching Identifiers: ${eventVinPrefix}`);
                console.log(`Target VIN Prefix Hash: ${targetVinPrefixHash}`);

                console.log(`Does VIN prefix match? ${eventVinPrefix === targetVinPrefixHash}`);

                if (eventVinPrefix === targetVinPrefixHash) {
                    console.log(`Event found for VIN: ${targetVinPrefixHash}`);
                    console.log(`Who: ${who.toHuman()}, Matching Identifiers: ${matchingIdentifiers.toHuman()}`);
                    fetchFromPinata(formatIdentifierToCID(matchingIdentifiers.toHuman()));
                }
            }
        });
    } catch (error) {
        console.error("Error fetching events:", error.message);
    }
}

// Define a target date for historical data retrieval, 30 seconds from now
const targetDate = new Date(Date.now() - 15 * 1000);

getEventsForVINAtTimestamp('1HGCM82633A337392', targetDate);

// Call the function with the target date
// getVehicleIdentifiers('Cruise', 'Bolt', '1HGCM82633A337392', targetDate);