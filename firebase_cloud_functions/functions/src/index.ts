import * as functions from "firebase-functions";

// // Start writing Firebase Functions
// // https://firebase.google.com/docs/functions/typescript
//

const region = "asia-southeast2";
const builderFunction = functions.region(region).https;

export const helloWorld = builderFunction.onRequest((request, response) => {
  functions.logger.info("Hello logs!", {structuredData: true});
  response.send("Hello from Firebase!");
});

// Imports the Google Cloud client library
const {PubSub} = require("@google-cloud/pubsub");

// [START import]
const admin = require("firebase-admin");
admin.initializeApp();
// const spawn = require("child-process-promise").spawn;
// const path = require("path");
// const os = require("os");
// const fs = require("fs");

// [PUB SUB]
const topicName = "TestApp";

// // Create and Deploy Your First Cloud Functions
// // https://firebase.google.com/docs/functions/write-firebase-functions
//
exports.helloWorld = functions.https.onRequest((request, response) => {
  functions.logger.info("Hello logs!", {structuredData: true});
  response.send("Hello from Firebase!");
});

exports.addTaskQueue = functions.storage.object().onFinalize(async (object) => {
  // const fileBucket = object.bucket;
  // The Storage bucket that contains the file.
  const filePath = object.name; // File path in the bucket.
  const contentType = object.contentType; // File content type.
  // const metageneration = object.metageneration;
  // Number of times metadata has been generated. New objects have a value of 1.
  const pubSubClient = new PubSub();
  functions.logger.info("content type :" + contentType);
  functions.logger.info("filePath :" + filePath);
  // Publishes the message as a string,
  // e.g. "Hello, world!" or JSON.stringify(someObject)
  if ( filePath !== undefined ) {
    const dataBuffer = Buffer.from(filePath );
    try {
      const messageId = await pubSubClient.topic(topicName).publish(dataBuffer);
      functions.logger.info(`Message ${messageId} published.`);
    } catch (error) {
      functions.logger.error(`Error while publishing: ${error.message}`);
      process.exitCode = 1;
    }
  }
});
