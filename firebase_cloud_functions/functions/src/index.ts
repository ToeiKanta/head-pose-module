import * as functions from "firebase-functions";
import {firestore} from "firebase-admin/lib/firestore";
import Timestamp = firestore.Timestamp;

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
const db = admin.firestore();
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

exports.createFirstUser = functions.auth.user().onCreate((user) => {
  const uid = user.uid;
  db.collection("users").doc(uid).set({"processes": []}).then(() => {
    functions.logger.info(`User ${uid} created.`);
  });
});

exports.onUserDeleted = functions.auth.user().onDelete((user) => {
  const uid = user.uid;
  db.collection("users").doc(uid).delete().then(() => {
    functions.logger.info(`User ${uid} deleted.`);
  });
});
interface ProcessType {
  id: string,
  status: "QUEUE" | "PROCESS" | "SUCCESS" | "FAILED",
  percent: number,
  // eslint-disable-next-line camelcase
  error_msg: string,
  // eslint-disable-next-line camelcase
  created_date: Timestamp,
  // eslint-disable-next-line camelcase
  updated_date: Timestamp
}
interface UserType {
  processes: Array<ProcessType>
}
exports.addTaskQueue = functions.storage.object().onFinalize(async (object) => {
  // const fileBucket = object.bucket;
  // The Storage bucket that contains the file.
  const filePath = object.name; // File path in the bucket.
  const contentType = object.contentType; // File content type.
  // const metageneration = object.metageneration;
  // Number of times metadata has been generated. New objects have a value of 1.
  functions.logger.debug("content type :" + contentType);
  functions.logger.debug("filePath :" + filePath);
  // @ts-ignore
  // if (!contentType.startsWith("video/")) {
  //   return console.log("This is not a video.");
  // }
  const f = filePath.split(".");
  const fileExe = f[f.length-1].toLowerCase();
  if (fileExe !== "mp4" && fileExe !== "mov") {
    return functions.logger.error("file :" + fileExe + "is not support");
  }
  const pubSubClient = new PubSub();
  if ( filePath !== undefined ) {
    // paths
    // 0 = videos, 1 = <user_id>, 2 = <process_id>, 3 = 'result' or 'upload'
    const paths = filePath.split("/");
    const uid = paths[1];
    const pid = paths[2];
    const usersRef = db.collection("users").doc(uid);
    const doc = await usersRef.get();
    functions.logger.info(" uid : " + uid + " - pid : " + pid);
    if (doc.exists) {
      functions.logger.debug("Document data:", doc.data());
      const data : UserType = doc.data();
      const processes: Array<ProcessType> = data.processes;
      processes.push({
        "id": pid,
        "status": "QUEUE",
        "percent": 0,
        "error_msg": "",
        "created_date": Timestamp.now(),
        "updated_date": Timestamp.now(),
      });
      data.processes = processes;
      await usersRef.set(data);
      // eslint-disable-next-line max-len
      functions.logger.info(" User : " + uid + " - Add process : " + pid + " to queue");
      functions.logger.debug("New document data:", data);
    } else {
      // eslint-disable-next-line max-len
      functions.logger.error("User : " + uid + " - Doesn't exists on Firestore Database");
    }
    const dataBuffer = Buffer.from(filePath);
    try {
      const messageId = await pubSubClient.topic(topicName).publish(dataBuffer);
      functions.logger.info(`Message queue ${messageId} published.`);
    } catch (error) {
      functions.logger.error(`Error while publishing: ${error.message}`);
      process.exitCode = 1;
    }
  }
});
