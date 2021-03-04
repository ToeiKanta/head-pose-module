import * as functions from "firebase-functions";
import {firestore} from "firebase-admin/lib/firestore";
import Timestamp = firestore.Timestamp;

// // Start writing Firebase Functions
// // https://firebase.google.com/docs/functions/typescript
//
// Imports the Google Cloud client library
const {Storage} = require("@google-cloud/storage");
const bucketName = "deepheadposeapp.appspot.com";

// Creates a client
const storage = new Storage();

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
// eslint-disable-next-line max-len
exports.helloWorld = functions.region(region).https
    .onRequest((request, response) => {
      functions.logger.info("Hello logs!", {structuredData: true});
      response.send("Hello from Firebase!");
    });

exports.createFirstUser = functions.region(region).auth.user()
    .onCreate((user) => {
      const uid = user.uid;
      db.collection("users").doc(uid)
          .set({"processes": []}, {merge: true}).then(() => {
            functions.logger.info(`User ${uid} created.`);
          });
    });

exports.onUserDeleted = functions.region(region).auth.user()
    .onDelete((user) => {
      const uid = user.uid;
      db.collection("users").doc(uid).delete().then(() => {
        functions.logger.info(`User ${uid} deleted.`);
      });
    });
interface ProcessType {
  id: string,
  status: "UPLOAD" | "SETUP" | "QUEUE" | "PROCESS" | "SUCCESS" | "FAILED",
  // eslint-disable-next-line camelcase
  owner: firestore.DocumentReference,
  // eslint-disable-next-line camelcase
  process_cycle: number,
  // eslint-disable-next-line camelcase
  file_path: string,
  percent: number,
  // eslint-disable-next-line camelcase
  error_msg: string,
  // eslint-disable-next-line camelcase
  created_date: Timestamp,
  // eslint-disable-next-line camelcase
  updated_date: Timestamp
}
interface UserType {
  processes: Array<firestore.DocumentReference>;
  // eslint-disable-next-line camelcase
  updated_date?: Timestamp;
}
const uploadVideoToPath = async (filePath:string) => {
  const f = filePath.split(".");
  const fileExe = f[f.length-1].toLowerCase();
  if (fileExe !== "mp4" && fileExe !== "mov") {
    return functions.logger.error("file :" + fileExe + "is not support");
  }
  if ( filePath !== undefined ) {
    // paths
    // 0 = videos, 1 = <user_id>, 2 = <process_id>, 3 = 'result' or 'upload'
    const paths = filePath.split("/");
    const uid = paths[1];
    const pid = paths[2];
    const usersRef = db.collection("users").doc(uid);
    const userDoc = await usersRef.get();
    functions.logger.info(" uid : " + uid + " - pid : " + pid);
    if (userDoc.exists) {
      functions.logger.debug("Document data:", userDoc.data());
      const userData : UserType = userDoc.data();
      const processesId = userData.processes;
      processesId.push(db.doc("processes/" + pid));
      userData.processes = processesId;
      await usersRef.update(userData);
      const processData: ProcessType = {
        "id": pid,
        "owner": db.collection("users").doc(uid),
        "file_path": filePath,
        "status": "UPLOAD",
        "percent": 0,
        "process_cycle": 0,
        "error_msg": "",
        "created_date": Timestamp.now(),
        "updated_date": Timestamp.now(),
      };
      db.collection("processes").doc(pid)
          .set(processData, {merge: true}).then(() => {
            functions.logger.info(`Process ${pid} created!!`);
          });
      // eslint-disable-next-line max-len
      functions.logger.info(" User : " + uid + " - Add process : " + pid + " to queue");
      functions.logger.debug("New document data:", userData);
    } else {
      // eslint-disable-next-line max-len
      functions.logger.error("User : " + uid + " - Doesn't exists on Firestore Database");
    }
  }
};
exports.addTaskQueue = functions.region(region).storage.object()
    .onFinalize(async (object) => {
      // const fileBucket = object.bucket;
      // The Storage bucket that contains the file.
      const filePath = object.name; // File path in the bucket.
      const contentType = object.contentType; // File content type.
      // const metageneration = object.metageneration;
      // Number of times metadata has been generated.
      // New objects have a value of 1.
      functions.logger.debug("content type :" + contentType);
      functions.logger.debug("filePath :" + filePath);
      // @ts-ignore
      // if (!contentType.startsWith("video/")) {
      //   return console.log("This is not a video.");
      // }
      uploadVideoToPath(filePath);
    });

exports.processUpdate = functions.region(region).firestore
    .document("processes/{processId}")
    .onUpdate(async (change, context) => {
      // https://firebase.google.com/docs/functions/firestore-events
      // Get an object representing the document
      // e.g. {'name': 'Marie', 'age': 66}
      const newValue = change.after.data();

      // ...or the previous value before this update
      const previousValue = change.before.data();

      // We'll only update if the name has changed.
      // This is crucial to prevent infinite loops.
      if (newValue.status == previousValue.status &&
          newValue.percent == previousValue.percent) {
        return null;
      }
      // if status change to QUEUE
      if (newValue.status != previousValue.status &&
          newValue.status === "QUEUE") {
        const pubSubClient = new PubSub();
        const dataBuffer = Buffer.from(newValue.file_path);
        try {
          // eslint-disable-next-line max-len
          const messageId = await pubSubClient.topic(topicName).publish(dataBuffer);
          functions.logger.info(`Message queue ${messageId} published.`);
        } catch (error) {
          functions.logger.error(`Error while publishing: ${error.message}`);
          process.exitCode = 1;
        }
      }
      // if status change to PROCESS
      if (newValue.status != previousValue.status &&
            newValue.status === "PROCESS") {
        await change.after.ref.set({
          process_cycle: newValue.process_cycle + 1,
        }, {merge: true});
      }
      // access a particular field as you would any JS property
      // const name = newValue.name;

      // Then return a promise of a set operation to update the count
      return change.after.ref.set({
        updated_date: Timestamp.now(),
      }, {merge: true});
    });
// https://firebase.google.com/docs/functions/firestore-events
exports.processDeleted = functions.region(region).firestore
    .document("processes/{processId}")
    .onDelete(async (snapshot, context) => {
      const process: FirebaseFirestore.DocumentData = snapshot.data();
      const processId = context.params.processId;
      const userDoc = await process.owner.get();
      if (!userDoc.exists) {
        console.log("No such document!");
      } else {
        const user : UserType = userDoc.data();
        const processDeleted = db.collection("processes").doc(processId);
        user.processes = user.processes.filter(
            (processRef) => processRef.path != processDeleted.path);
        process.owner.set({
          processes: user.processes,
        }, {merge: true});
      }
    });
exports.userProcessDeleted = functions.region(region).firestore
    .document("users/{userId}")
    .onUpdate(async (change, context) => {
      // https://firebase.google.com/docs/functions/firestore-events
      // Get an object representing the document
      // e.g. {'name': 'Marie', 'age': 66}
      const newValue = change.after.data();

      // ...or the previous value before this update
      const previousValue = change.before.data();

      // We'll only update if processes has changed.
      // This is crucial to prevent infinite loops.
      if (newValue.processes.length == previousValue.processes.length) {
        return null;
      }
      // if processes deleted
      if (newValue.processes.length < previousValue.processes.length) {
        const newPaths = newValue.processes
            .map((ref: { path: String; }) => ref.path);
        for (const data of previousValue.processes) {
          if (!newPaths.includes(data.path)) {
            await data.delete();
            functions.logger.info("Delete collection " + data.path + ".");
            // Deletes the file from the bucket
            const filename = "videos/" +
                context.params.userId +
                "/" + data.path.split("/")[1] + "/";
            const bucket = storage.bucket(bucketName);
            // await storage.bucket(bucketName).file(filename).delete();
            bucket.deleteFiles({
              prefix: filename,
            }, function(err: any) {
              if (!err) {
                // All files in the `images` directory have been deleted.
                functions.logger.info(`gs://${bucketName}/${filename} deleted.`);
              } else {
                functions.logger.error(err);
              }
            });
          }
        }
      }
      // Then return a promise of a set operation to update the count
      return change.after.ref.set({
        updated_date: Timestamp.now(),
      }, {merge: true});
    });
