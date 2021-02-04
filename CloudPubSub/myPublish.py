from google.cloud import pubsub_v1
from .consts import consts

def publish(message):
    publisher_options = pubsub_v1.types.PublisherOptions(enable_message_ordering=True)
    # Sending messages to the same region ensures they are received in order
    # even when multiple publishers are used.
    client_options = {"api_endpoint": "us-east1-pubsub.googleapis.com:443"}
    publisher = pubsub_v1.PublisherClient(
        publisher_options=publisher_options, client_options=client_options
    )
    # The `topic_path` method creates a fully qualified identifier
    # in the form `projects/{project_id}/topics/{topic_id}`
    topic_path = publisher.topic_path(consts.PUBSUB_INFO['project_id'], consts.PUBSUB_INFO['topic_id'])

    # Data must be a bytestring
    data = message.encode("utf-8")
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data=data)
    print(future.result())

    print(f"Published messages with ordering keys to {topic_path}.")