docker run -it --rm \
    -v .:/app \
    -p 1028:1028 \
    --name headpose-detection-dev \
    qhan1028/headpose_detection $@
    bash
