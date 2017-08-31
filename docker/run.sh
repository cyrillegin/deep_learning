BRO=$(cd $(dirname "${BASH_SOURCE[0]}")/../ && pwd -P)
DEEPUID=$UID
DEEPGID=$(id $USER -g)

HOSTPORT=8000
CONTAINERNAME=deeplearning

docker run -t -i -P \
    --privileged \
    -e "container=docker" \
    -v /sys/fs/cgroup:/sys/fs/cgroup \
    --rm=true \
    --name $CONTAINERNAME \
    -e DEEPUID=$DEEPUID \
    -e DEEPGID="$DEEPGID" \
    -e DEEPOSTYPE=$OSTYPE \
    -v $BRO:/home/deeplearning/ \
    -p $HOSTPORT:8000 \
    deeplearning/app \
