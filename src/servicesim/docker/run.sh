#!/usr/bin/env sh

# Runs a docker container with the image created by build.sh
# Requires
#   docker
#   nvidia-docker
#   an X server

until nvidia-docker ps
do
    echo "Waiting for docker server"
    sleep 1
done


# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

nvidia-docker run -it \
  -e DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=$XAUTH \
  -v "$XAUTH:$XAUTH" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "/etc/localtime:/etc/localtime:ro" \
  -v "/var/run/spnav.sock:/var/run/spnav.sock" \
  -v "/dev/input:/dev/input" \
  --rm=true \
  osrf/servicesim
