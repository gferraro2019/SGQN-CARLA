#!/bin/sh
gnome-terminal -- bash -c "cd /home/dcas/g.ferraro/Desktop/CARLA/CARLA_0.9.14 && sh ./CarlaUE4.sh -RenderOffScreen -carla-port=2000 ; exec bash"
gnome-terminal -- bash -c "cd /home/dcas/g.ferraro/Desktop/CARLA/CARLA_0.9.14 && sh ./CarlaUE4.sh -RenderOffScreen  -carla-port=2003 ; exec bash"
#gnome-terminal -- bash -c "cd /home/dcas/g.ferraro/Desktop/CARLA/CARLA_0.9.14 && sh ./CarlaUE4.sh -#carla-port=2006 -RenderOffScreen; exec bash"

