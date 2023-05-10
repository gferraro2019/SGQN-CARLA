#!/bin/sh
gnome-terminal -- bash -c "cd /home/dcas/g.ferraro/Desktop/CARLA/CARLA_0.9.14 && sh ./CarlaUE4.sh -carla-port=2000 -RenderOffScreen; exec bash"
gnome-terminal -- bash -c "cd /home/dcas/g.ferraro/Desktop/CARLA/CARLA_0.9.14 && sh ./CarlaUE4.sh -carla-port=2003 -RenderOffScreen; exec bash"
#gnome-terminal -- bash -c "cd /home/dcas/g.ferraro/Desktop/CARLA/CARLA_0.9.14 && sh ./CarlaUE4.sh -#carla-port=2006 -RenderOffScreen; exec bash"

