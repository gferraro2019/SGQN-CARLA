sh setup/install_mujoco_deps_old.sh
sh setup/install_envs.sh
mkdir -p datasets/places365_standard
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar 
tar -xf places365standard_easyformat.tar -C datasets/places365_standard
rm places365standard_easyformat.tar
