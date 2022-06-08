echo "Build ORBSLAM 2 ..."

cd orb-slam2
./build.sh
cd build
sudo make install
cd ..

echo "Build Python Wrapper ..."
cd ..
cd orbslam_python
mkdir build
cd build
cmake ..
make
sudo make install
