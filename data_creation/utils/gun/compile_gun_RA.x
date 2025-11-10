# gcc stuff
export PATH=/cvmfs/sft.cern.ch/lcg/contrib/gcc/11.2.0/x86_64-centos7/bin:$PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/contrib/gcc/11.2.0/x86_64-centos7/lib64:$LD_LIBRARY_PATH

# hepmc path
HEPMC3_PATH="/cvmfs/sw.hsf.org/spackages7/hepmc3/3.2.5/x86_64-centos7-gcc11.2.0-opt/rysg6"
export LD_LIBRARY_PATH=$HEPMC3_PATH/lib64:$LD_LIBRARY_PATH

# compile
g++ --std=c++11 -I${HEPMC3_PATH}/include -L${HEPMC3_PATH}/lib64 -lHepMC3 -o gun gun_random_angle.cpp
