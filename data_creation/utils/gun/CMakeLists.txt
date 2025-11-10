cmake_minimum_required(VERSION 3.10)
project(gun)

# Find the HEPMC3 package
find_package(HepMC3 3.2.0 REQUIRED)
find_package(Geant4 REQUIRED)
add_executable(gun gun.cpp) 
target_link_libraries(gun PUBLIC HepMC3::All ${Geant4_LIBRARIES} )

install(TARGETS gun DESTINATION ${CMAKE_INSTALL_BINDIR})


