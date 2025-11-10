! main03.cmnd.
! https://github.com/HEP-FCC/FCC-config/blob/winter2023/FCCee/Generator/Pythia8/p8_ee_Ztautau_ecm91.cmd
! This file contains commands to be read in for a Pythia8 run.
! Lines not beginning with a letter or digit are comments.
! Names are case-insensitive  -  but spellings-sensitive!
! The settings here are illustrative, not always physics-motivated.

! 1) Settings used in the main program.
Main:timesAllowErrors = 5          ! how many aborts before run stops
Stat:showProcessLevel = on
Main:numberOfEvents = 100         ! number of events to generate
Random:setSeed = on
! 2) Settings related to output in init(), next() and stat().
Init:showChangedSettings = on      ! list changed settings
Init:showChangedParticleData = off ! list changed particle data
Next:numberCount = 100             ! print message every n events
Next:numberShowInfo = 1            ! print event information n times
Next:numberShowProcess = 1         ! print process record n times
Next:numberShowEvent = 0           ! print event record n times

! 3) Beam parameter settings. Values below agree with default ones.
Beams:idA = 11                   ! first beam, e 
Beams:idB = -11                   ! second beam, e 

! Beam energy spread: 0.132% x 45.594 GeV = 0.0602 GeV
Beams:allowMomentumSpread  = off


! 4) Hard process : Z->qq at Ecm=91 GeV
Beams:eCM = 91.188  ! CM energy of collision
PartonLevel:ISR = on               ! no initial-state radiation
PartonLevel:FSR = on               ! no final-state radiation

WeakSingleBoson:ffbar2gmZ = on
23:onMode = off
23:onIfAny = 1 2


