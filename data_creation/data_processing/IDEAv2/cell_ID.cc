#include "podio/ROOTFrameReader.h"
#include "podio/ROOTLegacyReader.h" // For reading legacy files
#include "podio/Frame.h"
#include "edm4hep/SimCalorimeterHitCollection.h"
#include "DD4hep/Detector.h"

int main() {
  auto reader = podio::ROOTFrameReader();
  //reader.openFile("ALLEGRO_sim_edm4hep.root");
  reader.openFile("/afs/cern.ch/user/b/brfranco/work/public/Step1_LAr_podio_edm4hep.root");
  const auto metadata = podio::Frame(reader.readEntry("metadata", 0));
  const auto& cellIDEnc = metadata.getParameter<std::string>(podio::collMetadataParamName("ECalBarrelEta", "CellIDEncoding"));
  //dd4hep::DDSegmentation::BitFieldCoder decoder("system:4,cryo:1,type:3,subtype:3,layer:8,module:11,eta:9");
  dd4hep::DDSegmentation::BitFieldCoder decoder(cellIDEnc);
  std::cout << cellIDEnc << std::endl;
  for (size_t i = 0; i < reader.getEntries("events"); ++i) {
    std::cout << "Next event ------------------------- "  << i << std::endl;
    auto event = podio::Frame(reader.readNextEntry("events"));
    auto& simCalorimeterHits = event.get<edm4hep::SimCalorimeterHitCollection>("ECalBarrelEta");
    for (auto simCalorimeterHit = simCalorimeterHits.begin(), end = simCalorimeterHits.end(); simCalorimeterHit != end; ++simCalorimeterHit){
      auto cellID = simCalorimeterHit->getCellID();
      int layer = decoder.get(cellID, "layer");
      std::cout << layer  << std::endl;
    }
  }
  return 0;
}