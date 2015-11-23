#include "ToolChain.h"
#include "DummyTool.h"
#include "WCSimASCIReader.h"
#include "GPUProcessor.h"
#include "TriggerOutput.h"

int main(){

  ToolChain tools;
  WCSimASCIReader reader;
  GPUProcessor processor;
  TriggerOutput output;
  
  tools.Add("WCSim ASCI Reader",&reader,"configfiles/WCSimASCIReaderConfig");
  tools.Add("GPU Processor",&processor);
  tools.Add("Trigger Output",&output,"configfiles/TriggerOutputConfig");
 
  //tools.Remote(portnum);
  //tools.Interactive();
  
  tools.Initialise();
  tools.Execute();
  tools.Finalise();
  
  
  return 0;
  
}
