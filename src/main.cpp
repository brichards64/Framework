#include "ToolChain.h"
#include "DummyTool.h"

int main(){

  ToolChain tools;
  DummyTool dummytool;
    

  tools.Add("DummyTool",&dummytool,"configfiles/DummyToolConfig");

  //int portnum=24000;
  //tools.Remote(portnum);
  //tools.Interactive();
  
  tools.Initialise();
  tools.Execute();
  tools.Finalise();
  
  
  return 0;
  
}
