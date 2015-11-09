# SRC Directory
**************************


Source directory contains the main.cpp and mager framework code incuding:


The virtual Tool.h headder that Usertools should inherit from in order to work in the tool chain.

The ToolChain code for controlling and storing Tools

The Store code for inefficent mapped variable store used in  configuration file readin.

The Socket comunication class (work in progress)



***********************
Main.cpp Description
***********************

The main.cpp needs to be endited by the user in order to inculde the necessary tools for use in the DAQ. The current example code loads 1 tool known as DummyTool and then executes the functions Initialise Execute and Finalise on each tool sequenctially. 

Code Description:

     1) All UserTools that are required in the frame work need their individual headers included along with the ToolChain.h at tthe start of the code

     	    #include "ToolChain.h"
    	    #include "DummyTool.h"

    	    int main(){


     2) Each tool and the toolchain need to then be instansiated 


     	      ToolChain tools;
  	      DummyTool dummytool;

     3) The Tools must then be added to the tool chain one by one in order. Three arguments can be provided when adding the tool (1)tool name in the tool chain, (2) pointer to the tool, (3) conifguration file [optional] .

     	      tools.Add("DummyTool",&dummytool,"configfiles/DummyToolConfig");


      4) The ToolChain will will Initialise, then Execute and Finalise each tool in the chain in the order they have been added. The execution of this chain can be done in three ways.

      	     	       1) Precompiles execution (shown in main.cpp)
		       	  	      
				       tools.Initialise();
				       tools.Execute();
				       tools.Finalise();

		       2) Interactive execution. Initialise execute and finalise can be run via command line as well as start, stop, pause, unpause, and quit.

		       	  	      tools.Interactive();

		       3) Remote execution. Toolchain commands like interactive and can be passed between processes or computers via TCP.

					tools.Remote(portnum);


