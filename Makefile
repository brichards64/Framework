CUDAINC = -I$(CUDA_HOME)/include -I.
CUBALIB = -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math

DataModelInclude =
DataModelLib =
MyToolsInclude = $(CUDAINC)
MyToolsLib = $(CUDALIB)

all:  lib/libToolChain.so lib/libMyTools.so lib/libStore.so include/Tool.h lib/libSocketCom.so lib/libDataModel.so 

	g++ src/main.cpp -o main CUDA/daq_code.o -I include -L lib -lStore  -lToolChain -lDataModel -lSocketCom -lMyTools -lpthread $(DataModelInclude) $(DataModelLib) $(MyToolsInclude) $(MyToolsLib)

lib/libStore.so:

	cp src/Store/Store.h include/
	g++ --shared -c -fPIC -I inlcude src/Store/Store.cpp -o lib/libStore.so


include/Tool.h:

	cp src/Tool/Tool.h include/

lib/libSocketCom.so:

	cp src/SocketCom/SocketCom.h include/
	g++ -c --shared -fPIC src/SocketCom/SocketCom.cpp -I inlcude -lpthread -o lib/libSocketCom.so


lib/libToolChain.so: lib/libStore.so include/Tool.h lib/libSocketCom.so lib/libDataModel.so

	cp src/ToolChain/ToolChain.h include/
	g++ -c --shared -fPIC src/ToolChain/ToolChain.cpp -I include -lpthread -L /lib -lStore -lSocketCom -lDataModel -o lib/libToolChain.so $(DataModelInclude) $(DataModelLib)


clean: 

	rm -f include/*.h
	rm -f lib/*.so
	rm -f main
	rm -f UserTools/CUDA/daq_code.o


lib/libDataModel.so: lib/libStore.so

	cp DataModel/DataModel.h include/
	g++ -c --shared -fPIC DataModel/DataModel.cpp -I include -L lib -lStore -o lib/libDataModel.so $(DataModelInclude) $(DataModelLib)


lib/libMyTools.so: lib/libStore.so include/Tool.h lib/libDataModel.so UserTools/CUDA/daq_code.o

	cp UserTools/*.h include/
	g++  --shared  -fPIC UserTools/Unity.cpp UserTools/CUDA/daq_code.o -I include -L lib -lStore -lDataModel -o lib/libMyTools.so $(MyToolsInclude) $(MyToolsLib)

UserTools/CUDA/daq_code.o:

	cp UserTools/CUDA/*.h include/
	nvcc -c --shared -fPIC UserTools/CUDA/daq_code.cu -o UserTools/CUDA/daq_code.o -I include $(CUDAINC) $(NVCCFLAGS) $(CUDALIB)