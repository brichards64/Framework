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
	g++ --shared -c -I inlcude src/Store/Store.cpp -o lib/libStore.so


include/Tool.h:

	cp src/Tool/Tool.h include/

lib/libSocketCom.so:

	cp src/SocketCom/SocketCom.h include/
	g++ -c --shared src/SocketCom/SocketCom.cpp -I inlcude -lpthread -o lib/libSocketCom.so


lib/libToolChain.so: lib/libStore.so include/Tool.h lib/libSocketCom.so lib/libDataModel.so

	cp src/ToolChain/ToolChain.h include/
	g++ -c --shared src/ToolChain/ToolChain.cpp -I include -lpthread -L /lib -lStore -lSocketCom -lDataModel -o lib/libToolChain.so $(DataModelInclude) $(DataModelLib)


clean: 

	rm include/*.h
	rm lib/*.so
	rm main
	rm -f UserTools/CUDA/daq_code


lib/libDataModel.so: lib/libStore.so

	cp DataModel/DataModel.h include/
	g++ -c --shared DataModel/DataModel.cpp -I include -L lib -lStore -o lib/libDataModel.so $(DataModelInclude) $(DataModelLib)


lib/libMyTools.so: lib/libStore.so include/Tool.h lib/libDataModel.so CUDA/daq_code

	cp UserTools/*.h include/
	g++  --shared -c UserTools/Unity.cpp CUDA/daq_code.o -I include -L lib -lStore -lDataModel -o lib/libMyTools.so $(MyToolsInclude) $(MyToolsLib)

CUDA/daq_code: UserTools/CUDA/daq_code.cu Makefile
	nvcc -c UserTools/CUDA/daq_code.cu -o UserTools/CUDA/daq_code.o $(CUDAINC) $(NVCCFLAGS) $(CUDALIB)
