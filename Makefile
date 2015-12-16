
DataModelInclude =
DataModelLib =
MyToolsInclude =
MyToolsLib =

all: lib/libMyTools.so lib/libToolChain.so lib/libStore.so include/Tool.h  lib/libDataModel.so

	g++ src/main.cpp -o main -I include -L lib -lStore -lMyTools -lToolChain -lDataModel -lpthread $(DataModelInclude) $(DataModelLib) $(MyToolsInclude) $(MyToolsLib) -L /usr/lib64 -lzmq

lib/libStore.so:

	cp src/Store/Store.h include/
	g++ --shared -c -I inlcude src/Store/Store.cpp -o lib/libStore.so


include/Tool.h:

	cp src/Tool/Tool.h include/


lib/libToolChain.so: lib/libStore.so include/Tool.h lib/libDataModel.so lib/libMyTools.so

	cp src/ToolChain/*.h include/
	cp src/NodeDeamon/cppzmq/zmq.hpp include/
	g++ -c --shared src/ToolChain/ToolChain.cpp -I include -lpthread -L /lib -lStore -lDataModel -lMyTools -o lib/libToolChain.so $(DataModelInclude) $(DataModelLib) -L /usr/lib64 -lzmq


clean: 
	rm -f include/*.h
	rm -f lib/*.so
	rm -f main

lib/libDataModel.so: lib/libStore.so

	cp DataModel/DataModel.h include/
	g++ -c --shared DataModel/DataModel.cpp -I include -L lib -lStore -o lib/libDataModel.so $(DataModelInclude) $(DataModelLib)


lib/libMyTools.so: lib/libStore.so include/Tool.h lib/libDataModel.so

	cp UserTools/*.h include/
	cp UserTools/Factory/*.h include/
	g++  --shared -c UserTools/Factory/Factory.cpp -I include -L lib -lStore -lDataModel -o lib/libMyTools.so $(MyToolsInclude) $(MyToolsLib)

