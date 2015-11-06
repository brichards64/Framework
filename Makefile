
DataModelInclude =
DataModelLib =
MyToolsInclude =
MyToolsLib =

all: lib/libToolChain.so lib/libMyTools.so lib/libStore.so include/Tool.h lib/libSocketCom.so lib/libDataModel.so

	g++ src/main.cpp -o main -I include -L lib -lMyTools -lToolChain -lStore -lDataModel -lSocketCom -lpthread $(DataModelInclude) $(DataModelLib) $(MyToolsInclude) $(MyToolsLib)

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

	rm main
	rm lib/*.so
	rm include/*.h


lib/libDataModel.so: lib/libStore.so

	cp DataModel/DataModel.h include/
	g++ -c --shared DataModel/DataModel.cpp -I include -L lib -lStore -o lib/libDataModel.so $(DataModelInclude) $(DataModelLib)


lib/libMyTools.so: lib/libStore.so include/Tool.h lib/libDataModel.so

	cp UserTools/*.h include/
	g++  --shared -c UserTools/Unity.cpp -I include -L lib -lStore -lDataModel -o lib/libMyTools.so $(MyToolsInclude) $(MyToolsLib)

