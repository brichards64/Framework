
DataModelInclude = -L/usr/lib64/root -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lMathCore -lThread -pthread -lm -ldl -rdynamic -pthread -m64 -I/usr/include/root -I include -L /lib -lStore
DataModelLib =
MyToolsInclude = ../obj/PSEC4_EVAL.o  ../obj/stdUSBl.o  ../obj/ScopePipe.o -I ../include/ -L/usr/lib64/root -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lMathCore -lThread -pthread -lm -ldl -rdynamic -pthread -m64 -I/usr/include/root -lusb -I ../include
MyToolsLib =

all: lib/libToolChain.so lib/libMyTools.so lib/libStore.so include/Tool.h lib/libSocketCom.so lib/libDataModel.so

	g++ src/main.cpp -o main -I include -L lib -lMyTools -lToolChain -lDataModel -lSocketCom -lpthread $(DataModelInclude) $(DataModelLib) $(MyToolsInclude) $(MyToolsLib)

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

lib/libDataModel.so: lib/libStore.so

	cp DataModel/DataModel.h include/
	g++ -c --shared DataModel/DataModel.cpp -I include -L lib -lStore -o lib/libDataModel.so $(DataModelInclude) $(DataModelLib)


lib/libMyTools.so: lib/libStore.so include/Tool.h lib/libDataModel.so

	cp UserTools/*.h include/
	g++  --shared -c UserTools/Unity.cpp -I include -L lib -lStore -lDataModel -o lib/libMyTools.so $(MyToolsInclude) $(MyToolsLib)

