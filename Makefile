final:run
	@echo "ready to run this code. "
	./run
run:code/Detect.cpp main.cpp code/Utils.cpp code/Detector.cpp
# 	g++ -o YOLO2 yolo.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_dnn -lopencv_imgproc -lstdc++
# 	g++ -o YOLO2 YOLO2.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_dnn -lopencv_imgproc -lstdc++ 
	g++ -o run main.cpp code/Detect.cpp code/Detector.cpp code/Utils.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_dnn -lopencv_imgproc -lstdc++ 