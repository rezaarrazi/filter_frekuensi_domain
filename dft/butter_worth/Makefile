target = main
object = main.cpp
opencv_lib = -L"/usr/local/lib" -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
$(target):$(object)
	g++ $(object) -o $(target) $(opencv_lib)
