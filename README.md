# yolov3-sort-cpp

# download yolo model in models
  cd models

  wget https://pjreddie.com/media/files/yolov3.weights 

# cmake

  mkdir build & cd build

  cmake ..

  make -j4

  ./yolo-sort-app ../videos/<your test video>
