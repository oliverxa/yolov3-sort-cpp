# yolov3-sort-cpp

## 
```
libtorch version == 1.3.1
```
## download yolo weight in models
```
  cd models
  wget https://pjreddie.com/media/files/yolov3.weights 
```
## cmake
```
  mkdir build & cd build
  cmake ..
  make -j4
  ./yolo-sort-app ../videos/<your test video>
```
