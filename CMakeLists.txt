cmake_minimum_required (VERSION 3.1)
set(name "TRACKER")
set(libname "yolo")
project ( ${name} )

set(CMAKE_BUILD_TYPE "Release")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
endif()

# set output path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})    
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin/Debug)    
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin/Release)  

#c++ version
set (CMAKE_CXX_STANDARD 11)

#cuda
find_package(CUDA)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
#    -O3 
	-gencode=arch=compute_61,code=\"sm_61,compute_61\"
    )
	
#opencv 
if ( WIN32 )
	set(OpenCV_DIR "E:/libs/OpenCV/x64/vc14/lib")
	set(OpenCV_CONFIG_PATH "E:/libs/OpenCV")
	set(WIN_HEADER_PATH "${PROJECT_SOURCE_DIR}/winheadfile")
    include_directories(${WIN_HEADER_PATH})
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# opengl
find_package(OpenGL REQUIRED)
set(GLAD_INCLUDE_PATH "/home/lgh/Documents/gimbleMulti/include/glad")
set(GLFW_INCLUDE_PATH "/home/lgh/Documents/gimbleMulti/include/GLFW")
	set(GLFW_LIBRARY "/usr/lib/x86_64-linux-gnu/libglfw.so")
	set(GLEW_INCLUDE_PATH "/home/lgh/Documents/gimbleMulti/include")
	set(GLEW_LIBRARY "/usr/lib/x86_64-linux-gnu/libGLEW.so")
	set(GLM_INCLUDE_PATH "/home/lgh/Documents/gimbleMulti/include")
include_directories(${GLAD_INCLUDE_PATH}
						${GLFW_INCLUDE_PATH}
						${GLEW_INCLUDE_PATH}
						${GLM_INCLUDE_PATH}
                        )
add_definitions(
		#-DTW_STATIC
		#-DTW_NO_LIB_PRAGMA
		#-DTW_NO_DIRECT3D
		-DGLEW_STATIC
		#-D_CRT_SECURE_NO_WARNINGS
     )

# pointgrey camera sdk
if ( WIN32 )
    set(PTGREY_SDK_INCLUDE_DIR "C:/Program Files/Point Grey Research/Spinnaker/include")
	set(PTGREY_SDK_LIBRARY_DEBUG debug "C:/Program Files/Point Grey Research/Spinnaker/lib64/vs2015/Spinnakerd_v140.lib")
	set(PTGREY_SDK_LIBRARY_RELEASE  optimized "C:/Program Files/Point Grey Research/Spinnaker/lib64/vs2015/Spinnaker_v140.lib")
else()
    set(PTGREY_SDK_INCLUDE_DIR "/usr/include/spinnaker")
    set(PTGREY_SDK_LIBRARY_DEBUG debug "/usr/lib/libSpinnaker.so")
    set(PTGREY_SDK_LIBRARY_RELEASE optimized "/usr/lib/libSpinnaker.so")
endif()
include_directories(${PTGREY_SDK_INCLUDE_DIR})

# XIMEA camera
if ( WIN32 )
    set(XIMEA_SDK_INCLUDE_DIR "C:/XIMEA/API")
    set(XIMEA_SDK_LIBRARY_DEBUG debug "C:/XIMEA/API/x64/xiapi64.lib")
    set(XIMEA_SDK_LIBRARY_RELEASE optimized "C:/XIMEA/API/x64/xiapi64.lib")
else ()
    set(XIMEA_SDK_INCLUDE_DIR "/opt/XIMEA/include")
    set(XIMEA_SDK_LIBRARY  
        "/usr/lib/libm3api.so"
        #"/opt/XIMEA/lib/libxiapi_dng_store.so"
        #"/opt/XIMEA/lib/libXIMEA_GenTL.cti"
        #"/lib/libusb-1.0.so"
        #"/lib/libraw1394.so"
       )
endif()
include_directories(${XIMEA_SDK_INCLUDE_DIR})

#ceres solver
set(CERES_INCLUDE_DIR "/usr/local/include/ceres")
set (CERES_LIBS
    "/usr/local/lib/libceres.a"
    "/usr/lib/x86_64-linux-gnu/libglog.so"
)
include_directories(${CERES_INCLUDE_DIR}
					${CERES_INCLUDE_DIR}/ceres
					${CERES_INCLUDE_DIR}/glog)

add_definitions(
    -DGOOGLE_GLOG_DLL_DECL=
	-DGLOG_NO_ABBREVIATED_SEVERITIES
)

set(YOLO_HEADER_FILES
	# C++ header file
includes/activation_layer.h
includes/activations.h
includes/avgpool_layer.h
includes/batchnorm_layer.h
includes/blas.h
includes/box.h
includes/classifier.h
includes/col2im.h
includes/connected_layer.h
includes/convolutional_layer.h
includes/cost_layer.h
includes/crnn_layer.h
includes/crop_layer.h
includes/cuda.h
includes/darknet.h
includes/data.h
includes/deconvolutional_layer.h
includes/demo.h
includes/detection_layer.h
includes/dropout_layer.h
includes/gemm.h
includes/getopt.h
includes/gettimeofday.h
includes/gru_layer.h
includes/http_stream.h
includes/im2col.h
includes/image.h
includes/layer.h
includes/list.h
includes/local_layer.h
includes/matrix.h
includes/maxpool_layer.h
includes/network.h
includes/normalization_layer.h
includes/option_list.h
includes/parser.h
includes/region_layer.h
includes/reorg_layer.h
includes/reorg_old_layer.h
includes/rnn_layer.h
includes/route_layer.h
includes/shortcut_layer.h
includes/softmax_layer.h
includes/stb_image.h
includes/stb_image_write.h
includes/tree.h
includes/upsample_layer.h
includes/utils.h
includes/yolo_layer.h
)


set(YOLO_CUDA_FILES
    src/darknet/activation_kernels.cu
    src/darknet/avgpool_layer_kernels.cu
    src/darknet/blas_kernels.cu
    src/darknet/col2im_kernels.cu
    src/darknet/convolutional_kernels.cu
    src/darknet/crop_layer_kernels.cu
    src/darknet/deconvolutional_kernels.cu
    src/darknet/dropout_layer_kernels.cu
    src/darknet/im2col_kernels.cu
    src/darknet/maxpool_layer_kernels.cu
    src/darknet/network_kernels.cu
)
set(YOLO_SOURCE_FILES 
    src/darknet/activation_layer.c
    src/darknet/activations.c
src/darknet/art.c
src/darknet/avgpool_layer.c
src/darknet/batchnorm_layer.c
src/darknet/blas.c
src/darknet/box.c
src/darknet/captcha.c
src/darknet/cifar.c
src/darknet/classifier.c
src/darknet/coco.c
src/darknet/col2im.c
src/darknet/compare.c
src/darknet/connected_layer.c
src/darknet/convolutional_layer.c
src/darknet/cost_layer.c
src/darknet/cpu_gemm.c
src/darknet/crnn_layer.c
src/darknet/crop_layer.c
src/darknet/cuda.c
src/darknet/darknet.c
src/darknet/data.c
src/darknet/deconvolutional_layer.c
src/darknet/demo.c
src/darknet/detection_layer.c
src/darknet/detector.c
src/darknet/dice.c
src/darknet/dropout_layer.c
src/darknet/gemm.c
src/darknet/getopt.c
src/darknet/gettimeofday.c
src/darknet/go.c
src/darknet/gru_layer.c
src/darknet/im2col.c
src/darknet/image.c
src/darknet/layer.c
src/darknet/list.c
src/darknet/local_layer.c
src/darknet/matrix.c
src/darknet/maxpool_layer.c
src/darknet/network.c
src/darknet/nightmare.c
src/darknet/normalization_layer.c
src/darknet/option_list.c
src/darknet/parser.c
src/darknet/region_layer.c
src/darknet/reorg_layer.c
src/darknet/reorg_old_layer.c
src/darknet/rnn.c
src/darknet/rnn_layer.c
src/darknet/rnn_vid.c
src/darknet/route_layer.c
src/darknet/shortcut_layer.c
src/darknet/softmax_layer.c
src/darknet/super.c
src/darknet/swag.c
src/darknet/tag.c
src/darknet/tree.c
src/darknet/upsample_layer.c
src/darknet/utils.c
src/darknet/voxel.c
src/darknet/writing.c
src/darknet/yolo.c
src/darknet/yolo_layer.c
src/darknet/http_stream.cpp
)



set(GIMBLE_HEADER_FILES
    gimble/CameraDriver/Exceptions.h
    gimble/CameraDriver/helper_cuda.h
    gimble/CameraDriver/helper_string.h
    gimble/CameraDriver/NPPJpegCoder.h
    gimble/CameraDriver/GenCameraDriver.h
    gimble/CameraDriver/RealCameraDriver.h
    # point grey camera
    gimble/PointGrey/PointGreyCamera.h
    # XIMEA camera
    gimble/XIMEA/XIMEACamera.h
    # File camera
    gimble/FileCamera/FileCamera.h
	#GimUtil
	gimble/GimUtil/GimUtil.h
	#serial
	gimble/Serial/Serial.h
	#cost
	gimble/Cost/Cost.h
	#common
	gimble/Common/Common.hpp
	#RobustStitcher
	gimble/RobustStitcher/FeatureMatch.h
	gimble/RobustStitcher/CameraParamEstimator.h
    gimble/RobustStitcher/Compositor.h
    gimble/RobustStitcher/BundleAdjustment.h
	#OpenGL
	gimble/OpenGL/shader_m.h
	gimble/OpenGL/glad.c
	gimble/OpenGL/Display.h
	#KCF
)

set(GIMBLE_SOURCE_FILES 
    # point grey camera
    gimble/PointGrey/PointGreyCamera.cpp
    # XIMEA camera
    gimble/XIMEA/XIMEACamera.cpp
    # File camera
    gimble/FileCamera/FileCamera.cpp
    # C++ source file
    gimble/CameraDriver/NPPJpegCoder.cpp
    # generic camera driver
    gimble/CameraDriver/GenCameraDriver.cpp
	gimble/CameraDriver/GenCameraDriverImpl.cpp
    gimble/CameraDriver/RealCameraDriver.cpp
	#GimUtil
	gimble/GimUtil/GimUtil.cpp
	#serial
	gimble/Serial/Serial.cpp
	#cost
	gimble/Cost/Cost.cpp
	#RobustStitcher
	gimble/RobustStitcher/FeatureMatch.cpp
	gimble/RobustStitcher/CameraParamEstimator.cpp
    gimble/RobustStitcher/Compositor.cpp
    gimble/RobustStitcher/BundleAdjustment.cpp
	#OpenGL
	gimble/OpenGL/display.cpp
	#KCF
    # main function
    gimble/main.cpp
)

set(FEATURE
    ${PROJECT_SOURCE_DIR}/src/feature/FeatureTensor.cpp
    ${PROJECT_SOURCE_DIR}/src/feature/model.cpp
    ${PROJECT_SOURCE_DIR}/src/feature/dataType.h
    ${PROJECT_SOURCE_DIR}/src/feature/FeatureTensor.h
    ${PROJECT_SOURCE_DIR}/src/feature/model.h
)
set(MATCHING
    ${PROJECT_SOURCE_DIR}/src/matching/kalmanfilter.cpp
    ${PROJECT_SOURCE_DIR}/src/matching/linear_assignment.cpp
    ${PROJECT_SOURCE_DIR}/src/matching/nn_matching.cpp
    ${PROJECT_SOURCE_DIR}/src/matching/track.cpp
    ${PROJECT_SOURCE_DIR}/src/matching/tracker.cpp
    ${PROJECT_SOURCE_DIR}/src/matching/kalmanfilter.h
    ${PROJECT_SOURCE_DIR}/src/matching/linear_assignment.h
    ${PROJECT_SOURCE_DIR}/src/matching/nn_matching.h
    ${PROJECT_SOURCE_DIR}/src/matching/track.h
    ${PROJECT_SOURCE_DIR}/src/matching/tracker.h
)
set(THIRD
    ${PROJECT_SOURCE_DIR}/src/thirdPart/hungarianoper.cpp
    ${PROJECT_SOURCE_DIR}/src/thirdPart/munkres.cpp
    ${PROJECT_SOURCE_DIR}/src/thirdPart/boostmatrixadapter.cpp
    ${PROJECT_SOURCE_DIR}/src/thirdPart/adapter.cpp
    ${PROJECT_SOURCE_DIR}/src/thirdPart/hungarianoper.h
    ${PROJECT_SOURCE_DIR}/src/thirdPart/munkres.h
    ${PROJECT_SOURCE_DIR}/src/thirdPart/matrix.h
    ${PROJECT_SOURCE_DIR}/src/thirdPart/boostmatrixadapter.h
    ${PROJECT_SOURCE_DIR}/src/thirdPart/adapter.h
)
set(SOURCE
    ${PROJECT_SOURCE_DIR}/src/errmsg.cpp
    ${PROJECT_SOURCE_DIR}/src/errmsg.h
)

include_directories(
    /usr/local/cuda/include
    /home/lgh/.cache/bazel/_bazel_lgh/5c3057d6871aa63630396c83a6b7aeeb/external/eigen_archive/Eigen
    /home/lgh/.cache/bazel/_bazel_lgh/5c3057d6871aa63630396c83a6b7aeeb/external/eigen_archive
    /home/lgh/.cache/bazel/_bazel_lgh/5c3057d6871aa63630396c83a6b7aeeb/external/nsync/public
    /home/lgh/Downloads/tensorflow-r1.4/bazel-genfiles
    /home/lgh/Downloads/tensorflow-r1.4
    ${PROJECT_SOURCE_DIR}/includes
)

include_directories(
    #${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/gimble
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/gimble/CameraDriver
	${PROJECT_SOURCE_DIR}/gimble/GimUtil
	${PROJECT_SOURCE_DIR}/gimble/Serial
	${PROJECT_SOURCE_DIR}/gimble/XIMEA	
	${PROJECT_SOURCE_DIR}/gimble/PointGrey		
	${PROJECT_SOURCE_DIR}/gimble/Cost
	${PROJECT_SOURCE_DIR}/gimble/Common
	${PROJECT_SOURCE_DIR}/gimble/RobustStitcher
	${PROJECT_SOURCE_DIR}/gimble/OpenGL
)

set(CUDNN_LIBS
    /usr/local/cuda/lib64/libcudnn.so
)

set(TENSORFLOWLIB
    /home/lgh/Downloads/tensorflow-r1.4/bazel-bin/tensorflow/libtensorflow_cc.so
    /home/lgh/Downloads/tensorflow-r1.4/bazel-bin/tensorflow/libtensorflow_framework.so
)


add_definitions(-DGPU)
add_definitions(-DCUDNN)
add_definitions(-DOPENCV)
add_definitions(-lusb-1.0)

#cuda_add_library(${libname} SHARED
    #${YOLO_CUDA_FILES}
    #${YOLO_SOURCE_FILES}
    #${YOLO_HEADER_FILES}
	#src/darknet/YOLODetector.cpp
    #src/darknet/YOLODetector.hpp
    #OPTIONS -arch sm_61
#)
#target_link_libraries(${libname} 
    #${OpenCV_LIBS}
    #${CUDNN_LIBS} 
    #${CUDA_LIBRARIES}
    #${CUDA_CUBLAS_LIBRARIES}
    #${CUDA_curand_LIBRARY}
#)


cuda_add_executable(${name}
    ${FEATURE}
    ${MATCHING}
    ${THIRD}
    ${SOURCE}
    ${GIMBLE_HEADER_FILES}
    ${GIMBLE_SOURCE_FILES}
)

target_link_libraries(${name} 
    ${OpenCV_LIBS}
    ${CUDNN_LIBS} 
    ${TENSORFLOWLIB} 
	${CUDA_LIBRARIES}
	${CUDA_CUBLAS_LIBRARIES}
	${CUDA_curand_LIBRARY}
	${CUDA_npp_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CERES_LIBS}
    ${GLFW_LIBRARY}
    ${PTGREY_SDK_LIBRARY_DEBUG} 
	${XIMEA_SDK_LIBRARY} 
    ${PTGREY_SDK_LIBRARY_RELEASE} 
    #yolo
    ${PROJECT_BINARY_DIR}/libyolo.so
)
