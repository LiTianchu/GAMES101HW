# GAMES101HW

MY UCSB GAMES101 Computer Graphics homework implementations.  
Course Site: https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html

To Build on Windows
1. Download MinGW32 for windows, add the /bin/ folder into environment variables
2. Download CMake with version heigher than 3.10, add the /bin/ folder path into environment variables
3. Download OpenCV source code, create a /build/ folder in the hierarchy, compile OpenCV into the folder using CMake with MinGW Makefiles as Generator
4. Add the /bin/ folder of the built OpenCV into the environment variables
5. Download Eigen3, Eigen3 only need to import the header file so no need to build using CMake
6. Include the dependencies accordingly in the CMakeLists.txt and VS Code include paths