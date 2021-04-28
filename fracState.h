#pragma once

//Window width and height
#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

/*Max iterations before a complex number is considered to be in the mandelbrot set.
Increasing this will result in a better quality fractal with reduced performance.
Reducing this will result in a lower quality fractal with improved performance.*/
#define MAX_ITER 1024

enum RENDERTYPE { CPU, GPU };

//Used to keep track of pan/zoom state and current render method of the fractal
struct fracState {
    double xZoomScale;
    double yZoomScale;
    double xPanOffset;
    double yPanOffset;
    RENDERTYPE calcMethod;

    //Initializes the starting x and y offsets of the fractal
    void initialize(int xPan, int yPan, enum RENDERTYPE method)
    {
        xPanOffset = xPan;
        yPanOffset = yPan;
        xZoomScale = xPan / -2;
        yZoomScale = yPan / -1.5;
        calcMethod = method;
    }
};