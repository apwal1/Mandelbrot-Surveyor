#pragma once

enum RENDERTYPE { CPU, GPU };

//Used to keep track of the general state and settings of the fractal such as pan/zoom state, render mode, window dimensions, maximum iterations and FPS
struct fracState {
    double xZoomScale;
    double yZoomScale;
    double xPanOffset;
    double yPanOffset;
    RENDERTYPE calcMethod;
    int fps;
    const int windowWidth;
    const int windowHeight;
    const int maxIters;
    //Initializes the starting x and y offsets of the fractal, as well as the zoom scale based on those offsets
    fracState(int width, int height, int iters, enum RENDERTYPE method) : windowWidth(width), windowHeight(height), maxIters(iters)
    {
        xPanOffset = -width / 2;
        yPanOffset = -height / 2;
        xZoomScale = xPanOffset / -2;
        yZoomScale = yPanOffset / -1.5;
        calcMethod = method;
        fps = 0;
    }
};

struct RGB {
    char r;
    char g;
    char b;
};