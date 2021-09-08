#include "fracGen.hpp"

fracGen::fracGen(int windowWidth, int windowHeight, int maxIters) : WINDOW_WIDTH(windowWidth), WINDOW_HEIGHT(windowHeight), MAX_ITERS(maxIters)
{
    state = new fracState(windowWidth, windowHeight, maxIters, GPU);
    cudaMalloc((void**)&d_state, sizeof(fracState));
    result = new RGB[windowWidth * windowHeight];
    cudaMalloc((void**)&d_result, windowWidth * windowHeight * sizeof(RGB));

    threadPool = new fracThreadPool(NUM_CPU_THREADS, result, *state);
}

fracGen::~fracGen()
{
    delete state;
    delete[] result;
    delete threadPool;

    cudaFree(d_state);
    cudaFree(d_result);
}

void fracGen::start()
{
    //Initializes SDL video
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        logError("Error during SDL_Init", SDL_GetError, &initError, false);
    //Initializes SDL window
    if ((window = SDL_CreateWindow("Mandelbrot Surveyor", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, 0)) == NULL)
        logError("Error during SDL_CreateWindow", SDL_GetError, &initError, false);
    //Gets our window's surface
    if ((windowSurface = SDL_GetWindowSurface(window)) == NULL)
        logError("Error during SDL_GetWindowSurface", SDL_GetError, &initError, false);

    //Initializes SDL_ttf
    if (TTF_Init() == -1)
        logError("Error during TTF_Init", TTF_GetError, &initError, false);
    //Loads times new roman font
    if ((font = TTF_OpenFont("C:\\Windows\\Fonts\\times.ttf", 18)) == NULL)
        logError("Error during TTF_OpenFont", TTF_GetError, &initError, false);

    if (!initError)
    {
        while (!eventHandler())
        {
            auto start = std::chrono::high_resolution_clock::now();
            //If we are rendering the fractal in GPU mode, this runs
            if (state->calcMethod == GPU)
            {
                //Copy the state of the fractal to GPU mem
                cudaMemcpy(d_state, state, sizeof(fracState), cudaMemcpyHostToDevice);

                //Run the GPU calculations and wait for all threads to finish
                makeFracGPU << <NUM_BLOCKS, NUM_GPU_THREADS >> > (d_result, d_state);
                cudaDeviceSynchronize();

                //Copy results back from GPU mem
                cudaMemcpy(result, d_result, WINDOW_HEIGHT * WINDOW_WIDTH * sizeof(RGB), cudaMemcpyDeviceToHost);
            }
            //If we are rendering the fractal in CPU mode, this runs
            else
                threadPool->calcFrame();

            //Creates a surface out of our RGB result array and our ttf and displays it on the window
            fracSurface = SDL_CreateRGBSurfaceFrom((void*)result, WINDOW_WIDTH, WINDOW_HEIGHT, 24, WINDOW_WIDTH * 3, 0x0000FF, 0x00FF00, 0xFF0000, 0);
            ttfSurface = TTF_RenderText_Solid(font, getStateString().c_str(), { 255, 255, 255, 255 });
            SDL_BlitSurface(ttfSurface, NULL, fracSurface, NULL);
            SDL_BlitSurface(fracSurface, NULL, windowSurface, NULL);
            SDL_UpdateWindowSurface(window);
            SDL_FreeSurface(ttfSurface);
            SDL_FreeSurface(fracSurface);

            //Measures FPS and caps it at FPS_CAP by making the main thread sleep
            std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
            if (floor(1 / time.count()) > FPS_CAP)
            {
                state->fps = FPS_CAP;
                std::this_thread::sleep_for(std::chrono::milliseconds((int)floor(((1 / (double)FPS_CAP) - time.count()) * 1000)));
            }
            else
                state->fps = floor(1 / time.count());

            std::cout << getStateString() << std::endl;
        }
    }
    else
        logError("Init error", nullptr, nullptr, true);

    //Cleaning up SDL_ttf
    TTF_CloseFont(font);
    TTF_Quit();

    //Cleaning up SDL
    SDL_DestroyWindow(window);
    SDL_Quit();

    return;
}

/*Raises/lowers the zoom scale by a passed amount. Zooming will occur relative to the
  current mouse position. zoomAmount > 1 will result in a zoom-in, zoomAmount < 1 will result in a zoom-out*/
void fracGen::zoom(double zoomAmount, int mouseX, int mouseY)
{
    pair<double, double> beforeCenter = {
        ((double)mouseX + state->xPanOffset) / state->xZoomScale,
        ((double)mouseY + state->yPanOffset) / state->yZoomScale,
    };

    state->xZoomScale *= zoomAmount;
    state->yZoomScale *= zoomAmount;

    pair<double, double> afterCenter = {
        ((double)mouseX + state->xPanOffset) / state->xZoomScale,
        ((double)mouseY + state->yPanOffset) / state->yZoomScale,
    };

    state->xPanOffset += (beforeCenter.first - afterCenter.first) * state->xZoomScale;
    state->yPanOffset += (beforeCenter.second - afterCenter.second) * state->yZoomScale;
}

//Handles panning calculations
void fracGen::pan(const int& afterX, const int& afterY)
{
    state->xPanOffset += (double)(mousePanningCoords.x - afterX);
    state->yPanOffset += (double)(mousePanningCoords.y - afterY);
    mousePanningCoords.x = afterX;
    mousePanningCoords.y = afterY;
}

/*Handles mouse/keyboard events. Will return true if the user has chosen to close the window
or false otherwise*/
bool fracGen::eventHandler()
{
    static bool mousePanning;
    static bool modeKeyPressed = false;

    SDL_FlushEvents(SDL_TEXTEDITING, SDL_MOUSEMOTION);
    SDL_PollEvent(&event);
    SDL_FlushEvent(SDL_MOUSEWHEEL);

    if (event.type == SDL_QUIT || event.type == SDL_APP_TERMINATING)
        return true;

    //Holds the mouse coordinates
    int mouseX, mouseY;

    //Gets the keyboard and mouse state
    const unsigned char* kbState = SDL_GetKeyboardState(NULL);
    const unsigned int mouseState = SDL_GetMouseState(&mouseX, &mouseY);

    //Runs when user presses M key to switch render modes
    if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_m && !modeKeyPressed)
        modeKeyPressed = true;

    //Responds to the M key being released by switching render method
    if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_m && modeKeyPressed)
    {
        state->calcMethod == CPU ? state->calcMethod = GPU : state->calcMethod = CPU;
        modeKeyPressed = false;
    }

    //Runs when user starts panning
    if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT)
    {
        mousePanning = true;
        mousePanningCoords.x = event.button.x;
        mousePanningCoords.y = event.button.y;
    }

    //Runs when user stops panning
    if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT)
        mousePanning = false;

    //Handles panning events. Runs while the user is panning
    if (mousePanning)
        pan(mouseX, mouseY);
    else
    {
        //Handles keyboard zoom events (zoom in with W, zoom out with S)
        if (kbState[SDL_SCANCODE_W])
        {
            zoom(1.05, mouseX, mouseY);
            return false;
        }
        else if (kbState[SDL_SCANCODE_S])
        {
            zoom(0.95, mouseX, mouseY);
            return false;
        }
        //Handles scroll wheel zoom events 
        else if (event.type == SDL_MOUSEWHEEL)
        {
            if (event.wheel.y > 0)
                zoom(1.1, mouseX, mouseY);
            else if (event.wheel.y < 0)
                zoom(0.9, mouseX, mouseY);
            event.wheel.y = 0;
        }
    }

    return false;
}

void fracGen::logError(const char* errorMsg, const char* (*debugInfoFunc)(), bool* errorFlag, bool spawnWindow)
{
    if (errorMsg == nullptr)
        return;

    std::cerr << errorMsg;
    if (debugInfoFunc != nullptr)
        std::cerr << ": " << debugInfoFunc();
    std::cerr << std::endl;

    if (spawnWindow)
    {
        if (SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", errorMsg, NULL) != 0)
            logError("SDL_ShowSimpleMessageBox failed: ", SDL_GetError, nullptr, false);
    }
    if (errorFlag != nullptr)
        *errorFlag = true;
}

//Returns performance metric and fractal state as a string
std::string fracGen::getStateString()
{
    std::ostringstream strStream;
    strStream << "Rendered frame at " << state->fps << "fps\nMode: ";
    state->calcMethod == GPU ? strStream << "GPU" : strStream << "CPU";
    strStream << "\tZoom scale: " << (double)(state->yZoomScale / (state->windowHeight / 3.0));
    strStream << "\nPan X: " << state->xPanOffset << "\tPan Y: " << state->yPanOffset;
    return strStream.str();
}