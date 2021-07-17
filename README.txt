Mandelbrot Surveyor is a multithreaded Mandelbrot fractal rendering program written in C++ using the SDL2 library and CUDA C++.

Current features:
  - Panning (hold left click and drag the fractal)
  - Zooming (using W and S keys or mouse wheel)
  - GPU rendering using CUDA C++ (don't try running this project on a device without a CUDA-capable GPU; you will definitely run into problems)
  - CPU rendering (you can switch between GPU and CPU rendering with the M key)
  - Smooth coloring algorithm

Planned future features:
  - Dynamic color-palette choosing
  - Screenshots
  - Saving and loading of specific locations within the fractal
  - Cross-platform capabilities
  - Detection of CUDA-capable GPU and dynamically loading cuda DLL so that the project can be run on devices without CUDA in a CPU only mode

Potential future features:
  - Dynamic rotation of the Mandelbrot
  - Implementation of other fractals such as the Julia, Buddhabrot and Burning Ship fractals
