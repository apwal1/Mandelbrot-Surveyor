Mandelbrot Surveyor is a multithreaded Mandelbrot fractal rendering program written in C++ using the SDL2 library.
As of now it features panning and zooming capabilities, but many more features are planned for future updates.

Current features:
  - Panning (hold left click and drag the fractal)
  - Zooming (using W and S keys or mouse wheel)
  - GPU rendering (don't try running this project on a device without a CUDA-capable GPU; you will definitely run into problems)
  - CPU rendering (you can switch between GPU and CPU rendering with the M key)

Planned future features:
  - Smooth coloring algorithm
  - Dynamic color-palette choosing
  - Screenshots
  - Saving and loading of specific locations within the fractal
  - Cross-platform capabilities

Potential future features:
  - Dynamic rotation of the Mandelbrot
  - Implementation of other fractals such as the Julia, Buddhabrot and Burning Ship fractals
