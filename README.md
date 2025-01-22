### Website is nonfunctional right now

### Overview

This is a linear programming solver inspired by the IOR Tutorial from the "Introduction to Operations Research" textbook. It is meant to be a teaching aid, and show each iteration of the Simplex method in either Tableau or Matrix form, and the textbook's simplified Interior Point method.

The IOR tutorial is an .exe file which makes it impossible to run on a Mac device without a virtual machine. This is meant to alleviate that problem in a convenient website (deployed on the gh-pages branch).

### Structure

The project was built with Rust and it was developed with the Yew framework so that it could be compiled into WebAssembly and deployed as a static website on GitHub Pages - that is, running natively in a browser without server input.

### Issues / Future Plans

Currently, only the Simplex Tableau solver without artificial variables works correctly. There are some issues with displaying the matrix form and with passing correct artificial variables into the solvers, but the underlying solvers work correctly.

In addition, the intention is to allow the user to choose between Big M and Two-Phase for the Tableau solver. However, the logic and the display does not work fully yet.

I also intend on adding a graphical viewer for simple 2D Simplex problems - one that will graph the feasible region and show the correct iteration of feasible points.

Finally, the WASM app works locally with 'trunk serve,' but not when I attempt to build it into WASM and run it locally or on GitHub Pages. In addition, I have not tested how well the website runs on mobile.
