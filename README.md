stltfdrk4.py: single transmission line transient finite difference runge kutta 4; use python, numpy and scipy.
stltfdrk4.cpp: single transmission line transient finite difference runge kutta 4; use C++ with MFEM library.

cst.cpp: Create Single Transmission Line.
stlt.cpp: Single Transmission Line Transient.call cstl to create a single transmission line mesh.
call stlt to run the telegraph equation model.

stltfe2d_submesh.cpp: single transmission line transient finite element 2D using submesh. Done in serial mode.

stltfe2d_02.cpp: try with bounadry integrator ...

stltfe1d_03.cpp: return to 1D approach with time stepping, forward euler. stable for little over 300ns.

stltfe1d_04.cpp: return to 1D approach with time stepping, backward euler.

