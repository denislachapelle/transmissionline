#include "mfem.hpp"
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

int main()
{
    // Define mesh parameters
    int dim = 1; // 1D transmission line
    double length = 1.0; // Length of the line
    int n_elements = 100; // Number of elements

    // Create 1D mesh
    Mesh mesh(n_elements, 1, Element::Edge, false, length);

    // Define finite element spaces for V (voltage) and I (current)
    int order = 1; // Finite element order
    H1_FECollection fec_V(order, dim);      // Voltage space (continuous)
    DG_FECollection fec_I(order - 1, dim);  // Current space (discontinuous)

    FiniteElementSpace fes_V(&mesh, &fec_V);
    FiniteElementSpace fes_I(&mesh, &fec_I);

    // Ensure V and I have the same number of degrees of freedom
    Array<int> dof_map;
    fes_V.GetEssentialTrueDofs(dof_map);
    fes_I.GetEssentialTrueDofs(dof_map);

    // Print mesh information
    mesh.PrintInfo();
    cout << "Number of degrees of freedom for V: " << fes_V.GetNDofs() << endl;
    cout << "Number of degrees of freedom for I: " << fes_I.GetNDofs() << endl;

    // Set up boundary conditions
    // Assume voltage V is set at the left boundary (source)
    Vector essential_V(fes_V.GetNDofs());
    essential_V = 0.0;

    // Apply a time-dependent voltage source at the left boundary
    Array<int> ess_bdr;
    mesh.GetBoundaryElements(ess_bdr);
    fes_V.GetEssentialTrueDofs(ess_bdr);

    // Define the voltage source as a sinusoidal function of time
    auto voltage_source = [](double t) {
        return 1.0 * sin(2 * M_PI * 1.0 * t); // Sinusoidal voltage with frequency 1 Hz
    };

    // Initialize solutions for V and I (voltage and current)
    GridFunction V(&fes_V);
    GridFunction I(&fes_I);

    V = 0.0;
    I = 0.0;

    // Set up the mass and stiffness matrices
    // Constants for the telegrapher’s equation
    double L = 1.0e-6;  // Inductance per unit length
    double C = 1.0e-12; // Capacitance per unit length
    double R = 1.0e-3;  // Resistance per unit length
    double G = 1.0e-6;  // Conductance per unit length

    BilinearForm M_V(&fes_V);
    BilinearForm M_I(&fes_I);
    BilinearForm K_V(&fes_V);
    BilinearForm K_I(&fes_I);

    // Define the mass and stiffness integrators
    ConstantCoefficient rho_V(C);  // Capacitance for voltage
    ConstantCoefficient rho_I(L);  // Inductance for current

    M_V.AddDomainIntegrator(new MassIntegrator(rho_V));
    M_I.AddDomainIntegrator(new MassIntegrator(rho_I));

    ConstantCoefficient R_coeff(R); // Resistance
    ConstantCoefficient G_coeff(G); // Conductance

    K_V.AddDomainIntegrator(new MassIntegrator(R_coeff));
    K_I.AddDomainIntegrator(new MassIntegrator(G_coeff));

    M_V.Assemble();
    M_I.Assemble();
    K_V.Assemble();
    K_I.Assemble();

    M_V.Finalize();
    M_I.Finalize();
    K_V.Finalize();
    K_I.Finalize();

    // Set up time-stepping parameters
    double t = 0.0;       // Initial time
    double t_final = 1.0; // Final time
    double dt = 0.001;    // Time step size
    int num_steps = static_cast<int>(t_final / dt);

    // Time-stepping loop (Euler Forward method)
    for (int step = 0; step < num_steps; step++)
    {
        // Update the voltage source at each time step
        double source_voltage = voltage_source(t);

        // Apply the source voltage at the left boundary (boundary condition)
        V(0) = source_voltage; // Apply the voltage source to the left boundary

        // Solve the system: M_V * dV/dt = -K_V * I
        Vector rhs_V(fes_V.GetNDofs());
        M_V.Mult(V, rhs_V); // Compute the mass matrix * V
        K_V.Mult(I, rhs_V); // Subtract the stiffness matrix * I

        // Update voltage and current solutions
        V.Add(-dt, rhs_V);
        I.Add(-dt, rhs_V); // Updating current with the same equation (simplification)

        // Output the solution at each step
        if (step % 100 == 0)
        {
            cout << "Time step: " << step << " | Voltage V[0]: " << V(0) << " | Current I[0]: " << I(0) << endl;
        }

        // Increment time
        t += dt;
    }

    return 0;
}
