// Equation for species A:
// dA/dt + div(phi,A) = Da*laplacian(A) - C*A*B^2 + f*(1-A)
//
// Standard form:
// dA/dt + div(phi,A) - Da*laplacian(A) = -C*A*B^2 + f*(1-A)
//                                       = -C*A*B^2 + f - f*A

if (implicit)
{
    // Implicit treatment of source terms for stability
    // LHS: ddt(A) + div(phi,A) - laplacian(Da,A) + Sp(C*B*B,A) + Sp(f,A)
    // RHS: f (as a field)

    // Create source field: f is uniform
    volScalarField sourceA
    (
        IOobject
        (
            "sourceA",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        f
    );

    fvScalarMatrix AEqn
    (
        fvm::ddt(A)
      + fvm::div(phi, A)
      - fvm::laplacian(Da, A)
      + fvm::Sp(C*B*B, A)  // -C*A*B^2 moved to LHS as +Sp(C*B*B,A)
      + fvm::Sp(f, A)      // -f*A moved to LHS as +Sp(f,A)
     ==
        sourceA            // +f on RHS
    );

    AEqn.relax();
    AEqn.solve();
}
else
{
    // Explicit treatment of source terms
    volScalarField reactionA = f*(scalar(1) - A) - C*A*B*B;

    fvScalarMatrix AEqn
    (
        fvm::ddt(A)
      + fvm::div(phi, A)
      - fvm::laplacian(Da, A)
     ==
        reactionA
    );

    AEqn.relax();
    AEqn.solve();
}

// Bound A between 0 and 1 (physical concentrations)
A.max(scalar(0));
A.min(scalar(1));
