// Equation for species B:
// dB/dt + div(phi,B) = Db*laplacian(B) + C*A*B^2 - (f+k)*B
//
// Standard form:
// dB/dt + div(phi,B) - Db*laplacian(B) = C*A*B^2 - (f+k)*B

if (implicit)
{
    // Implicit treatment of source terms for stability
    // LHS: ddt(B) + div(phi,B) - laplacian(Db,B) + Sp(f+k,B)
    // RHS: C*A*B*B (production term, kept explicit for nonlinear term)

    fvScalarMatrix BEqn
    (
        fvm::ddt(B)
      + fvm::div(phi, B)
      - fvm::laplacian(Db, B)
      + fvm::Sp(f + k, B)    // -(f+k)*B moved to LHS as +Sp(f+k,B)
     ==
        C*A*B*B              // +C*A*B^2 on RHS (explicit)
    );

    BEqn.relax();
    BEqn.solve();
}
else
{
    // Explicit treatment of source terms
    volScalarField reactionB = C*A*B*B - (f + k)*B;

    fvScalarMatrix BEqn
    (
        fvm::ddt(B)
      + fvm::div(phi, B)
      - fvm::laplacian(Db, B)
     ==
        reactionB
    );

    BEqn.relax();
    BEqn.solve();
}

// Bound B between 0 and 1 (physical concentrations)
B.max(scalar(0));
B.min(scalar(1));
