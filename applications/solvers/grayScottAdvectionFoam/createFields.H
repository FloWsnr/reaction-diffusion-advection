// Read transport properties
Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

// Diffusion coefficients
dimensionedScalar Da("Da", dimArea/dimTime, transportProperties);
dimensionedScalar Db("Db", dimArea/dimTime, transportProperties);

// Reaction parameters
dimensionedScalar f("f", dimless/dimTime, transportProperties);
dimensionedScalar k("k", dimless/dimTime, transportProperties);

// Reaction coupling coefficient (for dimensional consistency)
// This is the rate constant for the A + 2B -> 3B reaction
// Default value is 1.0 s^-1
dimensionedScalar C
(
    "C",
    dimless/dimTime,
    transportProperties.getOrDefault<scalar>("C", 1.0)
);

// Read implicit flag for source term treatment
const bool implicit = transportProperties.getOrDefault<bool>("implicit", true);

Info<< "Parameters:" << nl
    << "    Da = " << Da.value() << " m^2/s" << nl
    << "    Db = " << Db.value() << " m^2/s" << nl
    << "    f  = " << f.value() << " 1/s" << nl
    << "    k  = " << k.value() << " 1/s" << nl
    << "    C  = " << C.value() << " 1/s" << nl
    << "    implicit = " << implicit << nl << endl;

// Species concentration fields
Info<< "Reading field A\n" << endl;
volScalarField A
(
    IOobject
    (
        "A",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field B\n" << endl;
volScalarField B
(
    IOobject
    (
        "B",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

// Velocity field for advection
Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

// Create flux field from velocity
#include "createPhi.H"
