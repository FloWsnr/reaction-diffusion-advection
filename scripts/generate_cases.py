#!/usr/bin/env python3
"""
Generate OpenFOAM case directories for Gray-Scott parameter sweep.

This script creates multiple simulation cases with different parameter
combinations for systematic exploration of the Gray-Scott parameter space.
"""

import os
import shutil
import yaml
import numpy as np
import itertools
from pathlib import Path
from typing import Dict, List, Any
import argparse


def load_parameters(config_path: str) -> Dict:
    """Load parameter configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_parameter_combinations(params: Dict) -> List[Dict]:
    """Generate all parameter combinations for sweep."""

    # Get individual parameter values
    Da_values = params['parameters']['Da']['values']
    Db_values = params['parameters']['Db']['values']

    # Generate f values (linspace or explicit)
    f_params = params['parameters']['f']
    if 'values' in f_params:
        f_values = f_params['values']
    else:
        f_values = np.linspace(
            f_params['min'],
            f_params['max'],
            f_params['num_samples']
        ).tolist()

    # Generate k values
    k_params = params['parameters']['k']
    if 'values' in k_params:
        k_values = k_params['values']
    else:
        k_values = np.linspace(
            k_params['min'],
            k_params['max'],
            k_params['num_samples']
        ).tolist()

    # Get C values (reaction rate constant)
    C_values = params['parameters'].get('C', {}).get('values', [1.0])

    U_mag_values = params['parameters']['U_mag']['values']
    U_angle_values = params['parameters']['U_angle']['values']

    # Generate all combinations
    combinations = []
    for Da, Db, f, k, C, U_mag, U_angle in itertools.product(
        Da_values, Db_values, f_values, k_values, C_values, U_mag_values, U_angle_values
    ):
        # Convert angle to velocity components
        U_angle_rad = np.radians(U_angle)
        Ux = U_mag * np.cos(U_angle_rad)
        Uy = U_mag * np.sin(U_angle_rad)

        combinations.append({
            'Da': Da,
            'Db': Db,
            'f': f,
            'k': k,
            'C': C,
            'Ux': Ux,
            'Uy': Uy,
            'U_mag': U_mag,
            'U_angle': U_angle
        })

    return combinations


def create_case_name(params: Dict, index: int) -> str:
    """Create unique case name from parameters."""
    return f"case_{index:05d}_Da{params['Da']:.2e}_Db{params['Db']:.2e}_f{params['f']:.4f}_k{params['k']:.4f}_U{params['U_mag']:.3f}"


def modify_transport_properties(case_dir: Path, params: Dict):
    """Modify transportProperties file with new parameters."""
    filepath = case_dir / "constant" / "transportProperties"

    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Da          Da [ 0 2 -1 0 0 0 0 ] {params['Da']:.6e};
Db          Db [ 0 2 -1 0 0 0 0 ] {params['Db']:.6e};
f           f  [ 0 0 -1 0 0 0 0 ] {params['f']:.6f};
k           k  [ 0 0 -1 0 0 0 0 ] {params['k']:.6f};
C           {params.get('C', 1.0):.6f};

implicit    true;

// ************************************************************************* //
"""

    with open(filepath, 'w') as f:
        f.write(content)


def modify_velocity_field(case_dir: Path, params: Dict):
    """Modify velocity field U with new advection velocity."""
    filepath = case_dir / "0.org" / "U"

    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({params['Ux']:.8f} {params['Uy']:.8f} 0);

boundaryField
{{
    left
    {{
        type            cyclic;
    }}
    right
    {{
        type            cyclic;
    }}
    bottom
    {{
        type            cyclic;
    }}
    top
    {{
        type            cyclic;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}

// ************************************************************************* //
"""

    with open(filepath, 'w') as f:
        f.write(content)


def generate_random_initial_conditions(case_dir: Path, config: Dict, seed: int):
    """Generate random initial perturbations using setFieldsDict."""
    np.random.seed(seed)

    ic_config = config['initial_conditions']
    domain_size = config['domain']['size']

    # Generate random patch locations
    num_patches = ic_config.get('num_patches', 5)
    patch_size = ic_config.get('patch_size', 0.2)

    regions = []
    for _ in range(num_patches):
        x_center = np.random.uniform(patch_size, domain_size - patch_size)
        y_center = np.random.uniform(patch_size, domain_size - patch_size)

        x_min = x_center - patch_size / 2
        x_max = x_center + patch_size / 2
        y_min = y_center - patch_size / 2
        y_max = y_center + patch_size / 2

        # Add some noise to perturbation values
        A_val = ic_config['A_perturbation'] + np.random.uniform(-0.1, 0.1)
        B_val = ic_config['B_perturbation'] + np.random.uniform(-0.05, 0.05)

        # Clamp values to valid range
        A_val = max(0.0, min(1.0, A_val))
        B_val = max(0.0, min(1.0, B_val))

        regions.append(f"""    boxToCell
    {{
        box ({x_min:.4f} {y_min:.4f} -1) ({x_max:.4f} {y_max:.4f} 1);
        fieldValues
        (
            volScalarFieldValue A {A_val:.4f}
            volScalarFieldValue B {B_val:.4f}
        );
    }}""")

    regions_str = "\n\n".join(regions)

    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue A 1
    volScalarFieldValue B 0
);

regions
(
{regions_str}
);

// ************************************************************************* //
"""

    filepath = case_dir / "system" / "setFieldsDict"
    with open(filepath, 'w') as f:
        f.write(content)


def save_case_metadata(case_dir: Path, params: Dict, index: int, seed: int):
    """Save case metadata for later reference."""
    metadata = {
        'index': index,
        'seed': seed,
        'parameters': params
    }

    filepath = case_dir / "case_metadata.yaml"
    with open(filepath, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


def create_run_script(case_dir: Path):
    """Create run script for the case."""
    content = """#!/bin/bash
# Run script for Gray-Scott advection case

# Source OpenFOAM environment (adjust path if needed)
if [ -f /usr/lib/openfoam/openfoam2412/etc/bashrc ]; then
    source /usr/lib/openfoam/openfoam2412/etc/bashrc
elif [ -f /opt/openfoam/openfoam2412/etc/bashrc ]; then
    source /opt/openfoam/openfoam2412/etc/bashrc
elif [ -n "$WM_PROJECT_DIR" ]; then
    # OpenFOAM already sourced
    :
else
    echo "Error: OpenFOAM environment not found"
    exit 1
fi

# Clean and prepare
rm -rf 0
cp -r 0.org 0

# Generate mesh
blockMesh > log.blockMesh 2>&1

# Set initial conditions
setFields > log.setFields 2>&1

# Run solver
grayScottAdvectionFoam > log.grayScottAdvectionFoam 2>&1

echo "Simulation complete"
"""

    filepath = case_dir / "Allrun"
    with open(filepath, 'w') as f:
        f.write(content)
    os.chmod(filepath, 0o755)


def create_clean_script(case_dir: Path):
    """Create clean script for the case."""
    content = """#!/bin/bash
# Clean script for Gray-Scott advection case

# Remove time directories (except 0.org)
foamListTimes -rm 2>/dev/null || rm -rf [0-9]* 2>/dev/null

# Remove generated mesh
rm -rf constant/polyMesh

# Remove logs
rm -f log.*

# Remove processor directories (if parallel)
rm -rf processor*

# Remove postProcessing
rm -rf postProcessing

# Remove VTK output
rm -rf VTK

echo "Case cleaned"
"""

    filepath = case_dir / "Allclean"
    with open(filepath, 'w') as f:
        f.write(content)
    os.chmod(filepath, 0o755)


def main():
    parser = argparse.ArgumentParser(
        description='Generate OpenFOAM cases for Gray-Scott parameter sweep'
    )
    parser.add_argument(
        '--config', '-c',
        default='parameters.yaml',
        help='Path to parameter configuration file'
    )
    parser.add_argument(
        '--base-case', '-b',
        default='../tutorials/grayScottAdvection/baseCase',
        help='Path to base case template'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./cases',
        help='Output directory for generated cases'
    )
    parser.add_argument(
        '--max-cases', '-m',
        type=int,
        default=None,
        help='Maximum number of cases to generate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print number of cases without generating'
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.resolve()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    base_case = Path(args.base_case)
    if not base_case.is_absolute():
        base_case = script_dir / base_case

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    # Load configuration
    config = load_parameters(config_path)

    # Generate parameter combinations
    combinations = generate_parameter_combinations(config)

    print(f"Total parameter combinations: {len(combinations)}")

    if args.dry_run:
        print("Dry run - no cases generated")
        return

    if args.max_cases:
        combinations = combinations[:args.max_cases]
        print(f"Limiting to {len(combinations)} cases")

    print(f"Generating {len(combinations)} cases...")

    # Check base case exists
    if not base_case.exists():
        print(f"Error: Base case not found at {base_case}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate cases
    for i, params in enumerate(combinations):
        case_name = create_case_name(params, i)
        case_dir = output_dir / case_name

        # Copy base case
        if case_dir.exists():
            shutil.rmtree(case_dir)
        shutil.copytree(base_case, case_dir)

        # Modify files
        modify_transport_properties(case_dir, params)
        modify_velocity_field(case_dir, params)

        # Generate random initial conditions with unique seed per case
        case_seed = args.seed + i
        generate_random_initial_conditions(case_dir, config, case_seed)

        # Save metadata
        save_case_metadata(case_dir, params, i, case_seed)

        # Create run and clean scripts
        create_run_script(case_dir)
        create_clean_script(case_dir)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{len(combinations)} cases")

    # Save master index
    index_data = {
        'num_cases': len(combinations),
        'config_file': str(config_path),
        'base_case': str(base_case),
        'seed': args.seed,
        'cases': [
            {
                'index': i,
                'name': create_case_name(p, i),
                'parameters': p
            }
            for i, p in enumerate(combinations)
        ]
    }

    index_file = output_dir / "case_index.yaml"
    with open(index_file, 'w') as f:
        yaml.dump(index_data, f, default_flow_style=False)

    print(f"\nGenerated {len(combinations)} cases in {output_dir}")
    print(f"Case index saved to {index_file}")


if __name__ == '__main__':
    main()
