#!/usr/bin/env python3
"""
Collect simulation data for ML dataset creation.
Parses OpenFOAM output and saves as NumPy arrays.
"""

import os
import re
import struct
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_openfoam_header(content: str) -> Dict:
    """Parse OpenFOAM file header to extract metadata."""
    header = {}

    # Extract class
    class_match = re.search(r'class\s+(\w+);', content)
    if class_match:
        header['class'] = class_match.group(1)

    # Extract object name
    object_match = re.search(r'object\s+(\w+);', content)
    if object_match:
        header['object'] = object_match.group(1)

    # Check format
    header['binary'] = 'format      binary' in content or 'format\tbinary' in content

    return header


def parse_ascii_scalar_field(content: str) -> Optional[np.ndarray]:
    """Parse ASCII format OpenFOAM scalar field."""
    # Find internalField section
    if 'internalField' not in content:
        return None

    # Check for nonuniform list
    match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', content)
    if match:
        num_values = int(match.group(1))
        # Find the start of values after '('
        start_idx = match.end()
        end_idx = content.find(')', start_idx)

        if end_idx == -1:
            return None

        values_str = content[start_idx:end_idx]
        # Parse values (handle both space and newline separated)
        values = []
        for line in values_str.strip().split('\n'):
            for val in line.strip().split():
                try:
                    values.append(float(val))
                except ValueError:
                    continue

        if len(values) == num_values:
            return np.array(values)

    # Check for uniform value
    match = re.search(r'internalField\s+uniform\s+([\d.eE+-]+)', content)
    if match:
        return float(match.group(1))

    return None


def parse_binary_scalar_field(filepath: Path, num_cells: int) -> Optional[np.ndarray]:
    """Parse binary format OpenFOAM scalar field."""
    with open(filepath, 'rb') as f:
        content = f.read()

    # Find header end (after FoamFile block)
    header_end = content.find(b'internalField')
    if header_end == -1:
        return None

    # Find the '(' that starts the binary data
    paren_start = content.find(b'(', header_end)
    if paren_start == -1:
        return None

    # The binary data starts right after '('
    data_start = paren_start + 1

    # Skip any whitespace/newlines
    while data_start < len(content) and content[data_start:data_start+1] in [b'\n', b' ', b'\t']:
        data_start += 1

    # Read the doubles (8 bytes each)
    try:
        values = struct.unpack(f'{num_cells}d', content[data_start:data_start + num_cells * 8])
        return np.array(values)
    except struct.error:
        return None


def parse_openfoam_field(filepath: Path, num_cells: int = None) -> Optional[np.ndarray]:
    """
    Parse OpenFOAM field file (binary or ASCII).

    Args:
        filepath: Path to the field file
        num_cells: Number of cells (required for binary parsing)

    Returns:
        numpy array of field values, or None if parsing failed
    """
    with open(filepath, 'rb') as f:
        raw_content = f.read()

    # Try to decode as text first
    try:
        content = raw_content.decode('utf-8')
        header = parse_openfoam_header(content)

        if header.get('binary', False):
            if num_cells is None:
                # Try to extract from the file
                match = re.search(r'nonuniform\s+List<scalar>\s*\n(\d+)', content)
                if match:
                    num_cells = int(match.group(1))
                else:
                    return None
            return parse_binary_scalar_field(filepath, num_cells)
        else:
            return parse_ascii_scalar_field(content)

    except UnicodeDecodeError:
        # File is binary, try binary parsing
        if num_cells is None:
            return None
        return parse_binary_scalar_field(filepath, num_cells)


def get_mesh_info(case_dir: Path) -> Dict:
    """Extract mesh information from blockMeshDict or polyMesh."""
    blockmesh_path = case_dir / "system" / "blockMeshDict"

    if blockmesh_path.exists():
        with open(blockmesh_path, 'r') as f:
            content = f.read()

        # Look for nx, ny, nz definitions
        nx_match = re.search(r'nx\s+(\d+)', content)
        ny_match = re.search(r'ny\s+(\d+)', content)

        if nx_match and ny_match:
            nx = int(nx_match.group(1))
            ny = int(ny_match.group(1))
            return {'nx': nx, 'ny': ny, 'num_cells': nx * ny}

        # Look for block definition hex (...) (nx ny nz)
        block_match = re.search(r'hex\s*\([^)]+\)\s*\((\d+)\s+(\d+)\s+(\d+)\)', content)
        if block_match:
            nx = int(block_match.group(1))
            ny = int(block_match.group(2))
            return {'nx': nx, 'ny': ny, 'num_cells': nx * ny}

    # Default values
    return {'nx': 128, 'ny': 128, 'num_cells': 16384}


def get_available_times(case_dir: Path) -> List[str]:
    """Get list of available time directories."""
    times = []
    for item in case_dir.iterdir():
        if item.is_dir():
            try:
                # Check if directory name is a valid number
                float(item.name)
                times.append(item.name)
            except ValueError:
                continue

    # Sort numerically
    return sorted(times, key=float)


def collect_case_data(case_dir: Path, time_dirs: List[str] = None,
                      fields: List[str] = ['A', 'B']) -> Dict:
    """
    Collect all data from a single case.

    Args:
        case_dir: Path to the case directory
        time_dirs: List of time directories to process (None = all)
        fields: List of field names to extract

    Returns:
        Dictionary with collected data
    """
    case_dir = Path(case_dir)

    # Load metadata
    metadata_path = case_dir / "case_metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
    else:
        metadata = {'parameters': {}}

    # Get mesh info
    mesh_info = get_mesh_info(case_dir)

    # Find available time directories
    available_times = get_available_times(case_dir)

    if not available_times:
        return {
            'metadata': metadata,
            'mesh': mesh_info,
            'times': [],
            'fields': {},
            'error': 'No time directories found'
        }

    # Filter time directories if specified
    if time_dirs is not None:
        time_dirs = [t for t in time_dirs if t in available_times]
    else:
        time_dirs = available_times

    # Collect field data
    data = {
        'metadata': metadata,
        'mesh': mesh_info,
        'times': [],
        'fields': {field: [] for field in fields}
    }

    for time_dir in time_dirs:
        time_path = case_dir / time_dir

        # Try to read all requested fields for this time
        time_data = {}
        all_fields_ok = True

        for field in fields:
            field_path = time_path / field
            if field_path.exists():
                field_data = parse_openfoam_field(field_path, mesh_info['num_cells'])
                if field_data is not None and isinstance(field_data, np.ndarray):
                    time_data[field] = field_data
                else:
                    all_fields_ok = False
                    break
            else:
                all_fields_ok = False
                break

        # Only add this timestep if all fields were successfully read
        if all_fields_ok and time_data:
            data['times'].append(float(time_dir))
            for field in fields:
                # Reshape to 2D
                field_2d = time_data[field].reshape(mesh_info['ny'], mesh_info['nx'])
                data['fields'][field].append(field_2d)

    # Convert lists to numpy arrays
    if data['times']:
        data['times'] = np.array(data['times'])
        for field in fields:
            data['fields'][field] = np.stack(data['fields'][field], axis=0)

    return data


def process_case(args) -> Tuple[int, Dict]:
    """Wrapper for parallel processing."""
    case_dir, time_dirs, fields = args
    try:
        data = collect_case_data(case_dir, time_dirs, fields)
        return (0, data)
    except Exception as e:
        return (1, {'error': str(e), 'case': str(case_dir)})


def save_dataset(output_path: Path, cases_data: List[Dict],
                 fields: List[str] = ['A', 'B']):
    """Save collected data to NumPy format."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all samples
    all_data = {field: [] for field in fields}
    all_times = []
    all_params = []

    for case_data in cases_data:
        if 'error' in case_data:
            continue

        if not case_data.get('times', []) is not None and len(case_data.get('times', [])) == 0:
            continue

        params = case_data['metadata'].get('parameters', {})

        for t_idx, time in enumerate(case_data['times']):
            all_times.append(time)

            for field in fields:
                all_data[field].append(case_data['fields'][field][t_idx])

            all_params.append({
                'case_index': case_data['metadata'].get('index', -1),
                'time': float(time),
                **{k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in params.items()}
            })

    if not all_times:
        print("No data collected!")
        return

    # Save arrays
    for field in fields:
        arr = np.stack(all_data[field])
        np.save(output_path / f'{field}.npy', arr)
        print(f"  Saved {field}.npy: shape {arr.shape}")

    np.save(output_path / 'times.npy', np.array(all_times))

    # Save parameters
    with open(output_path / 'parameters.yaml', 'w') as f:
        yaml.dump(all_params, f, default_flow_style=False)

    # Save dataset info
    info = {
        'num_samples': len(all_times),
        'fields': fields,
        'shape': {field: list(all_data[field][0].shape) for field in fields},
        'num_cases': len([c for c in cases_data if 'error' not in c]),
    }
    with open(output_path / 'dataset_info.yaml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False)

    print(f"\nSaved {len(all_times)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect Gray-Scott simulation data for ML'
    )
    parser.add_argument(
        '--cases-dir', '-d',
        default='./cases',
        help='Directory containing case folders'
    )
    parser.add_argument(
        '--output', '-o',
        default='./dataset',
        help='Output directory for collected data'
    )
    parser.add_argument(
        '--fields', '-f',
        nargs='+',
        default=['A', 'B'],
        help='Fields to extract (default: A B)'
    )
    parser.add_argument(
        '--times',
        nargs='+',
        default=None,
        help='Specific times to sample (default: all)'
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=None,
        help='Maximum number of cases to process'
    )
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=4,
        help='Number of parallel processes'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.resolve()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.is_absolute():
        cases_dir = script_dir / cases_dir

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    # Find case directories
    case_dirs = sorted([
        d for d in cases_dir.iterdir()
        if d.is_dir() and d.name.startswith('case_')
    ])

    if not case_dirs:
        print(f"No case directories found in {cases_dir}")
        return

    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]

    print(f"Collecting data from {len(case_dirs)} cases...")
    print(f"Fields: {args.fields}")
    if args.times:
        print(f"Times: {args.times}")

    # Process cases in parallel
    cases_data = []
    process_args = [(d, args.times, args.fields) for d in case_dirs]

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process_case, arg): arg[0] for arg in process_args}

        for i, future in enumerate(as_completed(futures)):
            case_dir = futures[future]
            status, data = future.result()

            if status == 0 and 'error' not in data:
                num_times = len(data.get('times', []))
                cases_data.append(data)
                print(f"  [{i+1}/{len(case_dirs)}] {case_dir.name}: {num_times} timesteps")
            else:
                error = data.get('error', 'Unknown error')
                print(f"  [{i+1}/{len(case_dirs)}] {case_dir.name}: FAILED - {error}")

    # Save dataset
    print(f"\nSaving dataset to {output_dir}...")
    save_dataset(output_dir, cases_data, args.fields)


if __name__ == '__main__':
    main()
