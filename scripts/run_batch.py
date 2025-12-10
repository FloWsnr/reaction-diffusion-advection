#!/usr/bin/env python3
"""
Batch execution manager for Gray-Scott simulations.
Supports parallel execution and progress tracking.
"""

import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import time
from datetime import datetime
import signal
import sys


def run_case(case_dir: Path, timeout: int = 14400) -> dict:
    """
    Run a single OpenFOAM case.

    Args:
        case_dir: Path to the case directory
        timeout: Maximum runtime in seconds (default 4 hours)

    Returns:
        Dictionary with run results
    """
    start_time = time.time()
    result = {
        'case': str(case_dir.name),
        'path': str(case_dir),
        'success': False,
        'runtime': 0,
        'error': None
    }

    try:
        # Run the Allrun script
        process = subprocess.run(
            ['./Allrun'],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        result['success'] = process.returncode == 0
        result['return_code'] = process.returncode

        # Store last part of output for debugging
        if process.stdout:
            result['stdout_tail'] = process.stdout[-2000:]
        if process.stderr:
            result['stderr_tail'] = process.stderr[-1000:]

        if not result['success']:
            result['error'] = f"Return code: {process.returncode}"

    except subprocess.TimeoutExpired:
        result['error'] = f"Timeout exceeded ({timeout}s)"
    except FileNotFoundError:
        result['error'] = "Allrun script not found"
    except PermissionError:
        result['error'] = "Permission denied running Allrun"
    except Exception as e:
        result['error'] = str(e)

    result['runtime'] = time.time() - start_time
    return result


def find_case_directories(cases_dir: Path, start: int = 0, end: int = None) -> list:
    """Find all case directories in the given path."""
    case_dirs = sorted([
        d for d in cases_dir.iterdir()
        if d.is_dir() and d.name.startswith('case_')
    ])

    # Apply range
    if end is not None:
        case_dirs = case_dirs[start:end]
    else:
        case_dirs = case_dirs[start:]

    return case_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Run batch Gray-Scott simulations'
    )
    parser.add_argument(
        '--cases-dir', '-d',
        default='./cases',
        help='Directory containing case folders'
    )
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=4,
        help='Number of parallel processes'
    )
    parser.add_argument(
        '--start', '-s',
        type=int,
        default=0,
        help='Start index (0-based)'
    )
    parser.add_argument(
        '--end', '-e',
        type=int,
        default=None,
        help='End index (exclusive)'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=14400,
        help='Timeout per case in seconds (default: 4 hours)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List cases without running'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    cases_dir = Path(args.cases_dir)
    if not cases_dir.is_absolute():
        cases_dir = script_dir / cases_dir

    # Find case directories
    case_dirs = find_case_directories(cases_dir, args.start, args.end)

    if not case_dirs:
        print(f"No case directories found in {cases_dir}")
        return

    print(f"Found {len(case_dirs)} cases to run")

    if args.dry_run:
        print("\nCases to run:")
        for i, case_dir in enumerate(case_dirs):
            print(f"  {i + args.start}: {case_dir.name}")
        return

    print(f"Running with {args.parallel} parallel processes")
    print(f"Timeout per case: {args.timeout}s")
    print()

    # Track results
    results = []
    successful = 0
    failed = 0

    start_time = datetime.now()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nInterrupted! Saving partial results...")
        save_results(cases_dir, results, successful, failed, start_time, interrupted=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all jobs
        futures = {
            executor.submit(run_case, case_dir, args.timeout): case_dir
            for case_dir in case_dirs
        }

        # Process results as they complete
        for future in as_completed(futures):
            case_dir = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    'case': str(case_dir.name),
                    'path': str(case_dir),
                    'success': False,
                    'runtime': 0,
                    'error': f"Future exception: {str(e)}"
                }

            results.append(result)

            if result['success']:
                successful += 1
                status = "OK"
            else:
                failed += 1
                status = f"FAILED: {result.get('error', 'Unknown error')}"

            elapsed = datetime.now() - start_time
            progress = successful + failed
            eta = (elapsed / progress * (len(case_dirs) - progress)) if progress > 0 else "N/A"

            print(f"[{progress}/{len(case_dirs)}] {result['case']}: {status} "
                  f"({result['runtime']:.1f}s) ETA: {eta}")

    # Save results
    save_results(cases_dir, results, successful, failed, start_time)

    print(f"\nBatch complete: {successful} successful, {failed} failed")
    print(f"Total time: {datetime.now() - start_time}")


def save_results(cases_dir: Path, results: list, successful: int, failed: int,
                 start_time: datetime, interrupted: bool = False):
    """Save batch results summary."""
    summary = {
        'total': len(results),
        'successful': successful,
        'failed': failed,
        'interrupted': interrupted,
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'total_time': str(datetime.now() - start_time),
        'results': results
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = cases_dir / f"batch_results_{timestamp}.yaml"

    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"Results saved to {summary_file}")


if __name__ == '__main__':
    main()
