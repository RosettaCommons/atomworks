set -e  # Exit on error

echo "Running from $PWD"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
: "${ATOMWORKS_CI_CONTAINER:?Set the AtomWorks CI container path}"
: "${ATOMWORKS_CI_TEST_PACK:?Set the AtomWorks CI test-pack path}"

# Give the container a job-local /tmp instead of the compute node's.
ci_tmp=$(mktemp -d "$PWD/.ci-tmp.XXXXXX")
trap 'rm -rf "$ci_tmp"' EXIT

run_in_container() {
    apptainer exec \
        --bind "$PWD:/workspace" \
        --bind "$ci_tmp:/tmp" \
        --pwd /workspace \
        --env PYTHONPATH=/workspace/src \
        --env TMPDIR=/tmp \
        "$ATOMWORKS_CI_CONTAINER" "$@"
}

# Extract the shared test pack (non-PDB fixtures). The PDB mirror itself is
# resolved by tests/conftest.py, which prefers the lab-wide frozen PDB copy.
mkdir -p tests/data
tar -xf "$ATOMWORKS_CI_TEST_PACK" -C tests/data

# Get max processes from environment variable, default to 24 if not set
N_CPU=${N_CPU:-24}
echo "Running tests with max $N_CPU CPUs"

# Ensure we can collect all tests (i.e. imports succeed)
echo "Testing imports by trying to collect all tests"
run_in_container pytest -m "not benchmark" --collect-only tests/

# Run the tests in coverage mode (with 24 CPUs)
run_in_container pytest -m "not benchmark" --cov=atomworks --cov-report=xml -n=auto --maxprocesses=$N_CPU --dist=worksteal tests/

# Require at least 80% coverage
run_in_container coverage report --fail-under=80

# Output the coverage in a format GitLab can parse
run_in_container coverage report | tail -n 1 | awk '{print "TOTAL", $NF}'
