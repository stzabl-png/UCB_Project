#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Affordance2Grasp pool-sim dependencies up to, but not including, smoke.
#
# Defaults match the current Vast setup:
#   /root/Affordance2Grasp   project repo
#   /root/isaac-sim          Isaac Sim standalone
#   /root/Project/curobo     cuRobo v0.7.7
#
# Usage:
#   bash scripts/setup_pool_server.sh
#   BRANCH=A6000 HF_MODE=full bash scripts/setup_pool_server.sh
#   HF_TOKEN=hf_xxx bash scripts/setup_pool_server.sh

ROOT_DIR="${ROOT_DIR:-/root}"
PROJ="${PROJ:-${ROOT_DIR}/Affordance2Grasp}"
REPO_URL="${REPO_URL:-https://github.com/stzabl-png/UCB_Project.git}"
BRANCH="${BRANCH:-titan}"

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-${ROOT_DIR}/isaac-sim}"
ISAAC_ZIP_URL="${ISAAC_ZIP_URL:-https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip}"
ISAAC_ZIP_PATH="${ISAAC_ZIP_PATH:-${ROOT_DIR}/isaac-sim-5.0.0.zip}"

CUROBO_DIR="${CUROBO_DIR:-${ROOT_DIR}/Project/curobo}"
CUROBO_REPO="${CUROBO_REPO:-https://github.com/NVlabs/curobo.git}"
CUROBO_TAG="${CUROBO_TAG:-v0.7.7}"

CONDA_ENV="${CONDA_ENV:-bundlesdf}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

HF_DATASET="${HF_DATASET:-UCBProject/hard_obj_grasp_collect_pipeline}"
# HF_MODE=minimal downloads only files needed for --no-auto-refill pool sim.
# HF_MODE=full downloads the whole dataset mirror.
# HF_MODE=skip skips HuggingFace download.
HF_MODE="${HF_MODE:-minimal}"

OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_ACCEPT_EULA

log() {
  printf '\n\033[1;34m==>\033[0m %s\n' "$*"
}

die() {
  printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_apt_install() {
  if [[ "${SKIP_APT:-0}" == "1" ]]; then
    log "Skipping apt install because SKIP_APT=1"
    return
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    die "apt install needs root. Re-run as root or set SKIP_APT=1 after installing deps manually."
  fi

  log "Installing system dependencies"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git git-lfs curl wget unzip ca-certificates \
    libxt6 libxmu6 libxi6 libxrender1 libxrandr2 libxinerama1 \
    libxcursor1 libxcomposite1 libxdamage1 libxfixes3 libxkbcommon-x11-0 \
    libglu1-mesa libgl1 libegl1
  git lfs install
}

ensure_conda_available() {
  if have_cmd conda; then
    return
  fi
  for conda_sh in \
    "${ROOT_DIR}/miniconda3/etc/profile.d/conda.sh" \
    "${ROOT_DIR}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "${conda_sh}" ]]; then
      # shellcheck disable=SC1090
      source "${conda_sh}"
      return
    fi
  done
  die "conda not found. Install Miniconda/Anaconda first, then re-run."
}

activate_conda_env() {
  ensure_conda_available
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
}

clone_or_update_project() {
  log "Preparing project repo at ${PROJ}"
  if [[ ! -d "${PROJ}/.git" ]]; then
    mkdir -p "$(dirname "${PROJ}")"
    git clone "${REPO_URL}" "${PROJ}"
  fi

  git -C "${PROJ}" fetch --all --tags
  git -C "${PROJ}" checkout "${BRANCH}"
  git -C "${PROJ}" pull --ff-only || {
    log "git pull --ff-only failed; continuing with current checkout"
  }
}

install_conda_env() {
  log "Preparing conda env ${CONDA_ENV}"
  ensure_conda_available
  eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  fi
  conda activate "${CONDA_ENV}"
  cd "${PROJ}"
  python -m pip install -U pip wheel
  if [[ -f requirements.txt ]]; then
    python -m pip install -r requirements.txt
  fi
  python -m pip install rtree huggingface-hub h5py
}

download_hf_data() {
  if [[ "${HF_MODE}" == "skip" ]]; then
    log "Skipping HuggingFace download because HF_MODE=skip"
    return
  fi

  log "Downloading HuggingFace dataset (${HF_MODE})"
  activate_conda_env
  cd "${PROJ}"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential || true
  elif ! hf auth whoami >/dev/null 2>&1; then
    log "No HuggingFace login detected; starting interactive hf auth login"
    hf auth login
  fi

  case "${HF_MODE}" in
    minimal)
      hf download "${HF_DATASET}" sim --repo-type dataset --local-dir "${PROJ}"
      hf download "${HF_DATASET}" output/obj_usd --repo-type dataset --local-dir "${PROJ}"
      hf download "${HF_DATASET}" output/grasp_collect_no_rot --repo-type dataset --local-dir "${PROJ}"
      ;;
    full)
      hf download "${HF_DATASET}" --repo-type dataset --local-dir "${PROJ}"
      ;;
    *)
      die "HF_MODE must be one of: minimal, full, skip"
      ;;
  esac

  rm -f "${PROJ}/output/grasp_collect_no_rot/sim_pool_registry.json"
}

install_isaac_sim() {
  log "Preparing Isaac Sim at ${ISAAC_SIM_PATH}"
  if [[ -x "${ISAAC_SIM_PATH}/python.sh" ]]; then
    log "Isaac Sim already present: ${ISAAC_SIM_PATH}/python.sh"
  else
    mkdir -p "${ISAAC_SIM_PATH}"
    if [[ ! -f "${ISAAC_ZIP_PATH}" ]]; then
      log "Downloading Isaac Sim 5.0 standalone zip"
      wget -c -O "${ISAAC_ZIP_PATH}" "${ISAAC_ZIP_URL}"
    fi
    log "Extracting Isaac Sim zip"
    unzip -q "${ISAAC_ZIP_PATH}" -d "${ISAAC_SIM_PATH}"
  fi

  [[ -x "${ISAAC_SIM_PATH}/python.sh" ]] || die "Isaac python.sh not found at ${ISAAC_SIM_PATH}/python.sh"

  log "Installing Isaac Python runtime dependencies"
  "${ISAAC_SIM_PATH}/python.sh" -m pip install -U pip wheel ninja tomli
  "${ISAAC_SIM_PATH}/python.sh" -m pip install h5py termcolor scipy
}

install_curobo() {
  log "Preparing cuRobo ${CUROBO_TAG} at ${CUROBO_DIR}"
  mkdir -p "$(dirname "${CUROBO_DIR}")"
  if [[ ! -d "${CUROBO_DIR}/.git" ]]; then
    git clone "${CUROBO_REPO}" "${CUROBO_DIR}"
  fi

  git -C "${CUROBO_DIR}" fetch --all --tags
  git -C "${CUROBO_DIR}" checkout "${CUROBO_TAG}"

  "${ISAAC_SIM_PATH}/python.sh" -m pip uninstall -y nvidia-curobo curobo 2>/dev/null || true
  cd "${CUROBO_DIR}"
  "${ISAAC_SIM_PATH}/python.sh" -m pip install -e . --no-build-isolation
}

verify_setup() {
  log "Running pre-smoke verification"
  activate_conda_env
  cd "${PROJ}"

  test -f sim/assets_franka/franka.usd && echo "OK franka"
  test -f sim/assets_scene/Collected_default_environment/default_environment.usd && echo "OK scene"
  ls output/obj_usd/oakink/*.usd >/dev/null 2>&1 && echo "OK object USD"
  echo "pool HDF5 count: $(ls output/grasp_collect_no_rot/candidates/pool/*_grasp.hdf5 2>/dev/null | wc -l)"
  echo "merged count: $(ls output/grasp_collect_no_rot/merged/*_merged.hdf5 2>/dev/null | wc -l)"
  test -f "${ISAAC_SIM_PATH}/python.sh" && echo "OK isaac"

  "${ISAAC_SIM_PATH}/python.sh" -c "
import h5py, termcolor, scipy
import curobo
from curobo.wrap.reacher.motion_gen import MotionGen
from curobo.types.math import Pose
print('OK Isaac Python deps + cuRobo', getattr(curobo, '__version__', 'unknown'))
"
}

print_next_steps() {
  cat <<EOF

Setup finished up to pre-smoke.

Before running smoke in a new shell:

  cd ${PROJ}
  conda activate ${CONDA_ENV}
  export ISAAC_SIM_PATH=${ISAAC_SIM_PATH}
  export OMNI_KIT_ACCEPT_EULA=YES

Recommended first smoke:

  python3 scripts/batch_sim_candidates_pool.py \\
    --outdir output/grasp_collect_smoke_gpu \\
    --pool-dir output/grasp_collect_no_rot/candidates/pool \\
    --merged-dir output/grasp_collect_no_rot/merged \\
    --max-rounds 1 \\
    --slots-per-round 2 \\
    --sim-gpu-ids 0 \\
    --sim-per-gpu 1 \\
    --same-gpu-stagger-s 0 \\
    --isaac-startup-slots-per-gpu 1 \\
    --sim-timeout 3600 \\
    --headless \\
    --no-auto-refill

EOF
}

main() {
  run_apt_install
  clone_or_update_project
  install_conda_env
  download_hf_data
  install_isaac_sim
  install_curobo
  verify_setup
  print_next_steps
}

main "$@"
