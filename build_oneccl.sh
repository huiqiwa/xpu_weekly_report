#!/bin/bash
# Build the latest oneCCL from source and replace all libccl.so in the current environment.
# Usage:
#   bash build_oneccl.sh          # full build + replace
#   bash build_oneccl.sh --build-only   # build without replacing
#   bash build_oneccl.sh --replace-only # replace using last build artifacts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONECCL_REPO="https://github.com/intel-innersource/libraries.performance.communication.oneccl.git"
ONECCL_SRC="/workspace/libraries.performance.communication.oneccl"
ONECCL_BRANCH="release/ccl_2022.0.0-gold"
BUILD_DIR="${ONECCL_SRC}/build"
BUILD_ARTIFACT="${BUILD_DIR}/src/libccl.so.1.0"
BACKUP_SUFFIX=".bak.$(date +%Y%m%d%H%M%S)"
MAKE_JOBS="$(nproc)"

GITHUB_TOKEN_FILE="$SCRIPT_DIR/.github_token"
if [ -f "$GITHUB_TOKEN_FILE" ]; then
  GITHUB_TOKEN=$(cat "$GITHUB_TOKEN_FILE")
  ONECCL_REPO="https://${GITHUB_TOKEN}@github.com/intel-innersource/libraries.performance.communication.oneccl.git"
fi


# --- Parse arguments ---
DO_BUILD=true
DO_REPLACE=true
if [[ "${1:-}" == "--build-only" ]]; then
    DO_REPLACE=false
elif [[ "${1:-}" == "--replace-only" ]]; then
    DO_BUILD=false
fi

log() { echo -e "\n\033[1;32m>>> $*\033[0m"; }
err() { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; exit 1; }

# ============================================================
# Step 1: Ensure source code exists and is up-to-date
# ============================================================
if [[ "$DO_BUILD" == true ]]; then
    if [[ ! -d "${ONECCL_SRC}/.git" ]]; then
        log "Cloning oneCCL repository..."
        git clone --branch "${ONECCL_BRANCH}" "${ONECCL_REPO}" "${ONECCL_SRC}"
    else
        log "Updating oneCCL source (${ONECCL_BRANCH})..."
        pushd "${ONECCL_SRC}" > /dev/null
        git fetch origin
        LOCAL=$(git rev-parse HEAD)
        REMOTE=$(git rev-parse "origin/${ONECCL_BRANCH}")
        if [[ "$LOCAL" != "$REMOTE" ]]; then
            log "New commits found, resetting to origin/${ONECCL_BRANCH}..."
            git reset --hard "origin/${ONECCL_BRANCH}"
        else
            log "Source already up-to-date ($(git log --oneline -1))"
            # Skip build if artifact already exists and was built from this commit
            if [[ -f "${BUILD_ARTIFACT}" ]]; then
                LAST_BUILD_COMMIT="$(cat "${BUILD_DIR}/.build_commit" 2>/dev/null || true)"
                if [[ "$LOCAL" == "$LAST_BUILD_COMMIT" ]]; then
                    log "Build artifact already up-to-date for commit ${LOCAL:0:10}, skipping build."
                    DO_BUILD=false
                fi
            fi
        fi
        popd > /dev/null
    fi
fi

if [[ "$DO_BUILD" == true ]]; then
    # ============================================================
    # Step 2: Build
    # ============================================================
    log "Configuring oneCCL (build dir: ${BUILD_DIR})..."
    mkdir -p "${BUILD_DIR}"
    pushd "${BUILD_DIR}" > /dev/null

    # Ensure oneAPI compiler is available
    if ! command -v icx &>/dev/null || ! command -v icpx &>/dev/null; then
        if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
            source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
        else
            err "icx/icpx not found and oneAPI setvars.sh is missing"
        fi
    fi

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCOMPUTE_BACKEND=dpcpp \
        -DCCL_ENABLE_ARCB=1 \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_FT=OFF \
        -DBUILD_REG_TESTS=OFF

    log "Building libccl.so with ${MAKE_JOBS} jobs..."
    make -j"${MAKE_JOBS}" ccl
    popd > /dev/null

    [[ -f "${BUILD_ARTIFACT}" ]] || err "Build artifact not found: ${BUILD_ARTIFACT}"

    # Record the commit used for this build
    git -C "${ONECCL_SRC}" rev-parse HEAD > "${BUILD_DIR}/.build_commit"
    log "Build succeeded: ${BUILD_ARTIFACT} ($(du -h "${BUILD_ARTIFACT}" | cut -f1))"
fi

# ============================================================
# Step 3: Discover all libccl.so to replace
# ============================================================
if [[ "$DO_REPLACE" == true ]]; then
    [[ -f "${BUILD_ARTIFACT}" ]] || err "No build artifact at ${BUILD_ARTIFACT}. Run build first."

    # Collect replacement targets (real files only, not symlinks):
    #  - oneAPI: /opt/intel/oneapi/ccl/*/lib/libccl.so.1.0
    #  - conda envs: /opt/miniforge3/envs/*/lib/libccl.so{,.1,.1.0}
    #    (conda packages copy files instead of using symlinks)
    declare -a TARGETS=()

    # oneAPI installed copies (real files, not symlinks)
    while IFS= read -r f; do
        TARGETS+=("$f")
    done < <(find /opt/intel/oneapi -name "libccl.so.1.0" -not -type l 2>/dev/null || true)

    # conda environment copies — match libccl.so, libccl.so.1, libccl.so.1.0 only
    for pattern in "libccl.so" "libccl.so.1" "libccl.so.1.0"; do
        while IFS= read -r f; do
            [[ -f "$f" && ! -L "$f" ]] && TARGETS+=("$f")
        done < <(find /opt/miniforge3/envs -name "$pattern" -not -type l 2>/dev/null || true)
    done

    if [[ ${#TARGETS[@]} -eq 0 ]]; then
        err "No libccl.so files found to replace"
    fi

    log "Found ${#TARGETS[@]} file(s) to check:"
    printf "    %s\n" "${TARGETS[@]}"

    # ============================================================
    # Step 4: Backup and replace (skip if identical)
    # ============================================================
    BUILD_MD5=$(md5sum "${BUILD_ARTIFACT}" | awk '{print $1}')
    REPLACED=0
    SKIPPED=0

    for target in "${TARGETS[@]}"; do
        TARGET_MD5=$(md5sum "${target}" | awk '{print $1}')
        if [[ "$BUILD_MD5" == "$TARGET_MD5" ]]; then
            echo "    [SKIP] ${target} (already identical)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        backup="${target}${BACKUP_SUFFIX}"
        log "Replacing: ${target}"
        echo "    Backup  -> ${backup}"
        cp -a "${target}" "${backup}"
        cp "${BUILD_ARTIFACT}" "${target}"
        echo "    Done    ($(du -h "${target}" | cut -f1))"
        REPLACED=$((REPLACED + 1))
    done

    log "Result: ${REPLACED} replaced, ${SKIPPED} skipped (already identical)"
    if [[ $REPLACED -gt 0 ]]; then
        # Verify
        log "Verification (md5sum):"
        md5sum "${BUILD_ARTIFACT}"
        for target in "${TARGETS[@]}"; do
            md5sum "${target}"
        done
        echo "    To restore, run:"
        for target in "${TARGETS[@]}"; do
            backup="${target}${BACKUP_SUFFIX}"
            [[ -f "$backup" ]] && echo "      cp '${backup}' '${target}'"
        done
    fi
fi
