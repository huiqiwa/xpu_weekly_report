# XPU Weekly Report

## Getting Started

### 1. Build the Docker Image

```bash
bash docker_build.sh
```

> The image is based on `intel/oneapi:2025.3.1-0-devel-ubuntu24.04` and installs Python, ccache, Miniforge, and other dependencies. The first build may take a while. Build logs are saved to `compose.txt`.

---

### 2. Create and Enter the Container

Start the container using `docker-compose.yml`:

```bash
# Start the container in the background
docker compose up -d

# Enter the container
docker compose exec node bash
```

Or start it directly with `docker run`:

```bash
docker run -it --rm \
  --privileged \
  --ipc=host \
  --network=host \
  --device /dev/dri:/dev/dri \
  -v /mnt:/mnt \
  -v $(pwd)/..:/workspace \
  -w /workspace/xpu_weekly_report \
  xpu-perf-weekly:latest \
  bash
```

> **Note:** The `container_name` in `docker-compose.yml` is `yupengzh-xpu-perf`. If multiple users share the same machine, update `name` and `container_name` in `docker-compose.yml` accordingly.

---

### 3. Set Up the Environment (Inside the Container)

After entering the container, activate the conda environment and install all dependencies:

```bash
# Option 1: Run the full setup automatically (recommended for first-time use)
bash setup_baremetal.sh

# Option 2: Activate an already-set-up environment
source activate_env.sh
```

`setup_baremetal.sh` performs the following steps:
1. Install system packages (ccache, intel-ocloc)
2. Create / activate the conda environment `xpu-perf-test` (Python 3.12)
3. Install PyTorch XPU, numpy, and ninja
4. Build vllm-xpu-kernels, auto-round, sycl-tla, and oneDNN
5. Prepare the xpu-perf repository

---

### 4. Run Tests

```bash
bash run_test.sh [REPORT_DIR]
```

- `REPORT_DIR`: Optional. Specifies the output directory for reports. Defaults to `reports/reports_<timestamp>`.
- Results are saved under the specified `REPORT_DIR` after the tests complete.

---

### 5. Stop / Remove the Container

```bash
docker compose down
```
