docker buildx create --use --name nolimit \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=-1 \
    --driver-opt env.HTTP_PROXY=http://proxy-ir.intel.com:911 \
    --driver-opt env.HTTPS_PROXY=http://proxy-ir.intel.com:912

docker buildx ls

docker buildx rm nolimit

docker buildx use nolimit


DOCKER_BUILDKIT=1 docker compose build  &> compose.txt
