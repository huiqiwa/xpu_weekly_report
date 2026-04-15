DOCKER_BUILDKIT=1 docker build \
    --secret id=github_token,src=.github_token \
    -t xpu-perf-weekly . &> compose.txt
