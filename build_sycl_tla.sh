mkdir build && cd build

cmake .. -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21   # 只生成 bmg-g21，不带 bmg-g31

cmake --build . --target \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim64 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim96 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim128 \
  06_xe_fmha_fwd_prefill_bfloat16_t_hdim192 \
  06_xe_fmha_fwd_decode_bfloat16_t_hdim64 \
  06_xe_fmha_fwd_decode_bfloat16_t_hdim96 \
  06_xe_fmha_fwd_decode_bfloat16_t_hdim128 \
  06_xe_fmha_fwd_decode_bfloat16_t_hdim192 \
  06_xe_fmha_fwd_prefill_float_e4m3_t_hdim64 \
  06_xe_fmha_fwd_prefill_float_e4m3_t_hdim96 \
  06_xe_fmha_fwd_prefill_float_e4m3_t_hdim128 \
  06_xe_fmha_fwd_prefill_float_e4m3_t_hdim192 \
  06_xe_fmha_fwd_decode_float_e4m3_t_hdim64 \
  06_xe_fmha_fwd_decode_float_e4m3_t_hdim96 \
  06_xe_fmha_fwd_decode_float_e4m3_t_hdim128 \
  06_xe_fmha_fwd_decode_float_e4m3_t_hdim192 \
  -j32