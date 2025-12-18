/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/tree.cuh>
#include <flashinfer/pos_enc.cuh>

#include "batch_tree_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          uint32_t TREE_WORDS_PER_Q,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchTreeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                               float* tmp_s, bool enable_pdl,
                                               cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace {

constexpr size_t kPrefillPlanInfoVecSize = 15;
constexpr size_t kTreePlanInfoVecSize = kPrefillPlanInfoVecSize + 6;

struct TreePlanInfo {
  PrefillPlanInfo prefill;
  int64_t tree_info_ptr;
  int64_t tree_info_stride_page;
  int64_t tree_info_stride_entry;
  int64_t tree_info_stride_word;
  int64_t tree_info_sche_len;
  int64_t max_tree_height;

  std::vector<int64_t> ToVector() const {
    std::vector<int64_t> vec = prefill.ToVector();
    vec.push_back(tree_info_ptr);
    vec.push_back(tree_info_stride_page);
    vec.push_back(tree_info_stride_entry);
    vec.push_back(tree_info_stride_word);
    vec.push_back(tree_info_sche_len);
    vec.push_back(max_tree_height);
    return vec;
  }

  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != kTreePlanInfoVecSize) {
      std::ostringstream err_msg;
      err_msg << "TreePlanInfo::FromVector: vec.size() should be " << kTreePlanInfoVecSize
              << ", but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    std::vector<int64_t> pre(vec.begin(), vec.begin() + kPrefillPlanInfoVecSize);
    prefill.FromVector(pre);
    tree_info_ptr = vec[kPrefillPlanInfoVecSize + 0];
    tree_info_stride_page = vec[kPrefillPlanInfoVecSize + 1];
    tree_info_stride_entry = vec[kPrefillPlanInfoVecSize + 2];
    tree_info_stride_word = vec[kPrefillPlanInfoVecSize + 3];
    tree_info_sche_len = vec[kPrefillPlanInfoVecSize + 4];
    max_tree_height = vec[kPrefillPlanInfoVecSize + 5];
  }
};

}  // namespace

Array<int64_t> BatchTreeWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr, TensorView kv_indptr,
    TensorView kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, TensorView tree_info, int64_t max_tree_height, bool causal,
    int64_t window_left, int64_t fixed_split_size,
    bool disable_split_kv, int64_t num_colocated_ctas = 0) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  PrefillPlanInfo plan_info;
  // Cache tree_info pointer/strides and MAX_SCHE_LEN (tree_info.size(2)) in the plan output.
  // tree_info: [max_num_pages, page_size, MAX_SCHE_LEN], uint32
  TVM_FFI_ICHECK_EQ(tree_info.ndim(), 3) << "tree_info must be a 3-D tensor";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(tree_info.dtype()), encode_dlpack_dtype(dl_uint32))
      << "tree_info must have dtype uint32";
  TVM_FFI_ICHECK_EQ(tree_info.device().device_type, float_workspace_buffer.device().device_type)
      << "tree_info must be on the same device type as workspace";
  TVM_FFI_ICHECK_EQ(tree_info.device().device_id, float_workspace_buffer.device().device_id)
      << "tree_info must be on the same CUDA device as workspace";
  TVM_FFI_ICHECK_EQ(tree_info.size(1), page_size) << "tree_info.page_size must match KV page_size";
  TVM_FFI_ICHECK(tree_info.size(2) >= 2) << "tree_info MAX_SCHE_LEN must be >= 2";
  TVM_FFI_ICHECK(max_tree_height >= 1) << "max_tree_height must be >= 1";
  const int64_t expected_tree_words_per_q = 1 + ((max_tree_height - 1 + 3) / 4);
  TVM_FFI_ICHECK_EQ(expected_tree_words_per_q, TREE_WORDS_PER_Q)
      << "TREE_WORDS_PER_Q mismatch: expected " << expected_tree_words_per_q
      << " from max_tree_height, but compiled TREE_WORDS_PER_Q is " << TREE_WORDS_PER_Q;
  TVM_FFI_ICHECK(tree_info.size(2) >= TREE_WORDS_PER_Q + 1)
      << "tree_info MAX_SCHE_LEN is insufficient for compiled TREE_WORDS_PER_Q";

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  cudaError_t status = PrefillPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr.data_ptr()),
      static_cast<IdType*>(kv_indptr.data_ptr()), total_num_rows, batch_size, num_qo_heads,
      num_kv_heads, head_dim_qk, head_dim_vo, page_size, enable_cuda_graph,
      /*sizeof_dtype_o=*/2, window_left, fixed_split_size, disable_split_kv, num_colocated_ctas,
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan tree(prefill) with error: " << cudaGetErrorString(status);

  TreePlanInfo tree_plan_info;
  tree_plan_info.prefill = plan_info;
  tree_plan_info.tree_info_ptr = reinterpret_cast<int64_t>(tree_info.data_ptr());
  tree_plan_info.tree_info_stride_page = tree_info.stride(0);
  tree_plan_info.tree_info_stride_entry = tree_info.stride(1);
  tree_plan_info.tree_info_stride_word = tree_info.stride(2);
  tree_plan_info.tree_info_sche_len = tree_info.size(2);
  tree_plan_info.max_tree_height = max_tree_height;
  return Array(tree_plan_info.ToVector());
}

void BatchTreeWithPagedKVCacheRun(TensorView float_workspace_buffer, TensorView int_workspace_buffer,
                                 Array<int64_t> plan_info_vec, TensorView q, TensorView paged_k_cache,
                                 TensorView paged_v_cache,
                                 TensorView qo_indptr,
                                 TensorView paged_kv_indptr, TensorView paged_kv_indices,
                                 TensorView paged_kv_last_page_len, TensorView o,
                                 Optional<TensorView> maybe_lse, int64_t mask_mode_code, int64_t layout,
                                 int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  TreePlanInfo tree_plan_info;
  tree_plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));
  PrefillPlanInfo plan_info = tree_plan_info.prefill;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  int64_t num_qo_heads = q.size(1);
  int64_t head_dim_qk = q.size(2);
  int64_t head_dim_vo = head_dim_qk;
  int64_t page_size = paged_k_cache.size(1);
  int64_t batch_size = paged_kv_last_page_len.size(0);
  int64_t num_kv_heads = (kv_layout == QKVLayout::kNHD) ? paged_k_cache.size(2) : paged_k_cache.size(1);

  // tree_info pointer/strides/MAX_SCHE_LEN/max_tree_height are cached in plan_info.
  TVM_FFI_ICHECK(tree_plan_info.max_tree_height >= 1) << "cached max_tree_height must be >= 1";
  const int64_t expected_tree_words_per_q = 1 + ((tree_plan_info.max_tree_height - 1 + 3) / 4);
  TVM_FFI_ICHECK_EQ(expected_tree_words_per_q, TREE_WORDS_PER_Q)
      << "cached max_tree_height mismatches compiled TREE_WORDS_PER_Q, please re-plan";
  TVM_FFI_ICHECK(tree_plan_info.tree_info_sche_len >= TREE_WORDS_PER_Q + 1)
      << "cached tree_info MAX_SCHE_LEN is insufficient for compiled TREE_WORDS_PER_Q";

  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides = paged_k_cache.strides().data();
  TVM_FFI_ICHECK_EQ(paged_k_cache.ndim(), paged_v_cache.ndim());
  for (int i = 0; i < paged_k_cache.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(paged_k_cache.stride(i), paged_v_cache.stride(i))
        << "k/v strides differs at " << i;
  }

  if (maybe_lse.has_value()) {
    const auto& lse = *maybe_lse;
    TVM_FFI_ICHECK_EQ(lse.size(0), q.size(0));
    TVM_FFI_ICHECK_EQ(lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  TVM_FFI_ICHECK(mask_mode == MaskMode::kNone)
      << "BatchTree only supports tree_mask, please pass mask_mode=NonCausal(kNone)";

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      PagedParams, [&] {
        PagedParams params;
        params.q = static_cast<DTypeQ*>(q.data_ptr());
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        params.paged_kv = paged_kv;
        params.tree_info = reinterpret_cast<uint32_t*>(tree_plan_info.tree_info_ptr);
        params.tree_info_stride_page = tree_plan_info.tree_info_stride_page;
        params.tree_info_stride_entry = tree_plan_info.tree_info_stride_entry;
        params.tree_info_stride_word = tree_plan_info.tree_info_stride_word;
        params.tree_info_sche_len = static_cast<uint32_t>(tree_plan_info.tree_info_sche_len);
        params.max_tree_height = static_cast<uint32_t>(tree_plan_info.max_tree_height);
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse =
            maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchTreeWithPagedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, TREE_WORDS_PER_Q, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchTreeWithPagedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}
