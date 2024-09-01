/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"

#include "cutlass/conv/convolution.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileSchedulerTag
>
class ConvUniversal<
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag,
  cute::enable_if_t<cute::is_base_of_v<cutlass::conv::KernelImplicitTmaWarpSpecializedSm90,
                                       typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static constexpr int NumSpatialDimensions = CollectiveMainloop::NumSpatialDimensions;
  static_assert(ArchTag::kMinComputeCapability >= 90);
  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(cute::is_void_v<TileSchedulerTag>,
    "TMA warp-specialized kernel does not support specializing the tile scheduler.");
  using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
      TileSchedulerTag, ArchTag, TileShape, ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;

  // Kernel level shared memory storage
  struct SharedStorage {
    union TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
    } pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{})) + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Host facing host arguments
  struct Arguments {
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel device entry point API
  struct Params {
    MainloopParams mainloop;
    EpilogueParams epilogue;
  };

  //
  // Methods
  //

  // Map user facing arguments to device facing params
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    auto mainloop_params = CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace);
    auto problem_shape_MNKL = append<4>(mainloop_params.problem_shape, Int<1>{});

    return {
      mainloop_params,
      CollectiveEpilogue::to_underlying_arguments(problem_shape_MNKL, args.epilogue, workspace)
    };
  }

  // Given arguemnts, returns true if the kernel can successfully compute upon them. False otherwise.
  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
    implementable &= CollectiveMainloop::can_implement(args.mainloop.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.mainloop.problem_shape.get_transformed_problem_shape_MNK(), args.epilogue);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // The CONV mainloop params problem shape will be the cute::Shape<> rank-3 MNK tuple we want for grid planning
    // Although conv problems do not have an L mode, we add it here to comply with the scheduler API
    auto linear_problem_shape_MNKL = make_shape(
        size<0>(params.mainloop.problem_shape), // M mode is linearized.
        shape<1>(params.mainloop.problem_shape),
        shape<2>(params.mainloop.problem_shape),
        Int<1>{});

    return cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::get_tiled_cta_shape_mnl(
        linear_problem_shape_MNKL, TileShape{}, ClusterShape{});
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    // Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
    #if ! defined(__CUDA_ARCH_FEAT_SM90_ALL)
      if constexpr(size<0>(typename TiledMma::AtomShape_MNK{}) == 64) {
        printf("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
        return;
      }
    #endif

    enum class WarpGroupRole {
      Producer = 0,
      Consumer = 1,
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int warp_idx   = canonical_warp_idx_sync();
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    int lane_predicate = cute::elect_one_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = 1; // 1 thread issues TMA load
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    // Separate out problem shape for convenience
    auto M = get<0>(params.mainloop.problem_shape);
    auto N = get<1>(params.mainloop.problem_shape);
    auto K = get<2>(params.mainloop.problem_shape);
    // output strides are coalesced so we linearize the output shape to match the shape/stride profiles
    auto linear_problem_shape_MNKL = make_shape(size(M), N, K, Int<1>{});

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mk = params.mainloop.tma_load_a.get_tma_tensor(make_shape(M, size(K)));
    Tensor mB_nk = params.mainloop.tma_load_b.get_tma_tensor(make_shape(N, K));

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    auto cta_tile_shape = TileShape{};                                                           // (BLK_M,BLK_N,BLK_K)
    TiledMma tiled_mma;

    // Make tiled views, defer the slice
    Tensor gA_mk = local_tile(mA_mk, cta_tile_shape, make_coord(_,_,_), Step<_1, X,_1>{});         // (BLK_M,BLK_K,m,k)
    Tensor gB_nk = local_tile(mB_nk, cta_tile_shape, make_coord(_,_,_), Step< X,_1,_1>{});         // (BLK_N,BLK_K,n,k)

    // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
    auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mk));
    auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nk));
    // The output shape M is linearized so the output coord M here should also be linearized.
    auto output_tile_coord = make_coord(int(blockIdx.x), n_coord, _, Int<0>{});

    // Slice with m_coord and n_coord
    Tensor gA = gA_mk(_,_,m_coord,_);                                                                // (BLK_M,BLK_K,k)
    Tensor gB = gB_nk(_,_,n_coord,_);                                                                // (BLK_N,BLK_K,k)

    // Get pipeline iterators and increments from tensor shapes
    auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
    auto k_tile_count = size<2>(gA);

    auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(cta_tile_shape);
    auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(cta_tile_shape);

    // Make sure pipeline init is visible to all producers and consumer CTAs in cluster
    if constexpr (size(ClusterShape{}) > 1) {
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    }
    else {
      __syncthreads();
    }

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue{params.epilogue, shared_storage.tensors.epilogue};

    if (warp_group_role == WarpGroupRole::Producer) {
      collective_mainloop.load(
        mainloop_pipeline,
        mainloop_pipe_producer_state,
        gA, params.mainloop.tma_load_a,
        gB, params.mainloop.tma_load_b,
        k_tile_iter, k_tile_count,
        thread_idx,
        shared_storage.tensors.mainloop
      );
      // Update starting mainloop pipeline state for the pipeline drain
      mainloop_pipe_producer_state.advance(k_tile_count);
      // Make sure mainloop consumer has been waited upon before issuing epilogue load
      collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

      if (collective_epilogue.is_producer_load_needed()) {
        collective_epilogue.load(
          epi_load_pipeline,
          epi_load_pipe_producer_state,
          linear_problem_shape_MNKL,
          cta_tile_shape,
          output_tile_coord,
          tiled_mma,
          warp_group_thread_idx,
          shared_storage.tensors.epilogue
        );
        // Update starting load pipeline state for the pipeline drain
        epi_load_pipe_producer_state.advance(c_tile_count);
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      }
    }
    else if (warp_group_role == WarpGroupRole::Consumer) {
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(cta_tile_shape));            // (MMA,MMA_M,MMA_N)

      collective_mainloop.mma(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        accumulators,
        k_tile_count,
        thread_idx,
        shared_storage.tensors.mainloop,
        params.mainloop
      );

      // Make sure the math instructions are done and free buffers before entering the epilogue
      collective_mainloop.mma_tail(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        k_tile_count
      );

      // Epilogue and write to gD
      collective_epilogue.store(
        epi_load_pipeline,
        epi_load_pipe_consumer_state,
        epi_store_pipeline,
        epi_store_pipe_producer_state,
        linear_problem_shape_MNKL,
        cta_tile_shape,
        output_tile_coord,
        accumulators,
        tiled_mma,
        warp_group_thread_idx,
        shared_storage.tensors.epilogue
      );
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::kernel
