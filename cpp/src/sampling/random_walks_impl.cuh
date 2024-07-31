/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "detail/graph_partition_utils.cuh"
#include "prims/per_v_random_select_transform_outgoing_e.cuh"
#include "prims/vertex_frontier.cuh"
#include "prims/property_op_utils.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/detail/nbr_intersection.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/graph_functions.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/optional.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>
#include <type_traits>

namespace cugraph {
namespace detail {

enum class random_walk_t{UNIFORM, BIASED, NODE2VEC};

inline uint64_t get_current_time_nanoseconds()
{
  auto cur = std::chrono::steady_clock::now();
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(cur.time_since_epoch()).count());
}

template <typename vertex_t, typename weight_t>
struct sample_edges_op_t {
  template <typename W = weight_t>
  __device__ std::enable_if_t<std::is_same_v<W, void>, vertex_t> operator()(
    vertex_t, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return dst;
  }

  template <typename W = weight_t>
  __device__ std::enable_if_t<!std::is_same_v<W, void>, thrust::tuple<vertex_t, W>> operator()(
    vertex_t, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, W w) const
  {
    return thrust::make_tuple(dst, w);
  }
};

template <typename vertex_t, typename bias_t>
struct biased_random_walk_e_bias_op_t {

  __device__ bias_t
  operator()(vertex_t, vertex_t, bias_t src_out_weight_sum, thrust::nullopt_t, bias_t weight) const
  {
    return weight / src_out_weight_sum;
  }
};

template <typename vertex_t, typename weight_t>
struct biased_sample_edges_op_t {
  __device__ thrust::tuple<vertex_t, weight_t>
  operator()(vertex_t, vertex_t dst, weight_t, thrust::nullopt_t, weight_t weight) const
  {
    return thrust::make_tuple(dst, weight);
  }
};

template <typename vertex_t, typename bias_t, typename weight_t>
struct node2vec_random_walk_e_bias_op_t {
  bias_t p_;
  bias_t q_;
  raft::device_span<size_t const> intersection_offsets_{};
  raft::device_span<vertex_t const> intersection_indices_{};
  raft::device_span<vertex_t const> current_vertices_{};
  raft::device_span<vertex_t const> prev_vertices_{};

  // Unweighted Bias Operator
  template <typename W = weight_t>
  __device__ std::enable_if_t<std::is_same_v<W, void>, bias_t> operator()(
        thrust::tuple<vertex_t, vertex_t> tagged_src,
        vertex_t dst,
        thrust::nullopt_t,
        thrust::nullopt_t,
        thrust::nullopt_t) const
  {
    //  Check tag (prev vert) for destination
    if(dst == thrust::get<1>(tagged_src)){
      return 1.0 / p_;
    }
    //  Search zipped vertices for tagged src
    auto lower_itr = thrust::lower_bound(
              thrust::seq, 
              thrust::make_zip_iterator(current_vertices_.begin(), prev_vertices_.begin()),
              thrust::make_zip_iterator(current_vertices_.end(), prev_vertices_.end()),
              tagged_src);
    auto low_idx = thrust::distance(thrust::make_zip_iterator(current_vertices_.begin(), 
                                                              prev_vertices_.begin()), 
                                    lower_itr);

    auto start_idx = intersection_offsets_[low_idx];
    auto end_idx = intersection_offsets_[low_idx + 1];
    auto itr = thrust::lower_bound(
        thrust::seq,
        intersection_indices_.begin() + start_idx, 
        intersection_indices_.begin() + end_idx, 
        dst);
    //  dst not in intersection
    if(itr == intersection_indices_.begin() + end_idx){
      return 1.0 / q_;
    }
    return 1.0;
  }

  //  Weighted Biase Operator
  template <typename W = weight_t>
  __device__ std::enable_if_t<!std::is_same_v<W, void>, bias_t> operator()(
        thrust::tuple<vertex_t, vertex_t> tagged_src,
        vertex_t dst,
        thrust::nullopt_t,
        thrust::nullopt_t,
        W w) const
  {
    //  Check tag (prev vert) for destination
    if(dst == thrust::get<1>(tagged_src)){
      return 1.0 / p_;
    }
    //  Search zipped vertices for tagged src
    auto lower_itr = thrust::lower_bound(
              thrust::seq, 
              thrust::make_zip_iterator(current_vertices_.begin(), prev_vertices_.begin()),
              thrust::make_zip_iterator(current_vertices_.end(), prev_vertices_.end()),
              tagged_src);
    auto low_idx = thrust::distance(thrust::make_zip_iterator(current_vertices_.begin(), 
                                                              prev_vertices_.begin()), 
                                    lower_itr);
    auto start_idx = intersection_offsets_[low_idx];
    auto end_idx = intersection_offsets_[low_idx + 1];
    auto itr = thrust::lower_bound(
        thrust::seq,
        intersection_indices_.begin() + start_idx, 
        intersection_indices_.begin() + end_idx, 
        dst);
    //  dst not in intersection
    if(itr == intersection_indices_.begin() + end_idx){
      return 1.0 / q_;
    }
    return 1.0;
  }
};

template <typename vertex_t, typename weight_t>
struct node2vec_sample_edges_op_t {
  template <typename W = weight_t>
  __device__ std::enable_if_t<std::is_same_v<W, void>, vertex_t> operator()(
        thrust::tuple<vertex_t, vertex_t> tagged_src,
        vertex_t dst,
        thrust::nullopt_t,
        thrust::nullopt_t,
        thrust::nullopt_t) const
  {
    return dst;
  }

  template <typename W = weight_t>
  __device__ std::enable_if_t<!std::is_same_v<W, void>, thrust::tuple<vertex_t, W>> operator()(
        thrust::tuple<vertex_t, vertex_t> tagged_src,
        vertex_t dst,
        thrust::nullopt_t,
        thrust::nullopt_t,
        W w) const
  {
    return thrust::make_tuple(dst, w);
  }
};

template <typename weight_t>
struct uniform_selector {
  raft::random::RngState& rng_state_;

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type> const& current_vertices,
    std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>> &previous_vertices)
  {
    using vertex_t = typename GraphViewType::vertex_type;

    // FIXME: add as a template parameter
    using tag_t = void;

    cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(
      handle, 1);

    vertex_frontier.bucket(0).insert(current_vertices.begin(), current_vertices.end());

    rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
    if (edge_weight_view) {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          *edge_weight_view,
          sample_edges_op_t<vertex_t, weight_t>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(
            thrust::make_tuple(cugraph::invalid_vertex_id<vertex_t>::value, weight_t{0.0})));

      minors  = std::move(std::get<0>(sample_e_op_results));
      weights = std::move(std::get<1>(sample_e_op_results));
    } else {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          sample_edges_op_t<vertex_t, void>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}));

      minors = std::move(sample_e_op_results);
    }
    return std::make_tuple(std::move(minors), std::move(weights));
  }
};

template <typename weight_t>
struct biased_selector {
  raft::random::RngState& rng_state_;

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type> const& current_vertices,
    std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>>& previous_vertices)
  {
    //  To do biased sampling, I need out_weights instead of out_degrees.
    //  Then I generate a random float between [0, out_weights[v]).  Then
    //  instead of making a decision based on the index I need to find
    //  upper_bound (or is it lower_bound) of the random number and
    //  the cumulative weight.

    //  Create vertex frontier
    using vertex_t = typename GraphViewType::vertex_type;

    using tag_t = void;

    cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(handle, 1);

    vertex_frontier.bucket(0).insert(current_vertices.begin(), current_vertices.end());

    // Create data structs for results
    rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
    rmm::device_uvector<weight_t> weights(0, handle.get_stream());

    auto vertex_weight_sum  = compute_out_weight_sums(handle, graph_view, *edge_weight_view);
    edge_src_property_t<GraphViewType, weight_t> edge_src_out_weight_sums(handle, graph_view);
    update_edge_src_property(handle, graph_view, vertex_weight_sum.data(), edge_src_out_weight_sums.mutable_view());
    auto [sample_offsets, sample_e_op_results] =
      cugraph::per_v_random_select_transform_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        edge_src_out_weight_sums.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        *edge_weight_view,
        biased_random_walk_e_bias_op_t<vertex_t, weight_t>{},
        edge_src_out_weight_sums.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        *edge_weight_view,
        biased_sample_edges_op_t<vertex_t, weight_t>{},
        rng_state_,
        size_t{1},
        true,
        std::make_optional(
            thrust::make_tuple(vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}, weight_t{0.0})));
    minors = std::move(std::get<0>(sample_e_op_results));
    weights = std::move(std::get<1>(sample_e_op_results));

    //  Return results
    return std::make_tuple(std::move(minors), std::move(weights));
  }
};

template <typename weight_t>
struct node2vec_selector {
  weight_t p_;
  weight_t q_;
  raft::random::RngState& rng_state_;

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type>& current_vertices,
    std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>>& previous_vertices)
  {
    //  To do node2vec, I need the following:
    //    1) transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v to compute the sum of the
    //       node2vec style weights
    //    2) Generate a random number between [0, output_from_trdnioeebv[v])
    //    3) a sampling value that lets me pick the correct edge based on the same computation
    //       (essentially weighted sampling, but with a function that computes the weight rather
    //       than just using the edge weights)

    //  Create vertex frontier
    using vertex_t = typename GraphViewType::vertex_type;
    
    using tag_t = vertex_t;

    cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(handle, 1);
    vertex_frontier.bucket(0).insert(thrust::make_zip_iterator(current_vertices.begin(),
                                                              (*previous_vertices).begin()), 
                                     thrust::make_zip_iterator(current_vertices.end(), 
                                                              (*previous_vertices).end()));         

    //  Zip previous and current vertices for nbr_intersection()
    auto intersection_pairs = thrust::make_zip_iterator(current_vertices.begin(), (*previous_vertices).begin());
    
    auto [intersection_offsets, intersection_indices] =
      detail::nbr_intersection(handle,
                               graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               intersection_pairs,
                               intersection_pairs + current_vertices.size(),
                               std::array<bool, 2>{true, true},
                               false);
      
    // Create data structs for results
    rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};

    if (edge_weight_view) {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          *edge_weight_view,
          node2vec_random_walk_e_bias_op_t<vertex_t, weight_t, weight_t>{
            p_, 
            q_, 
            raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()), 
            raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()), 
            raft::device_span<vertex_t const>(current_vertices.data(), current_vertices.size()), 
            raft::device_span<vertex_t const>((*previous_vertices).data(), (*previous_vertices).size())
          },
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          *edge_weight_view,
          node2vec_sample_edges_op_t<vertex_t, weight_t>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(
            thrust::make_tuple(vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}, weight_t{0.0})));
        minors = std::move(std::get<0>(sample_e_op_results));
        weights = std::move(std::get<1>(sample_e_op_results));
    } else {
      auto[ sample_offsets, sample_e_op_results] =
      cugraph::per_v_random_select_transform_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        node2vec_random_walk_e_bias_op_t<vertex_t, weight_t, void>{
          p_, 
          q_, 
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()), 
          raft::device_span<vertex_t const>(intersection_indices.data(), intersection_indices.size()), 
          raft::device_span<vertex_t const>(current_vertices.data(), current_vertices.size()), 
          raft::device_span<vertex_t const>((*previous_vertices).data(), (*previous_vertices).size())
        },
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        node2vec_sample_edges_op_t<vertex_t, void>{},
        rng_state_,
        size_t{1},
        true,
        std::make_optional(vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}));
      minors = std::move(sample_e_op_results);
    }

    //  Copy current vertices to previous vertices for two-order walk
    thrust::copy(
                handle.get_thrust_policy(), 
                current_vertices.begin(), 
                current_vertices.end(), 
                (*previous_vertices).data());

    return std::make_tuple(std::move(minors), std::move(weights));
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename random_selector_t>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
random_walk_impl(raft::handle_t const& handle,
                 graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                 std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                 raft::device_span<vertex_t const> start_vertices,
                 size_t max_length,
                 random_walk_t walk_type,
                 random_selector_t random_selector)
{
  rmm::device_uvector<vertex_t> result_vertices(start_vertices.size() * (max_length + 1),
                                                handle.get_stream());
  auto result_weights = edge_weight_view
                          ? std::make_optional<rmm::device_uvector<weight_t>>(
                              start_vertices.size() * max_length, handle.get_stream())
                          : std::nullopt;

  detail::scalar_fill(handle,
                      result_vertices.data(),
                      result_vertices.size(),
                      cugraph::invalid_vertex_id<vertex_t>::value);
  if (result_weights)
    detail::scalar_fill(handle, result_weights->data(), result_weights->size(), weight_t{0});

  rmm::device_uvector<vertex_t> current_vertices(start_vertices.size(), handle.get_stream());
  rmm::device_uvector<size_t> current_position(start_vertices.size(), handle.get_stream());
  rmm::device_uvector<int> current_gpu(0, handle.get_stream());
  auto new_weights = edge_weight_view
                       ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
                       : std::nullopt;

  auto previous_vertices = (walk_type == random_walk_t::NODE2VEC) 
                            ? std::make_optional<rmm::device_uvector<vertex_t>>(current_vertices.size(), handle.get_stream())
                            : std::nullopt;
  if (previous_vertices) {
    raft::copy(
      (*previous_vertices).data(), start_vertices.data(), start_vertices.size(), handle.get_stream());
  }
  raft::copy(
    current_vertices.data(), start_vertices.data(), start_vertices.size(), handle.get_stream());
  detail::sequence_fill(
    handle.get_stream(), current_position.data(), current_position.size(), size_t{0});

  if constexpr (multi_gpu) {
    current_gpu.resize(start_vertices.size(), handle.get_stream());

    detail::scalar_fill(
      handle, current_gpu.data(), current_gpu.size(), handle.get_comms().get_rank());
  }

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(current_vertices.size()),
    [current_verts = current_vertices.data(),
     result_verts  = result_vertices.data(),
     max_length] __device__(size_t i) { result_verts[i * (max_length + 1)] = current_verts[i]; });

  rmm::device_uvector<vertex_t> vertex_partition_range_lasts(
    graph_view.vertex_partition_range_lasts().size(), handle.get_stream());
  raft::update_device(vertex_partition_range_lasts.data(),
                      graph_view.vertex_partition_range_lasts().data(),
                      graph_view.vertex_partition_range_lasts().size(),
                      handle.get_stream());

  for (size_t level = 0; level < max_length; ++level) {
    if constexpr (multi_gpu) {
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      if  (previous_vertices) {
        std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position, previous_vertices),
                            std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          thrust::make_zip_iterator(current_vertices.begin(),
                                    current_gpu.begin(),
                                    current_position.begin(),
                                    previous_vertices->begin()),
          thrust::make_zip_iterator(current_vertices.end(),
                                    current_gpu.end(),
                                    current_position.end(),
                                    previous_vertices->end()),
          [key_func =
             cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
               {vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.size()},
               major_comm_size,
               minor_comm_size}] __device__(auto val) { return key_func(thrust::get<0>(val)); },
          handle.get_stream());
      } else {
        // Shuffle vertices to correct GPU to compute random indices
        std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position),
                              std::ignore) =
          cugraph::groupby_gpu_id_and_shuffle_values(
            handle.get_comms(),
            thrust::make_zip_iterator(current_vertices.begin(),
                                            current_gpu.begin(),
                                            current_position.begin()),
            thrust::make_zip_iterator(current_vertices.end(),
                                            current_gpu.end(),
                                            current_position.end()),
            [key_func =
               cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                 {vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.size()},
                 major_comm_size,
                 minor_comm_size}] __device__(auto val) { return key_func(thrust::get<0>(val)); },
            handle.get_stream());
      }
    }

    //  Sort for nbr_intersection, must sort all together
    if (previous_vertices) {
      if constexpr (multi_gpu){
        thrust::sort(handle.get_thrust_policy(), 
                     thrust::make_zip_iterator(current_vertices.begin(),
                                               (*previous_vertices).begin(),
                                               current_position.begin(),
                                               current_gpu.begin()),
                     thrust::make_zip_iterator(current_vertices.end(),
                                               (*previous_vertices).end(),
                                               current_position.end(),
                                               current_gpu.end()));
      } else {
        thrust::sort(handle.get_thrust_policy(), 
                     thrust::make_zip_iterator(current_vertices.begin(),
                                               (*previous_vertices).begin(),
                                               current_position.begin()),
                     thrust::make_zip_iterator(current_vertices.end(),
                                               (*previous_vertices).end(),
                                               current_position.end()));
      }
    }

    std::tie(current_vertices, new_weights) =
      random_selector.follow_random_edge(handle, graph_view, edge_weight_view, current_vertices, previous_vertices);

    // FIXME: remove_if has a 32-bit overflow issue
    // (https://github.com/NVIDIA/thrust/issues/1302) Seems unlikely here (the goal of
    // sampling is to extract small graphs) so not going to work around this for now.
    CUGRAPH_EXPECTS(
      current_vertices.size() < static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "remove_if will fail, current_vertices.size() is too large");
    size_t compacted_length{0};
    if constexpr (multi_gpu) {
      if (result_weights) {
        if (previous_vertices) {
          auto input_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                      new_weights->begin(),
                                                      current_gpu.begin(),
                                                      current_position.begin(),
                                                      previous_vertices->begin());

          compacted_length = thrust::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        } else {
          auto input_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                      new_weights->begin(),
                                                      current_gpu.begin(),
                                                      current_position.begin());

          compacted_length = thrust::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        }
      } else {
        if (previous_vertices) {
          auto input_iter = thrust::make_zip_iterator(current_vertices.begin(), 
                                                      current_gpu.begin(), 
                                                      current_position.begin(), 
                                                      previous_vertices->begin());

          compacted_length = thrust::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        } else {
          auto input_iter = thrust::make_zip_iterator(
            current_vertices.begin(), current_gpu.begin(), current_position.begin());

          compacted_length = thrust::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        }
      }
    } else {
      if (result_weights) {
        if (previous_vertices) {
          auto input_iter = thrust::make_zip_iterator(current_vertices.begin(), 
                                                      new_weights->begin(), 
                                                      current_position.begin(),
                                                      previous_vertices->begin());

          compacted_length = thrust::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        } else {
          auto input_iter = thrust::make_zip_iterator(
            current_vertices.begin(), new_weights->begin(), current_position.begin());

          compacted_length = thrust::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        }
      } else {
        if (previous_vertices) {
          auto input_iter =
            thrust::make_zip_iterator(current_vertices.begin(), 
                                      current_position.begin(), 
                                      previous_vertices->begin());

          compacted_length = thrust::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        } else {
          auto input_iter =
            thrust::make_zip_iterator(current_vertices.begin(), current_position.begin());

          compacted_length = thrust::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        }
      }
    }

    //  Moved out of if statements to cut down on code duplication
    current_vertices.resize(compacted_length, handle.get_stream());
    current_position.resize(compacted_length, handle.get_stream());
    if (result_weights) {new_weights->resize(compacted_length, handle.get_stream());}
    if (previous_vertices) {previous_vertices->resize(compacted_length, handle.get_stream());}
    if constexpr (multi_gpu) {
      current_gpu.resize(compacted_length, handle.get_stream());

      // Shuffle back to original GPU
      if (previous_vertices) {
        if (result_weights) {
          auto current_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                        new_weights->begin(),
                                                        current_gpu.begin(),
                                                        current_position.begin(),
                                                        previous_vertices->begin());

          std::forward_as_tuple(
            std::tie(current_vertices, *new_weights, current_gpu, current_position, *previous_vertices), std::ignore) =
            cugraph::groupby_gpu_id_and_shuffle_values(
              handle.get_comms(),
              current_iter,
              current_iter + current_vertices.size(),
              [] __device__(auto val) { return thrust::get<2>(val); },
              handle.get_stream());
        } else {
          auto current_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                        current_gpu.begin(), 
                                                        current_position.begin(),
                                                        previous_vertices->begin());

          std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position, *previous_vertices),
                                std::ignore) =
            cugraph::groupby_gpu_id_and_shuffle_values(
              handle.get_comms(),
              current_iter,
              current_iter + current_vertices.size(),
              [] __device__(auto val) { return thrust::get<1>(val); },
              handle.get_stream());
        }
      } else {
        if (result_weights) {
          auto current_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                        new_weights->begin(),
                                                        current_gpu.begin(),
                                                        current_position.begin());

          std::forward_as_tuple(
            std::tie(current_vertices, *new_weights, current_gpu, current_position), std::ignore) =
            cugraph::groupby_gpu_id_and_shuffle_values(
              handle.get_comms(),
              current_iter,
              current_iter + current_vertices.size(),
              [] __device__(auto val) { return thrust::get<2>(val); },
              handle.get_stream());
        } else {          
          auto current_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                        current_gpu.begin(), 
                                                        current_position.begin());

          std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position),
                                std::ignore) =
            cugraph::groupby_gpu_id_and_shuffle_values(
              handle.get_comms(),
              current_iter,
              current_iter + current_vertices.size(),
              [] __device__(auto val) { return thrust::get<1>(val); },
              handle.get_stream());
        }
      }
    }

    if (result_weights) {
      thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(
            current_vertices.begin(), new_weights->begin(), current_position.begin()),
          thrust::make_zip_iterator(
            current_vertices.end(), new_weights->end(), current_position.end()),
          [result_verts = result_vertices.data(),
           result_wgts  = result_weights->data(),
           level,
           max_length] __device__(auto tuple) {
            vertex_t v                                       = thrust::get<0>(tuple);
            weight_t w                                       = thrust::get<1>(tuple);
            size_t pos                                       = thrust::get<2>(tuple);
            result_verts[pos * (max_length + 1) + level + 1] = v;
            result_wgts[pos * max_length + level]            = w;
          });
    } else {
      thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(current_vertices.begin(), current_position.begin()),
          thrust::make_zip_iterator(current_vertices.end(), current_position.end()),
          [result_verts = result_vertices.data(), level, max_length] __device__(auto tuple) {
            vertex_t v                                       = thrust::get<0>(tuple);
            size_t pos                                       = thrust::get<1>(tuple);
            result_verts[pos * (max_length + 1) + level + 1] = v;
          });
    }
  }

  return std::make_tuple(std::move(result_vertices), std::move(result_weights));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     raft::device_span<vertex_t const> start_vertices,
                     size_t max_length,
                     raft::random::RngState& rng_state)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::random_walk_impl(handle,
                                  graph_view,
                                  edge_weight_view,
                                  start_vertices,
                                  max_length,
                                  detail::random_walk_t::UNIFORM,
                                  detail::uniform_selector<weight_t>{rng_state});
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                    edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                    raft::device_span<vertex_t const> start_vertices,
                    size_t max_length,
                    raft::random::RngState& rng_state)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::random_walk_impl(
    handle,
    graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>>{edge_weight_view},
    start_vertices,
    max_length,
    detail::random_walk_t::BIASED,
    detail::biased_selector<weight_t>{rng_state});
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                      std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                      raft::device_span<vertex_t const> start_vertices,
                      size_t max_length,
                      weight_t p,
                      weight_t q,
                      raft::random::RngState& rng_state)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::random_walk_impl(
    handle,
    graph_view,
    edge_weight_view,
    start_vertices,
    max_length,
    detail::random_walk_t::NODE2VEC,
    detail::node2vec_selector<weight_t>{
      p, q, rng_state});
}

}  // namespace cugraph
