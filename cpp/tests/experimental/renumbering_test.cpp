/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/experimental/graph.hpp>
#include <cugraph/experimental/graph_functions.hpp>
#include <cugraph/experimental/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 0;

struct Renumbering_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_Renumbering
  : public ::testing::TestWithParam<std::tuple<Renumbering_Usecase, input_usecase_t>> {
 public:
  Tests_Renumbering() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(Renumbering_Usecase const& renumbering_usecase,
                        input_usecase_t const& input_usecase)
  {
    raft::handle_t handle{};
    HighResClock hr_clock{};

    rmm::device_uvector<vertex_t> src_v(0, handle.get_stream_view());
    rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream_view());
    rmm::device_uvector<vertex_t> renumber_map_labels_v(0, handle.get_stream_view());
    cugraph::experimental::partition_t<vertex_t> partition{};
    vertex_t number_of_vertices{};
    edge_t number_of_edges{};
    bool symmetric{false};

    std::tie(src_v, dst_v, std::ignore, number_of_vertices, symmetric) =
      input_usecase.template construct_edgelist<vertex_t, edge_t, weight_t, true, false>(handle,
                                                                                         false);

    std::vector<vertex_t> h_vertices_before_v(2 * src_v.size());

    if (renumbering_usecase.check_correctness) {
      raft::update_host(h_vertices_before_v.data(), src_v.data(), src_v.size(), handle.get_stream());
      raft::update_host(h_vertices_before_v.data() + src_v.size(), dst_v.data(), dst_v.size(), handle.get_stream());
    }

    if (PERF) {
      handle.get_stream_view().synchronize();  // for consistent performance measurement
      hr_clock.start();
    }

    std::optional<std::tuple<vertex_t const*, vertex_t>> optional_local_vertex_span(
      std::make_tuple(nullptr, vertex_t{0}));  // std::nullopt,

    renumber_map_labels_v = cugraph::experimental::renumber_edgelist<vertex_t, edge_t, false>(
      handle, optional_local_vertex_span, src_v.data(), dst_v.data(), src_v.size());

    if (PERF) {
      handle.get_stream_view().synchronize();  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "renumbering took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (renumbering_usecase.check_correctness) {
      std::vector<vertex_t> h_vertices_after_v(2 * src_v.size());

      raft::update_host(h_vertices_after_v.data(), src_v.data(), src_v.size(), handle.get_stream());
      raft::update_host(h_vertices_after_v.data() + src_v.size(), dst_v.data(), dst_v.size(), handle.get_stream());

      EXPECT_TRUE(cugraph::test::renumbered_vectors_same(handle, h_vertices_before_v, h_vertices_after_v));
    }
  }
};

using Tests_Renumbering_File = Tests_Renumbering<cugraph::test::File_Usecase>;
using Tests_Renumbering_Rmat = Tests_Renumbering<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_Renumbering_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

// FIXME: add tests for type combinations
TEST_P(Tests_Renumbering_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Renumbering_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      Renumbering_Usecase{}, Renumbering_Usecase{}, Renumbering_Usecase{}, Renumbering_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

#if 0
INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_Renumbering_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      Renumbering_Usecase{}, Renumbering_Usecase{}, Renumbering_Usecase{}, Renumbering_Usecase{}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_large_tests,
  Tests_Renumbering_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Renumbering_Usecase{false},
                      Renumbering_Usecase{false},
                      Renumbering_Usecase{false},
                      Renumbering_Usecase{false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));
#endif

CUGRAPH_TEST_PROGRAM_MAIN()
