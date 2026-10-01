// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "dense_bias_relu_stream.h"
#include <type_traits>
using namespace adf;

template<typename ConfigT>
dense_stream_base<ConfigT>::dense_stream_base() {
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);

  static_assert(
        ConfigT::OUT_FEAT_SLICE * ConfigT::IN_FEAT_SLICE * sizeof(weight_t) <= ConfigT::BANK_BYTES,
        "Weight size per tile must not exceed one device memory bank");
  static_assert(
        ConfigT::IN_FEAT_SLICE % (2 * ConfigT::K) == 0,
        "IN_FEAT_SLICE must be divisible by 2*K");
  static_assert(
        ConfigT::OUT_FEAT_SLICE % (2 * ConfigT::N) == 0,
        "OUT_FEAT_SLICE must be divisible by 2*N");
  static_assert(
        ConfigT::padded_independent_extent % (2 * ConfigT::M) == 0,
        "padded_independent_extent must be divisible by 2*M");
  static_assert(
        ConfigT::padded_IN_FEAT == ConfigT::IN_FEAT_SLICE * ConfigT::CAS_LENGTH,
        "padded_IN_FEAT must equal IN_FEAT_SLICE * CAS_LENGTH");
  static_assert(
        ConfigT::PARALLELISM_CONTRACT_OUTER
            ? (ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE)
            : (ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE * ConfigT::CAS_NUM),
        "padded_OUT_FEAT must equal OUT_FEAT_SLICE * CAS_NUM ('inner') or OUT_FEAT_SLICE ('outer')");
  static_assert(!ConfigT::TRANSPOSE_INPUT, "a stream carries the LHS in linear order; it cannot be transposed");
}

template<int M, typename VT, int N>
struct stream_row_replicator;

template<typename VT, int N>
struct stream_row_replicator<2, VT, N> {
  static inline aie::vector<VT, 2 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row);
  }
};

template<typename VT, int N>
struct stream_row_replicator<4, VT, N> {
  static inline aie::vector<VT, 4 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row, row, row);
  }
};

template<typename VT, int N>
struct stream_row_replicator<8, VT, N> {
  static inline aie::vector<VT, 8 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row, row, row, row, row, row, row);
  }
};

template<typename ConfigT>
struct dense_stream_traits {
  using data_t       = typename ConfigT::data_t;
  using weight_t     = typename ConfigT::weight_t;
  using bias_t       = typename ConfigT::bias_t;
  using result_t     = typename ConfigT::result_t;
  using acc_scalar_t = typename ConfigT::acc_scalar_t;

  static constexpr int M     = ConfigT::M;
  static constexpr int K     = ConfigT::K;
  static constexpr int N     = ConfigT::N;
  static constexpr int colA  = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB  = ConfigT::OUT_FEAT_SLICE;
  static constexpr int rowA  = ConfigT::padded_independent_extent;
  static constexpr int SHIFT = ConfigT::SHIFT;
  // Two row groups per band: the 2x2 register blocking of the core.
  static constexpr int BAND_ROWS = 2 * M;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;

  // Streams move 128-bit chunks; a microtile row is either whole chunks or half of one.
  static constexpr int CHUNK_BYTES = 16;
  static constexpr int CHUNK_A = CHUNK_BYTES / sizeof(data_t);
  static constexpr int CHUNK_C = CHUNK_BYTES / sizeof(result_t);
  static constexpr int ROW_A_BYTES = K * sizeof(data_t);
  static constexpr int ROW_C_BYTES = N * sizeof(result_t);
};

// ---------------------------------------------------------------------------
// Stream I/O. A band is BAND_ROWS rows; it is received "raw" (linear rows, as the
// stream delivers them), re-tiled into the microtile-major layout the core reads, and
// its result is emitted as linear rows. The raw reads of the next band and the row
// writes of the previous band are folded into the core's pipelined loop so that the
// 32-bit stream ports work while the mmul unit does.
// ---------------------------------------------------------------------------

static constexpr unsigned umin(unsigned a, unsigned b) { return a < b ? a : b; }

template<typename ConfigT>
static inline void read_chunks(input_stream<typename ConfigT::data_t>* __restrict in,
                               typename ConfigT::data_t* __restrict raw, unsigned first, unsigned count)
{
  constexpr int CHUNK = dense_stream_traits<ConfigT>::CHUNK_A;
  for (unsigned c = first; c < first + count; ++c)
    chess_prepare_for_pipelining
  {
    aie::store_v(raw + c * CHUNK, readincr_v<CHUNK>(in));
  }
}

// The last chunk of an inference carries TLAST, framing the output tile like a DMA would.
template<typename ConfigT>
static inline void write_chunks(output_stream<typename ConfigT::result_t>* __restrict out,
                                const typename ConfigT::result_t* __restrict band, unsigned first, unsigned count,
                                bool last_of_inference)
{
  constexpr int CHUNK = dense_stream_traits<ConfigT>::CHUNK_C;
  for (unsigned c = first; c + 1 < first + count; ++c)
    chess_prepare_for_pipelining
  {
    writeincr(out, aie::load_v<CHUNK>(band + c * CHUNK));
  }
  if (count) {
    writeincr(out, aie::load_v<CHUNK>(band + (first + count - 1) * CHUNK), last_of_inference);
  }
}

// Raw linear rows -> microtile-major band: (row group g, tile i, row m) at ((g * colA/K + i) * M + m) * K.
template<typename ConfigT>
static inline void retile_band(const typename ConfigT::data_t* __restrict raw,
                               typename ConfigT::data_t* __restrict band)
{
  using T = dense_stream_traits<ConfigT>;
  using data_t = typename T::data_t;
  using MMUL = typename T::MMUL;
  constexpr int M = T::M, K = T::K, colA = T::colA, CHUNK = T::CHUNK_A;
  constexpr int tiles_per_group = colA / K;

  if constexpr (T::ROW_A_BYTES >= T::CHUNK_BYTES) {
    // A tile row is whole chunks: a straight copy into place.
    constexpr int chunks_per_row = T::ROW_A_BYTES / T::CHUNK_BYTES;
    for (unsigned r = 0; r < T::BAND_ROWS; ++r) {
      const data_t* src = raw + r * colA;
      data_t* row = band + (r / M) * tiles_per_group * MMUL::size_A + (r % M) * K;
      for (unsigned i = 0; i < tiles_per_group; ++i)
        chess_prepare_for_pipelining
      {
        for (unsigned t = 0; t < chunks_per_row; ++t)
          aie::store_v(row + i * MMUL::size_A + t * CHUNK, aie::load_v<CHUNK>(src + i * K + t * CHUNK));
      }
    }
  } else {
    // A tile row is half a chunk: zip two consecutive rows on 32-bit lanes so each chunk
    // pair becomes the row pair of two neighbouring tiles.
    static_assert(T::ROW_A_BYTES * 2 == T::CHUNK_BYTES && M % 2 == 0,
                  "stream staging needs a microtile row of 16 bytes, or 8 bytes with an even M");
    constexpr int chunks_per_row = colA / CHUNK;
    for (unsigned g = 0; g < T::BAND_ROWS / M; ++g) {
      data_t* group = band + g * tiles_per_group * MMUL::size_A;
      for (unsigned p = 0; p < M / 2; ++p) {
        const data_t* even_row = raw + (g * M + 2 * p) * colA;
        const data_t* odd_row  = even_row + colA;
        for (unsigned c = 0; c < chunks_per_row; ++c)
          chess_prepare_for_pipelining
        {
          auto even = aie::load_v<CHUNK>(even_row + c * CHUNK).template cast_to<int32>();
          auto odd  = aie::load_v<CHUNK>(odd_row + c * CHUNK).template cast_to<int32>();
          auto zipped = aie::interleave_zip(even, odd, T::ROW_A_BYTES / 4);
          data_t* tile = group + 2 * c * MMUL::size_A + 2 * p * K;
          aie::store_v(tile, zipped.first.template cast_to<data_t>());
          aie::store_v(tile + MMUL::size_A, zipped.second.template cast_to<data_t>());
        }
      }
    }
  }
}

// One finished tile pair -> rows of a colB-wide band.
template<typename ConfigT, typename MMUL>
static inline aie::vector<typename ConfigT::result_t, ConfigT::M * ConfigT::N>
finalize_tile(MMUL& C) {
  using result_t = typename ConfigT::result_t;
  auto v = C.template to_vector<result_t>(ConfigT::SHIFT);
  if constexpr (ConfigT::USE_RELU) {
    return aie::max(v, result_t(0));
  } else {
    return v;
  }
}

template<typename ConfigT>
static inline void store_tile_pair(typename ConfigT::result_t* __restrict band, int group, unsigned j,
                                   typename dense_stream_traits<ConfigT>::MMUL& C0,
                                   typename dense_stream_traits<ConfigT>::MMUL& C1)
{
  using T = dense_stream_traits<ConfigT>;
  using result_t = typename T::result_t;
  constexpr int M = T::M, N = T::N, colB = T::colB, CHUNK = T::CHUNK_C;
  result_t* rows = band + group * M * colB + j * N;
  auto v0 = finalize_tile<ConfigT>(C0);
  auto v1 = finalize_tile<ConfigT>(C1);
  if constexpr (T::ROW_C_BYTES >= T::CHUNK_BYTES) {
    constexpr int chunks_per_row = T::ROW_C_BYTES / T::CHUNK_BYTES;
    for (unsigned m = 0; m < M; ++m) {
      for (unsigned t = 0; t < chunks_per_row; ++t) {
        aie::store_v(rows + m * colB + t * CHUNK, v0.template extract<CHUNK>(m * chunks_per_row + t));
        aie::store_v(rows + m * colB + N + t * CHUNK, v1.template extract<CHUNK>(m * chunks_per_row + t));
      }
    }
  } else {
    static_assert(T::ROW_C_BYTES * 2 == T::CHUNK_BYTES && M % 2 == 0,
                  "stream emission needs a microtile row of 16 bytes, or 8 bytes with an even M");
    // Zipping the two tiles row by row yields rows of 2*N elements: one chunk each.
    auto zipped = aie::interleave_zip(v0.template cast_to<int32>(), v1.template cast_to<int32>(), T::ROW_C_BYTES / 4);
    auto lo = zipped.first.template cast_to<result_t>();
    auto hi = zipped.second.template cast_to<result_t>();
    for (unsigned m = 0; m < M / 2; ++m) {
      aie::store_v(rows + m * colB, lo.template extract<CHUNK>(m));
      aie::store_v(rows + (m + M / 2) * colB, hi.template extract<CHUNK>(m));
    }
  }
}

// One finished 2x2 block of tiles -> the band's microtiles in an output buffer, (row group g, tile j) at
// (g * colB/N + j) * size_C.
template<typename ConfigT, typename MMUL>
static inline void store_tiles(typename ConfigT::result_t* __restrict tiles, unsigned j,
                               MMUL& C00, MMUL& C01, MMUL& C10, MMUL& C11)
{
  constexpr int GROUP = ConfigT::OUT_FEAT_SLICE / ConfigT::N * MMUL::size_C;
  aie::store_v(tiles + j * MMUL::size_C, finalize_tile<ConfigT>(C00));
  aie::store_v(tiles + (j + 1) * MMUL::size_C, finalize_tile<ConfigT>(C01));
  aie::store_v(tiles + GROUP + j * MMUL::size_C, finalize_tile<ConfigT>(C10));
  aie::store_v(tiles + GROUP + (j + 1) * MMUL::size_C, finalize_tile<ConfigT>(C11));
}

// ---------------------------------------------------------------------------
// Band core: the 2x2 blocked mmul loop of dense_bias_relu.cpp over one band. With
// READ the raw chunks of the next band are read, with WRITE the row chunks of the
// previous band are written, from inside the K loop: a fixed number per tile-pair
// iteration so every loop keeps a constant trip count. With TILED the band's tiles go straight to
// the output buffer `band_c` points into.
// The first kernel of a chain, or a single one, seeds its accumulators with the bias.
// ---------------------------------------------------------------------------
template<typename ConfigT, bool CASC_IN, bool CASC_OUT, bool TILED, bool BIAS, bool READ, bool WRITE>
static inline void dense_stream_band(const typename ConfigT::data_t* __restrict pA,
                                     const typename ConfigT::weight_t* __restrict pB,
                                     const typename ConfigT::bias_t* __restrict pBias,
                                     input_cascade<typename ConfigT::acc_scalar_t>* inCascade,
                                     output_cascade<typename ConfigT::acc_scalar_t>* outCascade,
                                     typename ConfigT::result_t* __restrict band_c,
                                     input_stream<typename ConfigT::data_t>* __restrict in,
                                     typename ConfigT::data_t* __restrict raw_next,
                                     output_stream<typename ConfigT::result_t>* __restrict out,
                                     const typename ConfigT::result_t* __restrict prev_c)
{
  using T = dense_stream_traits<ConfigT>;
  using data_t   = typename T::data_t;
  using weight_t = typename T::weight_t;
  using bias_t   = typename T::bias_t;
  using MMUL     = typename T::MMUL;
  constexpr int M = T::M, K = T::K, N = T::N, colA = T::colA, colB = T::colB;
  constexpr bool BIAS_INIT = BIAS && !CASC_IN;
  constexpr unsigned CHUNK_A = T::CHUNK_A, CHUNK_C = T::CHUNK_C;
  constexpr unsigned NJ = colB / (2 * N);      // tile-pair iterations per band
  constexpr unsigned NI = colA / K - 1;        // pipelined K steps per tile pair (first step peeled)
  constexpr unsigned CHUNKS_IN  = READ  ? T::BAND_ROWS * colA / CHUNK_A : 0;
  constexpr unsigned CHUNKS_OUT = WRITE ? T::BAND_ROWS * colB / CHUNK_C : 0;
  // Each tile pair carries an even share of the band's I/O: RQ reads and WQ writes, one per
  // K step, in short constant-count loops (measured faster than folding the I/O into the
  // full K loop on AIE); chunks left over go in short tail loops.
  constexpr unsigned RQ = umin(CHUNKS_IN / NJ, NI), WQ = umin(CHUNKS_OUT / NJ, NI);
  constexpr unsigned N_BOTH = umin(RQ, WQ), N_READ = RQ - N_BOTH, N_WRITE = WQ - N_BOTH;
  constexpr unsigned N_ONLY = NI - N_BOTH - N_READ - N_WRITE;
  constexpr unsigned READ_TAIL = CHUNKS_IN - RQ * NJ, WRITE_TAIL = CHUNKS_OUT - WQ * NJ;
  unsigned rd = 0, wr = 0;

  auto read_one = [&]() {
    if constexpr (READ) { aie::store_v(raw_next + (rd++) * CHUNK_A, readincr_v<CHUNK_A>(in)); }
  };
  auto write_one = [&]() {
    if constexpr (WRITE) { writeincr(out, aie::load_v<CHUNK_C>(prev_c + (wr++) * CHUNK_C)); }
  };

  for (unsigned j = 0; j < colB / N; j += 2) {
    const data_t*   __restrict pA1 = pA;
    const data_t*   __restrict pA2 = pA + (colA / K) * MMUL::size_A;
    const weight_t* __restrict pB1 = pB + j * MMUL::size_B;
    const weight_t* __restrict pB2 = pB + (j + 1) * MMUL::size_B;

    MMUL C00, C01, C10, C11;
    if constexpr (CASC_IN) {
      C00 = readincr_v<MMUL::size_C>(inCascade);
      C01 = readincr_v<MMUL::size_C>(inCascade);
      C10 = readincr_v<MMUL::size_C>(inCascade);
      C11 = readincr_v<MMUL::size_C>(inCascade);
    } else if constexpr (BIAS_INIT) {
      auto bias_block_0 = stream_row_replicator<M, bias_t, N>::run(aie::load_v<N>(pBias + j * N));
      auto bias_block_1 = stream_row_replicator<M, bias_t, N>::run(aie::load_v<N>(pBias + (j + 1) * N));
      C00 = bias_block_0; C01 = bias_block_1;
      C10 = bias_block_0; C11 = bias_block_1;
    }

    aie::vector<data_t, MMUL::size_A> A0, A1;
    aie::vector<weight_t, MMUL::size_B> B0, B1;
    auto load_step = [&]() {
      A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
      A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
      B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * (colB / N);
      B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * (colB / N);
    };
    auto mac_step = [&]() {
      load_step();
      C00.mac(A0, B0); C01.mac(A0, B1);
      C10.mac(A1, B0); C11.mac(A1, B1);
    };

    load_step();
    if constexpr (CASC_IN || BIAS_INIT) {
      C00.mac(A0, B0); C01.mac(A0, B1);
      C10.mac(A1, B0); C11.mac(A1, B1);
    } else {
      C00.mul(A0, B0); C01.mul(A0, B1);
      C10.mul(A1, B0); C11.mul(A1, B1);
    }

    for (unsigned i = 0; i < N_BOTH; ++i)
      chess_prepare_for_pipelining
    {
      mac_step(); read_one(); write_one();
    }
    for (unsigned i = 0; i < N_READ; ++i)
      chess_prepare_for_pipelining
    {
      mac_step(); read_one();
    }
    for (unsigned i = 0; i < N_WRITE; ++i)
      chess_prepare_for_pipelining
    {
      mac_step(); write_one();
    }
    for (unsigned i = 0; i < N_ONLY; ++i)
      chess_prepare_for_pipelining
    {
      mac_step();
    }

    if constexpr (CASC_OUT) {
      writeincr(outCascade, C00.to_accum());
      writeincr(outCascade, C01.to_accum());
      writeincr(outCascade, C10.to_accum());
      writeincr(outCascade, C11.to_accum());
    } else if constexpr (TILED) {
      store_tiles<ConfigT>(band_c, j, C00, C01, C10, C11);
    } else {
      store_tile_pair<ConfigT>(band_c, 0, j, C00, C01);
      store_tile_pair<ConfigT>(band_c, 1, j, C10, C11);
    }
  }

  for (unsigned i = 0; i < READ_TAIL; ++i)
    chess_prepare_for_pipelining
  {
    read_one();
  }
  for (unsigned i = 0; i < WRITE_TAIL; ++i)
    chess_prepare_for_pipelining
  {
    write_one();
  }
}

// Bands of one inference: the first band is read up front and the last band is emitted
// at the end; every other read and write overlaps a neighbouring band's compute. Without
// STREAM_A the bands are read in place from the microtiles `a_tiles`; with TILED they are
// stored in `tiles` instead of emitted.
template<typename ConfigT, bool STREAM_A, bool CASC_IN, bool CASC_OUT, bool TILED, bool BIAS>
static inline void dense_stream_rows(input_stream<typename ConfigT::data_t>* __restrict in,
                                     const typename ConfigT::data_t* __restrict a_tiles,
                                     const typename ConfigT::weight_t* __restrict pB,
                                     const typename ConfigT::bias_t* __restrict pBias,
                                     input_cascade<typename ConfigT::acc_scalar_t>* inCascade,
                                     output_cascade<typename ConfigT::acc_scalar_t>* outCascade,
                                     output_stream<typename ConfigT::result_t>* __restrict out,
                                     typename ConfigT::result_t* __restrict tiles)
{
  using T = dense_stream_traits<ConfigT>;
  constexpr unsigned NB = T::rowA / T::BAND_ROWS;
  constexpr unsigned CHUNKS_IN = T::BAND_ROWS * T::colA / T::CHUNK_A;
  constexpr unsigned CHUNKS_OUT = T::BAND_ROWS * T::colB / T::CHUNK_C;
  constexpr unsigned A_BAND = STREAM_A ? T::BAND_ROWS * T::colA : T::CHUNK_A;
  alignas(32) static typename T::data_t   raw[2][A_BAND];
  alignas(32) static typename T::data_t   band_a[A_BAND];
  constexpr bool EMIT = !CASC_OUT && !TILED;
  alignas(32) static typename T::result_t band_c[2][EMIT ? T::BAND_ROWS * T::colB : T::CHUNK_C];

  auto band_at = [&](unsigned band, auto read, auto write) {
    if constexpr (STREAM_A) retile_band<ConfigT>(raw[band & 1], band_a);
    dense_stream_band<ConfigT, CASC_IN, CASC_OUT, TILED, BIAS, STREAM_A && decltype(read)::value,
                      EMIT && decltype(write)::value>(
        STREAM_A ? band_a : a_tiles + band * T::BAND_ROWS * T::colA, pB, pBias, inCascade, outCascade,
        TILED ? tiles + band * T::BAND_ROWS * T::colB : band_c[band & 1],
        in, raw[(band + 1) & 1], out, band_c[(band + 1) & 1]);
  };

  if constexpr (STREAM_A) read_chunks<ConfigT>(in, raw[0], 0, CHUNKS_IN);
  if constexpr (NB == 1) {
    band_at(0, std::false_type{}, std::false_type{});
  } else {
    band_at(0, std::true_type{}, std::false_type{});
    for (unsigned band = 1; band + 1 < NB; ++band) {
      band_at(band, std::true_type{}, std::true_type{});
    }
    band_at(NB - 1, std::false_type{}, std::true_type{});
  }
  if constexpr (EMIT) {
    write_chunks<ConfigT>(out, band_c[(NB - 1) & 1], 0, CHUNKS_OUT, true);
  }
}

// ---------------------------------------------------------------------------
// Kernel entry points
// ---------------------------------------------------------------------------

template<typename ConfigT>
void dense_single_stream<ConfigT>::run(input_stream<data_t>* ifm,
                                       const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                       const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                       output_stream<result_t>* ofm)
{
  dense_stream_rows<ConfigT, true, false, false, false, ConfigT::USE_BIAS>(
      ifm, nullptr, wts, bias, nullptr, nullptr, ofm, nullptr);
}

template<typename ConfigT>
void dense_first_stream<ConfigT>::run(input_stream<data_t>* ifm,
                                      const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                      const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                      output_cascade<acc_scalar_t>* outCascade)
{
  dense_stream_rows<ConfigT, true, false, true, false, ConfigT::USE_BIAS>(
      ifm, nullptr, wts, bias, nullptr, outCascade, nullptr, nullptr);
}

template<typename ConfigT>
void dense_middle_stream<ConfigT>::run(input_stream<data_t>* ifm,
                                       const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                       input_cascade<acc_scalar_t>* inCascade,
                                       output_cascade<acc_scalar_t>* outCascade)
{
  dense_stream_rows<ConfigT, true, true, true, false, false>(
      ifm, nullptr, wts, nullptr, inCascade, outCascade, nullptr, nullptr);
}

template<typename ConfigT>
void dense_last_stream<ConfigT>::run(input_stream<data_t>* ifm,
                                     const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                     input_cascade<acc_scalar_t>* inCascade,
                                     output_stream<result_t>* ofm)
{
  dense_stream_rows<ConfigT, true, true, false, false, false>(
      ifm, nullptr, wts, nullptr, inCascade, nullptr, ofm, nullptr);
}

template<typename ConfigT>
void dense_single_stream_in<ConfigT>::run(input_stream<data_t>* ifm,
                                          const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                          const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                          output_buffer<result_t>& ofm)
{
  dense_stream_rows<ConfigT, true, false, false, true, ConfigT::USE_BIAS>(
      ifm, nullptr, wts, bias, nullptr, nullptr, nullptr, ofm.data());
}

template<typename ConfigT>
void dense_last_stream_in<ConfigT>::run(input_stream<data_t>* ifm,
                                        const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                        input_cascade<acc_scalar_t>* inCascade,
                                        output_buffer<result_t>& ofm)
{
  dense_stream_rows<ConfigT, true, true, false, true, false>(
      ifm, nullptr, wts, nullptr, inCascade, nullptr, nullptr, ofm.data());
}

template<typename ConfigT>
void dense_single_stream_out<ConfigT>::run(input_buffer<data_t>& ifm,
                                           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                           const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                           output_stream<result_t>* ofm)
{
  dense_stream_rows<ConfigT, false, false, false, false, ConfigT::USE_BIAS>(
      nullptr, ifm.data(), wts, bias, nullptr, nullptr, ofm, nullptr);
}

template<typename ConfigT>
void dense_last_stream_out<ConfigT>::run(input_buffer<data_t>& ifm,
                                         const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                         input_cascade<acc_scalar_t>* inCascade,
                                         output_stream<result_t>* ofm)
{
  dense_stream_rows<ConfigT, false, true, false, false, false>(
      nullptr, ifm.data(), wts, nullptr, inCascade, nullptr, ofm, nullptr);
}
