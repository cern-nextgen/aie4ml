// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once

// How a kernel's accumulators start, and how its sums leave: from the bias (or zero) or from the cascade; stored as the
// layer's codes or sent down the cascade. A chain's kernels differ in these only: single (bias, store), first (bias,
// cascade), middle (cascade, cascade), last (cascade, store).
enum class dense_in { bias, cascade };
enum class dense_out { store, cascade };
