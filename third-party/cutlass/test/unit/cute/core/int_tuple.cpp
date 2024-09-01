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

#include "cutlass_unit_test.h"

#include <cute/layout.hpp>

TEST(CuTe_core, WeaklyCongruent)
{
  using namespace cute;

  auto a = _1{};
  auto b = _2{};
  EXPECT_TRUE (weakly_congruent(a, a));
  EXPECT_TRUE (weakly_congruent(b, b));
  EXPECT_TRUE (weakly_congruent(a, b));

  auto a0 = Shape<_1>{};
  auto b0 = Shape<_2>{};
  EXPECT_TRUE (weakly_congruent(a , a0));
  EXPECT_TRUE (weakly_congruent(b , b0));
  EXPECT_TRUE (weakly_congruent(a , b0));
  EXPECT_TRUE (weakly_congruent(b , a0));
  EXPECT_FALSE(weakly_congruent(a0, a ));
  EXPECT_FALSE(weakly_congruent(b0, b ));
  EXPECT_FALSE(weakly_congruent(a0, b ));
  EXPECT_FALSE(weakly_congruent(b0, a ));
  EXPECT_TRUE (weakly_congruent(a0, a0));
  EXPECT_TRUE (weakly_congruent(b0, b0));
  EXPECT_TRUE (weakly_congruent(a0, b0));
  
  auto a1 = Shape<_1, _1>{};
  EXPECT_TRUE (weakly_congruent(a , a1));
  EXPECT_FALSE(weakly_congruent(a0, a1));
  EXPECT_TRUE (weakly_congruent(a1, a1));

  auto a2 = Shape<_1, Shape<_1,_1>>{};
  EXPECT_TRUE (weakly_congruent(a , a2));
  EXPECT_FALSE(weakly_congruent(a0, a2));
  EXPECT_TRUE (weakly_congruent(a1, a2));

  auto b1 = Shape<_2, _2>{};
  EXPECT_TRUE (weakly_congruent(b , b1));
  EXPECT_FALSE(weakly_congruent(b0, b1));
  EXPECT_TRUE (weakly_congruent(a1, b1));

  auto b2 = Shape<_2, Shape<_2,_2>>{};
  EXPECT_FALSE(weakly_congruent(a2, b0));
  EXPECT_FALSE(weakly_congruent(a2, a1));
  EXPECT_TRUE (weakly_congruent(a2, b2));

  auto b3 = Shape<Shape<_2,_2>, Shape<_2,_2>>{};
  EXPECT_FALSE(weakly_congruent(a0, b3));
  EXPECT_TRUE (weakly_congruent(a1, b3));
  EXPECT_TRUE (weakly_congruent(a2, b3));
}

TEST(CuTe_core, WeaklyCompatible)
{
  using namespace cute;

  auto a = _16{};
  auto b = _12{};
  auto c = _8{};
  EXPECT_TRUE (weakly_compatible(a, a));
  EXPECT_TRUE (weakly_compatible(b, b));
  EXPECT_TRUE (weakly_compatible(c, c));
  EXPECT_FALSE(weakly_compatible(a, b)); 
  EXPECT_FALSE(weakly_compatible(a, c));
  EXPECT_TRUE (weakly_compatible(c, a));

  auto a0 = Shape<_16>{};
  EXPECT_TRUE (weakly_compatible(a0, a0));
  EXPECT_TRUE (weakly_compatible(a , a0));
  EXPECT_FALSE(weakly_compatible(a0, a ));
  EXPECT_TRUE (weakly_compatible(c , a0));
  EXPECT_FALSE(weakly_compatible(a0, c ));  
  EXPECT_FALSE(weakly_compatible(b , a0));
  EXPECT_FALSE(weakly_compatible(a0, b ));  

  auto a1 = Shape<_2,_8>{};
  EXPECT_TRUE (weakly_compatible(a1, a1));
  EXPECT_TRUE (weakly_compatible(a , a1));
  EXPECT_FALSE(weakly_compatible(a0, a1));
  EXPECT_FALSE(weakly_compatible(a1, a0));
  EXPECT_TRUE (weakly_compatible(a1, Shape<_2,Shape<_2,_4>>{}));

  auto a2 = Shape<Shape<_2,_8>>{};
  EXPECT_TRUE (weakly_compatible(a2, a2));
  EXPECT_TRUE (weakly_compatible(a , a2));
  EXPECT_TRUE (weakly_compatible(c , a2));
  EXPECT_TRUE (weakly_compatible(a0, a2));
  EXPECT_FALSE(weakly_compatible(a2, a0));

  auto a3 = Shape<Shape<_2,Shape<_4,_2>>>{};
  EXPECT_TRUE (weakly_compatible(a3, a3));
  EXPECT_TRUE (weakly_compatible(a , a3));
  EXPECT_TRUE (weakly_compatible(c , a3));
  EXPECT_TRUE (weakly_compatible(a0, a3));
  EXPECT_FALSE(weakly_compatible(a3, a0));
  EXPECT_TRUE (weakly_compatible(a2, a3));
  EXPECT_FALSE(weakly_compatible(a3, a2));
}
