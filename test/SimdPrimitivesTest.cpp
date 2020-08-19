// Copyright 2004-present Facebook. All Rights Reserved.

#include <gtest/gtest.h>
#include <smmintrin.h>
#include <array>
#include <random>
#include "deeplearning/projects/crypten/crypten/mpc/primitives/ot/AesSIMD.h"
using namespace ::testing;

TEST(simdPrimitivesTest, aesKey) {
  std::random_device rd;
  std::mt19937_64 e(rd());
  std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
  AES_KEY encKey;
  AES_KEY decKey;
  __m128i key = _mm_set_epi32(dist(e), dist(e), dist(e), dist(e));
  __m128i msg0 = _mm_set_epi32(dist(e), dist(e), dist(e), dist(e));
  auto msg = msg0;
  // encKey set here
  aesSetEncryptKey(key, &encKey);
  // decKey set here
  aesSetDecryptKey(key, &decKey);

  // encKey used here
  aesEcbEncryptBlks(&msg, 1, &encKey);
  auto enc = msg;
  // decKey used here
  aesEcbDecryptBlks(&enc, 1, &decKey);
  msg = _mm_xor_si128(msg0, enc);
  __m128i allOneMask =
      _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
  int rst = _mm_testz_si128(allOneMask, msg);
  ASSERT_EQ(rst, 1);
}

TEST(simdPrimitivesTest, keyedPRGTest) {
  std::random_device rd;
  std::mt19937_64 e(rd());
  std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
  AES_KEY encKey;
  AES_KEY decKey;
  __m128i key = _mm_set_epi32(dist(e), dist(e), dist(e), dist(e));
  keyedPRG prg(key, 16);
  std::array<unsigned char, 1024> tmp;
  tmp[512] = 1;
  prg.getRandomBytes(tmp.data(), 512);
  // make sure generate exactly 512 byte randomness
  ASSERT_EQ(tmp[512], 1);
}
