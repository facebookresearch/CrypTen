// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include <emmintrin.h>
#include <stdint.h>
#include <string.h>
#include <wmmintrin.h>
#include <xmmintrin.h>

/*
AES algorithm can be divided into two parts:
1. key schedule
2. encryption/decryption

A scheduled key can be repeatedly used as the origial key doesn' change.
Scheduling a key takes a significant portion of the total workload in
AES cipher. Therefore it would significantly improve the performance if
key scheduling can be amortized.
*/

/*
Existing SIMD instructions treat every __m128i register as an AES block.
As shown on Inte website
(https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=mm_aes&expand=2428,5304,5944,238,230)
AES encryption/decryption rounds can be executed in a signal instruction:
  * __m128i _mm_aesenc_si128 (__m128i a, __m128i RoundKey)
    Perform one round of an AES encryption flow on data (state) in a using the
round key in RoundKey, and store the result in dst.

  * __m128i _mm_aesenclast_si128 (__m128i a, __m128i RoundKey)
    Perform the last round of an AES encryption flow on data (state) in a using
the round key in RoundKey, and store the result in dst.

  * __m128i _mm_aesdec_si128 (__m128i a, __m128i RoundKey)
    Perform one round of an AES decryption flow on data (state) in a using the
round key in RoundKey, and store the result in dst.

  * __m128i _mm_aesdeclast_si128 (__m128i a, __m128i RoundKey)
    Perform the last round of an AES decryption flow on data (state) in a using
the round key in RoundKey, and store the result in dst.
*/

typedef struct {
  __m128i rd_key[11];
  unsigned int rounds;
} AES_KEY;

void aesSetEncryptKey(const __m128i& userkey, AES_KEY* key);

void aesEcbEncryptBlks(__m128i* blks, unsigned int nblks, const AES_KEY* key);

void aesSetDecryptKey_fast(AES_KEY* dkey, const AES_KEY* ekey);

void aesSetDecryptKey(const __m128i& userkey, AES_KEY* key);

void aesEcbDecryptBlks(__m128i* blks, unsigned nblks, const AES_KEY* key);

/*
a PRG from fixed key AES, intended to expand input with given indexes.
a fixed-key AES is viewed as a mapping \pi(x): 2^{128} -> 2^{128}

as suggested by https://eprint.iacr.org/2019/074.pdf, a robust hash function
can be realized by \pi(\pi(x) \xor i) xor \pi(x) if x is indistinguishable
from uniform random distributions. We can use this to construct a PRG:
     PRG(i):=\pi(\pi(key) \xor i) xor \pi(x)
*/

class prgFromFixedKeyAES {
 public:
  explicit prgFromFixedKeyAES(__m128i masterKey);

  // prepare the key of this PRG by applying \pi on each input in place.
  void setPrgKey(__m128i* input, const size_t size) const;

  // given a seed s, generates the corresponding randomness;
  void expand(
      __m128i* dst,
      const __m128i& prgKey,
      const uint32_t index,
      const size_t size) const;

 private:
  AES_KEY aesKey_;
};

extern prgFromFixedKeyAES aesPRG;

class keyedPRG {
 public:
  keyedPRG(__m128i key, size_t bufferSize);
  ~keyedPRG();
  void getRandomBytes(unsigned char* dst, size_t size);

 private:
  __m128i key_;
  int bufferIndex_ = 0;
  uint32_t prgIndex_ = 0;
  const size_t bufferSize_;
  __m128i* buffer_;
  unsigned char* pointer_;
  void generateRandomBytesInBuffer();
};
