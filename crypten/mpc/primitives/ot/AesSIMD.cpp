// Copyright 2004-present Facebook. All Rights Reserved.

#include "deeplearning/projects/crypten/crypten/mpc/primitives/ot/AesSIMD.h"

/*

High-level description of the algorithm(128 bit block, 10 round encryption)

0. Key schedule: round keys are derived from the cipher key using the AES key
schedule. AES requires a separate 128-bit round key block for each round plus
one more.

1. Initial round key addition:
  AddRoundKey: each byte of the state is combined with a byte of the round key
using bitwise xor.

2. 9 rounds:
  Each round consists of SubBytes, ShiftRows, MixColumns and AddRoundKey.
  _mm_aesenc_si128()/_mm_aesdec_si128() perform all these operations in
encryption/decryption.

3. last round:
  The last round of SubBytes, ShiftRows and AddRoundKey.
  _mm_aesenclast_si128()/_mm_aesdeclast_si128() perform all these operations in
encryption/decryption.
*/

// by default, we use 128-bit aes with 10 rounds.
// The following key schedule code is extracted from Intel documentation.
// see details at
// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
// page 25

inline __m128i aes128Assist(__m128i temp1, __m128i temp2) {
  __m128i temp3;
  temp2 = _mm_shuffle_epi32(temp2, 0xff);
  temp3 = _mm_slli_si128(temp1, 0x4);
  temp1 = _mm_xor_si128(temp1, temp3);
  temp3 = _mm_slli_si128(temp3, 0x4);
  temp1 = _mm_xor_si128(temp1, temp3);
  temp3 = _mm_slli_si128(temp3, 0x4);
  temp1 = _mm_xor_si128(temp1, temp3);
  temp1 = _mm_xor_si128(temp1, temp2);
  return temp1;
}

void aesSetEncryptKey(const __m128i& userkey, AES_KEY* key) {
  __m128i temp1, temp2;
  __m128i* kp = key->rd_key;
  kp[0] = temp1 = userkey;

  temp2 = _mm_aeskeygenassist_si128(temp1, 0x1);
  temp1 = aes128Assist(temp1, temp2);
  kp[1] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x2);
  temp1 = aes128Assist(temp1, temp2);
  kp[2] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x4);
  temp1 = aes128Assist(temp1, temp2);
  kp[3] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x8);
  temp1 = aes128Assist(temp1, temp2);
  kp[4] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x10);
  temp1 = aes128Assist(temp1, temp2);
  kp[5] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x20);
  temp1 = aes128Assist(temp1, temp2);
  kp[6] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x40);
  temp1 = aes128Assist(temp1, temp2);
  kp[7] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x80);
  temp1 = aes128Assist(temp1, temp2);
  kp[8] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x1b);
  temp1 = aes128Assist(temp1, temp2);
  kp[9] = temp1;
  temp2 = _mm_aeskeygenassist_si128(temp1, 0x36);
  temp1 = aes128Assist(temp1, temp2);
  kp[10] = temp1;
  key->rounds = 10;
}

// encryption scheme consists of 10 rounds
void aesEcbEncryptBlks(__m128i* blks, unsigned int nblks, const AES_KEY* key) {
  // Initial round key addition
  for (int i = 0; i < nblks; ++i) {
    blks[i] = _mm_xor_si128(blks[i], key->rd_key[0]);
  }

  // normal aes rounds
  for (int j = 1; j < key->rounds; ++j) {
    for (int i = 0; i < nblks; ++i) {
      blks[i] = _mm_aesenc_si128(blks[i], key->rd_key[j]);
    }
  }

  // last round
  for (int i = 0; i < nblks; ++i) {
    blks[i] = _mm_aesenclast_si128(blks[i], key->rd_key[key->rounds]);
  }
}

// generate a scheduled decryption key from a scheduled encryption key. The
// following code is extract from Intel document.
// see details at
// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
// page 48-49
void aesSetDecryptKey_fast(AES_KEY* dkey, const AES_KEY* ekey) {
  int j = 0;
  int i = ekey->rounds;
  dkey->rounds = ekey->rounds;
  dkey->rd_key[i--] = ekey->rd_key[j++];
  while (i > 0) {
    dkey->rd_key[i--] = _mm_aesimc_si128(ekey->rd_key[j++]);
  }
  dkey->rd_key[i] = ekey->rd_key[j];
}

void aesSetDecryptKey(const __m128i& userkey, AES_KEY* key) {
  AES_KEY tmpKey;
  aesSetEncryptKey(userkey, &tmpKey);
  aesSetDecryptKey_fast(key, &tmpKey);
}

void aesEcbDecryptBlks(__m128i* blks, unsigned nblks, const AES_KEY* key) {
  // Initial round key addition
  for (int i = 0; i < nblks; ++i) {
    blks[i] = _mm_xor_si128(blks[i], key->rd_key[0]);
  }

  // normal aes rounds
  for (int j = 1; j < key->rounds; ++j) {
    for (int i = 0; i < nblks; ++i) {
      blks[i] = _mm_aesdec_si128(blks[i], key->rd_key[j]);
    }
  }

  // last round
  for (int i = 0; i < nblks; ++i) {
    blks[i] = _mm_aesdeclast_si128(blks[i], key->rd_key[key->rounds]);
  }
}

// setup the Public fixed key
prgFromFixedKeyAES::prgFromFixedKeyAES(__m128i masterKey) {
  aesSetEncryptKey(masterKey, &aesKey_);
}

// setup the private PRG key
void prgFromFixedKeyAES::setPrgKey(__m128i* input, const size_t size) const {
  aesEcbEncryptBlks(input, size, &aesKey_);
}

void prgFromFixedKeyAES::expand(
    __m128i* dst,
    const __m128i& prgKey,
    const uint32_t index,
    const size_t size) const {
  for (uint64_t i = 0; i < size; i++) {
    dst[i] = _mm_xor_si128(prgKey, _mm_set_epi32(0, 0, 0, i + index));
  }
  aesEcbEncryptBlks(dst, size, &aesKey_);
  for (uint64_t i = 0; i < size; i++) {
    dst[i] = _mm_xor_si128(prgKey, dst[i]);
  }
}

// any arbitrary value is fine.
static const __m128i kMasterKey = _mm_set_epi32(0xaa, 0xbb, 0xcc, 0xdd);

prgFromFixedKeyAES aesPRG(kMasterKey);

keyedPRG::keyedPRG(__m128i key, size_t bufferSize)
    : bufferSize_(bufferSize * sizeof(__m128i)) {
  key_ = key;
  aesPRG.setPrgKey(&key, 1);
  buffer_ = new __m128i[bufferSize_ / sizeof(__m128i)];
  pointer_ = (unsigned char*)buffer_;
}

keyedPRG::~keyedPRG() {
  delete[] buffer_;
}

void keyedPRG::getRandomBytes(unsigned char* dst, size_t size) {
  if (bufferSize_ >= bufferIndex_ + size) {
    memcpy(dst, &pointer_[bufferIndex_], size);
    bufferIndex_ += size;
    if (bufferIndex_ >= bufferSize_) {
      generateRandomBytesInBuffer();
    }
  } else {
    auto tmp = bufferSize_ - bufferIndex_;
    memcpy(dst, &pointer_[bufferIndex_], tmp);
    generateRandomBytesInBuffer();
    getRandomBytes(&dst[tmp], size - tmp);
  }
}

void keyedPRG::generateRandomBytesInBuffer() {
  aesPRG.expand(buffer_, key_, prgIndex_, bufferSize_ / sizeof(__m128i));
  prgIndex_ += bufferSize_ / sizeof(__m128i);
  bufferIndex_ = 0;
}
