// ---------------------------------------------------------------------
//
// Copyright (C) 2013 - 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

// Trilinos ML is non-deterministic up to Trilinos 12.4 and then only when the
// random seed is initialized correctly. This test checks this. Like _01 but
// using our deal.II data structures.

// Note that we need to increase the size of the matrix to be >2000 from
// amg_nondeterministic_01, otherwise the default settings will just apply the
// coarse solver due to the default settings.

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include "../tests.h"



unsigned int mati[] = {
  0,   1,   2,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
  3,   3,   3,   3,   3,   3,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
  5,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
  6,   6,   6,   6,   6,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
  7,   7,   7,   7,   7,   7,   8,   8,   8,   8,   8,   8,   8,   8,   9,   10,  10,  10,  10,
  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  11,  11,  11,  11,  11,  11,  11,  11,  11,
  11,  11,  11,  12,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  14,  14,  14,
  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  15,  15,  15,  15,  15,  15,  15,  15,  15,
  15,  16,  16,  16,  16,  16,  16,  16,  16,  17,  17,  17,  17,  17,  17,  18,  19,  19,  19,
  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,
  19,  19,  19,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,
  20,  20,  20,  20,  20,  20,  20,  20,  20,  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,
  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  22,  22,  22,  22,
  22,  22,  22,  22,  22,  22,  22,  22,  22,  22,  22,  23,  23,  23,  23,  23,  23,  23,  23,
  23,  23,  23,  23,  23,  23,  23,  24,  25,  25,  25,  25,  25,  25,  25,  25,  25,  25,  25,
  25,  25,  25,  25,  26,  26,  26,  26,  26,  26,  26,  26,  26,  26,  26,  26,  26,  26,  26,
  27,  27,  27,  27,  27,  27,  27,  27,  27,  27,  27,  27,  27,  27,  27,  28,  28,  28,  28,
  28,  28,  28,  28,  28,  29,  29,  29,  29,  29,  29,  29,  29,  29,  30,  31,  32,  33,  33,
  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
  33,  34,  34,  34,  34,  34,  34,  34,  34,  35,  36,  36,  36,  36,  36,  36,  36,  36,  36,
  36,  36,  36,  36,  36,  37,  37,  37,  37,  37,  37,  37,  37,  37,  37,  37,  37,  38,  39,
  39,  39,  39,  39,  39,  39,  39,  39,  39,  39,  39,  40,  40,  40,  40,  40,  40,  40,  40,
  41,  41,  41,  41,  41,  41,  42,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,
  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,  44,  44,  44,  44,  44,  44,
  44,  44,  44,  44,  44,  44,  44,  44,  44,  45,  45,  45,  45,  45,  45,  45,  45,  45,  45,
  45,  45,  45,  45,  45,  46,  47,  47,  47,  47,  47,  47,  47,  47,  47,  47,  47,  47,  47,
  47,  47,  48,  48,  48,  48,  48,  48,  48,  48,  48,  49,  49,  49,  49,  49,  49,  49,  49,
  49,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,
  50,  50,  50,  50,  50,  50,  50,  51,  52,  52,  52,  52,  52,  52,  52,  52,  52,  52,  52,
  52,  52,  52,  52,  52,  52,  52,  52,  52,  52,  52,  52,  52,  52,  53,  53,  53,  53,  53,
  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,
  53,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  55,  55,  55,
  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  55,  56,  56,  56,  56,  56,  56,  56,
  56,  56,  56,  56,  56,  56,  56,  56,  57,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
  58,  58,  58,  58,  58,  59,  59,  59,  59,  59,  59,  59,  59,  59,  59,  59,  59,  59,  59,
  59,  60,  60,  60,  60,  60,  60,  60,  60,  60,  61,  61,  61,  61,  61,  61,  61,  61,  61,
  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,  62,
  62,  62,  62,  62,  62,  62,  63,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,
  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  65,  65,  65,  65,  65,  65,
  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,
  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  67,  67,  67,  67,
  67,  67,  67,  67,  67,  67,  67,  67,  67,  67,  67,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  69,  70,  70,  70,  70,  70,  70,  70,  70,  70,  70,  70,
  70,  70,  70,  70,  71,  71,  71,  71,  71,  71,  71,  71,  71,  71,  71,  71,  71,  71,  71,
  72,  72,  72,  72,  72,  72,  72,  72,  72,  73,  73,  73,  73,  73,  73,  73,  73,  73,  74,
  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,  74,
  74,  74,  74,  74,  74,  75,  76,  76,  76,  76,  76,  76,  76,  76,  76,  76,  76,  76,  76,
  76,  76,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  77,  78,  78,
  78,  78,  78,  78,  78,  78,  78,  78,  78,  78,  78,  78,  78,  79,  80,  80,  80,  80,  80,
  80,  80,  80,  80,  81,  81,  81,  81,  81,  81,  81,  81,  81,  82,  82,  82,  82,  82,  82,
  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,
  83,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  85,  85,  85,
  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  86,  86,  86,  86,  86,  86,  86,
  86,  86,  86,  86,  86,  86,  86,  86,  87,  88,  88,  88,  88,  88,  88,  88,  88,  88,  89,
  89,  89,  89,  89,  89,  89,  89,  89,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,
  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  91,  92,  92,  92,  92,  92,  92,  92,
  92,  92,  92,  92,  92,  92,  92,  92,  92,  92,  92,  92,  92,  93,  93,  93,  93,  93,  93,
  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  94,  94,
  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  95,  95,  95,  95,  95,  95,  95,  95,  95,
  95,  95,  95,  95,  95,  96,  96,  96,  96,  96,  96,  96,  96,  96,  96,  96,  96,  96,  96,
  96,  97,  98,  98,  98,  98,  98,  98,  98,  98,  98,  98,  98,  98,  98,  98,  98,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  100, 100, 100, 100, 100, 100,
  100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 101, 102, 103, 104, 105, 105, 105, 105,
  105, 105, 105, 105, 105, 105, 105, 105, 105, 106, 107, 107, 107, 107, 107, 107, 107, 107, 108,
  108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 109, 110, 110, 110, 110, 110, 110, 110,
  110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 112, 112, 112,
  112, 112, 112, 113, 113, 113, 113, 113, 113, 113, 113, 114, 114, 114, 114, 114, 114, 114, 114,
  114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 116, 116, 116, 116,
  116, 116, 116, 116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117,
  117, 117, 117, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 119,
  120, 120, 120, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 121, 121, 121, 122,
  123, 124, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126,
  126, 126, 126, 127, 128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 129, 129, 129, 129, 129};
int matj[] = {
  0,   1,   2,   3,   4,   6,   7,   8,   10,  11,  13,  14,  15,  16,  17,  19,  20,  21,  22,
  23,  25,  26,  27,  28,  29,  3,   4,   6,   8,   10,  13,  14,  16,  33,  34,  36,  39,  40,
  5,   3,   4,   6,   8,   10,  13,  14,  16,  19,  20,  22,  25,  26,  28,  33,  34,  36,  39,
  40,  43,  44,  47,  48,  3,   7,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  33,  37,
  39,  41,  43,  45,  47,  49,  3,   4,   6,   8,   10,  13,  14,  16,  9,   3,   4,   6,   8,
  10,  13,  14,  16,  19,  20,  22,  25,  26,  28,  3,   7,   11,  13,  15,  17,  19,  21,  23,
  25,  27,  29,  12,  3,   4,   6,   7,   8,   10,  11,  13,  14,  15,  16,  17,  3,   4,   6,
  8,   10,  13,  14,  16,  33,  34,  36,  39,  40,  3,   7,   11,  13,  15,  17,  33,  37,  39,
  41,  3,   4,   6,   8,   10,  13,  14,  16,  3,   7,   11,  13,  15,  17,  18,  3,   6,   7,
  10,  11,  19,  20,  21,  22,  23,  25,  26,  27,  28,  29,  50,  52,  53,  54,  55,  56,  58,
  59,  60,  61,  3,   6,   10,  19,  20,  22,  25,  26,  28,  33,  36,  43,  44,  47,  48,  50,
  52,  54,  56,  58,  60,  74,  76,  78,  80,  3,   7,   11,  19,  21,  23,  25,  27,  29,  33,
  37,  43,  45,  47,  49,  50,  53,  55,  56,  59,  61,  74,  77,  78,  81,  3,   6,   10,  19,
  20,  22,  25,  26,  28,  50,  52,  54,  56,  58,  60,  3,   7,   11,  19,  21,  23,  25,  27,
  29,  50,  53,  55,  56,  59,  61,  24,  3,   6,   7,   10,  11,  19,  20,  21,  22,  23,  25,
  26,  27,  28,  29,  3,   6,   10,  19,  20,  22,  25,  26,  28,  33,  36,  43,  44,  47,  48,
  3,   7,   11,  19,  21,  23,  25,  27,  29,  33,  37,  43,  45,  47,  49,  3,   6,   10,  19,
  20,  22,  25,  26,  28,  3,   7,   11,  19,  21,  23,  25,  27,  29,  30,  31,  32,  4,   6,
  7,   14,  15,  20,  21,  26,  27,  33,  34,  36,  37,  39,  40,  41,  43,  44,  45,  47,  48,
  49,  4,   6,   14,  33,  34,  36,  39,  40,  35,  4,   6,   14,  20,  26,  33,  34,  36,  39,
  40,  43,  44,  47,  48,  7,   15,  21,  27,  33,  37,  39,  41,  43,  45,  47,  49,  38,  4,
  6,   7,   14,  15,  33,  34,  36,  37,  39,  40,  41,  4,   6,   14,  33,  34,  36,  39,  40,
  7,   15,  33,  37,  39,  41,  42,  6,   7,   20,  21,  26,  27,  33,  36,  37,  43,  44,  45,
  47,  48,  49,  52,  53,  58,  59,  74,  76,  77,  78,  80,  81,  6,   20,  26,  33,  36,  43,
  44,  47,  48,  52,  58,  74,  76,  78,  80,  7,   21,  27,  33,  37,  43,  45,  47,  49,  53,
  59,  74,  77,  78,  81,  46,  6,   7,   20,  21,  26,  27,  33,  36,  37,  43,  44,  45,  47,
  48,  49,  6,   20,  26,  33,  36,  43,  44,  47,  48,  7,   21,  27,  33,  37,  43,  45,  47,
  49,  19,  20,  21,  22,  23,  50,  52,  53,  54,  55,  56,  58,  59,  60,  61,  62,  64,  65,
  66,  67,  68,  70,  71,  72,  73,  51,  19,  20,  22,  43,  44,  50,  52,  54,  56,  58,  60,
  62,  64,  66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  19,  21,  23,  43,  45,
  50,  53,  55,  56,  59,  61,  62,  65,  67,  68,  71,  73,  74,  77,  78,  81,  82,  85,  86,
  89,  19,  20,  22,  50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  19,  21,  23,
  50,  53,  55,  56,  59,  61,  62,  65,  67,  68,  71,  73,  19,  20,  21,  22,  23,  50,  52,
  53,  54,  55,  56,  58,  59,  60,  61,  57,  19,  20,  22,  43,  44,  50,  52,  54,  56,  58,
  60,  74,  76,  78,  80,  19,  21,  23,  43,  45,  50,  53,  55,  56,  59,  61,  74,  77,  78,
  81,  19,  20,  22,  50,  52,  54,  56,  58,  60,  19,  21,  23,  50,  53,  55,  56,  59,  61,
  50,  52,  53,  54,  55,  62,  64,  65,  66,  67,  68,  70,  71,  72,  73,  90,  92,  93,  94,
  95,  96,  98,  99,  100, 101, 63,  50,  52,  54,  62,  64,  66,  68,  70,  72,  74,  76,  82,
  84,  86,  88,  90,  92,  94,  96,  98,  100, 114, 116, 118, 120, 50,  53,  55,  62,  65,  67,
  68,  71,  73,  74,  77,  82,  85,  86,  89,  90,  93,  95,  96,  99,  101, 114, 117, 118, 121,
  50,  52,  54,  62,  64,  66,  68,  70,  72,  90,  92,  94,  96,  98,  100, 50,  53,  55,  62,
  65,  67,  68,  71,  73,  90,  93,  95,  96,  99,  101, 50,  52,  53,  54,  55,  62,  64,  65,
  66,  67,  68,  70,  71,  72,  73,  69,  50,  52,  54,  62,  64,  66,  68,  70,  72,  74,  76,
  82,  84,  86,  88,  50,  53,  55,  62,  65,  67,  68,  71,  73,  74,  77,  82,  85,  86,  89,
  50,  52,  54,  62,  64,  66,  68,  70,  72,  50,  53,  55,  62,  65,  67,  68,  71,  73,  20,
  21,  43,  44,  45,  52,  53,  58,  59,  64,  65,  70,  71,  74,  76,  77,  78,  80,  81,  82,
  84,  85,  86,  88,  89,  75,  20,  43,  44,  52,  58,  64,  70,  74,  76,  78,  80,  82,  84,
  86,  88,  21,  43,  45,  53,  59,  65,  71,  74,  77,  78,  81,  82,  85,  86,  89,  20,  21,
  43,  44,  45,  52,  53,  58,  59,  74,  76,  77,  78,  80,  81,  79,  20,  43,  44,  52,  58,
  74,  76,  78,  80,  21,  43,  45,  53,  59,  74,  77,  78,  81,  52,  53,  64,  65,  70,  71,
  74,  76,  77,  82,  84,  85,  86,  88,  89,  92,  93,  98,  99,  114, 116, 117, 118, 120, 121,
  83,  52,  64,  70,  74,  76,  82,  84,  86,  88,  92,  98,  114, 116, 118, 120, 53,  65,  71,
  74,  77,  82,  85,  86,  89,  93,  99,  114, 117, 118, 121, 52,  53,  64,  65,  70,  71,  74,
  76,  77,  82,  84,  85,  86,  88,  89,  87,  52,  64,  70,  74,  76,  82,  84,  86,  88,  53,
  65,  71,  74,  77,  82,  85,  86,  89,  62,  64,  65,  66,  67,  90,  92,  93,  94,  95,  96,
  98,  99,  100, 101, 105, 107, 108, 110, 111, 112, 113, 91,  62,  64,  66,  82,  84,  90,  92,
  94,  96,  98,  100, 108, 110, 112, 114, 116, 118, 120, 126, 128, 62,  65,  67,  82,  85,  90,
  93,  95,  96,  99,  101, 105, 107, 108, 111, 113, 114, 117, 118, 121, 125, 126, 129, 62,  64,
  66,  90,  92,  94,  96,  98,  100, 108, 110, 112, 62,  65,  67,  90,  93,  95,  96,  99,  101,
  105, 107, 108, 111, 113, 62,  64,  65,  66,  67,  90,  92,  93,  94,  95,  96,  98,  99,  100,
  101, 97,  62,  64,  66,  82,  84,  90,  92,  94,  96,  98,  100, 114, 116, 118, 120, 62,  65,
  67,  82,  85,  90,  93,  95,  96,  99,  101, 114, 117, 118, 121, 62,  64,  66,  90,  92,  94,
  96,  98,  100, 62,  65,  67,  90,  93,  95,  96,  99,  101, 102, 103, 104, 90,  93,  95,  105,
  107, 108, 111, 113, 114, 117, 125, 126, 129, 106, 90,  93,  95,  105, 107, 108, 111, 113, 90,
  92,  93,  94,  95,  105, 107, 108, 110, 111, 112, 113, 109, 90,  92,  94,  108, 110, 112, 114,
  116, 126, 128, 90,  93,  95,  105, 107, 108, 111, 113, 114, 117, 125, 126, 129, 90,  92,  94,
  108, 110, 112, 90,  93,  95,  105, 107, 108, 111, 113, 64,  65,  82,  84,  85,  92,  93,  98,
  99,  105, 110, 111, 114, 116, 117, 118, 120, 121, 125, 126, 128, 129, 115, 64,  82,  84,  92,
  98,  110, 114, 116, 118, 120, 126, 128, 65,  82,  85,  93,  99,  105, 111, 114, 117, 118, 121,
  125, 126, 129, 64,  65,  82,  84,  85,  92,  93,  98,  99,  114, 116, 117, 118, 120, 121, 119,
  64,  82,  84,  92,  98,  114, 116, 118, 120, 65,  82,  85,  93,  99,  114, 117, 118, 121, 122,
  123, 124, 93,  105, 111, 114, 117, 125, 126, 129, 92,  93,  105, 110, 111, 114, 116, 117, 125,
  126, 128, 129, 127, 92,  110, 114, 116, 126, 128, 93,  105, 111, 114, 117, 125, 126, 129, 129};
double matv[] = {
  2.15674e+24,  1.72585e+24,  5.01849e+24,  4.94998e+24,  2.31905e+22,  -1.25698e+22, -4.7717e+23,
  -1.43356e+23, 8.3908e+23,   -8.50862e+23, -1.03388e+24, -7.26744e+22, 6.63847e+23,  2.90617e+23,
  -1.98181e+24, -1.11194e+23, 4.27578e+22,  -6.73475e+22, -2.19461e+23, 2.94382e+23,  -9.85777e+23,
  -1.50944e+23, 3.71742e+23,  5.95036e+23,  -8.45701e+23, 2.31905e+22,  4.54564e+24,  -4.24468e+23,
  -1.24338e+24, 4.92253e+23,  -1.11363e+23, -6.80899e+23, -1.66133e+24, 2.1412e+22,   -1.01559e+24,
  4.24377e+23,  -8.38489e+22, -9.28075e+23, 5.07668e+24,  -1.25698e+22, -4.24468e+23, 9.17972e+24,
  4.82281e+23,  -2.46485e+24, -1.08554e+23, -6.76218e+23, -1.62028e+24, 1.05034e+22,  -3.91868e+23,
  2.12582e+23,  -3.79622e+22, -8.06616e+23, -5.4148e+23,  5.65405e+22,  4.34349e+23,  -2.00939e+24,
  -8.63343e+22, -9.66809e+23, 9.0469e+22,   7.20161e+23,  -1.65961e+23, -2.11006e+24, -4.7717e+23,
  9.64564e+24,  -2.25958e+23, 3.06653e+23,  -3.09079e+24, -7.05565e+23, -9.70839e+22, 8.54425e+22,
  4.07668e+23,  5.70103e+23,  -2.9604e+24,  -1.78436e+24, -7.97466e+23, 1.92965e+22,  7.97591e+23,
  -2.06563e+24, -7.74852e+22, 2.56331e+23,  5.00419e+23,  -9.22386e+23, -1.43356e+23, -1.24338e+24,
  4.82281e+23,  6.73286e+24,  -4.41848e+23, 4.45062e+23,  -1.04594e+24, -1.78377e+24, 8.19331e+24,
  8.3908e+23,   4.92253e+23,  -2.46485e+24, -4.41848e+23, 1.36199e+25,  4.34134e+23,  -1.08467e+24,
  -1.77561e+24, -1.46365e+23, 7.78109e+23,  -3.85007e+23, 1.4311e+23,   -2.22809e+24, -2.00298e+24,
  -8.50862e+23, -2.25958e+23, 1.55014e+25,  -5.53032e+23, -2.13434e+24, -5.71849e+24, 4.13327e+23,
  2.88723e+23,  3.13618e+23,  -1.63915e+24, -9.90917e+23, -5.49112e+24, 8.74891e+24,  -1.03388e+24,
  -1.11363e+23, -1.08554e+23, 3.06653e+23,  4.45062e+23,  4.34134e+23,  -5.53032e+23, 7.69772e+24,
  -1.20562e+23, -3.7104e+23,  1.84464e+24,  -2.71937e+24, -7.26744e+22, -6.80899e+23, -6.76218e+23,
  -1.04594e+24, -1.08467e+24, -1.20562e+23, 1.51182e+25,  -5.73288e+24, -1.22214e+23, -1.54346e+24,
  -1.50242e+24, -1.60913e+22, -4.8176e+24,  6.63847e+23,  -3.09079e+24, -2.13434e+24, -3.7104e+23,
  1.39824e+25,  -1.76747e+24, 4.40397e+23,  -6.36859e+23, -8.05327e+23, -1.26983e+24, 2.90617e+23,
  -1.66133e+24, -1.62028e+24, -1.78377e+24, -1.77561e+24, 1.84464e+24,  -5.73288e+24, 2.15884e+25,
  -1.98181e+24, -7.05565e+23, -5.71849e+24, -2.71937e+24, -1.76747e+24, 2.1746e+25,   4.09803e+24,
  -1.11194e+23, 1.05034e+22,  -9.70839e+22, -1.46365e+23, 4.13327e+23,  5.03315e+24,  5.24046e+22,
  -3.85558e+23, 1.47208e+24,  -1.35226e+24, -9.7808e+23,  -1.55715e+22, 4.84552e+23,  5.47585e+21,
  -1.31078e+24, 3.00243e+23,  7.13032e+22,  8.22448e+21,  -3.49267e+23, 6.18903e+22,  8.92724e+23,
  -2.83224e+23, 2.09651e+22,  9.79334e+23,  3.75992e+23,  4.27578e+22,  -3.91868e+23, 7.78109e+23,
  5.24046e+22,  9.3568e+24,   -1.89011e+24, -1.97311e+23, -1.07209e+24, -2.86938e+24, 4.10364e+21,
  1.54634e+23,  2.11679e+23,  -1.48728e+24, -2.50088e+22, 8.66224e+22,  -4.04347e+22, -1.85812e+23,
  2.4756e+23,   2.69939e+23,  -1.59713e+24, -8.7401e+23,  -1.25864e+23, 5.34698e+23,  5.71791e+23,
  -1.66359e+24, -6.73475e+22, 8.54425e+22,  2.88723e+23,  -3.85558e+23, 9.88784e+24,  -6.96799e+23,
  4.45023e+23,  -2.90456e+24, -1.29475e+24, -1.07222e+23, 3.75276e+23,  -7.04199e+23, -3.99475e+23,
  6.1106e+23,   -1.38441e+24, 1.86487e+23,  -1.20613e+23, 8.19272e+23,  -9.11457e+23, -2.37952e+24,
  -3.29011e+24, -3.32754e+21, -4.78825e+21, -1.32603e+23, 3.65806e+23,  -2.19461e+23, 2.12582e+23,
  -3.85007e+23, 1.47208e+24,  -1.89011e+24, 1.4196e+25,   7.62876e+23,  -3.1767e+22,  -2.46587e+24,
  2.72683e+23,  5.79824e+23,  -9.25358e+22, -8.1378e+23,  -1.75649e+24, -3.24772e+24, 2.94382e+23,
  4.07668e+23,  3.13618e+23,  -1.35226e+24, -6.96799e+23, 1.56564e+25,  -1.15266e+24, -1.45286e+24,
  -5.39376e+24, -8.40737e+23, 4.04253e+22,  2.11472e+22,  3.18598e+24,  2.71858e+23,  -4.61191e+24,
  8.95619e+24,  -9.85777e+23, -3.79622e+22, 5.70103e+23,  1.4311e+23,   -1.63915e+24, -9.7808e+23,
  -1.97311e+23, 4.45023e+23,  7.62876e+23,  -1.15266e+24, 7.50532e+24,  -1.05206e+23, -3.13731e+23,
  1.81785e+24,  -2.81935e+24, -1.50944e+23, -8.06616e+23, -2.22809e+24, -1.55715e+22, -1.07209e+24,
  -3.1767e+22,  -1.05206e+23, 1.56398e+25,  -5.60363e+24, -5.29799e+22, -4.23447e+23, -4.6346e+23,
  -2.75099e+24, 4.69561e+21,  -4.68905e+24, 3.71742e+23,  -2.9604e+24,  -9.90917e+23, 4.84552e+23,
  -2.90456e+24, -1.45286e+24, -3.13731e+23, 1.34608e+25,  -1.89672e+24, 6.98781e+23,  -1.71583e+24,
  5.7153e+23,   -1.22629e+24, -7.27749e+23, -1.39838e+24, 5.95036e+23,  -5.4148e+23,  -2.00298e+24,
  5.47585e+21,  -2.86938e+24, -2.46587e+24, 1.81785e+24,  -5.60363e+24, 2.24978e+25,  -8.45701e+23,
  -1.78436e+24, -5.49112e+24, -1.31078e+24, -1.29475e+24, -5.39376e+24, -2.81935e+24, -1.89672e+24,
  2.08365e+25,  2.51238e+24,  3.59576e+24,  4.41426e+24,  2.1412e+22,   5.65405e+22,  -7.97466e+23,
  -1.22214e+23, 4.40397e+23,  4.10364e+21,  -1.07222e+23, -5.29799e+22, 6.98781e+23,  5.8278e+24,
  -1.05496e+23, 3.14588e+23,  9.47464e+23,  -2.3807e+24,  4.27831e+23,  -9.66118e+23, 1.71199e+23,
  -3.41097e+22, 3.9097e+23,   -2.29943e+24, 1.42285e+23,  -2.03179e+24, -1.01559e+24, 4.34349e+23,
  -1.54346e+24, -1.05496e+23, 6.89012e+24,  -5.32027e+22, 2.74064e+23,  -3.48067e+24, 9.82352e+24,
  4.24377e+23,  -2.00939e+24, -1.50242e+24, 1.54634e+23,  -4.23447e+23, 3.14588e+23,  -5.32027e+22,
  1.39876e+25,  2.84314e+23,  -3.46969e+24, -4.00004e+23, 2.32348e+22,  5.94211e+23,  -3.77544e+24,
  1.92965e+22,  -6.36859e+23, 3.75276e+23,  -1.71583e+24, 9.47464e+23,  1.84566e+25,  -2.39489e+24,
  -8.28714e+24, 2.72025e+23,  7.20344e+23,  -1.23834e+24, -7.98139e+24, 6.65223e+24,  -8.38489e+22,
  -8.63343e+22, 7.97591e+23,  -1.60913e+22, -8.05327e+23, -2.3807e+24,  2.74064e+23,  2.84314e+23,
  -2.39489e+24, 7.93474e+24,  9.31819e+23,  4.49296e+22,  -9.28075e+23, -9.66809e+23, -4.8176e+24,
  4.27831e+23,  -3.48067e+24, -3.46969e+24, 9.31819e+23,  2.03753e+25,  -2.06563e+24, -1.26983e+24,
  -9.66118e+23, -8.28714e+24, 4.49296e+22,  2.44652e+25,  5.56645e+24,  9.0469e+22,   -7.74852e+22,
  2.11679e+23,  -7.04199e+23, -4.6346e+23,  5.7153e+23,   1.71199e+23,  -4.00004e+23, 2.72025e+23,
  8.50531e+24,  3.59435e+23,  5.42073e+23,  -2.37068e+24, 1.66602e+24,  -1.53655e+24, 2.47094e+22,
  -1.91384e+23, -2.02749e+23, 1.02309e+24,  -1.14643e+23, -1.11595e+23, 7.71564e+23,  2.75217e+24,
  5.5637e+23,   -3.45852e+24, 7.20161e+23,  -1.48728e+24, -2.75099e+24, -3.41097e+22, 2.32348e+22,
  3.59435e+23,  1.52469e+25,  1.28692e+22,  -4.39792e+24, 2.02434e+23,  -7.81113e+23, 5.25552e+23,
  2.95352e+23,  -1.84614e+24, -5.20694e+24, 2.56331e+23,  -3.99475e+23, -1.22629e+24, 3.9097e+23,
  7.20344e+23,  5.42073e+23,  1.81803e+25,  -1.69467e+24, -7.85046e+24, 7.74058e+23,  -3.19616e+24,
  7.283e+21,    4.48227e+23,  -1.03443e+23, -7.04144e+24, 7.00245e+24,  -1.65961e+23, 5.00419e+23,
  -2.50088e+22, 6.1106e+23,   4.69561e+21,  -7.27749e+23, -2.29943e+24, 5.94211e+23,  -1.23834e+24,
  -2.37068e+24, 1.28692e+22,  -1.69467e+24, 7.60967e+24,  8.83117e+23,  -1.36831e+23, -2.11006e+24,
  8.66224e+22,  -4.68905e+24, 1.42285e+23,  -3.77544e+24, 1.66602e+24,  -4.39792e+24, 8.83117e+23,
  2.15983e+25,  -9.22386e+23, -1.38441e+24, -1.39838e+24, -2.03179e+24, -7.98139e+24, -1.53655e+24,
  -7.85046e+24, -1.36831e+23, 2.32422e+25,  3.00243e+23,  -4.04347e+22, 1.86487e+23,  2.72683e+23,
  -8.40737e+23, 9.60816e+24,  -2.15129e+23, 2.15129e+23,  -1.99117e+24, 1.99117e+24,  -9.65305e+23,
  1.75431e+23,  -7.5964e+23,  -3.97096e+23, 2.66931e+24,  -3.00243e+23, -1.86487e+23, 4.04347e+22,
  8.40737e+23,  -2.72683e+23, -9.65305e+23, 7.5964e+23,   -1.75431e+23, -2.66931e+24, 3.97096e+23,
  4.80408e+24,  7.13032e+22,  -1.85812e+23, 5.79824e+23,  2.47094e+22,  2.02434e+23,  -2.15129e+23,
  9.41268e+24,  -1.3454e+24,  6.19341e+23,  -1.75311e+24, -2.24172e+24, 8.22448e+21,  -1.20613e+23,
  4.04253e+22,  9.24394e+22,  -2.0139e+24,  -8.41236e+22, -5.04239e+23, -9.95045e+23, 2.62725e+23,
  -3.72973e+23, -1.91384e+23, 7.74058e+23,  8.5707e+23,   -2.65947e+24, 8.22448e+21,  -1.20613e+23,
  4.04253e+22,  -1.91384e+23, 7.74058e+23,  2.15129e+23,  9.41268e+24,  -1.3454e+24,  -9.24394e+22,
  -2.0139e+24,  -8.41236e+22, 7.13032e+22,  -1.85812e+23, 5.79824e+23,  -6.19341e+23, -1.75311e+24,
  -2.24172e+24, 5.04239e+23,  -9.95045e+23, -8.5707e+23,  -2.65947e+24, 2.47094e+22,  2.02434e+23,
  -2.62725e+23, -3.72973e+23, -3.49267e+23, 2.4756e+23,   -9.25358e+22, -1.99117e+24, -1.3454e+24,
  1.45606e+25,  -2.17274e+24, -4.6608e+23,  -3.51968e+24, 6.18903e+22,  8.19272e+23,  2.11472e+22,
  -5.05634e+20, -2.75293e+24, -3.97441e+24, 6.18903e+22,  8.19272e+23,  2.11472e+22,  1.99117e+24,
  -1.3454e+24,  1.45606e+25,  5.05634e+20,  -2.75293e+24, -3.97441e+24, -3.49267e+23, 2.4756e+23,
  -9.25358e+22, 2.17274e+24,  -4.6608e+23,  -3.51968e+24, 8.92724e+23,  2.69939e+23,  -9.11457e+23,
  -8.1378e+23,  3.18598e+24,  -9.65305e+23, 6.19341e+23,  -9.24394e+22, -2.17274e+24, 5.05634e+20,
  1.58185e+25,  9.67609e+22,  1.78565e+22,  -4.9018e+24,  4.70184e+24,  8.11652e+24,  -2.83224e+23,
  -1.59713e+24, -1.75649e+24, -2.02749e+23, -7.81113e+23, 1.75431e+23,  -1.75311e+24, -4.6608e+23,
  9.67609e+22,  1.40287e+25,  -3.87943e+24, 7.06635e+23,  -2.14861e+24, -2.95058e+23, -3.17227e+24,
  2.09651e+22,  -2.37952e+24, 2.71858e+23,  1.02309e+24,  -3.19616e+24, -7.5964e+23,  -2.0139e+24,
  -2.75293e+24, 1.78565e+22,  1.50719e+25,  -3.62092e+24, -1.89869e+23, 9.33359e+21,  4.50214e+23,
  -2.91516e+24, 9.79334e+23,  -8.7401e+23,  -3.24772e+24, -3.97096e+23, -2.24172e+24, -3.51968e+24,
  -4.9018e+24,  -3.87943e+24, 2.07577e+25,  3.75992e+23,  -3.29011e+24, -4.61191e+24, 2.66931e+24,
  -8.41236e+22, -3.97441e+24, 4.70184e+24,  -3.62092e+24, 2.25766e+25,  -3.00243e+23, 8.22448e+21,
  7.13032e+22,  6.18903e+22,  -3.49267e+23, 5.03315e+24,  -3.85558e+23, 5.24046e+22,  -1.35226e+24,
  1.47208e+24,  -8.92724e+23, 2.09651e+22,  -2.83224e+23, 3.75992e+23,  9.79334e+23,  -1.11194e+23,
  -9.70839e+22, 1.05034e+22,  4.13327e+23,  -1.46365e+23, -9.7808e+23,  4.84552e+23,  -1.55715e+22,
  -1.31078e+24, 5.47585e+21,  4.09803e+24,  -1.86487e+23, -1.20613e+23, 8.19272e+23,  -3.85558e+23,
  9.88784e+24,  -6.96799e+23, 9.11457e+23,  -2.37952e+24, -3.29011e+24, 3.32754e+21,  -4.78825e+21,
  -7.04199e+23, -3.99475e+23, 1.32603e+23,  3.65806e+23,  -6.73475e+22, 8.54425e+22,  2.88723e+23,
  4.45023e+23,  -2.90456e+24, -1.29475e+24, -1.07222e+23, 3.75276e+23,  6.1106e+23,   -1.38441e+24,
  4.04347e+22,  -1.85812e+23, 2.4756e+23,   5.24046e+22,  9.3568e+24,   -1.89011e+24, -2.69939e+23,
  -1.59713e+24, -8.7401e+23,  1.25864e+23,  5.34698e+23,  2.11679e+23,  -1.48728e+24, -5.71791e+23,
  -1.66359e+24, 4.27578e+22,  -3.91868e+23, 7.78109e+23,  -1.97311e+23, -1.07209e+24, -2.86938e+24,
  4.10364e+21,  1.54634e+23,  -2.50088e+22, 8.66224e+22,  8.40737e+23,  4.04253e+22,  2.11472e+22,
  -1.35226e+24, -6.96799e+23, 1.56564e+25,  -3.18598e+24, 2.71858e+23,  -4.61191e+24, 2.94382e+23,
  4.07668e+23,  3.13618e+23,  -1.15266e+24, -1.45286e+24, -5.39376e+24, -2.72683e+23, 5.79824e+23,
  -9.25358e+22, 1.47208e+24,  -1.89011e+24, 1.4196e+25,   8.1378e+23,   -1.75649e+24, -3.24772e+24,
  -2.19461e+23, 2.12582e+23,  -3.85007e+23, 7.62876e+23,  -3.1767e+22,  -2.46587e+24, -9.65305e+23,
  9.24394e+22,  -6.19341e+23, -5.05634e+20, 2.17274e+24,  -8.92724e+23, 9.11457e+23,  -2.69939e+23,
  -3.18598e+24, 8.1378e+23,   1.58185e+25,  -1.78565e+22, -9.67609e+22, -4.70184e+24, 4.9018e+24,
  7.70196e+24,  7.5964e+23,   -2.0139e+24,  -2.75293e+24, 2.09651e+22,  -2.37952e+24, 2.71858e+23,
  -1.78565e+22, 1.50719e+25,  -3.62092e+24, 1.89869e+23,  9.33359e+21,  1.02309e+24,  -3.19616e+24,
  -4.50214e+23, -2.91516e+24, -1.75431e+23, -1.75311e+24, -4.6608e+23,  -2.83224e+23, -1.59713e+24,
  -1.75649e+24, -9.67609e+22, 1.40287e+25,  -3.87943e+24, -7.06635e+23, -2.14861e+24, -2.02749e+23,
  -7.81113e+23, 2.95058e+23,  -3.17227e+24, -2.66931e+24, -8.41236e+22, -3.97441e+24, 3.75992e+23,
  -3.29011e+24, -4.61191e+24, -4.70184e+24, -3.62092e+24, 2.25766e+25,  3.97096e+23,  -2.24172e+24,
  -3.51968e+24, 9.79334e+23,  -8.7401e+23,  -3.24772e+24, 4.9018e+24,   -3.87943e+24, 2.07577e+25,
  -1.25864e+23, -3.32754e+21, -1.14643e+23, 5.25552e+23,  7.283e+21,    -5.04239e+23, 5.04239e+23,
  7.06635e+23,  -1.89869e+23, 3.32754e+21,  1.25864e+23,  1.89869e+23,  -7.06635e+23, 9.92513e+24,
  -1.13297e+23, 1.13297e+23,  -3.20809e+24, -2.34707e+24, 2.15731e+23,  1.14643e+23,  -7.283e+21,
  -5.25552e+23, -3.20809e+24, -2.15731e+23, 2.34707e+24,  4.96257e+24,  5.34698e+23,  -1.11595e+23,
  2.95352e+23,  -9.95045e+23, -2.14861e+24, -4.78825e+21, 9.33359e+21,  -1.13297e+23, 1.62221e+25,
  -5.71433e+23, -5.57266e+24, 7.71564e+23,  4.48227e+23,  -2.88454e+24, -6.18416e+24, -4.78825e+21,
  7.71564e+23,  4.48227e+23,  -9.95045e+23, 9.33359e+21,  5.34698e+23,  -2.14861e+24, 1.13297e+23,
  1.62221e+25,  2.88454e+24,  -6.18416e+24, -1.11595e+23, 2.95352e+23,  5.71433e+23,  -5.57266e+24,
  5.71791e+23,  -1.32603e+23, 2.75217e+24,  -1.84614e+24, -1.03443e+23, 2.62725e+23,  -8.5707e+23,
  -2.95058e+23, 4.50214e+23,  -3.20809e+24, -5.71433e+23, 2.88454e+24,  1.41094e+25,  -1.89809e+24,
  1.53457e+24,  7.4049e+24,   -1.66359e+24, 5.5637e+23,   -5.20694e+24, -3.72973e+23, -3.17227e+24,
  -2.34707e+24, -5.57266e+24, -1.89809e+24, 2.11973e+25,  3.65806e+23,  -3.45852e+24, -7.04144e+24,
  -2.65947e+24, -2.91516e+24, 2.15731e+23,  -6.18416e+24, 1.53457e+24,  2.36433e+25,  -1.91384e+23,
  2.47094e+22,  -7.04199e+23, 2.11679e+23,  1.02309e+24,  -2.02749e+23, 1.14643e+23,  7.71564e+23,
  -1.11595e+23, 8.50531e+24,  5.42073e+23,  3.59435e+23,  -2.75217e+24, -3.45852e+24, 5.5637e+23,
  -7.74852e+22, 9.0469e+22,   5.7153e+23,   -4.6346e+23,  1.71199e+23,  2.72025e+23,  -4.00004e+23,
  -2.37068e+24, -1.53655e+24, 1.66602e+24,  5.56645e+24,  7.74058e+23,  -3.99475e+23, -3.19616e+24,
  -7.283e+21,   4.48227e+23,  5.42073e+23,  1.81803e+25,  1.03443e+23,  -7.04144e+24, 2.56331e+23,
  -1.22629e+24, 3.9097e+23,   7.20344e+23,  -1.69467e+24, -7.85046e+24, 2.02434e+23,  -1.48728e+24,
  -7.81113e+23, -5.25552e+23, 2.95352e+23,  3.59435e+23,  1.52469e+25,  1.84614e+24,  -5.20694e+24,
  7.20161e+23,  -2.75099e+24, -3.41097e+22, 2.32348e+22,  1.28692e+22,  -4.39792e+24, 8.5707e+23,
  -2.62725e+23, 1.32603e+23,  -5.71791e+23, -4.50214e+23, 2.95058e+23,  -3.20809e+24, -2.88454e+24,
  5.71433e+23,  -2.75217e+24, 1.03443e+23,  1.84614e+24,  1.41094e+25,  -1.53457e+24, 1.89809e+24,
  6.70446e+24,  -2.65947e+24, 3.65806e+23,  -2.91516e+24, -2.15731e+23, -6.18416e+24, -3.45852e+24,
  -7.04144e+24, -1.53457e+24, 2.36433e+25,  -3.72973e+23, -1.66359e+24, -3.17227e+24, 2.34707e+24,
  -5.57266e+24, 5.5637e+23,   -5.20694e+24, 1.89809e+24,  2.11973e+25,  -1.11194e+23, -6.73475e+22,
  4.27578e+22,  2.94382e+23,  -2.19461e+23, 4.94998e+24,  -4.7717e+23,  -1.25698e+22, -8.50862e+23,
  8.3908e+23,   -9.85777e+23, 3.71742e+23,  -1.50944e+23, -8.45701e+23, 5.95036e+23,  2.31905e+22,
  -1.43356e+23, -1.03388e+24, 6.63847e+23,  -7.26744e+22, -1.98181e+24, 2.90617e+23,  5.01849e+24,
  -9.70839e+22, 8.54425e+22,  4.07668e+23,  -7.74852e+22, 2.56331e+23,  -4.7717e+23,  9.64564e+24,
  -2.25958e+23, 5.70103e+23,  -2.9604e+24,  -1.78436e+24, 3.06653e+23,  -3.09079e+24, -7.05565e+23,
  -7.97466e+23, 1.92965e+22,  5.00419e+23,  -9.22386e+23, 7.97591e+23,  -2.06563e+24, 1.05034e+22,
  -3.91868e+23, 2.12582e+23,  9.0469e+22,   7.20161e+23,  -1.25698e+22, 9.17972e+24,  -2.46485e+24,
  -3.79622e+22, -8.06616e+23, -5.4148e+23,  -4.24468e+23, 4.82281e+23,  -1.08554e+23, -6.76218e+23,
  -1.62028e+24, 5.65405e+22,  -2.00939e+24, -1.65961e+23, -2.11006e+24, 4.34349e+23,  -8.63343e+22,
  -9.66809e+23, 4.13327e+23,  2.88723e+23,  3.13618e+23,  -8.50862e+23, -2.25958e+23, 1.55014e+25,
  -1.63915e+24, -9.90917e+23, -5.49112e+24, -5.53032e+23, -2.13434e+24, -5.71849e+24, -1.46365e+23,
  7.78109e+23,  -3.85007e+23, 8.3908e+23,   -2.46485e+24, 1.36199e+25,  1.4311e+23,   -2.22809e+24,
  -2.00298e+24, 4.92253e+23,  -4.41848e+23, 4.34134e+23,  -1.08467e+24, -1.77561e+24, -9.7808e+23,
  4.45023e+23,  -1.97311e+23, -1.15266e+24, 7.62876e+23,  -9.85777e+23, 5.70103e+23,  -3.79622e+22,
  -1.63915e+24, 1.4311e+23,   7.50532e+24,  -3.13731e+23, -1.05206e+23, -2.81935e+24, 1.81785e+24,
  8.95619e+24,  4.84552e+23,  -2.90456e+24, -1.45286e+24, 5.7153e+23,   -1.22629e+24, 3.71742e+23,
  -2.9604e+24,  -9.90917e+23, -3.13731e+23, 1.34608e+25,  -1.89672e+24, 6.98781e+23,  -1.71583e+24,
  -7.27749e+23, -1.39838e+24, -1.55715e+22, -1.07209e+24, -3.1767e+22,  -4.6346e+23,  -2.75099e+24,
  -1.50944e+23, -8.06616e+23, -2.22809e+24, -1.05206e+23, 1.56398e+25,  -5.60363e+24, -5.29799e+22,
  -4.23447e+23, 4.69561e+21,  -4.68905e+24, -1.31078e+24, -1.29475e+24, -5.39376e+24, -8.45701e+23,
  -1.78436e+24, -5.49112e+24, -2.81935e+24, -1.89672e+24, 2.08365e+25,  5.47585e+21,  -2.86938e+24,
  -2.46587e+24, 5.95036e+23,  -5.4148e+23,  -2.00298e+24, 1.81785e+24,  -5.60363e+24, 2.24978e+25,
  1.72585e+24,  2.15674e+24,  5.07668e+24,  2.31905e+22,  -4.24468e+23, 4.92253e+23,  4.54564e+24,
  -1.24338e+24, -1.11363e+23, -6.80899e+23, -1.66133e+24, 2.1412e+22,   4.24377e+23,  -1.01559e+24,
  -8.38489e+22, -9.28075e+23, 8.19331e+24,  -1.43356e+23, 4.82281e+23,  -4.41848e+23, -1.24338e+24,
  6.73286e+24,  4.45062e+23,  -1.04594e+24, -1.78377e+24, -1.03388e+24, 3.06653e+23,  -1.08554e+23,
  -5.53032e+23, 4.34134e+23,  -1.11363e+23, 4.45062e+23,  7.69772e+24,  -3.7104e+23,  -1.20562e+23,
  -2.71937e+24, 1.84464e+24,  8.74891e+24,  6.63847e+23,  -3.09079e+24, -2.13434e+24, -3.7104e+23,
  1.39824e+25,  -1.76747e+24, 4.40397e+23,  -6.36859e+23, -8.05327e+23, -1.26983e+24, -7.26744e+22,
  -6.76218e+23, -1.08467e+24, -6.80899e+23, -1.04594e+24, -1.20562e+23, 1.51182e+25,  -5.73288e+24,
  -1.22214e+23, -1.50242e+24, -1.54346e+24, -1.60913e+22, -4.8176e+24,  -1.98181e+24, -7.05565e+23,
  -5.71849e+24, -2.71937e+24, -1.76747e+24, 2.1746e+25,   2.90617e+23,  -1.62028e+24, -1.77561e+24,
  -1.66133e+24, -1.78377e+24, 1.84464e+24,  -5.73288e+24, 2.15884e+25,  -1.07222e+23, 4.10364e+21,
  1.71199e+23,  3.9097e+23,   -3.41097e+22, -7.97466e+23, 5.65405e+22,  6.98781e+23,  -5.29799e+22,
  2.1412e+22,   4.40397e+23,  -1.22214e+23, 5.8278e+24,   9.47464e+23,  3.14588e+23,  -2.29943e+24,
  -2.03179e+24, 1.42285e+23,  -1.05496e+23, -2.3807e+24,  -9.66118e+23, 4.27831e+23,  4.41426e+24,
  3.75276e+23,  2.72025e+23,  7.20344e+23,  1.92965e+22,  -1.71583e+24, -6.36859e+23, 9.47464e+23,
  1.84566e+25,  -1.23834e+24, -7.98139e+24, -2.39489e+24, -8.28714e+24, 1.54634e+23,  -4.00004e+23,
  2.32348e+22,  -2.00939e+24, -4.23447e+23, 4.24377e+23,  -1.50242e+24, 3.14588e+23,  1.39876e+25,
  5.94211e+23,  -3.77544e+24, -5.32027e+22, 2.84314e+23,  -3.46969e+24, 6.1106e+23,   -2.50088e+22,
  -2.37068e+24, -1.69467e+24, 1.28692e+22,  5.00419e+23,  -1.65961e+23, -7.27749e+23, 4.69561e+21,
  -2.29943e+24, -1.23834e+24, 5.94211e+23,  7.60967e+24,  -1.36831e+23, 8.83117e+23,  7.00245e+24,
  -1.38441e+24, -1.53655e+24, -7.85046e+24, -9.22386e+23, -1.39838e+24, -2.03179e+24, -7.98139e+24,
  -1.36831e+23, 2.32422e+25,  8.66224e+22,  1.66602e+24,  -4.39792e+24, -2.11006e+24, -4.68905e+24,
  1.42285e+23,  -3.77544e+24, 8.83117e+23,  2.15983e+25,  3.59576e+24,  2.51238e+24,  9.82352e+24,
  4.34349e+23,  -1.01559e+24, -1.54346e+24, -1.05496e+23, -5.32027e+22, 6.89012e+24,  2.74064e+23,
  -3.48067e+24, 7.97591e+23,  -8.63343e+22, -8.38489e+22, -8.05327e+23, -1.60913e+22, -2.3807e+24,
  -2.39489e+24, 2.84314e+23,  2.74064e+23,  7.93474e+24,  4.49296e+22,  9.31819e+23,  6.65223e+24,
  -2.06563e+24, -1.26983e+24, -9.66118e+23, -8.28714e+24, 4.49296e+22,  2.44652e+25,  -9.66809e+23,
  -9.28075e+23, -4.8176e+24,  4.27831e+23,  -3.46969e+24, -3.48067e+24, 9.31819e+23,  2.03753e+25,
  2.03753e+25};


void
run()
{
  int                dim = 2300;
  int                ierr;
  const unsigned int myid    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int       numproc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  IndexSet owned(dim);

  if (myid == 0)
    owned.add_range(0, dim / 2);
  else
    owned.add_range(dim / 2, dim);
  if (numproc == 1)
    owned.add_range(0, dim);


  TrilinosWrappers::SparsityPattern sp(owned);

  unsigned int n = sizeof(mati) / sizeof(*mati);

  for (unsigned int i = 0; i < n; ++i)
    sp.add(mati[i], matj[i]);

  for (unsigned int i = 0; i < dim; ++i)
    sp.add(i, i);

  sp.compress();

  TrilinosWrappers::SparseMatrix mat(sp);
  TrilinosWrappers::MPI::Vector  x1(owned, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector  x2(owned, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector  b(owned, MPI_COMM_WORLD);

  for (unsigned int i = 0; i < dim; ++i)
    mat.set(i, i, 1.0);


  for (unsigned int i = 0; i < n; ++i)
    {
      mat.set(mati[i], matj[i], matv[i]);
      b(mati[i]) = 1.0;
    }
  mat.compress(VectorOperation::insert);
  b.compress(VectorOperation::insert);

  deallog << "SA:" << std::endl;

  TrilinosWrappers::PreconditionAMG::AdditionalData data;
  data.elliptic = true;

  {
    TrilinosWrappers::PreconditionAMG prec;
    prec.initialize(mat, data);
    prec.vmult(x1, b);
  }
  {
    TrilinosWrappers::PreconditionAMG prec;
    prec.initialize(mat, data);
    prec.vmult(x2, b);
  }

  if (myid == 0)
    {
      for (unsigned int j = 0; j < 10; ++j)
        {
          if (x1[j] != x2[j])
            deallog << "FAIL: j=" << j << ": " << x1[j] << " != " << x2[j] << std::endl;
        }
    }

  deallog << "NSSA:" << std::endl;
  data.elliptic = false;

  {
    TrilinosWrappers::PreconditionAMG prec;
    //      prec.initialize(mat, parameter_list);
    prec.initialize(mat, data);
    prec.vmult(x1, b);
  }
  {
    TrilinosWrappers::PreconditionAMG prec;
    //      prec.initialize(mat, parameter_list);
    prec.initialize(mat, data);
    prec.vmult(x2, b);
  }

  if (myid == 0)
    {
      for (unsigned int j = 0; j < 10; ++j)
        {
          if (x1[j] != x2[j])
            deallog << "FAIL: j=" << j << ": " << x1[j] << " != " << x2[j] << std::endl;
        }
    }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    log;
  run();
}
