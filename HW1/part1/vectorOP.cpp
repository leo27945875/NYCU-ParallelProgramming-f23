#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float xVal, result, threshold;
  __pp_vec_int xExp, zeros, ones;
  __pp_mask maskAll, maskOp, maskIsZero, maskNotZero, maskIsClamped;

  maskAll    = _pp_init_ones();
  maskOp     = _pp_init_ones();
  maskIsZero = _pp_init_ones();  // [maskIsZero] will be filled with ones automatically after every while-loop, so just initialize [maskIsZero] at the begining.

  _pp_vset_float(threshold, 9.999999f, maskAll);
  _pp_vset_int(zeros, 0, maskAll);
  _pp_vset_int(ones , 1, maskAll);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    if (i + VECTOR_WIDTH > N)
      maskOp = _pp_init_ones(N - i);

    // Initialize result array
    _pp_vset_float(result, 1.0f, maskOp);

    // Load data
    _pp_vload_float(xVal, values    + i, maskOp);
    _pp_vload_int  (xExp, exponents + i, maskOp);

    // Loop to get exp results
    _pp_veq_int(maskIsZero, xExp, zeros, maskOp);
    while (_pp_cntbits(maskIsZero) != VECTOR_WIDTH){
      maskNotZero = _pp_mask_not(maskIsZero);
      _pp_vmult_float(result, result, xVal, maskNotZero);
      _pp_vsub_int(xExp, xExp, ones, maskNotZero);
      _pp_veq_int(maskIsZero, xExp, zeros, maskOp);
    }

    // Clamp result value
    _pp_vgt_float(maskIsClamped, result, threshold, maskOp);
    _pp_vset_float(result, 9.999999f, maskIsClamped);
    
    // Store result
    _pp_vstore_float(output + i, result, maskOp);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_vec_float x, interleaved;
  __pp_mask maskAll;

  float result = 0.f;

  maskAll = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(x, values + i, maskAll);
    
    int L = VECTOR_WIDTH;
    while (L != 1){
      _pp_hadd_float(x, x);
      _pp_interleave_float(interleaved, x);
      _pp_vmove_float(x, interleaved, maskAll);
      L >>= 1;
    }

    result += x.value[0];
  }

  return result;
}