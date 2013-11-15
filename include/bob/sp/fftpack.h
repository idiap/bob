/*
 * This file is from the NumPy library:
 *   https://github.com/numpy/numpy/blob/master/numpy/fft/fftpack.h
 *
 * Copyright (c) 2005-2011, NumPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *   * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *   * Neither the name of the NumPy Developers nor the names of any
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * This file is part of tela the Tensor Language.
 * Copyright (c) 1994-1995 Pekka Janhunen
 */

#ifdef __cplusplus
extern "C" {
#endif

#define DOUBLE

#ifdef DOUBLE
#define Treal double
#else
#define Treal float
#endif

extern void cfftf(int N, Treal data[], const Treal wrk[]);
extern void cfftb(int N, Treal data[], const Treal wrk[]);
extern void cffti(int N, Treal wrk[]);

extern void rfftf(int N, Treal data[], const Treal wrk[]);
extern void rfftb(int N, Treal data[], const Treal wrk[]);
extern void rffti(int N, Treal wrk[]);

#ifdef __cplusplus
}
#endif
