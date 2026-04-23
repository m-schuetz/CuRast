/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */




#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.
#define C_norm 0.3535533905932737f // 1 / (8^0.5)
#define MODULO_8(x) ((x) & 7)
void CUDAsubroutineInplaceIDCTvector(float* Vect0, int Step)
{
	float* Vect1 = Vect0 + Step;
	float* Vect2 = Vect1 + Step;
	float* Vect3 = Vect2 + Step;
	float* Vect4 = Vect3 + Step;
	float* Vect5 = Vect4 + Step;
	float* Vect6 = Vect5 + Step;
	float* Vect7 = Vect6 + Step;

	float Y04P = (*Vect0) + (*Vect4);
	float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

	float Y04P2b6ePP = Y04P + Y2b6eP;
	float Y04P2b6ePM = Y04P - Y2b6eP;
	float Y7f1aP3c5dPP = C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
	float Y7a1fM3d5cMP = C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

	float Y04M = (*Vect0) - (*Vect4);
	float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

	float Y04M2e6bMP = Y04M + Y2e6bM;
	float Y04M2e6bMM = Y04M - Y2e6bM;
	float Y1c7dM3f5aPM = C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
	float Y1d7cP3a5fMM = C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

	(*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
	(*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
	(*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
	(*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

	(*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
	(*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
	(*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
	(*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}