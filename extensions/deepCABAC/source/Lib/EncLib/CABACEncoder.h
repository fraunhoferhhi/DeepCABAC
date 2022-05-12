/*
The copyright in this software is being made available under this Software
Copyright License. This software may be subject to other third party and
contributor rights, including patent rights, and no such rights are
granted under this license.
Copyright (c) 1995 - 2019 Fraunhofer-Gesellschaft zur Förderung der
angewandten Forschung e.V. (Fraunhofer)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted for purpose of testing the functionalities of
this software provided that the following conditions are met:
*     Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
*     Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
*     Neither the names of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
*/
#ifndef __CABACENC__
#define __CABACENC__

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "CommonLib/ContextModel.h"
#include "CommonLib/ContextModeler.h"
#include "CommonLib/Bitstream.h"
#include "BinEncoder.h"
#include "BitEstimator.h"
#include "CABACEncoderIf.h"
#include <bitset>
#include <limits>

namespace py = pybind11;

extern uint32_t  g_NumGtxFlags;

class CABACEncoder : public CABACEncoderIf
{
public:
    CABACEncoder( std::vector<uint8_t>* pBytestream );
    ~CABACEncoder() {}

    void      startCabacEncoding      ();
    void      terminateCabacEncoding  ();
    void      initCtxMdls             ();

    void      encodeSideinfo          ( float32_t stepsize, py::array_t<float32_t, py::array::c_style> Weights );
    void      encodeWeightsRD         ( float32_t* pWeights, float32_t* pIntervals, float32_t stepsize, float32_t lambda, uint32_t layerWidth, uint32_t numWeights );
    void      encodeWeightsRD         ( float32_t* pWeights, float32_t   Interval,  float32_t stepsize, float32_t lambda, uint32_t layerWidth, uint32_t numWeights );
    void      encodeWeightsRD         ( int8_t *pWeights, uint32_t numWeights);



protected:
    float32_t estimateAndDecideWeight ( int32_t& bestWeightInt, float32_t origWeight, float32_t weightInterval, float32_t stepsize, float32_t lambda, bool newNorm = false );

    uint64_t  estimateWeightVal       ( int32_t weightInt );
    void      encodeWeightVal         ( int32_t weightInt );
private:
    std::vector<SBMPCtx> m_CtxStore;
    ContextModeler       m_CtxModeler;
    BinEnc               m_BinEncoder;
    BitEst               m_BitEstimator;
};

#endif // !__CABACENCIF__
