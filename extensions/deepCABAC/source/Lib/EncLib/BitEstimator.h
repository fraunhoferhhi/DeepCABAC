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
#ifndef __BITEST__
#define __BITEST__

#include "CommonLib/ContextModel.h"

extern uint32_t g_NumGtxFlags;

class BitEst
{
public:
    BitEst() {}
    ~BitEst() {}
public:
    void      startBinEncoder      ( );
    void      setByteStreamBuf     ( std::vector<uint8_t> *byteStreamBuf );

    void      encodeBin            ( uint32_t bin,       SBMPCtx &ctxMdl );
    void      encodeBinNoUpdate    ( uint32_t bin, const SBMPCtx &ctxMdl );
    void      encodeBinEP          ( uint32_t bin                        );
    void      encodeBinsEP         ( uint32_t bins, uint32_t numBins     );

    void      encodeRemAbsLevel    ( uint32_t remAbsLevel, uint32_t goRicePar, bool useLimitedPrefixLength, uint32_t maxLog2TrDynamicRange );
    void      encodeRemAbsLevelNew ( uint32_t remAbsLevel, std::vector<SBMPCtx>& ctxStore                                                  );


    void      terminate_write      ();

    void      resetBitCounter      ();
    uint64_t  getBitCounter        ();
private:
    uint64_t m_EstScaledBits;
    static const uint32_t   m_auiGoRiceRange[ 10 ];
};

#endif // !__BITEST__
