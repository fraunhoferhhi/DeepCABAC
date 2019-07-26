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
#include "BitEstimator.h"

const uint32_t BitEst::m_auiGoRiceRange[ 10 ] =
{
    6, 5, 6, 3, 3, 3, 3, 3, 3, 3
};


void BitEst::startBinEncoder()
{
    m_EstScaledBits = 0;
}

void BitEst::setByteStreamBuf( std::vector<uint8_t> *byteStreamBuf )
{}

void BitEst::encodeBin( uint32_t bin, SBMPCtx &ctxMdl )
{
    CHECK( 1, "SHOULD CURRENTLY NOT BE NECESSARY , REMOVE CHECK IF NEEDED!" );
    encodeBinNoUpdate(bin, ctxMdl);
    ctxMdl.updateState(-(int32_t)bin);
}

void BitEst::encodeBinNoUpdate( uint32_t bin, const SBMPCtx &ctxMdl )
{
    m_EstScaledBits += ctxMdl.getBits().scaledEstBits[bin]; //The 15 MSB of state should be the sum of state1 + state1
}

void BitEst::encodeBinEP( uint32_t bin )
{
    m_EstScaledBits += 1 << 15;
}

void BitEst::encodeBinsEP( uint32_t bins, uint32_t numBins )
{
    m_EstScaledBits += numBins << 15;
}

void BitEst::encodeRemAbsLevel( uint32_t remAbsLevel, uint32_t goRicePar, bool useLimitedPrefixLength, uint32_t maxLog2TrDynamicRange )
{
    uint32_t bins = (uint32_t)remAbsLevel;
    const uint32_t threshold = m_auiGoRiceRange[ goRicePar ] << goRicePar;
    if (bins < threshold)
    {
        m_EstScaledBits += (uint64_t)((bins >> goRicePar) + 1 + goRicePar) << 15;
    }
    else if (useLimitedPrefixLength)
    {
        const uint32_t  maxPrefixLength = 32 - 3 - maxLog2TrDynamicRange;
        uint32_t        prefixLength    = 0;
        uint32_t        codeValue       = (bins >> goRicePar) - 3;
        uint32_t        suffixLength;
        if (codeValue >= ((1u << maxPrefixLength) - 1))
        {
            prefixLength = maxPrefixLength;
            suffixLength = maxLog2TrDynamicRange;
        }
        else
        {
            while (codeValue > ((2u << prefixLength) - 2))
            {
                prefixLength++;
            }
            suffixLength = prefixLength + goRicePar + 1; //+1 for the separator bit
        }
        m_EstScaledBits += (uint64_t)(3 + prefixLength + suffixLength) << 15;
    }
    else
    {
        uint32_t length = goRicePar;
        uint32_t delta  = 1 << length;
        bins           -= threshold;
        while (bins >= delta)
        {
            bins -= delta;
            delta = 1 << (++length);
        }
        m_EstScaledBits += (uint64_t)(m_auiGoRiceRange[ goRicePar ] + 1 + (length << 1) - goRicePar) << 15;
    }
}

void BitEst::encodeRemAbsLevelNew( uint32_t remAbsLevel, std::vector<SBMPCtx>& ctxStore )
{
    uint32_t remLevel = uint32_t(remAbsLevel);
    uint32_t log2NumElemNextGroup = 0;

    uint32_t remAbsBaseLevel = 0;
    uint32_t ctxIdx =  ( 6 + g_NumGtxFlags*2 ) ;

    if (remLevel > 0)
    {
        encodeBinNoUpdate( 1, ctxStore[ ctxIdx ]);
        remAbsBaseLevel += (1 << log2NumElemNextGroup);
        ctxIdx++;
        log2NumElemNextGroup++;
    }
    else
    {
        encodeBinNoUpdate(0, ctxStore[ ctxIdx ]);
        return;
    }

    while (remLevel > (remAbsBaseLevel + (1 << log2NumElemNextGroup) - 1))
    {
        encodeBinNoUpdate(1, ctxStore[ ctxIdx ]);
        remAbsBaseLevel += (1 << log2NumElemNextGroup);
        ctxIdx++;
        log2NumElemNextGroup++;
    }

    encodeBinNoUpdate(0, ctxStore[ ctxIdx ]);
    m_EstScaledBits += (log2NumElemNextGroup << 15);
}


void BitEst::terminate_write()
{}

void BitEst::resetBitCounter()
{
    m_EstScaledBits = 0;
}

uint64_t BitEst::getBitCounter()
{
    return m_EstScaledBits;
}
