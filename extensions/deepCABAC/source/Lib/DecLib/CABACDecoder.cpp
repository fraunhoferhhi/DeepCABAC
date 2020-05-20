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
#include "CABACDecoder.h"
#include <iostream>

CABACDecoder::CABACDecoder()
{
    m_CtxStore.resize( 6 + 19 * 2 + 32); // NumOfGtxFlags is max 19
    initCtxMdls();
}

void CABACDecoder::initCtxMdls()
{
    for (uint32_t ctxId = 0; ctxId < m_CtxStore.size(); ctxId++)
    {
        m_CtxStore[ ctxId ].initState();
    }
}

void CABACDecoder::startCabacDecoding( uint8_t* pBytestream )
{
    m_BinDecoder.setByteStreamBuf(pBytestream);
    initCtxMdls();
    m_BinDecoder.startBinDecoder();
    m_CtxModeler.init();
}

void CABACDecoder::decodeStepsize( float32_t &stepsize )
{
  FloatUIntUnion uiTf;
  uint32_t bins = m_BinDecoder.decodeBinsEP(16);
  bins += (m_BinDecoder.decodeBinsEP(16)) << 16;
  uiTf.ui = bins;

  stepsize = uiTf.f;
}

void CABACDecoder::decodeSideinfo( std::vector<uint32_t>* pDimensions, float32_t& stepsize )
{
  g_NumGtxFlags = 4 + m_BinDecoder.decodeBinsEP(4);

  uint32_t dimensionFlag = m_BinDecoder.decodeBinsEP(2);
  int32_t  numDims       = 1 << dimensionFlag;  // 0 for 1, 1 for 2, 2 for 4

  uint32_t currDim      = 0;

  for (int i = 0; i < numDims; i++)
  {
      currDim = m_BinDecoder.decodeBinsEP(16);
      pDimensions->push_back(currDim);
  }

  FloatUIntUnion uiTf;
  uint32_t bins = m_BinDecoder.decodeBinsEP(16);
  bins += (m_BinDecoder.decodeBinsEP(16)) << 16;
  uiTf.ui = bins;

  stepsize = uiTf.f;
}

void CABACDecoder::decodeWeights( int32_t* pWeights, uint32_t layerWidth, uint32_t numWeights )
{
    m_CtxModeler.resetNeighborCtx();
    for (uint32_t posInMat = 0; posInMat < numWeights; posInMat++)
    {
        pWeights[posInMat] = 0;
        decodeWeightVal( pWeights[ posInMat ] );
        m_CtxModeler.updateNeighborCtx(pWeights[ posInMat ], posInMat, layerWidth);
    }
}

void CABACDecoder::decodeWeightVal( int32_t& decodedIntVal )
{
    int32_t sigctx = m_CtxModeler.getSigCtxId();
    uint32_t sigFlag = m_BinDecoder.decodeBin(m_CtxStore[ sigctx ]);

    if (sigFlag)
    {
        decodedIntVal++;
        uint32_t signFlag = 0;
        uint32_t maxAbsPositive = (1 << (BITS_WEIGHT_INTS - 1)) - 1;
        uint32_t maxAbsNegative = (1 << (BITS_WEIGHT_INTS - 1));

        if (maxAbsNegative == 0 || maxAbsPositive == 0)
        {
            signFlag = maxAbsNegative ? 1 : 0;
        }
        else
        {
            int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
            signFlag = m_BinDecoder.decodeBin(m_CtxStore[ signCtx ]);
        }

        int32_t intermediateVal = signFlag ? -1 : 1;

        uint32_t maxAbsVal = signFlag ? maxAbsNegative : maxAbsPositive;
        int32_t ctxIdx = m_CtxModeler.getGtxCtxId(intermediateVal, 0);
        uint32_t grXFlag = 0;

        if (maxAbsVal > 1)
        {
            grXFlag = m_BinDecoder.decodeBin(m_CtxStore[ ctxIdx ]); //greater1
        }

        uint32_t numGreaterFlagsDecoded = 1;

        while (grXFlag && (numGreaterFlagsDecoded < g_NumGtxFlags))
        {
            decodedIntVal++;

            if (decodedIntVal >= maxAbsVal)
            {
                grXFlag = 0;
                break;
            }

            ctxIdx =  m_CtxModeler.getGtxCtxId(intermediateVal, numGreaterFlagsDecoded);
            grXFlag = m_BinDecoder.decodeBin(m_CtxStore[ ctxIdx ]);
            numGreaterFlagsDecoded++;
        }

        if (grXFlag)
        {
            decodedIntVal++;

            if (std::abs(decodedIntVal) < maxAbsVal)
            {
                uint32_t remAbsLevel = m_BinDecoder.decodeRemAbsLevelNew(m_CtxStore);
                decodedIntVal += remAbsLevel;
            }
        }
        decodedIntVal = signFlag ? -decodedIntVal : decodedIntVal;
    }
}

uint32_t CABACDecoder::getBytesRead()
{
  return m_BinDecoder.getBytesRead();
}

uint32_t CABACDecoder::terminateCabacDecoding()
{
  if( m_BinDecoder.decodeBinTrm() )
  {
    m_BinDecoder.finish();
    return m_BinDecoder.getBytesRead();
  }
  CHECK(1, "Terminating Bin not received!");
}
