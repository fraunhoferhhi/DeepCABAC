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

#include "CABACEncoder.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

CABACEncoder::CABACEncoder( std::vector<uint8_t>* pBytestream )
{
    m_CtxStore.resize( 6 + g_NumGtxFlags * 2 + 32);

    initCtxMdls();
    m_BinEncoder.setByteStreamBuf(pBytestream);
}

void CABACEncoder::initCtxMdls()
{
    for (uint32_t ctxId = 0; ctxId < m_CtxStore.size() ; ctxId++)
    {
        m_CtxStore[ctxId].initState();
    }
}

void CABACEncoder::startCabacEncoding()
{
    initCtxMdls();
    m_BinEncoder.startBinEncoder();
    m_CtxModeler.init();
}

void CABACEncoder::terminateCabacEncoding()
{
    m_BinEncoder.encodeBinTrm(1);
    m_BinEncoder.finish();
}

uint64_t CABACEncoder::estimateWeightVal( int32_t weightInt )
{
    m_BitEstimator.resetBitCounter();

    uint32_t sigFlag = weightInt != 0 ? 1 : 0;
    int32_t sigctx = m_CtxModeler.getSigCtxId();

    m_BitEstimator.encodeBinNoUpdate(sigFlag, m_CtxStore[ sigctx ]);

    if (sigFlag)
    {
        uint32_t signFlag = weightInt < 0 ? 1 : 0;
        uint32_t maxAbsPositive = (1 << (BITS_WEIGHT_INTS-1)) - 1;
        uint32_t maxAbsNegative = (1 << (BITS_WEIGHT_INTS-1));

        if (maxAbsNegative != 0 && maxAbsPositive != 0)
        {
            int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
            m_BitEstimator.encodeBinNoUpdate(signFlag, m_CtxStore[ signCtx ]);
        }

        uint32_t remAbsLevel = abs(weightInt) - 1;
        uint32_t maxAbsVal = signFlag ? maxAbsNegative : maxAbsPositive;
        uint32_t grXFlag = remAbsLevel ? 1 : 0; //greater1
        int32_t ctxIdx = m_CtxModeler.getGtxCtxId(weightInt, 0);

        if (maxAbsVal > 1)
        {
            m_BitEstimator.encodeBinNoUpdate(grXFlag, m_CtxStore[ ctxIdx ]);
        }

        uint32_t numGreaterFlagsCoded = 1;

        while (grXFlag && numGreaterFlagsCoded < g_NumGtxFlags && (numGreaterFlagsCoded + 1) < maxAbsVal)
        {
            remAbsLevel--;
            grXFlag = remAbsLevel ? 1 : 0;
            ctxIdx = m_CtxModeler.getGtxCtxId(weightInt, numGreaterFlagsCoded );
            m_BitEstimator.encodeBinNoUpdate(grXFlag, m_CtxStore[ ctxIdx ]);
            numGreaterFlagsCoded++;
        }

        if (grXFlag && (numGreaterFlagsCoded + 1) < maxAbsVal)
        {
            remAbsLevel--;
            m_BitEstimator.encodeRemAbsLevelNew(remAbsLevel, m_CtxStore);
        }
    }
    return ( m_BitEstimator.getBitCounter() );
}

void  CABACEncoder::encodeWeightVal( int32_t weightInt )
{
    uint32_t maxAbsPositive = (1 << (BITS_WEIGHT_INTS - 1)) - 1;
    uint32_t maxAbsNegative = (1 << (BITS_WEIGHT_INTS - 1));

    CHECK( !( weightInt >= -(int32_t(maxAbsNegative)) ) || !( weightInt <= int32_t(maxAbsPositive) ) , printf("Value to encode %i  exceeds %i bits (stepsize is too small)!", weightInt,  BITS_WEIGHT_INTS))

    uint32_t sigFlag = weightInt != 0 ? 1 : 0;
    int32_t sigctx = m_CtxModeler.getSigCtxId();

    m_BinEncoder.encodeBin(sigFlag, m_CtxStore[ sigctx ]);

    if (sigFlag)
    {
        uint32_t signFlag = weightInt < 0 ? 1 : 0;

        if (maxAbsNegative != 0 && maxAbsPositive != 0)
        {
            int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
            m_BinEncoder.encodeBin(signFlag, m_CtxStore[ signCtx ]);
        }

        uint32_t remAbsLevel = abs(weightInt) - 1;
        uint32_t maxAbsVal = signFlag ? maxAbsNegative : maxAbsPositive;
        uint32_t grXFlag = remAbsLevel ? 1 : 0; //greater1
        int32_t ctxIdx = m_CtxModeler.getGtxCtxId( weightInt, 0 );

        if (maxAbsVal > 1)
        {
            m_BinEncoder.encodeBin(grXFlag, m_CtxStore[ ctxIdx ]);
        }

        uint32_t numGreaterFlagsCoded = 1;

        while (grXFlag && (numGreaterFlagsCoded < g_NumGtxFlags) && ((numGreaterFlagsCoded + 1) < maxAbsVal))
        {
            remAbsLevel--;
            grXFlag = remAbsLevel ? 1 : 0;
            ctxIdx =  m_CtxModeler.getGtxCtxId(weightInt, numGreaterFlagsCoded);
            m_BinEncoder.encodeBin(grXFlag, m_CtxStore[ ctxIdx ]);
            numGreaterFlagsCoded++;
        }

        if (grXFlag && ((numGreaterFlagsCoded + 1) < maxAbsVal))
        {
            remAbsLevel--;
            m_BinEncoder.encodeRemAbsLevelNew(remAbsLevel, m_CtxStore);
        }
    }
}

float32_t CABACEncoder::estimateAndDecideWeight( int32_t& bestWeightInt, float32_t origWeight, float32_t weightInterval, float32_t stepsize, float32_t lambda, bool newNorm )
{
    float32_t bestDist   = 0.0;
    float32_t bitrate    = 0.0;
    float32_t distortion = 0.0;
    float32_t diff       = 0.0;
    float32_t minCost    = std::numeric_limits<float32_t>::max(); //set to max initially
    float32_t currCost   = 0.0;

    const float32_t oneOver2_15 = 1.0 / (1 << 15);

    float32_t upperBound = origWeight + 3*weightInterval;
    float32_t lowerBound = origWeight - 3*weightInterval;

    int32_t   intWeight  = (int32_t)(( std::fabs(origWeight) + (stepsize*0.5) ) / ( stepsize ));
              intWeight  = origWeight >= 0.0 ? intWeight : -intWeight;

    bestWeightInt = intWeight;

    float32_t normInterval = newNorm ? 6*stepsize : weightInterval;

    for (int i = 0; (i + intWeight) >= -(1 << (BITS_WEIGHT_INTS-1)); i--)
    {
        if( (intWeight + i)*stepsize < lowerBound && !newNorm )
        {
            break;
        }
        else if ( newNorm && (i < -1) )
        {
          break;
        }

        bitrate = estimateWeightVal(intWeight + i) * oneOver2_15;
        diff = (origWeight - ((intWeight + i)*stepsize)) / (normInterval*(float32_t)0.5);
        distortion = (diff * diff);


        currCost = distortion + (bitrate*lambda);

        if (currCost < minCost)
        {
            bestWeightInt = intWeight + i;
            minCost = currCost;
            bestDist = distortion;
        }
    }

    for (int i = 1; (i + intWeight) < (1 << (BITS_WEIGHT_INTS-1)); i++)
    {
        if ( (intWeight + i)*stepsize > upperBound && !newNorm )
        {
            break;
        }
        else if( newNorm && (i > 1) )
        {
          break;
        }

        bitrate = estimateWeightVal(intWeight + i ) * oneOver2_15;
        diff = (origWeight - ((intWeight + i)*stepsize)) / (normInterval*(float32_t)0.5);
        distortion = (diff * diff);

        currCost = distortion + (bitrate*lambda);

        if (currCost < minCost)
        {
            bestWeightInt = intWeight + i;
            minCost = currCost;
            bestDist = distortion;
        }
    }
    return bestDist;
}

void CABACEncoder::encodeSideinfo( float32_t stepsize, py::array_t<float32_t, py::array::c_style> Weights )
{
  m_BinEncoder.encodeBinsEP(g_NumGtxFlags - 4, 4);

  py::buffer_info bi_Weights = Weights.request();
  uint32_t dimensionFlag = bi_Weights.ndim >> 1; //indicates the number of Dimensions: dimensionFlag == 0 -> Dimsize == 1 // dimensionFlag == 1 -> Dimsize == 2 // dimensionFlag == 2 -> Dimsize == 4
  uint32_t currDim       = 0;

  m_BinEncoder.encodeBinsEP(dimensionFlag, 2);

  for (uint32_t i = 0; i < bi_Weights.ndim; i++)
  {
    m_BinEncoder.encodeBinsEP(bi_Weights.shape[i], 16);
  }

  FloatUIntUnion fTui;

  fTui.f = stepsize;
  m_BinEncoder.encodeBinsEP(fTui.ui & 0xFFFFu, 16);
  m_BinEncoder.encodeBinsEP((fTui.ui >> 16), 16);
}

void CABACEncoder::encodeWeightsRD( float32_t* pWeights, float32_t* pIntervals, float32_t stepsize, float32_t lambda, uint32_t layerWidth, uint32_t numWeights )
{
  int32_t bestIntVal = 0;
  double distSum     = 0.0;
  m_CtxModeler.resetNeighborCtx();
  for (uint32_t posInMat = 0; posInMat < numWeights; posInMat++)
  {
    bestIntVal = 0;
    distSum += estimateAndDecideWeight( bestIntVal, pWeights[ posInMat ], pIntervals[ posInMat ], stepsize, lambda );
    encodeWeightVal( bestIntVal );
    m_CtxModeler.updateNeighborCtx( bestIntVal, posInMat, layerWidth );
  }
}

void CABACEncoder::encodeWeightsRD( float32_t* pWeights, float32_t Interval, float32_t stepsize, float32_t lambda, uint32_t layerWidth, uint32_t numWeights )
{
  int32_t bestIntVal = 0;
  double distSum     = 0.0;
  m_CtxModeler.resetNeighborCtx();

  for (uint32_t posInMat = 0; posInMat < numWeights; posInMat++)
  {
    bestIntVal = 0;
    distSum += estimateAndDecideWeight( bestIntVal, pWeights[ posInMat ], Interval, stepsize, lambda );
    encodeWeightVal( bestIntVal );
    m_CtxModeler.updateNeighborCtx( bestIntVal, posInMat, layerWidth );
  }
}

void CABACEncoder::encodeWeightsRD(int8_t *pWeights, uint32_t numWeights)
{
    int32_t bestIntVal = 0;
    double distSum = 0.0;
    m_CtxModeler.resetNeighborCtx();

    for (uint32_t posInMat = 0; posInMat < numWeights; posInMat++)
    {
        bestIntVal = static_cast<int32_t>(pWeights[posInMat]);
        encodeWeightVal(bestIntVal);
        m_CtxModeler.updateNeighborCtx(bestIntVal, posInMat, numWeights);
    }
}
