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
#include "ContextModeler.h"


void ContextModeler::init()
{
    neighborWeightVal = 0;
}

void ContextModeler::resetNeighborCtx()
{
    init();
}


int32_t ContextModeler::getSigCtxId()
{
    int32_t ctxId = 0;

    if (neighborWeightVal != 0)
    {
        ctxId = neighborWeightVal < 0 ? 1 : 2;
    }

    return ctxId;
}

int32_t ContextModeler::getSignFlagCtxId()
{
    int32_t ctxId = 3;

    if (neighborWeightVal != 0)
    {
        ctxId = neighborWeightVal < 0 ? 4 : 5;
    }

    return ctxId;
}

int32_t ContextModeler::getGtxCtxId( int32_t currWeighVal, uint32_t numGtxFlagsCoded )
{
    int32_t offset =  6;

    int32_t ctxId  = 0;

    ctxId = currWeighVal > 0 ? (numGtxFlagsCoded << 1) : 1 + (numGtxFlagsCoded << 1);

    return (ctxId + offset);
}


void ContextModeler::updateNeighborCtx( int32_t currWeightVal, uint32_t posInMat, uint32_t layerWidth )
{
    if (posInMat % layerWidth == layerWidth - 1)
    {
        neighborWeightVal = 0;
    }
    else
    {
        neighborWeightVal = currWeightVal;
    }
}
