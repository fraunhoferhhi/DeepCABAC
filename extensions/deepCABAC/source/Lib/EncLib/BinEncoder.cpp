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
#include <random>
#include <algorithm>

#include "BinEncoder.h"
#include <bitset>
#include <iostream>

#if _WIN32
inline uint32_t __builtin_clz(uint32_t x)
{
    unsigned long position;
    _BitScanReverse(&position, x);
    return 31 - position;
}
#endif

const uint32_t BinEnc::m_auiGoRiceRange[ 10 ] =
{
    6, 5, 6, 3, 3, 3, 3, 3, 3, 3
};


void BinEnc::startBinEncoder()
{
    m_Low                = 0;
    m_Range              = 510;
    m_BitsLeft           = 23;
    m_NumBufferedBytes   = 0;
}


void BinEnc::setByteStreamBuf( std::vector<uint8_t> *byteStreamBuf )
{
    m_ByteBuf = byteStreamBuf;
}


void BinEnc::encodeBin( uint32_t bin, SBMPCtx &ctxMdl )
{
    encodeBinNoUpdate( bin, ctxMdl );
    ctxMdl.updateState( -(int32_t)bin );
}


void BinEnc::encodeBinNoUpdate( uint32_t bin, const SBMPCtx &ctxMdl )
{
    uint32_t rlps = ctxMdl.getRLPS( m_Range );
    m_Range -= rlps;

    int32_t minusBin = -(int32_t)bin;

    if (minusBin == ctxMdl.getMinusMPS() )
    {
        if (m_Range < 256)
        {
            m_Range += m_Range;
            m_Low += m_Low;
            m_BitsLeft -= 1;
            if (m_BitsLeft < 12)
                write_out();
        }
    }
    else
    {
        uint32_t n = __builtin_clz(rlps) - 23;
        m_Low += m_Range;
        m_Range = rlps << n;
        m_Low <<= n;
        m_BitsLeft -= n;
        if (m_BitsLeft < 12)
            write_out();
    }
}


void BinEnc::encodeBinEP( uint32_t bin )
{
    m_Low <<= 1;
    if (bin)
    {
        m_Low += m_Range;
    }
    m_BitsLeft--;
    if (m_BitsLeft < 12)
    {
        write_out();
    }
}


void BinEnc::encodeBinsEP( uint32_t bins, uint32_t numBins )
{
    CHECK( bins > ( 1u << numBins ), printf( "%i can not be coded with %i EP-Bins", bins, numBins ) )
    
    if (m_Range == 256)
    {
        uint32_t remBins = numBins;
        while (remBins > 0)
        {
            uint32_t binsToCode = std::min<uint32_t>(remBins, 8); //code bytes if able to take advantage of the system's byte-write function
            uint32_t binMask    = (1 << binsToCode) - 1;
            uint32_t newBins    = (bins >> (remBins - binsToCode)) & binMask;
            m_Low               = (m_Low << binsToCode) + (newBins << 8); //range is known to be 256
            remBins            -= binsToCode;
            m_BitsLeft         -= binsToCode;
            if (m_BitsLeft < 12)
            {
                write_out();
            }
        }

        return;
    }
    while (numBins > 8)
    {
        numBins          -= 8;
        uint32_t pattern  = bins >> numBins;
        m_Low           <<= 8;
        m_Low            += m_Range * pattern;
        bins             -= pattern << numBins;
        m_BitsLeft       -= 8;
        if (m_BitsLeft < 12)
        {
            write_out();
        }
    }
    m_Low     <<= numBins;
    m_Low      += m_Range * bins;
    m_BitsLeft -= numBins;
    if (m_BitsLeft < 12)
    {
        write_out();
    }
}


void BinEnc::encodeRemAbsLevel( uint32_t remAbsLevel, uint32_t goRicePar, bool useLimitedPrefixLength, uint32_t maxLog2TrDynamicRange )
{   
    uint32_t bins = (uint32_t)remAbsLevel;

    const uint32_t threshold = m_auiGoRiceRange[ goRicePar ] << goRicePar;
    if (bins < threshold)
    {
        const uint32_t bitMask  = (1 << goRicePar) - 1;
        const uint32_t length   = (bins >> goRicePar) + 1;
        encodeBinsEP((1 << length) - 2, length);
        encodeBinsEP(bins & bitMask, goRicePar);
    }
    else if (useLimitedPrefixLength)
    {
        const uint32_t  maxPrefixLength = 32 - 3 - maxLog2TrDynamicRange;
        uint32_t        prefixLength    = 0;
        uint32_t        codeValue       = (bins >> goRicePar) - 3;
        uint32_t        suffixLength;
        if ((uint32_t)codeValue >= (uint32_t)((1 << maxPrefixLength) - 1)) //CORRECT BUGFIX (Cast) ??? //REMOVE
        {
            prefixLength = maxPrefixLength;
            suffixLength = maxLog2TrDynamicRange;
        }
        else
        {
            while ((uint32_t)codeValue > (uint32_t)((2 << prefixLength) - 2)) //CORRECT BUGFIX (Cast) ??? //REMOVE
            {
                prefixLength++;
            }
            suffixLength = prefixLength + goRicePar + 1; //+1 for the separator bit
        }
        const uint32_t totalPrefixLength  = prefixLength + 3;
        const uint32_t bitMask            = (1 << goRicePar) - 1;
        const uint32_t prefix             = (1 << totalPrefixLength) - 1;
        const uint32_t suffix             = ((codeValue - ((1 << prefixLength) - 1)) << goRicePar) | (bins & bitMask);
        encodeBinsEP(prefix, totalPrefixLength); //prefix
        encodeBinsEP(suffix, suffixLength); //separator, suffix, and rParam bits
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
        uint32_t numBin = m_auiGoRiceRange[ goRicePar ] + length + 1 - goRicePar;
        encodeBinsEP((1 << numBin) - 2, numBin);
        encodeBinsEP(bins, length);
    }
}

void BinEnc::encodeRemAbsLevelNew( uint32_t remAbsLevel, std::vector<SBMPCtx>& ctxStore )
{
    uint32_t remLevel = uint32_t(remAbsLevel);
    uint32_t log2NumElemNextGroup = 0;

    uint32_t remAbsBaseLevel = 0;
    uint32_t ctxIdx = (6 + g_NumGtxFlags * 2);

    if (remLevel > 0)
    {
        encodeBin(1, ctxStore[ ctxIdx ]);
        remAbsBaseLevel += (1 << log2NumElemNextGroup);
        ctxIdx++;
        log2NumElemNextGroup++;
    }
    else
    {
        encodeBin(0, ctxStore[ ctxIdx ]);
        return;
    }

    while ( remLevel >  (remAbsBaseLevel + (1 << log2NumElemNextGroup) -1 ) )
    {
        encodeBin(1 , ctxStore[ctxIdx]);
        remAbsBaseLevel += (1 << log2NumElemNextGroup);
        ctxIdx++;
        log2NumElemNextGroup++;
    }

    uint32_t bins = remLevel - remAbsBaseLevel;

    encodeBin(0, ctxStore[ ctxIdx ]);
    encodeBinsEP( bins, log2NumElemNextGroup );
}


void BinEnc::write_out()
{
    uint32_t lead_byte = m_Low >> (24 - m_BitsLeft);
    m_BitsLeft += 8;
    m_Low &= 0xffffffffu >> m_BitsLeft;
    if (lead_byte == 0xff)
    {
        m_NumBufferedBytes++;
    }
    else
    {
        if (m_NumBufferedBytes > 0)
        {
            uint32_t carry      = lead_byte >> 8;
            uint8_t  byte       = m_BufferedByte + carry;
            m_BufferedByte       = lead_byte & 0xff;
            m_ByteBuf->push_back(byte);
            byte                = (0xff + carry) & 0xff;
            while (m_NumBufferedBytes > 1)
            {
                m_ByteBuf->push_back(byte);
                m_NumBufferedBytes--;
            }
        }
        else
        {
            m_NumBufferedBytes = 1;
            m_BufferedByte      = lead_byte;
        }
    }
}

void BinEnc::encodeBinTrm( unsigned bin )
{
  m_Range -= 2;
  if( bin )
  {
    m_Low      += m_Range;
    m_Low     <<= 7;
    m_Range     = 2 << 7;
    m_BitsLeft -= 7;
  }
  else if( m_Range >= 256 )
  {
    return;
  }
  else
  {
    m_Low     <<= 1;
    m_Range   <<= 1;
    m_BitsLeft--;
  }
  if( m_BitsLeft < 12 )
  {
    write_out();
  }
}

void BinEnc::finish()
{
  if( m_Low >> ( 32 - m_BitsLeft ) )
  {
    m_ByteBuf->push_back( m_BufferedByte + 1 );
    while( m_NumBufferedBytes > 1 )
    {
      m_ByteBuf->push_back( 0x00 );
      m_NumBufferedBytes--;
    }
    m_Low -= 1 << ( 32 - m_BitsLeft );
  }
  else
  {
    if( m_NumBufferedBytes > 0 )
    {
      m_ByteBuf->push_back( m_BufferedByte );
    }
    while( m_NumBufferedBytes > 1 )
    {
      m_ByteBuf->push_back( 0xff );
      m_NumBufferedBytes--;
    }
  }

  // add trailing 1
   m_Low >>= 8;
   m_Low <<= 1;
   m_Low++;
   m_BitsLeft--;
   // left align
   m_Low <<= (32 - (24-m_BitsLeft) );
   // write out starting from the leftmost byte
   for( int i = 0; i < 24 - m_BitsLeft; i+=8 )
   {
     m_ByteBuf->push_back( (m_Low >> (24-i)) & 0xFF );
   }
}
