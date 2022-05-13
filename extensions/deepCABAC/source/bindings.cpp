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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Lib/CommonLib/TypeDef.h>
#include <Lib/EncLib/CABACEncoder.h>
#include <Lib/DecLib/CABACDecoder.h>
#include <iostream>
#include <tuple>

namespace py = pybind11;

uint32_t g_NumGtxFlags = 10;

class Encoder
{
public:
  Encoder();
  ~Encoder();
  void                  encodeWeightsRD ( py::array_t<float32_t, py::array::c_style> Weights, py::array_t<float32_t, py::array::c_style> Intervals, float32_t stepsize, float32_t lambda );
  void                  encodeWeightsRD ( py::array_t<float32_t, py::array::c_style> Weights,             float32_t                      Interval,  float32_t stepsize, float32_t lambda );
  void encodeWeightsRD(py::array_t<int8_t, py::array::c_style> Weights);

  py::array_t<uint8_t, py::array::c_style>  finish();
private:
  std::vector<uint8_t>  m_Bytestream;
  CABACEncoder*         m_CABACEncoder;
};

Encoder::Encoder()
{
  m_CABACEncoder = new CABACEncoder( &m_Bytestream );
  m_CABACEncoder->startCabacEncoding();
}

Encoder::~Encoder()
{
  delete m_CABACEncoder;
}

void Encoder::encodeWeightsRD( py::array_t<float32_t, py::array::c_style> Weights, py::array_t<float32_t, py::array::c_style> Intervals, float32_t stepsize, float32_t lambda )
{
  m_CABACEncoder->encodeSideinfo( stepsize, Weights );

  py::buffer_info bi_Weights    = Weights.request();
  float32_t* pWeights           = (float32_t*) bi_Weights.ptr;

  py::buffer_info bi_Intervals  = Intervals.request();
  float32_t* pIntervals         = (float32_t*) bi_Intervals.ptr;

  uint32_t layerWidth = 1;
  uint32_t numWeights = 1;
  for( long int idx = 0; idx < bi_Weights.ndim; idx++ )
  {
    numWeights *= bi_Weights.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= bi_Weights.shape[idx];
  }
  m_CABACEncoder->encodeWeightsRD( pWeights, pIntervals, stepsize, lambda, layerWidth, numWeights );
}

void Encoder::encodeWeightsRD( py::array_t<float32_t, py::array::c_style> Weights, float32_t Interval, float32_t stepsize, float32_t lambda )
{
  m_CABACEncoder->encodeSideinfo( stepsize, Weights );

  py::buffer_info bi_Weights    = Weights.request();
  float32_t* pWeights           = (float32_t*) bi_Weights.ptr;

  uint32_t layerWidth = 1;
  uint32_t numWeights = 1;
  for( long int idx = 0; idx < bi_Weights.ndim; idx++ )
  {
    numWeights *= bi_Weights.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= bi_Weights.shape[idx];
  }
  m_CABACEncoder->encodeWeightsRD( pWeights, Interval, stepsize, lambda, layerWidth, numWeights );
}

void Encoder::encodeWeightsRD(py::array_t<int8_t, py::array::c_style> Weights)
{
    // m_CABACEncoder->encodeSideinfo( stepsize, Weights );

    py::buffer_info bi_Weights = Weights.request();

    uint32_t numWeights = 1;
    for (long int idx = 0; idx < bi_Weights.ndim; idx++)
    {
        numWeights *= bi_Weights.shape[idx];
        if (idx == 0)
        {
            continue;
        }
    }
    int8_t *pWeights_8 = (int8_t*)bi_Weights.ptr;
    m_CABACEncoder->encodeWeightsRD(pWeights_8, numWeights);
}

py::array_t<uint8_t, py::array::c_style> Encoder::finish()
{
  m_CABACEncoder->terminateCabacEncoding();

  auto Result = py::array_t<uint8_t, py::array::c_style>(m_Bytestream.size());
  py::buffer_info bi_Result = Result.request();
  uint8_t* pResult = (uint8_t*) bi_Result.ptr;

  for( size_t idx = 0; idx < m_Bytestream.size(); idx ++ )
  {
    pResult[idx] = m_Bytestream.at(idx);
  }
  return Result;
}

class Decoder
{
public:
  Decoder();
  ~Decoder();
  void                                        getStream     ( py::array_t<uint8_t, py::array::c_style> Bytestream );
  py::array_t<float32_t, py::array::c_style>  decodeWeights ();
  py::array_t<int8_t, py::array::c_style>     decodeWeights(uint32_t numWeights);
  std::tuple<py::array_t<int8_t, py::array::c_style>, float32_t>  decodeWeights_wstepsize ();
  uint32_t                                    finish        ();

private:
  CABACDecoder* m_CABACDecoder;
};

Decoder::Decoder()
{
  m_CABACDecoder = new CABACDecoder;
}

Decoder::~Decoder()
{
  delete m_CABACDecoder;
}

void Decoder::getStream( py::array_t<uint8_t, py::array::c_style> Bytestream )
{
  py::buffer_info bi_Bytestream = Bytestream.request();
  uint8_t* pBytestream          = (uint8_t*) bi_Bytestream.ptr;
  m_CABACDecoder->startCabacDecoding( pBytestream );
}

py::array_t<float32_t, py::array::c_style>  Decoder::decodeWeights()
{
  std::vector<uint32_t> dimensions;
  float32_t stepsize;

  m_CABACDecoder->decodeSideinfo( &dimensions, stepsize );

  uint32_t numWeights = 1;
  uint32_t layerWidth = 1;
  for( size_t idx = 0; idx < dimensions.size(); idx++ )
  {
    numWeights *= dimensions.at(idx);
    if( idx == 0 ) { continue; }
    layerWidth *= dimensions.at(idx);
  }

  auto Weights = py::array_t<int32_t,   py::array::c_style>(numWeights);
  auto Rec     = py::array_t<float32_t, py::array::c_style>(numWeights);

  py::buffer_info bi_Weights = Weights.request();
  int32_t* pWeights = (int32_t*)    bi_Weights.ptr;

  py::buffer_info bi_Rec     = Rec.request();
  float32_t* pRec   = (float32_t*)  bi_Rec.ptr;

  m_CABACDecoder->decodeWeights( pWeights, layerWidth, numWeights );
  for( uint32_t i = 0; i < numWeights; i++ )
  {
    pRec[i] = pWeights[i] * stepsize;
  }

  Rec.resize({dimensions});
  return Rec;
}

py::array_t<int8_t, py::array::c_style> Decoder::decodeWeights(uint32_t numWeights)
{
    auto Weights = py::array_t<int8_t, py::array::c_style>(numWeights);
    py::buffer_info bi_Weights = Weights.request();
    int8_t *pWeights_8 = (int8_t *)bi_Weights.ptr;
    m_CABACDecoder->decodeWeights(pWeights_8, numWeights);
    return Weights;
}

std::tuple<py::array_t<int8_t, py::array::c_style>, float32_t>  Decoder::decodeWeights_wstepsize()
{
  std::vector<uint32_t> dimensions;
  float32_t stepsize;

  m_CABACDecoder->decodeSideinfo( &dimensions, stepsize );

  uint32_t numWeights = 1;
  uint32_t layerWidth = 1;
  for( size_t idx = 0; idx < dimensions.size(); idx++ )
  {
    numWeights *= dimensions.at(idx);
    if( idx == 0 ) { continue; }
    layerWidth *= dimensions.at(idx);
  }

  auto Weights = py::array_t<int32_t,   py::array::c_style>(numWeights);
  auto Rec     = py::array_t<int8_t, py::array::c_style>(numWeights);

  py::buffer_info bi_Weights = Weights.request();
  int32_t* pWeights = (int32_t*)    bi_Weights.ptr;

  py::buffer_info bi_Rec     = Rec.request();
  int8_t* pRec   = (int8_t*)  bi_Rec.ptr;
  m_CABACDecoder->decodeWeights( pWeights, layerWidth, numWeights );
  for( uint32_t i = 0; i < numWeights; i++)
  {
    pRec[i] = (int8_t)pWeights[i];
  }    

  Rec.resize({dimensions});
  return {Rec, stepsize};
}

uint32_t Decoder::finish()
{
  uint32_t bytesRead = m_CABACDecoder->terminateCabacDecoding();
  return bytesRead;
}


PYBIND11_MODULE(deepCABAC, m) 
{
    py::class_<Encoder>(m, "Encoder")
        .def( py::init<>())
        // binding overloaded functions gets a bit messy.
        // there is a nicer syntax when relying on C++14, but this is compatible to C++11 as well:
        .def( "encodeWeightsRD", (void (Encoder::*) (py::array_t<float32_t, py::array::c_style>, py::array_t<float32_t, py::array::c_style>, float32_t, float32_t)) &Encoder::encodeWeightsRD )
        .def( "encodeWeightsRD", (void (Encoder::*) (py::array_t<float32_t, py::array::c_style>, float32_t, float32_t, float32_t))                                  &Encoder::encodeWeightsRD )
        .def( "encodeWeightsRD", (void (Encoder::*) (py::array_t<int8_t, py::array::c_style>)                                    )                                  &Encoder::encodeWeightsRD )
        .def( "finish",          &Encoder::finish          );

    py::class_<Decoder>(m, "Decoder")
        .def( py::init<>())
        .def( "getStream",     &Decoder::getStream,    py::keep_alive<1, 2>())
        .def( "decodeWeights", (py::array_t<float32_t, py::array::c_style> (Decoder::*) ()) &Decoder::decodeWeights )
        .def( "decodeWeights", (py::array_t<int8_t, py::array::c_style> (Decoder::*) (uint32_t)) &Decoder::decodeWeights )
        .def( "decodeWeights_wstepsize", (std::tuple<py::array_t<int8_t, py::array::c_style>, float32_t> (Decoder::*) ()) &Decoder::decodeWeights_wstepsize )
        .def( "finish",        &Decoder::finish        );
}
