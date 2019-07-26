#!/usr/bin/env python3
'''
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
'''
from os.path import dirname, realpath, join
import deepCABAC, numpy

addSteps    = 256
fixInterval = 0.1
_lambda     = 0.001


with open( join(dirname(realpath(__file__)), "./conv1.weight.txt"), "r" ) as _f:
    dims_w  = numpy.array( _f.readline().strip(" ,\n").split(","), dtype=numpy.int )
    weights = numpy.fromfile( _f, dtype = numpy.float32, sep="\n" )

with open( join(dirname(realpath(__file__)), "./conv1.interv.txt"), "r" ) as _f:
    dims_i = numpy.array( _f.readline().strip(" ,\n").split(","), dtype=numpy.int )
    interv = numpy.fromfile( _f, dtype = numpy.float32, sep="\n" )

assert numpy.array_equal(dims_w, dims_i), "weights and intervals dimension missmatch!"


weightAddHalfInterv = ( numpy.abs( weights ) + interv / 2. )
maxIdx              = numpy.argmax( weightAddHalfInterv )

weightsRange = 2*weightAddHalfInterv[ maxIdx ]
minInterv    = numpy.min( interv )

nSteps       = numpy.ceil( weightsRange / minInterv )
nSteps      += addSteps

weights, interv = weights.reshape(dims_w), interv.reshape(dims_i)

print( "Weights Range     : {}".format( weightsRange ) )
print( "min Interval      : {}".format( minInterv    ) )

stepsize = weightsRange / nSteps

print( "Quantizer stepsize: {}".format( stepsize ) )


# Encode/Decode using interval matrix providing importance interval per weight
encA, decA = deepCABAC.Encoder(), deepCABAC.Decoder()

encA.encodeWeightsRD( weights, interv, stepsize, _lambda )
streamA = encA.finish()
decA.getStream( streamA )
recA = decA.decodeWeights()
decA.finish()

# Encode/Decode using a fixed interval which is the same across all the weights
encB, decB = deepCABAC.Encoder(), deepCABAC.Decoder()

encB.encodeWeightsRD( weights, fixInterval, stepsize, _lambda )
streamB = encB.finish()
decB.getStream( streamB )
recB = decB.decodeWeights()
decB.finish()


print( "Bitstream size (Interval matrix): {}".format( streamA.size ) )
print( "MSE            (Interval matrix): {}".format( numpy.square((weights - recA)).mean() ) )
print( "Bitstream size (fixed Interval ): {}".format( streamB.size ) )
print( "MSE            (fixed Interval ): {}".format( numpy.square((weights - recB)).mean() ) )
