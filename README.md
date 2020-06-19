# DeepCABAC for Python

This code implements the encoding and decoding process of DeepCABAC as described in _(insert paper link)_.

Python binding of the C++ Implementation of deepCABAC using pybind11.

## How to install

Tested on

- Ubuntu 18.04, gcc 7.4
- Windows 7, Visual Studio 2019
- Windows 10, Visual Studio 2019
- OS X Mojave 10.14, Xcode 10.2.1

You need python >= 3.6 with working pip:

From the root of this repository, run

```
pip install .
```

to install deepCABAC extension.

## Examples

### Encoding Pytorch model weights
```
import deepCABAC
import torchvision.models as models

model = models.resnet50(pretrained=True)
encoder = deepCABAC.Encoder()

interv = 0.1
stepsize = 15
_lambda = 0.

for name, param in model.state_dict().items():
    if '.weight' in name:
        encoder.encodeWeightsRD( weights, interv, stepsize, _lambda )
    else:
        encoder.encodeWeightsRD( weights, interv, stepsize + 4, _lambda )
    
stream = encoder.finish()
with open('weights.bin', 'wb') as f:
    f.write(stream)
```

### Decoding Pytorch model weights
```
import deepCABAC
import torchvision.models as models

model = models.resnet50(pretrained=False)
decoder = deepCABAC.Decoder()

with open('weights.bin', 'rb') as f:
    stream = f.read()

decoder.getStream(stream)
state_dict = model.state_dict()
for name in state_dict.keys():
    state_dict[name] = torch.tensor(decA.decodeWeights())
decoder.finish()

model.load_state_dict(state_dict)

# evaluate(model)
```

### Debugging

If you want to debug the module, on Ubuntu with gdb you can use:

```
CFLAGS='-Wall -O0 -g' pip install .
```


Find simple usage example in `./Tests`.
