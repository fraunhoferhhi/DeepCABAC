import torch
import deepCABAC
import torchvision.models as models
import numpy as np
from tqdm import tqdm


def main():
    # encoding
    model = models.resnet18(pretrained=True)
    encoder = deepCABAC.Encoder()

    interv = 0.1
    stepsize = 2**(-0.5*15)
    stepsize_other = 2**(-0.5*19)
    _lambda = 0.

    for name, param in tqdm(model.state_dict().items()):
        if '.num_batches_tracked' in name:
            continue
        param = param.cpu().numpy()
        if '.weight' in name:
            encoder.encodeWeightsRD(param, interv, stepsize, _lambda)
        else:
            encoder.encodeWeightsRD(param, interv, stepsize_other, _lambda)

    stream = encoder.finish().tobytes()
    print("Compressed size: {:2f} MB".format(1e-6 * len(stream)))
    with open('weights.bin', 'wb') as f:
        f.write(stream)

    # decoding
    model = models.resnet18(pretrained=False)
    decoder = deepCABAC.Decoder()

    with open('weights.bin', 'rb') as f:
        stream = f.read()

    decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
    state_dict = model.state_dict()

    for name in tqdm(state_dict.keys()):
        if '.num_batches_tracked' in name:
            continue
        param = decoder.decodeWeights()
        state_dict[name] = torch.tensor(param)
    decoder.finish()

    model.load_state_dict(state_dict)

    # evaluate(model)


if __name__ == '__main__':
    main()
