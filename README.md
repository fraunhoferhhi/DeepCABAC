# deepCABAC for Python

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

### Debugging

If you want to debug the module, on Ubuntu with gdb you can use:

```
CFLAGS='-Wall -O0 -g' pip install .
```


Find simple usage example in `./Tests`.
