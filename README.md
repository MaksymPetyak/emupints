# Emupints
emupints (Emulators for PINTS) provides an interface to apply machine learning techniques such as Gaussian Processes (GPs) and Bayesian Neural Networks (BNNs) to [PINTS](https://github.com/pints-team/pints).
For a good first tutorial see this [example notebook](examples/emulator-logistic-model.ipynb)

## Installing emupints

 There are a number of libraries required to use emupints, including: `pints` `GPy` `edward`
 See [requirements.txt](requirements.txt) for more info. 

These can easily be installed using `pip`. To do this, first make sure you have the latest version of pip installed:

```
$ pip install --upgrade pip
```

Then navigate to the path where you downloaded emupints to, and install both Pints and its dependencies by typing:

```
$ pip install .
```

Since there are a number of dependencies, in case something goes wrong you can try install without them, although some functionality might not be available. This might be desirable if you are only interested in applying Gaussian Processes and already have GPy, for instance.

```
$ pip install --no-deps .
```