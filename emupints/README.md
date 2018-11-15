# Emulators
All emulators inherit from base **Emulator** class. <br/>
Generally, using an emulator follows `__init__()`, `set_parameters()`, `fit()`, `__call__()` pattern.

```python
emu = Emulator(likelihood, X_train, y_train)
emu.set_parameters(**kwargs)
emu.fit()
emu(x_test)
```
All emulators are callable, and have outputs and inputs matching dimensions of _likelihood_ provided. For some cases we also want to know uncertainty, in which case `predict()` method should be used. 
