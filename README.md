# OpenAI Gym herding environment

Installation
============
### 1. Install pyopencl
https://wiki.tiker.net/PyOpenCL/Installation
### 2. Install herding package

```
pip install <package_directory>
```
Running
============

## 2. Create environment
You can directly create Herding class object and specify the parameters.
```python
import herding

env = herding.Herding(
    dog_count=3,
    sheep_count=20,
    sheep_type='simple'
)
```
## 3. Manual steering
You can play the scenario yourself. 
```python
import herding

herding.play()
```
The play function also accepts custom created Herding environment.
```python
import herding

env = herding.Herding(
    sheep_count=15,
    max_movement_speed=10,
    rotation_mode='centered'
)

herding.play(env)
```
