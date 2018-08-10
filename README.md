# OpenAI Gym herding environment

Installation
============
```
pip install -e <package_directory>
```
Running
============
## 1. OpenAi Gym env
The environment can be created by gym.make. Available scenarios:
* herding-singleDog-v0
```python
import gym
import herding

env = gym.make(gym)
```
## 2. Use own configuration
You can directly create Herding class object and specify the parameters.
```python
import herding

env = herding.Herding(
    dog_count=3,
    sheep_count=20,
    sheep_type=herding.constants.SheepType.COMPLEX
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
    rotation_mode=herding.constants.RotationMode.LOCKED_ON_HERD_CENTRE
)

herding.play(env)
```