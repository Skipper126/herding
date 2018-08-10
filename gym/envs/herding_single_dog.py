from .assets import Herding


class HerdingSingleDog(Herding):

        def __init__(self):
            super().__init__(
                dog_count=1
            )
