from ..robot import BaseWalker
from ..joystick_policy.joystick_policy import JoystickPolicy


class WalkerEnvironment:
    @property
    def robot(self) -> BaseWalker:
        pass

class JoystickEnvironment:
    @property
    def joystick_policy(self) -> JoystickPolicy:
        pass

    def set_joystick_policy(self, joystick_policy: JoystickPolicy):
        pass

    @property
    def is_resetter_policy(self) -> bool:
        return False