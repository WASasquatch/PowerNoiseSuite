import random
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

peacock_rambling = ["Hrruuh-hhruh!!!", "Squaaack!", "Cock-a-doodle-dooo!", "Squeaa-squee!!!", "kweh..", "wark!", "MooOoo!!!", "Sup."]
print(f"\033[1;32m[Power Noise Suite]: 🦚🦚🦚 \033[93m\033[3m{random.choice(peacock_rambling)}\033[0m 🦚🦚🦚")
print(f"\033[1;32m[Power Noise Suite]:\033[0m Tamed \033[93m{len(NODE_CLASS_MAPPINGS)}\033[0m wild nodes.")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']