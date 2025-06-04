
from .multiview import MULTIVIEW_NODE_CLASS_MAPPINGS, MULTIVIEW_NODE_DISPLAY_NAME_MAPPINGS

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
}

NODE_CLASS_MAPPINGS.update(MULTIVIEW_NODE_CLASS_MAPPINGS)


# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
}

NODE_DISPLAY_NAME_MAPPINGS.update(MULTIVIEW_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

