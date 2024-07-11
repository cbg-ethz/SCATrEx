# Functions to make a colormap out of a tree
import matplotlib
import seaborn as sns
from colorsys import rgb_to_hls
from . import constants

def adjust_color(rgba, lightness_scale, saturation_scale):
    # scale the lightness (The values should be between 0 and 1)
    rgb = rgba[:-1]
    a = rgba[-1]
    hls = rgb_to_hls(*rgb)
    lightness = max(0,min(1, hls[1] * lightness_scale))
    saturation = max(0,min(1,hls[2] * saturation_scale))
    rgb = sns.set_hls_values(color = rgb, h = None, l = lightness, s = saturation)
    rgba = rgb + (a,)
    hex = matplotlib.colors.to_hex(rgba, keep_alpha=True)
    return hex


def make_tree_colormap(tree, base_color, brightness_mult=0.7, saturation_mult=1.3):
    """
    Updates the tree dictionary with colors defined from node depth and breadth
    tree: nested dictionary containing {node: children} with children a list of also dictionaries {child: children}
    base_color: HEX code for base color
    saturation_mult: how much to change saturation for each step in depth, centered at 1
    brightness_mult: how much to change brightness for each step in breadth, centered at 1
    """
    base_color_rgba = matplotlib.colors.ColorConverter.to_rgba(base_color)

    tree['color'] = base_color

    # Traverse tree to update out_dict
    def descend(root, depth=1, breadth=1):
        for i, child in enumerate(root['children']):
            breadth += i
            color = adjust_color(base_color_rgba, brightness_mult**depth, saturation_mult**breadth)
            child['color'] = color
            descend(child, depth=depth+1, breadth=breadth)
    descend(tree)


def make_color_palette(n_categories):
    """
    Adapted from scanpy.plotting._utils._set_default_colors_for_categorical_obs
    """
    from scanpy.plotting import palettes
    # check if default matplotlib palette has enough colors
    if n_categories <= len(constants.CLONES_PAL):
        palette = constants.CLONES_PAL
    elif n_categories <= 20:
        palette = palettes.default_20
    elif n_categories <= 28:
        palette = palettes.default_28
    elif n_categories <= len(palettes.default_102):  # 103 colors
        palette = palettes.default_102
    else:
        palette = ["grey" for _ in range(n_categories)]
    
    return palette


