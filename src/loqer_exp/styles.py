import seaborn


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def set_default_style():
    seaborn.set_theme(style="whitegrid")


IMPERIAL_COLOR_TO_HEX = {
    "ic_dark": "#232333",
    "ic_navy_blue": "#000080",
    "ic_saddle_brown": "#8b4513",
    "ic_teal": "#008080",
    "ic_medium_violet_red": "#c71585",
    "ic_indigo": "#4b0082",
    "ic_crimson": "#dc143c",
    "ic_orange_red": "#ff4500",
    "ic_dark_green": "#006400",
    "ic_slate_gray": "#708090",
    "ic_imperial_blue": "#0000cd",
    "ic_yellow": "#ffff00",
    "ic_turquoise": "#40e0d0",
    "ic_violet": "#ee82ee",
    "ic_medium_blue_slate": "#7b68ee",
    "ic_red": "#ff0000",
    "ic_dark_orange": "#ff8c00",
    "ic_spring_green": "#00ff7f",
    "ic_white_smoke": "#f5f5f5",
    "ic_deep_sky_blue": "#00bfff",
    "ic_khaki": "#f0e68c",
    "ic_pale_turquoise": "#afeeee",
    "ic_light_pink": "#ffb6c1",
    "ic_lavender": "#e6e6fa",
    "ic_salmon": "#fa8072",
    "ic_orange": "#ffa500",
    "ic_pale_green": "#98fb98",
}


def get_ic_color(name):
    return IMPERIAL_COLOR_TO_HEX[name]


CZ_COLOR_TO_HEX = {
    # red
    "cz_darkred": "#8C000F",
    "cz_red": "#E50000",
    "cz_lightred": "#FC5A50",
    # orange
    "cz_darkorange": "#D2691E",
    "cz_orange": "#F97306",
    "cz_lightorange": "#FFA500",
    # green
    "cz_darkgreen": "#006400",
    "cz_green": "#15B01A",
    "cz_lightgreen": "#7FECAD",
    # blue
    "cz_darkblue": "#030764",
    "cz_blue": "#0343DF",
    "cz_lightblue": "#80ccf9",
    # purple
    "cz_darkpurple": "#35063E",
    "cz_purple": "#7E1E9C",
    "cz_lightpurple": "#C79FEF",
    # grey
    "cz_darkgrey": "#424242",
    "cz_grey": "#848484",
    "cz_lightgrey": "#D0D0D0",
}


def get_cz_color(name):
    return CZ_COLOR_TO_HEX[name]


def plot_palette(palette_name: str, figsize=(5, 3)):
    import matplotlib.pyplot as plt

    if palette_name == "cz":
        palette = CZ_COLOR_TO_HEX
    elif palette_name == "ic":
        palette = IMPERIAL_COLOR_TO_HEX
    else:
        raise ValueError(f"Unknown palette name: {palette_name}")

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, color) in enumerate(palette.items()):
        ax.bar(i, 1, color=color, label=name)
    # place the legend under the plot
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    ax.xaxis.set_ticklabels([])
    plt.show()
